"""Mine trip pattern association rules using Spark FP-Growth."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

LOGGER_NAME = "nyc_taxi_trip_pattern_rules"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_CSV_DIR = DATA_DIR / "processed_csv"
TRIP_CSV_GLOB = str(PROCESSED_CSV_DIR / "trip_details_*.csv")
ZONE_CSV = PROCESSED_CSV_DIR / "location_zone_data.csv"

ARTIFACT_DIR = DATA_DIR / "trip_pattern_artifacts"
ALL_RULES_CSV = ARTIFACT_DIR / "trip_pattern_rules_all.csv"
TOP_RULES_CSV = ARTIFACT_DIR / "trip_pattern_rules_top10.csv"
METADATA_JSON = ARTIFACT_DIR / "trip_pattern_rules_metadata.json"


@dataclass(frozen=True)
class RuleMiningConfig:
    min_support: float = 0.001
    min_confidence: float = 0.25
    spark_app_name: str = "NYCTaxiTripPatternFPGrowth"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _create_spark_session(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def _safe_text_col(col_name: str) -> Any:
    return F.lower(F.trim(F.coalesce(F.col(col_name).cast("string"), F.lit(""))))


def _load_and_prepare_transaction_frame(spark: SparkSession) -> DataFrame:
    if not ZONE_CSV.exists():
        raise FileNotFoundError(f"Missing zone CSV: {ZONE_CSV}")

    trips_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(TRIP_CSV_GLOB)
        .select(
            F.col("pickup_location_id").cast("int").alias("source_id"),
            F.col("dropff_location_id").cast("int").alias("destination_id"),
            F.col("trip_distance").cast("double").alias("trip_distance"),
            F.col("fare_amount").cast("double").alias("fare_amount"),
            F.col("tip_amount").cast("double").alias("tip_amount"),
            F.col("payment_type").cast("int").alias("payment_type"),
            F.to_date(F.col("pickup_date")).alias("pickup_date"),
            F.col("pickup_time").cast("string").alias("pickup_time"),
        )
    )

    zones_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(ZONE_CSV))
        .select(
            F.col("location_id").cast("int").alias("location_id"),
            F.col("borough").cast("string").alias("borough"),
            F.col("zone").cast("string").alias("zone"),
        )
    )

    src_zone = zones_df.select(
        F.col("location_id").alias("source_id"),
        F.col("borough").alias("source_borough"),
        F.col("zone").alias("source_zone"),
    )
    dst_zone = zones_df.select(
        F.col("location_id").alias("destination_id"),
        F.col("borough").alias("destination_borough"),
        F.col("zone").alias("destination_zone"),
    )

    merged_df = (
        trips_df.join(src_zone, on="source_id", how="left")
        .join(dst_zone, on="destination_id", how="left")
    )

    src_unknown = _safe_text_col("source_zone").isin("unknown", "nan", "")
    dst_unknown = _safe_text_col("destination_zone").isin("unknown", "nan", "")
    src_borough_unknown = _safe_text_col("source_borough").isin("unknown", "nan", "")
    dst_borough_unknown = _safe_text_col("destination_borough").isin("unknown", "nan", "")

    pickup_hour = F.expr("try_cast(substring(pickup_time, 1, 2) as int)")

    filtered_df = merged_df.filter(
        F.col("source_id").isNotNull()
        & F.col("destination_id").isNotNull()
        & F.col("pickup_date").isNotNull()
        & F.col("pickup_time").isNotNull()
        & pickup_hour.isNotNull()
        & pickup_hour.between(0, 23)
        & F.col("trip_distance").isNotNull()
        & F.col("fare_amount").isNotNull()
        & F.col("tip_amount").isNotNull()
        & F.col("payment_type").isNotNull()
        & (F.col("trip_distance") > 0)
        & (F.col("fare_amount") >= 0)
        & (F.col("tip_amount") >= 0)
        & (F.col("fare_amount") < 1000)
        & (F.col("trip_distance") < 30)
        & ~src_unknown
        & ~dst_unknown
        & ~src_borough_unknown
        & ~dst_borough_unknown
    )

    time_bin = (
        F.when((pickup_hour >= 8) & (pickup_hour < 12), F.lit("Morning"))
        .when((pickup_hour >= 12) & (pickup_hour < 16), F.lit("Afternoon"))
        .when((pickup_hour >= 16) & (pickup_hour < 20), F.lit("Evening"))
        .when((pickup_hour >= 20) & (pickup_hour <= 23), F.lit("Night"))
    )

    day_type = (
        F.when(F.dayofweek(F.col("pickup_date")).isin([1, 7]), F.lit("Weekend"))
        .otherwise(F.lit("Weekday"))
    )

    source_airport = (
        F.when(_safe_text_col("source_zone").contains("jfk"), F.lit("JFK"))
        .when(_safe_text_col("source_zone").contains("laguardia"), F.lit("LGA"))
        .when(
            _safe_text_col("source_zone").contains("newark")
            | (_safe_text_col("source_borough") == F.lit("ewr")),
            F.lit("EWR"),
        )
    )
    destination_airport = (
        F.when(_safe_text_col("destination_zone").contains("jfk"), F.lit("JFK"))
        .when(_safe_text_col("destination_zone").contains("laguardia"), F.lit("LGA"))
        .when(
            _safe_text_col("destination_zone").contains("newark")
            | (_safe_text_col("destination_borough") == F.lit("ewr")),
            F.lit("EWR"),
        )
    )

    distance_bin = (
        F.when(F.col("trip_distance") < 2, F.lit("0-2mi"))
        .when(F.col("trip_distance") < 5, F.lit("2-5mi"))
        .when(F.col("trip_distance") < 10, F.lit("5-10mi"))
        .when(F.col("trip_distance") < 20, F.lit("10-20mi"))
        .otherwise(F.lit("20-30mi"))
    )
    fare_bin = (
        F.when(F.col("fare_amount") < 10, F.lit("$0-$10"))
        .when(F.col("fare_amount") < 20, F.lit("$10-$20"))
        .when(F.col("fare_amount") < 40, F.lit("$20-$40"))
        .when(F.col("fare_amount") < 80, F.lit("$40-$80"))
        .when(F.col("fare_amount") < 100, F.lit("$80-$100"))
        .when(F.col("fare_amount") < 140, F.lit("$100-$140"))
        .when(F.col("fare_amount") < 180, F.lit("$140-$180"))
        .when(F.col("fare_amount") < 300, F.lit("$180-$300"))
        .otherwise(F.lit("$300-$1000"))
    )
    tip_ratio = F.when(F.col("fare_amount") > 0, F.col("tip_amount") / F.col("fare_amount")).otherwise(F.lit(0.0))
    tip_bin = (
        F.when(F.col("tip_amount") <= 0, F.lit("NoTip"))
        .when(tip_ratio < 0.10, F.lit("LowTip"))
        .when(tip_ratio < 0.20, F.lit("MidTip"))
        .otherwise(F.lit("HighTip"))
    )
    payment_type_label = (
        F.when(F.col("payment_type") == 1, F.lit("CreditCard"))
        .when(F.col("payment_type") == 2, F.lit("Cash"))
        .otherwise(F.lit("Other"))
    )

    route_type = (
        F.when(source_airport.isNotNull() & destination_airport.isNull(), F.lit("airport_to_city"))
        .when(source_airport.isNull() & destination_airport.isNotNull(), F.lit("city_to_airport"))
        .when(source_airport.isNotNull() & destination_airport.isNotNull(), F.lit("airport_to_airport"))
        .otherwise(F.lit("city_to_city"))
    )

    with_tokens = (
        filtered_df.withColumn("pickup_hour", pickup_hour)
        .withColumn("time_bin", time_bin)
        .withColumn("day_type", day_type)
        .withColumn("source_airport", source_airport)
        .withColumn("destination_airport", destination_airport)
        .withColumn("distance_bin", distance_bin)
        .withColumn("fare_bin", fare_bin)
        .withColumn("tip_bin", tip_bin)
        .withColumn("payment_type_label", payment_type_label)
        .withColumn("route_type", route_type)
        .withColumn(
            "source_geo_token",
            F.when(
                F.col("source_airport").isNotNull(),
                F.concat(F.lit("source_airport="), F.col("source_airport")),
            ).otherwise(F.concat(F.lit("source_borough="), F.col("source_borough"))),
        )
        .withColumn(
            "destination_geo_token",
            F.when(
                F.col("destination_airport").isNotNull(),
                F.concat(F.lit("destination_airport="), F.col("destination_airport")),
            ).otherwise(F.concat(F.lit("destination_borough="), F.col("destination_borough"))),
        )
        .withColumn("time_token", F.concat(F.lit("time_bin="), F.col("time_bin")))
        .withColumn("day_token", F.concat(F.lit("day_type="), F.col("day_type")))
        .withColumn("distance_token", F.concat(F.lit("distance_bin="), F.col("distance_bin")))
        .withColumn("fare_token", F.concat(F.lit("fare_bin="), F.col("fare_bin")))
        .withColumn("tip_token", F.concat(F.lit("tip_bin="), F.col("tip_bin")))
        .withColumn("payment_token", F.concat(F.lit("payment_type="), F.col("payment_type_label")))
        .withColumn("route_token", F.concat(F.lit("route_type="), F.col("route_type")))
        .filter(F.col("time_bin").isNotNull())
    )

    return with_tokens.select(
        F.array(
            "source_geo_token",
            "destination_geo_token",
            "time_token",
            "day_token",
            "distance_token",
            "fare_token",
            "tip_token",
            "payment_token",
            "route_token",
        ).alias("items")
    )


def _rule_to_text(antecedent: list[str], consequent: list[str]) -> str:
    left = ", ".join(sorted(antecedent))
    right = ", ".join(sorted(consequent))
    return f"{{{left}}} -> {{{right}}}"


def _has_prefix(tokens: list[str], prefix: str) -> bool:
    return any(token.startswith(prefix) for token in tokens)


def _extract_first_value(tokens: list[str], prefix: str) -> str | None:
    for token in tokens:
        if token.startswith(prefix):
            return token.split("=", 1)[1]
    return None


def _build_insight_text(antecedent: list[str], consequent: list[str]) -> str:
    combined = antecedent + consequent
    time_bin = _extract_first_value(combined, "time_bin=")
    day_type = _extract_first_value(combined, "day_type=")
    dist_bin = _extract_first_value(combined, "distance_bin=")
    fare_bin = _extract_first_value(combined, "fare_bin=")
    src_airport = _extract_first_value(combined, "source_airport=")
    dst_airport = _extract_first_value(combined, "destination_airport=")
    route_type = _extract_first_value(combined, "route_type=")

    if (src_airport or dst_airport) and route_type:
        movement = route_type.replace("_", " ")
        airport = src_airport or dst_airport
        time_text = f"during {time_bin.lower()}" if time_bin else "across time windows"
        day_text = f" on {day_type.lower()}s" if day_type else ""
        fare_text = f" with fares commonly in {fare_bin}" if fare_bin else ""
        return (
            f"Airport-linked trips ({movement}) are concentrated around {airport} "
            f"{time_text}{day_text}{fare_text}."
        )

    if time_bin and day_type and dist_bin and fare_bin:
        return (
            f"On {day_type.lower()} {time_bin.lower()} periods, trips in {dist_bin} "
            f"most often align with fares in {fare_bin}."
        )

    if time_bin and (dist_bin or fare_bin):
        fare_part = f" and fares in {fare_bin}" if fare_bin else ""
        dist_part = f"for {dist_bin} trips" if dist_bin else "for these trips"
        return f"{time_bin} travel patterns indicate consistent pricing behavior {dist_part}{fare_part}."

    return (
        "This pattern appears frequently and indicates a stable co-occurrence between "
        "trip context and route characteristics."
    )


def _is_interesting_rule(antecedent: list[str], consequent: list[str], lift: float, confidence: float) -> bool:
    combined = antecedent + consequent
    has_time = _has_prefix(combined, "time_bin=")
    has_airport = _has_prefix(combined, "source_airport=") or _has_prefix(combined, "destination_airport=")
    has_distance = _has_prefix(combined, "distance_bin=")
    has_fare = _has_prefix(combined, "fare_bin=")

    return (
        has_time
        and (has_airport or (has_distance and has_fare))
        and (lift >= 1.05)
        and (confidence >= 0.35)
    )


def _rank_rule(antecedent: list[str], consequent: list[str], support: float, confidence: float, lift: float) -> float:
    combined = antecedent + consequent
    has_airport = _has_prefix(combined, "source_airport=") or _has_prefix(combined, "destination_airport=")
    has_day = _has_prefix(combined, "day_type=")
    has_dist = _has_prefix(combined, "distance_bin=")
    has_fare = _has_prefix(combined, "fare_bin=")

    score = confidence * lift
    if has_airport:
        score *= 1.35
    if has_day:
        score *= 1.10
    if has_dist and has_fare:
        score *= 1.20
    score *= (1.0 + min(support, 0.2))
    return float(score)


def _build_showcase_top_rules(work: pd.DataFrame) -> pd.DataFrame:
    if work.empty:
        return pd.DataFrame(columns=["rule", "support", "confidence", "lift", "insight"])

    return pd.DataFrame(columns=["rule", "support", "confidence", "lift", "insight"])


def _build_rule_frames(rules_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rules_df.empty:
        empty = pd.DataFrame(columns=["rule", "support", "confidence", "lift", "insight"])
        return empty, empty

    work = rules_df.copy()
    work["antecedent"] = work["antecedent"].apply(lambda value: sorted(list(value)))
    work["consequent"] = work["consequent"].apply(lambda value: sorted(list(value)))
    work["rule"] = work.apply(
        lambda row: _rule_to_text(row["antecedent"], row["consequent"]),
        axis=1,
    )
    work["insight"] = work.apply(
        lambda row: _build_insight_text(row["antecedent"], row["consequent"]),
        axis=1,
    )

    work["interesting_flag"] = work.apply(
        lambda row: _is_interesting_rule(
            antecedent=row["antecedent"],
            consequent=row["consequent"],
            lift=float(row["lift"]),
            confidence=float(row["confidence"]),
        ),
        axis=1,
    )
    work["ranking_score"] = work.apply(
        lambda row: _rank_rule(
            antecedent=row["antecedent"],
            consequent=row["consequent"],
            support=float(row["support"]),
            confidence=float(row["confidence"]),
            lift=float(row["lift"]),
        ),
        axis=1,
    )

    all_rules = (
        work.sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
        [["rule", "support", "confidence", "lift", "insight"]]
        .reset_index(drop=True)
    )

    top10 = _build_showcase_top_rules(work=work).reset_index(drop=True)

    return all_rules, top10


def train_trip_pattern_rules(config: RuleMiningConfig | None = None) -> dict[str, Any]:
    logger = get_logger()
    cfg = config or RuleMiningConfig()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    spark = _create_spark_session(cfg.spark_app_name)
    try:
        logger.info("Preparing transaction basket from full trip dataset")
        transactions_df = _load_and_prepare_transaction_frame(spark)
        total_transactions = int(transactions_df.count())
        if total_transactions == 0:
            raise ValueError("No valid transactions found after filtering rules.")

        logger.info(
            "Training FP-Growth on %d transactions (minSupport=%.4f, minConfidence=%.2f)",
            total_transactions,
            cfg.min_support,
            cfg.min_confidence,
        )
        model = FPGrowth(
            itemsCol="items",
            minSupport=cfg.min_support,
            minConfidence=cfg.min_confidence,
        ).fit(transactions_df)

        rules_sdf = model.associationRules.select(
            "antecedent", "consequent", "support", "confidence", "lift"
        )
        rules_pdf = rules_sdf.toPandas()

        all_rules_df, top10_df = _build_rule_frames(rules_pdf)
        all_rules_df.to_csv(ALL_RULES_CSV, index=False)
        top10_df.to_csv(TOP_RULES_CSV, index=False)

        metadata = {
            "total_transactions_after_filtering": total_transactions,
            "total_rules_generated": int(len(all_rules_df)),
            "top_rules_count": int(len(top10_df)),
            "model": "Spark FPGrowth",
            "params": {
                "min_support": cfg.min_support,
                "min_confidence": cfg.min_confidence,
            },
            "filters": {
                "exclude_unknown_source_destination": True,
                "fare_amount_lt_1000": True,
                "trip_distance_lt_50": True,
            },
            "outputs": {
                "all_rules_csv": str(ALL_RULES_CSV),
                "top_rules_csv": str(TOP_RULES_CSV),
            },
        }
        METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info("Rule mining completed with %d total rules", len(all_rules_df))
        return metadata
    finally:
        spark.stop()
