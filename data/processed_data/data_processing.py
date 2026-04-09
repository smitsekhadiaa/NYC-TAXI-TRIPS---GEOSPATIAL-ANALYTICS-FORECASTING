"""NYC taxi trip data processing pipeline using PySpark (Python API).

This module builds:
- `mp`: dict where key is 3-letter month (e.g., 'jan') and value is a processed
  Spark DataFrame for that month.
- `all_trip_details`: an optional combined Spark DataFrame for all months.
"""

from __future__ import annotations

import calendar
import logging
import os
import re
import sys
from pathlib import Path
from shutil import which
from typing import Dict, Iterable, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

LOGGER_NAME = "nyc_taxi_data_processing"
MONTH_FILE_REGEX = re.compile(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet$")

DROP_COLUMNS = [
    "VendorID",
    "RatecodeID",
    "store_and_fwd_flag",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
]

SORT_COLUMNS = ["pickup_date", "pickup_time", "dropff_date", "dropff_time"]
MONTH_TO_NUMBER = {calendar.month_abbr[i].lower(): i for i in range(1, 13)}
RAW_SRC_DATA_DIR = Path(__file__).resolve().parents[1] / "raw_src_data"
MAX_TRIP_DURATION_SECONDS = 24 * 60 * 60

mp: Dict[str, DataFrame] = {}
all_trip_details: Optional[DataFrame] = None


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


def create_spark_session(app_name: str = "NYCTripDataProcessing") -> SparkSession:
    _configure_java_runtime()
    _configure_python_runtime_for_spark()
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "America/New_York")
        .getOrCreate()
    )


def discover_parquet_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("yellow_tripdata_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")
    return files


def _configure_java_runtime() -> None:
    env_prefix = Path(sys.prefix).resolve()
    candidate_java_homes = [
        env_prefix,
        env_prefix / "lib" / "jvm",
    ]

    for java_home in candidate_java_homes:
        java_bin = java_home / "bin" / "java"
        if java_bin.exists():
            os.environ["JAVA_HOME"] = str(java_home)
            current_path = os.environ.get("PATH", "")
            java_bin_dir = str(java_bin.parent)
            path_entries = current_path.split(":") if current_path else []
            if not path_entries or path_entries[0] != java_bin_dir:
                os.environ["PATH"] = (
                    f"{java_bin_dir}:{current_path}" if current_path else java_bin_dir
                )
            return

    if which("java") is None:
        raise RuntimeError(
            "Java runtime not found. Install openjdk in nycprojectenv to run PySpark."
        )


def _configure_python_runtime_for_spark() -> None:
    python_executable = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable


def parse_month_key(file_path: Path) -> str:
    match = MONTH_FILE_REGEX.search(file_path.name)
    if not match:
        raise ValueError(f"Unexpected file name format: {file_path.name}")

    month_number = int(match.group(2))
    return calendar.month_abbr[month_number].lower()


def _column_lookup(df: DataFrame) -> dict[str, str]:
    return {col_name.lower(): col_name for col_name in df.columns}


def _resolve_column(df: DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = _column_lookup(df)
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found:
            return found
    return None


def _require_column(df: DataFrame, candidates: Iterable[str], label: str) -> str:
    found = _resolve_column(df, candidates)
    if not found:
        raise ValueError(f"Required column for '{label}' not found. Candidates={list(candidates)}")
    return found


def _drop_existing_columns(df: DataFrame, candidate_columns: Iterable[str]) -> DataFrame:
    lookup = _column_lookup(df)
    to_drop = []
    for candidate in candidate_columns:
        actual = lookup.get(candidate.lower())
        if actual:
            to_drop.append(actual)

    if to_drop:
        df = df.drop(*to_drop)
    return df


def transform_trip_dataframe(
    df: DataFrame,
    logger: logging.Logger,
    sort_output: bool = True,
) -> DataFrame:
    df = df.dropna(how="any")

    pickup_dt_col = _require_column(df, ["tpep_pickup_datetime"], "pickup datetime")
    dropff_dt_col = _require_column(
        df,
        ["tpep_dropff_datetime", "tpep_dropoff_datetime"],
        "dropoff datetime",
    )

    pickup_location_col = _require_column(df, ["PULocationID"], "pickup location id")
    dropff_location_col = _require_column(df, ["DOLocationID"], "dropoff location id")
    trip_distance_col = _require_column(df, ["trip_distance"], "trip distance")

    fare_amount_col = _require_column(df, ["fare_amount"], "fare amount")
    tip_amount_col = _require_column(df, ["tip_amount"], "tip amount")

    extra_col = _require_column(df, ["extra"], "extra")
    tolls_col = _require_column(df, ["tolls_amount"], "tolls amount")
    congestion_col = _require_column(df, ["congestion_surcharge"], "congestion surcharge")
    airport_fee_col = _require_column(df, ["airport_fee", "Airport_fee"], "airport fee")

    df = (
        df.withColumn("pickup_date", F.to_date(F.col(pickup_dt_col)))
        .withColumn("pickup_time", F.date_format(F.col(pickup_dt_col), "HH:mm:ss"))
        .withColumn("dropff_date", F.to_date(F.col(dropff_dt_col)))
        .withColumn("dropff_time", F.date_format(F.col(dropff_dt_col), "HH:mm:ss"))
        .withColumn("_pickup_ts", F.to_timestamp(F.col(pickup_dt_col)))
        .withColumn("_dropff_ts", F.to_timestamp(F.col(dropff_dt_col)))
        .withColumn("trip_distance", F.col(trip_distance_col).cast("double"))
        .withColumn("fare_amount", F.col(fare_amount_col).cast("double"))
        .withColumn("tip_amount", F.col(tip_amount_col).cast("double"))
        .withColumn(
            "extra_amount",
            F.col(extra_col)
            + F.col(tolls_col)
            + F.col(congestion_col)
            + F.col(airport_fee_col),
        )
        .withColumn(
            "total_trip_amount",
            F.col(fare_amount_col) + F.col(tip_amount_col) + F.col("extra_amount"),
        )
    )

    if pickup_location_col != "pickup_location_id":
        df = df.withColumnRenamed(pickup_location_col, "pickup_location_id")
    if dropff_location_col != "dropff_location_id":
        df = df.withColumnRenamed(dropff_location_col, "dropff_location_id")

    df = (
        df.filter(
            (F.col("trip_distance") > 0)
            & (F.col("pickup_location_id") != F.col("dropff_location_id"))
            & F.col("_pickup_ts").isNotNull()
            & F.col("_dropff_ts").isNotNull()
            & (F.col("_dropff_ts") >= F.col("_pickup_ts"))
            & (
                (F.col("_dropff_ts").cast("long") - F.col("_pickup_ts").cast("long"))
                <= F.lit(MAX_TRIP_DURATION_SECONDS)
            )
            & (F.col("fare_amount") >= 0)
        )
        .dropDuplicates()
    )

    df = _drop_existing_columns(
        df,
        [
            *DROP_COLUMNS,
            "_pickup_ts",
            "_dropff_ts",
            "tpep_pickup_datetime",
            "tpep_dropff_datetime",
            "tpep_dropoff_datetime",
        ],
    )

    logger.debug("Transformed monthly dataframe columns: %s", df.columns)
    if sort_output:
        return df.orderBy(*SORT_COLUMNS)
    return df


def build_monthly_dataframe_map(
    spark: SparkSession,
    parquet_files: Iterable[Path],
    logger: logging.Logger,
    sort_output: bool = True,
) -> Dict[str, DataFrame]:
    month_map: Dict[str, DataFrame] = {}

    for file_path in parquet_files:
        month_key = parse_month_key(file_path)
        logger.info("Processing file '%s' as month key '%s'", file_path.name, month_key)

        monthly_df = spark.read.parquet(str(file_path))
        month_map[month_key] = transform_trip_dataframe(
            monthly_df,
            logger,
            sort_output=sort_output,
        )

    return month_map


def build_all_trip_details(
    month_map: Dict[str, DataFrame],
    logger: logging.Logger,
    sort_output: bool = True,
) -> Optional[DataFrame]:
    if not month_map:
        logger.warning("Month map is empty. Skipping combined dataframe creation.")
        return None

    try:
        ordered_keys = sorted(month_map.keys(), key=lambda key: MONTH_TO_NUMBER.get(key, 99))
        combined_df = month_map[ordered_keys[0]]

        for month_key in ordered_keys[1:]:
            combined_df = combined_df.unionByName(month_map[month_key])

        if sort_output:
            return combined_df.orderBy(*SORT_COLUMNS)
        return combined_df
    except Exception:
        logger.exception(
            "Failed to build all_trip_details due to runtime/memory constraints. Returning None."
        )
        return None


def build_processed_trip_data(
    data_dir: str | Path = RAW_SRC_DATA_DIR,
    spark: Optional[SparkSession] = None,
    create_combined_df: bool = True,
    sort_output: bool = True,
) -> Tuple[Dict[str, DataFrame], Optional[DataFrame]]:
    logger = get_logger()
    data_path = Path(data_dir)

    if spark is None:
        spark = create_spark_session()

    parquet_files = discover_parquet_files(data_path)
    logger.info("Found %d monthly parquet files under '%s'", len(parquet_files), data_path)

    month_map = build_monthly_dataframe_map(
        spark,
        parquet_files,
        logger,
        sort_output=sort_output,
    )
    combined_df = (
        build_all_trip_details(month_map, logger, sort_output=sort_output)
        if create_combined_df
        else None
    )

    return month_map, combined_df


def initialize_dataframes(
    data_dir: str | Path = RAW_SRC_DATA_DIR,
    spark: Optional[SparkSession] = None,
    create_combined_df: bool = True,
    force_refresh: bool = False,
    sort_output: bool = True,
) -> Tuple[Dict[str, DataFrame], Optional[DataFrame]]:
    global mp, all_trip_details

    if mp and not force_refresh:
        if create_combined_df and all_trip_details is None:
            all_trip_details = build_all_trip_details(mp, get_logger(), sort_output=sort_output)
        return mp, all_trip_details

    mp, all_trip_details = build_processed_trip_data(
        data_dir=data_dir,
        spark=spark,
        create_combined_df=create_combined_df,
        sort_output=sort_output,
    )
    return mp, all_trip_details


def get_processed_dataframes(
    data_dir: str | Path = RAW_SRC_DATA_DIR,
    spark: Optional[SparkSession] = None,
    create_combined_df: bool = True,
    force_refresh: bool = False,
    sort_output: bool = True,
) -> Tuple[Dict[str, DataFrame], Optional[DataFrame]]:
    return initialize_dataframes(
        data_dir=data_dir,
        spark=spark,
        create_combined_df=create_combined_df,
        force_refresh=force_refresh,
        sort_output=sort_output,
    )


def main() -> None:
    logger = get_logger()
    local_mp, local_all_trip_details = initialize_dataframes(create_combined_df=True)

    logger.info("Initialized mp with keys: %s", sorted(local_mp.keys()))
    if local_all_trip_details is None:
        logger.warning("all_trip_details was not created.")
    else:
        logger.info("all_trip_details DataFrame was created successfully.")


if __name__ == "__main__":
    main()
