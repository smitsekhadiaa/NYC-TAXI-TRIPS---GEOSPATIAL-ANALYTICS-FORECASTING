"""Train and serve anomaly detection artifacts for NYC taxi trips."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_CSV_DIR = DATA_DIR / "processed_csv"
LOCATION_ZONE_CSV = PROCESSED_CSV_DIR / "location_zone_data.csv"

ANOMALY_DIR = DATA_DIR / "anomaly_artifacts"
MODELS_DIR = ANOMALY_DIR / "models"
OUTPUTS_DIR = ANOMALY_DIR / "outputs"
SUMMARY_PATH = ANOMALY_DIR / "anomaly_summary.json"
THRESHOLDS_PATH = ANOMALY_DIR / "anomaly_thresholds.json"

EXTREME_SPEED_MODEL_PATH = MODELS_DIR / "extreme_speed_isolation_forest.pkl"
FARE_OUTLIER_MODEL_PATH = MODELS_DIR / "fare_outlier_isolation_forest.pkl"
FEATURE_BUNDLE_PATH = MODELS_DIR / "anomaly_feature_bundle.pkl"

EXTREME_SPEED_OUTPUT_PATH = OUTPUTS_DIR / "extreme_speed_anomalies.csv"
FARE_OUTLIER_OUTPUT_PATH = OUTPUTS_DIR / "fare_outlier_anomalies.csv"

RANDOM_STATE = 42
TRAIN_SAMPLE_PROBABILITY = 0.02
MAX_TRAIN_SAMPLE_ROWS = 500_000
SCORE_CHUNK_SIZE = 150_000
TARGET_ANOMALY_RATE = 0.01
CONTEXT_SAMPLE_PROBABILITY = 0.08
CONTEXT_MAX_SAMPLE_ROWS = 900_000

SPEED_FEATURE_COLUMNS = [
    "speed_mph",
    "trip_distance_miles",
    "eta_minutes",
    "fare_amount",
    "fare_per_mile",
    "pickup_hour",
    "day_of_week",
    "is_weekend",
    "pickup_hour_sin",
    "pickup_hour_cos",
]

FARE_FEATURE_COLUMNS = [
    "fare_amount",
    "trip_distance_miles",
    "eta_minutes",
    "fare_per_mile",
    "fare_per_minute",
    "pickup_hour",
    "day_of_week",
    "is_weekend",
    "context_median_fare",
    "context_iqr_fare",
    "context_count_log1p",
    "fare_to_context_median_ratio",
    "pickup_hour_sin",
    "pickup_hour_cos",
]

RAW_USECOLS = [
    "pickup_location_id",
    "dropff_location_id",
    "pickup_date",
    "pickup_time",
    "dropff_date",
    "dropff_time",
    "fare_amount",
    "trip_distance",
]


@dataclass(frozen=True)
class ContextStats:
    context_df: pd.DataFrame
    route_df: pd.DataFrame
    global_defaults: dict[str, float]


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _validate_inputs() -> None:
    if not LOCATION_ZONE_CSV.exists():
        raise FileNotFoundError(f"Missing required file: {LOCATION_ZONE_CSV}")
    trip_paths = sorted(PROCESSED_CSV_DIR.glob("trip_details_*.csv"))
    if not trip_paths:
        raise FileNotFoundError(f"No trip detail CSV files found in: {PROCESSED_CSV_DIR}")


def _hour_label(hour: int) -> str:
    return f"{int(hour):02d}:00"


def _load_zone_lookup() -> tuple[dict[int, str], dict[int, str]]:
    zone_df = pd.read_csv(LOCATION_ZONE_CSV, usecols=["location_id", "borough", "zone"])
    zone_df = zone_df.dropna(subset=["location_id"]).copy()
    zone_df["location_id"] = zone_df["location_id"].astype(int)
    zone_df["borough"] = zone_df["borough"].astype(str)
    zone_df["zone"] = zone_df["zone"].astype(str)

    zone_lookup: dict[int, str] = {}
    borough_lookup: dict[int, str] = {}
    for row in zone_df.itertuples(index=False):
        location_id = int(row.location_id)
        zone_lookup[location_id] = str(row.zone)
        borough_lookup[location_id] = str(row.borough)
    return zone_lookup, borough_lookup


def _load_training_sample() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    sampled_parts: list[pd.DataFrame] = []
    sampled_rows = 0

    trip_paths = sorted(PROCESSED_CSV_DIR.glob("trip_details_*.csv"))
    for trip_path in trip_paths:
        for raw_chunk in pd.read_csv(trip_path, usecols=RAW_USECOLS, chunksize=SCORE_CHUNK_SIZE):
            chunk = _prepare_raw_chunk(raw_chunk)
            if chunk.empty:
                continue

            sample_mask = rng.random(len(chunk)) <= TRAIN_SAMPLE_PROBABILITY
            if not sample_mask.any():
                continue

            sampled_chunk = chunk.loc[sample_mask].copy()
            remaining = MAX_TRAIN_SAMPLE_ROWS - sampled_rows
            if remaining <= 0:
                break
            if len(sampled_chunk) > remaining:
                sampled_chunk = sampled_chunk.iloc[:remaining].copy()

            sampled_parts.append(sampled_chunk)
            sampled_rows += len(sampled_chunk)

            if sampled_rows >= MAX_TRAIN_SAMPLE_ROWS:
                break
        if sampled_rows >= MAX_TRAIN_SAMPLE_ROWS:
            break

    if not sampled_parts:
        raise ValueError("Training sample is empty. Cannot train anomaly models.")
    sample_df = pd.concat(sampled_parts, ignore_index=True)
    if sample_df.empty:
        raise ValueError("Training sample is empty. Cannot train anomaly models.")
    return sample_df


def _load_context_stats() -> ContextStats:
    rng = np.random.default_rng(RANDOM_STATE + 11)
    sampled_parts: list[pd.DataFrame] = []
    sampled_rows = 0

    trip_paths = sorted(PROCESSED_CSV_DIR.glob("trip_details_*.csv"))
    for trip_path in trip_paths:
        for raw_chunk in pd.read_csv(trip_path, usecols=RAW_USECOLS, chunksize=SCORE_CHUNK_SIZE):
            chunk = _prepare_raw_chunk(raw_chunk)
            if chunk.empty:
                continue

            sample_mask = rng.random(len(chunk)) <= CONTEXT_SAMPLE_PROBABILITY
            if not sample_mask.any():
                continue

            sampled_chunk = chunk.loc[
                sample_mask,
                ["source_id", "destination_id", "pickup_hour", "is_weekend", "fare_amount"],
            ].copy()
            remaining = CONTEXT_MAX_SAMPLE_ROWS - sampled_rows
            if remaining <= 0:
                break
            if len(sampled_chunk) > remaining:
                sampled_chunk = sampled_chunk.iloc[:remaining].copy()

            sampled_parts.append(sampled_chunk)
            sampled_rows += len(sampled_chunk)
            if sampled_rows >= CONTEXT_MAX_SAMPLE_ROWS:
                break
        if sampled_rows >= CONTEXT_MAX_SAMPLE_ROWS:
            break

    if not sampled_parts:
        raise ValueError("Context sample is empty. Cannot build anomaly context stats.")

    sample_df = pd.concat(sampled_parts, ignore_index=True)
    sample_df["fare_amount"] = _safe_numeric(sample_df["fare_amount"], default_value=0.0)

    context_group = sample_df.groupby(["source_id", "destination_id", "pickup_hour", "is_weekend"])
    context_df = (
        context_group["fare_amount"]
        .agg(context_trip_count="count", context_fare_median="median")
        .reset_index()
    )
    context_q1 = context_group["fare_amount"].quantile(0.25).rename("context_fare_q1").reset_index()
    context_q3 = context_group["fare_amount"].quantile(0.75).rename("context_fare_q3").reset_index()
    context_df = context_df.merge(context_q1, on=["source_id", "destination_id", "pickup_hour", "is_weekend"])
    context_df = context_df.merge(context_q3, on=["source_id", "destination_id", "pickup_hour", "is_weekend"])

    route_group = sample_df.groupby(["source_id", "destination_id"])
    route_df = (
        route_group["fare_amount"]
        .agg(route_trip_count="count", route_fare_median="median")
        .reset_index()
    )
    route_q1 = route_group["fare_amount"].quantile(0.25).rename("route_fare_q1").reset_index()
    route_q3 = route_group["fare_amount"].quantile(0.75).rename("route_fare_q3").reset_index()
    route_df = route_df.merge(route_q1, on=["source_id", "destination_id"])
    route_df = route_df.merge(route_q3, on=["source_id", "destination_id"])

    global_defaults = {
        "global_fare_q1": float(sample_df["fare_amount"].quantile(0.25)),
        "global_fare_median": float(sample_df["fare_amount"].quantile(0.5)),
        "global_fare_q3": float(sample_df["fare_amount"].quantile(0.75)),
        "global_context_count": max(int(len(sample_df)), 1),
    }
    return ContextStats(
        context_df=context_df,
        route_df=route_df,
        global_defaults=global_defaults,
    )


def _safe_numeric(series: pd.Series, default_value: float = 0.0) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    if cleaned.notna().any():
        return cleaned.fillna(float(cleaned.median()))
    return cleaned.fillna(default_value)


def _with_time_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    hour_angle = 2.0 * np.pi * (df["pickup_hour"].astype(float) / 24.0)
    df["pickup_hour_sin"] = np.sin(hour_angle)
    df["pickup_hour_cos"] = np.cos(hour_angle)
    return df


def _attach_context_features(
    frame: pd.DataFrame,
    context_stats: ContextStats,
) -> pd.DataFrame:
    context_df = context_stats.context_df
    route_df = context_stats.route_df
    defaults = context_stats.global_defaults

    enriched = frame.merge(
        context_df,
        on=["source_id", "destination_id", "pickup_hour", "is_weekend"],
        how="left",
    )
    enriched = enriched.merge(route_df, on=["source_id", "destination_id"], how="left")

    for col_name in ["context_fare_q1", "context_fare_median", "context_fare_q3"]:
        route_col = col_name.replace("context", "route")
        fallback_value = defaults[f"global_fare_{col_name.split('_')[-1]}"]
        enriched[col_name] = enriched[col_name].fillna(enriched[route_col]).fillna(fallback_value)

    enriched["context_trip_count"] = (
        enriched["context_trip_count"]
        .fillna(enriched["route_trip_count"])
        .fillna(defaults["global_context_count"])
    )

    enriched["context_iqr_fare"] = np.maximum(
        enriched["context_fare_q3"] - enriched["context_fare_q1"],
        0.0,
    )
    enriched["context_median_fare"] = enriched["context_fare_median"]
    enriched["context_count_log1p"] = np.log1p(enriched["context_trip_count"].astype(float))
    enriched["fare_to_context_median_ratio"] = (
        enriched["fare_amount"] / np.clip(enriched["context_median_fare"], 0.1, None)
    )

    numeric_cols = [
        "context_fare_q1",
        "context_fare_median",
        "context_fare_q3",
        "context_median_fare",
        "context_trip_count",
        "context_iqr_fare",
        "context_count_log1p",
        "fare_to_context_median_ratio",
    ]
    for col_name in numeric_cols:
        enriched[col_name] = _safe_numeric(enriched[col_name], default_value=0.0)

    return enriched


def _build_speed_model() -> IsolationForest:
    return IsolationForest(
        n_estimators=180,
        max_samples=120_000,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _build_fare_model() -> IsolationForest:
    return IsolationForest(
        n_estimators=180,
        max_samples=120_000,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _calibrate_speed_thresholds(sample_df: pd.DataFrame) -> dict[str, Any]:
    candidate_score_q = [0.98, 0.985, 0.99]
    candidate_speed_q = [0.99, 0.995, 0.997]

    best: dict[str, Any] | None = None
    best_gap = float("inf")

    for score_q in candidate_score_q:
        for speed_q in candidate_speed_q:
            score_thr_by_hour = (
                sample_df.groupby("pickup_hour")["speed_anomaly_score"].quantile(score_q).to_dict()
            )
            speed_thr_by_hour = (
                sample_df.groupby("pickup_hour")["speed_mph"].quantile(speed_q).to_dict()
            )

            speed_score_thr = sample_df["pickup_hour"].map(score_thr_by_hour)
            speed_mph_thr = sample_df["pickup_hour"].map(speed_thr_by_hour)
            flags = (
                (sample_df["speed_anomaly_score"] >= speed_score_thr)
                & (sample_df["speed_mph"] >= speed_mph_thr)
            )
            anomaly_rate = float(flags.mean())
            gap = abs(anomaly_rate - TARGET_ANOMALY_RATE)

            if gap < best_gap:
                best_gap = gap
                best = {
                    "score_quantile": float(score_q),
                    "speed_quantile": float(speed_q),
                    "score_threshold_by_hour": {
                        int(hour): float(value) for hour, value in score_thr_by_hour.items()
                    },
                    "speed_threshold_by_hour": {
                        int(hour): float(value) for hour, value in speed_thr_by_hour.items()
                    },
                    "estimated_anomaly_rate": anomaly_rate,
                }

    if best is None:
        raise ValueError("Failed to calibrate speed anomaly thresholds.")

    global_score_threshold = float(sample_df["speed_anomaly_score"].quantile(best["score_quantile"]))
    global_speed_threshold = float(sample_df["speed_mph"].quantile(best["speed_quantile"]))

    per_hour_payload = []
    for hour in range(24):
        per_hour_payload.append(
            {
                "pickup_hour": int(hour),
                "hour_label": _hour_label(hour),
                "speed_score_threshold": float(
                    best["score_threshold_by_hour"].get(hour, global_score_threshold)
                ),
                "speed_mph_threshold": float(
                    best["speed_threshold_by_hour"].get(hour, global_speed_threshold)
                ),
            }
        )

    best["global_speed_score_threshold"] = global_score_threshold
    best["global_speed_mph_threshold"] = global_speed_threshold
    best["per_hour_thresholds"] = per_hour_payload
    return best


def _calibrate_fare_thresholds(sample_df: pd.DataFrame) -> dict[str, Any]:
    candidate_score_q = [0.98, 0.985, 0.99]
    candidate_iqr_multiplier = [1.5, 2.0, 2.5, 3.0]
    candidate_ratio_q = [0.97, 0.98, 0.99]

    best: dict[str, Any] | None = None
    best_gap = float("inf")

    for score_q in candidate_score_q:
        score_threshold = float(sample_df["fare_anomaly_score"].quantile(score_q))

        for iqr_multiplier in candidate_iqr_multiplier:
            context_upper = (
                sample_df["context_fare_median"] + iqr_multiplier * sample_df["context_iqr_fare"]
            )
            for ratio_q in candidate_ratio_q:
                ratio_threshold = float(sample_df["fare_to_context_median_ratio"].quantile(ratio_q))
                flags = (
                    (sample_df["fare_anomaly_score"] >= score_threshold)
                    & (sample_df["fare_amount"] >= context_upper)
                    & (sample_df["fare_to_context_median_ratio"] >= ratio_threshold)
                )
                anomaly_rate = float(flags.mean())
                gap = abs(anomaly_rate - TARGET_ANOMALY_RATE)

                if gap < best_gap:
                    best_gap = gap
                    best = {
                        "score_quantile": float(score_q),
                        "score_threshold": score_threshold,
                        "iqr_multiplier": float(iqr_multiplier),
                        "ratio_quantile": float(ratio_q),
                        "ratio_threshold": ratio_threshold,
                        "estimated_anomaly_rate": anomaly_rate,
                    }

    if best is None:
        raise ValueError("Failed to calibrate fare anomaly thresholds.")

    return best


def _prepare_raw_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    frame = chunk.copy()

    frame["source_id"] = pd.to_numeric(frame["pickup_location_id"], errors="coerce")
    frame["destination_id"] = pd.to_numeric(frame["dropff_location_id"], errors="coerce")
    frame["trip_distance_miles"] = pd.to_numeric(frame["trip_distance"], errors="coerce")
    frame["fare_amount"] = pd.to_numeric(frame["fare_amount"], errors="coerce")

    pickup_dt = pd.to_datetime(
        frame["pickup_date"].astype(str) + " " + frame["pickup_time"].astype(str),
        errors="coerce",
    )
    dropoff_dt = pd.to_datetime(
        frame["dropff_date"].astype(str) + " " + frame["dropff_time"].astype(str),
        errors="coerce",
    )

    frame["pickup_timestamp"] = pickup_dt
    frame["eta_minutes"] = (dropoff_dt - pickup_dt).dt.total_seconds() / 60.0
    frame["pickup_hour"] = pickup_dt.dt.hour
    frame["day_of_week"] = (pickup_dt.dt.dayofweek + 1) % 7
    frame["is_weekend"] = frame["day_of_week"].isin([0, 6]).astype(int)

    valid_mask = (
        frame["source_id"].notna()
        & frame["destination_id"].notna()
        & (frame["source_id"] != frame["destination_id"])
        & frame["trip_distance_miles"].notna()
        & (frame["trip_distance_miles"] > 0)
        & frame["fare_amount"].notna()
        & (frame["fare_amount"] >= 0)
        & frame["eta_minutes"].notna()
        & (frame["eta_minutes"] > 0)
        & (frame["eta_minutes"] <= 240)
        & frame["pickup_hour"].between(0, 23, inclusive="both")
    )

    frame = frame.loc[valid_mask].copy()
    if frame.empty:
        return frame

    frame["source_id"] = frame["source_id"].astype(int)
    frame["destination_id"] = frame["destination_id"].astype(int)
    frame["pickup_hour"] = frame["pickup_hour"].astype(int)
    frame["day_of_week"] = frame["day_of_week"].astype(int)

    frame["speed_mph"] = frame["trip_distance_miles"] / (frame["eta_minutes"] / 60.0)
    frame["fare_per_mile"] = frame["fare_amount"] / np.clip(frame["trip_distance_miles"], 1e-3, None)
    frame["fare_per_minute"] = frame["fare_amount"] / np.clip(frame["eta_minutes"], 1e-3, None)

    for col_name in [
        "trip_distance_miles",
        "fare_amount",
        "eta_minutes",
        "speed_mph",
        "fare_per_mile",
        "fare_per_minute",
    ]:
        frame[col_name] = _safe_numeric(frame[col_name], default_value=0.0)

    frame["pickup_date"] = frame["pickup_timestamp"].dt.date.astype(str)
    frame["pickup_time"] = frame["pickup_timestamp"].dt.strftime("%H:%M:%S")
    return frame[
        [
            "source_id",
            "destination_id",
            "pickup_date",
            "pickup_time",
            "pickup_timestamp",
            "pickup_hour",
            "day_of_week",
            "is_weekend",
            "trip_distance_miles",
            "fare_amount",
            "eta_minutes",
            "speed_mph",
            "fare_per_mile",
            "fare_per_minute",
        ]
    ]


def _fit_models_and_thresholds(context_stats: ContextStats) -> tuple[IsolationForest, IsolationForest, dict[str, Any], pd.DataFrame]:
    sample_df = _load_training_sample()
    sample_df = _attach_context_features(sample_df, context_stats=context_stats)
    sample_df = _with_time_cycle_features(sample_df)

    for col_name in SPEED_FEATURE_COLUMNS + FARE_FEATURE_COLUMNS:
        sample_df[col_name] = _safe_numeric(sample_df[col_name], default_value=0.0)

    speed_model = _build_speed_model()
    fare_model = _build_fare_model()

    speed_model.fit(sample_df[SPEED_FEATURE_COLUMNS])
    fare_model.fit(sample_df[FARE_FEATURE_COLUMNS])

    sample_df["speed_anomaly_score"] = -speed_model.decision_function(sample_df[SPEED_FEATURE_COLUMNS])
    sample_df["fare_anomaly_score"] = -fare_model.decision_function(sample_df[FARE_FEATURE_COLUMNS])

    speed_thresholds = _calibrate_speed_thresholds(sample_df)
    fare_thresholds = _calibrate_fare_thresholds(sample_df)

    threshold_bundle = {
        "target_anomaly_rate": TARGET_ANOMALY_RATE,
        "speed": speed_thresholds,
        "fare": fare_thresholds,
    }
    return speed_model, fare_model, threshold_bundle, sample_df


def _append_csv_rows(df: pd.DataFrame, path: Path, include_header: bool) -> None:
    if df.empty:
        return
    df.to_csv(path, mode="a", index=False, header=include_header)


def _prepare_anomaly_rows(
    frame: pd.DataFrame,
    zone_lookup: dict[int, str],
    borough_lookup: dict[int, str],
    score_column: str,
    anomaly_type: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["origin_zone"] = result["source_id"].map(zone_lookup).fillna("Unknown")
    result["origin_borough"] = result["source_id"].map(borough_lookup).fillna("Unknown")
    result["destination_zone"] = result["destination_id"].map(zone_lookup).fillna("Unknown")
    result["destination_borough"] = result["destination_id"].map(borough_lookup).fillna("Unknown")

    result["origin_label"] = result["origin_zone"] + " (" + result["origin_borough"] + ")"
    result["destination_label"] = result["destination_zone"] + " (" + result["destination_borough"] + ")"
    result["anomaly_type"] = anomaly_type
    result["anomaly_score"] = result[score_column].astype(float)

    result["pickup_timestamp"] = pd.to_datetime(result["pickup_timestamp"], errors="coerce")
    result["pickup_timestamp"] = result["pickup_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    columns = [
        "pickup_timestamp",
        "pickup_hour",
        "source_id",
        "destination_id",
        "origin_label",
        "destination_label",
        "fare_amount",
        "trip_distance_miles",
        "eta_minutes",
        "speed_mph",
        "fare_to_context_median_ratio",
        "context_median_fare",
        "context_iqr_fare",
        "anomaly_score",
        "anomaly_type",
    ]
    for col_name in columns:
        if col_name not in result.columns:
            result[col_name] = np.nan

    return result[columns].copy()


def _top_od_pairs(
    counter: Counter[tuple[int, int]],
    zone_lookup: dict[int, str],
    borough_lookup: dict[int, str],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (source_id, destination_id), frequency in counter.most_common(top_n):
        source_zone = zone_lookup.get(source_id, "Unknown")
        source_borough = borough_lookup.get(source_id, "Unknown")
        destination_zone = zone_lookup.get(destination_id, "Unknown")
        destination_borough = borough_lookup.get(destination_id, "Unknown")

        rows.append(
            {
                "source_id": int(source_id),
                "destination_id": int(destination_id),
                "origin": f"{source_zone} ({source_borough})",
                "destination": f"{destination_zone} ({destination_borough})",
                "anomaly_frequency": int(frequency),
            }
        )
    return rows


def _finalize_hourly_variation(hour_totals: np.ndarray, anomaly_totals: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hour in range(24):
        total_count = int(hour_totals[hour])
        anomaly_count = int(anomaly_totals[hour])
        percentage = (anomaly_count / total_count * 100.0) if total_count > 0 else 0.0
        rows.append(
            {
                "pickup_hour": int(hour),
                "hour_label": _hour_label(hour),
                "total_records": total_count,
                "anomaly_count": anomaly_count,
                "anomaly_percentage": round(float(percentage), 4),
            }
        )
    return rows


def _score_full_dataset(
    speed_model: IsolationForest,
    fare_model: IsolationForest,
    context_stats: ContextStats,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    zone_lookup, borough_lookup = _load_zone_lookup()

    if EXTREME_SPEED_OUTPUT_PATH.exists():
        EXTREME_SPEED_OUTPUT_PATH.unlink()
    if FARE_OUTLIER_OUTPUT_PATH.exists():
        FARE_OUTLIER_OUTPUT_PATH.unlink()

    write_speed_header = True
    write_fare_header = True

    total_valid_records = 0
    hour_totals = np.zeros(24, dtype=np.int64)

    speed_hour_counts = np.zeros(24, dtype=np.int64)
    fare_hour_counts = np.zeros(24, dtype=np.int64)

    speed_od_counter: Counter[tuple[int, int]] = Counter()
    fare_od_counter: Counter[tuple[int, int]] = Counter()

    speed_top_df = pd.DataFrame()
    fare_top_df = pd.DataFrame()

    speed_score_threshold_by_hour = {
        int(item["pickup_hour"]): float(item["speed_score_threshold"])
        for item in thresholds["speed"]["per_hour_thresholds"]
    }
    speed_mph_threshold_by_hour = {
        int(item["pickup_hour"]): float(item["speed_mph_threshold"])
        for item in thresholds["speed"]["per_hour_thresholds"]
    }

    global_speed_score_threshold = float(thresholds["speed"]["global_speed_score_threshold"])
    global_speed_mph_threshold = float(thresholds["speed"]["global_speed_mph_threshold"])

    fare_score_threshold = float(thresholds["fare"]["score_threshold"])
    fare_iqr_multiplier = float(thresholds["fare"]["iqr_multiplier"])
    fare_ratio_threshold = float(thresholds["fare"]["ratio_threshold"])

    trip_paths = sorted(PROCESSED_CSV_DIR.glob("trip_details_*.csv"))
    for trip_path in trip_paths:
        for raw_chunk in pd.read_csv(trip_path, usecols=RAW_USECOLS, chunksize=SCORE_CHUNK_SIZE):
            chunk = _prepare_raw_chunk(raw_chunk)
            if chunk.empty:
                continue

            chunk = _attach_context_features(chunk, context_stats=context_stats)
            chunk = _with_time_cycle_features(chunk)

            for col_name in SPEED_FEATURE_COLUMNS + FARE_FEATURE_COLUMNS:
                chunk[col_name] = _safe_numeric(chunk[col_name], default_value=0.0)

            chunk["speed_anomaly_score"] = -speed_model.decision_function(chunk[SPEED_FEATURE_COLUMNS])
            chunk["fare_anomaly_score"] = -fare_model.decision_function(chunk[FARE_FEATURE_COLUMNS])

            chunk["speed_score_threshold"] = (
                chunk["pickup_hour"].map(speed_score_threshold_by_hour).fillna(global_speed_score_threshold)
            )
            chunk["speed_mph_threshold"] = (
                chunk["pickup_hour"].map(speed_mph_threshold_by_hour).fillna(global_speed_mph_threshold)
            )

            chunk["is_extreme_speed_anomaly"] = (
                (chunk["speed_anomaly_score"] >= chunk["speed_score_threshold"])
                & (chunk["speed_mph"] >= chunk["speed_mph_threshold"])
            )

            chunk["context_upper_fare_threshold"] = (
                chunk["context_median_fare"] + fare_iqr_multiplier * chunk["context_iqr_fare"]
            )
            chunk["is_fare_outlier_anomaly"] = (
                (chunk["fare_anomaly_score"] >= fare_score_threshold)
                & (chunk["fare_amount"] >= chunk["context_upper_fare_threshold"])
                & (chunk["fare_to_context_median_ratio"] >= fare_ratio_threshold)
            )

            total_valid_records += int(len(chunk))
            hour_counts = chunk["pickup_hour"].value_counts()
            for hour_value, count_value in hour_counts.items():
                hour_totals[int(hour_value)] += int(count_value)

            speed_mask = chunk["is_extreme_speed_anomaly"]
            fare_mask = chunk["is_fare_outlier_anomaly"]

            if speed_mask.any():
                speed_chunk = chunk.loc[speed_mask].copy()
                speed_counts = speed_chunk["pickup_hour"].value_counts()
                for hour_value, count_value in speed_counts.items():
                    speed_hour_counts[int(hour_value)] += int(count_value)

                speed_pair_counts = speed_chunk.groupby(["source_id", "destination_id"]).size()
                for (source_id, destination_id), count_value in speed_pair_counts.items():
                    speed_od_counter[(int(source_id), int(destination_id))] += int(count_value)

                speed_rows = _prepare_anomaly_rows(
                    frame=speed_chunk,
                    zone_lookup=zone_lookup,
                    borough_lookup=borough_lookup,
                    score_column="speed_anomaly_score",
                    anomaly_type="extreme_speed",
                )
                _append_csv_rows(speed_rows, EXTREME_SPEED_OUTPUT_PATH, include_header=write_speed_header)
                write_speed_header = False

                speed_top_df = pd.concat([speed_top_df, speed_rows], ignore_index=True)
                speed_top_df = speed_top_df.nlargest(10, columns="anomaly_score").reset_index(drop=True)

            if fare_mask.any():
                fare_chunk = chunk.loc[fare_mask].copy()
                fare_counts = fare_chunk["pickup_hour"].value_counts()
                for hour_value, count_value in fare_counts.items():
                    fare_hour_counts[int(hour_value)] += int(count_value)

                fare_pair_counts = fare_chunk.groupby(["source_id", "destination_id"]).size()
                for (source_id, destination_id), count_value in fare_pair_counts.items():
                    fare_od_counter[(int(source_id), int(destination_id))] += int(count_value)

                fare_rows = _prepare_anomaly_rows(
                    frame=fare_chunk,
                    zone_lookup=zone_lookup,
                    borough_lookup=borough_lookup,
                    score_column="fare_anomaly_score",
                    anomaly_type="fare_outlier",
                )
                _append_csv_rows(fare_rows, FARE_OUTLIER_OUTPUT_PATH, include_header=write_fare_header)
                write_fare_header = False

                fare_top_df = pd.concat([fare_top_df, fare_rows], ignore_index=True)
                fare_top_df = fare_top_df.nlargest(10, columns="anomaly_score").reset_index(drop=True)

    speed_total = int(speed_hour_counts.sum())
    fare_total = int(fare_hour_counts.sum())

    speed_variation = _finalize_hourly_variation(hour_totals, speed_hour_counts)
    fare_variation = _finalize_hourly_variation(hour_totals, fare_hour_counts)

    total_safe = max(total_valid_records, 1)

    speed_top_records = speed_top_df.copy()
    if not speed_top_records.empty:
        speed_top_records["fare_amount"] = speed_top_records["fare_amount"].round(2)
        speed_top_records["trip_distance_miles"] = speed_top_records["trip_distance_miles"].round(3)
        speed_top_records["eta_minutes"] = speed_top_records["eta_minutes"].round(2)
        speed_top_records["speed_mph"] = speed_top_records["speed_mph"].round(2)
        speed_top_records["anomaly_score"] = speed_top_records["anomaly_score"].round(6)

    fare_top_records = fare_top_df.copy()
    if not fare_top_records.empty:
        fare_top_records["fare_amount"] = fare_top_records["fare_amount"].round(2)
        fare_top_records["trip_distance_miles"] = fare_top_records["trip_distance_miles"].round(3)
        fare_top_records["eta_minutes"] = fare_top_records["eta_minutes"].round(2)
        fare_top_records["fare_to_context_median_ratio"] = fare_top_records[
            "fare_to_context_median_ratio"
        ].round(3)
        fare_top_records["context_median_fare"] = fare_top_records["context_median_fare"].round(2)
        fare_top_records["context_iqr_fare"] = fare_top_records["context_iqr_fare"].round(2)
        fare_top_records["anomaly_score"] = fare_top_records["anomaly_score"].round(6)

    payload = {
        "total_valid_records": int(total_valid_records),
        "extreme_speed": {
            "total_anomaly_records": speed_total,
            "anomaly_percentage": round((speed_total / total_safe) * 100.0, 4),
            "hourly_variation": speed_variation,
            "top_od_pairs": _top_od_pairs(speed_od_counter, zone_lookup, borough_lookup, top_n=10),
            "top_10_records": speed_top_records.to_dict(orient="records"),
            "output_csv": str(EXTREME_SPEED_OUTPUT_PATH),
        },
        "fare_outlier": {
            "total_anomaly_records": fare_total,
            "anomaly_percentage": round((fare_total / total_safe) * 100.0, 4),
            "hourly_variation": fare_variation,
            "top_od_pairs": _top_od_pairs(fare_od_counter, zone_lookup, borough_lookup, top_n=10),
            "top_10_records": fare_top_records.to_dict(orient="records"),
            "output_csv": str(FARE_OUTLIER_OUTPUT_PATH),
        },
    }
    return payload


def train_anomaly_detection_models() -> dict[str, Any]:
    _ensure_dirs()
    _validate_inputs()

    context_stats = _load_context_stats()
    speed_model, fare_model, threshold_bundle, sample_df = _fit_models_and_thresholds(context_stats)

    joblib.dump(speed_model, EXTREME_SPEED_MODEL_PATH)
    joblib.dump(fare_model, FARE_OUTLIER_MODEL_PATH)

    feature_bundle = {
        "speed_feature_columns": SPEED_FEATURE_COLUMNS,
        "fare_feature_columns": FARE_FEATURE_COLUMNS,
        "thresholds": threshold_bundle,
        "training_sample_size": int(len(sample_df)),
        "training_sample_probability": TRAIN_SAMPLE_PROBABILITY,
        "max_training_sample_rows": MAX_TRAIN_SAMPLE_ROWS,
    }
    joblib.dump(feature_bundle, FEATURE_BUNDLE_PATH)

    scored_payload = _score_full_dataset(
        speed_model=speed_model,
        fare_model=fare_model,
        context_stats=context_stats,
        thresholds=threshold_bundle,
    )

    summary_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "models": {
            "extreme_speed_model": str(EXTREME_SPEED_MODEL_PATH),
            "fare_outlier_model": str(FARE_OUTLIER_MODEL_PATH),
            "feature_bundle": str(FEATURE_BUNDLE_PATH),
        },
        "thresholds_json": str(THRESHOLDS_PATH),
        "thresholds": {
            "extreme_speed": {
                "model": "IsolationForest",
                "target_anomaly_rate": TARGET_ANOMALY_RATE,
                "selected_score_quantile": threshold_bundle["speed"]["score_quantile"],
                "selected_speed_quantile": threshold_bundle["speed"]["speed_quantile"],
                "global_speed_score_threshold": threshold_bundle["speed"][
                    "global_speed_score_threshold"
                ],
                "global_speed_mph_threshold": threshold_bundle["speed"][
                    "global_speed_mph_threshold"
                ],
                "per_hour_thresholds": threshold_bundle["speed"]["per_hour_thresholds"],
            },
            "fare_outlier": {
                "model": "IsolationForest",
                "target_anomaly_rate": TARGET_ANOMALY_RATE,
                "selected_score_quantile": threshold_bundle["fare"]["score_quantile"],
                "score_threshold": threshold_bundle["fare"]["score_threshold"],
                "iqr_multiplier": threshold_bundle["fare"]["iqr_multiplier"],
                "selected_ratio_quantile": threshold_bundle["fare"]["ratio_quantile"],
                "ratio_threshold": threshold_bundle["fare"]["ratio_threshold"],
                "fare_upper_formula": "context_median_fare + iqr_multiplier * context_iqr_fare",
            },
        },
        "anomalies": {
            "extreme_speed": scored_payload["extreme_speed"],
            "fare_outlier": scored_payload["fare_outlier"],
        },
        "total_valid_records": scored_payload["total_valid_records"],
    }
    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as outfile:
        json.dump(summary_payload["thresholds"], outfile, indent=2)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as outfile:
        json.dump(summary_payload, outfile, indent=2)

    return summary_payload


def get_anomaly_detection_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Anomaly summary not found. Run backend CLI command: "
            "'flask --app run.py train-anomaly-models'"
        )
    with open(SUMMARY_PATH, "r", encoding="utf-8") as infile:
        return json.load(infile)
