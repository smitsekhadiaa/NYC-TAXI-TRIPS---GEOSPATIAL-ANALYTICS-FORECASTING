"""Train and serve fare/ETA forecasting models from processed trip CSV data."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOGGER_NAME = "nyc_taxi_trip_forecasting"
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_CSV_DIR = DATA_DIR / "processed_csv"
TRIP_CSV_GLOB = str(PROCESSED_CSV_DIR / "trip_details_*.csv")
LOCATION_COORDINATES_CSV = PROCESSED_CSV_DIR / "location_coordinates_data.csv"

FORECAST_DIR = DATA_DIR / "forecasting_artifacts"
MODELS_DIR = FORECAST_DIR / "models"
OUTPUTS_DIR = FORECAST_DIR / "outputs"
METRICS_DIR = FORECAST_DIR / "metrics"
METADATA_PATH = FORECAST_DIR / "forecasting_metadata.json"

FARE_MODEL_PATH = MODELS_DIR / "fare_model.pkl"
ETA_MODEL_PATH = MODELS_DIR / "eta_model.pkl"
FEATURE_BUNDLE_PATH = MODELS_DIR / "feature_bundle.pkl"
TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions_oct_nov.csv"
TEST_METRICS_PATH = METRICS_DIR / "test_metrics.json"

TRAIN_MONTH_START = 1
TRAIN_MONTH_END = 9
TEST_MONTH_START = 10
TEST_MONTH_END = 11
SERVING_DATE_START = date(2025, 10, 1)
SERVING_DATE_END = date(2025, 12, 31)

FEATURE_COLUMNS = [
    "source_id",
    "destination_id",
    "pickup_hour",
    "day_of_week",
    "is_weekend",
    "month",
    "route_distance_km",
    "route_hist_avg_fare",
    "route_hist_avg_eta",
    "route_hist_trip_count_log1p",
]
ETA_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "predicted_fare_feature"]
ETA_TARGET_CLIP_QUANTILE = 0.999


@dataclass(frozen=True)
class ForecastModels:
    fare_model: HistGradientBoostingRegressor
    eta_model: HistGradientBoostingRegressor
    feature_bundle: dict[str, Any]


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


def _ensure_artifact_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _validate_input_files() -> None:
    trip_csv_paths = sorted(PROCESSED_CSV_DIR.glob("trip_details_*.csv"))
    if not trip_csv_paths:
        raise FileNotFoundError(f"No trip CSV files found in {PROCESSED_CSV_DIR}")
    if not LOCATION_COORDINATES_CSV.exists():
        raise FileNotFoundError(f"Missing location coordinates CSV: {LOCATION_COORDINATES_CSV}")


def _load_aggregated_trip_frame() -> pd.DataFrame:
    trip_glob_literal = TRIP_CSV_GLOB.replace("'", "''")
    sql = f"""
    WITH raw AS (
        SELECT
            CAST(pickup_location_id AS INTEGER) AS source_id,
            CAST(dropff_location_id AS INTEGER) AS destination_id,
            CAST(pickup_date AS DATE) AS pickup_date,
            TRY_CAST(SUBSTR(CAST(pickup_time AS VARCHAR), 1, 2) AS INTEGER) AS pickup_hour,
            CAST(fare_amount AS DOUBLE) AS fare_amount,
            date_diff(
                'minute',
                TRY_CAST(CAST(pickup_date AS VARCHAR) || ' ' || CAST(pickup_time AS VARCHAR) AS TIMESTAMP),
                TRY_CAST(CAST(dropff_date AS VARCHAR) || ' ' || CAST(dropff_time AS VARCHAR) AS TIMESTAMP)
            ) AS eta_minutes
        FROM read_csv_auto(
            '{trip_glob_literal}',
            header = true,
            union_by_name = true
        )
    ),
    valid AS (
        SELECT
            source_id,
            destination_id,
            pickup_date,
            pickup_hour,
            fare_amount,
            eta_minutes,
            CAST(strftime(pickup_date, '%m') AS INTEGER) AS month,
            CAST(strftime(pickup_date, '%w') AS INTEGER) AS day_of_week
        FROM raw
        WHERE source_id IS NOT NULL
          AND destination_id IS NOT NULL
          AND source_id <> destination_id
          AND pickup_date IS NOT NULL
          AND pickup_hour BETWEEN 0 AND 23
          AND fare_amount >= 0
          AND eta_minutes > 0
          AND eta_minutes <= 1440
    )
    SELECT
        source_id,
        destination_id,
        pickup_hour,
        day_of_week,
        CASE WHEN day_of_week IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
        month,
        AVG(fare_amount) AS avg_fare_amount,
        AVG(eta_minutes) AS avg_eta_minutes,
        COUNT(*) AS trip_count
    FROM valid
    WHERE month BETWEEN 1 AND 11
    GROUP BY
        source_id,
        destination_id,
        pickup_hour,
        day_of_week,
        is_weekend,
        month
    """
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("PRAGMA threads=4")
        aggregated = connection.execute(sql).df()
    finally:
        connection.close()

    if aggregated.empty:
        raise ValueError("Aggregated trip frame is empty. Cannot train forecasting models.")
    return aggregated


def _split_train_test(aggregated_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = aggregated_df.loc[
        (aggregated_df["month"] >= TRAIN_MONTH_START) & (aggregated_df["month"] <= TRAIN_MONTH_END)
    ].copy()
    test_df = aggregated_df.loc[
        (aggregated_df["month"] >= TEST_MONTH_START) & (aggregated_df["month"] <= TEST_MONTH_END)
    ].copy()

    if train_df.empty:
        raise ValueError("Training split (Jan-Sep) is empty.")
    if test_df.empty:
        raise ValueError("Test split (Oct-Nov) is empty.")
    return train_df, test_df


def _build_route_prior_frame(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    tmp_df = train_df.copy()
    tmp_df["fare_weighted_sum"] = tmp_df["avg_fare_amount"] * tmp_df["trip_count"]
    tmp_df["eta_weighted_sum"] = tmp_df["avg_eta_minutes"] * tmp_df["trip_count"]

    route_prior_df = (
        tmp_df.groupby(["source_id", "destination_id"], as_index=False)
        .agg(
            route_hist_trip_count=("trip_count", "sum"),
            fare_weighted_sum=("fare_weighted_sum", "sum"),
            eta_weighted_sum=("eta_weighted_sum", "sum"),
        )
    )
    route_prior_df["route_hist_avg_fare"] = (
        route_prior_df["fare_weighted_sum"] / route_prior_df["route_hist_trip_count"]
    )
    route_prior_df["route_hist_avg_eta"] = (
        route_prior_df["eta_weighted_sum"] / route_prior_df["route_hist_trip_count"]
    )
    route_prior_df["route_hist_trip_count_log1p"] = np.log1p(route_prior_df["route_hist_trip_count"])
    route_prior_df = route_prior_df[
        [
            "source_id",
            "destination_id",
            "route_hist_avg_fare",
            "route_hist_avg_eta",
            "route_hist_trip_count_log1p",
        ]
    ].copy()

    global_defaults = {
        "route_hist_avg_fare": float(
            np.average(train_df["avg_fare_amount"], weights=train_df["trip_count"])
        ),
        "route_hist_avg_eta": float(
            np.average(train_df["avg_eta_minutes"], weights=train_df["trip_count"])
        ),
        "route_hist_trip_count_log1p": 0.0,
    }
    return route_prior_df, global_defaults


def _haversine_km_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    radius_km = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_km * c


def _load_coordinates_frame() -> pd.DataFrame:
    coords_df = pd.read_csv(LOCATION_COORDINATES_CSV)
    coords_df = coords_df.dropna(subset=["location_id", "lat", "long"]).copy()
    coords_df["location_id"] = coords_df["location_id"].astype(int)
    coords_df["lat"] = coords_df["lat"].astype(float)
    coords_df["long"] = coords_df["long"].astype(float)
    return coords_df


def _add_route_distance_feature(frame: pd.DataFrame, coords_df: pd.DataFrame) -> pd.DataFrame:
    source_coords = coords_df.rename(
        columns={
            "location_id": "source_id",
            "lat": "source_lat",
            "long": "source_long",
        }
    )
    destination_coords = coords_df.rename(
        columns={
            "location_id": "destination_id",
            "lat": "destination_lat",
            "long": "destination_long",
        }
    )

    merged = (
        frame.merge(source_coords, on="source_id", how="left")
        .merge(destination_coords, on="destination_id", how="left")
    )
    merged["route_distance_km"] = _haversine_km_vectorized(
        lat1=merged["source_lat"].astype(float).to_numpy(),
        lon1=merged["source_long"].astype(float).to_numpy(),
        lat2=merged["destination_lat"].astype(float).to_numpy(),
        lon2=merged["destination_long"].astype(float).to_numpy(),
    )
    merged["route_distance_km"] = merged["route_distance_km"].replace([np.inf, -np.inf], np.nan)
    median_distance = float(merged["route_distance_km"].dropna().median())
    merged["route_distance_km"] = merged["route_distance_km"].fillna(median_distance)
    return merged.drop(
        columns=["source_lat", "source_long", "destination_lat", "destination_long"]
    )


def _prepare_feature_frame(
    frame: pd.DataFrame,
    route_prior_df: pd.DataFrame,
    global_defaults: dict[str, float],
    coords_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched = frame.merge(
        route_prior_df,
        on=["source_id", "destination_id"],
        how="left",
    )
    for col_name, default_value in global_defaults.items():
        enriched[col_name] = enriched[col_name].fillna(default_value)

    enriched = _add_route_distance_feature(enriched, coords_df)

    for col_name in ("source_id", "destination_id", "pickup_hour", "day_of_week", "is_weekend", "month"):
        enriched[col_name] = enriched[col_name].astype(int)

    feature_frame = enriched[FEATURE_COLUMNS].copy()
    return feature_frame


def _build_fare_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=500,
        max_depth=8,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def _build_eta_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=450,
        max_depth=10,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_trip_forecasting_models() -> dict:
    logger = get_logger()
    _ensure_artifact_dirs()
    _validate_input_files()

    logger.info("Loading aggregated trip features from processed CSVs")
    aggregated_df = _load_aggregated_trip_frame()
    train_df, test_df = _split_train_test(aggregated_df)

    coords_df = _load_coordinates_frame()
    route_prior_df, global_defaults = _build_route_prior_frame(train_df)

    logger.info("Preparing train/test feature matrices")
    train_X = _prepare_feature_frame(train_df, route_prior_df, global_defaults, coords_df)
    test_X = _prepare_feature_frame(test_df, route_prior_df, global_defaults, coords_df)

    sample_weight_train = train_df["trip_count"].astype(float).to_numpy()
    sample_weight_test = test_df["trip_count"].astype(float).to_numpy()

    train_y_fare = train_df["avg_fare_amount"].astype(float).to_numpy()
    train_y_eta = train_df["avg_eta_minutes"].astype(float).to_numpy()
    test_y_fare = test_df["avg_fare_amount"].astype(float).to_numpy()
    test_y_eta = test_df["avg_eta_minutes"].astype(float).to_numpy()

    fare_model = _build_fare_model()
    eta_model = _build_eta_model()

    logger.info("Training fare model")
    fare_model.fit(train_X, train_y_fare, sample_weight=sample_weight_train)
    logger.info("Training ETA model")
    eta_clip_upper = float(np.quantile(train_y_eta, ETA_TARGET_CLIP_QUANTILE))
    eta_train_transformed = np.log1p(np.clip(train_y_eta, 0.0, eta_clip_upper))
    eta_train_X = train_X.copy()
    eta_test_X = test_X.copy()
    eta_train_X["predicted_fare_feature"] = fare_model.predict(train_X)
    eta_test_X["predicted_fare_feature"] = fare_model.predict(test_X)
    eta_model.fit(eta_train_X[ETA_FEATURE_COLUMNS], eta_train_transformed, sample_weight=sample_weight_train)

    test_pred_fare = fare_model.predict(test_X)
    test_pred_eta = np.expm1(eta_model.predict(eta_test_X[ETA_FEATURE_COLUMNS]))

    metrics_payload = {
        "split": {
            "train_months": [TRAIN_MONTH_START, TRAIN_MONTH_END],
            "test_months": [TEST_MONTH_START, TEST_MONTH_END],
        },
        "sample_sizes": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_weighted_rows": int(sample_weight_train.sum()),
            "test_weighted_rows": int(sample_weight_test.sum()),
        },
        "fare_model_test_metrics": _regression_metrics(test_y_fare, test_pred_fare),
        "eta_model_test_metrics": _regression_metrics(test_y_eta, test_pred_eta),
    }

    test_predictions_df = test_df[
        [
            "source_id",
            "destination_id",
            "pickup_hour",
            "day_of_week",
            "is_weekend",
            "month",
            "trip_count",
        ]
    ].copy()
    test_predictions_df["actual_avg_fare"] = test_y_fare
    test_predictions_df["predicted_avg_fare"] = test_pred_fare
    test_predictions_df["abs_error_fare"] = np.abs(test_y_fare - test_pred_fare)
    test_predictions_df["actual_avg_eta_minutes"] = test_y_eta
    test_predictions_df["predicted_avg_eta_minutes"] = test_pred_eta
    test_predictions_df["abs_error_eta_minutes"] = np.abs(test_y_eta - test_pred_eta)

    route_prior_lookup = {
        (int(row.source_id), int(row.destination_id)): {
            "route_hist_avg_fare": float(row.route_hist_avg_fare),
            "route_hist_avg_eta": float(row.route_hist_avg_eta),
            "route_hist_trip_count_log1p": float(row.route_hist_trip_count_log1p),
        }
        for row in route_prior_df.itertuples(index=False)
    }
    coord_lookup = {
        int(row.location_id): (float(row.lat), float(row.long))
        for row in coords_df.itertuples(index=False)
    }
    feature_bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "global_defaults": global_defaults,
        "route_prior_lookup": route_prior_lookup,
        "coord_lookup": coord_lookup,
        "eta_model_target_transform": "log1p_expm1",
        "eta_uses_predicted_fare_feature": True,
        "serving_date_start": SERVING_DATE_START.isoformat(),
        "serving_date_end": SERVING_DATE_END.isoformat(),
    }

    logger.info("Saving forecasting model artifacts")
    joblib.dump(fare_model, FARE_MODEL_PATH)
    joblib.dump(eta_model, ETA_MODEL_PATH)
    joblib.dump(feature_bundle, FEATURE_BUNDLE_PATH)

    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as outfile:
        json.dump(metrics_payload, outfile, indent=2)
    test_predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    metadata_payload = {
        "models": {
            "fare_model": str(FARE_MODEL_PATH),
            "eta_model": str(ETA_MODEL_PATH),
            "feature_bundle": str(FEATURE_BUNDLE_PATH),
        },
        "outputs": {
            "test_predictions_csv": str(TEST_PREDICTIONS_PATH),
            "test_metrics_json": str(TEST_METRICS_PATH),
        },
        "serving_window": {
            "start_date": SERVING_DATE_START.isoformat(),
            "end_date": SERVING_DATE_END.isoformat(),
            "hour_range_inclusive": [0, 24],
        },
        "split": metrics_payload["split"],
        "sample_sizes": metrics_payload["sample_sizes"],
        "fare_model_test_metrics": metrics_payload["fare_model_test_metrics"],
        "eta_model_test_metrics": metrics_payload["eta_model_test_metrics"],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as outfile:
        json.dump(metadata_payload, outfile, indent=2)

    _load_models_cached.cache_clear()
    return metadata_payload


def _haversine_single(
    source_lat: float,
    source_long: float,
    destination_lat: float,
    destination_long: float,
) -> float:
    radius_km = 6371.0
    phi1 = math.radians(source_lat)
    phi2 = math.radians(destination_lat)
    delta_phi = math.radians(destination_lat - source_lat)
    delta_lambda = math.radians(destination_long - source_long)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_km * c


@lru_cache(maxsize=1)
def _load_models_cached() -> ForecastModels:
    if not FARE_MODEL_PATH.exists() or not ETA_MODEL_PATH.exists() or not FEATURE_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "Trip forecasting model artifacts not found. Run backend CLI command: "
            "'flask --app run.py train-trip-forecast'"
        )

    fare_model = joblib.load(FARE_MODEL_PATH)
    eta_model = joblib.load(ETA_MODEL_PATH)
    feature_bundle = joblib.load(FEATURE_BUNDLE_PATH)
    return ForecastModels(
        fare_model=fare_model,
        eta_model=eta_model,
        feature_bundle=feature_bundle,
    )


def _normalize_pickup_datetime(pickup_date: date, pickup_hour: int) -> datetime:
    if pickup_hour < 0 or pickup_hour > 24:
        raise ValueError("pickup_hour must be in range [0, 24].")

    if pickup_hour == 24:
        return datetime.combine(pickup_date, datetime.min.time()) + timedelta(days=1)
    return datetime.combine(pickup_date, datetime.min.time()) + timedelta(hours=pickup_hour)


def _build_single_feature_row(
    source_id: int,
    destination_id: int,
    pickup_dt: datetime,
    feature_bundle: dict[str, Any],
) -> pd.DataFrame:
    route_prior_lookup = feature_bundle["route_prior_lookup"]
    global_defaults = feature_bundle["global_defaults"]
    coord_lookup = feature_bundle["coord_lookup"]

    route_prior = route_prior_lookup.get((source_id, destination_id), global_defaults)

    source_coords = coord_lookup.get(source_id)
    destination_coords = coord_lookup.get(destination_id)
    if source_coords and destination_coords:
        route_distance_km = _haversine_single(
            source_lat=source_coords[0],
            source_long=source_coords[1],
            destination_lat=destination_coords[0],
            destination_long=destination_coords[1],
        )
    else:
        route_distance_km = 0.0

    day_of_week = int(pickup_dt.strftime("%w"))
    is_weekend = 1 if day_of_week in (0, 6) else 0

    row = {
        "source_id": int(source_id),
        "destination_id": int(destination_id),
        "pickup_hour": int(pickup_dt.hour),
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "month": int(pickup_dt.month),
        "route_distance_km": float(route_distance_km),
        "route_hist_avg_fare": float(route_prior["route_hist_avg_fare"]),
        "route_hist_avg_eta": float(route_prior["route_hist_avg_eta"]),
        "route_hist_trip_count_log1p": float(route_prior["route_hist_trip_count_log1p"]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)