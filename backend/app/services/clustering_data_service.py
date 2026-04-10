"""Service layer for serving precomputed clustering map payloads."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

LOGGER_NAME = "nyc_taxi_clustering_data_service"
VALID_SEGMENTS = {"morning", "afternoon", "evening", "night", "all"}

CLUSTER_COLORS = {
    0: [183, 28, 28, 235],    # dark red
    1: [27, 94, 32, 235],     # dark green
    2: [245, 196, 0, 235],    # dark yellow
    3: [13, 71, 161, 235],    # dark blue
    4: [74, 20, 140, 235],    # dark purple
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CLUSTERING_DIR = DATA_DIR / "clustering_artifacts"
OUTPUTS_DIR = CLUSTERING_DIR / "outputs"
METADATA_PATH = CLUSTERING_DIR / "clustering_metadata.json"
PROCESSED_CSV_DIR = DATA_DIR / "processed_csv"
LOCATION_ZONE_CSV = PROCESSED_CSV_DIR / "location_zone_data.csv"
LOCATION_COORDINATES_CSV = PROCESSED_CSV_DIR / "location_coordinates_data.csv"


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


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


@lru_cache(maxsize=1)
def _load_metadata() -> dict:
    _ensure_exists(METADATA_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as infile:
        return json.load(infile)


@lru_cache(maxsize=8)
def _load_segment_points(segment: str) -> pd.DataFrame:
    csv_path = OUTPUTS_DIR / f"sample_points_labeled_{segment}.csv"
    _ensure_exists(csv_path)
    df = pd.read_csv(csv_path)
    return df.dropna(subset=["long", "lat", "cluster_id"]).reset_index(drop=True)


@lru_cache(maxsize=8)
def _load_segment_centers(segment: str) -> pd.DataFrame:
    csv_path = OUTPUTS_DIR / f"cluster_centers_{segment}.csv"
    _ensure_exists(csv_path)
    df = pd.read_csv(csv_path)
    return df.dropna(subset=["center_long", "center_lat", "cluster_id"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_borough_centroids() -> pd.DataFrame:
    _ensure_exists(LOCATION_ZONE_CSV)
    _ensure_exists(LOCATION_COORDINATES_CSV)

    zones_df = pd.read_csv(LOCATION_ZONE_CSV)
    coordinates_df = pd.read_csv(LOCATION_COORDINATES_CSV)

    merged_df = zones_df.merge(
        coordinates_df,
        left_on="location_id",
        right_on="location_id",
        how="inner",
    )
    merged_df["borough"] = merged_df["borough"].astype(str).str.strip()
    valid_borough_mask = ~merged_df["borough"].str.lower().isin(["", "nan", "unknown"])
    merged_df = merged_df.loc[valid_borough_mask].copy()

    borough_centroids = (
        merged_df.groupby("borough", as_index=False)
        .agg(
            long=("long", "mean"),
            lat=("lat", "mean"),
            location_count=("location_id", "nunique"),
        )
        .sort_values("borough")
        .reset_index(drop=True)
    )
    return borough_centroids


def _stratified_sample(points_df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(points_df) <= max_points:
        return points_df.copy()

    working_df = points_df.reset_index(drop=False).rename(columns={"index": "_row_id"})
    total_rows = len(working_df)
    sampled_chunks: list[pd.DataFrame] = []

    grouped = working_df.groupby("cluster_id", sort=True)
    for _, cluster_df in grouped:
        proportional_n = max(1, int(round(max_points * len(cluster_df) / total_rows)))
        sample_n = min(len(cluster_df), proportional_n)
        sampled_chunks.append(cluster_df.sample(n=sample_n, random_state=42))

    sampled_df = pd.concat(sampled_chunks, ignore_index=True)
    sampled_df = sampled_df.drop_duplicates(subset=["_row_id"]).reset_index(drop=True)

    if len(sampled_df) < max_points:
        remaining_df = working_df.loc[
            ~working_df["_row_id"].isin(sampled_df["_row_id"])
        ].reset_index(drop=True)
        needed = min(max_points - len(sampled_df), len(remaining_df))
        if needed > 0:
            sampled_df = pd.concat(
                [sampled_df, remaining_df.sample(n=needed, random_state=42)],
                ignore_index=True,
            )

    if len(sampled_df) > max_points:
        sampled_df = sampled_df.sample(n=max_points, random_state=42).reset_index(drop=True)
    return sampled_df.drop(columns=["_row_id"]).reset_index(drop=True)


def _compute_bounds(points_df: pd.DataFrame, centers_df: pd.DataFrame) -> dict:
    long_series = pd.concat(
        [points_df["long"], centers_df["center_long"]], ignore_index=True
    ).astype(float)
    lat_series = pd.concat(
        [points_df["lat"], centers_df["center_lat"]], ignore_index=True
    ).astype(float)

    if long_series.empty or lat_series.empty:
        return {
            "x_long_min": -74.10,
            "x_long_max": -73.70,
            "y_lat_min": 40.55,
            "y_lat_max": 40.92,
        }

    pad_long = 0.01
    pad_lat = 0.01
    return {
        "x_long_min": float(long_series.min() - pad_long),
        "x_long_max": float(long_series.max() + pad_long),
        "y_lat_min": float(lat_series.min() - pad_lat),
        "y_lat_max": float(lat_series.max() + pad_lat),
    }


def get_clustering_map_payload(segment: str, max_points: int = 25_000) -> dict:
    normalized_segment = segment.strip().lower()
    if normalized_segment not in VALID_SEGMENTS:
        raise ValueError(
            f"Invalid segment '{segment}'. Valid values: {sorted(VALID_SEGMENTS)}"
        )

    logger = get_logger()
    logger.info(
        "Building clustering payload for segment=%s with max_points=%d",
        normalized_segment,
        max_points,
    )

    metadata = _load_metadata()
    points_df = _load_segment_points(normalized_segment)
    centers_df = _load_segment_centers(normalized_segment)
    borough_df = _load_borough_centroids()

    total_points_available = int(len(points_df))
    full_cluster_counts = (
        points_df.assign(cluster_id=points_df["cluster_id"].astype(int))
        .groupby("cluster_id", as_index=False)
        .size()
        .rename(columns={"size": "total_count"})
    )
    full_cluster_count_lookup = {
        int(row.cluster_id): int(row.total_count)
        for row in full_cluster_counts.itertuples(index=False)
    }
    sampled_points_df = _stratified_sample(points_df, max_points=max_points)

    sampled_points_df = sampled_points_df.copy()
    sampled_points_df["cluster_id"] = sampled_points_df["cluster_id"].astype(int)
    sampled_points_df["color_rgba"] = sampled_points_df["cluster_id"].map(CLUSTER_COLORS)
    sampled_points_df = sampled_points_df.dropna(subset=["color_rgba"]).reset_index(drop=True)

    cluster_counts_display = (
        sampled_points_df.groupby("cluster_id", as_index=False)
        .size()
        .rename(columns={"size": "display_count"})
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )
    cluster_count_lookup = {
        int(row.cluster_id): int(row.display_count)
        for row in cluster_counts_display.itertuples(index=False)
    }
    cluster_summary = []
    for cluster_id in sorted(CLUSTER_COLORS.keys()):
        total_count = int(full_cluster_count_lookup.get(cluster_id, 0))
        display_count = int(cluster_count_lookup.get(cluster_id, 0))
        pct_of_total = (display_count / total_count * 100.0) if total_count > 0 else 0.0
        cluster_summary.append(
            {
                "cluster_id": int(cluster_id),
                "display_count": display_count,
                "total_count": total_count,
                "display_pct_of_cluster": round(pct_of_total, 2),
                "color_rgba": CLUSTER_COLORS[int(cluster_id)],
            }
        )

    bounds = _compute_bounds(sampled_points_df, centers_df)

    return {
        "segment": normalized_segment,
        "selected_k": int(metadata.get("selected_k", 5)),
        "total_points_available": total_points_available,
        "display_points_count": int(len(sampled_points_df)),
        "bounds": bounds,
        "cluster_summary": cluster_summary,
        "points": sampled_points_df.to_dict(orient="records"),
        "centers": centers_df.assign(cluster_id=centers_df["cluster_id"].astype(int)).to_dict(
            orient="records"
        ),
        "borough_labels": borough_df.to_dict(orient="records"),
    }
