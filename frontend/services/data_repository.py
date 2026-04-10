"""Centralized data access for frontend pages.

Dynamic data reads use CSV files under data/processed_csv.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from config import PROCESSED_CSV_DIR, STATISTICS_JSON_PATH

TRIP_CSV_GLOB = str(PROCESSED_CSV_DIR / "trip_details_*.csv")
LOCATION_ZONE_CSV = PROCESSED_CSV_DIR / "location_zone_data.csv"
LOCATION_COORDINATES_CSV = PROCESSED_CSV_DIR / "location_coordinates_data.csv"
DATE_DAY_CSV = PROCESSED_CSV_DIR / "date_day_data.csv"


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


@st.cache_data(show_spinner=False)
def load_statistics() -> dict:
    _ensure_file_exists(STATISTICS_JSON_PATH)
    with open(STATISTICS_JSON_PATH, "r", encoding="utf-8") as infile:
        return json.load(infile)


@st.cache_data(show_spinner=False)
def get_date_bounds_from_statistics() -> tuple[date, date]:
    stats = load_statistics()
    return date.fromisoformat(stats["start_date"]), date.fromisoformat(stats["end_date"])


@st.cache_data(show_spinner=False)
def load_location_zone_data() -> pd.DataFrame:
    _ensure_file_exists(LOCATION_ZONE_CSV)
    df = pd.read_csv(LOCATION_ZONE_CSV)
    df["borough"] = df["borough"].astype(str)
    valid_mask = ~df["borough"].str.strip().str.lower().isin(["unknown", "nan", ""])
    return df.loc[valid_mask].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_location_coordinates_data() -> pd.DataFrame:
    _ensure_file_exists(LOCATION_COORDINATES_CSV)
    return pd.read_csv(LOCATION_COORDINATES_CSV)


@st.cache_data(show_spinner=False)
def get_borough_stats_frame() -> pd.DataFrame:
    stats = load_statistics()
    return pd.DataFrame(stats.get("avg_trips_by_location_for_each_borough", []))


@st.cache_data(show_spinner=False)
def get_time_of_day_stats_frame() -> pd.DataFrame:
    stats = load_statistics()
    order = ["morning", "afternoon", "evening", "night"]
    payload = stats.get("avg_trips_by_time_of_day", {})
    records = [{"time_of_day": bucket, "avg_trips": payload.get(bucket, 0.0)} for bucket in order]
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def get_monthly_trip_counts_frame() -> pd.DataFrame:
    start_date, end_date = get_date_bounds_from_statistics()
    trip_glob_literal = TRIP_CSV_GLOB.replace("'", "''")

    with duckdb.connect(database=":memory:") as connection:
        connection.execute("PRAGMA threads=4")
        month_counts = connection.execute(
            f"""
            SELECT
                EXTRACT(MONTH FROM pickup_dt)::INTEGER AS month_no,
                COUNT(*) AS trip_count
            FROM (
                SELECT TRY_CAST(pickup_date AS DATE) AS pickup_dt
                FROM read_csv_auto(
                    '{trip_glob_literal}',
                    header = true,
                    union_by_name = true
                )
            )
            WHERE pickup_dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            GROUP BY month_no
            ORDER BY month_no
            """,
            [start_date.isoformat(), end_date.isoformat()],
        ).df()

    if month_counts.empty:
        return pd.DataFrame(columns=["month_no", "month", "trip_count"])

    full_month_range = pd.DataFrame(
        {"month_no": list(range(start_date.month, end_date.month + 1))}
    )
    full_month_range["month"] = pd.to_datetime(
        {"year": 2025, "month": full_month_range["month_no"], "day": 1}
    ).dt.strftime("%b")

    merged = full_month_range.merge(month_counts, on="month_no", how="left")
    merged["trip_count"] = merged["trip_count"].fillna(0).astype(int)
    return merged[["month_no", "month", "trip_count"]]


@st.cache_data(show_spinner=False)
def get_weekday_weekend_avg_frame() -> pd.DataFrame:
    stats = load_statistics()
    start_date, end_date = get_date_bounds_from_statistics()
    _ensure_file_exists(DATE_DAY_CSV)

    date_df = pd.read_csv(DATE_DAY_CSV)
    date_df["date"] = pd.to_datetime(date_df["date"], errors="coerce")
    date_df["is_weekend"] = pd.to_numeric(date_df["is_weekend"], errors="coerce").fillna(0).astype(int)
    window_mask = (date_df["date"] >= pd.Timestamp(start_date)) & (date_df["date"] <= pd.Timestamp(end_date))
    date_df = date_df.loc[window_mask]

    weekday_days = int((date_df["is_weekend"] == 0).sum())
    weekend_days = int((date_df["is_weekend"] == 1).sum())

    totals = stats.get("weekday_weekend_trip_totals", {})
    weekday_total = int(totals.get("total_no_of_weekday_trips", 0))
    weekend_total = int(totals.get("total_no_of_weekend_trips", 0))

    weekday_avg = (weekday_total / weekday_days) if weekday_days else 0.0
    weekend_avg = (weekend_total / weekend_days) if weekend_days else 0.0

    return pd.DataFrame(
        [
            {"segment": "Weekday", "avg_trips": weekday_avg},
            {"segment": "Weekend", "avg_trips": weekend_avg},
        ]
    )


@st.cache_data(show_spinner=False)
def get_weekday_weekend_time_of_day_stats_frame() -> pd.DataFrame:
    start_date, end_date = get_date_bounds_from_statistics()
    trip_glob_literal = TRIP_CSV_GLOB.replace("'", "''")
    date_day_literal = str(DATE_DAY_CSV).replace("'", "''")
    _ensure_file_exists(DATE_DAY_CSV)

    with duckdb.connect(database=":memory:") as connection:
        connection.execute("PRAGMA threads=4")
        frame = connection.execute(
            f"""
            WITH date_ref AS (
                SELECT
                    CAST(date AS DATE) AS trip_date,
                    CAST(is_weekend AS INTEGER) AS is_weekend
                FROM read_csv_auto('{date_day_literal}', header = true)
                WHERE CAST(date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ),
            bucketed_trips AS (
                SELECT
                    date_ref.is_weekend,
                    CASE
                        WHEN pickup_hour >= 8 AND pickup_hour < 12 THEN 'morning'
                        WHEN pickup_hour >= 12 AND pickup_hour < 16 THEN 'afternoon'
                        WHEN pickup_hour >= 16 AND pickup_hour < 20 THEN 'evening'
                        WHEN pickup_hour >= 20 AND pickup_hour <= 23 THEN 'night'
                    END AS time_of_day
                FROM (
                    SELECT
                        TRY_CAST(pickup_date AS DATE) AS pickup_dt,
                        TRY_CAST(SUBSTR(CAST(pickup_time AS VARCHAR), 1, 2) AS INTEGER) AS pickup_hour
                    FROM read_csv_auto(
                        '{trip_glob_literal}',
                        header = true,
                        union_by_name = true
                    )
                ) trips
                JOIN date_ref ON trips.pickup_dt = date_ref.trip_date
                WHERE pickup_hour BETWEEN 0 AND 23
            ),
            total_trips_by_segment_bucket AS (
                SELECT
                    is_weekend,
                    time_of_day,
                    COUNT(*) AS total_trips
                FROM bucketed_trips
                WHERE time_of_day IS NOT NULL
                GROUP BY is_weekend, time_of_day
            ),
            day_counts AS (
                SELECT
                    is_weekend,
                    COUNT(*) AS day_count
                FROM date_ref
                GROUP BY is_weekend
            ),
            all_time_buckets AS (
                SELECT * FROM (VALUES ('morning'), ('afternoon'), ('evening'), ('night')) AS t(time_of_day)
            ),
            all_segments AS (
                SELECT * FROM (VALUES (0), (1)) AS s(is_weekend)
            )
            SELECT
                all_segments.is_weekend,
                all_time_buckets.time_of_day,
                COALESCE(total_trips_by_segment_bucket.total_trips, 0)::DOUBLE
                    / NULLIF(day_counts.day_count, 0) AS avg_trips
            FROM all_segments
            CROSS JOIN all_time_buckets
            JOIN day_counts ON day_counts.is_weekend = all_segments.is_weekend
            LEFT JOIN total_trips_by_segment_bucket
                ON total_trips_by_segment_bucket.is_weekend = all_segments.is_weekend
               AND total_trips_by_segment_bucket.time_of_day = all_time_buckets.time_of_day
            ORDER BY
                all_segments.is_weekend,
                CASE all_time_buckets.time_of_day
                    WHEN 'morning' THEN 1
                    WHEN 'afternoon' THEN 2
                    WHEN 'evening' THEN 3
                    WHEN 'night' THEN 4
                END
            """,
            [start_date.isoformat(), end_date.isoformat()],
        ).df()

    if frame.empty:
        return pd.DataFrame(columns=["segment", "time_of_day", "avg_trips"])

    frame["segment"] = frame["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    frame["time_of_day"] = frame["time_of_day"].astype(str).str.title()
    frame["avg_trips"] = pd.to_numeric(frame["avg_trips"], errors="coerce").fillna(0.0)
    return frame[["segment", "time_of_day", "avg_trips"]]


@st.cache_data(show_spinner=False)
def get_top_frequent_pairs_frame() -> pd.DataFrame:
    stats = load_statistics()
    return pd.DataFrame(stats.get("top_10_frequent_source_destination_pairs", []))


@st.cache_data(show_spinner=False)
def get_top_tipped_pairs_frame() -> pd.DataFrame:
    stats = load_statistics()
    return pd.DataFrame(stats.get("top_10_most_tipped_source_destination_pairs", []))


@st.cache_data(ttl=600, show_spinner=False, max_entries=128)
def query_trip_arcs(
    start_date: date,
    end_date: date,
    start_hour: int,
    end_hour: int,
    max_pairs: int = 400,
) -> pd.DataFrame:
    """Build source-destination arc data from processed CSV files."""
    if start_date > end_date:
        raise ValueError("start_date cannot be greater than end_date")
    if start_hour < 0 or start_hour > 23:
        raise ValueError("start_hour must be in range [0, 23]")
    if end_hour < 1 or end_hour > 24:
        raise ValueError("end_hour must be in range [1, 24]")
    if start_hour >= end_hour:
        raise ValueError("start_hour must be lower than end_hour")

    trip_glob_literal = TRIP_CSV_GLOB.replace("'", "''")

    with duckdb.connect(database=":memory:") as connection:
        connection.execute("PRAGMA threads=4")

        pairs_df = connection.execute(
            f"""
            WITH filtered_trips AS (
                SELECT
                    pickup_location_id AS source_id,
                    dropff_location_id AS destination_id
                FROM read_csv_auto(
                    '{trip_glob_literal}',
                    header = true,
                    union_by_name = true
                )
                WHERE pickup_location_id IS NOT NULL
                  AND dropff_location_id IS NOT NULL
                  AND pickup_location_id <> dropff_location_id
                  AND pickup_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                  AND TRY_CAST(SUBSTR(CAST(pickup_time AS VARCHAR), 1, 2) AS INTEGER) >= ?
                  AND TRY_CAST(SUBSTR(CAST(pickup_time AS VARCHAR), 1, 2) AS INTEGER) < ?
            )
            SELECT
                source_id,
                destination_id,
                COUNT(*) AS trip_count
            FROM filtered_trips
            GROUP BY source_id, destination_id
            ORDER BY trip_count DESC
            LIMIT ?
            """,
            [
                start_date.isoformat(),
                end_date.isoformat(),
                start_hour,
                end_hour,
                max_pairs,
            ],
        ).df()

    if pairs_df.empty:
        return pairs_df

    coordinates_df = load_location_coordinates_data()
    zones_df = load_location_zone_data()[["location_id", "zone"]]

    source_coords = coordinates_df.rename(
        columns={
            "location_id": "source_id",
            "lat": "source_lat",
            "long": "source_long",
        }
    )
    destination_coords = coordinates_df.rename(
        columns={
            "location_id": "destination_id",
            "lat": "destination_lat",
            "long": "destination_long",
        }
    )

    source_zones = zones_df.rename(columns={"location_id": "source_id", "zone": "source_zone"})
    destination_zones = zones_df.rename(
        columns={"location_id": "destination_id", "zone": "destination_zone"}
    )

    arc_df = (
        pairs_df.merge(source_coords, on="source_id", how="left")
        .merge(destination_coords, on="destination_id", how="left")
        .merge(source_zones, on="source_id", how="left")
        .merge(destination_zones, on="destination_id", how="left")
        .dropna(
            subset=[
                "source_lat",
                "source_long",
                "destination_lat",
                "destination_long",
            ]
        )
        .reset_index(drop=True)
    )

    if arc_df.empty:
        return arc_df

    min_count = float(arc_df["trip_count"].min())
    max_count = float(arc_df["trip_count"].max())

    if max_count == min_count:
        arc_df["line_width"] = 3.0
    else:
        arc_df["line_width"] = 1.5 + (
            (arc_df["trip_count"] - min_count) / (max_count - min_count)
        ) * 10.0

    return arc_df
