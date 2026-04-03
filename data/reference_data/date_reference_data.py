"""Reference date/day data builders for NYC taxi project."""

from __future__ import annotations

import calendar
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

LOGGER_NAME = "nyc_taxi_date_reference"

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "processed_data"
if str(PROCESSED_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESSED_DATA_DIR))

from data_processing import initialize_dataframes  # noqa: E402

MONTH_TO_NUMBER = {calendar.month_abbr[i].lower(): i for i in range(1, 13)}

date_day_data: Optional[DataFrame] = None
date_map: Dict[str, DataFrame] = {}
trip_details_map: Dict[str, DataFrame] = {}


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


def _build_trip_details_map(month_map: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    """Convert {'jan': df} -> {'trip_details_jan': df}."""
    return {f"trip_details_{month_key}": df for month_key, df in month_map.items()}


def _extract_unique_dates_from_trip_details_map(
    monthly_trip_details_map: Dict[str, DataFrame],
) -> DataFrame:
    ordered_keys = sorted(
        monthly_trip_details_map.keys(),
        key=lambda key: MONTH_TO_NUMBER.get(key.rsplit("_", 1)[-1], 99),
    )
    if not ordered_keys:
        raise ValueError("Month map is empty. Cannot build date_day_data.")

    base_df = (
        monthly_trip_details_map[ordered_keys[0]]
        .select(F.col("pickup_date").alias("date"))
        .distinct()
    )
    for month_key in ordered_keys[1:]:
        monthly_dates = (
            monthly_trip_details_map[month_key]
            .select(F.col("pickup_date").alias("date"))
            .distinct()
        )
        base_df = base_df.unionByName(monthly_dates)

    return base_df.distinct()


def build_date_day_data(
    spark: Optional[SparkSession] = None,
    monthly_trip_details_map: Optional[Dict[str, DataFrame]] = None,
    sort_output: bool = False,
) -> DataFrame:
    global trip_details_map

    if monthly_trip_details_map is None:
        month_map, _ = initialize_dataframes(
            spark=spark,
            create_combined_df=False,
            sort_output=sort_output,
        )
        monthly_trip_details_map = _build_trip_details_map(month_map)

    trip_details_map.clear()
    trip_details_map.update(monthly_trip_details_map)
    unique_dates_df = _extract_unique_dates_from_trip_details_map(trip_details_map)

    return unique_dates_df.select("date").orderBy(F.col("date").asc())


def initialize_date_map(
    spark: Optional[SparkSession] = None,
    monthly_trip_details_map: Optional[Dict[str, DataFrame]] = None,
    force_refresh: bool = False,
    sort_output: bool = False,
) -> Dict[str, DataFrame]:
    global date_day_data, date_map

    if date_map and date_day_data is not None and not force_refresh:
        return date_map

    logger = get_logger()
    logger.info("Building date_day_data from monthly trip data map")

    date_day_data = build_date_day_data(
        spark=spark,
        monthly_trip_details_map=monthly_trip_details_map,
        sort_output=sort_output,
    )
    date_map.clear()
    date_map["date_day_data"] = date_day_data
    return date_map


def get_date_map(
    spark: Optional[SparkSession] = None,
    monthly_trip_details_map: Optional[Dict[str, DataFrame]] = None,
    force_refresh: bool = False,
    sort_output: bool = False,
) -> Dict[str, DataFrame]:
    """Public accessor for date reference dataframes."""
    return initialize_date_map(
        spark=spark,
        monthly_trip_details_map=monthly_trip_details_map,
        force_refresh=force_refresh,
        sort_output=sort_output,
    )


def main() -> None:
    logger = get_logger()
    local_map = initialize_date_map()
    logger.info("Initialized date_map with keys: %s", sorted(local_map.keys()))


if __name__ == "__main__":
    main()
