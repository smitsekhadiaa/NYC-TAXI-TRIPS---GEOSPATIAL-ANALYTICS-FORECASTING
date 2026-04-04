"""Export processed and reference dataframes to local CSV files."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_SRC_DATA_DIR = DATA_DIR / "raw_src_data"
OUTPUT_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
REFERENCE_DATA_DIR = DATA_DIR / "reference_data"

for path in (PROCESSED_DATA_DIR, REFERENCE_DATA_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import data_processing as dp  # noqa: E402
import date_reference_data as drd  # noqa: E402
import location_reference_data as lrd  # noqa: E402

LOGGER_NAME = "nyc_taxi_export_csv"
TRIP_DETAILS_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "pickup_location_id",
    "dropff_location_id",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "pickup_date",
    "pickup_time",
    "dropff_date",
    "dropff_time",
    "extra_amount",
    "total_trip_amount",
]


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


def _safe_remove(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _write_spark_df_single_csv(df: SparkDataFrame, output_csv_path: Path) -> int:
    temp_dir = output_csv_path.with_suffix(".tmp")
    _safe_remove(temp_dir)
    _safe_remove(output_csv_path)

    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(temp_dir))
    )

    part_files = list(temp_dir.glob("part-*.csv"))
    if not part_files:
        raise FileNotFoundError(f"No CSV part file created at {temp_dir}")

    shutil.move(str(part_files[0]), str(output_csv_path))
    _safe_remove(temp_dir)

    return df.count()


def _write_pandas_df_csv(df: pd.DataFrame, output_csv_path: Path) -> int:
    _safe_remove(output_csv_path)
    df.to_csv(output_csv_path, index=False)
    return int(len(df))


def _format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.2f} MB"
    return f"{size_bytes / 1024**3:.2f} GB"


def _collect_export_targets() -> Dict[str, object]:
    spark = dp.create_spark_session(app_name="NYCTaxiCSVExporter")

    month_map, _ = dp.get_processed_dataframes(
        data_dir=RAW_SRC_DATA_DIR,
        spark=spark,
        create_combined_df=False,
        force_refresh=True,
        sort_output=False,
    )

    trip_details_map = {f"trip_details_{month_key}": df for month_key, df in month_map.items()}
    ordered_trip_keys = sorted(
        trip_details_map.keys(),
        key=lambda key: dp.MONTH_TO_NUMBER.get(key.rsplit("_", 1)[-1], 99),
    )

    locations_map = lrd.get_locations_map(
        base_data_dir=RAW_SRC_DATA_DIR,
        force_refresh=False,
    )
    date_map = drd.get_date_map(
        spark=spark,
        monthly_trip_details_map=trip_details_map,
        force_refresh=True,
        sort_output=False,
    )

    location_zone_data = locations_map["location_zone_data"].copy()
    location_zone_data["borough"] = location_zone_data["borough"].astype(str)
    valid_mask = ~location_zone_data["borough"].str.strip().str.lower().isin(
        ["unknown", "nan", ""]
    )
    location_zone_data = location_zone_data[valid_mask].reset_index(drop=True)

    valid_location_ids = set(location_zone_data["location_id"].astype(int).tolist())
    location_coordinates_data = locations_map["location_coordinates_data"].copy()
    location_coordinates_data = (
        location_coordinates_data[
            location_coordinates_data["location_id"].astype(int).isin(valid_location_ids)
        ]
        .reset_index(drop=True)
    )

    export_targets: Dict[str, object] = {}
    for key in ordered_trip_keys:
        export_targets[key] = trip_details_map[key]

    export_targets["date_day_data"] = date_map["date_day_data"]
    export_targets["location_coordinates_data"] = location_coordinates_data
    export_targets["location_zone_data"] = location_zone_data

    return export_targets


def _deduplicate_trip_csv_exports(
    output_dir: Path,
    ordered_trip_table_names: list[str],
    logger: logging.Logger,
) -> dict[str, int]:
    if not ordered_trip_table_names:
        return {}

    month_keys = [table_name.rsplit("_", 1)[-1] for table_name in ordered_trip_table_names]
    month_order_case = " ".join(
        f"WHEN '{month_key}' THEN {idx + 1}" for idx, month_key in enumerate(month_keys)
    )
    projection = ", ".join(TRIP_DETAILS_COLUMNS)
    partition_cols = ", ".join(TRIP_DETAILS_COLUMNS)
    glob_path = str(output_dir / "trip_details_*.csv")

    temp_dir = output_dir / "_dedup_trip_csv_tmp"
    _safe_remove(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Applying cross-month duplicate cleanup to exported trip CSVs")
    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW deduped_trip_rows AS
            WITH source_rows AS (
                SELECT
                    {projection},
                    lower(split_part(split_part(filename, 'trip_details_', 2), '.csv', 1)) AS month_key
                FROM read_csv_auto(
                    '{glob_path}',
                    header = true,
                    union_by_name = true,
                    filename = true
                )
            ),
            ranked AS (
                SELECT
                    {projection},
                    month_key,
                    ROW_NUMBER() OVER (
                        PARTITION BY {partition_cols}
                        ORDER BY CASE month_key {month_order_case} ELSE 999 END
                    ) AS row_rank
                FROM source_rows
            )
            SELECT
                {projection},
                month_key
            FROM ranked
            WHERE row_rank = 1
            """
        )

        for month_key in month_keys:
            temp_csv_path = temp_dir / f"trip_details_{month_key}.csv"
            connection.execute(
                f"""
                COPY (
                    SELECT {projection}
                    FROM deduped_trip_rows
                    WHERE month_key = '{month_key}'
                )
                TO '{str(temp_csv_path)}' (HEADER, DELIMITER ',')
                """
            )

        month_counts_rows = connection.execute(
            """
            SELECT month_key, COUNT(*) AS row_count
            FROM deduped_trip_rows
            GROUP BY month_key
            """
        ).fetchall()
    finally:
        connection.close()

    valid_table_names = {f"trip_details_{month_key}" for month_key in month_keys}
    row_counts = {}
    for month_key, row_count in month_counts_rows:
        table_name = f"trip_details_{str(month_key)}"
        if table_name in valid_table_names:
            row_counts[table_name] = int(row_count)

    for month_key in month_keys:
        table_name = f"trip_details_{month_key}"
        final_csv_path = output_dir / f"{table_name}.csv"
        rebuilt_csv_path = temp_dir / f"{table_name}.csv"
        _safe_remove(final_csv_path)
        shutil.move(str(rebuilt_csv_path), str(final_csv_path))

    _safe_remove(temp_dir)
    return row_counts


def export_all_csvs() -> dict:
    logger = get_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_targets = _collect_export_targets()
    manifest: dict = {"output_dir": str(OUTPUT_DIR), "files": {}}

    for name, dataframe in export_targets.items():
        output_csv_path = OUTPUT_DIR / f"{name}.csv"
        logger.info("Exporting %s to %s", name, output_csv_path)

        if isinstance(dataframe, SparkDataFrame):
            row_count = _write_spark_df_single_csv(dataframe, output_csv_path)
        elif isinstance(dataframe, pd.DataFrame):
            row_count = _write_pandas_df_csv(dataframe, output_csv_path)
        else:
            raise TypeError(f"Unsupported dataframe type for export: {name}")

        file_size = output_csv_path.stat().st_size
        manifest["files"][name] = {
            "csv_path": str(output_csv_path),
            "rows": int(row_count),
            "size_bytes": int(file_size),
            "size_human": _format_bytes(file_size),
        }

    trip_table_names = [name for name in export_targets.keys() if name.startswith("trip_details_")]
    deduped_trip_counts = _deduplicate_trip_csv_exports(
        output_dir=OUTPUT_DIR,
        ordered_trip_table_names=trip_table_names,
        logger=logger,
    )
    for table_name, row_count in deduped_trip_counts.items():
        csv_path = OUTPUT_DIR / f"{table_name}.csv"
        file_size = csv_path.stat().st_size
        manifest["files"][table_name] = {
            "csv_path": str(csv_path),
            "rows": int(row_count),
            "size_bytes": int(file_size),
            "size_human": _format_bytes(file_size),
        }

    manifest_path = OUTPUT_DIR / "export_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)

    logger.info("Export complete. Manifest written to %s", manifest_path)
    return manifest


def main() -> None:
    export_all_csvs()


if __name__ == "__main__":
    main()
