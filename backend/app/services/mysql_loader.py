"""MySQL loader for NYC taxi CSV exports."""

from __future__ import annotations

import calendar
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

from app.config import Settings

LOGGER_NAME = "nyc_taxi_mysql_loader"
VALID_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TRIP_CSV_RE = re.compile(r"^trip_details_([a-z]{3})\.csv$")
MONTH_TO_NUMBER = {calendar.month_abbr[i].lower(): i for i in range(1, 13)}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_CSV_DIR = PROJECT_ROOT / "data" / "processed_csv"
REFERENCE_TABLE_CSVS = {
    "location_zone_data": "location_zone_data.csv",
    "location_coordinates_data": "location_coordinates_data.csv",
    "date_day_data": "date_day_data.csv",
}


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


def _build_sqlalchemy_url(settings: Settings, with_database: bool) -> URL:
    return URL.create(
        drivername="mysql+pymysql",
        username=settings.mysql_user,
        password=settings.mysql_password,
        host=settings.mysql_host,
        port=settings.mysql_port,
        database=settings.mysql_database if with_database else None,
        query={"charset": settings.mysql_charset},
    )


def _create_engine(settings: Settings, with_database: bool) -> Engine:
    return create_engine(
        _build_sqlalchemy_url(settings, with_database=with_database),
        pool_pre_ping=True,
        future=True,
    )


def create_database_if_not_exists(settings: Settings) -> None:
    logger = get_logger()
    server_engine = _create_engine(settings, with_database=False)
    create_db_sql = (
        f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
        f"CHARACTER SET {settings.mysql_charset} COLLATE {settings.mysql_charset}_unicode_ci"
    )

    logger.info("Ensuring database '%s' exists", settings.mysql_database)
    with server_engine.begin() as connection:
        connection.execute(text(create_db_sql))


def _validate_table_name(table_name: str) -> None:
    if not VALID_TABLE_NAME_RE.match(table_name):
        raise ValueError(f"Invalid table name: {table_name}")


def _drop_table_if_exists(engine: Engine, table_name: str) -> None:
    _validate_table_name(table_name)
    with engine.begin() as connection:
        connection.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))


def _trip_table_name_and_order(csv_path: Path) -> Tuple[str, int]:
    match = TRIP_CSV_RE.match(csv_path.name)
    if not match:
        raise ValueError(f"Unexpected trip CSV file name: {csv_path.name}")

    month_key = match.group(1).lower()
    month_order = MONTH_TO_NUMBER.get(month_key, 99)
    return f"trip_details_{month_key}", month_order


def _collect_csv_tables_to_load() -> List[Tuple[str, Path]]:
    trip_csv_files = sorted(PROCESSED_CSV_DIR.glob("trip_details_*.csv"))
    if not trip_csv_files:
        raise FileNotFoundError(f"No trip CSV files found in {PROCESSED_CSV_DIR}")

    trip_specs: List[Tuple[int, str, Path]] = []
    for csv_path in trip_csv_files:
        table_name, month_order = _trip_table_name_and_order(csv_path)
        trip_specs.append((month_order, table_name, csv_path))
    trip_specs.sort(key=lambda item: item[0])

    table_files: List[Tuple[str, Path]] = [(table_name, csv_path) for _, table_name, csv_path in trip_specs]

    for table_name, file_name in REFERENCE_TABLE_CSVS.items():
        csv_path = PROCESSED_CSV_DIR / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required CSV: {csv_path}")
        table_files.append((table_name, csv_path))

    return table_files


def _write_csv_file_to_table(
    engine: Engine,
    table_name: str,
    csv_path: Path,
    logger: logging.Logger,
    chunk_size: int,
) -> int:
    _drop_table_if_exists(engine, table_name)
    logger.info("Loading CSV '%s' into table '%s'", csv_path.name, table_name)

    total_rows = 0
    wrote_any_rows = False

    for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        chunk_df.to_sql(
            name=table_name,
            con=engine,
            if_exists="append" if wrote_any_rows else "replace",
            index=False,
            chunksize=chunk_size,
            method="multi",
        )
        total_rows += int(len(chunk_df))
        wrote_any_rows = True

    if not wrote_any_rows:
        # Keep schema creation behavior for empty CSVs with headers.
        empty_df = pd.read_csv(csv_path, nrows=0)
        empty_df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            method="multi",
        )

    return total_rows


def load_all_dataframes_to_mysql(settings: Settings) -> Dict[str, int]:
    """Create database and load processed CSV tables into MySQL."""
    logger = get_logger()

    create_database_if_not_exists(settings)
    database_engine = _create_engine(settings, with_database=True)

    table_csv_files = _collect_csv_tables_to_load()
    summary: Dict[str, int] = {}
    chunk_size = max(1, int(settings.spark_insert_batch_size))

    for table_name, csv_path in table_csv_files:
        _validate_table_name(table_name)
        row_count = _write_csv_file_to_table(
            engine=database_engine,
            table_name=table_name,
            csv_path=csv_path,
            logger=logger,
            chunk_size=chunk_size,
        )

        summary[table_name] = row_count
        logger.info("Loaded table '%s' with %d rows", table_name, row_count)

    return summary
