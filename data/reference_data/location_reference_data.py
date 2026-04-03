"""Reference location data builders for NYC taxi project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import pandas as pd

LOGGER_NAME = "nyc_taxi_location_reference"
RAW_SRC_DATA_DIR = Path(__file__).resolve().parents[1] / "raw_src_data"

# Public module-level references requested by user.
location_zone_data: pd.DataFrame | None = None
location_coordinates_data: pd.DataFrame | None = None
locations_map: Dict[str, pd.DataFrame] = {}


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


def _build_paths(base_data_dir: Path) -> Tuple[Path, Path]:
    csv_path = base_data_dir / "taxi_zone_lookup.csv"
    shp_path = base_data_dir / "taxi_zones" / "taxi_zones.shp"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing taxi zone lookup CSV: {csv_path}")
    if not shp_path.exists():
        raise FileNotFoundError(f"Missing taxi zones shapefile: {shp_path}")

    return csv_path, shp_path


def build_location_zone_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "LocationID": "location_id",
            "Borough": "borough",
            "Zone": "zone",
        }
    )

    service_zone_col = next((c for c in df.columns if c.lower() == "service_zone"), None)
    if service_zone_col:
        df = df.drop(columns=[service_zone_col])

    required_cols = ["location_id", "borough", "zone"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in location zone data: {missing_cols}")

    return df[required_cols].sort_values("location_id").reset_index(drop=True)


def build_location_coordinates_data(
    shp_path: Path,
    location_zone_df: pd.DataFrame,
) -> pd.DataFrame:
    # TODO: read shapefile with geopandas
    # TODO: compute centroids from geometry (EPSG:2263 -> project to WGS84 EPSG:4326)
    # TODO: merge with location_zone_df on location_id
    # Placeholder: return empty dataframe with expected schema
    return pd.DataFrame(columns=["location_id", "lat", "long"])


def initialize_locations_map(
    base_data_dir: str | Path = RAW_SRC_DATA_DIR,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    global location_zone_data, location_coordinates_data, locations_map

    if locations_map and not force_refresh:
        return locations_map

    logger = get_logger()
    data_dir = Path(base_data_dir)
    csv_path, shp_path = _build_paths(data_dir)

    logger.info("Building location_zone_data from %s", csv_path)
    location_zone_data = build_location_zone_data(csv_path)

    logger.info("Building location_coordinates_data from %s", shp_path)
    location_coordinates_data = build_location_coordinates_data(
        shp_path=shp_path,
        location_zone_df=location_zone_data,
    )

    locations_map.clear()
    locations_map["location_zone_data"] = location_zone_data
    locations_map["location_coordinates_data"] = location_coordinates_data
    return locations_map


def get_locations_map(
    base_data_dir: str | Path = RAW_SRC_DATA_DIR,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Public accessor for location reference dataframes."""
    return initialize_locations_map(base_data_dir=base_data_dir, force_refresh=force_refresh)


def main() -> None:
    logger = get_logger()
    local_map = initialize_locations_map()
    logger.info("Initialized locations_map with keys: %s", sorted(local_map.keys()))


if __name__ == "__main__":
    main()
