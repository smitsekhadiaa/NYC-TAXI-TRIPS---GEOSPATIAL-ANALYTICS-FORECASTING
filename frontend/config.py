"""Configuration constants for Streamlit frontend."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_CSV_DIR = DATA_DIR / "processed_csv"
ANALYSIS_DIR = DATA_DIR / "data_analysis"
STATISTICS_JSON_PATH = ANALYSIS_DIR / "statistics.json"
TRIP_PATTERN_ARTIFACT_DIR = DATA_DIR / "trip_pattern_artifacts"
TRIP_PATTERN_ALL_RULES_CSV = TRIP_PATTERN_ARTIFACT_DIR / "trip_pattern_rules_all.csv"
TRIP_PATTERN_TOP_RULES_CSV = TRIP_PATTERN_ARTIFACT_DIR / "trip_pattern_rules_top10.csv"
TRIP_PATTERN_METADATA_JSON = TRIP_PATTERN_ARTIFACT_DIR / "trip_pattern_rules_metadata.json"

APP_TITLE = "NYC Taxi Demand Intelligence"
APP_SUBTITLE = "Geospatial Analytics and Forecasting Workspace"
BACKEND_BASE_URL = "http://127.0.0.1:5001"
