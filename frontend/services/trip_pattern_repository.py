"""Load mined trip pattern rule artifacts for frontend insights page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    TRIP_PATTERN_ALL_RULES_CSV,
    TRIP_PATTERN_METADATA_JSON,
    TRIP_PATTERN_TOP_RULES_CSV,
)


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required artifact: {path}. "
            "Run backend command: flask --app run.py train-trip-pattern-rules"
        )


@st.cache_data(show_spinner=False)
def load_pattern_metadata() -> dict:
    _ensure_file_exists(TRIP_PATTERN_METADATA_JSON)
    with open(TRIP_PATTERN_METADATA_JSON, "r", encoding="utf-8") as infile:
        return json.load(infile)


@st.cache_data(show_spinner=False)
def load_all_rules() -> pd.DataFrame:
    _ensure_file_exists(TRIP_PATTERN_ALL_RULES_CSV)
    df = pd.read_csv(TRIP_PATTERN_ALL_RULES_CSV)
    for col in ("support", "confidence", "lift"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_top_rules() -> pd.DataFrame:
    _ensure_file_exists(TRIP_PATTERN_TOP_RULES_CSV)
    df = pd.read_csv(TRIP_PATTERN_TOP_RULES_CSV)
    for col in ("support", "confidence", "lift"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_airport_spotlight_rules(limit: int = 6) -> pd.DataFrame:
    rules_df = load_top_rules()
    if rules_df.empty:
        return rules_df

    airport_mask = (
        rules_df["rule"].str.contains("airport", case=False, na=False)
        | rules_df["insight"].str.contains("airport", case=False, na=False)
    )
    spotlight = rules_df.loc[airport_mask].copy()
    spotlight = spotlight.sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
    return spotlight.head(limit).reset_index(drop=True)
