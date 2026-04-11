"""Frontend API client for trip fare/ETA forecasting."""

from __future__ import annotations

import json
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from config import BACKEND_BASE_URL


@st.cache_data(ttl=300, show_spinner=False, max_entries=8)
def fetch_forecast_metadata() -> dict:
    endpoint = f"{BACKEND_BASE_URL}/api/trip-forecast/metadata"
    try:
        with urlopen(endpoint, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Backend returned HTTP {exc.code} for forecast metadata API: {error_text}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not reach backend forecast API. "
            "Ensure Flask backend is running on http://127.0.0.1:5001."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Invalid forecast metadata response format.")
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def predict_trip(
    source_location_id: int,
    destination_location_id: int,
    pickup_date: date,
    pickup_hour: int,
) -> dict:
    endpoint = f"{BACKEND_BASE_URL}/api/trip-forecast/predict"
    request_body = json.dumps(
        {
            "source_location_id": int(source_location_id),
            "destination_location_id": int(destination_location_id),
            "pickup_date": pickup_date.isoformat(),
            "pickup_hour": int(pickup_hour),
        }
    ).encode("utf-8")

    request_obj = Request(
        endpoint,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request_obj, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Backend returned HTTP {exc.code} for trip prediction API: {error_text}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not reach backend prediction API. "
            "Ensure Flask backend is running on http://127.0.0.1:5001."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Invalid trip prediction response format.")
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload
