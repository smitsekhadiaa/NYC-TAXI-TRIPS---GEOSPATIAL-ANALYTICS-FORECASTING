"""Frontend API client for anomaly detection payloads."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import streamlit as st

from config import BACKEND_BASE_URL


@st.cache_data(ttl=300, show_spinner=False, max_entries=8)
def fetch_anomaly_summary() -> dict:
    endpoint = f"{BACKEND_BASE_URL}/api/anomaly-detection/summary"
    try:
        with urlopen(endpoint, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Backend returned HTTP {exc.code} for anomaly summary API: {error_text}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not reach backend anomaly API. "
            "Ensure Flask backend is running on http://127.0.0.1:5001."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Invalid anomaly summary response format.")
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload
