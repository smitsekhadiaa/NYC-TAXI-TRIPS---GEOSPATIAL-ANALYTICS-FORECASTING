"""Frontend data access for clustering map payload."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import streamlit as st

from config import BACKEND_BASE_URL

VALID_SEGMENTS = ("morning", "afternoon", "evening", "night", "all")


@st.cache_data(ttl=300, show_spinner=False, max_entries=32)
def fetch_clustering_payload(segment: str, max_points: int = 25_000) -> dict:
    normalized_segment = segment.strip().lower()
    if normalized_segment not in VALID_SEGMENTS:
        raise ValueError(
            f"Invalid segment '{segment}'. Valid values: {', '.join(VALID_SEGMENTS)}"
        )

    query_params = urlencode({"max_points": int(max_points)})
    endpoint = f"{BACKEND_BASE_URL}/api/clustering/{normalized_segment}?{query_params}"

    try:
        with urlopen(endpoint, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Backend returned HTTP {exc.code} for clustering API: {error_text}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not reach backend clustering API. "
            f"Ensure Flask backend is running on {BACKEND_BASE_URL}."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Invalid clustering API response format.")
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload
