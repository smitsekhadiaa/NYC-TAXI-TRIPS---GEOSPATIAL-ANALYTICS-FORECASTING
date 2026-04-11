"""Trip fare and ETA prediction page."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import streamlit as st

from services import data_repository as data_repo
from services import trip_forecast_repository as forecast_repo

MILES_PER_KM = 0.621371
BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]


def _inject_prediction_styles() -> None:
    st.markdown(
        """
        <style>
        .forecast-hero {
            background: linear-gradient(145deg, rgba(15, 76, 129, 0.96), rgba(7, 40, 72, 0.96));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            box-shadow: 0 14px 35px rgba(10, 34, 57, 0.2);
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.9rem;
        }

        .forecast-hero-title {
            color: #ffffff;
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: 0.2px;
        }

        .forecast-hero-subtitle {
            color: rgba(234, 243, 255, 0.95);
            font-size: 0.95rem;
            margin-top: 0.35rem;
            margin-bottom: 0;
            font-weight: 600;
        }

        .forecast-result-shell {
            border: 1px solid var(--border);
            background: #ffffff;
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 8px 22px rgba(20, 56, 91, 0.06);
        }

        .forecast-route-badge {
            display: inline-block;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            background: rgba(15, 76, 129, 0.1);
            color: #0b3558;
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _borough_sort_key(name: str) -> tuple[int, str]:
    if name in BOROUGH_ORDER:
        return (BOROUGH_ORDER.index(name), name)
    return (len(BOROUGH_ORDER), name)


@st.cache_data(show_spinner=False)
def _build_location_catalog() -> pd.DataFrame:
    zone_df = data_repo.load_location_zone_data()
    zone_df = zone_df[["location_id", "borough", "zone"]].dropna(subset=["location_id"]).copy()
    zone_df["location_id"] = zone_df["location_id"].astype(int)
    zone_df["borough"] = zone_df["borough"].astype(str).str.strip()
    zone_df["zone"] = zone_df["zone"].astype(str).str.strip()

    coords_df = data_repo.load_location_coordinates_data()
    coords_df = coords_df[["location_id", "lat", "long"]].dropna(subset=["location_id"]).copy()
    coords_df["location_id"] = coords_df["location_id"].astype(int)
    coords_df["lat"] = pd.to_numeric(coords_df["lat"], errors="coerce")
    coords_df["long"] = pd.to_numeric(coords_df["long"], errors="coerce")

    merged = zone_df.merge(coords_df, on="location_id", how="left")
    merged = merged.sort_values(["borough", "zone", "location_id"]).reset_index(drop=True)
    merged["option_label"] = merged.apply(
        lambda row: f"{int(row['location_id'])} - {row['zone']}",
        axis=1,
    )
    return merged


def _format_hour(hour: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    human_hour = 12 if hour % 12 == 0 else hour % 12
    if hour == 24:
        return "24:00 (next day)"
    return f"{hour:02d}:00 ({human_hour}:00 {suffix})"


def _haversine_km(
    source_lat: float,
    source_long: float,
    destination_lat: float,
    destination_long: float,
) -> float:
    radius_km = 6371.0
    phi1 = math.radians(source_lat)
    phi2 = math.radians(destination_lat)
    delta_phi = math.radians(destination_lat - source_lat)
    delta_lambda = math.radians(destination_long - source_long)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_km * c


def _select_location(
    prefix: str,
    title: str,
    catalog_df: pd.DataFrame,
    borough_options: list[str],
) -> tuple[int, dict]:
    borough = st.selectbox(
        f"{title} Borough",
        options=borough_options,
        key=f"{prefix}_borough",
    )
    borough_df = catalog_df.loc[catalog_df["borough"] == borough].copy()
    if borough_df.empty:
        raise ValueError(f"No locations found for borough: {borough}")

    location_ids = borough_df["location_id"].astype(int).tolist()
    label_lookup = {
        int(row.location_id): str(row.option_label)
        for row in borough_df.itertuples(index=False)
    }
    location_key = f"{prefix}_location"
    previous_location = st.session_state.get(location_key)
    if previous_location not in location_ids:
        st.session_state[location_key] = int(location_ids[0])

    location_id = st.selectbox(
        f"{title} Location",
        options=location_ids,
        format_func=lambda value: label_lookup.get(int(value), str(value)),
        key=location_key,
    )

    row = borough_df.loc[borough_df["location_id"] == int(location_id)].iloc[0].to_dict()
    return int(location_id), row


def render() -> None:
    _inject_prediction_styles()

    st.markdown(
        """
        <div class="forecast-hero">
            <p class="forecast-hero-title">Trip Fare & ETA Prediction</p>
            <p class="forecast-hero-subtitle">
                Select pickup and drop off along with date and time)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        metadata = forecast_repo.fetch_forecast_metadata()
    except Exception as exc:
        st.error(str(exc))
        return

    serving_window = metadata.get("serving_window", {})
    min_date = date.fromisoformat(serving_window.get("start_date", "2025-10-01"))
    max_date = date.fromisoformat(serving_window.get("end_date", "2025-12-31"))

    catalog_df = _build_location_catalog()
    if catalog_df.empty:
        st.error("No location IDs available for prediction input.")
        return

    borough_options = sorted(catalog_df["borough"].dropna().unique().tolist(), key=_borough_sort_key)
    if not borough_options:
        st.error("No borough data available for prediction input.")
        return

    if "trip_forecast_latest_result" not in st.session_state:
        st.session_state["trip_forecast_latest_result"] = None

    pickup_col, drop_col = st.columns([1, 1], gap="large")
    with pickup_col:
        source_id, source_row = _select_location(
            prefix="forecast_source",
            title="Pickup",
            catalog_df=catalog_df,
            borough_options=borough_options,
        )
    with drop_col:
        destination_id, destination_row = _select_location(
            prefix="forecast_destination",
            title="Drop",
            catalog_df=catalog_df,
            borough_options=borough_options,
        )

    date_col, time_col = st.columns([1, 1], gap="large")
    with date_col:
        pickup_date = st.date_input(
            "Pickup Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="forecast_pickup_date",
        )
    with time_col:
        pickup_hour = st.select_slider(
            "Pickup Time",
            options=list(range(0, 25)),
            value=12,
            format_func=_format_hour,
            key="forecast_pickup_hour",
        )

    submitted = st.button(
        "Estimate Base Fare and ETA",
        type="primary",
        use_container_width=True,
    )

    if submitted:
        if int(source_id) == int(destination_id):
            st.error("Pickup and drop locations must be different.")
            return

        with st.spinner("Predicting fare and ETA..."):
            try:
                prediction = forecast_repo.predict_trip(
                    source_location_id=int(source_id),
                    destination_location_id=int(destination_id),
                    pickup_date=pickup_date,
                    pickup_hour=int(pickup_hour),
                )
            except Exception as exc:
                st.error(str(exc))
                return

        st.session_state["trip_forecast_latest_result"] = {
            "prediction": prediction,
            "source": source_row,
            "destination": destination_row,
        }

    result_payload = st.session_state.get("trip_forecast_latest_result")
    if not result_payload:
        return

    prediction = result_payload["prediction"]
    source_row = result_payload["source"]
    destination_row = result_payload["destination"]

    source_label = f"{source_row['zone']} ({source_row['borough']})"
    destination_label = f"{destination_row['zone']} ({destination_row['borough']})"
    route_label = f"{source_label} → {destination_label}"

    trip_distance_miles: float | None = None
    if pd.notna(source_row.get("lat")) and pd.notna(source_row.get("long")) and pd.notna(
        destination_row.get("lat")
    ) and pd.notna(destination_row.get("long")):
        trip_distance_miles = _haversine_km(
            float(source_row["lat"]),
            float(source_row["long"]),
            float(destination_row["lat"]),
            float(destination_row["long"]),
        ) * MILES_PER_KM

    st.markdown(
        """
        <div class="forecast-result-shell">
            <div class="forecast-route-badge">Estimated Trip Summary</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col_1, metric_col_2, metric_col_3 = st.columns([1, 1, 1], gap="large")
    metric_col_1.metric(
        "Estimated Base Fare",
        f"${float(prediction.get('predicted_fare_amount', 0.0)):.2f}",
    )
    metric_col_2.metric(
        "Estimated ETA",
        f"{float(prediction.get('predicted_eta_minutes', 0.0)):.2f} min",
    )
    metric_col_3.metric(
        "Trip Distance",
        "N/A" if trip_distance_miles is None else f"{trip_distance_miles:.2f} miles",
    )

    st.markdown(
        f"""
        **Route:** {route_label}  
        **Location IDs:** {prediction.get("source_location_id")} → {prediction.get("destination_location_id")}  
        **Pickup Slot:** {prediction.get("pickup_date")} at {_format_hour(int(prediction.get("pickup_hour", 0)))}
        """
    )
