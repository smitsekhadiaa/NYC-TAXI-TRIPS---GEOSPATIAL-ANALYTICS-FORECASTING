"""Dashboard page implementation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from components import dashboard_components as components
from services import data_repository as repository
from utils.formatting import format_hour_label

NYC_DEFAULT_VIEW = pdk.ViewState(latitude=40.7306, longitude=-73.9352, zoom=10, pitch=35)
MAP_CHART_HEIGHT = 500
MAP_SECTION_PANEL_HEIGHT = 650
FILTER_CARD_ESTIMATED_HEIGHT = 530
ARC_COLORS = [
    [129, 199, 132, 210],  # light green
    [255, 235, 59, 215],   # yellow
    [255, 167, 38, 220],   # orange
    [229, 57, 53, 230],    # red
]


def _prepare_arc_styling(arc_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    if arc_df.empty:
        return arc_df, []

    styled_df = arc_df.copy()
    min_count = float(styled_df["trip_count"].min())
    max_count = float(styled_df["trip_count"].max())

    if max_count == min_count:
        styled_df["line_width"] = 4.0
        styled_df["source_color"] = [ARC_COLORS[2]] * len(styled_df)
        styled_df["target_color"] = [[183, 28, 28, 225]] * len(styled_df)
        styled_df["trip_range_label"] = [f"{int(min_count):,} trips"] * len(styled_df)
        legend = [
            {
                "color": ARC_COLORS[2],
                "label": f"All flows: {int(min_count):,} trips",
            }
        ]
        return styled_df, legend

    styled_df["line_width"] = 1.2 + ((styled_df["trip_count"] - min_count) / (max_count - min_count)) * 10.0

    edges = np.linspace(min_count, max_count, 5)
    boundary_points = edges[1:-1]
    styled_df["bucket_index"] = np.searchsorted(boundary_points, styled_df["trip_count"], side="right")

    legend: list[dict] = []
    range_labels: list[str] = []
    for bucket_idx in range(4):
        lower = int(np.floor(edges[bucket_idx]))
        upper = int(np.ceil(edges[bucket_idx + 1]))
        if bucket_idx > 0:
            lower += 1
        range_labels.append(f"{lower:,} - {upper:,} trips")
        legend.append({"color": ARC_COLORS[bucket_idx], "label": range_labels[bucket_idx]})

    styled_df["trip_range_label"] = styled_df["bucket_index"].map(
        {idx: label for idx, label in enumerate(range_labels)}
    )
    styled_df["source_color"] = styled_df["bucket_index"].map(
        {idx: color for idx, color in enumerate(ARC_COLORS)}
    )
    styled_df["target_color"] = styled_df["bucket_index"].map(
        {
            idx: [max(color[0] - 25, 0), max(color[1] - 25, 0), max(color[2] - 25, 0), color[3]]
            for idx, color in enumerate(ARC_COLORS)
        }
    )

    styled_df = styled_df.drop(columns=["bucket_index"])
    return styled_df, legend


def _render_arc_legend(legend_items: list[dict]) -> None:
    if not legend_items:
        return

    st.markdown("**Flow Color Scheme (by trip count)**")
    legend_columns = st.columns(len(legend_items), gap="small")

    for column, item in zip(legend_columns, legend_items):
        red, green, blue, _alpha = item["color"]
        with column:
            st.markdown(
                f"""
                <div class="map-legend-item">
                    <span class="map-legend-swatch" style="background: rgba({red}, {green}, {blue}, 0.95);"></span>
                    <span class="map-legend-text">{item["label"]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption("Arc width is proportional to number of trips.")


def _build_trip_arc_map(arc_df: pd.DataFrame) -> pdk.Deck:
    if arc_df.empty:
        return pdk.Deck(
            initial_view_state=NYC_DEFAULT_VIEW,
            map_style="light",
            map_provider="carto",
        )

    center_latitude = (
        arc_df["source_lat"].mean() + arc_df["destination_lat"].mean()
    ) / 2.0
    center_longitude = (
        arc_df["source_long"].mean() + arc_df["destination_long"].mean()
    ) / 2.0

    view_state = pdk.ViewState(
        latitude=float(center_latitude),
        longitude=float(center_longitude),
        zoom=10.2,
        pitch=38,
    )

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_df,
        get_source_position="[source_long, source_lat]",
        get_target_position="[destination_long, destination_lat]",
        get_source_color="source_color",
        get_target_color="target_color",
        get_width="line_width",
        width_scale=1,
        width_min_pixels=1,
        width_max_pixels=16,
        pickable=True,
        auto_highlight=True,
    )

    return pdk.Deck(
        layers=[arc_layer],
        initial_view_state=view_state,
        tooltip={
            "html": (
                "<b>{source_zone}</b> → <b>{destination_zone}</b><br/>"
                "Trips: <b>{trip_count}</b><br/>"
                "Range: <b>{trip_range_label}</b>"
            ),
            "style": {"backgroundColor": "#0e2745", "color": "white"},
        },
        map_provider="carto",
        map_style="light",
    )


def _render_filter_panel(
    min_date: date,
    max_date: date,
) -> tuple[date, date, int, int, int]:
    with st.container(border=True):
        st.markdown("#### Filters")
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="dashboard_start_date",
        )
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="dashboard_end_date",
        )

        start_hour = st.selectbox(
            "Start Hour",
            options=list(range(0, 24)),
            index=0,
            format_func=format_hour_label,
            key="dashboard_start_hour",
        )
        end_hour = st.selectbox(
            "End Hour",
            options=list(range(1, 25)),
            index=23,
            format_func=format_hour_label,
            key="dashboard_end_hour",
        )
        max_pairs = st.slider(
            "Max Arc Pairs",
            min_value=50,
            max_value=700,
            value=350,
            step=50,
            key="dashboard_max_pairs",
            help="Higher values show more route pairs, but may be visually denser.",
        )

    return start_date, end_date, int(start_hour), int(end_hour), int(max_pairs)


def render() -> None:
    stats = repository.load_statistics()
    min_date, max_date = repository.get_date_bounds_from_statistics()

    components.render_hero(stats)
    components.render_kpi_cards(stats)
    st.markdown("")

    st.markdown("### Source-Destination Trip Flow Map")
    filter_col, map_col = st.columns([1.05, 2.8], gap="large")

    with filter_col:
        with st.container(height=MAP_SECTION_PANEL_HEIGHT, border=False):
            top_spacer_height = max((MAP_SECTION_PANEL_HEIGHT - FILTER_CARD_ESTIMATED_HEIGHT) // 2, 0)
            if top_spacer_height > 0:
                st.markdown(
                    f"<div style='height:{top_spacer_height}px;'></div>",
                    unsafe_allow_html=True,
                )

            start_date, end_date, start_hour, end_hour, max_pairs = _render_filter_panel(
                min_date=min_date,
                max_date=max_date,
            )

            if top_spacer_height > 0:
                st.markdown(
                    f"<div style='height:{top_spacer_height}px;'></div>",
                    unsafe_allow_html=True,
                )

    with map_col:
        with st.container(border=True, height=MAP_SECTION_PANEL_HEIGHT):
            if start_date > end_date:
                st.error("Start date must be earlier than or equal to end date.")
                return
            if start_hour >= end_hour:
                st.error("Start hour must be lower than end hour.")
                return

            with st.spinner("Loading filtered trip flows from processed CSV data..."):
                arc_df = repository.query_trip_arcs(
                    start_date=start_date,
                    end_date=end_date,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    max_pairs=max_pairs,
                )

            styled_arc_df, legend_items = _prepare_arc_styling(arc_df)
            if styled_arc_df.empty:
                st.warning("No trip flows found for selected filters.")
            else:
                st.caption(f"Displaying {len(styled_arc_df)} aggregated route arcs.")

            deck = _build_trip_arc_map(arc_df=styled_arc_df)
            st.pydeck_chart(deck, use_container_width=True, height=MAP_CHART_HEIGHT)
            _render_arc_legend(legend_items)

    _, frequent_mid_col, _ = st.columns([0.08, 0.84, 0.08], gap="small")
    with frequent_mid_col:
        components.render_top_frequent_pairs_table()

    _, snapshot_mid_col, _ = st.columns([0.08, 0.84, 0.08], gap="small")
    with snapshot_mid_col:
        st.markdown("### Trip Pattern Snapshot")
        components.render_borough_line_chart()
        components.render_monthly_trip_bar_chart()

        pie_col_left, pie_col_right = st.columns(2, gap="large")
        with pie_col_left:
            components.render_time_of_day_pie()
        with pie_col_right:
            components.render_weekday_weekend_pie()

        components.render_weekday_weekend_time_of_day_table()

    _, tipped_mid_col, _ = st.columns([0.08, 0.84, 0.08], gap="small")
    with tipped_mid_col:
        components.render_top_tipped_pairs_table()

    st.markdown(
        """
        <div class="app-footer">
            NYC Taxi Demand Intelligence Platform • Built with Streamlit, Flask, PySpark, and MySQL
        </div>
        """,
        unsafe_allow_html=True,
    )
