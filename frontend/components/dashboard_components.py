"""Reusable dashboard UI components."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services import data_repository as repository
from utils.formatting import format_float, format_int, format_iso_date

BLUE_SHADES = ["#0B3B75", "#1457A6", "#2A74C9", "#5A96D6"]
BLUE_PAIR = {"weekday": "#134A8E", "weekend": "#4F8ECF"}
MONTH_COLORS = [
    "#123B6D",
    "#1A4F8A",
    "#2363A6",
    "#2D77C0",
    "#3A89CB",
    "#4A98D3",
    "#5BA5DA",
    "#6CB2E1",
    "#7CBEE7",
    "#8CC9EC",
    "#9CD3F0",
]


def _add_value_legend_to_right(
    fig: go.Figure,
    legend_items: list[tuple[str, str, float]],
    digits: int = 2,
    title: str = "Actual Values",
) -> None:
    for label, color, value in legend_items:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": color},
                name=f"{label}: {value:,.{digits}f}",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.update_layout(
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "title": {"text": title},
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "#d8e1ec",
            "borderwidth": 1,
        }
    )


def render_hero(stats: dict) -> None:
    start_date = format_iso_date(stats.get("start_date", "2025-01-01"))
    end_date = format_iso_date(stats.get("end_date", "2025-11-30"))
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">NYC Taxi Demand Intelligence Platform</div>
            <div class="hero-subtitle">Integrated Geospatial Analytics for Urban Mobility ({start_date} to {end_date})</div>
            <div class="hero-mission">
                "Transform NYC taxi trip data into operational intelligence that drives smarter
                demand planning, movement optimization, and location-level decision making."
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(stats: dict) -> None:
    cards = [
        ("Total Trips", format_int(stats.get("total_no_of_trips", 0))),
        ("Start Date", format_iso_date(stats.get("start_date", "2025-01-01"))),
        ("End Date", format_iso_date(stats.get("end_date", "2025-11-30"))),
        (
            "Avg Trips / Day",
            format_float(float(stats.get("avg_no_of_trips_per_day", 0.0)), digits=2),
        ),
    ]

    columns = st.columns(4, gap="small")
    for column, (label, value) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_borough_line_chart() -> None:
    borough_df = repository.get_borough_stats_frame().copy()
    if borough_df.empty:
        st.info("No borough statistics available.")
        return

    borough_df = borough_df.sort_values("avg_trips_per_location", ascending=False).reset_index(
        drop=True
    )
    borough_df["point_label"] = borough_df["avg_trips_per_location"].apply(
        lambda value: f"{value/1000:.1f}k" if value >= 1000 else f"{value:.0f}"
    )

    marker_colors = [MONTH_COLORS[idx % len(MONTH_COLORS)] for idx in range(len(borough_df))]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=borough_df["borough"],
            y=borough_df["avg_trips_per_location"],
            mode="lines+markers+text",
            line={"width": 3, "color": BLUE_SHADES[1]},
            marker={"size": 9, "color": marker_colors},
            text=borough_df["point_label"],
            textposition="top center",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Avg trips per location: %{y:,.2f}<br>"
                "Locations: %{customdata[0]}<extra></extra>"
            ),
            customdata=borough_df[["location_count"]].values,
            showlegend=False,
        )
    )

    legend_items = [
        (
            str(row["borough"]),
            marker_colors[idx],
            float(row["avg_trips_per_location"]),
        )
        for idx, (_row_idx, row) in enumerate(borough_df.iterrows())
    ]
    _add_value_legend_to_right(fig, legend_items=legend_items, digits=2)

    fig.update_layout(
        title="Average Trips per Location by Borough",
        xaxis_title="Borough",
        yaxis_title="Avg Trips per Location",
        margin={"l": 10, "r": 240, "t": 48, "b": 10},
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_monthly_trip_bar_chart() -> None:
    monthly_df = repository.get_monthly_trip_counts_frame().copy()
    if monthly_df.empty:
        st.info("No monthly trip data available.")
        return

    bar_colors = [MONTH_COLORS[idx % len(MONTH_COLORS)] for idx in range(len(monthly_df))]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly_df["month"],
            y=monthly_df["trip_count"],
            marker={"color": bar_colors},
            hovertemplate="<b>%{x}</b><br>Trips: %{y:,}<extra></extra>",
            showlegend=False,
        )
    )

    legend_items = [
        (
            str(row["month"]),
            bar_colors[idx],
            float(row["trip_count"]),
        )
        for idx, (_row_idx, row) in enumerate(monthly_df.iterrows())
    ]
    _add_value_legend_to_right(fig, legend_items=legend_items, digits=0)

    fig.update_layout(
        title="Number of Trips per Month (Jan to Nov)",
        xaxis_title="Month",
        yaxis_title="Number of Trips",
        margin={"l": 10, "r": 240, "t": 48, "b": 10},
        height=340,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_time_of_day_pie() -> None:
    time_df = repository.get_time_of_day_stats_frame().copy()
    if time_df.empty:
        st.info("No time-of-day statistics available.")
        return

    time_df["time_of_day"] = time_df["time_of_day"].str.title()
    fig = px.pie(
        time_df,
        values="avg_trips",
        names="time_of_day",
        hole=0.45,
        color="time_of_day",
        color_discrete_sequence=BLUE_SHADES,
    )
    fig.update_traces(
        textinfo="percent",
        customdata=time_df[["avg_trips"]].values,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Avg trips per day: %{customdata[0]:,.2f}<extra></extra>"
        ),
    )
    fig.update_layout(
        title="Avg Trips by Time of Day",
        margin={"l": 8, "r": 8, "t": 46, "b": 8},
        height=300,
        legend={"orientation": "h", "y": -0.1},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_weekday_weekend_pie() -> None:
    summary_df = repository.get_weekday_weekend_avg_frame().copy()
    if summary_df.empty:
        st.info("No weekday/weekend summary data available.")
        return

    fig = px.pie(
        summary_df,
        values="avg_trips",
        names="segment",
        hole=0.45,
        color="segment",
        color_discrete_map={
            "Weekday": BLUE_PAIR["weekday"],
            "Weekend": BLUE_PAIR["weekend"],
        },
    )
    fig.update_traces(
        textinfo="percent",
        customdata=summary_df[["avg_trips"]].values,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Avg trips per day: %{customdata[0]:,.2f}<extra></extra>"
        ),
    )
    fig.update_layout(
        title="Weekday vs Weekend Avg Trips",
        margin={"l": 8, "r": 8, "t": 46, "b": 8},
        height=300,
        legend={"orientation": "h", "y": -0.1},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_weekday_weekend_time_of_day_table() -> None:
    frame = repository.get_weekday_weekend_time_of_day_stats_frame().copy()
    if frame.empty:
        st.info("No weekday/weekend time-of-day table data available.")
        return

    ordered_columns = ["Morning", "Afternoon", "Evening", "Night"]
    ordered_rows = ["Weekday", "Weekend"]

    table_df = (
        frame.pivot(index="segment", columns="time_of_day", values="avg_trips")
        .reindex(index=ordered_rows, columns=ordered_columns)
        .fillna(0.0)
    )
    table_df.columns.name = "Time of Day"
    table_df.index.name = ""
    table_df = table_df.round(2)

    st.markdown("#### Avg Trips by Time of Day (Weekday vs Weekend)")
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=False,
    )


def render_top_frequent_pairs_table() -> None:
    frequent_pairs_df = repository.get_top_frequent_pairs_frame()
    frequent_cols = [
        "source_id",
        "destination_id",
        "source_zone",
        "destination_zone",
        "distance_km",
        "frequency_of_trips",
        "avg_fare_amount",
    ]

    st.markdown("#### Top Frequent Source-Destination Pairs")
    if frequent_pairs_df.empty:
        st.info("No source-destination frequency data available.")
        return

    st.dataframe(
        frequent_pairs_df[frequent_cols],
        use_container_width=True,
        hide_index=True,
        height=265,
    )


def render_top_tipped_pairs_table() -> None:
    tipped_pairs_df = repository.get_top_tipped_pairs_frame()
    tipped_cols = [
        "source_id",
        "destination_id",
        "source_zone",
        "destination_zone",
        "avg_tip_amount",
        "frequency_of_trips",
    ]

    st.markdown("#### Top Most Tipped Source-Destination Pairs")
    if tipped_pairs_df.empty:
        st.info("No tipping pair data available.")
        return

    st.dataframe(
        tipped_pairs_df[tipped_cols],
        use_container_width=True,
        hide_index=True,
        height=300,
    )
