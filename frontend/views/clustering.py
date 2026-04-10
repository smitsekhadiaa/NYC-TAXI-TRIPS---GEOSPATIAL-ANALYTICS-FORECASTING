"""Clustering tab: interactive map of precomputed MiniBatchKMeans results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from services.clustering_insights import (
    build_full_day_summary,
    build_period_cards,
)
from services.clustering_repository import fetch_clustering_payload

MAP_HEIGHT = 620
MAP_PANEL_HEIGHT = 700
LEFT_PANEL_EST_HEIGHT = 470
MAX_POINT_OPTIONS = [value for value in range(5_000, 70_000, 5_000)] + ["All"]
SEGMENT_OPTIONS = {
    "Full Day": "all",
    "Morning (08:00-12:00)": "morning",
    "Afternoon (12:00-16:00)": "afternoon",
    "Evening (16:00-20:00)": "evening",
    "Night (20:00-24:00)": "night",
}
CLUSTER_COLORS = {
    0: "#B71C1C",  # dark red
    1: "#1B5E20",  # dark green
    2: "#F5C400",  # dark yellow
    3: "#0D47A1",  # dark blue
    4: "#4A148C",  # dark purple
}
PERIOD_CARD_BG = {
    "morning": "#FFF7D6",
    "afternoon": "#FFEBD9",
    "evening": "#F1E6FF",
    "night": "#E7EEFF",
}


def _to_view_state() -> pdk.ViewState:
    return pdk.ViewState(
        longitude=-73.94,
        latitude=40.73,
        zoom=11.0,
        pitch=30,
        bearing=0,
    )


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _lighten_rgb(rgb: tuple[int, int, int], strength: float = 0.56) -> list[int]:
    red, green, blue = rgb
    return [
        int(red + (255 - red) * strength),
        int(green + (255 - green) * strength),
        int(blue + (255 - blue) * strength),
        145,
    ]


def _cross_2d(origin: np.ndarray, point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(
        (point_a[0] - origin[0]) * (point_b[1] - origin[1])
        - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
    )


def _convex_hull(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return points

    sorted_idx = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[sorted_idx]

    lower: list[np.ndarray] = []
    for point in pts:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in pts[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _build_cluster_region_polygons(points_df: pd.DataFrame, centers_df: pd.DataFrame) -> pd.DataFrame:
    if points_df.empty:
        return pd.DataFrame(columns=["cluster_id", "polygon", "fill_rgba", "line_rgba", "long", "lat"])

    center_map: dict[int, tuple[float, float]] = {}
    if not centers_df.empty:
        for row in centers_df.itertuples(index=False):
            center_map[int(row.cluster_id)] = (float(row.center_long), float(row.center_lat))

    polygons: list[dict] = []
    grouped = points_df.groupby("cluster_id", sort=True)
    for cluster_id, cluster_points in grouped:
        points = cluster_points[["long", "lat"]].drop_duplicates().to_numpy(dtype=np.float64)
        if points.shape[0] < 3:
            continue

        hull = _convex_hull(points)
        if hull.shape[0] < 3:
            continue

        dark_rgb = _hex_to_rgb(CLUSTER_COLORS[int(cluster_id)])
        fill_rgba = _lighten_rgb(dark_rgb, strength=0.56)
        line_rgba = [dark_rgb[0], dark_rgb[1], dark_rgb[2], 190]
        center_long, center_lat = center_map.get(
            int(cluster_id),
            (float(points[:, 0].mean()), float(points[:, 1].mean())),
        )

        polygons.append(
            {
                "cluster_id": int(cluster_id),
                "polygon": hull.tolist(),
                "fill_rgba": fill_rgba,
                "line_rgba": line_rgba,
                "long": center_long,
                "lat": center_lat,
            }
        )

    return pd.DataFrame(polygons)


def _build_center_cross_paths(centers_df: pd.DataFrame) -> pd.DataFrame:
    if centers_df.empty:
        return pd.DataFrame(columns=["path"])

    half_diagonal = 0.004
    rows: list[dict] = []
    for record in centers_df.to_dict(orient="records"):
        center_long = float(record["center_long"])
        center_lat = float(record["center_lat"])
        rows.append(
            {
                "path": [
                    [center_long - half_diagonal, center_lat - half_diagonal],
                    [center_long + half_diagonal, center_lat + half_diagonal],
                ]
            }
        )
        rows.append(
            {
                "path": [
                    [center_long - half_diagonal, center_lat + half_diagonal],
                    [center_long + half_diagonal, center_lat - half_diagonal],
                ]
            }
        )
    return pd.DataFrame(rows)


def _build_cluster_map(payload: dict) -> pdk.Deck:
    points_df = pd.DataFrame(payload.get("points", []))
    centers_df = pd.DataFrame(payload.get("centers", []))
    borough_df = pd.DataFrame(payload.get("borough_labels", []))

    if points_df.empty:
        return pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=40.73, longitude=-73.94, zoom=10.2, pitch=0),
            map_provider="carto",
            map_style="light",
        )

    points_df = points_df.copy()
    points_df["cluster_id"] = points_df["cluster_id"].astype(int)

    center_cross_paths_df = _build_center_cross_paths(centers_df=centers_df)
    cluster_polygons_df = _build_cluster_region_polygons(points_df=points_df, centers_df=centers_df)
    view_state = _to_view_state()

    layers: list[pdk.Layer] = []
    if not cluster_polygons_df.empty:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=cluster_polygons_df,
                get_polygon="polygon",
                get_fill_color="fill_rgba",
                get_line_color="line_rgba",
                get_line_width=1.8,
                stroked=True,
                filled=True,
                pickable=True,
                opacity=0.45,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=points_df,
            get_position="[long, lat]",
            get_fill_color="color_rgba",
            get_radius=58,
            radius_min_pixels=1,
            radius_max_pixels=5,
            pickable=True,
            auto_highlight=True,
            stroked=False,
            opacity=0.86,
        )
    )

    if not centers_df.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=center_cross_paths_df,
                get_path="path",
                get_color=[0, 0, 0, 235],
                get_width=2,
                width_units="pixels",
                width_min_pixels=2,
                width_max_pixels=2,
                pickable=False,
                opacity=0.95,
            )
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=centers_df,
                get_position="[center_long, center_lat]",
                get_radius=80,
                radius_min_pixels=2,
                radius_max_pixels=4,
                get_fill_color=[0, 0, 0, 245],
                pickable=False,
                opacity=0.95,
            )
        )

    if not borough_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=borough_df,
                get_position="[long, lat]",
                get_radius=120,
                radius_min_pixels=2,
                radius_max_pixels=5,
                get_fill_color=[21, 40, 60, 210],
                pickable=False,
                opacity=0.9,
            )
        )
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=borough_df,
                get_position="[long, lat]",
                get_text="borough",
                get_color=[22, 38, 56, 235],
                get_size=15,
                get_alignment_baseline="'bottom'",
                get_pixel_offset=[0, -12],
                pickable=False,
            )
        )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light",
        tooltip={
            "html": (
                "<b>Cluster:</b> {cluster_id}<br/>"
                "<b>Longitude:</b> {long}<br/>"
                "<b>Latitude:</b> {lat}"
            ),
            "style": {"backgroundColor": "#0d223a", "color": "white"},
        },
    )


def _render_cluster_legend(cluster_summary: list[dict], show_title: bool = True) -> None:
    if not cluster_summary:
        return

    if show_title:
        st.markdown("**No of points per cluster**")
    for cluster_item in cluster_summary:
        cluster_id = int(cluster_item["cluster_id"])
        hex_color = CLUSTER_COLORS.get(cluster_id, "#4A627A")
        display_count = int(cluster_item["display_count"])
        st.markdown(
            f"""
            <div class="map-legend-item">
                <span class="map-legend-swatch" style="background:{hex_color};"></span>
                <span class="map-legend-text">Cluster {cluster_id}: {display_count:,}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_full_day_card(summary: dict) -> None:
    if not summary:
        st.info("Full Day insight unavailable.")
        return

    st.markdown("#### Full Day")
    row_1_col_1, row_1_col_2 = st.columns(2, gap="small")
    row_2_col_1, row_2_col_2 = st.columns(2, gap="small")

    row_1_col_1.metric("Top Borough", str(summary["top_borough"]))
    row_1_col_2.metric(
        "Top Borough %",
        f"{float(summary['top_borough_pct']):.2f}%",
    )
    row_2_col_1.metric("2nd Borough", str(summary["second_borough"]))
    row_2_col_2.metric(
        "2nd Borough %",
        f"{float(summary['second_borough_pct']):.2f}%",
    )

    st.markdown(
        (
            f"**Top 2 Boroughs:** {summary['top_2_boroughs']}  \n"
            f"**Top 2 Borough %:** {float(summary['top_2_borough_pct']):.2f}%"
        )
    )
    st.markdown(
        (
            f"**Most Concentrated:** {summary.get('most_concentrated_period', 'N/A')}  \n"
            f"**Most Distributed:** {summary.get('most_distributed_period', 'N/A')}"
        )
    )


def _render_period_card(card: dict, segment: str) -> None:
    if not card:
        card = {
            "label": segment.title(),
            "line_1": "No insight available.",
            "line_2": "No insight available.",
        }

    bg_color = PERIOD_CARD_BG.get(segment, "#ECF1F9")
    st.markdown(
        f"""
        <div style="
            background:{bg_color};
            border:1px solid #d4ddeb;
            border-radius:12px;
            padding:0.7rem 0.75rem;
            min-height:165px;
        ">
            <div style="font-weight:800;font-size:0.95rem;color:#203248;margin-bottom:0.35rem;">
                {card['label']}
            </div>
            <div style="font-size:0.84rem;color:#2f465e;line-height:1.45;">
                {card['line_1']}<br/>
                {card['line_2']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_insights_section(full_day_summary: dict, period_cards: dict[str, dict]) -> None:
    st.markdown("### Clustering Insights")
    left_col, right_col = st.columns([1.05, 1.45], gap="large")

    with left_col:
        with st.container(border=True):
            _render_full_day_card(full_day_summary)

    with right_col:
        st.markdown("#### Time-of-Day Insights")
        top_left, top_right = st.columns(2, gap="small")
        bottom_left, bottom_right = st.columns(2, gap="small")

        with top_left:
            _render_period_card(period_cards.get("morning", {}), "morning")
        with top_right:
            _render_period_card(period_cards.get("afternoon", {}), "afternoon")
        with bottom_left:
            _render_period_card(period_cards.get("evening", {}), "evening")
        with bottom_right:
            _render_period_card(period_cards.get("night", {}), "night")


def render() -> None:
    st.markdown("### Spatial Clustering Map")
    st.caption(
        "KMeans clustering output for geospatial points. "
        "Cluster centers are shown as black crosses. Borough labels include all 6 borough groups in the dataset."
    )

    filter_col, map_col = st.columns([1, 2.8], gap="large")

    with filter_col:
        with st.container(height=MAP_PANEL_HEIGHT, border=False):
            top_spacer_height = max((MAP_PANEL_HEIGHT - LEFT_PANEL_EST_HEIGHT) // 2, 0)
            if top_spacer_height > 0:
                st.markdown(
                    f"<div style='height:{top_spacer_height}px;'></div>",
                    unsafe_allow_html=True,
                )

            with st.container(border=True):
                st.markdown("#### Filters")
                selected_label = st.selectbox(
                    "Time Segment",
                    options=list(SEGMENT_OPTIONS.keys()),
                    index=0,
                    key="clustering_segment",
                )
                max_points_selection = st.select_slider(
                    "Max Points on Map",
                    options=MAX_POINT_OPTIONS,
                    value=50_000,
                    key="clustering_max_points",
                    help="Select display volume. Use 'All' to plot every available point for the selected segment.",
                )
                st.markdown("**No of points per cluster**")
                legend_placeholder = st.empty()

            if top_spacer_height > 0:
                st.markdown(
                    f"<div style='height:{top_spacer_height}px;'></div>",
                    unsafe_allow_html=True,
                )

    segment = SEGMENT_OPTIONS[selected_label]
    max_points_query = 0 if max_points_selection == "All" else int(max_points_selection)
    with st.spinner("Loading clustering map data from backend..."):
        try:
            payload = fetch_clustering_payload(segment=segment, max_points=max_points_query)
        except Exception as exc:
            st.error(str(exc))
            return

    full_day_payload: dict = {}
    time_of_day_payloads: dict[str, dict] = {}
    try:
        full_day_payload = fetch_clustering_payload(segment="all", max_points=0)
    except Exception:
        full_day_payload = {}

    for segment_name in ("morning", "afternoon", "evening", "night"):
        try:
            time_of_day_payloads[segment_name] = fetch_clustering_payload(
                segment=segment_name,
                max_points=0,
            )
        except Exception:
            continue

    with legend_placeholder.container():
        _render_cluster_legend(payload.get("cluster_summary", []), show_title=False)

    with map_col:
        with st.container(height=MAP_PANEL_HEIGHT, border=True):
            deck = _build_cluster_map(payload=payload)
            st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)

    insights_full_day = build_full_day_summary(
        full_day_payload=full_day_payload or payload,
        period_payloads=time_of_day_payloads,
    )
    period_cards = build_period_cards(time_of_day_payloads)
    _render_insights_section(
        full_day_summary=insights_full_day,
        period_cards=period_cards,
    )
