"""Insight builders for clustering tab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEGMENT_LABELS = {
    "morning": "Morning",
    "afternoon": "Afternoon",
    "evening": "Evening",
    "night": "Night",
    "all": "Full Day",
}


@dataclass(frozen=True)
class ClusterSnapshot:
    cluster_id: int
    total_count: int
    share_pct: float
    center_long: float
    center_lat: float
    nearest_borough: str


def _nearest_borough_name(center_long: float, center_lat: float, borough_df: pd.DataFrame) -> str:
    if borough_df.empty:
        return "Unknown"

    distances = (
        (borough_df["long"].astype(float) - center_long) ** 2
        + (borough_df["lat"].astype(float) - center_lat) ** 2
    )
    nearest_idx = distances.idxmin()
    borough_name = str(borough_df.loc[nearest_idx, "borough"]).strip()
    return borough_name if borough_name else "Unknown"


def _cluster_snapshots(payload: dict) -> list[ClusterSnapshot]:
    centers_df = pd.DataFrame(payload.get("centers", []))
    summary_df = pd.DataFrame(payload.get("cluster_summary", []))
    borough_df = pd.DataFrame(payload.get("borough_labels", []))
    if centers_df.empty or summary_df.empty:
        return []

    merged_df = centers_df.merge(
        summary_df[["cluster_id", "total_count"]],
        on="cluster_id",
        how="left",
    ).fillna({"total_count": 0})
    merged_df["cluster_id"] = merged_df["cluster_id"].astype(int)
    merged_df["total_count"] = merged_df["total_count"].astype(int)

    total_points = int(merged_df["total_count"].sum())
    snapshots: list[ClusterSnapshot] = []
    for row in merged_df.itertuples(index=False):
        share_pct = (float(row.total_count) / total_points * 100.0) if total_points > 0 else 0.0
        snapshots.append(
            ClusterSnapshot(
                cluster_id=int(row.cluster_id),
                total_count=int(row.total_count),
                share_pct=round(share_pct, 2),
                center_long=float(row.center_long),
                center_lat=float(row.center_lat),
                nearest_borough=_nearest_borough_name(
                    center_long=float(row.center_long),
                    center_lat=float(row.center_lat),
                    borough_df=borough_df,
                ),
            )
        )

    return sorted(snapshots, key=lambda item: item.total_count, reverse=True)


def _borough_share_from_points(payload: dict) -> list[tuple[str, float]]:
    points_df = pd.DataFrame(payload.get("points", []))
    borough_df = pd.DataFrame(payload.get("borough_labels", []))
    if points_df.empty or borough_df.empty:
        return []

    point_coords = points_df[["long", "lat"]].astype(float).to_numpy()
    borough_coords = borough_df[["long", "lat"]].astype(float).to_numpy()
    borough_names = borough_df["borough"].astype(str).to_numpy()

    # Assign each point to nearest borough centroid (6 borough groups in this dataset).
    diff = point_coords[:, None, :] - borough_coords[None, :, :]
    distances_sq = np.sum(diff * diff, axis=2)
    nearest_idx = np.argmin(distances_sq, axis=1)
    nearest_boroughs = borough_names[nearest_idx]

    borough_counts = pd.Series(nearest_boroughs).value_counts()
    total_count = int(len(nearest_boroughs))
    borough_share = [
        (str(borough), round(float(count) / total_count * 100.0, 2))
        for borough, count in borough_counts.items()
    ]
    return borough_share


def build_full_day_summary(full_day_payload: dict, period_payloads: dict[str, dict]) -> dict:
    snapshots = _cluster_snapshots(full_day_payload)
    if not snapshots:
        return {}

    top_1 = snapshots[0]
    top_2 = snapshots[1] if len(snapshots) > 1 else None

    ranked_boroughs = _borough_share_from_points(full_day_payload)
    top_borough = ranked_boroughs[0][0] if ranked_boroughs else "N/A"
    top_borough_pct = round(float(ranked_boroughs[0][1]), 2) if ranked_boroughs else 0.0
    second_borough = ranked_boroughs[1][0] if len(ranked_boroughs) > 1 else top_borough
    second_borough_pct = round(float(ranked_boroughs[1][1]), 2) if len(ranked_boroughs) > 1 else 0.0
    top_2_borough_names = "N/A"
    top_2_borough_pct = top_borough_pct
    if len(ranked_boroughs) >= 2:
        top_2_borough_names = f"{ranked_boroughs[0][0]} + {ranked_boroughs[1][0]}"
        top_2_borough_pct = round(float(ranked_boroughs[0][1] + ranked_boroughs[1][1]), 2)
    elif ranked_boroughs:
        top_2_borough_names = ranked_boroughs[0][0]

    segment_rows: list[dict] = []
    for segment in ("morning", "afternoon", "evening", "night"):
        payload = period_payloads.get(segment)
        if not payload:
            continue
        segment_snapshots = _cluster_snapshots(payload)
        if not segment_snapshots:
            continue
        primary = segment_snapshots[0]
        segment_rows.append(
            {
                "segment": segment,
                "share_pct": primary.share_pct,
                "top2_share_pct": round(
                    primary.share_pct
                    + (segment_snapshots[1].share_pct if len(segment_snapshots) > 1 else primary.share_pct),
                    2,
                ),
                "borough": primary.nearest_borough,
            }
        )

    most_concentrated_period = ""
    most_distributed_period = ""
    if segment_rows:
        segment_df = pd.DataFrame(segment_rows)
        most_concentrated_period = str(
            SEGMENT_LABELS.get(
                segment_df.sort_values("share_pct", ascending=False).iloc[0]["segment"], ""
            )
        )
        most_distributed_period = str(
            SEGMENT_LABELS.get(
                segment_df.sort_values("top2_share_pct", ascending=True).iloc[0]["segment"], ""
            )
        )

    return {
        "title": "Full Day",
        "dominant_cluster_id": top_1.cluster_id,
        "dominant_cluster_share_pct": round(float(top_1.share_pct), 2),
        "second_cluster_id": top_2.cluster_id if top_2 else top_1.cluster_id,
        "second_cluster_share_pct": round(float(top_2.share_pct), 2) if top_2 else round(float(top_1.share_pct), 2),
        "top_borough": top_borough,
        "top_borough_pct": top_borough_pct,
        "second_borough": second_borough,
        "second_borough_pct": second_borough_pct,
        "top_2_boroughs": top_2_borough_names,
        "top_2_borough_pct": top_2_borough_pct,
        "most_concentrated_period": most_concentrated_period,
        "most_distributed_period": most_distributed_period,
    }


def build_period_cards(period_payloads: dict[str, dict]) -> dict[str, dict]:
    segment_metrics: dict[str, dict] = {}
    ordered_segments = ("morning", "afternoon", "evening", "night")

    for segment in ordered_segments:
        payload = period_payloads.get(segment) or {}
        snapshots = _cluster_snapshots(payload)
        if not snapshots:
            continue

        dominant = snapshots[0]
        second_share = snapshots[1].share_pct if len(snapshots) > 1 else dominant.share_pct
        top2_share = round(float(dominant.share_pct + second_share), 2)

        borough_share = _borough_share_from_points(payload)
        top_borough = borough_share[0][0] if borough_share else "N/A"
        top_borough_pct = borough_share[0][1] if borough_share else 0.0
        second_borough = borough_share[1][0] if len(borough_share) > 1 else top_borough
        second_borough_pct = borough_share[1][1] if len(borough_share) > 1 else 0.0

        segment_metrics[segment] = {
            "dominant_cluster_id": int(dominant.cluster_id),
            "dominant_share": round(float(dominant.share_pct), 2),
            "top2_share": top2_share,
            "top_borough": top_borough,
            "top_borough_pct": round(float(top_borough_pct), 2),
            "second_borough": second_borough,
            "second_borough_pct": round(float(second_borough_pct), 2),
        }

    if not segment_metrics:
        return {
            segment: {"label": SEGMENT_LABELS[segment], "line_1": "No insight available.", "line_2": ""}
            for segment in ordered_segments
        }

    cards: dict[str, dict] = {}
    for segment in ordered_segments:
        if segment not in segment_metrics:
            cards[segment] = {
                "label": SEGMENT_LABELS[segment],
                "line_1": f"{SEGMENT_LABELS[segment]}: no cluster insights available.",
                "line_2": "",
            }
            continue

        metric = segment_metrics[segment]
        if segment == "morning":
            line_1 = (
                f"Morning: Cluster {metric['dominant_cluster_id']} has max points "
                f"({metric['dominant_share']:.2f}%)."
            )
            line_2 = (
                f"Most morning trips are in {metric['top_borough']} ({metric['top_borough_pct']:.2f}%), "
                f"followed by {metric['second_borough']} ({metric['second_borough_pct']:.2f}%)."
            )
        elif segment == "afternoon":
            evening_top2 = segment_metrics.get("evening", {}).get("top2_share")
            if evening_top2 is not None:
                concentration_gap = round(float(metric["top2_share"] - float(evening_top2)), 2)
                gap_text = f"{concentration_gap:.2f} percentage points higher than evening"
            else:
                gap_text = "higher than evening"

            line_1 = (
                f"Afternoon: Cluster {metric['dominant_cluster_id']} has max points "
                f"({metric['dominant_share']:.2f}%). Top 2 clusters hold {metric['top2_share']:.2f}%."
            )
            line_2 = (
                f"Afternoon is concentrated: top-2 share is {gap_text}."
            )
        elif segment == "evening":
            afternoon_top2 = segment_metrics.get("afternoon", {}).get("top2_share")
            if afternoon_top2 is not None:
                spread_gap = round(float(afternoon_top2 - metric["top2_share"]), 2)
                gap_text = f"{spread_gap:.2f} percentage points lower than afternoon"
            else:
                gap_text = "lower than afternoon"

            line_1 = (
                f"Evening: Cluster {metric['dominant_cluster_id']} has max points "
                f"({metric['dominant_share']:.2f}%). Top 2 clusters hold {metric['top2_share']:.2f}%."
            )
            line_2 = (
                f"Evening is more distributed: top-2 share is {gap_text}."
            )
        else:
            line_1 = (
                f"Night: Cluster {metric['dominant_cluster_id']} has max points "
                f"({metric['dominant_share']:.2f}%)."
            )
            line_2 = (
                f"Most night trips are in {metric['top_borough']} ({metric['top_borough_pct']:.2f}%), "
                f"followed by {metric['second_borough']} ({metric['second_borough_pct']:.2f}%)."
            )

        cards[segment] = {"label": SEGMENT_LABELS[segment], "line_1": line_1, "line_2": line_2}
    return cards
