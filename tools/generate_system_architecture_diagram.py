from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "architecture"
PNG_PATH = OUT_DIR / "system_architecture_high_level.png"
SVG_PATH = OUT_DIR / "system_architecture_high_level.svg"


def draw_box(ax, x, y, w, h, title, subtitle, fc="#f7fbff", ec="#2b6cb0"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=11, weight="bold", color="#16324f")
    ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=8.5, color="#2d3748")


def draw_arrow(ax, x1, y1, x2, y2, text=None, color="#1a365d", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=3,
    )
    ax.add_patch(arr)
    if text:
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(xm, ym + 0.015, text, fontsize=8, color="#1a365d", ha="center", va="center")


def icon_database(ax, cx, cy, s=0.038, color="#2563eb"):
    h = s * 0.86
    w = s * 1.5
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="#dbeafe", edgecolor=color, lw=1.5, zorder=4))
    ax.add_patch(Ellipse((cx, cy + h / 2), w, h * 0.28, facecolor="#bfdbfe", edgecolor=color, lw=1.5, zorder=5))
    ax.add_patch(Ellipse((cx, cy), w, h * 0.28, fill=False, edgecolor=color, lw=1.2, zorder=5))
    ax.add_patch(Ellipse((cx, cy - h / 2), w, h * 0.28, fill=False, edgecolor=color, lw=1.2, zorder=5))


def icon_spark(ax, cx, cy, s=0.042, color="#ea580c"):
    bolt = Polygon(
        [
            (cx - s * 0.18, cy + s * 0.30),
            (cx + s * 0.02, cy + s * 0.05),
            (cx - s * 0.05, cy + s * 0.05),
            (cx + s * 0.18, cy - s * 0.30),
            (cx, cy - s * 0.02),
            (cx + s * 0.07, cy - s * 0.02),
        ],
        closed=True,
        facecolor="#fed7aa",
        edgecolor=color,
        lw=1.5,
        zorder=5,
    )
    ax.add_patch(bolt)
    for r in (0.14, 0.22, 0.30):
        ax.add_patch(Arc((cx - s * 0.30, cy + s * 0.32), s * r * 2.2, s * r * 1.6, theta1=300, theta2=30, color=color, lw=1.2, zorder=5))


def icon_flask(ax, cx, cy, s=0.042, color="#059669"):
    neck_w = s * 0.18
    neck_h = s * 0.30
    body_w = s * 0.70
    body_h = s * 0.55
    ax.add_patch(Rectangle((cx - neck_w / 2, cy + body_h * 0.15), neck_w, neck_h, facecolor="#d1fae5", edgecolor=color, lw=1.4, zorder=5))
    body = Polygon(
        [
            (cx - body_w / 2, cy - body_h / 2),
            (cx + body_w / 2, cy - body_h / 2),
            (cx + neck_w / 2, cy + body_h * 0.15),
            (cx - neck_w / 2, cy + body_h * 0.15),
        ],
        closed=True,
        facecolor="#d1fae5",
        edgecolor=color,
        lw=1.4,
        zorder=5,
    )
    ax.add_patch(body)
    ax.add_patch(Rectangle((cx - body_w * 0.38, cy - body_h * 0.20), body_w * 0.76, body_h * 0.16, facecolor="#6ee7b7", edgecolor="none", zorder=6))


def icon_streamlit(ax, cx, cy, s=0.042, color="#be123c"):
    ax.add_patch(Circle((cx, cy), s * 0.35, facecolor="#ffe4e6", edgecolor=color, lw=1.4, zorder=5))
    tri = Polygon(
        [
            (cx - s * 0.30, cy + s * 0.20),
            (cx + s * 0.20, cy + s * 0.36),
            (cx + s * 0.02, cy - s * 0.24),
        ],
        closed=True,
        facecolor="#fecdd3",
        edgecolor=color,
        lw=1.4,
        zorder=6,
    )
    ax.add_patch(tri)


def icon_service(ax, cx, cy, s=0.035, color="#7c3aed"):
    ax.add_patch(Circle((cx, cy), s * 0.27, facecolor="#ede9fe", edgecolor=color, lw=1.3, zorder=6))
    for k in range(8):
        ang = k * 45
        import math

        x = cx + (s * 0.44) * math.cos(math.radians(ang))
        y = cy + (s * 0.44) * math.sin(math.radians(ang))
        ax.add_patch(Circle((x, y), s * 0.05, facecolor="#c4b5fd", edgecolor=color, lw=0.8, zorder=6))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 11.25), dpi=300)
    fig.patch.set_facecolor("#eef4fb")
    ax.set_facecolor("#eef4fb")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "NYC Taxi Trips - High-Level System Architecture",
        ha="center",
        va="center",
        fontsize=24,
        weight="bold",
        color="#0f4c81",
    )

    # Main pipeline nodes
    draw_box(ax, 0.04, 0.70, 0.13, 0.15, "NYC Open Data", "Public TLC datasets")
    icon_database(ax, 0.055, 0.84)

    draw_box(ax, 0.21, 0.70, 0.14, 0.15, "Staging Layer", "raw_src_data\nmonthly parquet")
    icon_database(ax, 0.225, 0.84, s=0.034, color="#4c51bf")

    draw_box(ax, 0.39, 0.70, 0.15, 0.15, "Transform + QA", "PySpark ETL\ncleaning + feature prep")
    icon_spark(ax, 0.405, 0.84)

    draw_box(ax, 0.58, 0.70, 0.14, 0.15, "Processed Data", "processed_csv\nstatistics.json")
    icon_database(ax, 0.595, 0.84, s=0.034, color="#0f766e")

    # MySQL path
    draw_box(ax, 0.52, 0.45, 0.14, 0.15, "MySQL Loader", "CSV -> tables\nchunked inserts")
    icon_service(ax, 0.537, 0.585)

    draw_box(ax, 0.70, 0.45, 0.13, 0.15, "MySQL", "trip_details_*\nreference tables")
    icon_database(ax, 0.715, 0.585)

    # ML path
    draw_box(ax, 0.04, 0.28, 0.19, 0.13, "Clustering Service", "MiniBatchKMeans\nOD cluster artifacts")
    icon_service(ax, 0.058, 0.392)

    draw_box(ax, 0.27, 0.28, 0.19, 0.13, "Forecasting Service", "Fare + ETA models\n(.pkl + metrics)")
    icon_service(ax, 0.288, 0.392)

    draw_box(ax, 0.50, 0.28, 0.19, 0.13, "Anomaly Service", "Isolation Forest\nspeed + fare outliers")
    icon_service(ax, 0.518, 0.392)

    draw_box(ax, 0.73, 0.28, 0.12, 0.13, "Model Artifacts", "models + outputs\nJSON / CSV / PKL")
    icon_database(ax, 0.745, 0.392, s=0.03, color="#7c3aed")

    # API and UI
    draw_box(ax, 0.70, 0.70, 0.14, 0.15, "Flask API", "routes + services\n/health /api/*")
    icon_flask(ax, 0.715, 0.84)

    draw_box(ax, 0.87, 0.70, 0.11, 0.15, "Streamlit UI", "Dashboard\nClustering\nForecasting\nAnomaly")
    icon_streamlit(ax, 0.885, 0.84)

    # Arrows - top pipeline
    draw_arrow(ax, 0.17, 0.775, 0.21, 0.775)
    draw_arrow(ax, 0.35, 0.775, 0.39, 0.775)
    draw_arrow(ax, 0.54, 0.775, 0.58, 0.775)

    # Branch to MySQL path
    draw_arrow(ax, 0.62, 0.70, 0.59, 0.60, text="CSV feed", rad=0.0)
    draw_arrow(ax, 0.66, 0.525, 0.70, 0.525, text="load tables")

    # Branch to ML services
    draw_arrow(ax, 0.60, 0.70, 0.15, 0.41, text="training data", rad=0.15)
    draw_arrow(ax, 0.60, 0.70, 0.38, 0.41, rad=0.08)
    draw_arrow(ax, 0.60, 0.70, 0.61, 0.41, rad=-0.02)

    draw_arrow(ax, 0.23, 0.345, 0.27, 0.345)
    draw_arrow(ax, 0.46, 0.345, 0.50, 0.345)
    draw_arrow(ax, 0.69, 0.345, 0.73, 0.345)

    # ML and DB to API
    draw_arrow(ax, 0.83, 0.525, 0.77, 0.70, text="query results", rad=0.10)
    draw_arrow(ax, 0.79, 0.41, 0.77, 0.70, text="load models", rad=-0.08)

    # API to UI
    draw_arrow(ax, 0.84, 0.775, 0.87, 0.775, text="REST JSON")

    # Legend
    ax.text(0.04, 0.12, "Legend:", fontsize=11, weight="bold", color="#16324f")
    ax.text(0.11, 0.12, "Database icon = storage layers (staging / MySQL / artifacts)", fontsize=9, color="#2d3748")
    ax.text(0.11, 0.09, "Spark icon = ETL + quality checks", fontsize=9, color="#2d3748")
    ax.text(0.11, 0.06, "Flask icon = backend API", fontsize=9, color="#2d3748")
    ax.text(0.11, 0.03, "Streamlit icon = frontend UI", fontsize=9, color="#2d3748")
    ax.text(0.52, 0.06, "Shared service icon = reusable backend/ML services", fontsize=9, color="#2d3748")

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)

    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
