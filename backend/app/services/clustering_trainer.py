"""Train MiniBatchKMeans clustering models for NYC taxi OD points.

Design goals:
1) Fixed K=5 (per user requirement) across all 5 segment models.
2) Numeric stability for large data (float64 + chunked training).
3) Strict centroid refinement from full assigned points so centers align with clusters.
4) Consistent plotting axes across all segment PNGs (x=long, y=lat).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

LOGGER_NAME = "nyc_taxi_clustering_trainer"
RANDOM_STATE = 42


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    start_hour: Optional[int]
    end_hour: Optional[int]


SEGMENTS: tuple[SegmentSpec, ...] = (
    SegmentSpec(name="morning", start_hour=8, end_hour=12),
    SegmentSpec(name="afternoon", start_hour=12, end_hour=16),
    SegmentSpec(name="evening", start_hour=16, end_hour=20),
    SegmentSpec(name="night", start_hour=20, end_hour=24),
    SegmentSpec(name="all", start_hour=None, end_hour=None),
)

# Plot colors requested by user.
PLOT_COLORS = ["#E53935", "#43A047", "#FDD835", "#1E88E5", "#8E24AA"]


@dataclass(frozen=True)
class ClusteringConfig:
    n_clusters: int = 5
    chunk_size: int = 300_000
    batch_size: int = 8_192
    n_init: int = 12
    max_iter: int = 220
    reassignment_ratio: float = 0.0
    tuning_sample_size: int = 250_000
    silhouette_sample_size: int = 20_000
    plotting_sample_size: int = 70_000
    plot_lon_margin: float = 0.01
    plot_lat_margin: float = 0.01


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _project_paths() -> dict:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    processed_csv_dir = data_dir / "processed_csv"
    clustering_dir = data_dir / "clustering_artifacts"

    return {
        "project_root": project_root,
        "processed_csv_dir": processed_csv_dir,
        "coords_csv": processed_csv_dir / "location_coordinates_data.csv",
        "clustering_dir": clustering_dir,
        "models_dir": clustering_dir / "models",
        "plots_dir": clustering_dir / "plots",
        "outputs_dir": clustering_dir / "outputs",
        "diagnostics_dir": clustering_dir / "diagnostics",
    }


def _get_trip_csv_paths(processed_csv_dir: Path) -> list[Path]:
    trip_files = sorted(processed_csv_dir.glob("trip_details_*.csv"))
    if not trip_files:
        raise FileNotFoundError(f"No trip CSV files found in {processed_csv_dir}")
    return trip_files


def _build_coordinate_lookup(
    coords_csv: Path,
    config: ClusteringConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    if not coords_csv.exists():
        raise FileNotFoundError(f"Missing location coordinates CSV: {coords_csv}")

    coords_df = pd.read_csv(coords_csv)
    coords_df = coords_df.dropna(subset=["location_id", "lat", "long"]).copy()
    coords_df["location_id"] = coords_df["location_id"].astype("int32")

    lon_min = float(coords_df["long"].min()) - config.plot_lon_margin
    lon_max = float(coords_df["long"].max()) + config.plot_lon_margin
    lat_min = float(coords_df["lat"].min()) - config.plot_lat_margin
    lat_max = float(coords_df["lat"].max()) + config.plot_lat_margin

    max_location_id = int(coords_df["location_id"].max())
    lat_lookup = np.full(max_location_id + 1, np.nan, dtype=np.float64)
    lon_lookup = np.full(max_location_id + 1, np.nan, dtype=np.float64)

    lat_lookup[coords_df["location_id"].to_numpy()] = coords_df["lat"].astype("float64").to_numpy()
    lon_lookup[coords_df["location_id"].to_numpy()] = coords_df["long"].astype("float64").to_numpy()

    return lon_lookup, lat_lookup, (lon_min, lon_max, lat_min, lat_max)


def _points_from_location_ids(
    location_ids: np.ndarray,
    lon_lookup: np.ndarray,
    lat_lookup: np.ndarray,
) -> np.ndarray:
    valid_range_mask = (location_ids >= 0) & (location_ids < lon_lookup.shape[0])
    if not np.any(valid_range_mask):
        return np.empty((0, 2), dtype=np.float64)

    safe_ids = location_ids[valid_range_mask]
    lons = lon_lookup[safe_ids]
    lats = lat_lookup[safe_ids]
    finite_mask = np.isfinite(lons) & np.isfinite(lats)
    if not np.any(finite_mask):
        return np.empty((0, 2), dtype=np.float64)

    # Feature order: [long, lat] to match plotting x/y convention.
    return np.column_stack((lons[finite_mask], lats[finite_mask])).astype(np.float64)


def _extract_points_for_rows(
    pickup_ids: np.ndarray,
    dropoff_ids: np.ndarray,
    row_mask: np.ndarray,
    lon_lookup: np.ndarray,
    lat_lookup: np.ndarray,
) -> np.ndarray:
    if not np.any(row_mask):
        return np.empty((0, 2), dtype=np.float64)

    masked_pickup = pickup_ids[row_mask]
    masked_dropoff = dropoff_ids[row_mask]

    pickup_points = _points_from_location_ids(masked_pickup, lon_lookup, lat_lookup)
    dropoff_points = _points_from_location_ids(masked_dropoff, lon_lookup, lat_lookup)

    if pickup_points.size == 0 and dropoff_points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if pickup_points.size == 0:
        return dropoff_points
    if dropoff_points.size == 0:
        return pickup_points
    return np.vstack((pickup_points, dropoff_points))


def _append_random_sample(
    existing_sample: np.ndarray,
    new_points: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if new_points.size == 0:
        return existing_sample

    candidate = new_points
    if len(candidate) > max_points:
        selected_idx = rng.choice(len(candidate), size=max_points, replace=False)
        candidate = candidate[selected_idx]

    if existing_sample.size == 0:
        combined = candidate
    else:
        combined = np.vstack((existing_sample, candidate))

    if len(combined) <= max_points:
        return combined

    selected_idx = rng.choice(len(combined), size=max_points, replace=False)
    return combined[selected_idx]


def _iter_trip_chunks(csv_paths: Iterable[Path], chunk_size: int) -> Iterable[pd.DataFrame]:
    usecols = ["pickup_location_id", "dropff_location_id", "pickup_time"]
    for csv_path in csv_paths:
        for chunk in pd.read_csv(
            csv_path,
            usecols=usecols,
            chunksize=chunk_size,
            low_memory=False,
        ):
            yield chunk


def _chunk_to_arrays(chunk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pickup_ids = pd.to_numeric(chunk["pickup_location_id"], errors="coerce").fillna(-1).astype(
        "int32"
    ).to_numpy()
    dropoff_ids = pd.to_numeric(chunk["dropff_location_id"], errors="coerce").fillna(-1).astype(
        "int32"
    ).to_numpy()
    pickup_hours = pd.to_numeric(
        chunk["pickup_time"].astype(str).str.slice(0, 2),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    return pickup_ids, dropoff_ids, pickup_hours


def _create_model(config: ClusteringConfig) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=config.n_clusters,
        batch_size=config.batch_size,
        n_init=config.n_init,
        max_iter=config.max_iter,
        random_state=RANDOM_STATE,
        reassignment_ratio=config.reassignment_ratio,
    )


def _collect_reference_sample(
    csv_paths: list[Path],
    lon_lookup: np.ndarray,
    lat_lookup: np.ndarray,
    config: ClusteringConfig,
) -> np.ndarray:
    logger = get_logger()
    rng = np.random.default_rng(RANDOM_STATE)
    sample_points = np.empty((0, 2), dtype=np.float64)

    logger.info(
        "Collecting representative sample for diagnostics (target=%d points)",
        config.tuning_sample_size,
    )

    for chunk in _iter_trip_chunks(csv_paths=csv_paths, chunk_size=config.chunk_size):
        pickup_ids, dropoff_ids, _pickup_hours = _chunk_to_arrays(chunk)
        all_rows_mask = np.ones(len(pickup_ids), dtype=bool)
        points = _extract_points_for_rows(
            pickup_ids=pickup_ids,
            dropoff_ids=dropoff_ids,
            row_mask=all_rows_mask,
            lon_lookup=lon_lookup,
            lat_lookup=lat_lookup,
        )
        sample_points = _append_random_sample(
            existing_sample=sample_points,
            new_points=points,
            max_points=config.tuning_sample_size,
            rng=rng,
        )
        if len(sample_points) >= config.tuning_sample_size:
            break

    if len(sample_points) < config.n_clusters:
        raise RuntimeError(
            f"Not enough sample points for clustering diagnostics. Collected={len(sample_points)}"
        )

    logger.info("Collected %d points for diagnostics", len(sample_points))
    return sample_points


def _diagnostics_for_fixed_k(sample_points: np.ndarray, config: ClusteringConfig) -> list[dict]:
    model = _create_model(config)
    labels = model.fit_predict(sample_points)
    inertia = float(model.inertia_)
    silhouette = float(
        silhouette_score(
            sample_points,
            labels,
            sample_size=min(config.silhouette_sample_size, len(sample_points)),
            random_state=RANDOM_STATE,
        )
    )
    return [
        {
            "k": int(config.n_clusters),
            "inertia": inertia,
            "silhouette": silhouette,
        }
    ]


def _fit_segment_models(
    csv_paths: list[Path],
    lon_lookup: np.ndarray,
    lat_lookup: np.ndarray,
    config: ClusteringConfig,
) -> tuple[dict[str, MiniBatchKMeans], dict[str, np.ndarray], dict[str, int]]:
    logger = get_logger()
    rng = np.random.default_rng(RANDOM_STATE)

    models = {segment.name: _create_model(config) for segment in SEGMENTS}
    sample_points = {
        segment.name: np.empty((0, 2), dtype=np.float64)
        for segment in SEGMENTS
    }
    point_counts = {segment.name: 0 for segment in SEGMENTS}

    logger.info("Training 5 MiniBatchKMeans(K=5) models on all available monthly CSV files")
    for chunk_idx, chunk in enumerate(
        _iter_trip_chunks(csv_paths=csv_paths, chunk_size=config.chunk_size),
        start=1,
    ):
        pickup_ids, dropoff_ids, pickup_hours = _chunk_to_arrays(chunk)
        valid_hour_mask = np.isfinite(pickup_hours)

        for segment in SEGMENTS:
            if segment.name == "all":
                segment_mask = np.ones(len(pickup_ids), dtype=bool)
            else:
                segment_mask = (
                    valid_hour_mask
                    & (pickup_hours >= float(segment.start_hour))
                    & (pickup_hours < float(segment.end_hour))
                )

            segment_points = _extract_points_for_rows(
                pickup_ids=pickup_ids,
                dropoff_ids=dropoff_ids,
                row_mask=segment_mask,
                lon_lookup=lon_lookup,
                lat_lookup=lat_lookup,
            )
            if segment_points.size == 0:
                continue

            models[segment.name].partial_fit(segment_points)
            point_counts[segment.name] += int(len(segment_points))
            sample_points[segment.name] = _append_random_sample(
                existing_sample=sample_points[segment.name],
                new_points=segment_points,
                max_points=config.plotting_sample_size,
                rng=rng,
            )

        if chunk_idx % 10 == 0:
            logger.info("Processed %d chunks for model fitting", chunk_idx)

    for segment in SEGMENTS:
        if point_counts[segment.name] == 0:
            raise RuntimeError(f"No points collected for segment '{segment.name}'.")

    return models, sample_points, point_counts


def _refine_centers_from_full_assignments(
    models: dict[str, MiniBatchKMeans],
    csv_paths: list[Path],
    lon_lookup: np.ndarray,
    lat_lookup: np.ndarray,
    config: ClusteringConfig,
) -> dict[str, dict]:
    """Recompute cluster centers as exact mean of assigned points (chunked).

    This makes centers robust and keeps them inside observed cluster regions.
    """
    logger = get_logger()
    n_clusters = config.n_clusters
    sums = {segment.name: np.zeros((n_clusters, 2), dtype=np.float64) for segment in SEGMENTS}
    counts = {segment.name: np.zeros(n_clusters, dtype=np.int64) for segment in SEGMENTS}

    logger.info("Refining centers from full assignments (second pass)")
    for chunk_idx, chunk in enumerate(
        _iter_trip_chunks(csv_paths=csv_paths, chunk_size=config.chunk_size),
        start=1,
    ):
        pickup_ids, dropoff_ids, pickup_hours = _chunk_to_arrays(chunk)
        valid_hour_mask = np.isfinite(pickup_hours)

        for segment in SEGMENTS:
            if segment.name == "all":
                segment_mask = np.ones(len(pickup_ids), dtype=bool)
            else:
                segment_mask = (
                    valid_hour_mask
                    & (pickup_hours >= float(segment.start_hour))
                    & (pickup_hours < float(segment.end_hour))
                )

            points = _extract_points_for_rows(
                pickup_ids=pickup_ids,
                dropoff_ids=dropoff_ids,
                row_mask=segment_mask,
                lon_lookup=lon_lookup,
                lat_lookup=lat_lookup,
            )
            if points.size == 0:
                continue

            labels = models[segment.name].predict(points)
            np.add.at(sums[segment.name], labels, points)
            np.add.at(counts[segment.name], labels, 1)

        if chunk_idx % 10 == 0:
            logger.info("Processed %d chunks for center refinement", chunk_idx)

    refinement_summary: dict[str, dict] = {}
    for segment in SEGMENTS:
        name = segment.name
        current_centers = models[name].cluster_centers_.copy()
        refined_centers = current_centers.copy()

        for cluster_id in range(n_clusters):
            if counts[name][cluster_id] > 0:
                refined_centers[cluster_id] = sums[name][cluster_id] / counts[name][cluster_id]

        models[name].cluster_centers_ = refined_centers
        refinement_summary[name] = {
            "cluster_counts": counts[name].tolist(),
            "empty_clusters": int(np.sum(counts[name] == 0)),
        }

    return refinement_summary


def _save_k_diagnostics_plot(diagnostics: list[dict], output_path: Path) -> None:
    diag_df = pd.DataFrame(diagnostics)
    k_value = int(diag_df.loc[0, "k"])
    inertia = float(diag_df.loc[0, "inertia"])
    silhouette = float(diag_df.loc[0, "silhouette"])

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.axis("off")
    ax.text(
        0.02,
        0.75,
        f"Fixed K Diagnostics\nK = {k_value}",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(0.02, 0.47, f"Inertia: {inertia:,.4f}", fontsize=12)
    ax.text(0.02, 0.29, f"Silhouette: {silhouette:,.6f}", fontsize=12)
    ax.text(0.02, 0.10, "Model: MiniBatchKMeans", fontsize=11, color="#4f4f4f")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_cluster_visualization_png(
    segment_name: str,
    model: MiniBatchKMeans,
    sampled_points: np.ndarray,
    output_path: Path,
    axis_bounds: tuple[float, float, float, float],
) -> None:
    if sampled_points.size == 0:
        return

    labels = model.predict(sampled_points)
    centers = model.cluster_centers_
    unique_labels = sorted(np.unique(labels).tolist())

    fig, ax = plt.subplots(figsize=(9, 7))
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_points = sampled_points[cluster_mask]
        if cluster_points.size == 0:
            continue

        color = PLOT_COLORS[cluster_id % len(PLOT_COLORS)]
        ax.scatter(
            cluster_points[:, 0],  # long
            cluster_points[:, 1],  # lat
            s=8,
            alpha=0.62,
            color=color,
            label=str(cluster_id),
        )

    ax.scatter(
        centers[:, 0],  # center_long
        centers[:, 1],  # center_lat
        s=160,
        marker="X",
        color="black",
        linewidths=0.6,
        edgecolors="white",
        label="center",
        zorder=5,
    )
    lon_min, lon_max, lat_min, lat_max = axis_bounds
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(f"{segment_name.title()} MiniBatchKMeans, K={model.n_clusters}")
    ax.set_xlabel("long")
    ax.set_ylabel("lat")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_artifacts(
    models: dict[str, MiniBatchKMeans],
    sampled_points: dict[str, np.ndarray],
    diagnostics: list[dict],
    point_counts: dict[str, int],
    refinement_summary: dict[str, dict],
    axis_bounds: tuple[float, float, float, float],
    paths: dict,
    config: ClusteringConfig,
) -> dict:
    for key in ("clustering_dir", "models_dir", "plots_dir", "outputs_dir", "diagnostics_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "selected_k": int(config.n_clusters),
        "models": {},
        "plots": {},
        "outputs": {},
        "diagnostics": {},
    }

    for segment in SEGMENTS:
        segment_name = segment.name
        model = models[segment_name]
        points = sampled_points[segment_name]

        model_path = paths["models_dir"] / f"minibatch_kmeans_{segment_name}.pkl"
        centers_csv_path = paths["outputs_dir"] / f"cluster_centers_{segment_name}.csv"
        sampled_csv_path = paths["outputs_dir"] / f"sample_points_labeled_{segment_name}.csv"
        plot_path = paths["plots_dir"] / f"{segment_name}_clusters.png"

        joblib.dump(model, model_path)

        centers_df = pd.DataFrame(
            {
                "cluster_id": np.arange(model.n_clusters),
                "center_long": model.cluster_centers_[:, 0],
                "center_lat": model.cluster_centers_[:, 1],
            }
        )
        centers_df.to_csv(centers_csv_path, index=False)

        if points.size > 0:
            point_labels = model.predict(points)
            sampled_df = pd.DataFrame(
                {
                    "long": points[:, 0],
                    "lat": points[:, 1],
                    "cluster_id": point_labels,
                }
            )
            sampled_df.to_csv(sampled_csv_path, index=False)

        _save_cluster_visualization_png(
            segment_name=segment_name,
            model=model,
            sampled_points=points,
            output_path=plot_path,
            axis_bounds=axis_bounds,
        )

        summary["models"][segment_name] = str(model_path)
        summary["plots"][segment_name] = str(plot_path)
        summary["outputs"][segment_name] = {
            "centers_csv": str(centers_csv_path),
            "sample_points_csv": str(sampled_csv_path),
            "points_used_for_training": int(point_counts[segment_name]),
            "points_used_for_plot": int(len(points)),
            "refinement": refinement_summary.get(segment_name, {}),
        }

    diagnostics_json_path = paths["diagnostics_dir"] / "k_diagnostics.json"
    diagnostics_plot_path = paths["diagnostics_dir"] / "k_diagnostics.png"
    metadata_json_path = paths["clustering_dir"] / "clustering_metadata.json"

    with open(diagnostics_json_path, "w", encoding="utf-8") as out_file:
        json.dump(diagnostics, out_file, indent=2)
    _save_k_diagnostics_plot(diagnostics=diagnostics, output_path=diagnostics_plot_path)

    lon_min, lon_max, lat_min, lat_max = axis_bounds
    metadata = {
        "selected_k": int(config.n_clusters),
        "segments": [segment.name for segment in SEGMENTS],
        "point_counts": point_counts,
        "axis_bounds": {
            "x_long_min": lon_min,
            "x_long_max": lon_max,
            "y_lat_min": lat_min,
            "y_lat_max": lat_max,
        },
        "config": {
            "chunk_size": config.chunk_size,
            "batch_size": config.batch_size,
            "n_init": config.n_init,
            "max_iter": config.max_iter,
            "reassignment_ratio": config.reassignment_ratio,
            "tuning_sample_size": config.tuning_sample_size,
            "silhouette_sample_size": config.silhouette_sample_size,
            "plotting_sample_size": config.plotting_sample_size,
            "plot_colors": PLOT_COLORS,
            "random_state": RANDOM_STATE,
        },
        "refinement_summary": refinement_summary,
    }
    with open(metadata_json_path, "w", encoding="utf-8") as out_file:
        json.dump(metadata, out_file, indent=2)

    summary["diagnostics"] = {
        "k_diagnostics_json": str(diagnostics_json_path),
        "k_diagnostics_plot": str(diagnostics_plot_path),
        "metadata_json": str(metadata_json_path),
    }
    return summary


def train_minibatch_kmeans_models(config: Optional[ClusteringConfig] = None) -> dict:
    """Train and persist 5 MiniBatchKMeans models from processed monthly CSV files."""
    logger = get_logger()
    cfg = config or ClusteringConfig()
    paths = _project_paths()

    trip_csv_paths = _get_trip_csv_paths(paths["processed_csv_dir"])
    lon_lookup, lat_lookup, axis_bounds = _build_coordinate_lookup(paths["coords_csv"], cfg)

    logger.info("Using %d trip CSV files for clustering", len(trip_csv_paths))
    logger.info("Fixed K=%d (user-specified)", cfg.n_clusters)

    diagnostics_sample = _collect_reference_sample(
        csv_paths=trip_csv_paths,
        lon_lookup=lon_lookup,
        lat_lookup=lat_lookup,
        config=cfg,
    )
    diagnostics = _diagnostics_for_fixed_k(diagnostics_sample, cfg)

    models, sampled_points, point_counts = _fit_segment_models(
        csv_paths=trip_csv_paths,
        lon_lookup=lon_lookup,
        lat_lookup=lat_lookup,
        config=cfg,
    )
    refinement_summary = _refine_centers_from_full_assignments(
        models=models,
        csv_paths=trip_csv_paths,
        lon_lookup=lon_lookup,
        lat_lookup=lat_lookup,
        config=cfg,
    )
    summary = _save_artifacts(
        models=models,
        sampled_points=sampled_points,
        diagnostics=diagnostics,
        point_counts=point_counts,
        refinement_summary=refinement_summary,
        axis_bounds=axis_bounds,
        paths=paths,
        config=cfg,
    )
    logger.info("Clustering training completed. Artifacts in %s", paths["clustering_dir"])
    return summary
