"""HTTP routes for backend service."""

from __future__ import annotations

from datetime import date

from flask import Blueprint, current_app, jsonify, request

from app.config import load_settings
from app.services.anomaly_detection_service import (
    get_anomaly_detection_summary,
    train_anomaly_detection_models,
)
from app.services.clustering_data_service import get_clustering_map_payload
from app.services.mysql_loader import load_all_dataframes_to_mysql
from app.services.trip_forecasting_service import (
    get_forecasting_metadata,
    predict_fare_and_eta,
)

api_blueprint = Blueprint("api", __name__)


@api_blueprint.get("/health")
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@api_blueprint.post("/api/load-mysql")
def load_mysql() -> tuple:
    settings = load_settings()
    current_app.logger.info("Starting MySQL data load")
    summary = load_all_dataframes_to_mysql(settings)
    return jsonify({"database": settings.mysql_database, "tables_loaded": summary}), 200


@api_blueprint.get("/api/clustering/<segment>")
def get_clustering(segment: str) -> tuple:
    max_points = request.args.get("max_points", default=25_000, type=int)
    try:
        payload = get_clustering_map_payload(segment=segment, max_points=max_points)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        current_app.logger.exception("Failed to build clustering payload")
        return jsonify({"error": f"Internal server error: {exc}"}), 500

    return jsonify(payload), 200


@api_blueprint.get("/api/trip-forecast/metadata")
def trip_forecast_metadata() -> tuple:
    try:
        payload = get_forecasting_metadata()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # pragma: no cover
        current_app.logger.exception("Failed to load trip forecast metadata")
        return jsonify({"error": f"Internal server error: {exc}"}), 500

    return jsonify(payload), 200


@api_blueprint.post("/api/trip-forecast/predict")
def trip_forecast_predict() -> tuple:
    payload = request.get_json(silent=True) or {}
    source_location_id = payload.get("source_location_id")
    destination_location_id = payload.get("destination_location_id")
    pickup_date_text = payload.get("pickup_date")
    pickup_hour = payload.get("pickup_hour")

    if source_location_id is None or destination_location_id is None:
        return jsonify({"error": "source_location_id and destination_location_id are required."}), 400
    if pickup_date_text is None or pickup_hour is None:
        return jsonify({"error": "pickup_date and pickup_hour are required."}), 400

    try:
        pickup_date_value = date.fromisoformat(str(pickup_date_text))
        result = predict_fare_and_eta(
            source_location_id=int(source_location_id),
            destination_location_id=int(destination_location_id),
            pickup_date_value=pickup_date_value,
            pickup_hour=int(pickup_hour),
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        current_app.logger.exception("Failed to generate trip forecast")
        return jsonify({"error": f"Internal server error: {exc}"}), 500

    return jsonify(result), 200


@api_blueprint.get("/api/anomaly-detection/summary")
def anomaly_detection_summary() -> tuple:
    try:
        payload = get_anomaly_detection_summary()
    except FileNotFoundError as exc:
        current_app.logger.warning("Anomaly summary missing, attempting auto-training")
        try:
            payload = train_anomaly_detection_models()
        except Exception as train_exc:  # pragma: no cover
            current_app.logger.exception("Auto-training anomaly models failed")
            return jsonify({"error": f"{exc}. Auto-training failed: {train_exc}"}), 404
    except Exception as exc:  # pragma: no cover
        current_app.logger.exception("Failed to load anomaly detection summary")
        return jsonify({"error": f"Internal server error: {exc}"}), 500

    return jsonify(payload), 200
