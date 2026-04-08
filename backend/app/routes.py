"""HTTP routes for backend service."""

from flask import Blueprint, jsonify, request
from app.services.anomaly_detection_service import (
    get_anomaly_detection_summary,
    train_anomaly_detection_models,
)

api_blueprint = Blueprint("api", __name__)


@api_blueprint.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@api_blueprint.get("/api/clustering/<segment>")
def get_clustering(segment):
    # TODO: implement clustering logic
    return jsonify({"message": f"Clustering for {segment} not implemented yet"}), 501



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
    except FileNotFoundError:
        current_app.logger.warning("Anomaly summary missing, attempting auto-training")
        payload = train_anomaly_detection_models()
    except Exception as exc:
        current_app.logger.exception("Failed to load anomaly detection summary")
        return jsonify({"error": f"Internal server error: {exc}"}), 500

    return jsonify(payload), 200
