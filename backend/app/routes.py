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


@api_blueprint.post("/api/trip-forecast/predict")
def trip_forecast_predict():
    data = request.get_json()
    
    # TODO: implement prediction logic
    return jsonify({"message": "Prediction endpoint not implemented yet"}), 501


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
