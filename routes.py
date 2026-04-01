"""HTTP routes for backend service."""

from flask import Blueprint, jsonify, request

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
def anomaly_detection_summary():
    # TODO: implement anomaly detection
    return jsonify({"message": "Anomaly detection not implemented yet"}), 501