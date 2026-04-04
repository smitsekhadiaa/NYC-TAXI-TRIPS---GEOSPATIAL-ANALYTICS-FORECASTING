"""Flask CLI commands for backend service."""

from __future__ import annotations

import click
from flask import Flask

from app.config import load_settings
from app.services.anomaly_detection_service import train_anomaly_detection_models
from app.services.clustering_trainer import train_minibatch_kmeans_models
from app.services.mysql_loader import load_all_dataframes_to_mysql
from app.services.trip_pattern_rule_mining_service import train_trip_pattern_rules
from app.services.trip_forecasting_service import train_trip_forecasting_models


def register_cli_commands(app: Flask) -> None:
    @app.cli.command("load-mysql")
    def load_mysql_command() -> None:
        """Create nyc_taxi_db and load all required dataframes into MySQL tables."""
        settings = load_settings()
        summary = load_all_dataframes_to_mysql(settings)

        click.echo(f"Database: {settings.mysql_database}")
        for table_name, row_count in summary.items():
            click.echo(f"- {table_name}: {row_count} rows")

    @app.cli.command("train-clustering")
    def train_clustering_command() -> None:
        """Train and save 5 MiniBatchKMeans clustering models and PNG visualizations."""
        summary = train_minibatch_kmeans_models()

        click.echo("Clustering training completed.")
        click.echo(f"Selected K: {summary['selected_k']}")
        click.echo("Saved model files:")
        for segment_name, model_path in summary["models"].items():
            click.echo(f"- {segment_name}: {model_path}")

    @app.cli.command("train-trip-forecast")
    def train_trip_forecast_command() -> None:
        """Train and save fare/ETA forecasting models from processed CSV data."""
        summary = train_trip_forecasting_models()

        click.echo("Trip forecasting training completed.")
        click.echo("Saved model files:")
        for model_name, model_path in summary["models"].items():
            click.echo(f"- {model_name}: {model_path}")
        click.echo("Saved outputs:")
        for output_name, output_path in summary["outputs"].items():
            click.echo(f"- {output_name}: {output_path}")

    @app.cli.command("train-anomaly-models")
    def train_anomaly_models_command() -> None:
        """Train and save anomaly detection models and outputs."""
        summary = train_anomaly_detection_models()

        click.echo("Anomaly detection training completed.")
        click.echo("Saved model files:")
        for model_name, model_path in summary["models"].items():
            click.echo(f"- {model_name}: {model_path}")
        click.echo("Saved output files:")
        click.echo(f"- extreme_speed_csv: {summary['anomalies']['extreme_speed']['output_csv']}")
        click.echo(f"- fare_outlier_csv: {summary['anomalies']['fare_outlier']['output_csv']}")

    @app.cli.command("train-anamoly-models")
    def train_anamoly_models_command() -> None:
        """Alias for train-anomaly-models (spelling-compatible)."""
        train_anomaly_models_command()

    @app.cli.command("train-trip-pattern-rules")
    def train_trip_pattern_rules_command() -> None:
        """Train FP-Growth trip pattern rules and save insight CSV artifacts."""
        summary = train_trip_pattern_rules()

        click.echo("Trip pattern rule mining completed.")
        click.echo(f"- total_transactions_after_filtering: {summary['total_transactions_after_filtering']}")
        click.echo(f"- total_rules_generated: {summary['total_rules_generated']}")
        click.echo(f"- top_rules_count: {summary['top_rules_count']}")
        click.echo(f"- all_rules_csv: {summary['outputs']['all_rules_csv']}")
        click.echo(f"- top_rules_csv: {summary['outputs']['top_rules_csv']}")
