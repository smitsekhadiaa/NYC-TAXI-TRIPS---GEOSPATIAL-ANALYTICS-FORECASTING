"""Flask application factory."""

from __future__ import annotations

import logging

from flask import Flask

from app.cli import register_cli_commands
from app.routes import api_blueprint


def create_app() -> Flask:
    app = Flask(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app.register_blueprint(api_blueprint)
    register_cli_commands(app)
    return app
