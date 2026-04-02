"""Entrypoint for running Flask backend locally."""

from __future__ import annotations

from app import create_app
from app.config import load_settings

app = create_app()


if __name__ == "__main__":
    settings = load_settings()
    app.run(
        host=settings.flask_host,
        port=settings.flask_port,
        debug=settings.flask_debug,
    )
