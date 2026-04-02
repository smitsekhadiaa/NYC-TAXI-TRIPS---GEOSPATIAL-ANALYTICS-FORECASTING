"""Application configuration for Flask backend."""

from dataclasses import dataclass


@dataclass
class Settings:
    flask_host: str = "0.0.0.0"
    flask_port: int = 5000
    flask_debug: bool = True


def load_settings() -> Settings:
    return Settings()