"""Application configuration for Flask backend."""

"""Application configuration for Flask backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    flask_host: str
    flask_port: int
    flask_debug: bool
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    mysql_charset: str
    spark_insert_batch_size: int


DEFAULT_DATABASE_NAME = "nyc_taxi_db"


def load_settings() -> Settings:
    return Settings(
        flask_host="0.0.0.0",
        flask_port=5001,
        flask_debug=True,
        mysql_host="127.0.0.1",
        mysql_port=3306,
        mysql_user="root",
        mysql_password="rootpassword",
        mysql_database=DEFAULT_DATABASE_NAME,
        mysql_charset="utf8mb4",
        spark_insert_batch_size=5000,
    )
