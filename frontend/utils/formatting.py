"""Formatting helpers for frontend presentation."""

from __future__ import annotations

from datetime import date, datetime


def format_int(value: int | float) -> str:
    return f"{int(value):,}"


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def format_iso_date(iso_date: str) -> str:
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    return dt.strftime("%b %d, %Y")


def format_hour_label(hour: int) -> str:
    if hour == 24:
        return "24:00"
    return f"{hour:02d}:00"


def date_to_iso(value: date) -> str:
    return value.isoformat()

