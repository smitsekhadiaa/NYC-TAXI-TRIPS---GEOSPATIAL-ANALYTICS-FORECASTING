"""Trip pattern insights dashboard page."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from services import trip_pattern_repository as repository
from utils.formatting import format_float, format_int

BLOCK_TITLES = [
    "EWR Airport Insight (Weekday)",
    "EWR Airport Insight (Weekend)",
    "LGA Airport Insight (Weekday)",
    "LGA Airport Insight (Weekend)",
    "Morning Trip Pattern",
    "Night Trip Pattern",
    "Tipping Pattern",
    "Short Distance, High Fare Pattern",
    "Cash Payment Pattern",
    "Credit Card Payment Pattern",
]


def _parse_rule(rule_text: str) -> tuple[list[str], list[str]]:
    match = re.match(r"^\{(.*)\}\s*->\s*\{(.*)\}$", str(rule_text).strip())
    if not match:
        return [], []

    antecedent = [item.strip() for item in match.group(1).split(",") if item.strip()]
    consequent = [item.strip() for item in match.group(2).split(",") if item.strip()]
    return antecedent, consequent


def _token_to_text(token: str) -> str:
    if token.startswith("day_type="):
        value = token.split("=", 1)[1]
        return f"it is a {value.lower()} trip"
    if token.startswith("time_bin="):
        value = token.split("=", 1)[1]
        return f"time of day is {value}"
    if token.startswith("source_borough="):
        value = token.split("=", 1)[1]
        return f"pickup is in {value}"
    if token.startswith("destination_borough="):
        value = token.split("=", 1)[1]
        return f"drop-off is in {value}"
    if token.startswith("source_airport="):
        value = token.split("=", 1)[1]
        return f"pickup is from {value} airport"
    if token.startswith("destination_airport="):
        value = token.split("=", 1)[1]
        return f"drop-off is to {value} airport"
    if token.startswith("distance_bin="):
        value = token.split("=", 1)[1]
        return f"distance is in {value}"
    if token.startswith("fare_bin="):
        value = token.split("=", 1)[1]
        return f"fare is in {value}"
    if token.startswith("tip_bin="):
        value = token.split("=", 1)[1]
        return f"tip behavior is {value}"
    if token.startswith("payment_type="):
        value = token.split("=", 1)[1]
        if value == "CreditCard":
            return "payment mode is credit card"
        if value == "Cash":
            return "payment mode is cash"
        return f"payment mode is {value.lower()}"
    if token.startswith("route_type="):
        value = token.split("=", 1)[1].replace("_", " ")
        return f"route type is {value}"
    return token.replace("_", " ")


def _join_phrases(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _rule_insight_from_rule(rule_text: str, confidence: float | None = None) -> str:
    antecedent, consequent = _parse_rule(rule_text)
    if not antecedent and not consequent:
        return "This rule captures a recurring co-occurrence pattern in trips."

    left = _join_phrases([_token_to_text(token) for token in antecedent])
    right = _join_phrases([_token_to_text(token) for token in consequent])

    has_confidence = confidence is not None and confidence == confidence
    if left and right and has_confidence:
        return f"When {left}, {right} is seen in about {confidence * 100:.1f}% of such trips."
    if left and right:
        return f"When {left}, it is likely that {right}."
    if right and has_confidence:
        return f"This pattern indicates {right} in about {confidence * 100:.1f}% of matching trips."
    if right:
        return f"This pattern indicates {right}."
    return f"This pattern is associated with {left}."


def _render_kpis(metadata: dict, top_rules_df: pd.DataFrame) -> None:
    total_transactions = int(metadata.get("total_transactions_after_filtering", 0))
    total_rules = int(metadata.get("total_rules_generated", 0))
    avg_confidence = float(top_rules_df["confidence"].mean()) if not top_rules_df.empty else 0.0
    avg_lift = float(top_rules_df["lift"].mean()) if not top_rules_df.empty else 0.0

    col_1, col_2, col_3, col_4 = st.columns(4, gap="small")
    col_1.metric("Transactions Used", format_int(total_transactions))
    col_2.metric("Rules Mined", format_int(total_rules))
    col_3.metric("Top-10 Avg Confidence", f"{format_float(avg_confidence, 3)}")
    col_4.metric("Top-10 Avg Lift", f"{format_float(avg_lift, 3)}")


def _render_rule_blocks(top_rules_df) -> None:
    st.markdown("### Trip Pattern Blocks")
    if top_rules_df.empty:
        st.info("No top insight rules available.")
        return

    left_col, right_col = st.columns(2, gap="large")
    for idx, row in top_rules_df.iterrows():
        block_title = BLOCK_TITLES[idx] if idx < len(BLOCK_TITLES) else f"Pattern Insight {idx + 1}"
        target_col = left_col if idx % 2 == 0 else right_col
        with target_col:
            generated_insight = _rule_insight_from_rule(str(row["rule"]), float(row["confidence"]))
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-card-title">{block_title}</div>
                    <div class="insight-card-rule">{row["rule"]}</div>
                    <div class="insight-card-text">{generated_insight}</div>
                    <div class="insight-card-metrics">
                        <span class="insight-chip">Support: {float(row["support"]):.4f}</span>
                        <span class="insight-chip">Confidence: {float(row["confidence"]):.4f}</span>
                        <span class="insight-chip">Lift: {float(row["lift"]):.3f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render() -> None:
    st.markdown("## Trip Pattern Insights")
    st.caption(
        "Insight blocks generated from FP-Growth mining on full cleaned trip data. "
        "Filters applied: unknown OD removed, fare < $1000, distance < 30 miles."
    )

    try:
        metadata = repository.load_pattern_metadata()
        top_rules_df = repository.load_top_rules()
    except Exception as exc:
        st.error(str(exc))
        return

    _render_kpis(metadata=metadata, top_rules_df=top_rules_df)
    st.markdown("")
    _render_rule_blocks(top_rules_df=top_rules_df)

    st.markdown("")
    st.download_button(
        label="Download All Rules CSV",
        data=repository.load_all_rules().to_csv(index=False).encode("utf-8"),
        file_name="trip_pattern_rules_all.csv",
        mime="text/csv",
        key="download_trip_pattern_rules_csv",
    )
