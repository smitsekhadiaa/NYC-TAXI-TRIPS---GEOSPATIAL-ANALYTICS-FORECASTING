"""Main Streamlit entrypoint for NYC taxi frontend."""

from __future__ import annotations

import streamlit as st
from streamlit_option_menu import option_menu

from components.styles import inject_global_styles
from config import APP_SUBTITLE, APP_TITLE
from views import clustering, dashboard, time_series_prediction


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🚕",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_global_styles()

    st.markdown(f"#### {APP_SUBTITLE}")

    selected_page = option_menu(
        menu_title=None,
        options=["Dashboard", "Clustering", "Trip fare and eta prediction"],
        icons=["speedometer2", "diagram-3", "graph-up-arrow"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {
                "padding": "0.3rem 0.6rem",
                "background-color": "#ffffff",
                "border": "1px solid #d8e1ec",
                "border-radius": "12px",
            },
            "icon": {"color": "#0f4c81", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px",
                "font-weight": "700",
                "text-align": "center",
                "color": "#35506a",
                "border-radius": "8px",
                "margin": "0 0.25rem",
            },
            "nav-link-selected": {
                "background-color": "#0f4c81",
                "color": "white",
            },
        },
    )
    st.markdown("")

    if selected_page == "Dashboard":
        dashboard.render()
    elif selected_page == "Clustering":
        clustering.render()
    elif selected_page == "Trip fare and eta prediction":
        time_series_prediction.render()


if __name__ == "__main__":
    main()
