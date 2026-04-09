"""Global CSS styling for Streamlit frontend."""

from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Playfair+Display:ital,wght@1,600&display=swap');

        :root {
            --bg: #f3f6fb;
            --surface: #ffffff;
            --primary: #0f4c81;
            --primary-dark: #0b3558;
            --accent: #178a9f;
            --text: #1d2733;
            --muted: #607085;
            --border: #d8e1ec;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 20%, #eaf2fb 0%, transparent 35%),
                radial-gradient(circle at 90% 10%, #e9f5f7 0%, transparent 30%),
                var(--bg);
            color: var(--text);
            font-family: 'Manrope', sans-serif;
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 1rem;
            padding-bottom: 1.8rem;
        }

        .hero-card {
            background: linear-gradient(145deg, rgba(15, 76, 129, 0.98), rgba(9, 46, 78, 0.98));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 1.8rem 1.9rem;
            box-shadow: 0 14px 35px rgba(10, 34, 57, 0.2);
            margin-bottom: 1rem;
        }

        .hero-title {
            color: #ffffff;
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.45rem;
            letter-spacing: 0.2px;
        }

        .hero-subtitle {
            color: rgba(234, 243, 255, 0.95);
            font-size: 0.98rem;
            margin-bottom: 0.95rem;
            font-weight: 600;
        }

        .hero-mission {
            font-family: 'Playfair Display', serif;
            font-style: italic;
            color: #eaf3ff;
            font-size: 1.08rem;
            line-height: 1.5;
            border-left: 3px solid rgba(255, 255, 255, 0.55);
            padding-left: 0.9rem;
            margin-top: 0.2rem;
        }

        .kpi-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            min-height: 92px;
            box-shadow: 0 6px 18px rgba(20, 56, 91, 0.07);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .kpi-label {
            color: var(--muted);
            font-weight: 700;
            font-size: 0.86rem;
            letter-spacing: 0.2px;
            margin-bottom: 0.45rem;
        }

        .kpi-value {
            color: var(--primary-dark);
            font-size: 1.35rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .map-legend-item {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.35rem 0.5rem;
            min-height: 44px;
            font-size: 0.78rem;
            color: #3c4f64;
        }

        .map-legend-swatch {
            width: 12px;
            height: 12px;
            border-radius: 3px;
            display: inline-block;
            flex: 0 0 auto;
        }

        .map-legend-text {
            line-height: 1.2;
            font-weight: 600;
        }

        .app-footer {
            margin-top: 1.4rem;
            padding: 0.85rem 0.6rem;
            text-align: center;
            font-size: 0.82rem;
            font-weight: 600;
            color: #62758d;
            border-top: 1px solid #d8e1ec;
            letter-spacing: 0.12px;
        }

        .insight-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #d8e1ec;
            border-left: 4px solid #0f4c81;
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.7rem;
            box-shadow: 0 5px 16px rgba(15, 76, 129, 0.08);
        }

        .insight-card-title {
            font-size: 0.84rem;
            font-weight: 800;
            letter-spacing: 0.2px;
            color: #0f4c81;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }

        .insight-card-rule {
            font-size: 0.84rem;
            line-height: 1.4;
            color: #183147;
            margin-bottom: 0.42rem;
            font-weight: 700;
        }

        .insight-card-text {
            font-size: 0.86rem;
            line-height: 1.45;
            color: #2e4a62;
            margin-bottom: 0.5rem;
        }

        .insight-card-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
        }

        .insight-chip {
            font-size: 0.76rem;
            font-weight: 700;
            color: #0f4c81;
            background: #eaf3ff;
            border: 1px solid #cfe3fb;
            border-radius: 999px;
            padding: 0.18rem 0.52rem;
        }

        .panel-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 6px 18px rgba(20, 56, 91, 0.06);
        }

        .section-title {
            color: var(--primary-dark);
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        [data-testid="stSidebarNav"] {
            display: none !important;
        }

        .stDeployButton {
            display: none !important;
        }

        [data-testid="stToolbar"] {
            display: none !important;
        }

        [data-testid="stHeaderActionElements"] {
            display: none !important;
        }

        @media (max-width: 900px) {
            .main .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            .hero-title {
                font-size: 1.65rem;
            }

            .hero-mission {
                font-size: 0.97rem;
            }

            .kpi-value {
                font-size: 1.28rem;
            }

            .map-legend-item {
                font-size: 0.74rem;
                min-height: 40px;
                padding: 0.3rem 0.4rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
