"""
GENX Observatory — Metric Card Component
"""

import streamlit as st
from observatory.theme.tokens import safe_text, tone_class


def metric_card(label: str, value: str, tone: str = "neutral", compact: bool = False):
    size_class = "genx-metric-card compact" if compact else "genx-metric-card"
    st.markdown(
        f"""
        <div class="{size_class} {tone_class(tone)}">
            <div class="genx-metric-label">{safe_text(label)}</div>
            <div class="genx-metric-value">{safe_text(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
