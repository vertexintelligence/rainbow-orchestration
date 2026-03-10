"""
GENX Observatory — Radar Card Component
"""

import streamlit as st
from observatory.theme.tokens import safe_text, tone_class


def radar_card(label: str, value: str, tone: str = "neutral"):
    st.markdown(
        f"""
        <div class="genx-radar-card {tone_class(tone)}">
            <div class="genx-radar-label">{safe_text(label)}</div>
            <div class="genx-radar-value">{safe_text(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
