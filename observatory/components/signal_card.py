"""
GENX Observatory — Signal Card Component
"""

import streamlit as st
from observatory.theme.tokens import safe_text


def render_signal_card(message: str) -> None:
    st.markdown(
        f"""
        <div class="genx-signal-card">
            {safe_text(message)}
        </div>
        """,
        unsafe_allow_html=True,
    )
