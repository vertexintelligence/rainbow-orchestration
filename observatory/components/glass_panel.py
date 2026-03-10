"""
GENX Observatory — Glass Panel Components

Section titles, info cards, pills, and section dividers.
"""

import streamlit as st
from observatory.theme.tokens import safe_text, tone_class


def section_title(title: str, subtitle: str = "", tier: str = "secondary") -> None:
    wrap_class = "genx-section-wrap"
    if tier == "primary":
        wrap_class += " genx-section-wrap--primary"

    st.markdown(
        f"""
        <div class="{wrap_class}">
            <div class="genx-section-title">{safe_text(title)}</div>
            <div class="genx-section-caption">{safe_text(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_divider() -> None:
    st.markdown('<div class="genx-section-divider"></div>', unsafe_allow_html=True)


def pill(label: str, value: str, tone: str = "neutral") -> str:
    return f"""
    <div class="genx-pill {tone_class(tone)}">
        <span class="genx-pill-label">{safe_text(label)}</span>
        <span class="genx-pill-value">{safe_text(value)}</span>
    </div>
    """


def info_card(title: str, body: str, tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="genx-info-card {tone_class(tone)}">
            <div class="genx-info-card-title">{safe_text(title)}</div>
            <div class="genx-info-card-body">{safe_text(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
