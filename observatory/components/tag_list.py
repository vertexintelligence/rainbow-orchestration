"""
GENX Observatory — Tag List Component

Provides both st.markdown-calling and HTML-returning variants.
"""

import streamlit as st
from observatory.theme.tokens import safe_text, tone_class


def tag_list_html(title: str, items, tone: str = "neutral") -> str:
    """Return tag list as an HTML string for composition into larger blocks."""
    items = items or []
    title_html = f'<div class="genx-mini-title">{safe_text(title)}</div>'

    if items:
        chips = "".join(
            f'<span class="genx-chip {tone_class(tone)}">{safe_text(item)}</span>'
            for item in items
        )
        content = f'<div class="genx-chip-row">{chips}</div>'
    else:
        content = '<div class="genx-muted">No entries.</div>'

    return title_html + content


def render_tag_list(title: str, items, tone: str = "neutral") -> None:
    """Render tag list as a standalone glass card."""
    inner = tag_list_html(title, items, tone)
    st.markdown(
        f'<div class="genx-glass-card">{inner}</div>',
        unsafe_allow_html=True,
    )
