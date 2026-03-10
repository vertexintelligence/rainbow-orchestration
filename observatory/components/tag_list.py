"""
GENX Observatory — Tag List Component
"""

import streamlit as st
from observatory.theme.tokens import safe_text, tone_class


def render_tag_list(title: str, items, tone: str = "neutral", inline: bool = False) -> None:
    items = items or []

    if not inline:
        st.markdown('<div class="genx-glass-card">', unsafe_allow_html=True)

    st.markdown(f'<div class="genx-mini-title">{safe_text(title)}</div>', unsafe_allow_html=True)

    if items:
        chips = "".join(
            f'<span class="genx-chip {tone_class(tone)}">{safe_text(item)}</span>'
            for item in items
        )
        st.markdown(f'<div class="genx-chip-row">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="genx-muted">No entries.</div>', unsafe_allow_html=True)

    if not inline:
        st.markdown("</div>", unsafe_allow_html=True)
