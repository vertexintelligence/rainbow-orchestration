"""
GENX Observatory — KV Grid Components

Key-value grid renderer and council session card.
"""

import streamlit as st
from observatory.theme.tokens import safe_text, tone_class
from observatory.components.glass_panel import info_card
from observatory.components.tag_list import render_tag_list


def render_kv_grid(title: str, data: dict, columns: int = 2) -> None:
    items = list((data or {}).items())
    if not items:
        info_card(title, "No data available.")
        return

    rows = [items[i:i + columns] for i in range(0, len(items), columns)]

    st.markdown('<div class="genx-glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="genx-mini-title">{safe_text(title)}</div>', unsafe_allow_html=True)

    for row in rows:
        cols = st.columns(columns)
        for idx in range(columns):
            with cols[idx]:
                if idx < len(row):
                    k, v = row[idx]
                    st.markdown(
                        f"""
                        <div class="genx-kv-card">
                            <div class="genx-kv-label">{safe_text(k)}</div>
                            <div class="genx-kv-value">{safe_text(v)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_kv_grid_inline(data: dict, columns: int = 2) -> None:
    items = list((data or {}).items())
    rows = [items[i:i + columns] for i in range(0, len(items), columns)]

    for row in rows:
        cols = st.columns(columns)
        for idx in range(columns):
            with cols[idx]:
                if idx < len(row):
                    k, v = row[idx]
                    st.markdown(
                        f"""
                        <div class="genx-kv-card">
                            <div class="genx-kv-label">{safe_text(k)}</div>
                            <div class="genx-kv-value">{safe_text(v)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def render_council_session_card(data: dict) -> None:
    if not data:
        info_card("Council Session", "No council telemetry available.")
        return

    top_fields = {
        "session_id": data.get("session_id", "unknown"),
        "recommendation": data.get("recommendation", "unknown"),
        "authority_action": data.get("authority_action", "unknown"),
        "support_count": data.get("support_count", 0),
        "dissent_count": data.get("dissent_count", 0),
        "confidence_score": data.get("confidence_score", 0.0),
        "aggregate_risk_score": data.get("aggregate_risk_score", 0.0),
        "alignment_status": data.get("alignment_status", "unknown"),
    }

    st.markdown('<div class="genx-glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="genx-mini-title">Council Session Core</div>', unsafe_allow_html=True)

    render_kv_grid_inline(top_fields, columns=2)
    render_tag_list("Supporting Roles", data.get("supporting_roles", []), tone="good", inline=True)
    render_tag_list("Dissenting Roles", data.get("dissenting_roles", []), tone="warn", inline=True)

    st.markdown("</div>", unsafe_allow_html=True)
