"""
GENX Observatory — KV Grid Components

Self-contained key-value grid and council session card.
All HTML emitted in single st.markdown() calls.
"""

import streamlit as st
from observatory.theme.tokens import safe_text, tone_class
from observatory.components.glass_panel import info_card
from observatory.components.tag_list import tag_list_html


def _kv_card_html(key: str, value) -> str:
    """Return a single KV card as an HTML string."""
    return f"""
    <div class="genx-kv-card">
        <div class="genx-kv-label">{safe_text(key)}</div>
        <div class="genx-kv-value">{safe_text(value)}</div>
    </div>
    """


def render_kv_grid(title: str, data: dict, columns: int = 2) -> None:
    """Render a key-value grid inside a glass card. Single st.markdown call."""
    items = list((data or {}).items())
    if not items:
        info_card(title, "No data available.")
        return

    cards_html = "".join(_kv_card_html(k, v) for k, v in items)

    st.markdown(
        f"""
        <div class="genx-glass-card">
            <div class="genx-mini-title">{safe_text(title)}</div>
            <div class="genx-kv-grid" style="grid-template-columns: repeat({columns}, 1fr);">
                {cards_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_council_session_card(data: dict) -> None:
    """Render the council session card. Single st.markdown call."""
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

    kv_cards = "".join(_kv_card_html(k, v) for k, v in top_fields.items())
    support_html = tag_list_html("Supporting Roles", data.get("supporting_roles", []), tone="good")
    dissent_html = tag_list_html("Dissenting Roles", data.get("dissenting_roles", []), tone="warn")

    st.markdown(
        f"""
        <div class="genx-glass-card">
            <div class="genx-mini-title">Council Session Core</div>
            <div class="genx-kv-grid" style="grid-template-columns: repeat(2, 1fr);">
                {kv_cards}
            </div>
            {support_html}
            {dissent_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
