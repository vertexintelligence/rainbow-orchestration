"""Mission Radar section renderer."""

import streamlit as st
from observatory.components.glass_panel import section_title
from observatory.components.radar_card import radar_card_html
from observatory.theme.tokens import status_tone


def render_mission_radar_section(state: dict) -> None:
    governance = state.get("governance", {})
    risk_monitor = state.get("risk_monitor", {})
    mission_control = state.get("mission_control", {})

    section_title(
        "Mission Radar",
        "Immediate command view across live mission pressure, governance routing, and operational risk.",
        tier="primary",
    )

    cards = "".join([
        radar_card_html("Active Missions", str(mission_control.get("active_count", 0)), "neutral"),
        radar_card_html("Review Required", str(mission_control.get("review_required_count", 0)), "warn"),
        radar_card_html("Blocked-like", str(mission_control.get("blocked_like_count", 0)), "bad"),
        radar_card_html(
            "Aggregate Risk",
            str(risk_monitor.get("aggregate_risk_score", governance.get("aggregate_risk_score", 0.0))),
            "warn",
        ),
        radar_card_html("Confidence", str(governance.get("confidence_score", 0.0)), "good"),
        radar_card_html("Authority Route", governance.get("authority_action", "unknown"), "neutral"),
    ])

    st.markdown(
        f"""
        <div class="genx-glass-panel">
            <div class="genx-radar-grid">{cards}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
