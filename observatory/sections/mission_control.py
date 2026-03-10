"""Autonomous Mission Control section renderer."""

import streamlit as st
from observatory.components.glass_panel import section_title, section_divider
from observatory.components.metric_card import metric_card
from observatory.components.kv_grid import render_kv_grid
from observatory.components.styled_table import render_styled_table


def render_mission_control_section(state: dict) -> None:
    section_title(
        "Autonomous Mission Control",
        "Live mission distribution, decision spread, and top active control surfaces.",
    )

    mission_control = state.get("mission_control", {})
    mission_telemetry = state.get("mission_telemetry", {})

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        metric_card("Active Missions", str(mission_control.get("active_count", 0)), "neutral")
    with mc2:
        metric_card("Engineering Missions", str(mission_control.get("engineering_count", 0)), "neutral")
    with mc3:
        metric_card("Review Required", str(mission_control.get("review_required_count", 0)), "warn")
    with mc4:
        metric_card("Blocked-like", str(mission_control.get("blocked_like_count", 0)), "bad")

    mission_left, mission_right = st.columns(2)
    with mission_left:
        render_kv_grid("Mission Distribution", mission_telemetry.get("by_domain", {}), columns=1)
    with mission_right:
        render_kv_grid("Decision Distribution", mission_telemetry.get("by_decision", {}), columns=1)

    render_styled_table("Top Active Missions", mission_control.get("top_active_missions", []))

    section_divider()
