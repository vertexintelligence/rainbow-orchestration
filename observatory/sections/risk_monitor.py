"""Risk Monitor section renderer."""

import streamlit as st
from observatory.components.glass_panel import section_title
from observatory.components.metric_card import metric_card
from observatory.components.kv_grid import render_kv_grid
from observatory.theme.tokens import status_tone


def render_risk_monitor_section(state: dict) -> None:
    section_title(
        "Risk Monitor",
        "Escalation posture, dissent pressure, execution exposure, and live governance risk intelligence.",
    )

    risk_monitor = state.get("risk_monitor", {})

    rk1, rk2, rk3, rk4, rk5 = st.columns(5)
    with rk1:
        metric_card("Risk Band", str(risk_monitor.get("live_risk_band", "unknown")), status_tone(risk_monitor.get("live_risk_band", "unknown")))
    with rk2:
        metric_card("Escalation Score", str(risk_monitor.get("governance_escalation_score", 0.0)), "warn")
    with rk3:
        metric_card("Dissent Severity", str(risk_monitor.get("dissent_severity", "unknown")), "warn")
    with rk4:
        metric_card("Tool Exposure", str(risk_monitor.get("tool_exposure_score", 0.0)), "neutral")
    with rk5:
        metric_card("Mission Risk", str(risk_monitor.get("mission_risk_index", 0.0)), "neutral")

    risk_left, risk_right = st.columns(2)
    with risk_left:
        render_kv_grid(
            "Risk Ratios",
            {
                "dissent_ratio": risk_monitor.get("dissent_ratio", 0.0),
                "blocked_execution_ratio": risk_monitor.get("blocked_execution_ratio", 0.0),
                "mission_review_ratio": risk_monitor.get("mission_review_ratio", 0.0),
            },
            columns=1,
        )
    with risk_right:
        render_kv_grid(
            "Risk Core",
            {
                "aggregate_risk_score": risk_monitor.get("aggregate_risk_score", 0.0),
                "confidence_score": risk_monitor.get("confidence_score", 0.0),
            },
            columns=1,
        )

    st.markdown("---")
