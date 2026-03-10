"""System Status section renderer."""

import streamlit as st
from observatory.components.glass_panel import section_title
from observatory.components.metric_card import metric_card
from observatory.theme.tokens import status_tone


def render_system_status_section(state: dict) -> None:
    system = state.get("system", {})

    section_title(
        "System Status",
        "Foundational operational posture of the GENX civilization stack.",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Environment", system.get("environment", "UNKNOWN"), "neutral")
    with col2:
        metric_card("System Status", system.get("status", "unknown"), status_tone(system.get("status", "unknown")))
    with col3:
        metric_card("Phase", system.get("phase", "unknown"), "neutral")
    with col4:
        metric_card("Registered Tools", str(state.get("tools", {}).get("registered_tools", 0)), "neutral")

    st.markdown("---")
