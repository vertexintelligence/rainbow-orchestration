"""Mission Intelligence section renderer."""

import streamlit as st
from observatory.components.glass_panel import section_title
from observatory.components.styled_table import render_styled_table


def render_mission_intelligence_section(state: dict) -> None:
    section_title(
        "Mission Intelligence",
        "Operational mission ledger across domains, objectives, and decision outcomes.",
    )

    render_styled_table("Mission Ledger", state.get("missions", []))

    st.markdown("---")
