"""Council Telemetry section renderer."""

import streamlit as st
from observatory.components.glass_panel import section_title
from observatory.components.kv_grid import render_council_session_card
from observatory.components.styled_table import render_styled_table


def render_council_telemetry_section(state: dict) -> None:
    section_title(
        "Council Telemetry",
        "Live council alignment, dissent posture, and governance event flow.",
    )

    ct_left, ct_right = st.columns(2)
    with ct_left:
        render_council_session_card(state.get("council_telemetry", {}))
    with ct_right:
        render_styled_table("Governance Timeline", state.get("governance_timeline", []))

    st.markdown("---")
