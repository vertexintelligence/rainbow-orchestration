"""Tool Telemetry section renderer."""

import streamlit as st
from observatory.components.glass_panel import section_title, section_divider
from observatory.components.metric_card import metric_card
from observatory.components.kv_grid import render_kv_grid
from observatory.components.styled_table import render_styled_table


def render_tool_telemetry_section(state: dict) -> None:
    section_title(
        "Tool Telemetry",
        "Execution reliability, sandbox/live exposure, and recent tool behavior.",
    )

    tt = state.get("tool_telemetry", {})
    tt1, tt2, tt3, tt4, tt5 = st.columns(5)
    with tt1:
        metric_card("Total Executions", str(tt.get("total_executions", 0)), "neutral", compact=True)
    with tt2:
        metric_card("Sandbox", str(tt.get("sandbox_count", 0)), "neutral", compact=True)
    with tt3:
        metric_card("Live", str(tt.get("live_count", 0)), "neutral", compact=True)
    with tt4:
        metric_card("Approved", str(tt.get("approved_count", 0)), "good", compact=True)
    with tt5:
        metric_card("Blocked", str(tt.get("blocked_count", 0)), "bad", compact=True)

    tool_left, tool_right = st.columns(2)
    with tool_left:
        render_kv_grid("Reliability Summary", tt.get("reliability", {}), columns=1)
    with tool_right:
        render_styled_table("Recent Tool Activity", tt.get("recent_activity", []))

    section_divider()
