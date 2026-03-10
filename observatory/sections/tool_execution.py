"""Tool Execution Panel section renderer."""

import streamlit as st
from observatory.components.glass_panel import section_title, section_divider
from observatory.components.kv_grid import render_kv_grid


def render_tool_execution_section(state: dict) -> None:
    section_title(
        "Tool Execution Panel",
        "Approved tool state, latest execution outcome, and operational execution context.",
    )

    tool_left, tool_right = st.columns(2)
    with tool_left:
        render_kv_grid("Tool Usage Summary", state.get("tools", {}).get("tool_usage_summary", {}), columns=1)
    with tool_right:
        render_kv_grid("Latest Tool Execution", state.get("tools", {}).get("latest_tool_execution", {}), columns=2)

    section_divider()
