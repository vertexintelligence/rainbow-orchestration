"""
GENX Observatory — Command Rail Component
"""

import textwrap
import streamlit as st
from observatory.theme.tokens import safe_text, tone_class, status_tone


def rail_item(label: str, value: str, tone: str = "neutral") -> str:
    return f"""
<div class="genx-rail-item {tone_class(tone)}">
    <div class="genx-rail-label">{safe_text(label)}</div>
    <div class="genx-rail-value">{safe_text(value)}</div>
</div>
""".strip()


def render_command_rail(state: dict) -> None:
    system = state.get("system", {})
    governance = state.get("governance", {})
    risk_monitor = state.get("risk_monitor", {})
    mission_control = state.get("mission_control", {})

    rail_html = textwrap.dedent(f"""
    <div class="genx-rail">
        <div class="genx-rail-title">Command Rail</div>
        <div class="genx-rail-subtitle">
            Command spine for mission, governance, and risk posture.
        </div>

        <div class="genx-rail-stack">
            {rail_item("Environment", system.get("environment", "UNKNOWN"), "neutral")}
            {rail_item("Status", system.get("status", "unknown"), status_tone(system.get("status", "unknown")))}
            {rail_item("Council Route", governance.get("authority_action", "unknown"), "neutral")}
            {rail_item("Risk Band", risk_monitor.get("live_risk_band", "unknown"), status_tone(risk_monitor.get("live_risk_band", "unknown")))}
            {rail_item("Active Missions", str(mission_control.get("active_count", 0)), "neutral")}
            {rail_item("Signals", str(len(state.get("alerts", []))), "warn")}
        </div>
    </div>
    """).strip()

    st.markdown(rail_html, unsafe_allow_html=True)
