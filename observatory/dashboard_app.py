"""
GENX Observatory Dashboard App

Run with:
    streamlit run ~/genx/observatory/dashboard_app.py
"""

import os
import sys

# ============================================================
# Ensure GENX root is on the Python path
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GENX_ROOT = os.path.dirname(CURRENT_DIR)

if GENX_ROOT not in sys.path:
    sys.path.insert(0, GENX_ROOT)

import streamlit as st

from observatory.state.loader import load_dashboard_state
from observatory.theme.css import inject_stylesheet
from observatory.sections.command_rail import render_command_rail_section
from observatory.sections.mission_radar import render_mission_radar_section
from observatory.sections.system_status import render_system_status_section
from observatory.sections.council_telemetry import render_council_telemetry_section
from observatory.sections.tool_execution import render_tool_execution_section
from observatory.sections.tool_telemetry import render_tool_telemetry_section
from observatory.sections.knowledge_graph import render_knowledge_graph_section
from observatory.sections.mission_intelligence import render_mission_intelligence_section
from observatory.sections.mission_control import render_mission_control_section
from observatory.sections.risk_monitor import render_risk_monitor_section
from observatory.sections.signals import render_signals_section
from observatory.sections.raw_state import render_raw_state_section

# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="GENX Observatory",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# Load state + inject theme
# ============================================================

state = load_dashboard_state()
inject_stylesheet()

# ============================================================
# Render sections in order
# ============================================================

render_command_rail_section(state)
render_mission_radar_section(state)
render_system_status_section(state)
render_council_telemetry_section(state)
render_tool_execution_section(state)
render_tool_telemetry_section(state)
render_knowledge_graph_section(state)
render_mission_intelligence_section(state)
render_mission_control_section(state)
render_risk_monitor_section(state)
render_signals_section(state)
render_raw_state_section(state)
