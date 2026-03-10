"""Knowledge Graph Intelligence section renderer."""

import streamlit as st
from observatory.components.glass_panel import section_title
from observatory.components.metric_card import metric_card


def render_knowledge_graph_section(state: dict) -> None:
    section_title(
        "Knowledge Graph Intelligence",
        "Structural memory and linked operational intelligence across missions, sessions, and decisions.",
    )

    kg = state.get("knowledge_graph", {})
    kg1, kg2, kg3, kg4, kg5, kg6 = st.columns(6)
    with kg1:
        metric_card("Nodes", str(kg.get("node_count", 0)), "neutral", compact=True)
    with kg2:
        metric_card("Edges", str(kg.get("edge_count", 0)), "neutral", compact=True)
    with kg3:
        metric_card("Missions", str(kg.get("mission_count", 0)), "neutral", compact=True)
    with kg4:
        metric_card("Sessions", str(kg.get("session_count", 0)), "neutral", compact=True)
    with kg5:
        metric_card("Decisions", str(kg.get("decision_count", 0)), "neutral", compact=True)
    with kg6:
        metric_card("Tool Executions", str(kg.get("tool_execution_count", 0)), "neutral", compact=True)

    st.markdown("---")
