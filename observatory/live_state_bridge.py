"""
GENX Live State Bridge

Builds observatory state from live GENX telemetry inputs.
"""

from knowledge.graph_query import summarize_graph
from knowledge.strategic_queries import (
    missions_requiring_evaluator_review,
    tool_usage_summary,
)


def build_live_observatory_state(
    graph_store,
    latest_governance=None,
    latest_tool_execution=None,
    latest_missions=None,
) -> dict:
    """
    Build the live observatory state payload.
    """

    latest_governance = latest_governance or {}
    latest_tool_execution = latest_tool_execution or {}
    latest_missions = latest_missions or []

    graph_summary = summarize_graph(graph_store)
    tool_usage = tool_usage_summary(graph_store)
    evaluator_review_missions = missions_requiring_evaluator_review(graph_store)

    alerts = []

    if latest_governance.get("authority_action"):
        alerts.append(
            f"Council routed latest mission to {latest_governance['authority_action']}."
        )

    if latest_governance.get("dissenting_roles"):
        alerts.append(
            f"Dissent active from: {', '.join(latest_governance['dissenting_roles'])}."
        )

    if latest_tool_execution.get("tool") and latest_tool_execution.get("mode"):
        alerts.append(
            f"{latest_tool_execution['tool']} executed in mode {latest_tool_execution['mode']}."
        )

    if evaluator_review_missions:
        alerts.append(
            f"Missions requiring evaluator review: {len(evaluator_review_missions)}."
        )

    return {
        "system": {
            "name": "GENX Observatory",
            "environment": "RAINBOW",
            "status": "operational",
            "phase": "live_telemetry_active",
        },
        "governance": {
            "last_council_recommendation": latest_governance.get(
                "last_council_recommendation", "unknown"
            ),
            "authority_action": latest_governance.get(
                "authority_action", "unknown"
            ),
            "allowed_to_continue_layers": latest_governance.get(
                "allowed_to_continue_layers", False
            ),
            "confidence_score": latest_governance.get(
                "confidence_score", 0.0
            ),
            "aggregate_risk_score": latest_governance.get(
                "aggregate_risk_score", 0.0
            ),
            "dissenting_roles": latest_governance.get(
                "dissenting_roles", []
            ),
            "supporting_roles": latest_governance.get(
                "supporting_roles", []
            ),
        },
        "tools": {
            "registered_tools": len(tool_usage),
            "tool_usage_summary": tool_usage,
            "latest_tool_execution": latest_tool_execution,
        },
        "knowledge_graph": graph_summary,
        "missions": latest_missions,
        "alerts": alerts,
    }
