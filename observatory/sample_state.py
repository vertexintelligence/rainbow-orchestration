"""
GENX Observatory Sample State

Provides initial structured data for the RAINBOW Intelligence Dashboard.
Replace later with live orchestrator / graph / tool feeds.
"""


def load_observatory_state() -> dict:
    return {
        "system": {
            "name": "GENX Observatory",
            "environment": "RAINBOW",
            "status": "operational",
            "phase": "civilization_memory_active",
        },
        "governance": {
            "last_council_recommendation": "request_evaluator_review",
            "authority_action": "route_evaluator_review",
            "allowed_to_continue_layers": False,
            "confidence_score": 0.689,
            "aggregate_risk_score": 0.497,
            "dissenting_roles": ["safety"],
            "supporting_roles": ["analyst", "evaluator", "planner", "historian"],
        },
        "tools": {
            "registered_tools": 1,
            "tool_usage_summary": {
                "claude_code": 1,
            },
            "latest_tool_execution": {
                "tool": "claude_code",
                "mode": "sandbox_exec",
                "status": "approved",
                "execution_type": "sandbox_stub",
            },
        },
        "knowledge_graph": {
            "node_count": 4,
            "edge_count": 3,
            "mission_count": 1,
            "session_count": 1,
            "decision_count": 1,
            "tool_execution_count": 1,
        },
        "missions": [
            {
                "mission_id": "M-TEST-200",
                "domain": "engineering",
                "objective": "Analyze repository structure and propose safe refactor plan",
                "decision": "request_evaluator_review",
                "tool": "claude_code",
            }
        ],
        "alerts": [
            "Council routed latest mission to evaluator_review.",
            "Safety dissent remains active on promotion path.",
            "Claude Code is currently sandbox-only.",
        ],
    }
