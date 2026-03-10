"""
GENX Governance Timeline

Creates a simple governance event timeline for observatory display.
"""


def build_governance_timeline(
    mission_id: str,
    session_id: str,
    recommendation: str,
    authority_action: str,
    tool_id: str,
    tool_mode: str,
) -> list:
    return [
        {
            "step": 1,
            "event": "mission_registered",
            "detail": f"Mission {mission_id} registered",
        },
        {
            "step": 2,
            "event": "council_session_started",
            "detail": f"Council session {session_id} started",
        },
        {
            "step": 3,
            "event": "recommendation_issued",
            "detail": f"Council recommended {recommendation}",
        },
        {
            "step": 4,
            "event": "authority_routed",
            "detail": f"Authority action: {authority_action}",
        },
        {
            "step": 5,
            "event": "tool_execution_recorded",
            "detail": f"Tool {tool_id} executed in mode {tool_mode}",
        },
    ]
