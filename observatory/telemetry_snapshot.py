"""
GENX Telemetry Snapshot

Produces a live observatory snapshot from current GENX telemetry sources.
"""
from observatory.risk_monitor import build_risk_monitor
from knowledge.graph_store import GraphStore
from knowledge.graph_memory_bridge import GraphMemoryBridge

from observatory.live_state_bridge import build_live_observatory_state
from observatory.council_telemetry import build_council_telemetry
from observatory.governance_timeline import build_governance_timeline
from observatory.tool_telemetry import build_tool_telemetry
from observatory.mission_telemetry import build_mission_telemetry
from observatory.mission_control_snapshot import build_mission_control_snapshot


def load_live_telemetry_snapshot() -> dict:
    """
    Build a live telemetry snapshot for the observatory.
    """

    store = GraphStore()
    bridge = GraphMemoryBridge(store)

    mission = {
        "mission_id": "M-LIVE-001",
        "objective": "Analyze repository structure and propose safe refactor plan",
        "domain": "engineering",
        "authority_class": "B",
    }

    session = {
        "session_id": "CS-LIVE-001",
        "mission_id": "M-LIVE-001",
        "status": "DECISION_FORMED",
    }

    decision = {
        "session_id": "CS-LIVE-001",
        "mission_id": "M-LIVE-001",
        "consensus_recommendation": "request_evaluator_review",
        "confidence_score": 0.689,
    }

    tool_result = {
        "tool": "claude_code",
        "mode": "sandbox_exec",
        "status": "approved",
        "execution_type": "sandbox_stub",
        "mission_id": "M-LIVE-001",
        "session_id": "CS-LIVE-001",
    }

    bridge.record_mission(mission)
    bridge.record_session(session)
    bridge.record_decision(decision)
    bridge.record_tool_execution(tool_result)

    latest_governance = {
        "last_council_recommendation": "request_evaluator_review",
        "authority_action": "route_evaluator_review",
        "allowed_to_continue_layers": False,
        "confidence_score": 0.689,
        "aggregate_risk_score": 0.497,
        "dissenting_roles": ["safety"],
        "supporting_roles": ["analyst", "evaluator", "planner", "historian"],
    }

    latest_missions = [
        {
            "mission_id": "M-LIVE-001",
            "domain": "engineering",
            "objective": "Analyze repository structure and propose safe refactor plan",
            "decision": "request_evaluator_review",
            "tool": "claude_code",
        },
        {
            "mission_id": "M-LIVE-002",
            "domain": "engineering",
            "objective": "Review adapter reliability over recent tool runs",
            "decision": "approved",
            "tool": "claude_code",
        },
        {
            "mission_id": "M-LIVE-003",
            "domain": "governance",
            "objective": "Audit council dissent trend",
            "decision": "block",
            "tool": "claude_code",
        },
    ]

    council_telemetry = build_council_telemetry(
        session_id="CS-LIVE-001",
        recommendation="request_evaluator_review",
        authority_action="route_evaluator_review",
        supporting_roles=["analyst", "evaluator", "planner", "historian"],
        dissenting_roles=["safety"],
        confidence_score=0.689,
        aggregate_risk_score=0.497,
    )

    governance_timeline = build_governance_timeline(
        mission_id="M-LIVE-001",
        session_id="CS-LIVE-001",
        recommendation="request_evaluator_review",
        authority_action="route_evaluator_review",
        tool_id="claude_code",
        tool_mode="sandbox_exec",
    )

    state = build_live_observatory_state(
        graph_store=store,
        latest_governance=latest_governance,
        latest_tool_execution=tool_result,
        latest_missions=latest_missions,
    )

    execution_history = [
        {
            "tool": "claude_code",
            "mode": "sandbox_exec",
            "status": "approved",
            "mission_id": "M-LIVE-001",
            "session_id": "CS-LIVE-001",
        },
        {
            "tool": "claude_code",
            "mode": "sandbox_exec",
            "status": "approved",
            "mission_id": "M-LIVE-002",
            "session_id": "CS-LIVE-002",
        },
        {
            "tool": "claude_code",
            "mode": "live_exec",
            "status": "blocked",
            "mission_id": "M-LIVE-003",
            "session_id": "CS-LIVE-003",
        },
    ]


    tool_telemetry = build_tool_telemetry(execution_history)
    mission_telemetry = build_mission_telemetry(latest_missions)
    mission_control = build_mission_control_snapshot(mission_telemetry)

    risk_monitor = build_risk_monitor(
        governance=latest_governance,
        council_telemetry=council_telemetry,
        tool_telemetry=tool_telemetry,
        mission_control=mission_control,
    )

    state["council_telemetry"] = council_telemetry
    state["governance_timeline"] = governance_timeline
    state["tool_telemetry"] = tool_telemetry
    state["mission_telemetry"] = mission_telemetry
    state["mission_control"] = mission_control
    state["risk_monitor"] = risk_monitor

    return state
