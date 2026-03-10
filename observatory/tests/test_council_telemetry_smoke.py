"""
GENX Council Telemetry Smoke Test
"""

from observatory.council_telemetry import build_council_telemetry
from observatory.governance_timeline import build_governance_timeline


def main():
    council = build_council_telemetry(
        session_id="CS-LIVE-001",
        recommendation="request_evaluator_review",
        authority_action="route_evaluator_review",
        supporting_roles=["analyst", "evaluator", "planner", "historian"],
        dissenting_roles=["safety"],
        confidence_score=0.689,
        aggregate_risk_score=0.497,
    )

    timeline = build_governance_timeline(
        mission_id="M-LIVE-001",
        session_id="CS-LIVE-001",
        recommendation="request_evaluator_review",
        authority_action="route_evaluator_review",
        tool_id="claude_code",
        tool_mode="sandbox_exec",
    )

    print("\n[GENX COUNCIL TELEMETRY SMOKE TEST]")
    print(council)
    print(timeline)


if __name__ == "__main__":
    main()
