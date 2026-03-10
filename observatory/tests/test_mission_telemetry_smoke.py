"""
GENX Mission Telemetry Smoke Test
"""

from observatory.mission_telemetry import build_mission_telemetry
from observatory.mission_control_snapshot import build_mission_control_snapshot


def main():
    missions = [
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

    mission_telemetry = build_mission_telemetry(missions)
    mission_control = build_mission_control_snapshot(mission_telemetry)

    print("\n[GENX MISSION TELEMETRY SMOKE TEST]")
    print(mission_telemetry)
    print(mission_control)


if __name__ == "__main__":
    main()
