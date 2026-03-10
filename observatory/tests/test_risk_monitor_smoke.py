"""
GENX Risk Monitor Smoke Test
"""

from observatory.risk_monitor import build_risk_monitor


def main():
    governance = {
        "confidence_score": 0.689,
        "aggregate_risk_score": 0.497,
    }

    council_telemetry = {
        "support_count": 4,
        "dissent_count": 1,
    }

    tool_telemetry = {
        "total_executions": 3,
        "live_count": 1,
        "blocked_count": 1,
    }

    mission_control = {
        "active_count": 3,
        "review_required_count": 1,
        "blocked_like_count": 2,
    }

    risk = build_risk_monitor(
        governance=governance,
        council_telemetry=council_telemetry,
        tool_telemetry=tool_telemetry,
        mission_control=mission_control,
    )

    print("\n[GENX RISK MONITOR SMOKE TEST]")
    print(risk)


if __name__ == "__main__":
    main()
