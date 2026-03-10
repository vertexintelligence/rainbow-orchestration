"""
GENX Tool Telemetry Smoke Test
"""

from observatory.tool_telemetry import build_tool_telemetry


def main():
    executions = [
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

    telemetry = build_tool_telemetry(executions)

    print("\n[GENX TOOL TELEMETRY SMOKE TEST]")
    print(telemetry)


if __name__ == "__main__":
    main()
