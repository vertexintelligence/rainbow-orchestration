"""
GENX Tool Telemetry

Provides tool-focused telemetry for the observatory.
"""


def build_tool_telemetry(executions: list) -> dict:
    total_executions = len(executions)
    sandbox_count = 0
    live_count = 0
    approved_count = 0
    blocked_count = 0
    by_tool = {}
    recent_activity = []

    for item in executions:
        tool_id = item.get("tool", "unknown")
        mode = item.get("mode", "unknown")
        status = item.get("status", "unknown")

        if mode == "sandbox_exec":
            sandbox_count += 1
        elif mode == "live_exec":
            live_count += 1

        if status == "approved":
            approved_count += 1
        elif status == "blocked":
            blocked_count += 1

        by_tool[tool_id] = by_tool.get(tool_id, 0) + 1

        recent_activity.append({
            "tool": tool_id,
            "mode": mode,
            "status": status,
            "mission_id": item.get("mission_id", ""),
            "session_id": item.get("session_id", ""),
        })

    reliability = {}
    for tool_id, count in by_tool.items():
        tool_approved = sum(
            1 for item in executions
            if item.get("tool") == tool_id and item.get("status") == "approved"
        )
        reliability[tool_id] = {
            "total": count,
            "approved": tool_approved,
            "approval_rate": round(tool_approved / count, 3) if count else 0.0,
        }

    return {
        "total_executions": total_executions,
        "sandbox_count": sandbox_count,
        "live_count": live_count,
        "approved_count": approved_count,
        "blocked_count": blocked_count,
        "by_tool": by_tool,
        "reliability": reliability,
        "recent_activity": recent_activity[-10:],
    }
