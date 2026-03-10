"""
GENX Mission Telemetry

Provides mission-focused telemetry for the observatory.
"""


def build_mission_telemetry(missions: list) -> dict:
    total_missions = len(missions)
    by_domain = {}
    by_decision = {}
    by_tool = {}
    active_missions = []
    mission_activity = []

    for mission in missions:
        domain = mission.get("domain", "unknown")
        decision = mission.get("decision", "unknown")
        tool = mission.get("tool", "unknown")
        mission_id = mission.get("mission_id", "unknown")

        by_domain[domain] = by_domain.get(domain, 0) + 1
        by_decision[decision] = by_decision.get(decision, 0) + 1
        by_tool[tool] = by_tool.get(tool, 0) + 1

        active_missions.append({
            "mission_id": mission_id,
            "domain": domain,
            "decision": decision,
            "tool": tool,
        })

        mission_activity.append({
            "mission_id": mission_id,
            "objective": mission.get("objective", ""),
            "domain": domain,
            "decision": decision,
            "tool": tool,
        })

    return {
        "total_missions": total_missions,
        "by_domain": by_domain,
        "by_decision": by_decision,
        "by_tool": by_tool,
        "active_missions": active_missions,
        "mission_activity": mission_activity[-10:],
    }
