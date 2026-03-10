"""
GENX Mission Control Snapshot

Builds an active mission-control summary for the observatory.
"""


def build_mission_control_snapshot(mission_telemetry: dict) -> dict:
    active_missions = mission_telemetry.get("active_missions", [])
    by_decision = mission_telemetry.get("by_decision", {})

    blocked_like = (
        by_decision.get("block", 0)
        + by_decision.get("request_evaluator_review", 0)
    )

    return {
        "active_count": len(active_missions),
        "engineering_count": mission_telemetry.get("by_domain", {}).get("engineering", 0),
        "review_required_count": by_decision.get("request_evaluator_review", 0),
        "blocked_like_count": blocked_like,
        "top_active_missions": active_missions[:5],
    }
