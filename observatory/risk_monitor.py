"""
GENX Risk Monitor

Builds live risk intelligence from governance, tool telemetry,
and mission telemetry.
"""


def build_risk_monitor(
    governance: dict,
    council_telemetry: dict,
    tool_telemetry: dict,
    mission_control: dict,
) -> dict:
    confidence_score = governance.get("confidence_score", 0.0)
    aggregate_risk_score = governance.get("aggregate_risk_score", 0.0)

    dissent_count = council_telemetry.get("dissent_count", 0)
    support_count = council_telemetry.get("support_count", 0)

    blocked_count = tool_telemetry.get("blocked_count", 0)
    live_count = tool_telemetry.get("live_count", 0)
    total_executions = tool_telemetry.get("total_executions", 0)

    review_required_count = mission_control.get("review_required_count", 0)
    blocked_like_count = mission_control.get("blocked_like_count", 0)
    active_count = mission_control.get("active_count", 0)

    dissent_ratio = round(
        dissent_count / (support_count + dissent_count), 3
    ) if (support_count + dissent_count) > 0 else 0.0

    blocked_ratio = round(
        blocked_count / total_executions, 3
    ) if total_executions > 0 else 0.0

    live_exposure_ratio = round(
        live_count / total_executions, 3
    ) if total_executions > 0 else 0.0

    mission_review_ratio = round(
        review_required_count / active_count, 3
    ) if active_count > 0 else 0.0

    mission_block_ratio = round(
        blocked_like_count / active_count, 3
    ) if active_count > 0 else 0.0

    governance_escalation_score = round(
        (aggregate_risk_score * 0.35)
        + (dissent_ratio * 0.20)
        + (blocked_ratio * 0.20)
        + (mission_block_ratio * 0.15)
        + ((1.0 - confidence_score) * 0.10),
        3,
    )

    if governance_escalation_score >= 0.75:
        live_risk_band = "critical"
    elif governance_escalation_score >= 0.50:
        live_risk_band = "elevated"
    elif governance_escalation_score >= 0.25:
        live_risk_band = "guarded"
    else:
        live_risk_band = "stable"

    if dissent_count >= 2:
        dissent_severity = "high"
    elif dissent_count == 1:
        dissent_severity = "moderate"
    else:
        dissent_severity = "low"

    return {
        "live_risk_band": live_risk_band,
        "governance_escalation_score": governance_escalation_score,
        "dissent_severity": dissent_severity,
        "dissent_ratio": dissent_ratio,
        "tool_exposure_score": live_exposure_ratio,
        "blocked_execution_ratio": blocked_ratio,
        "mission_risk_index": mission_block_ratio,
        "mission_review_ratio": mission_review_ratio,
        "aggregate_risk_score": aggregate_risk_score,
        "confidence_score": confidence_score,
    }
