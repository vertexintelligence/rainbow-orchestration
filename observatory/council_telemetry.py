"""
GENX Council Telemetry

Provides council-focused telemetry for the observatory.
"""


def build_council_telemetry(
    session_id: str,
    recommendation: str,
    authority_action: str,
    supporting_roles: list,
    dissenting_roles: list,
    confidence_score: float,
    aggregate_risk_score: float,
) -> dict:
    return {
        "session_id": session_id,
        "recommendation": recommendation,
        "authority_action": authority_action,
        "supporting_roles": supporting_roles,
        "dissenting_roles": dissenting_roles,
        "support_count": len(supporting_roles),
        "dissent_count": len(dissenting_roles),
        "confidence_score": confidence_score,
        "aggregate_risk_score": aggregate_risk_score,
        "alignment_status": (
            "contested" if dissenting_roles else "aligned"
        ),
    }
