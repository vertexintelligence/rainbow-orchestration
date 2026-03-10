"""Mission Intelligence section renderer."""

from observatory.components.glass_panel import section_title, section_divider
from observatory.components.styled_table import render_styled_table


def render_mission_intelligence_section(state: dict) -> None:
    section_title(
        "Mission Intelligence",
        "Operational mission ledger across domains, objectives, and decision outcomes.",
    )

    render_styled_table("Mission Ledger", state.get("missions", []))

    section_divider()
