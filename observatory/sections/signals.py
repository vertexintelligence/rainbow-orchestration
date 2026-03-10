"""Observatory Signals section renderer."""

from observatory.components.glass_panel import section_title, section_divider
from observatory.components.signal_card import render_signal_card


def render_signals_section(state: dict) -> None:
    section_title(
        "Observatory Signals",
        "Immediate warnings and notable live system conditions.",
    )

    for alert in state.get("alerts", []):
        render_signal_card(alert)

    section_divider()
