"""
GENX Observatory — Radar Card Component

Returns HTML string for composition into radar grids.
"""

from observatory.theme.tokens import safe_text, tone_class


def radar_card_html(label: str, value: str, tone: str = "neutral") -> str:
    return f"""
    <div class="genx-radar-card {tone_class(tone)}">
        <div class="genx-radar-label">{safe_text(label)}</div>
        <div class="genx-radar-value">{safe_text(value)}</div>
    </div>
    """
