"""
GENX Observatory — Styled Table Component

Renders list-of-dicts as a dark-themed HTML table inside a glass card.
Detects known status values and applies tone coloring.
"""

import streamlit as st
from observatory.theme.tokens import safe_text
from observatory.components.glass_panel import info_card

_TONE_VALUES = {
    "approved": "good",
    "operational": "good",
    "active": "good",
    "good": "good",
    "blocked": "bad",
    "block": "bad",
    "critical": "bad",
    "sandbox_exec": "neutral",
    "live_exec": "warn",
    "request_evaluator_review": "warn",
    "guarded": "warn",
    "review": "warn",
    "warning": "warn",
}


def _cell_html(value: str) -> str:
    """Render a table cell with optional tone coloring."""
    tone = _TONE_VALUES.get(str(value).lower().strip(), "")
    tone_cls = f" genx-table-td--{tone}" if tone else ""
    return f'<td class="genx-table-td{tone_cls}">{safe_text(value)}</td>'


def render_styled_table(title: str, rows: list) -> None:
    if not rows:
        info_card(title, "No data available.")
        return

    headers = list(rows[0].keys())

    header_html = "".join(
        f'<th class="genx-table-th">{safe_text(h)}</th>' for h in headers
    )

    body_html = ""
    for row in rows:
        cells = "".join(_cell_html(row.get(h, "")) for h in headers)
        body_html += f"<tr>{cells}</tr>"

    st.markdown(
        f"""
        <div class="genx-glass-card">
            <div class="genx-mini-title">{safe_text(title)}</div>
            <table class="genx-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{body_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
