"""
GENX Observatory — Styled Table Component

Renders list-of-dicts as a dark-themed HTML table inside a glass card.
"""

import streamlit as st
from observatory.theme.tokens import safe_text
from observatory.components.glass_panel import info_card


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
        cells = "".join(
            f'<td class="genx-table-td">{safe_text(row.get(h, ""))}</td>'
            for h in headers
        )
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
