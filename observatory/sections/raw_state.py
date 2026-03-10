"""Raw Observatory State section renderer."""

import streamlit as st
from observatory.components.glass_panel import section_title


def render_raw_state_section(state: dict) -> None:
    section_title(
        "Raw Observatory State",
        "Canonical state payload for direct inspection and debugging.",
    )

    with st.expander("Raw Observatory State"):
        st.json(state)
