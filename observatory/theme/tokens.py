"""
GENX Observatory Theme Tokens

Pure functions for text safety, tone mapping, and CSS class resolution.
No Streamlit dependency.
"""

import html


def safe_text(value) -> str:
    if value is None:
        return "unknown"
    return html.escape(str(value))


def status_tone(value: str) -> str:
    if value is None:
        return "neutral"
    v = str(value).lower()

    if v in {"operational", "approved", "active", "good"}:
        return "good"

    if v in {"guarded", "review", "request_evaluator_review", "warn", "warning"}:
        return "warn"

    if v in {"blocked", "block", "critical", "bad"}:
        return "bad"

    return "neutral"


def tone_class(name: str) -> str:
    mapping = {
        "good": "genx-tone-good",
        "warn": "genx-tone-warn",
        "bad": "genx-tone-bad",
        "neutral": "genx-tone-neutral",
    }
    return mapping.get(name, "genx-tone-neutral")
