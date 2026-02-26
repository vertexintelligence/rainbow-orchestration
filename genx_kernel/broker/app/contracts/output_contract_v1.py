from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

CONTRACT_NAME = "GENX_OUTPUT_CONTRACT_V1"
CONTRACT_VERSION = "1.0.0"
DEFAULT_TAGS = ["genx", "brokered", "safe"]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def normalize_tags(tags: Optional[List[str]]) -> List[str]:
    """Return exactly 3 tags by padding/truncating in a deterministic way."""
    cleaned = _as_string_list(tags)
    if not cleaned:
        cleaned = list(DEFAULT_TAGS)

    i = 0
    while len(cleaned) < 3:
        cleaned.append(DEFAULT_TAGS[i % len(DEFAULT_TAGS)])
        i += 1

    return cleaned[:3]


def _extract_structured(raw_output: str) -> Dict[str, Any]:
    """Best-effort extraction from model output. Falls back to plain text coercion."""
    text = (raw_output or "").strip()
    if not text:
        return {}

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    lines = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
    title = lines[0][:120] if lines else "Assistant Response"
    summary = text[:300]
    items = [{"text": ln} for ln in lines[1:6]]
    return {
        "title": title,
        "summary": summary,
        "items": items,
    }


def render_elite_markdown(payload: Dict[str, Any]) -> str:
    """Render the elite markdown string returned in assistant.content."""
    title = payload.get("title") or "Assistant Response"
    summary = payload.get("summary") or ""
    items = payload.get("items") or []
    actions = payload.get("actions") or []
    artifacts = payload.get("artifacts") or []

    parts = [f"## {title}"]
    if summary:
        parts.append(summary)

    if items:
        parts.append("### Key Points")
        for item in items:
            text = (item.get("text") if isinstance(item, dict) else str(item)).strip()
            if text:
                parts.append(f"- {text}")

    if actions:
        parts.append("### Actions")
        for action in actions:
            text = (action.get("text") if isinstance(action, dict) else str(action)).strip()
            if text:
                parts.append(f"- {text}")

    if artifacts:
        parts.append("### Artifacts")
        for artifact in artifacts:
            text = (artifact.get("text") if isinstance(artifact, dict) else str(artifact)).strip()
            if text:
                parts.append(f"- {text}")

    return "\n\n".join(parts).strip()


def coerce_output_contract_v1(
    *,
    raw_output: str,
    request_id: str,
    actor: str,
    thread_id: str,
    model: str,
    tags: Optional[List[str]] = None,
    route: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Coerce any model output into GENX_OUTPUT_CONTRACT_V1."""
    structured = _extract_structured(raw_output)

    payload = {
        "kind": str(structured.get("kind") or "assistant_reply"),
        "title": str(structured.get("title") or "Assistant Response"),
        "summary": str(structured.get("summary") or (raw_output or "").strip()[:300]),
        "items": structured.get("items") if isinstance(structured.get("items"), list) else [],
        "actions": structured.get("actions") if isinstance(structured.get("actions"), list) else [],
        "artifacts": structured.get("artifacts") if isinstance(structured.get("artifacts"), list) else [],
        "signals": {
            "tags": normalize_tags(tags if tags is not None else structured.get("tags")),
            "warnings": _as_string_list(structured.get("warnings")),
        },
    }

    render = {"md": render_elite_markdown(payload)}

    route_obj = {
        "decision": "allow",
        "policy_statuses": {
            "firewall": "not_evaluated",
            "policy": "not_evaluated",
        },
        "confidence": 0.5,
        "fallback_used": False,
    }
    if isinstance(route, dict):
        route_obj.update(route)

    return {
        "meta": {
            "contract": CONTRACT_NAME,
            "version": CONTRACT_VERSION,
            "request_id": request_id,
            "timestamp": _now_iso(),
            "actor": actor,
            "thread_id": thread_id,
            "privacy": "local_broker",
            "model": {
                "provider": "ollama",
                "name": model,
            },
        },
        "route": route_obj,
        "payload": payload,
        "render": render,
    }
