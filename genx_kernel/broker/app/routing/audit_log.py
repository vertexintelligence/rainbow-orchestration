from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


AUDIT_EVENT_VERSION = "AUDIT_EVENT_V1"
DEFAULT_AUDIT_LOG_PATH = "runtime/audit/events.jsonl"
AUDIT_LOG_ENV = "GENX_AUDIT_LOG_PATH"


@dataclass(frozen=True)
class AuditEvent:
    event_version: str
    ts_unix_ms: int

    policy_version: str
    actor: str
    endpoint: str
    requested_model: str
    chosen_model: str
    reason: str
    escalation: bool
    tools_disabled: bool
    decision_hash: str

    # Optional/extended metadata (kept stable + nullable)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    message_count: Optional[int] = None
    has_system_prompt: Optional[bool] = None
    risk_score: Optional[float] = None
    task_type: Optional[str] = None


def _now_unix_ms() -> int:
    return int(time.time() * 1000)


def audit_log_path() -> str:
    return os.environ.get(AUDIT_LOG_ENV, DEFAULT_AUDIT_LOG_PATH)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)


def _to_compact_json(obj: Dict[str, Any]) -> str:
    # Deterministic key ordering = stable for diffing + compliance review
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def append_audit_event(event: AuditEvent, path: Optional[str] = None) -> str:
    """
    Append-only JSONL. Returns the resolved path written to.
    Atomic enough for our use-case: O_APPEND + single write per line.
    """
    p = path or audit_log_path()
    _ensure_parent_dir(p)

    line = _to_compact_json(asdict(event)) + "\n"

    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)

    return p


def build_audit_event(*,
    policy_version: str,
    actor: str,
    endpoint: str,
    requested_model: str,
    chosen_model: str,
    reason: str,
    escalation: bool,
    tools_disabled: bool,
    decision_hash: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    message_count: Optional[int] = None,
    has_system_prompt: Optional[bool] = None,
    risk_score: Optional[float] = None,
    task_type: Optional[str] = None,
) -> AuditEvent:
    return AuditEvent(
        event_version=AUDIT_EVENT_VERSION,
        ts_unix_ms=_now_unix_ms(),
        policy_version=policy_version,
        actor=actor,
        endpoint=endpoint,
        requested_model=requested_model,
        chosen_model=chosen_model,
        reason=reason,
        escalation=escalation,
        tools_disabled=tools_disabled,
        decision_hash=decision_hash,
        max_tokens=max_tokens,
        temperature=temperature,
        message_count=message_count,
        has_system_prompt=has_system_prompt,
        risk_score=risk_score,
        task_type=task_type,
    )
