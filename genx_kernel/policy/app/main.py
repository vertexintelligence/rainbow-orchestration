from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="genx-policy", version="1.3.0")

AUDIT_PATH = os.getenv("AUDIT_PATH", "/data/audit/audit.jsonl")

DENY_INTENTS = {
    "shell_exec",
    "net_scan",
    "exfiltrate",
    "download_malware",
}

ALLOW_INTENTS = {
    "ping",
    "run_code",
    "sandbox_run",
}

ALLOW_LANGUAGES = {"python"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit_write(event: Dict[str, Any]) -> None:
    p = Path(AUDIT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


class FirewallResult(BaseModel):
    ok: bool = True
    allowed: bool = True
    reason: Optional[str] = None
    score: Optional[int] = None
    threshold: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class DecideRequest(BaseModel):
    request_id: str
    actor: str
    intent: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    firewall: FirewallResult = Field(default_factory=FirewallResult)


class DecideResponse(BaseModel):
    ok: bool = True
    decision: str  # allow | deny | review
    reason: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "policy", "version": app.version}


@app.post("/v1/decide", response_model=DecideResponse)
def decide(req: DecideRequest) -> DecideResponse:
    t0 = time.time()

    if req.firewall and req.firewall.ok and req.firewall.allowed is False:
        out = DecideResponse(decision="deny", reason="blocked_by_firewall")
    elif req.intent in DENY_INTENTS:
        out = DecideResponse(decision="deny", reason="intent_hard_denied")
    elif req.intent in ALLOW_INTENTS:
        if req.intent in {"run_code", "sandbox_run"}:
            lang = (req.payload or {}).get("language")
            if lang and lang not in ALLOW_LANGUAGES:
                out = DecideResponse(decision="deny", reason="language_not_allowed")
            else:
                out = DecideResponse(decision="allow", reason=None)
        else:
            out = DecideResponse(decision="allow", reason=None)
    else:
        out = DecideResponse(decision="review", reason="intent_requires_review")

    out.meta["elapsed_ms"] = int((time.time() - t0) * 1000)

    _audit_write(
        {
            "ts": _now_iso(),
            "request_id": req.request_id,
            "stage": "policy",
            "actor": req.actor,
            "intent": req.intent,
            "decision": out.decision,
            "reason": out.reason,
            "payload": req.payload,
            "firewall": req.firewall.model_dump() if req.firewall else None,
            "elapsed_ms": out.meta["elapsed_ms"],
        }
    )

    return out
