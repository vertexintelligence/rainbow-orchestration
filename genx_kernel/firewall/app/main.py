from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="genx-firewall", version="1.1.0")

AUDIT_PATH = os.getenv("AUDIT_PATH", "/data/audit/audit.jsonl")
FIREWALL_THRESHOLD = int(os.getenv("FIREWALL_THRESHOLD", "60"))

DENY_KEYWORDS = {
    "rm -rf": 100,
    "exfil": 90,
    "net_scan": 80,
    "shell": 70,
    "exec": 70,
    "powershell": 70,
    "curl http": 60,
    "wget http": 60,
}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _audit_write(event: Dict[str, Any]) -> None:
    fp = Path(AUDIT_PATH)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def _score(intent: str, payload: Dict[str, Any]) -> int:
    text = (intent or "").lower() + " " + json.dumps(payload, default=str).lower()
    score = 0
    for k, pts in DENY_KEYWORDS.items():
        if k in text:
            score = max(score, pts)
    return score

class InspectRequest(BaseModel):
    request_id: str
    actor: str = "unknown"
    intent: str
    payload: Dict[str, Any] = Field(default_factory=dict)

class InspectResponse(BaseModel):
    ok: bool = True
    allowed: bool = True
    reason: Optional[str] = None
    score: int = 0
    threshold: int = FIREWALL_THRESHOLD
    meta: Dict[str, Any] = Field(default_factory=dict)

@app.get("/health")
def health():
    return {"ok": True, "service": "firewall", "version": app.version}

@app.post("/v1/inspect", response_model=InspectResponse)
def inspect(req: InspectRequest):
    t0 = time.time()
    score = _score(req.intent, req.payload)
    allowed = score < FIREWALL_THRESHOLD
    reason = None if allowed else "firewall_threshold_exceeded"

    out = InspectResponse(
        ok=True,
        allowed=allowed,
        reason=reason,
        score=score,
        threshold=FIREWALL_THRESHOLD,
        meta={"elapsed_ms": int((time.time() - t0) * 1000)},
    )

    _audit_write({
        "ts": _now_iso(),
        "service": "firewall",
        "stage": "inspect",
        "request_id": req.request_id,
        "actor": req.actor,
        "intent": req.intent,
        "allowed": out.allowed,
        "reason": out.reason,
        "score": out.score,
        "threshold": out.threshold,
        "payload": req.payload,
        "meta": out.meta,
    })
    return out
