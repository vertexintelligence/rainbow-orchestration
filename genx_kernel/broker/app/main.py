from __future__ import annotations
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import urllib.request
import urllib.error

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


# ============================================================
# ENV + CONFIG
# ============================================================

FIREWALL_URL = os.getenv("FIREWALL_URL", "http://firewall:8786/v1/inspect")
POLICY_URL = os.getenv("POLICY_URL", "http://policy:8788/v1/decide")
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://sandbox:8789/v1/run")

# Ollama (OpenAI-compatible endpoint is required for chat)
# Verified working on your node:
#   http://host.docker.internal:11434/v1/chat/completions
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")

AUDIT_PATH = os.getenv("AUDIT_PATH", "/data/audit/audit.jsonl")

FIREWALL_THRESHOLD = int(os.getenv("FIREWALL_THRESHOLD", "60"))

HMAC_KEYS_JSON = os.getenv("HMAC_KEYS_JSON", "{}")
ACTOR_KEYS_JSON = os.getenv("ACTOR_KEYS_JSON", "{}")

HMAC_SKEW_SECONDS = int(os.getenv("HMAC_SKEW_SECONDS", "120"))
HMAC_NONCE_TTL_SECONDS = int(os.getenv("HMAC_NONCE_TTL_SECONDS", "900"))

try:
    HMAC_KEYS: Dict[str, str] = json.loads(HMAC_KEYS_JSON) if HMAC_KEYS_JSON else {}
except Exception:
    HMAC_KEYS = {}

try:
    ACTOR_KEYS: Dict[str, str] = json.loads(ACTOR_KEYS_JSON) if ACTOR_KEYS_JSON else {}
except Exception:
    ACTOR_KEYS = {}


# ============================================================
# AUDIT
# ============================================================

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _audit_write(obj: Dict[str, Any]) -> None:
    # Keep audit always best-effort but never crash requests.
    try:
        os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ============================================================
# HMAC SECURITY
# ============================================================

_NONCES: Dict[str, float] = {}


def _cleanup_nonces(now: float) -> None:
    # Remove expired nonces
    expired = [n for n, exp in _NONCES.items() if exp <= now]
    for n in expired:
        _NONCES.pop(n, None)


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _verify_hmac(path: str, method: str, body_bytes: bytes, headers: Dict[str, str]) -> Tuple[str, str]:
    """
    Returns (actor, kid) if valid, otherwise raises HTTPException.
    Canonical:
      METHOD\nPATH\nTS\nNONCE\nBODY_SHA
    """
    if not HMAC_KEYS or not ACTOR_KEYS:
        raise HTTPException(status_code=401, detail="hmac_not_configured")

    actor = headers.get("x-genx-actor", "").strip()
    kid_hdr = headers.get("x-genx-key-id", "").strip()
    ts_hdr = headers.get("x-genx-ts", "").strip()
    nonce = headers.get("x-genx-nonce", "").strip()
    sig = headers.get("x-genx-sig", "").strip()

    if not actor or not ts_hdr or not nonce or not sig:
        raise HTTPException(status_code=401, detail="hmac_missing_headers")

    expected_kid = ACTOR_KEYS.get(actor, "")
    if not expected_kid:
        raise HTTPException(status_code=401, detail="hmac_unknown_actor")

    # Allow client to send key-id header, but enforce it matches the actor mapping.
    if kid_hdr and kid_hdr != expected_kid:
        raise HTTPException(status_code=401, detail="hmac_key_id_mismatch")

    secret = HMAC_KEYS.get(expected_kid, "")
    if not secret:
        raise HTTPException(status_code=401, detail="hmac_unknown_key")

    try:
        ts = int(ts_hdr)
    except Exception:
        raise HTTPException(status_code=401, detail="hmac_bad_timestamp")

    now = int(time.time())
    if abs(now - ts) > HMAC_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail="hmac_timestamp_skew")

    # nonce replay protection
    _cleanup_nonces(time.time())
    if nonce in _NONCES:
        raise HTTPException(status_code=401, detail="hmac_replay_nonce")

    body_sha = _sha256_hex(body_bytes)
    canonical = f"{method.upper()}\n{path}\n{ts}\n{nonce}\n{body_sha}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected_sig, sig):
        raise HTTPException(status_code=401, detail="hmac_bad_signature")

    _NONCES[nonce] = time.time() + HMAC_NONCE_TTL_SECONDS
    _cleanup_nonces(time.time())
    return actor, expected_kid


# ============================================================
# HTTP UTILS
# ============================================================

def _post_json(url: str, payload: Dict[str, Any], timeout_s: float = 10.0) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            if not raw:
                return {"ok": False, "error": "empty_response"}
            return json.loads(raw.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return {"ok": False, "http_status": e.code, "error": body[:600]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:600]}


# ============================================================
# MODELS
# ============================================================

class RouteRequest(BaseModel):
    actor: str
    intent: str
    payload: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatPayload(BaseModel):
    thread_id: str
    messages: List[ChatMessage]
    max_tokens: int = 200
    temperature: float = 0.2
    model: str = "mistral:latest"


class ChatRequest(BaseModel):
    actor: str
    intent: str = "chat"
    payload: ChatPayload


# ============================================================
# CHAT MEMORY (lightweight, local-only)
# ============================================================

_CHAT_STORE: Dict[str, List[Dict[str, str]]] = {}
_CHAT_STORE_EXP: Dict[str, float] = {}
CHAT_TTL_SECONDS = int(os.getenv("CHAT_TTL_SECONDS", "21600"))  # 6 hours default


def _chat_store_get(thread_id: str) -> List[Dict[str, str]]:
    now = time.time()
    exp = _CHAT_STORE_EXP.get(thread_id, 0)
    if exp and exp < now:
        _CHAT_STORE.pop(thread_id, None)
        _CHAT_STORE_EXP.pop(thread_id, None)
    return _CHAT_STORE.get(thread_id, [])


def _chat_store_put(thread_id: str, hist: List[Dict[str, str]]) -> None:
    _CHAT_STORE[thread_id] = hist
    _CHAT_STORE_EXP[thread_id] = time.time() + CHAT_TTL_SECONDS


# ============================================================
# OLLAMA CHAT (OpenAI compatible endpoint)
# ============================================================

def _ollama_chat_openai(messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float) -> str:
    url = f"{OLLAMA_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=f"ollama_http_error:{e.code}:{body[:200]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ollama_error:{str(e)[:200]}")

    try:
        obj = json.loads(raw)
        choices = obj.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").lstrip()
    except Exception:
        raise HTTPException(status_code=502, detail="ollama_bad_json")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="GenX Broker", version="1.0.0")


@app.middleware("http")
async def hmac_middleware(request: Request, call_next):
    # Only enforce on API routes (not docs/openapi).
    path = request.url.path
    if path.startswith("/v1/"):
        body_bytes = await request.body()
        headers_lc = {k.lower(): v for k, v in request.headers.items()}
        actor, kid = _verify_hmac(path, request.method, body_bytes, headers_lc)

        # Attach for downstream usage / auditing
        request.state.actor = actor
        request.state.kid = kid

    return await call_next(request)


@app.get("/health")
def health():
    return {"ok": True, "ts": _now_iso()}


# ============================================================
# /v1/route  (firewall → policy → sandbox)
# ============================================================

# --- ROUTE HANDLER (canonical) ----------------------------------------------

@app.post("/v1/route")
async def v1_route(req: RouteRequest, request: Request):
    t0 = time.time()
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    # 0) AUTH / SIGNATURE GATE (already implemented in your code)
    # (Assumes you already validated headers and signature before reaching here.)

    # 1) FIREWALL
    firewall_out = _post_json(FIREWALL_URL, req.model_dump(), timeout_s=8.0)
    if not firewall_out.get("ok", False):
        raise HTTPException(status_code=502, detail="firewall unavailable")
    if not firewall_out.get("allowed", False):
        resp = {
            "ok": False,
            "request_id": request_id,
            "firewall": firewall_out,
            "policy": None,
            "sandbox": None,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
        _audit_write({
            "ts": _now_iso(),
            "request_id": request_id,
            "stage": "route",
            "actor": req.actor,
            "intent": req.intent,
            "firewall": firewall_out,
            "policy": None,
            "sandbox": None,
            "elapsed_ms": resp["elapsed_ms"],
        })
        return resp

    # 2) POLICY
    policy_out = _post_json(POLICY_URL, req.model_dump(), timeout_s=8.0)
    if not policy_out.get("ok", False):
        raise HTTPException(status_code=502, detail="policy unavailable")

    decision = policy_out.get("decision", "deny")

    # 3) SANDBOX (allow-only)
    sandbox_out: Optional[Dict[str, Any]] = None
    if decision == "allow" and req.intent in {"run_code", "sandbox_run"}:
        payload = req.payload or {}
        language = payload.get("language")
        code = payload.get("code")
        if not language or not code:
            raise HTTPException(
                status_code=422,
                detail="payload must include {language, code} for run_code"
            )
        sandbox_out = _post_json(
            SANDBOX_URL,
            {"language": language, "code": code},
            timeout_s=30.0,
        )

    resp = {
        "ok": (decision == "allow"),
        "request_id": request_id,
        "firewall": firewall_out,
        "policy": policy_out,
        "sandbox": sandbox_out,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }

    _audit_write({
        "ts": _now_iso(),
        "request_id": request_id,
        "stage": "route",
        "actor": req.actor,
        "intent": req.intent,
        "firewall": firewall_out,
        "policy": policy_out,
        "payload": (req.payload if isinstance(req.payload, dict) else None),
        "sandbox": sandbox_out,
        "elapsed_ms": resp["elapsed_ms"],
    })

    return resp


def _system_guardrails() -> Dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are GenX running locally. Follow user instructions exactly. "
            "If user says 'Say only: <TEXT>', output exactly <TEXT> and nothing else. "
            "No links. No suggestions. No extra words."
        )
    }

# ============================================================
# /v1/chat  (secure local chat → Ollama)
# ============================================================

@app.post("/v1/chat")
async def v1_chat(obj: ChatRequest, request: Request):
    started = time.time()
    req_id = f"req_{uuid.uuid4().hex[:12]}"

    # Use actor validated by middleware (prevents spoofing)
    actor = getattr(request.state, "actor", obj.actor)  # fallback if middleware ever bypassed
    thread_id = obj.payload.thread_id

    # Build history: stored + incoming
    hist = _chat_store_get(thread_id)

    # Ensure guardrails are always first (single system message)
    hist = [_system_guardrails()] + [m for m in hist if m.get("role") != "system"]

    # Append incoming messages (exactly once)
    for m in obj.payload.messages:
        hist.append({"role": m.role, "content": m.content})

    # Call Ollama
    assistant_text = _ollama_chat_openai(
        messages=hist,
        model=obj.payload.model,
        max_tokens=obj.payload.max_tokens,
        temperature=obj.payload.temperature,
    )

    assistant_msg = {"role": "assistant", "content": assistant_text}
    hist.append(assistant_msg)
    _chat_store_put(thread_id, hist)

    elapsed_ms = int((time.time() - started) * 1000)

    _audit_write({
        "ts": _now_iso(),
        "request_id": req_id,
        "stage": "chat",
        "actor": actor,
        "thread_id": thread_id,
        "model": obj.payload.model,
        "elapsed_ms": elapsed_ms,
    })
    return {
          "ok": True,
          "request_id": req_id,
          "actor": actor,
          "thread_id": thread_id,
          "model": obj.payload.model,
          "assistant": assistant_msg,
          "elapsed_ms": elapsed_ms,
        
    }

