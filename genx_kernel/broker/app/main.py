from __future__ import annotations
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import logging

import urllib.request
import urllib.error

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .routing.auto_model_policy import (
    AutoModelInputs,
    ChatMessage as PolicyChatMessage,
    ModelCatalog,
    build_actor_policy,
    resolve_auto_model,
)

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse

AUTH_WINDOW_SECONDS = int(os.getenv("GENX_AUTH_WINDOW_SECONDS", "300"))
NONCE_TTL_SECONDS = int(os.getenv("GENX_NONCE_TTL_SECONDS", str(AUTH_WINDOW_SECONDS)))

# -----------------------------
# Auth errors (typed)
# -----------------------------
class AuthError(Exception):
    def __init__(self, code: str, detail: str, status_code: int = 401):
        self.code = code
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)

# -----------------------------
# Key registry loader
# -----------------------------
def _load_auth_keys() -> Dict[str, Dict[str, bytes]]:
    """
    Expects GENX_AUTH_KEYS_JSON like:
    {
      "rainbow": {"kid1": "base64:....", "kid2": "hex:...."},
      "public": {"kid1": "base64:...."}
    }
    """
    raw = os.getenv("GENX_AUTH_KEYS_JSON", "").strip()
    if not raw:
        return {}

    try:
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid GENX_AUTH_KEYS_JSON: {e}") from e

    out: Dict[str, Dict[str, bytes]] = {}
    for actor, kids in (data or {}).items():
        if not isinstance(kids, dict):
            continue
        out[actor] = {}
        for kid, secret in kids.items():
            if not isinstance(secret, str):
                continue
            secret = secret.strip()
            if secret.startswith("base64:"):
                b = base64.b64decode(secret[len("base64:"):])
            elif secret.startswith("hex:"):
                b = bytes.fromhex(secret[len("hex:"):])
            else:
                # allow raw string (NOT recommended) but support for local testing
                b = secret.encode("utf-8")
            out[actor][kid] = b
    return out

_AUTH_KEYS_CACHE: Optional[Dict[str, Dict[str, bytes]]] = None

def _get_actor_secret(actor: str, kid: str) -> bytes:
    global _AUTH_KEYS_CACHE
    if _AUTH_KEYS_CACHE is None:
        _AUTH_KEYS_CACHE = _load_auth_keys()

    actor_map = (_AUTH_KEYS_CACHE or {}).get(actor)
    if not actor_map:
        raise AuthError("AUTH_UNKNOWN_ACTOR", f"Unknown actor '{actor}'", 401)

    secret = actor_map.get(kid)
    if not secret:
        raise AuthError("AUTH_UNKNOWN_KID", f"Unknown kid '{kid}' for actor '{actor}'", 401)
    return secret

# -----------------------------
# Nonce replay guard (simple TTL map)
# -----------------------------
_NONCE_STORE: Dict[str, float] = {}  # key = f"{actor}:{nonce}" -> expires_at_epoch

def _nonce_seen(actor: str, nonce: str) -> bool:
    now = time.time()

    # opportunistic cleanup (fast enough for v1)
    # remove expired entries
    if _NONCE_STORE:
        expired_keys = [k for k, exp in _NONCE_STORE.items() if exp <= now]
        for k in expired_keys:
            _NONCE_STORE.pop(k, None)

    k = f"{actor}:{nonce}"
    exp = _NONCE_STORE.get(k)
    return exp is not None and exp > now

def _nonce_mark(actor: str, nonce: str) -> None:
    _NONCE_STORE[f"{actor}:{nonce}"] = time.time() + float(NONCE_TTL_SECONDS)

# -----------------------------
# Signature v1
# -----------------------------
def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _build_string_to_sign(
    actor: str,
    kid: str,
    timestamp: str,
    nonce: str,
    method: str,
    path: str,
    body_sha256_hex: str,
) -> bytes:
    # Canonical (stable) format
    s = "\n".join([
        "v1",
        actor,
        kid,
        timestamp,
        nonce,
        method.upper(),
        path,
        body_sha256_hex,
        "",
    ])
    return s.encode("utf-8")

def _parse_signature(sig_header: str) -> Tuple[str, str]:
    """
    Accept: "v1=<hex>"
    """
    if not sig_header:
        raise AuthError("AUTH_MISSING_SIGNATURE", "Missing X-GenX-Signature", 401)
    sig_header = sig_header.strip()
    if not sig_header.startswith("v1="):
        raise AuthError("AUTH_BAD_SIGNATURE_FORMAT", "Bad signature format", 401)
    return ("v1", sig_header[len("v1="):].strip())

async def _verify_hmac_request(request: Request) -> Tuple[str, str]:
    """
    Backward-compatible wrapper.

    Tests monkeypatch `_verify_hmac`.
    Week3 middleware calls `_verify_hmac_request`, so we route through `_verify_hmac`
    to keep tests valid while preserving the Week3 request-based auth flow.
    """
    body_bytes = await request.body()
    headers_lc = {k.lower(): v for k, v in request.headers.items()}

    actor, kid = _verify_hmac(
        request.url.path,
        request.method,
        body_bytes,
        headers_lc,
    )
    return actor, kid

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


logger = logging.getLogger("genx.broker")

DEFAULT_BUCKETS = {
    "FAST": ["fast_primary", "fast_secondary", "last_known_good_fast"],
    "REASONING": ["reasoning_primary", "reasoning_secondary", "last_known_good_reasoning"],
    "SAFE_SMALL": ["safe_primary", "safe_secondary"],
}

AVAILABLE_LOCAL_MODELS = {
    m.strip()
    for m in os.getenv(
        "AVAILABLE_LOCAL_MODELS",
        "fast_primary,fast_secondary,last_known_good_fast,reasoning_primary,reasoning_secondary,last_known_good_reasoning,safe_primary,safe_secondary",
    ).split(",")
    if m.strip()
}
REMOTE_MODELS = {
    m.strip()
    for m in os.getenv("REMOTE_MODELS", "").split(",")
    if m.strip()
}


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
    allow_remote_models: bool = False


class ChatRequest(BaseModel):
    actor: str
    intent: str = "chat"
    payload: ChatPayload


class ModelDecisionMeta(BaseModel):
    chosen_model: str
    reason: str
    escalation: bool


class ChatResponse(BaseModel):
    ok: bool
    request_id: str
    actor: str
    thread_id: str
    model: str
    assistant: Dict[str, str]
    elapsed_ms: int
    meta: Optional[ModelDecisionMeta] = None


class RouteResponse(BaseModel):
    ok: bool
    request_id: str
    firewall: Dict[str, Any]
    policy: Optional[Dict[str, Any]] = None
    sandbox: Optional[Dict[str, Any]] = None
    elapsed_ms: int
    meta: Optional[ModelDecisionMeta] = None


def _build_model_catalog() -> ModelCatalog:
    return ModelCatalog(
        bucket_models=DEFAULT_BUCKETS,
        available_models=frozenset(AVAILABLE_LOCAL_MODELS | REMOTE_MODELS),
        remote_models=frozenset(REMOTE_MODELS),
    )


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

import logging

_log_auth = logging.getLogger("genx.broker.auth")

def _emit_auth_audit_safe(event: dict) -> None:
    """
    Fail-safe auth audit emitter.
    Middleware must NEVER throw because of logging/audit.
    """
    try:
        _log_auth.info(json.dumps(event, separators=(",", ":"), sort_keys=True))
    except Exception:
        pass

app = FastAPI(title="GenX Broker", version="1.0.0")

def _auth_bypass_path(path: str) -> bool:
    # Public endpoints (no HMAC required)
    if path in ("/health", "/openapi.json"):
        return True
    if path.startswith("/docs"):
        return True
    return False

@app.middleware("http")
async def hmac_middleware(request: Request, call_next):
    # Bypass docs/health/openapi
    if _auth_bypass_path(request.url.path):
        return await call_next(request)

    # Only enforce on v1 surface (tight perimeter)
    if not request.url.path.startswith("/v1/"):
        return await call_next(request)

    try:
        actor, kid = await _verify_hmac_request(request)
        request.state.actor = actor
        request.state.kid = kid

        _emit_auth_audit_safe({
            "event_type": "AUTH_OK",
            "actor": actor,
            "kid": kid,
            "path": request.url.path,
            "method": request.method,
            "ts": int(time.time()),
        })

        return await call_next(request)

    except AuthError as e:
        _emit_auth_audit_safe({
            "event_type": e.code,
            "actor": (request.headers.get("X-GenX-Actor") or "").strip() or None,
            "kid": (request.headers.get("X-GenX-Key-Id") or "").strip() or None,
            "path": request.url.path,
            "method": request.method,
            "ts": int(time.time()),
            "detail": e.detail,
        })
        return JSONResponse(
            status_code=e.status_code,
            content={"ok": False, "error": e.code, "detail": e.detail},
        )


# ============================================================
# /v1/route  (firewall → policy → sandbox)
# ============================================================

# --- ROUTE HANDLER (canonical) ----------------------------------------------

@app.post("/v1/route", response_model=RouteResponse)
async def v1_route(req: RouteRequest, request: Request):
    t0 = time.time()
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    allow_remote_models = bool((req.payload or {}).get("allow_remote_models", False))
    requested_model = str((req.payload or {}).get("model", "auto"))
    catalog = _build_model_catalog()
    actor_policy = build_actor_policy(req.actor, allow_remote_models=allow_remote_models)
    route_messages = [PolicyChatMessage(role="user", content=req.intent)]
    try:
        decision_meta = resolve_auto_model(
            AutoModelInputs(
            actor=req.actor,
            endpoint="route",
            requested_model=requested_model,
            max_tokens=int((req.payload or {}).get("max_tokens", 200)),
            temperature=(req.payload or {}).get("temperature"),
            message_count=1,
            has_system_prompt=False,
            risk_score=(req.payload or {}).get("risk_score"),
            allow_remote_models=allow_remote_models,
            messages=route_messages,
            ),
            catalog=catalog,
            policy=actor_policy,
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=f"auto_model_selection_failed:{str(e)}")

    if decision_meta.tools_disabled and isinstance(req.payload, dict) and req.payload.get("tools"):
        req.payload["tools"] = []

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
            "meta": decision_meta.__dict__,
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
            "model_meta": decision_meta.__dict__,
        })
        logger.info(
            "auto_model actor=%s endpoint=route requested_model=%s chosen_model=%s reason=%s escalation=%s",
            req.actor,
            requested_model,
            decision_meta.chosen_model,
            decision_meta.reason,
            decision_meta.escalation,
        )
        return resp

    policy_out = _post_json(POLICY_URL, req.model_dump(), timeout_s=8.0)
    if not policy_out.get("ok", False):
        raise HTTPException(status_code=502, detail="policy unavailable")

    decision = policy_out.get("decision", "deny")

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
        "meta": decision_meta.__dict__,
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
        "model_meta": decision_meta.__dict__,
    })
    logger.info(
        "auto_model actor=%s endpoint=route requested_model=%s chosen_model=%s reason=%s escalation=%s",
        req.actor,
        requested_model,
        decision_meta.chosen_model,
        decision_meta.reason,
        decision_meta.escalation,
    )

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

@app.post("/v1/chat", response_model=ChatResponse)
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

    allow_remote_models = bool(getattr(obj.payload, "allow_remote_models", False))
    catalog = _build_model_catalog()
    actor_policy = build_actor_policy(actor, allow_remote_models=allow_remote_models)
    try:
        decision_meta = resolve_auto_model(
            AutoModelInputs(
            actor=actor,
            endpoint="chat",
            requested_model=obj.payload.model,
            max_tokens=obj.payload.max_tokens,
            temperature=obj.payload.temperature,
            message_count=len(obj.payload.messages),
            has_system_prompt=any(m.role == "system" for m in obj.payload.messages),
            risk_score=None,
            allow_remote_models=allow_remote_models,
                messages=[PolicyChatMessage(role=m.role, content=m.content) for m in obj.payload.messages],
            ),
            catalog=catalog,
            policy=actor_policy,
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=f"auto_model_selection_failed:{str(e)}")

    # Call Ollama
    assistant_text = _ollama_chat_openai(
        messages=hist,
        model=decision_meta.chosen_model,
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
        "model": decision_meta.chosen_model,
        "requested_model": obj.payload.model,
        "model_meta": decision_meta.__dict__,
        "elapsed_ms": elapsed_ms,
    })
    logger.info(
        "auto_model actor=%s endpoint=chat requested_model=%s chosen_model=%s reason=%s escalation=%s",
        actor,
        obj.payload.model,
        decision_meta.chosen_model,
        decision_meta.reason,
        decision_meta.escalation,
    )
    return {
        "ok": True,
        "request_id": req_id,
        "actor": actor,
        "thread_id": thread_id,
        "model": decision_meta.chosen_model,
        "assistant": assistant_msg,
        "elapsed_ms": elapsed_ms,
        "meta": decision_meta.__dict__,
    }

