# DUNBAR_OS — Auto Model Policy v1 (Broker)

## 1) Purpose
This document defines Broker-side deterministic routing for `model="auto"` in `/v1/chat` and `/v1/route`, and policy-first downgrade behavior when a requested concrete model is forbidden or unavailable. The Broker is the governance layer: policy constraints are applied before availability fallback so forbidden models are never selected.

## 2) What is enforced
- `model="auto"` is resolved deterministically from actor + endpoint + request features (`max_tokens`, temperature, message count, system prompt flag, optional risk score).
- Policy-first controls are authoritative:
  - actor `public` is forced to SAFE_SMALL bucket.
  - actor `public` has tool usage disabled.
  - actor `public` cannot use remote models.
- Requested model handling:
  - allowed + available => keep requested model.
  - forbidden => deterministic downgrade (`reason=DOWNGRADE_FORBIDDEN`, `escalation=true`).
  - unavailable => deterministic fallback (`reason=FALLBACK_UNAVAILABLE`).
- Remote/cloud model selection is blocked unless **both** request and actor policy permit remote usage.
- Response compatibility is preserved; model-routing details are returned in `meta`.

### Deterministic bucket logic (v1)
| Bucket | Preference chain |
|---|---|
| FAST | `fast_primary` → `fast_secondary` → `last_known_good_fast` |
| REASONING | `reasoning_primary` → `reasoning_secondary` → `last_known_good_reasoning` |
| SAFE_SMALL | `safe_primary` → `safe_secondary` |

Selection defaults:
- `rainbow`: `max_tokens >= 300` => REASONING, else FAST.
- `public`: always SAFE_SMALL.

## 3) How to use
```bash
pwd
ls -la
```

### Chat (auto)
```bash
curl -sS http://localhost:8787/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'x-genx-actor: rainbow' \
  -H 'x-genx-key-id: <kid>' \
  -H 'x-genx-ts: <unix_ts>' \
  -H 'x-genx-nonce: <nonce>' \
  -H 'x-genx-sig: <hmac_hex>' \
  -d '{
    "actor":"rainbow",
    "intent":"chat",
    "payload":{
      "thread_id":"thread-1",
      "model":"auto",
      "max_tokens":350,
      "temperature":0.2,
      "messages":[{"role":"user","content":"Plan this architecture."}]
    }
  }'
```

### Route (auto)
```bash
curl -sS http://localhost:8787/v1/route \
  -H 'Content-Type: application/json' \
  -H 'x-genx-actor: rainbow' \
  -H 'x-genx-key-id: <kid>' \
  -H 'x-genx-ts: <unix_ts>' \
  -H 'x-genx-nonce: <nonce>' \
  -H 'x-genx-sig: <hmac_hex>' \
  -d '{
    "actor":"rainbow",
    "intent":"ping",
    "payload":{"model":"auto","max_tokens":100}
  }'
```

### Health and status checks
```bash
curl -sS http://localhost:8787/health
```

TODO: VERIFY in `genx_kernel/broker/app/main.py` (`/v1/status` route) — endpoint not present in current broker file.

## 4) How to test
```bash
pytest -q tests/test_auto_model_policy.py
pytest -q
```

### OpenAPI check (chosen_model visible)
```bash
curl -sS http://localhost:8787/openapi.json | jq '.paths["/v1/chat"].post.responses["200"]'
curl -sS http://localhost:8787/openapi.json | jq '.paths["/v1/route"].post.responses["200"]'
```

## 5) Failure modes + exact fixes
- Forbidden requested model.
  - Symptom: `meta.reason=DOWNGRADE_FORBIDDEN`, `meta.escalation=true`.
  - Fix: request an actor-allowed model, or use `model="auto"`.
- Unavailable selected model.
  - Symptom: `meta.reason=FALLBACK_UNAVAILABLE`.
  - Fix: add model to `AVAILABLE_LOCAL_MODELS` or ensure a fallback model is loaded.
- No allowed model available.
  - Symptom: broker returns `503 auto_model_selection_failed:NO_ALLOWED_MODEL_AVAILABLE`.
  - Fix: ensure at least one actor-allowed model exists in the active catalog.

## 6) Security notes (threat model + mitigations)
- Threat: model escalation via request tampering.
  - Mitigation: HMAC-authenticated actor identity + actor policy before model selection.
- Threat: remote model exfiltration.
  - Mitigation: remote models require both request opt-in and actor-policy permit; `public` hard-denied.
- Threat: unsafe tool invocation by low-trust actors.
  - Mitigation: `public` actor forces `tools_disabled`; route payload tools are stripped.
- Threat: prompt injection.
  - Mitigation: broker injects system guardrails and keeps policy decision out of user-controlled content.

## 7) Next upgrades
- Richer deterministic task classification (coding/planning/vision weighting).
- Dynamic load-aware routing with deterministic tie-breakers.
- Per-project actor profiles and signed policy bundles.
- Catalog freshness checks and health-based but deterministic model readiness state.
