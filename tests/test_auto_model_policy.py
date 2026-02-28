from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from genx_kernel.broker.app import main
from genx_kernel.broker.app.routing.auto_model_policy import (
    AutoModelInputs,
    ChatMessage,
    ModelCatalog,
    build_actor_policy,
    resolve_auto_model,
)


def _catalog(available: set[str]) -> ModelCatalog:
    return ModelCatalog(
        bucket_models={
            "FAST": ["fast_primary", "fast_secondary", "last_known_good_fast"],
            "REASONING": ["reasoning_primary", "reasoning_secondary", "last_known_good_reasoning"],
            "SAFE_SMALL": ["safe_primary", "safe_secondary"],
        },
        available_models=frozenset(available),
        remote_models=frozenset({"remote_ultra"}),
    )


def _inputs(**kwargs) -> AutoModelInputs:
    base = dict(
        actor="rainbow",
        endpoint="chat",
        requested_model="auto",
        max_tokens=200,
        temperature=0.2,
        message_count=1,
        has_system_prompt=False,
        risk_score=None,
        allow_remote_models=False,
        messages=[ChatMessage(role="user", content="hello")],
    )
    base.update(kwargs)
    return AutoModelInputs(**base)


def test_rainbow_chooses_fast_under_300_tokens() -> None:
    decision = resolve_auto_model(_inputs(max_tokens=299), _catalog({"fast_primary"}), build_actor_policy("rainbow"))
    assert decision.chosen_model == "fast_primary"
    assert decision.reason == "AUTO_FAST"


def test_rainbow_chooses_reasoning_at_or_above_300_tokens() -> None:
    decision = resolve_auto_model(
        _inputs(max_tokens=300),
        _catalog({"reasoning_primary", "fast_primary"}),
        build_actor_policy("rainbow"),
    )
    assert decision.chosen_model == "reasoning_primary"
    assert decision.reason == "AUTO_REASONING"


def test_public_actor_forces_safe_small_and_tools_disabled() -> None:
    decision = resolve_auto_model(
        _inputs(actor="public", max_tokens=999, allow_remote_models=True, requested_model="auto"),
        _catalog({"safe_primary", "reasoning_primary", "remote_ultra"}),
        build_actor_policy("public", allow_remote_models=True),
    )
    assert decision.chosen_model == "safe_primary"
    assert decision.reason == "PUBLIC_SAFE"
    assert decision.tools_disabled is True


def test_forbidden_model_downgrades_with_escalation() -> None:
    decision = resolve_auto_model(
        _inputs(actor="public", requested_model="reasoning_primary"),
        _catalog({"safe_primary", "reasoning_primary"}),
        build_actor_policy("public"),
    )
    assert decision.chosen_model == "safe_primary"
    assert decision.escalation is True
    assert decision.reason == "DOWNGRADE_FORBIDDEN"


def test_unavailable_model_fallback() -> None:
    decision = resolve_auto_model(
        _inputs(requested_model="reasoning_primary", max_tokens=400),
        _catalog({"reasoning_secondary", "fast_primary"}),
        build_actor_policy("rainbow"),
    )
    assert decision.chosen_model == "reasoning_secondary"
    assert decision.reason == "FALLBACK_UNAVAILABLE"


def test_determinism_same_input_same_output() -> None:
    inputs = _inputs(max_tokens=300, requested_model="auto")
    catalog = _catalog({"reasoning_primary", "fast_primary"})
    policy = build_actor_policy("rainbow")

    d1 = resolve_auto_model(inputs, catalog, policy)
    d2 = resolve_auto_model(inputs, catalog, policy)

    assert d1.chosen_model == d2.chosen_model
    assert d1.reason == d2.reason


def test_api_chat_auto_includes_chosen_model(monkeypatch) -> None:
    monkeypatch.setattr(main, "_verify_hmac", lambda *args, **kwargs: ("rainbow", "kid"))
    monkeypatch.setattr(main, "_ollama_chat_openai", lambda **kwargs: "ok")

    client = TestClient(main.app)
    payload = {
        "actor": "rainbow",
        "intent": "chat",
        "payload": {
            "thread_id": "t1",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
            "temperature": 0.1,
            "model": "auto",
        },
    }
    resp = client.post("/v1/chat", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["meta"]["chosen_model"]


def test_api_route_auto_includes_chosen_model(monkeypatch) -> None:
    monkeypatch.setattr(main, "_verify_hmac", lambda *args, **kwargs: ("rainbow", "kid"))

    def fake_post_json(url: str, payload: dict, timeout_s: float = 10.0):
        if "inspect" in url:
            return {"ok": True, "allowed": True}
        if "decide" in url:
            return {"ok": True, "decision": "allow"}
        return {"ok": True}

    monkeypatch.setattr(main, "_post_json", fake_post_json)

    client = TestClient(main.app)
    payload = {
        "actor": "rainbow",
        "intent": "ping",
        "payload": {"model": "auto", "max_tokens": 100},
    }
    r1 = client.post("/v1/route", json=payload)
    r2 = client.post("/v1/route", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    b1 = r1.json()
    b2 = r2.json()
    assert b1["ok"] is True
    assert b1["meta"]["chosen_model"] == b2["meta"]["chosen_model"]
