from types import SimpleNamespace

import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genx_kernel.broker.app import main
from genx_kernel.broker.app.main import ChatPayload, ChatRequest


def test_contract_v1_response_contains_contract_and_markdown(monkeypatch):
    monkeypatch.setattr(main, "_ollama_chat_openai", lambda **kwargs: "plain output")

    req = ChatRequest(
        actor="tester",
        payload=ChatPayload(
            thread_id="th_1",
            messages=[{"role": "user", "content": "hello"}],
            model="mistral:latest",
            output_format="contract_v1",
            tags=["a"],
        ),
    )
    request = SimpleNamespace(state=SimpleNamespace(actor="tester"))

    resp = asyncio.run(main.v1_chat(req, request))

    assert resp["assistant"]["role"] == "assistant"
    assert isinstance(resp["assistant"]["content"], str)
    assert resp["assistant"]["content"].startswith("## ")
    assert "assistant_contract" in resp
    tags = resp["assistant_contract"]["payload"]["signals"]["tags"]
    assert len(tags) == 3
    assert tags == ["a", "genx", "brokered"]


def test_legacy_default_response_unchanged_without_contract(monkeypatch):
    monkeypatch.setattr(main, "_ollama_chat_openai", lambda **kwargs: "raw model text")

    req = ChatRequest(
        actor="tester",
        payload=ChatPayload(
            thread_id="th_legacy",
            messages=[{"role": "user", "content": "hello"}],
            model="mistral:latest",
        ),
    )
    request = SimpleNamespace(state=SimpleNamespace(actor="tester"))

    resp = asyncio.run(main.v1_chat(req, request))

    assert resp["assistant"]["content"] == "raw model text"
    assert "assistant_contract" not in resp
