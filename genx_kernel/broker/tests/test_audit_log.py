import json
import os
from pathlib import Path

from genx_kernel.broker.app.routing.audit_log import (
    append_audit_event,
    build_audit_event,
)

def test_append_audit_event_writes_jsonl(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("GENX_AUDIT_LOG_PATH", str(log_path))

    ev = build_audit_event(
        policy_version="AUTO_MODEL_POLICY_V1.0",
        actor="rainbow",
        endpoint="chat",
        requested_model="auto",
        chosen_model="fast_primary",
        reason="AUTO_FAST",
        escalation=False,
        tools_disabled=False,
        decision_hash="abc123",
        max_tokens=100,
        temperature=0.2,
        message_count=2,
        has_system_prompt=True,
        risk_score=None,
        task_type="general",
    )

    append_audit_event(ev)

    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1

    obj = json.loads(lines[0])
    assert obj["event_version"] == "AUDIT_EVENT_V1"
    assert obj["policy_version"] == "AUTO_MODEL_POLICY_V1.0"
    assert obj["actor"] == "rainbow"
    assert obj["chosen_model"] == "fast_primary"
    assert obj["decision_hash"] == "abc123"
    assert "ts_unix_ms" in obj
