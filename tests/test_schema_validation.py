import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "SCHEMAS" / "daily_log.schema.json"
MIN_PATH = ROOT / "EXAMPLES" / "daily_log_event.min.json"
FULL_PATH = ROOT / "EXAMPLES" / "daily_log_event.full.json"
CLI_PATH = ROOT / "bin" / "validate_daily_log"

HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None

if HAS_JSONSCHEMA:
    from jsonschema import Draft202012Validator


def _load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema dependency not installed")
def test_examples_are_valid() -> None:
    validator = Draft202012Validator(_load(SCHEMA_PATH))
    for path in (MIN_PATH, FULL_PATH):
        errors = list(validator.iter_errors(_load(path)))
        assert not errors, f"Expected valid payload for {path}, got: {[e.message for e in errors]}"


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema dependency not installed")
def test_broken_payload_fails() -> None:
    validator = Draft202012Validator(_load(SCHEMA_PATH))
    payload = copy.deepcopy(_load(MIN_PATH))

    payload["log"]["signal_tags"]["tags"] = ["only", "two"]
    payload["log"]["title"]["title_text"] = "X" * 60
    payload.pop("truth_planes")

    errors = list(validator.iter_errors(payload))
    assert errors, "Expected validation errors for broken payload"

    messages = "\n".join(e.message for e in errors)
    assert "too long" in messages
    assert "too short" in messages
    assert "truth_planes" in messages


def test_cli_fail_closed_when_dependency_missing() -> None:
    result = subprocess.run(
        [str(CLI_PATH), str(MIN_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )

    if HAS_JSONSCHEMA:
        assert result.returncode == 0, result.stderr + result.stdout
        assert "VALID:" in result.stdout
    else:
        assert result.returncode != 0
        assert "missing dependency 'jsonschema'" in result.stderr
