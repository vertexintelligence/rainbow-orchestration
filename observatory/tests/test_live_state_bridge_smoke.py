"""
GENX Live State Bridge Smoke Test
"""

import pytest

try:
    from observatory.telemetry_snapshot import load_live_telemetry_snapshot
    HAS_KNOWLEDGE = True
except ImportError:
    HAS_KNOWLEDGE = False


@pytest.mark.skipif(not HAS_KNOWLEDGE, reason="knowledge module not available")
def test_live_state_bridge_smoke():
    state = load_live_telemetry_snapshot()

    assert "system" in state
    assert "governance" in state
    assert "tools" in state
    assert "knowledge_graph" in state
    assert "missions" in state
    assert "alerts" in state


def main():
    if not HAS_KNOWLEDGE:
        print("SKIP: knowledge module not available")
        return

    state = load_live_telemetry_snapshot()

    print("\n[GENX LIVE STATE BRIDGE SMOKE TEST]")
    print(state["system"])
    print(state["governance"])
    print(state["tools"])
    print(state["knowledge_graph"])
    print(state["missions"])
    print(state["alerts"])


if __name__ == "__main__":
    main()
