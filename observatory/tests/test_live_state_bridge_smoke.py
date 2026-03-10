"""
GENX Live State Bridge Smoke Test
"""

from observatory.telemetry_snapshot import load_live_telemetry_snapshot


def main():
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
