"""
GENX Observatory Sample State Smoke Test
"""

from observatory.sample_state import load_observatory_state


def main():
    state = load_observatory_state()

    print("\n[GENX OBSERVATORY SAMPLE STATE SMOKE TEST]")
    print(state["system"])
    print(state["governance"])
    print(state["tools"])
    print(state["knowledge_graph"])
    print(state["missions"])


if __name__ == "__main__":
    main()
