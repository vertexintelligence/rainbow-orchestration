"""
GENX Observatory State Loader

Loads live telemetry snapshot with fallback to sample state.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GENX_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if GENX_ROOT not in sys.path:
    sys.path.insert(0, GENX_ROOT)

from observatory.sample_state import load_observatory_state
from observatory.telemetry_snapshot import load_live_telemetry_snapshot


def load_dashboard_state() -> dict:
    """Load live telemetry snapshot, falling back to sample state on error."""
    try:
        return load_live_telemetry_snapshot()
    except Exception:
        return load_observatory_state()
