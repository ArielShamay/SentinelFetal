#!/usr/bin/env python
"""Smoke test: Pre-AI imports only, no heavy ML modules."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("error")

# Minimal pre-AI imports
from src.utils.runtime_config import load_runtime_config  # noqa: F401
from src.v6.pre_ai.invariants import WarmupError, assert_raw_invariants  # noqa: F401
from src.v6.pre_ai.quality_gate import quality_gate  # noqa: F401
from src.v6.pre_ai.windowing import window_iter  # noqa: F401

for blocked in ("torch", "tensorflow", "momentfm"):
    if blocked in sys.modules:
        raise RuntimeError(f"IMPORT_POLLUTION: {blocked} loaded in sys.modules")

print("SMOKE_IMPORT_PRE_AI_OK")
