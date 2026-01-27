"""V6 Pre-AI pipeline modules (strict, no AI)."""

from src.v6.pre_ai.invariants import WarmupError, assert_raw_invariants
from src.v6.pre_ai.quality_gate import quality_gate
from src.v6.pre_ai.windowing import window_iter
from src.v6.pre_ai.pipeline import run_pre_ai

__all__ = [
    "WarmupError",
    "assert_raw_invariants",
    "quality_gate",
    "window_iter",
    "run_pre_ai",
]
