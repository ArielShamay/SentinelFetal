"""V6 windowing utilities (20m/5m at 4 Hz)."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from src.v6.pre_ai.invariants import WarmupError


def window_iter(
    fhr: np.ndarray,
    uc: np.ndarray,
    fs_hz: float,
    window_minutes: float,
    stride_minutes: float,
    min_window_minutes: float,
) -> Iterable[Tuple[int, int, np.ndarray, np.ndarray]]:
    window_samples = int(window_minutes * 60 * fs_hz)
    stride_samples = int(stride_minutes * 60 * fs_hz)
    min_window_samples = int(min_window_minutes * 60 * fs_hz)

    if window_samples < min_window_samples:
        raise ValueError(
            f"WINDOWING | window_minutes {window_minutes} < min_window_minutes {min_window_minutes}"
        )

    if len(fhr) < window_samples:
        raise WarmupError(
            f"WARMUP_ERROR: signal too short for window len={len(fhr)} < {window_samples}"
        )

    start = 0
    while start + window_samples <= len(fhr):
        end = start + window_samples
        if (end - start) < min_window_samples:
            raise WarmupError(
                f"WARMUP_ERROR: window too short start={start} end={end}"
            )
        yield start, end, fhr[start:end], uc[start:end]
        start += stride_samples
