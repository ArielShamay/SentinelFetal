"""RAW invariant checks for V6 Pre-AI pipeline."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


class WarmupError(RuntimeError):
    """Raised when attempting to process a window shorter than the minimum."""


def _to_1d_array(signal: Any) -> np.ndarray:
    arr = np.asarray(signal)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _preview(arr: np.ndarray) -> Tuple[list, list]:
    head = arr[:5].tolist()
    tail = arr[-5:].tolist() if len(arr) > 5 else arr.tolist()
    return head, tail


def assert_raw_invariants(
    fhr: Any,
    uc: Any,
    fs_hz: float,
    min_case_minutes: float,
    min_window_minutes: float,
    tag: str = "RAW",
) -> Tuple[np.ndarray, np.ndarray]:
    if fhr is None or uc is None:
        raise ValueError(f"{tag} | signal is None")

    fhr_arr = _to_1d_array(fhr)
    uc_arr = _to_1d_array(uc)

    if not np.issubdtype(fhr_arr.dtype, np.number):
        raise ValueError(f"{tag} | non-numeric FHR dtype={fhr_arr.dtype}")
    if not np.issubdtype(uc_arr.dtype, np.number):
        raise ValueError(f"{tag} | non-numeric UC dtype={uc_arr.dtype}")

    if len(fhr_arr) != len(uc_arr):
        raise ValueError(
            f"{tag} | FHR/UC length mismatch fhr={len(fhr_arr)} uc={len(uc_arr)}"
        )

    n = int(len(fhr_arr))
    if fs_hz <= 0:
        raise ValueError(f"{tag} | invalid fs_hz={fs_hz}")

    duration_min = n / fs_hz / 60.0
    if duration_min < min_window_minutes:
        raise WarmupError(
            f"WARMUP_ERROR: {tag} duration {duration_min:.2f} min < min_window {min_window_minutes}"
        )

    expected_min = int(min_case_minutes * 60 * fs_hz)
    if n < expected_min:
        head, tail = _preview(fhr_arr)
        raise ValueError(
            f"{tag} | too short: n={n} ({duration_min:.2f} min), "
            f"expected >= {min_case_minutes:.2f} min (min_samples={expected_min}). "
            f"head={head} tail={tail}"
        )

    return fhr_arr, uc_arr
