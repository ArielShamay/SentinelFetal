"""Signal invariants for raw CTG data.

These checks are for RAW signals only. Do NOT use interpolated or cleaned data
as proof of validity.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def _to_1d_array(signal: Any) -> np.ndarray:
    arr = np.asarray(signal)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _preview_values(arr: np.ndarray) -> Tuple[list, list]:
    head = arr[:5].tolist()
    tail = arr[-5:].tolist() if len(arr) > 5 else arr.tolist()
    return head, tail


def assert_signal_length(signal: Any, fs: float, min_minutes: float, tag: str) -> None:
    if signal is None:
        raise ValueError(f"{tag} | signal is None")
    arr = _to_1d_array(signal)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{tag} | non-numeric signal dtype={arr.dtype}")

    n = int(len(arr))
    mins = n / fs / 60.0 if fs else float("inf")
    expected_min = int(min_minutes * 60 * fs)

    if n < expected_min:
        head, tail = _preview_values(arr)
        raise ValueError(
            f"{tag} | too short: n={n} ({mins:.2f} min), expected >= {min_minutes:.2f} min "
            f"(min_samples={expected_min}). head={head} tail={tail}"
        )


def assert_pair_aligned(fhr: Any, uc: Any, fs: float) -> None:
    if fhr is None or uc is None:
        raise ValueError("FHR/UC alignment check failed: one or both signals are None")

    fhr_arr = _to_1d_array(fhr)
    uc_arr = _to_1d_array(uc)

    if not np.issubdtype(fhr_arr.dtype, np.number) or not np.issubdtype(uc_arr.dtype, np.number):
        raise ValueError(
            f"FHR/UC alignment check failed: non-numeric dtype fhr={fhr_arr.dtype} uc={uc_arr.dtype}"
        )

    if len(fhr_arr) != len(uc_arr):
        fhr_min = len(fhr_arr) / fs / 60.0 if fs else float("inf")
        uc_min = len(uc_arr) / fs / 60.0 if fs else float("inf")
        raise ValueError(
            f"FHR/UC alignment check failed: len(fhr)={len(fhr_arr)} ({fhr_min:.2f} min) "
            f"len(uc)={len(uc_arr)} ({uc_min:.2f} min)"
        )
