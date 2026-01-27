"""Small-gap interpolation utilities for Pre-AI ingest."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def fill_small_gaps(
    arr: np.ndarray,
    max_gap_samples: int,
    method: str = "linear",
) -> Tuple[np.ndarray, Dict[str, float]]:
    data = np.asarray(arr, dtype=float)
    filled = data.copy()

    if data.size == 0:
        stats = {
            "filled_gap_count": 0,
            "filled_samples": 0,
            "max_gap": 0,
            "total_nan_before": 0,
            "total_nan_after": 0,
        }
        return filled, stats

    isnan = ~np.isfinite(filled)
    total_nan_before = int(np.sum(isnan))

    max_gap = 0
    filled_gap_count = 0
    filled_samples = 0

    idx = 0
    n = len(filled)
    while idx < n:
        if not isnan[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and isnan[idx]:
            idx += 1
        end = idx
        gap_len = end - start
        max_gap = max(max_gap, gap_len)

        if gap_len <= max_gap_samples:
            left = start - 1
            right = end
            if left >= 0 and right < n and np.isfinite(filled[left]) and np.isfinite(filled[right]):
                if method != "linear":
                    raise ValueError(f"Unsupported fill method: {method}")
                y0 = filled[left]
                y1 = filled[right]
                step = (y1 - y0) / (gap_len + 1)
                for i in range(gap_len):
                    filled[start + i] = y0 + step * (i + 1)
                filled_gap_count += 1
                filled_samples += gap_len
        # else: leave NaNs as-is

    total_nan_after = int(np.sum(~np.isfinite(filled)))

    stats = {
        "filled_gap_count": filled_gap_count,
        "filled_samples": filled_samples,
        "max_gap": max_gap,
        "total_nan_before": total_nan_before,
        "total_nan_after": total_nan_after,
    }
    return filled, stats
