"""RAW quality gate for V6 Pre-AI pipeline (no AI dependencies)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.v6.pre_ai.invariants import WarmupError
from src.v6.pre_ai.quality_policy import load_quality_policy


def quality_gate(
    fhr_window: np.ndarray,
    uc_window: np.ndarray | None,
    fs_hz: float,
    min_window_minutes: float,
) -> Tuple[str, Dict]:
    """Compute quality on RAW windows only.

    Returns (quality_class, diagnostics).
    """
    policy = load_quality_policy()
    window_policy = policy.get("window", {})
    if fs_hz <= 0:
        raise ValueError(f"QUALITY | invalid fs_hz={fs_hz}")

    n = int(len(fhr_window))
    if n == 0:
        raise WarmupError("WARMUP_ERROR: empty window")

    min_samples = int(min_window_minutes * 60 * fs_hz)
    if n < min_samples:
        raise WarmupError(
            f"WARMUP_ERROR: window too short n={n} < {min_samples}"
        )

    if uc_window is not None and len(uc_window) != n:
        raise ValueError(
            f"QUALITY | FHR/UC length mismatch fhr={n} uc={len(uc_window)}"
        )

    window = np.asarray(fhr_window, dtype=float)
    finite = np.isfinite(window)
    valid_frac = float(np.mean(finite)) if window.size else 0.0
    nan_frac = float(1.0 - valid_frac)
    clean = window[finite]
    zeros_frac = float(np.mean((window == 0.0) & finite)) if window.size else 1.0

    isnan = ~finite
    max_nan_run = 0
    current_run = 0
    for flag in isnan:
        if flag:
            current_run += 1
            if current_run > max_nan_run:
                max_nan_run = current_run
        else:
            current_run = 0

    std = float(np.std(clean)) if clean.size else 0.0
    mad = float(np.median(np.abs(clean - np.median(clean)))) if clean.size else 0.0
    diffs = np.diff(clean) if clean.size > 1 else np.array([])
    abs_diffs = np.abs(diffs)
    jump_count_gt25 = int(np.sum(abs_diffs > 25.0)) if abs_diffs.size else 0
    max_abs_jump = float(np.max(abs_diffs)) if abs_diffs.size else 0.0
    flatline_ratio = float(np.mean(abs_diffs < 0.5)) if abs_diffs.size else 1.0
    out_of_range_frac = float(np.mean((clean < 60.0) | (clean > 220.0))) if clean.size else 1.0
    unique_ratio = float(len(np.unique(clean)) / len(clean)) if clean.size else 0.0

    hard_low = (
        nan_frac > float(window_policy.get("hard_low_nan_frac", 0.60))
        or max_nan_run >= int(window_policy.get("hard_low_max_nan_run", 200))
        or unique_ratio < float(window_policy.get("hard_low_unique_ratio", 0.05))
        or std < float(window_policy.get("hard_low_std_min", 1.0))
        or std > float(window_policy.get("hard_low_std_max", 50.0))
        or out_of_range_frac > float(window_policy.get("hard_low_out_of_range_frac", 0.20))
    )

    if hard_low:
        quality_class = "LOW"
    elif (
        nan_frac > float(window_policy.get("med_nan_frac", 0.35))
        or zeros_frac > float(window_policy.get("med_zero_frac", 0.50))
        or flatline_ratio > float(window_policy.get("med_flatline_ratio", 0.90))
        or jump_count_gt25 > int(window_policy.get("med_jump_count_gt25", 10))
        or max_abs_jump > float(window_policy.get("med_max_abs_jump", 30.0))
    ):
        quality_class = "MED"
    else:
        quality_class = "HIGH"

    diag = {
        "valid_frac": valid_frac,
        "nan_frac": nan_frac,
        "zeros_frac": zeros_frac,
        "max_nan_run": max_nan_run,
        "std": std,
        "mad": mad,
        "max_abs_jump": max_abs_jump,
        "jump_count_gt25": jump_count_gt25,
        "unique_ratio": unique_ratio,
        "out_of_range_frac": out_of_range_frac,
        "flatline_ratio": flatline_ratio,
        "quality_class": quality_class,
        "hard_low": hard_low,
        "policy": {
            "hard_low_nan_frac": float(window_policy.get("hard_low_nan_frac", 0.60)),
            "hard_low_max_nan_run": int(window_policy.get("hard_low_max_nan_run", 200)),
            "hard_low_out_of_range_frac": float(window_policy.get("hard_low_out_of_range_frac", 0.20)),
            "med_nan_frac": float(window_policy.get("med_nan_frac", 0.35)),
            "med_zero_frac": float(window_policy.get("med_zero_frac", 0.50)),
            "med_flatline_ratio": float(window_policy.get("med_flatline_ratio", 0.90)),
            "med_jump_count_gt25": int(window_policy.get("med_jump_count_gt25", 10)),
            "med_max_abs_jump": float(window_policy.get("med_max_abs_jump", 30.0)),
        },
    }

    return quality_class, diag
