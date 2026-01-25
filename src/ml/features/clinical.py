"""Clinical context and rule-proxy features for V5 Super-Features.

Provides deterministic, leakage-safe clinical meta-features and a lightweight
rule score proxy to inject clinical priors into the ML pipeline.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


def _fraction_valid(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sum(~np.isnan(x)) / x.size)


def _nan_median(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.nanmedian(x))


def _stv_proxy(fhr: np.ndarray) -> float:
    if fhr.size < 2:
        return 0.0
    diffs = np.diff(fhr)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return 0.0
    return float(np.mean(np.abs(diffs)))


def _fhr_std(fhr: np.ndarray) -> float:
    if fhr.size == 0:
        return 0.0
    return float(np.nanstd(fhr))


def _count_contractions(uc: Optional[np.ndarray], fs: float) -> int:
    if uc is None or uc.size == 0:
        return 0
    valid = uc[~np.isnan(uc)]
    if valid.size == 0:
        return 0
    baseline = np.nanmedian(valid)
    threshold = baseline + 10.0  # simple elevation heuristic
    above = np.where(uc > threshold, 1, 0)
    contractions = 0
    in_peak = False
    start_idx = 0
    min_width = int(10 * fs)  # require 10s elevation
    min_gap = int(30 * fs)    # enforce spacing to avoid overcounting
    last_end = -min_gap
    for i, val in enumerate(above):
        if val and not in_peak:
            in_peak = True
            start_idx = i
        elif not val and in_peak:
            in_peak = False
            if i - start_idx >= min_width and start_idx - last_end >= min_gap:
                contractions += 1
                last_end = i
    if in_peak and (len(above) - start_idx) >= min_width and start_idx - last_end >= min_gap:
        contractions += 1
    return contractions


def compute_rule_score(baseline_fhr: float, stv: float, fhr_std: float) -> int:
    """Heuristic rule proxy.

    Returns 0=normal, 1=suspicious, 2=pathological.
    """
    if baseline_fhr <= 110 or baseline_fhr >= 160 or stv < 2.0:
        return 2
    if baseline_fhr <= 120 or baseline_fhr >= 150 or stv < 5.0 or fhr_std < 5.0:
        return 1
    return 0


def compute_clinical_features(
    fhr_signal: np.ndarray,
    uc_signal: Optional[np.ndarray],
    fs: float,
    start_time_min: float,
) -> Dict[str, float]:
    """Compute clinical meta-features for a window.

    Returns a dictionary with 7 clinical meta-features. Handles NaNs gracefully
    and falls back to safe defaults when signal quality is too low.
    """
    fhr = np.asarray(fhr_signal, dtype=float)
    fhr[(fhr < 50) | (fhr > 210)] = np.nan
    quality = _fraction_valid(fhr)

    if quality < 0.10 or fhr.size == 0:
        return {
            "baseline_fhr": 0.0,
            "stv_proxy": 0.0,
            "fhr_std": 0.0,
            "uc_contractions": 0.0,
            "uc_rate": 0.0,
            "time_since_start_min": float(start_time_min),
            "signal_quality": quality,
        }

    baseline = _nan_median(fhr)
    stv = _stv_proxy(fhr)
    std = _fhr_std(fhr)

    contractions = _count_contractions(uc_signal, fs)
    duration_min = fhr.size / fs / 60.0 if fs > 0 else 0.0
    uc_rate = contractions / duration_min if duration_min > 0 else 0.0

    return {
        "baseline_fhr": baseline,
        "stv_proxy": stv,
        "fhr_std": std,
        "uc_contractions": float(contractions),
        "uc_rate": float(uc_rate),
        "time_since_start_min": float(start_time_min),
        "signal_quality": quality,
    }


__all__ = [
    "compute_clinical_features",
    "compute_rule_score",
]
