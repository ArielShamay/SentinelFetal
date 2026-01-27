"""Standardized ingest schema and record-quality helpers for V6 Pre-AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from src.v6.pre_ai.quality_policy import load_quality_policy

FHR_MIN_BPM = 60.0
FHR_MAX_BPM = 220.0


@dataclass
class StandardizedRecord:
    patient_id: str
    fhr_raw: np.ndarray
    uc_raw: np.ndarray
    fs_hz: float
    source: str
    labels: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    record_quality: Dict[str, Any] = field(default_factory=dict)
    fhr_filled: Optional[np.ndarray] = None
    uc_filled: Optional[np.ndarray] = None


def _as_1d_float(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 1:
        out = out.reshape(-1)
    return out


def compute_record_quality(
    fhr_raw: Any,
    uc_raw: Any | None,
    fs_hz: float,
) -> Dict[str, Any]:
    fhr = _as_1d_float(fhr_raw)
    finite = np.isfinite(fhr)
    valid_frac = float(np.mean(finite)) if fhr.size else 0.0
    nan_frac = float(1.0 - valid_frac)
    zeros_frac = float(np.mean((fhr == 0.0) & finite)) if fhr.size else 1.0
    clean = fhr[finite]
    out_of_range_frac = (
        float(np.mean((clean < FHR_MIN_BPM) | (clean > FHR_MAX_BPM)))
        if clean.size
        else 1.0
    )
    duration_minutes = float(fhr.size / fs_hz / 60.0) if fs_hz > 0 else 0.0

    uc_valid_frac = None
    uc_nan_frac = None
    if uc_raw is not None:
        uc = _as_1d_float(uc_raw)
        uc_finite = np.isfinite(uc)
        uc_valid_frac = float(np.mean(uc_finite)) if uc.size else 0.0
        uc_nan_frac = float(1.0 - uc_valid_frac)

    flags = []
    if duration_minutes <= 0:
        flags.append("DURATION_INVALID")
    if nan_frac > 0.60:
        flags.append("FHR_NAN_FRAC_HIGH")
    if zeros_frac > 0.90:
        flags.append("FHR_ZERO_FRAC_HIGH")
    if out_of_range_frac > 0.20:
        flags.append("FHR_OUT_OF_RANGE_HIGH")

    record_quality = {
        "fhr_valid_frac": valid_frac,
        "fhr_nan_frac": nan_frac,
        "fhr_zeros_frac": zeros_frac,
        "fhr_out_of_range_frac": out_of_range_frac,
        "uc_valid_frac": uc_valid_frac,
        "uc_nan_frac": uc_nan_frac,
        "duration_minutes": duration_minutes,
        "flags": flags,
    }

    return record_quality


def record_reject_reason(record_quality: Dict[str, Any], policy: Dict[str, Any] | None = None) -> str | None:
    policy = policy or load_quality_policy()
    rec_policy = policy.get("record", {})

    nan_frac = float(record_quality.get("fhr_nan_frac", 1.0))
    zeros_frac = float(record_quality.get("fhr_zeros_frac", 1.0))
    out_of_range_frac = float(record_quality.get("fhr_out_of_range_frac", 1.0))
    duration = float(record_quality.get("duration_minutes", 0.0))

    if duration <= 0:
        return "DURATION_INVALID"
    if nan_frac > float(rec_policy.get("max_fhr_nan_frac_for_keep", 0.80)):
        return "FHR_NAN_FRAC"
    if out_of_range_frac > float(rec_policy.get("max_fhr_out_of_range_frac_for_keep", 0.60)):
        return "FHR_OUT_OF_RANGE"
    if zeros_frac > float(rec_policy.get("max_fhr_zero_frac_for_keep", 0.95)):
        return "FHR_ZERO_FRAC"
    return None


def record_is_reject(record_quality: Dict[str, Any], policy: Dict[str, Any] | None = None) -> bool:
    return record_reject_reason(record_quality, policy) is not None
