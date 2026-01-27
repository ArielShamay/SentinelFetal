"""CTU-CHB WFDB loader (V6 standardized ingest)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from src.v6.pre_ai.quality_policy import load_quality_policy
from .gap_fill import fill_small_gaps
from .schema import StandardizedRecord, compute_record_quality

FHR_CANDIDATES = {
    "fhr",
    "fhr1",
    "fetalheartrate",
    "fetalheartrate1",
    "fetal_heart_rate",
    "fetal heart rate",
    "fhr-1",
}

UC_CANDIDATES = {
    "uc",
    "toco",
    "ua",
    "uterinecontraction",
    "uterine_contraction",
    "uterine activity",
    "uterineactivity",
}


@dataclass
class LoadResult:
    record: StandardizedRecord | None
    skip_reason: str | None


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _select_channel(sig_names: Iterable[str], candidates: set[str]) -> int | None:
    normalized = [_norm(n) for n in sig_names]
    for cand in candidates:
        cand_norm = _norm(cand)
        for idx, name in enumerate(normalized):
            if name == cand_norm:
                return idx
    return None


def _resample_to_fs(signal: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return signal.astype(float)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("SCIPY_NOT_AVAILABLE") from exc

    ratio = Fraction(fs_out, fs_in).limit_denominator(1000)
    resampled = resample_poly(signal, ratio.numerator, ratio.denominator)
    expected = int(round(len(signal) * fs_out / fs_in))
    if len(resampled) != expected:
        raise RuntimeError(
            f"RESAMPLE_LEN_MISMATCH len={len(resampled)} expected={expected}"
        )
    return resampled.astype(float)


def discover_ctu_records(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.hea"))


def load_ctu_record(hea_path: Path) -> LoadResult:
    policy = load_quality_policy()
    interp = policy.get("interpolation", {})
    max_gap = int(interp.get("max_gap_samples_to_fill", 10))
    method = str(interp.get("fill_method", "linear"))
    do_not_fill = bool(interp.get("do_not_fill_if_gap_too_large", True))

    try:
        import wfdb
    except ImportError:
        return LoadResult(None, "WFDB_NOT_AVAILABLE")

    if not hea_path.exists():
        return LoadResult(None, "FILE_NOT_FOUND")

    record_path = hea_path.with_suffix("")
    try:
        record = wfdb.rdrecord(str(record_path))
    except Exception as exc:
        return LoadResult(None, f"WFDB_READ_FAIL: {exc}")

    sig_names = list(record.sig_name or [])
    fhr_idx = _select_channel(sig_names, FHR_CANDIDATES)
    uc_idx = _select_channel(sig_names, UC_CANDIDATES)

    if fhr_idx is None or uc_idx is None:
        return LoadResult(None, "CHANNELS_NOT_FOUND")

    try:
        raw = np.asarray(record.p_signal, dtype=float)
    except Exception as exc:
        return LoadResult(None, f"SIGNAL_READ_FAIL: {exc}")

    if raw.ndim != 2 or raw.shape[0] == 0:
        return LoadResult(None, "EMPTY_SIGNAL")

    fhr = raw[:, fhr_idx].reshape(-1)
    uc = raw[:, uc_idx].reshape(-1)

    if len(fhr) != len(uc):
        return LoadResult(None, "ALIGNMENT_FAIL")

    fs_in = float(getattr(record, "fs", 0.0) or 0.0)
    if fs_in <= 0:
        return LoadResult(None, "INVALID_FS")

    fs_out = 4.0
    try:
        fhr_rs = _resample_to_fs(fhr, fs_in, fs_out)
        uc_rs = _resample_to_fs(uc, fs_in, fs_out)
    except Exception as exc:
        return LoadResult(None, str(exc))

    if len(fhr_rs) != len(uc_rs):
        return LoadResult(None, "ALIGNMENT_FAIL")

    if not do_not_fill:
        max_gap = max(max_gap, len(fhr_rs))
    record_quality = compute_record_quality(fhr_rs, uc_rs, fs_out)
    fhr_filled, fhr_gap_stats = fill_small_gaps(fhr_rs, max_gap_samples=max_gap, method=method)
    uc_filled, uc_gap_stats = fill_small_gaps(uc_rs, max_gap_samples=max_gap, method=method)

    patient_id = hea_path.stem
    meta = {
        "source_path": str(hea_path),
        "original_fs": fs_in,
        "selected_fhr_channel": sig_names[fhr_idx],
        "selected_uc_channel": sig_names[uc_idx],
        "sig_names": sig_names,
        "original_length": int(len(fhr)),
        "resampled": fs_in != fs_out,
        "gap_fill": {
            "fhr": fhr_gap_stats,
            "uc": uc_gap_stats,
            "policy": {
                "max_gap_samples_to_fill": max_gap,
                "fill_method": method,
                "do_not_fill_if_gap_too_large": do_not_fill,
            },
        },
    }

    record_obj = StandardizedRecord(
        patient_id=patient_id,
        fhr_raw=fhr_rs,
        uc_raw=uc_rs,
        fhr_filled=fhr_filled,
        uc_filled=uc_filled,
        fs_hz=fs_out,
        source="CTU-CHB",
        labels={},
        meta=meta,
        record_quality=record_quality,
    )
    return LoadResult(record_obj, None)
