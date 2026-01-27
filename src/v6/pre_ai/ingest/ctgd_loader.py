"""CTGDL CSV loader (V6 standardized ingest)."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np

from src.v6.pre_ai.quality_policy import load_quality_policy
from .gap_fill import fill_small_gaps
from .schema import StandardizedRecord, compute_record_quality

COLUMN_MAP = {
    "fhr_candidates": [
        "FHR", "fhr", "FetalHeartRate", "Fetal_Heart_Rate", "heart_rate",
        "fhr_1", "fhr1", "fetal_heart", "fetalheart", "fetalheartrate",
        "signal_0", "signal0", "trace_1", "trace1", "channel_0", "channel0",
        "hr", "heartrate", "bpm", "bpm_fhr", "fh", "fetal", "signal(0)", "trace(1)",
    ],
    "uc_candidates": [
        "UC", "uc", "UterineContraction", "Uterine_Contraction", "toco", "UA",
        "uc_1", "uc1", "uterine_activity", "uterineactivity", "toco_1", "ua_1",
        "pressure", "contractions", "cont",
    ],
    "time_candidates": ["t", "time", "timestamp", "seconds", "sec"],
}


@dataclass
class LoadResult:
    record: StandardizedRecord | None
    skip_reason: str | None
    suggestions: dict | None = None


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _select_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {_norm(col): col for col in columns}
    for cand in candidates:
        cand_norm = _norm(cand)
        if cand_norm in normalized:
            return normalized[cand_norm]
    return None


def _numeric_column_stats(df) -> list[dict]:
    stats = []
    for col in df.columns:
        try:
            series = df[col]
            values = series.to_numpy()
            values = values.astype(float)
        except Exception:
            try:
                import pandas as pd
                values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            except Exception:
                continue
        finite = np.isfinite(values)
        finite_frac = float(np.mean(finite)) if values.size else 0.0
        if finite_frac == 0.0:
            continue
        clean = values[finite]
        in_range = np.mean((clean >= 60.0) & (clean <= 200.0)) if clean.size else 0.0
        stats.append({
            "column": col,
            "finite_frac": finite_frac,
            "in_range_frac": float(in_range),
            "min": float(np.min(clean)) if clean.size else None,
            "median": float(np.median(clean)) if clean.size else None,
            "max": float(np.max(clean)) if clean.size else None,
        })
    stats.sort(key=lambda x: (x["in_range_frac"], x["finite_frac"]), reverse=True)
    return stats


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


def load_ctgdl_record(csv_path: Path) -> LoadResult:
    policy = load_quality_policy()
    interp = policy.get("interpolation", {})
    max_gap = int(interp.get("max_gap_samples_to_fill", 10))
    method = str(interp.get("fill_method", "linear"))
    do_not_fill = bool(interp.get("do_not_fill_if_gap_too_large", True))

    if not csv_path.exists():
        return LoadResult(None, "FILE_NOT_FOUND")

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return LoadResult(None, f"CSV_READ_FAIL: {exc}")

    if df.empty:
        return LoadResult(None, "EMPTY_FILE")
    if len(df.index) < 1000:
        return LoadResult(None, "FILE_TOO_SHORT")

    fhr_col = _select_column(df.columns, COLUMN_MAP["fhr_candidates"])
    uc_col = _select_column(df.columns, COLUMN_MAP["uc_candidates"])
    time_col = _select_column(df.columns, COLUMN_MAP["time_candidates"])
    auto_selected = False

    if fhr_col is None:
        stats = _numeric_column_stats(df)
        suggestions = {
            "path": str(csv_path),
            "columns": [str(c) for c in df.columns],
            "top_candidates": stats[:5],
        }
        if stats:
            best = stats[0]
            if (
                best["in_range_frac"] >= 0.85
                and best["finite_frac"] >= 0.90
                and best["median"] is not None
                and 90.0 <= best["median"] <= 170.0
            ):
                fhr_col = best["column"]
                auto_selected = True
        if fhr_col is None:
            return LoadResult(None, "FHR_COLUMN_MISSING", suggestions=suggestions)

    if uc_col is None:
        return LoadResult(None, "UC_COLUMN_MISSING")

    fhr = pd.to_numeric(df[fhr_col], errors="coerce").to_numpy(dtype=float)
    uc = pd.to_numeric(df[uc_col], errors="coerce").to_numpy(dtype=float)

    if len(fhr) != len(uc):
        return LoadResult(None, "ALIGNMENT_FAIL")

    fs_in = 4.0
    time_unit = None
    if time_col is not None:
        t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        t_clean = t_raw[np.isfinite(t_raw)]
        if t_clean.size >= 2:
            diffs = np.diff(t_clean)
            diffs = diffs[diffs > 0]
            if diffs.size:
                dt = float(np.median(diffs))
                if dt > 10.0:
                    dt = dt / 1000.0
                    time_unit = "ms"
                else:
                    time_unit = "s"
                if dt > 0:
                    fs_in = 1.0 / dt

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

    patient_id = csv_path.stem
    meta = {
        "source_path": str(csv_path),
        "selected_fhr_column": fhr_col,
        "selected_uc_column": uc_col,
        "selected_time_column": time_col,
        "auto_selected_fhr_column": auto_selected,
        "original_fs": fs_in,
        "time_unit": time_unit,
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
        source="CTGDL",
        labels={},
        meta=meta,
        record_quality=record_quality,
    )
    return LoadResult(record_obj, None, suggestions=None)
