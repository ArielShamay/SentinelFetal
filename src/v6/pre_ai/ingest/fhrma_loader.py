"""FHRMA / FSdataset loader (V6 standardized ingest)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np

from src.v6.pre_ai.quality_policy import load_quality_policy
from .gap_fill import fill_small_gaps
from .schema import StandardizedRecord, compute_record_quality

COLUMN_MAP = {
    "fhr_candidates": ["FHR", "fhr", "fetal_hr", "fetalheartrate", "heart_rate"],
    "uc_candidates": ["UC", "uc", "toco", "uterine", "ua", "contraction"],
    "time_candidates": ["t", "time", "timestamp", "seconds", "sec"],
    "label_candidates": ["label", "quality", "noise", "class", "annotation"],
}


@dataclass
class LoadResult:
    record: StandardizedRecord | None
    skip_reason: str | None
    details: dict | None = None


_MIN_SAMPLES = int(4 * 60 * 20)
_PLAUSIBLE_MIN = 50.0
_PLAUSIBLE_MAX = 220.0


def _plausibility_stats(arr: np.ndarray) -> dict:
    finite = np.isfinite(arr)
    finite_frac = float(np.mean(finite)) if arr.size else 0.0
    clean = arr[finite]
    in_range = float(np.mean((clean >= _PLAUSIBLE_MIN) & (clean <= _PLAUSIBLE_MAX))) if clean.size else 0.0
    median = float(np.median(clean)) if clean.size else None
    jump_frac = float(np.mean(np.abs(np.diff(clean)) > 50.0)) if clean.size > 1 else 0.0
    return {
        "finite_frac": finite_frac,
        "in_range_frac": in_range,
        "median": median,
        "jump_frac": jump_frac,
        "samples": int(arr.size),
    }


def _is_plausible(stats: dict) -> bool:
    median = stats.get("median")
    if median is None:
        return False
    return (
        stats.get("in_range_frac", 0.0) >= 0.50
        and 60.0 <= float(median) <= 200.0
    )


def _infer_fs_hz(length: int) -> float | None:
    if length < _MIN_SAMPLES:
        return None
    duration_min = length / 4.0 / 60.0
    if 20.0 <= duration_min <= 360.0:
        return 4.0
    return None


def _decode_generic_binary(path: Path) -> tuple[np.ndarray | None, dict]:
    candidates = [
        ("int16_le", "<i2"),
        ("int16_be", ">i2"),
        ("float32_le", "<f4"),
        ("float32_be", ">f4"),
        ("uint16_le", "<u2"),
    ]
    candidate_stats = []
    for name, dtype in candidates:
        arr = np.fromfile(path, dtype=np.dtype(dtype))
        if arr.size < _MIN_SAMPLES:
            candidate_stats.append({"candidate": name, "reason": "too_short", "samples": int(arr.size)})
            continue
        stats = _plausibility_stats(arr.astype(float, copy=False))
        candidate_stats.append({"candidate": name, "stats": stats})
        if _is_plausible(stats):
            return arr.astype(float), {"decode_method": name, "stats": stats, "candidates": candidate_stats}
    return None, {"decode_method": None, "candidates": candidate_stats}


def _decode_structured_binary(path: Path) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    suffix = path.suffix.lower()
    try:
        with open(path, "rb") as f:
            np.fromfile(f, dtype=np.uint32, count=1)
            if suffix == ".dat":
                data = np.fromfile(f, dtype=np.uint16)
                if data.size < 2:
                    return None, None, {"decode_method": "structured_dat", "reason": "too_short"}
                data = data.reshape((2, -1), order="F")
                fhr = data[0] / 100.0
                uc = data[1] / 100.0
            elif suffix in {".fhr", ".rcf"}:
                data = np.fromfile(f, dtype=np.uint16)
                if data.size < 3:
                    return None, None, {"decode_method": "structured_fhr", "reason": "too_short"}
                data = data.reshape((3, -1), order="F")
                fhr = data[0] / 4.0
                f.seek(4, 0)
                data8 = np.fromfile(f, dtype=np.uint8)
                if data8.size < 6:
                    return None, None, {"decode_method": "structured_fhr", "reason": "too_short_uc"}
                data8 = data8.reshape((6, -1), order="F")
                uc = data8[4] / 2.0
            else:
                data = np.fromfile(f, dtype=np.uint16)
                if data.size < 4:
                    return None, None, {"decode_method": "structured_fhrm", "reason": "too_short"}
                data = data.reshape((4, -1), order="F")
                fhr = data[0] / 4.0
                f.seek(4, 0)
                data8 = np.fromfile(f, dtype=np.uint8)
                if data8.size < 8:
                    return None, None, {"decode_method": "structured_fhrm", "reason": "too_short_uc"}
                data8 = data8.reshape((8, -1), order="F")
                uc = data8[6] / 2.0
    except Exception as exc:
        return None, None, {"decode_method": "structured_fail", "error": str(exc)}

    stats = _plausibility_stats(np.asarray(fhr, dtype=float))
    if not _is_plausible(stats):
        return None, None, {"decode_method": "structured_fail", "stats": stats}
    return np.asarray(fhr, dtype=float), np.asarray(uc, dtype=float), {"decode_method": "structured_fhrma", "stats": stats}


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _select_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {_norm(col): col for col in columns}
    for cand in candidates:
        cand_norm = _norm(cand)
        if cand_norm in normalized:
            return normalized[cand_norm]
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


def discover_fhrma_records(root: Path) -> list[Path]:
    if not root.exists():
        return []
    patterns = ["*.csv", "*.CSV", "*.mat", "*.MAT", "*.fhr", "*.fhrm", "*.dat", "*.rcf", "*.rcfm"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted(set(files))


def _mat_candidate_arrays(mat: dict) -> list[dict]:
    candidates: list[dict] = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            continue
        if arr.size < 10:
            continue
        if arr.ndim == 1:
            candidates.append({"key": key, "array": arr})
        elif arr.ndim == 2:
            if 1 in arr.shape:
                candidates.append({"key": key, "array": arr.reshape(-1)})
            else:
                # Treat columns as separate candidates
                for idx in range(arr.shape[1]):
                    col = arr[:, idx]
                    candidates.append({"key": f"{key}[:,{idx}]", "array": col.reshape(-1)})
    return candidates


def _score_fhr_like(arr: np.ndarray) -> dict:
    finite = np.isfinite(arr)
    finite_frac = float(np.mean(finite)) if arr.size else 0.0
    clean = arr[finite]
    in_range = float(np.mean((clean >= 50.0) & (clean <= 210.0))) if clean.size else 0.0
    median = float(np.median(clean)) if clean.size else None
    return {
        "finite_frac": finite_frac,
        "in_range_frac": in_range,
        "median": median,
    }


def _find_fs_from_mat(mat: dict) -> float | None:
    for key in ("fs", "fs_hz", "sampling_rate", "sample_rate", "hz"):
        if key in mat:
            try:
                val = float(np.asarray(mat[key]).reshape(-1)[0])
                if val > 0:
                    return val
            except Exception:
                continue
    return None


def _extract_label_from_mat(mat: dict) -> dict:
    for key in ("label", "quality", "noise", "class", "annotation"):
        if key in mat:
            try:
                value = mat[key]
                if isinstance(value, np.ndarray) and value.size:
                    value = value.reshape(-1)[0]
                return {"quality_label": value}
            except Exception:
                return {"quality_label": None}
    return {}


def load_fhrma_record(csv_path: Path) -> LoadResult:
    policy = load_quality_policy()
    interp = policy.get("interpolation", {})
    max_gap = int(interp.get("max_gap_samples_to_fill", 10))
    method = str(interp.get("fill_method", "linear"))
    do_not_fill = bool(interp.get("do_not_fill_if_gap_too_large", True))

    if not csv_path.exists():
        return LoadResult(None, "FILE_NOT_FOUND")

    suffix = csv_path.suffix.lower()
    details: dict = {}
    if suffix in {".fhr", ".fhrm", ".rcf", ".rcfm", ".dat"}:
        fhr = None
        uc = None
        labels = {}
        time_col = None
        label_col = None

        fhr_candidate, cand_details = _decode_generic_binary(csv_path)
        if fhr_candidate is not None:
            fs_in = _infer_fs_hz(len(fhr_candidate))
            if fs_in is None:
                return LoadResult(None, "FS_UNKNOWN", details=cand_details)
            fhr = fhr_candidate
            uc = np.full_like(fhr, np.nan, dtype=float)
            fhr_col = cand_details.get("decode_method", "binary")
            uc_col = "missing"
            details.update(cand_details)
            details["fs_inferred"] = True
            details["uc_missing"] = True
        else:
            fhr_s, uc_s, struct_details = _decode_structured_binary(csv_path)
            details.update(cand_details)
            details.update(struct_details)
            if fhr_s is None or uc_s is None:
                return LoadResult(None, "BINARY_DECODE_FAIL", details=details)
            fhr = fhr_s
            uc = uc_s
            fs_in = 4.0
            fhr_col = struct_details.get("decode_method", "structured")
            uc_col = "structured_uc"
            details["fs_inferred"] = False
            details["uc_missing"] = False
    elif suffix == ".mat":
        try:
            from scipy.io import loadmat
        except Exception as exc:
            return LoadResult(None, "SCIPY_NOT_AVAILABLE", details={"error": str(exc)})

        try:
            mat = loadmat(csv_path)
        except Exception as exc:
            return LoadResult(None, f"MAT_READ_FAIL: {exc}")

        fs_in = _find_fs_from_mat(mat) or 4.0
        label = _extract_label_from_mat(mat)
        candidates = _mat_candidate_arrays(mat)
        if not candidates:
            keys = {k: str(np.asarray(v).shape) for k, v in mat.items() if not k.startswith("__")}
            return LoadResult(None, "MAT_KEYS_UNKNOWN", details={"keys": keys})

        scored = []
        for cand in candidates:
            stats = _score_fhr_like(cand["array"])
            scored.append({**cand, **stats})
        scored.sort(key=lambda x: (x["in_range_frac"], x["finite_frac"]), reverse=True)
        fhr_candidate = scored[0] if scored else None
        if fhr_candidate is None:
            keys = {k: str(np.asarray(v).shape) for k, v in mat.items() if not k.startswith("__")}
            return LoadResult(None, "MAT_KEYS_UNKNOWN", details={"keys": keys})

        fhr = np.asarray(fhr_candidate["array"], dtype=float).reshape(-1)
        # Find UC candidate with same length
        uc = None
        for cand in scored[1:]:
            arr = np.asarray(cand["array"], dtype=float).reshape(-1)
            if len(arr) == len(fhr):
                uc = arr
                break
        if uc is None:
            keys = {k: str(np.asarray(v).shape) for k, v in mat.items() if not k.startswith("__")}
            return LoadResult(None, "UC_COLUMN_MISSING", details={"keys": keys})

        fhr_col = "MAT"
        uc_col = "MAT"
        time_col = None
        label_col = None
        labels = label
        details["decode_method"] = "mat"
    else:
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except Exception:
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, sep=r"\s+", engine="python")
            except Exception as exc:
                return LoadResult(None, f"CSV_READ_FAIL: {exc}")

        if df.empty:
            return LoadResult(None, "EMPTY_FILE")

        fhr_col = _select_column(df.columns, COLUMN_MAP["fhr_candidates"])
        uc_col = _select_column(df.columns, COLUMN_MAP["uc_candidates"])
        time_col = _select_column(df.columns, COLUMN_MAP["time_candidates"])
        label_col = _select_column(df.columns, COLUMN_MAP["label_candidates"])

        if fhr_col is None:
            return LoadResult(None, "FHR_COLUMN_MISSING")
        if uc_col is None:
            return LoadResult(None, "UC_COLUMN_MISSING")

        fhr_series = df[fhr_col]
        uc_series = df[uc_col]
        try:
            import pandas as pd
            fhr = pd.to_numeric(fhr_series, errors="coerce").to_numpy(dtype=float)
            uc = pd.to_numeric(uc_series, errors="coerce").to_numpy(dtype=float)
        except Exception:
            fhr = fhr_series.to_numpy(dtype=float, copy=False)
            uc = uc_series.to_numpy(dtype=float, copy=False)

        labels = {}
        if label_col is not None:
            try:
                labels["quality_label"] = df[label_col].iloc[0]
            except Exception:
                labels["quality_label"] = None

        fs_in = 4.0
        details["decode_method"] = "csv"

    if len(fhr) != len(uc):
        return LoadResult(None, "ALIGNMENT_FAIL")

    time_unit = None
    if suffix != ".mat" and time_col is not None:
        t_raw = df[time_col].to_numpy(dtype=float, copy=False)
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
        "selected_fhr_column": fhr_col if suffix != ".mat" else "MAT",
        "selected_uc_column": uc_col if suffix != ".mat" else "MAT",
        "selected_time_column": time_col if suffix != ".mat" else None,
        "selected_label_column": label_col if suffix != ".mat" else None,
        "original_fs": fs_in,
        "time_unit": time_unit,
        "original_length": int(len(fhr)),
        "resampled": fs_in != fs_out,
        "decode_method": details.get("decode_method"),
        "decode_stats": details.get("stats"),
        "fs_inferred": details.get("fs_inferred"),
        "uc_missing": details.get("uc_missing"),
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
        source="FHRMA",
        labels=labels,
        meta=meta,
        record_quality=record_quality,
    )
    return LoadResult(record_obj, None, details=details)
