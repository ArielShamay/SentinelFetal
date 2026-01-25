"""Trace generator and gauntlet signal lengths to root-cause short signals.

Runs generation entrypoint and gauntlet loader, asserts 30-minute durations
for both FHR and UC, and prints diagnostics plus stack traces on failure.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CTG
from src.synthetic.generate_gauntlet import run_patient, OUTPUT_DIR, DURATION_SEC
from src.synthetic.run_gauntlet_stress import load_gauntlet

EXPECTED_DURATION_SEC = 30 * 60
EXPECTED_FS = float(CTG.SAMPLING_RATE)
EXPECTED_SAMPLES = int(EXPECTED_DURATION_SEC * EXPECTED_FS)


def _assert_len(name: str, sig: List[float] | np.ndarray, fs: float, expected: int) -> None:
    if sig is None:
        raise ValueError(f"CRITICAL: {name} is None")
    n = len(sig)
    mins = n / fs / 60.0 if fs else float('inf')
    if n < expected:
        stack = "".join(traceback.format_stack())
        raise ValueError(
            f"CRITICAL: {name} too short: n={n} ({mins:.2f} min), expected={expected} (30.00 min)\nStack:\n{stack}"
        )


def trace_generation() -> Tuple[int, int]:
    print("=== Generation probe ===")
    trace = run_patient("trace_probe", [], duration_sec=DURATION_SEC)
    fhr = trace["fhr"]
    uc = trace["uc"]
    fs = float(trace.get("sampling_rate", EXPECTED_FS))
    print(f"generator duration_sec param: {DURATION_SEC}, fs: {fs}")
    print(f"returned lengths: fhr={len(fhr)}, uc={len(uc)}")
    print(f"computed duration: {len(fhr)/fs/60:.2f} min")
    if len(fhr) != len(uc):
        raise ValueError(f"CRITICAL: generator mismatch fhr={len(fhr)} uc={len(uc)}")
    _assert_len("FHR(gen)", fhr, fs, EXPECTED_SAMPLES)
    _assert_len("UC(gen)", uc, fs, EXPECTED_SAMPLES)
    return len(fhr), len(uc)


def _load_case_lengths() -> List[Tuple[str, int, int, float]]:
    df = load_gauntlet()
    lengths: List[Tuple[str, int, int, float]] = []
    for _, row in df.iterrows():
        fhr = json.loads(row["fhr"])
        uc = json.loads(row["uc"])
        fs = float(row.get("sampling_rate", EXPECTED_FS))
        lengths.append((row["case_id"], len(fhr), len(uc), fs))
    return lengths


def trace_gauntlet_rows() -> None:
    print("=== Gauntlet CSV probe ===")
    lengths = _load_case_lengths()
    fhr_samples = [l[1] for l in lengths]
    uc_samples = [l[2] for l in lengths]
    fs_values = [l[3] for l in lengths]
    min_fhr, mean_fhr, max_fhr = np.min(fhr_samples), np.mean(fhr_samples), np.max(fhr_samples)
    min_uc, mean_uc, max_uc = np.min(uc_samples), np.mean(uc_samples), np.max(uc_samples)
    print(f"Loaded {len(lengths)} cases from {OUTPUT_DIR}")
    print(f"FHR samples (min/mean/max): {min_fhr} / {mean_fhr:.1f} / {max_fhr}")
    print(f"UC  samples (min/mean/max): {min_uc} / {mean_uc:.1f} / {max_uc}")

    for case_id, fhr_n, uc_n, fs in lengths:
        if fhr_n != uc_n:
            stack = "".join(traceback.format_stack())
            raise ValueError(
                f"CRITICAL: length mismatch case={case_id} fhr={fhr_n} uc={uc_n} fs={fs}\nStack:\n{stack}"
            )
        _assert_len(f"FHR({case_id})", [0] * fhr_n, fs, EXPECTED_SAMPLES)
        _assert_len(f"UC({case_id})", [0] * uc_n, fs, EXPECTED_SAMPLES)


def trace() -> None:
    trace_generation()
    trace_gauntlet_rows()
    print("Trace complete: all signals >=30 minutes and lengths match.")


if __name__ == "__main__":
    trace()
