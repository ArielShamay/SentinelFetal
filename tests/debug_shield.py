import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.smart_hybrid_logic import compute_signal_quality

NOISE_PATH = Path("data/synthetic_gauntlet/synthetic_noise_check.csv")


def load_case(case_id: str):
    df = pd.read_csv(NOISE_PATH)
    if case_id:
        row = df[df["case_id"] == case_id].iloc[0]
    else:
        row = df.iloc[0]
    fhr = np.array(json.loads(row["fhr"]), dtype=float)
    return row["case_id"], fhr


def raw_diagnostics(fhr: np.ndarray):
    finite = np.isfinite(fhr)
    valid_frac = float(np.mean(finite)) if fhr.size else 0.0
    zeros_frac = float(np.mean(fhr == 0)) if fhr.size else 0.0
    std = float(np.nanstd(fhr)) if fhr.size else 0.0
    diffs = np.diff(fhr)
    abs_diffs = np.abs(diffs)
    max_jump = float(np.nanmax(abs_diffs)) if abs_diffs.size else 0.0
    jump_count = int(np.sum(abs_diffs > 25.0)) if abs_diffs.size else 0
    flat_segments = int(np.sum(abs_diffs < 0.5)) if abs_diffs.size else 0
    return {
        "valid_frac": valid_frac,
        "zeros_frac": zeros_frac,
        "std": std,
        "max_abs_jump": max_jump,
        "jump_count_gt25": jump_count,
        "flat_segments": flat_segments,
        "nan_frac": 1.0 - valid_frac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=str, default=None, help="Case id from synthetic_noise_check.csv")
    args = parser.parse_args()

    cid, fhr = load_case(args.case_id)
    print(f"Loaded case: {cid}, length={len(fhr)}")

    diag = raw_diagnostics(fhr)
    print("RAW diagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    metrics = compute_signal_quality(fhr, raw=True, return_metrics=True)
    print("Shield metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
if __name__ == "__main__":
    main()
