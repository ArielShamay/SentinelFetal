"""Calibrate smart logic thresholds with persistence guardrails (V5)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
VAL_DEFAULT = PROJECT_ROOT / "models" / "ensemble_v5" / "validation_preds_v5.csv"
ENSEMBLE_CFG_DEFAULT = PROJECT_ROOT / "config" / "ensemble_v5_optuna.yaml"
OUT_PATH = PROJECT_ROOT / "config" / "smart_logic_v5_thresholds.yaml"


@dataclass
class Metrics:
    recall: float
    precision: float
    f2: float
    alert_rate: float


def load_weights(path: Path) -> Tuple[float, float, float]:
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    w = cfg.get("weights", {}) if isinstance(cfg, dict) else {}
    return (
        float(w.get("xgboost", 1.0)),
        float(w.get("random_forest", 0.0)),
        float(w.get("sgd_classifier", 0.0)),
    )


def load_windows(path: Path):
    data: Dict[str, Dict] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id", row.get("pid", "p"))
            if pid not in data:
                data[pid] = {"label": int(row.get("true_label", row.get("label", 0))), "probs": []}
            data[pid]["probs"].append({
                "xgb": float(row.get("xgb_prob", 0.0)),
                "rf": float(row.get("rf_prob", 0.0)),
                "sgd": float(row.get("sgd_prob", 0.0)),
            })
    return data


def has_persistent_alert(probs: List[float], thr: float, k: int, n: int) -> bool:
    if len(probs) < n:
        return False
    for i in range(0, len(probs) - n + 1):
        window = probs[i:i + n]
        if sum(1 for p in window if p >= thr) >= k:
            return True
    return False


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Metrics:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if precision > 0 and recall > 0 else 0.0
    alert_rate = (tp + fp) / len(y_true) if y_true else 0.0
    return Metrics(recall, precision, f2, alert_rate)


def main(args):
    if not args.val.exists():
        raise FileNotFoundError(f"Missing validation file: {args.val}")

    weights = load_weights(args.ensemble_cfg)
    data = load_windows(args.val)

    k = args.k
    n = args.n

    grid = np.arange(args.t_min, args.t_max + 1e-9, args.t_step)

    best = None
    best_thr = None
    best_metrics = None

    insufficient_windows = sum(1 for rec in data.values() if len(rec["probs"]) < n)

    # Baseline alert-rate constraint (AI-only persistence at ensemble threshold)
    baseline_alert_rate = None
    if args.max_alert_rate is None:
        y_true_base = []
        y_pred_base = []
        for pid, rec in data.items():
            if len(rec["probs"]) < n:
                continue
            wx, wr, ws = weights
            probs = [wx * p["xgb"] + wr * p["rf"] + ws * p["sgd"] for p in rec["probs"]]
            y_true_base.append(rec["label"])
            y_pred_base.append(1 if has_persistent_alert(probs, args.ensemble_threshold, k, n) else 0)
        baseline_alert_rate = compute_metrics(y_true_base, y_pred_base).alert_rate if y_true_base else 1.0
        max_alert_rate = baseline_alert_rate
    else:
        max_alert_rate = args.max_alert_rate

    for thr in grid:
        y_true = []
        y_pred = []
        for pid, rec in data.items():
            if len(rec["probs"]) < n:
                continue
            wx, wr, ws = weights
            probs = [wx * p["xgb"] + wr * p["rf"] + ws * p["sgd"] for p in rec["probs"]]
            y_true.append(rec["label"])
            y_pred.append(1 if has_persistent_alert(probs, thr, k, n) else 0)

        metrics = compute_metrics(y_true, y_pred)
        if metrics.recall < args.recall_guardrail:
            continue
        if metrics.alert_rate > max_alert_rate:
            continue
        score = metrics.f2 - args.alert_penalty * metrics.alert_rate
        if best is None or score > best:
            best = score
            best_thr = float(thr)
            best_metrics = metrics

    if best_thr is None:
        raise RuntimeError("No threshold met recall guardrail; try lowering guardrail")

    t_high = best_thr
    t_low = max(0.0, t_high - 0.10)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "t_high": t_high,
                "t_low": t_low,
                "persistence_k": k,
                "persistence_n": n,
                "persistence_window_minutes": args.persistence_window_minutes,
                "recall_guardrail": args.recall_guardrail,
                "insufficient_windows": insufficient_windows,
            },
            f,
            sort_keys=False,
        )

    print("Calibration report")
    print(f"t_high: {t_high:.2f}")
    print(f"t_low: {t_low:.2f}")
    print(f"Recall: {best_metrics.recall:.3f}")
    print(f"Precision: {best_metrics.precision:.3f}")
    print(f"AlertRate: {best_metrics.alert_rate:.3f}")
    print(f"F2: {best_metrics.f2:.3f}")
    if baseline_alert_rate is not None:
        print(f"Baseline alert rate (AI-only @ {args.ensemble_threshold:.2f}): {baseline_alert_rate:.3f}")
    print(f"Max alert rate constraint: {max_alert_rate:.3f}")
    print(f"Insufficient windows (<{n}): {insufficient_windows}")
    print(f"Saved thresholds to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=Path, default=VAL_DEFAULT)
    parser.add_argument("--ensemble_cfg", type=Path, default=ENSEMBLE_CFG_DEFAULT)
    parser.add_argument("--t_min", type=float, default=0.20)
    parser.add_argument("--t_max", type=float, default=0.80)
    parser.add_argument("--t_step", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--persistence_window_minutes", type=int, default=15)
    parser.add_argument("--recall_guardrail", type=float, default=0.85)
    parser.add_argument("--alert_penalty", type=float, default=0.5)
    parser.add_argument("--ensemble_threshold", type=float, default=0.3844)
    parser.add_argument("--max_alert_rate", type=float, default=None)
    args = parser.parse_args()
    main(args)
