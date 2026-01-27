"""Validate Smart Hybrid Logic on real validation set.

Uses validation_preds_v5.csv (window-level) and applies persistence aggregation plus smart hybrid logic.
Outputs a metrics table comparing:
- AI-only persistence
- Old Hybrid (AI OR rule>=0.5)
- Smart Tiered Logic (t_high/t_low configurable)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.smart_hybrid_logic import SmartLogicConfig, evaluate, load_threshold_config, persistence_alert
from src.utils.runtime_config import load_runtime_config, apply_strict_warnings

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
VAL_PATH = PROJECT_ROOT / "models" / "ensemble_v5" / "validation_preds_v5.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "ensemble_v5_optuna.yaml"
LOGIC_CONFIG_PATH = PROJECT_ROOT / "config" / "logic_v5_2.yaml"
SMART_CFG_PATH = PROJECT_ROOT / "config" / "smart_logic_v5_thresholds.yaml"
RUNTIME_CFG = load_runtime_config()
SAFE_WARNING_ALLOWLIST = [
    # Add allowlisted warnings here if they are proven safe.
]


def load_config(path: Path) -> Tuple[float, Tuple[float, float, float]]:
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    thr = float(cfg.get("threshold", 0.5))
    w = cfg.get("weights", {})
    weights = (
        float(w.get("xgboost", 1.0)),
        float(w.get("random_forest", 0.0)),
        float(w.get("sgd_classifier", 0.0)),
    )
    return thr, weights


def calibrate_thresholds(path: Path, thr: float, weights: Tuple[float, float, float]) -> SmartLogicConfig:
    _, labels, data = load_patient_windows(path)
    neg_probs = []
    for pid, lbl in zip(data.keys(), labels):
        if lbl != 0:
            continue
        rec = data[pid]
        wx, wr, ws = weights
        probs = [wx * x + wr * r + ws * s for x, r, s in zip(rec["xgb"], rec["rf"], rec["sgd"])]
        neg_probs.extend(probs)
    if not neg_probs:
        raise RuntimeError("No negative windows found for calibration")
    t_high = float(np.percentile(neg_probs, 95))
    t_low = float(np.percentile(neg_probs, 75))
    cfg = SmartLogicConfig(t_high=t_high, t_low=t_low)
    LOGIC_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOGIC_CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump({
            "t_high": t_high,
            "t_low": t_low,
            "q_min": cfg.q_min,
            "suspicious_rule": cfg.suspicious_rule,
            "pathological_rule": cfg.pathological_rule,
            "source": "calibrated_from_validation_negatives",
        }, f, sort_keys=False)
    return cfg


def load_patient_windows(path: Path):
    data = defaultdict(lambda: {"label": None, "xgb": [], "rf": [], "sgd": [], "rule_score": []})
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id", row.get("pid", "p"))
            lbl = int(row.get("true_label", row.get("label", 0)))
            rec = data[pid]
            if rec["label"] is None:
                rec["label"] = lbl
            rec["xgb"].append(float(row["xgb_prob"]))
            rec["rf"].append(float(row.get("rf_prob", 0.0)))
            rec["sgd"].append(float(row.get("sgd_prob", 0.0)))
            if "rule_score" in row and row["rule_score"] not in (None, ""):
                rec["rule_score"].append(float(row["rule_score"]))
            else:
                rec["rule_score"].append(0.0)
    pids = list(data.keys())
    labels = [data[p]["label"] for p in pids]
    return pids, labels, data


def persistence_counts(thr: float, weights: Tuple[float, float, float], rec) -> Tuple[int, float, float]:
    wx, wr, ws = weights
    probs = [wx * x + wr * r + ws * s for x, r, s in zip(rec["xgb"], rec["rf"], rec["sgd"])]
    count = sum(1 for p in probs if p >= thr)
    max_prob = max(probs)
    rule_mean = float(np.mean(rec["rule_score"])) if rec.get("rule_score") else 0.0
    return count, max_prob, rule_mean


def metrics(y_true: List[int], y_pred: List[int]):
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if precision > 0 and recall > 0 else 0.0
    alert_rate = (tp + fp) / len(y_true) if y_true else 0.0
    fp_per_patient = (tp + fp - tp) / len(y_true) if y_true else 0.0
    return recall, precision, f2, alert_rate, fp_per_patient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=Path, default=VAL_PATH, help="Path to validation window CSV")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Ensemble weight/threshold config")
    parser.add_argument("--use_config", type=Path, default=SMART_CFG_PATH, help="Smart logic thresholds config")
    parser.add_argument("--t_high", type=float, default=None, help="High-tier AI threshold (optional)")
    parser.add_argument("--t_low", type=float, default=None, help="Low-tier AI threshold (optional)")
    parser.add_argument("--q_min", type=float, default=None, help="Quality minimum for AI tiers")
    parser.add_argument("--suspicious_rule", type=float, default=0.5, help="Rule threshold for tier2")
    parser.add_argument("--pathological_rule", type=float, default=0.8, help="Rule threshold for tier3")
    return parser.parse_args()


def main(args):
    apply_strict_warnings(RUNTIME_CFG.strict_mode, SAFE_WARNING_ALLOWLIST)
    print(f"STRICT_MODE: {RUNTIME_CFG.strict_mode}")
    print(f"Min window minutes: {RUNTIME_CFG.min_window_minutes}")
    if not args.val.exists():
        raise FileNotFoundError(f"Missing validation file: {args.val}")
    thr, weights = load_config(args.config)
    pids, labels, data = load_patient_windows(args.val)

    window_minutes = None
    if args.use_config and args.use_config.exists():
        cfg, k, n, window_minutes = load_threshold_config(args.use_config)
    else:
        cfg = calibrate_thresholds(args.val, thr, weights)
        k, n = 2, 3
    if window_minutes is not None and window_minutes < RUNTIME_CFG.min_window_minutes:
        raise RuntimeError(
            "STRICT_WINDOWING: persistence_window_minutes "
            f"{window_minutes} < min_window_minutes {RUNTIME_CFG.min_window_minutes}"
        )

    if args.t_high is not None:
        cfg.t_high = args.t_high
    if args.t_low is not None:
        cfg.t_low = args.t_low
    if args.q_min is not None:
        cfg.q_min = args.q_min
    cfg.suspicious_rule = args.suspicious_rule
    cfg.pathological_rule = args.pathological_rule

    ai_preds = []
    hybrid_preds = []
    smart_preds = []
    insufficient_windows = 0

    for pid in pids:
        rec = data[pid]
        wx, wr, ws = weights
        probs = [wx * x + wr * r + ws * s for x, r, s in zip(rec["xgb"], rec["rf"], rec["sgd"])]
        rule_scores = rec.get("rule_score", [0.0] * len(probs))

        if len(probs) < n:
            insufficient_windows += 1
            continue

        ai_flags = [p >= thr for p in probs]
        ai_pred = 1 if persistence_alert(ai_flags, k, n) else 0
        ai_preds.append(ai_pred)

        rule_mean = float(np.mean(rule_scores)) if rule_scores else 0.0
        hybrid_pred = 1 if (ai_pred == 1 or rule_mean >= 0.5) else 0
        hybrid_preds.append(hybrid_pred)

        sig_quality = 1.0  # validation windows assumed high quality
        smart_flags = []
        for p, r in zip(probs, rule_scores):
            smart_res = evaluate(p, r, cfg, quality_class="HIGH", signal_quality=sig_quality)
            smart_flags.append(smart_res.alert)
        smart_pred = 1 if persistence_alert(smart_flags, k, n) else 0
        smart_preds.append(smart_pred)

    ai_metrics = metrics(labels, ai_preds)
    hybrid_metrics = metrics(labels, hybrid_preds)
    smart_metrics = metrics(labels, smart_preds)

    header = ["Method", "Recall", "Precision", "Alert Rate", "F2", "FP/Patient", "Insufficient Windows"]
    rows = [
        ["AI-only (persistence)", *[f"{m:.3f}" for m in ai_metrics], str(insufficient_windows)],
        ["Old Hybrid (AI OR Rule)", *[f"{m:.3f}" for m in hybrid_metrics], str(insufficient_windows)],
        ["Smart Tiered Logic", *[f"{m:.3f}" for m in smart_metrics], str(insufficient_windows)],
    ]

    print("|" + "|".join(header) + "|")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        print("|" + "|".join(r) + "|")


if __name__ == "__main__":
    main(parse_args())
