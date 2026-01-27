"""Optuna search for V5 persistence-mode ensemble weights/threshold (Golden Point).

Reads window-level probabilities (validation_preds_v5.csv) and optimizes patient-level
F2 under safety guardrails. Saves best config to config/ensemble_v5_optuna.yaml and
plots/summary to REPORTS/optimization_v5_<timestamp>/.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

VAL_PATH = PROJECT_ROOT / "models" / "ensemble_v5" / "validation_preds_v5.csv"
OUTPUT_DIR_BASE = PROJECT_ROOT / "REPORTS" / "optimization_v5_{}".format(datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
CONFIG_OUT = OUTPUT_DIR_BASE / "config" / "ensemble_v5_optuna.yaml"
REPORT_OUT = OUTPUT_DIR_BASE / "hybrid_validation_report_v5.md"  # placeholder for phase D
PLOT_RECALL_THR = OUTPUT_DIR_BASE / "recall_vs_threshold.png"
PLOT_PR = OUTPUT_DIR_BASE / "precision_recall.png"

THRESH_MIN = 0.25
THRESH_MAX = 0.85

WEIGHT_BOUNDS = {
    "xgb": (0.10, 0.90),
    "rf": (0.10, 0.90),
    "sgd": (0.00, 0.40),
}

np.random.seed(42)


def load_patient_windows(path: Path):
    data = defaultdict(lambda: {"label": None, "xgb": [], "rf": [], "sgd": []})
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            lbl = int(row["true_label"])
            rec = data[pid]
            if rec["label"] is None:
                rec["label"] = lbl
            rec["xgb"].append(float(row["xgb_prob"]))
            rec["rf"].append(float(row["rf_prob"]))
            rec["sgd"].append(float(row["sgd_prob"]))
    pids = list(data.keys())
    labels = [data[p]["label"] for p in pids]
    return pids, labels, data


def persistence_predict(thr: float, weights: Tuple[float, float, float], data) -> List[int]:
    wx, wr, ws = weights
    preds = []
    for pid, rec in data.items():
        probs = [wx * x + wr * r + ws * s for x, r, s in zip(rec["xgb"], rec["rf"], rec["sgd"])]
        count = sum(1 for p in probs if p > thr)
        preds.append(1 if count >= 2 else 0)
    return preds


def metrics(y_true: List[int], y_pred: List[int]):
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if precision > 0 and recall > 0 else 0.0
    alert_rate = (tp + fp) / len(y_true) if y_true else 0.0
    return recall, precision, f2, alert_rate


def objective_factory(y_true: List[int], data):
    def objective(trial: optuna.Trial):
        thr = trial.suggest_float("threshold", THRESH_MIN, THRESH_MAX)
        wx = trial.suggest_float("w_xgb", WEIGHT_BOUNDS["xgb"][0], WEIGHT_BOUNDS["xgb"][1])
        wr = trial.suggest_float("w_rf", WEIGHT_BOUNDS["rf"][0], WEIGHT_BOUNDS["rf"][1])
        ws = 1.0 - wx - wr
        # enforce weight constraints
        if ws < WEIGHT_BOUNDS["sgd"][0] or ws > WEIGHT_BOUNDS["sgd"][1]:
            return 0.0
        weights = (wx, wr, ws)
        y_pred = persistence_predict(thr, weights, data)
        recall, precision, f2, _ = metrics(y_true, y_pred)
        if precision < 0.30:
            return 0.0
        score = f2
        if recall < 0.75:
            score *= 0.1
        return score

    return objective


def evaluate_curve(weights: Tuple[float, float, float], y_true: List[int], data, thresholds: List[float]):
    recalls = []
    precisions = []
    for thr in thresholds:
        y_pred = persistence_predict(thr, weights, data)
        recall, precision, _, _ = metrics(y_true, y_pred)
        recalls.append(recall)
        precisions.append(precision)
    return recalls, precisions


def save_plots(weights: Tuple[float, float, float], y_true: List[int], data, out_recall: Path, out_pr: Path):
    thresholds = np.linspace(THRESH_MIN, THRESH_MAX, 50)
    recalls, precisions = evaluate_curve(weights, y_true, data, thresholds)

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, precisions, label="Precision")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Recall/Precision vs Threshold (Persistence)")
    plt.legend()
    plt.grid(True)
    out_recall.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_recall, bbox_inches="tight")
    plt.close()

    # Precision-Recall curve (by sweeping thresholds)
    plt.figure(figsize=(6, 4))
    plt.plot(recalls, precisions, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Persistence)")
    plt.grid(True)
    plt.savefig(out_pr, bbox_inches="tight")
    plt.close()


def main():
    if not VAL_PATH.exists():
        print(f"Missing validation file: {VAL_PATH}")
        sys.exit(1)

    pids, labels, data = load_patient_windows(VAL_PATH)

    OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_factory(labels, data), n_trials=100, show_progress_bar=False)

    best = study.best_trial.params
    thr = best["threshold"]
    wx = best["w_xgb"]
    wr = best["w_rf"]
    ws = 1.0 - wx - wr
    weights = (wx, wr, ws)

    y_pred = persistence_predict(thr, weights, data)
    recall, precision, f2, alert_rate = metrics(labels, y_pred)

    cfg = {
        "weights": {"xgboost": round(wx, 4), "random_forest": round(wr, 4), "sgd_classifier": round(ws, 4)},
        "threshold": round(thr, 4),
        "aggregation_mode": "persistence",
        "objective": "F2",
        "feature_spec": "v5_super_features",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study_best_value": study.best_value,
    }

    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_OUT.open("w") as f:
        yaml.safe_dump(cfg, f)

    save_plots(weights, labels, data, PLOT_RECALL_THR, PLOT_PR)

    summary = {
        "recall": recall,
        "precision": precision,
        "f2": f2,
        "alert_rate": alert_rate,
        "weights": cfg["weights"],
        "threshold": cfg["threshold"],
    }
    with (OUTPUT_DIR_BASE / "optuna_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("Best params:", cfg)
    print(f"Metrics: recall={recall:.3f}, precision={precision:.3f}, f2={f2:.3f}, alert_rate={alert_rate:.3f}")
    print(f"Config saved to {CONFIG_OUT}")
    print(f"Plots saved to {PLOT_RECALL_THR} and {PLOT_PR}")


if __name__ == "__main__":
    main()
