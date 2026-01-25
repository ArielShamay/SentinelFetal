"""Experiment A: Compare ensemble vs. XGBoost-solo under persistence logic.

Rules:
- Threshold comes from config/ensemble_v5_optuna.yaml (or --config override).
- Ensemble uses weights from config; XGB solo uses weight 1.0 on xgb_prob only.
- Persistence decision: alert if at least 2 windows exceed threshold.
- Winner: if XGB precision > Ensemble precision AND (Ensemble recall - XGB recall) < 0.03.
- Saves metrics and winner to optimization_v5_latest/engine_selection.json for Experiment B.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "ensemble_v5_optuna.yaml"
VAL_PATH = PROJECT_ROOT / "models" / "ensemble_v5" / "validation_preds_v5.csv"
OUT_PATH = PROJECT_ROOT / "optimization_v5_latest" / "engine_selection.json"


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


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
    for rec in data.values():
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


def decide_winner(rec_ens, rec_xgb):
    ens_recall, ens_prec, ens_f2, ens_alert = rec_ens
    xgb_recall, xgb_prec, xgb_f2, xgb_alert = rec_xgb
    recall_drop = ens_recall - xgb_recall
    if xgb_prec > ens_prec and recall_drop < 0.03:
        return "xgb_solo", {
            "recall_drop": recall_drop,
            "precision_gain": xgb_prec - ens_prec,
            "winner_metrics": {
                "recall": xgb_recall,
                "precision": xgb_prec,
                "f2": xgb_f2,
                "alert_rate": xgb_alert,
            },
        }
    return "ensemble", {
        "recall_drop": recall_drop,
        "precision_gain": xgb_prec - ens_prec,
        "winner_metrics": {
            "recall": ens_recall,
            "precision": ens_prec,
            "f2": ens_f2,
            "alert_rate": ens_alert,
        },
    }


def main(config_path: Path = DEFAULT_CONFIG):
    if not VAL_PATH.exists():
        print(f"Missing validation file: {VAL_PATH}")
        sys.exit(1)
    config = load_config(config_path)
    thr = float(config.get("threshold", 0.38))
    w = config.get("weights", {})
    ens_weights = (
        float(w.get("xgboost", 0.0)),
        float(w.get("random_forest", 0.0)),
        float(w.get("sgd_classifier", 0.0)),
    )

    _, labels, data = load_patient_windows(VAL_PATH)
    ens_preds = persistence_predict(thr, ens_weights, data)
    rec_ens = metrics(labels, ens_preds)

    xgb_preds = persistence_predict(thr, (1.0, 0.0, 0.0), data)
    rec_xgb = metrics(labels, xgb_preds)

    winner, info = decide_winner(rec_ens, rec_xgb)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(
            {
                "config_path": str(config_path),
                "threshold": thr,
                "ensemble_weights": ens_weights,
                "metrics": {
                    "ensemble": {
                        "recall": rec_ens[0],
                        "precision": rec_ens[1],
                        "f2": rec_ens[2],
                        "alert_rate": rec_ens[3],
                    },
                    "xgb_solo": {
                        "recall": rec_xgb[0],
                        "precision": rec_xgb[1],
                        "f2": rec_xgb[2],
                        "alert_rate": rec_xgb[3],
                    },
                },
                "winner": winner,
                "decision": info,
            },
            f,
            indent=2,
        )

    def fmt(rec):
        return f"recall={rec[0]:.3f}, precision={rec[1]:.3f}, f2={rec[2]:.3f}, alert_rate={rec[3]:.3f}"

    print(f"Config threshold={thr:.4f}, weights={ens_weights}")
    print(f"Ensemble: {fmt(rec_ens)}")
    print(f"XGB solo: {fmt(rec_xgb)}")
    print(f"Winner: {winner} (saved to {OUT_PATH})")


if __name__ == "__main__":
    cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    main(cfg)
