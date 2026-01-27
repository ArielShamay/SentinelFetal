"""Experiment B: Grid search for tiered hybrid logic (Smart Gate).

- Uses validation_preds_v5.csv and rule proxy (real rule_score if present, otherwise simulated 50% TP, ~60% precision).
- Engine selection: auto-read REPORTS/optimization_v5_latest/engine_selection.json (written by Experiment A).
  Fallback: run internal selection with same criteria.
- Tiered logic:
    Tier 1: if prob > T_high -> alert
    Tier 2: if T_low < prob <= T_high and rule suspicious -> alert
    Tier 3: if rule pathological -> alert
- Grid:
    T_high in [0.55..0.85] step 0.05
    T_low  in [0.25..0.50] step 0.05
    constraint: T_high > T_low
- Objective: recall >= 0.85, minimize alert_rate. Tie-break: higher recall, then higher F2.
- Outputs comparison table and saves logic_v5_1.yaml if pass (recall>=0.85 and alert_rate<=0.45).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
VAL_PATH = PROJECT_ROOT / "models" / "ensemble_v5" / "validation_preds_v5.csv"
CONFIG_DEFAULT = PROJECT_ROOT / "config" / "ensemble_v5_optuna.yaml"
ENGINE_SELECTION = PROJECT_ROOT / "REPORTS" / "optimization_v5_latest" / "engine_selection.json"
LOGIC_CONFIG_OUT = PROJECT_ROOT / "config" / "logic_v5_1.yaml"
TABLE_OUT = PROJECT_ROOT / "REPORTS" / "optimization_v5_latest" / "logic_comparison_table.txt"

np.random.seed(42)


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_patient_windows(path: Path):
    data = defaultdict(lambda: {"label": None, "xgb": [], "rf": [], "sgd": [], "rule_score": []})
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
            if "rule_score" in row and row["rule_score"] not in (None, ""):
                rec["rule_score"].append(float(row["rule_score"]))
    pids = list(data.keys())
    labels = [data[p]["label"] for p in pids]
    return pids, labels, data


def persistence_probabilities(weights: Tuple[float, float, float], data) -> List[float]:
    wx, wr, ws = weights
    # Aggregate patient-level probability as max of weighted window probabilities.
    agg = []
    for rec in data.values():
        probs = [wx * x + wr * r + ws * s for x, r, s in zip(rec["xgb"], rec["rf"], rec["sgd"])]
        agg.append(max(probs))
    return agg


def persistence_alerts(thr: float, weights: Tuple[float, float, float], data) -> List[int]:
    wx, wr, ws = weights
    preds = []
    for rec in data.values():
        probs = [wx * x + wr * r + ws * s for x, r, s in zip(rec["xgb"], rec["rf"], rec["sgd"])]
        count = sum(1 for p in probs if p > thr)
        preds.append(1 if count >= 2 else 0)
    return preds


def simulate_rule_flags(labels: List[int], desired_tp_rate: float = 0.5, desired_precision: float = 0.6):
    labels_arr = np.array(labels)
    n_pos = int(labels_arr.sum())
    n_neg = len(labels_arr) - n_pos

    tp_expected = int(round(n_pos * desired_tp_rate))
    fp_expected = int(round(tp_expected * (1 / desired_precision - 1))) if desired_precision > 0 else 0

    p_pos = desired_tp_rate
    p_neg = fp_expected / n_neg if n_neg > 0 else 0.0
    p_neg = min(max(p_neg, 0.0), 1.0)

    rng = np.random.default_rng(42)
    suspicious = []
    pathological = []
    for lbl in labels_arr:
        if lbl == 1:
            susp = rng.random() < p_pos
            patho = susp and rng.random() < 0.3  # 30% of positive suspicious are pathological
        else:
            susp = rng.random() < p_neg
            patho = susp and rng.random() < 0.05  # rare pathological false alarms
        suspicious.append(1 if susp else 0)
        pathological.append(1 if patho else 0)
    return suspicious, pathological


def derive_rule_flags(data, labels):
    # If rule_score provided, use thresholds to map to flags. Else simulate.
    any_rule = any(rec["rule_score"] for rec in data.values())
    if not any_rule:
        return simulate_rule_flags(labels)

    suspicious = []
    pathological = []
    for rec in data.values():
        scores = rec["rule_score"]
        max_score = max(scores) if scores else 0.0
        susp = max_score >= 0.5
        patho = max_score >= 0.8
        suspicious.append(1 if susp else 0)
        pathological.append(1 if patho else 0)
    return suspicious, pathological


def metrics(y_true: List[int], y_pred: List[int]):
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if precision > 0 and recall > 0 else 0.0
    alert_rate = (tp + fp) / len(y_true) if y_true else 0.0
    return recall, precision, f2, alert_rate


def choose_engine(config_path: Path) -> Tuple[str, Tuple[float, float, float], float, Dict]:
    if ENGINE_SELECTION.exists():
        with ENGINE_SELECTION.open("r") as f:
            sel = json.load(f)
        winner = sel.get("winner", "ensemble")
        ens_w = sel.get("ensemble_weights")
        thr = float(sel.get("threshold", 0.38))
        if winner == "xgb_solo":
            return "xgb_solo", (1.0, 0.0, 0.0), thr, sel
        return "ensemble", tuple(ens_w), thr, sel

    # fallback: quick selection
    cfg = load_config(config_path)
    thr = float(cfg.get("threshold", 0.38))
    w = cfg.get("weights", {})
    ens_weights = (
        float(w.get("xgboost", 0.0)),
        float(w.get("random_forest", 0.0)),
        float(w.get("sgd_classifier", 0.0)),
    )
    _, labels, data = load_patient_windows(VAL_PATH)
    ens_preds = persistence_alerts(thr, ens_weights, data)
    xgb_preds = persistence_alerts(thr, (1.0, 0.0, 0.0), data)
    ens_rec = metrics(labels, ens_preds)
    xgb_rec = metrics(labels, xgb_preds)
    recall_drop = ens_rec[0] - xgb_rec[0]
    if xgb_rec[1] > ens_rec[1] and recall_drop < 0.03:
        return "xgb_solo", (1.0, 0.0, 0.0), thr, {}
    return "ensemble", ens_weights, thr, {}


def grid_search(engine_weights, patient_probs, labels, suspicious_flags, pathological_flags):
    best = None
    for t_high in np.arange(0.55, 0.851, 0.05):
        for t_low in np.arange(0.25, 0.501, 0.05):
            if t_high <= t_low:
                continue
            preds = []
            for prob, susp, patho in zip(patient_probs, suspicious_flags, pathological_flags):
                if patho:
                    preds.append(1)
                elif prob > t_high:
                    preds.append(1)
                elif prob > t_low and susp:
                    preds.append(1)
                else:
                    preds.append(0)
            rec = metrics(labels, preds)
            recall_ok = rec[0] >= 0.85
            if not best:
                best = (rec, t_high, t_low, recall_ok)
                continue
            best_rec, _, _, best_ok = best
            if recall_ok:
                if not best_ok:
                    best = (rec, t_high, t_low, recall_ok)
                else:
                    if rec[3] < best_rec[3] - 1e-9 or (
                        abs(rec[3] - best_rec[3]) < 1e-9 and (rec[0] > best_rec[0] or (abs(rec[0]-best_rec[0]) < 1e-9 and rec[2] > best_rec[2]))
                    ):
                        best = (rec, t_high, t_low, recall_ok)
            else:
                if not best_ok and rec[0] > best_rec[0]:
                    best = (rec, t_high, t_low, recall_ok)
    return best


def format_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def main(engine_override: str | None = None, config_path: Path = CONFIG_DEFAULT):
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Missing validation file: {VAL_PATH}")

    engine, weights, thr, sel_meta = choose_engine(config_path)
    if engine_override:
        engine = engine_override
        weights = (1.0, 0.0, 0.0) if engine_override == "xgb_solo" else weights

    _, labels, data = load_patient_windows(VAL_PATH)
    patient_probs = persistence_probabilities(weights, data)
    suspicious_flags, pathological_flags = derive_rule_flags(data, labels)

    # Baselines
    ai_alerts = persistence_alerts(thr, weights, data)
    baseline_hybrid = [1 if a or s else 0 for a, s in zip(ai_alerts, suspicious_flags)]
    base_metrics = metrics(labels, baseline_hybrid)

    xgb_or = persistence_alerts(thr, (1.0, 0.0, 0.0), data)
    xgb_hybrid = [1 if a or s else 0 for a, s in zip(xgb_or, suspicious_flags)]
    xgb_metrics = metrics(labels, xgb_hybrid)

    # Grid search
    best = grid_search(weights, patient_probs, labels, suspicious_flags, pathological_flags)
    if not best:
        raise RuntimeError("Grid search failed to evaluate.")
    best_rec, t_high, t_low, recall_ok = best

    # Pass/fail for production logic
    pass_logic = best_rec[0] >= 0.85 and best_rec[3] <= 0.45
    if pass_logic:
        LOGIC_CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
        with LOGIC_CONFIG_OUT.open("w") as f:
            yaml.safe_dump(
                {
                    "engine": engine,
                    "weights": {
                        "xgboost": weights[0],
                        "random_forest": weights[1],
                        "sgd_classifier": weights[2],
                    },
                    "threshold_base": thr,
                    "tiered_thresholds": {
                        "t_high": float(t_high),
                        "t_low": float(t_low),
                    },
                    "rule_assumptions": {
                        "suspicious": "rule_score>=0.5 or simulated TP50 precision60",
                        "pathological": "rule_score>=0.8 or simulated rare patho (30% of TP, 5% of FP)",
                    },
                    "metrics": {
                        "recall": best_rec[0],
                        "precision": best_rec[1],
                        "f2": best_rec[2],
                        "alert_rate": best_rec[3],
                    },
                },
                f,
                sort_keys=False,
            )

    # Reporting
    table = [
        [
            "Old Hybrid (Simple OR)",
            format_pct(base_metrics[0]),
            format_pct(base_metrics[1]),
            format_pct(base_metrics[3]),
            f"{base_metrics[2]:.3f}",
            "Baseline",
        ],
        [
            "XGBoost Solo (Simple OR)",
            format_pct(xgb_metrics[0]),
            format_pct(xgb_metrics[1]),
            format_pct(xgb_metrics[3]),
            f"{xgb_metrics[2]:.3f}",
            "Did it win Exp A?",
        ],
        [
            "Smart Tiered Logic (Best)",
            format_pct(best_rec[0]),
            format_pct(best_rec[1]),
            format_pct(best_rec[3]),
            f"{best_rec[2]:.3f}",
            f"T_high={t_high:.2f}, T_low={t_low:.2f}, engine={engine}",
        ],
    ]

    header = ["Method", "Recall", "Precision", "Alert Rate", "F2-Score", "Config / Notes"]
    lines = ["|" + "|".join(header) + "|"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in table:
        lines.append("|" + "|".join(row) + "|")

    table_md = "\n".join(lines)
    print(table_md)
    verdict = "✅ SentinelFetal V5.1 PASSES LOGIC VALIDATION — Ready for Shadow Mode." if pass_logic else "⚠️ Logic Improvement Insufficient — Alert load remains high."
    print(verdict)

    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_OUT.open("w", encoding="utf-8") as f:
        f.write(table_md)
        f.write("\n\n")
        f.write(verdict)

    if pass_logic:
        print(f"Saved logic config to {LOGIC_CONFIG_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["ensemble", "xgb_solo"], default=None, help="Override engine choice")
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT, help="Config path for weights/threshold")
    args = parser.parse_args()
    main(engine_override=args.engine, config_path=args.config)
