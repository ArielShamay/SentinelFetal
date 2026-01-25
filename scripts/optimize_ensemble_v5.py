import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple, Optional

import yaml
from sklearn.metrics import roc_auc_score

VALIDATION_PATH = Path("models/ensemble_v4/validation_preds.csv")
OUTPUT_CONFIG = Path("config/ensemble_v5.yaml")


@dataclass
class Candidate:
    mode: str
    thr: float
    weights: Tuple[float, float, float]
    recall: float
    precision: float
    f2: float
    auc: float
    fp_per_patient: float
    pct_alerted: float
    guardrail_ok: bool


def load_predictions(path: Path) -> Tuple[List[str], Dict[str, Dict[str, List[float]]]]:
    patients: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"label": None, "xgb": [], "rf": [], "sgd": []})
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            label = int(row["true_label"])
            patients[pid]["label"] = label
            patients[pid]["xgb"].append(float(row["xgb_prob"]))
            patients[pid]["rf"].append(float(row["rf_prob"]))
            patients[pid]["sgd"].append(float(row["sgd_prob"]))
    pids = list(patients.keys())
    return pids, patients


def build_weight_grid() -> List[Tuple[float, float, float]]:
    weights: List[Tuple[float, float, float]] = []
    for wx in [i / 100 for i in range(10, 61, 5)]:
        for wr in [i / 100 for i in range(20, 71, 5)]:
            ws = round(1.0 - wx - wr, 2)
            if ws < -1e-9 or ws > 0.30 + 1e-9:
                continue
            if wx < 0.10 - 1e-9 or wx > 0.60 + 1e-9:
                continue
            if wr < 0.20 - 1e-9 or wr > 0.70 + 1e-9:
                continue
            if ws < 0:
                continue
            weights.append((round(wx, 2), round(wr, 2), round(ws, 2)))
    return weights


def evaluate_mode(
    mode: str,
    thr: float,
    weights: Tuple[float, float, float],
    pids: List[str],
    patients: Dict[str, Dict[str, List[float]]],
    labels: List[int]
) -> Candidate:
    wx, wr, ws = weights
    preds = []
    probs_for_auc = []
    for pid in pids:
        rec = patients[pid]
        win_probs = [wx * x + wr * r + ws * s for x, r, s in zip(rec["xgb"], rec["rf"], rec["sgd"])]
        if mode == "any":
            patient_prob = max(win_probs)
            pred = 1 if patient_prob >= thr else 0
        else:
            count = sum(1 for p in win_probs if p > thr)
            pred = 1 if count >= 2 else 0
            patient_prob = max(win_probs)
        preds.append(pred)
        probs_for_auc.append(patient_prob)

    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if precision > 0 and recall > 0 else 0.0

    try:
        auc = roc_auc_score(labels, probs_for_auc) if len(set(labels)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    fp_per_patient = fp / len(labels)
    pct_alerted = (tp + fp) / len(labels)
    guardrail_ok = precision >= 0.35 and thr >= 0.40

    return Candidate(
        mode=mode,
        thr=thr,
        weights=weights,
        recall=recall,
        precision=precision,
        f2=f2,
        auc=auc,
        fp_per_patient=fp_per_patient,
        pct_alerted=pct_alerted,
        guardrail_ok=guardrail_ok,
    )


def pick_best(cands: List[Candidate]) -> Tuple[Optional[Candidate], Optional[Candidate]]:
    guarded = [c for c in cands if c.guardrail_ok]
    fallback = [c for c in cands if not c.guardrail_ok]

    best_guarded = max(guarded, key=lambda c: (c.f2, c.recall, c.precision), default=None)
    best_fallback = max(fallback, key=lambda c: (c.precision, c.recall), default=None)
    return best_guarded, best_fallback


def main():
    pids, patients = load_predictions(VALIDATION_PATH)
    labels = [patients[pid]["label"] for pid in pids]

    threshold_grid = [round(t, 2) for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]
    weight_grid = build_weight_grid()

    any_candidates: List[Candidate] = []
    persistence_candidates: List[Candidate] = []

    for w in weight_grid:
        for thr in threshold_grid:
            any_candidates.append(evaluate_mode("any", thr, w, pids, patients, labels))
            persistence_candidates.append(evaluate_mode("persistence", thr, w, pids, patients, labels))

    best_any, fb_any = pick_best(any_candidates)
    best_persist, fb_persist = pick_best(persistence_candidates)

    chosen: Candidate
    status: str

    if best_any is None and best_persist is None:
        # Fallback rule: pick highest precision overall
        cand_pool = [c for c in [fb_any, fb_persist] if c is not None]
        if not cand_pool:
            print("No configurations evaluated.")
            return
        chosen = max(cand_pool, key=lambda c: (c.precision, c.recall))
        status = "not_acceptable"
    else:
        # At least one satisfies guardrail
        cand_any = best_any or fb_any
        cand_persist = best_persist or fb_persist
        if cand_persist and cand_any:
            if cand_persist.precision > cand_any.precision and cand_persist.recall >= 0.85:
                chosen = cand_persist
            else:
                chosen = cand_any
        else:
            chosen = cand_any or cand_persist  # whichever exists
        status = "acceptable"

    def fmt_val(c: Optional[Candidate], key: str) -> str:
        if c is None:
            return "N/A"
        if key == "thr":
            return f"{c.thr:.2f}"
        if key == "weights":
            return f"{c.weights[0]:.2f}/{c.weights[1]:.2f}/{c.weights[2]:.2f}"
        return f"{getattr(c, key):.3f}"

    print("| Metric | Best Any-Window | Best Persistence |")
    print("| --- | --- | --- |")
    metrics = [
        ("Threshold", "thr"),
        ("Weights (XGB/RF/SGD)", "weights"),
        ("Recall", "recall"),
        ("Precision", "precision"),
        ("F2-Score", "f2"),
        ("FP / Patient", "fp_per_patient"),
        ("% Patients Alerted", "pct_alerted"),
        ("ROC-AUC", "auc"),
    ]
    for label, key in metrics:
        print(f"| {label} | {fmt_val(best_any, key)} | {fmt_val(best_persist, key)} |")

    # Save config
    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "weights": {
            "xgboost": round(chosen.weights[0], 3),
            "random_forest": round(chosen.weights[1], 3),
            "sgd_classifier": round(chosen.weights[2], 3),
        },
        "threshold": float(round(chosen.thr, 3)),
        "aggregation_mode": "any" if chosen.mode == "any" else "persistence",
        "objective": "F2",
        "timestamp": datetime.utcnow().isoformat(),
    }
    with OUTPUT_CONFIG.open("w") as f:
        yaml.safe_dump(cfg, f)
    print(f"Saved config to {OUTPUT_CONFIG}")
    print(f"Chosen: mode={chosen.mode}, thr={chosen.thr:.2f}, weights={chosen.weights}, guardrail_ok={chosen.guardrail_ok}, status={status}")

    if status != "acceptable":
        print("Final status: NOT ACCEPTABLE (precision guardrail not met)")


if __name__ == "__main__":
    main()
