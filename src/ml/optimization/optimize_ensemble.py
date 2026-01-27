"""
Ensemble optimization for SentinelFetal V4.0.
- Tunes ensemble weights (XGB, RF, SGD) under FPR constraint
- Tunes decision threshold for recall/precision tradeoff
- Computes patient-level metrics
- Generates plots: ROC, PR, Calibration
- Saves patient-level predictions and optimized config
"""

import json
import pickle
import itertools
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# Reuse data loader and feature extractor from training script
from src.ml.training.train_v4_ensemble import CTUCHBDataLoader, CTGFeatureExtractor
DATA_DIR = PROJECT_ROOT / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
MODELS_DIR = PROJECT_ROOT / "models" / "ensemble_v4"
CONFIG_PATH = PROJECT_ROOT / "config" / "ensemble_v4.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "REPORTS" / "ensemble_v4"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def load_config() -> Dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_models(config: Dict) -> Tuple[Dict[str, object], object]:
    artifacts = config.get("artifacts", {})
    models = {}
    scaler = None

    xgb_path = PROJECT_ROOT / artifacts.get("xgboost_path", "models/ensemble_v4/xgboost_calibrated.pkl")
    rf_path = PROJECT_ROOT / artifacts.get("random_forest_path", "models/ensemble_v4/random_forest_calibrated.pkl")
    sgd_path = PROJECT_ROOT / artifacts.get("sgd_path", "models/ensemble_v4/sgd_classifier_calibrated.pkl")
    scaler_path = PROJECT_ROOT / artifacts.get("scaler_path", "models/ensemble_v4/feature_scaler.pkl")

    with open(xgb_path, "rb") as f:
        models["xgboost"] = pickle.load(f)
    with open(rf_path, "rb") as f:
        models["random_forest"] = pickle.load(f)
    with open(sgd_path, "rb") as f:
        models["sgd_classifier"] = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return models, scaler


def load_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    loader = CTUCHBDataLoader(DATA_DIR)
    fhr_signals, labels, patient_ids = loader.load_all_records()
    extractor = CTGFeatureExtractor()

    X_list = []
    y_list = []
    pid_list = []
    for fhr, label, pid in zip(fhr_signals, labels, patient_ids):
        feats = extractor.extract(fhr, fs=4.0)
        if np.sum(np.abs(feats)) == 0:
            continue
        X_list.append(feats)
        y_list.append(label)
        pid_list.append(pid)
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, pid_list


def predict_all(models: Dict[str, object], scaler, X: np.ndarray) -> Dict[str, np.ndarray]:
    Xs = scaler.transform(X)
    outputs = {}
    for name, model in models.items():
        proba = model.predict_proba(Xs)[:, 1]
        outputs[name] = proba
    return outputs


def weighted_ensemble(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    total = 0.0
    score = np.zeros_like(next(iter(preds.values())))
    for name, w in weights.items():
        if name in preds:
            score += w * preds[name]
            total += w
    if total > 0 and abs(total - 1.0) > 1e-6:
        score /= total
    return score


def optimize_weights(preds: Dict[str, np.ndarray], y: np.ndarray, step: float = 0.05) -> Tuple[Dict[str, float], float, float]:
    best_weights = None
    best_recall = -1
    best_threshold = 0.5

    grid = np.arange(0, 1 + 1e-6, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-6 or w3 > 1.0:
                continue
            weights = {"xgboost": w1, "random_forest": w2, "sgd_classifier": w3}
            ens = weighted_ensemble(preds, weights)
            fpr, tpr, thr = roc_curve(y, ens)
            valid = fpr <= 0.15
            if not np.any(valid):
                continue
            idx = np.argmax(tpr[valid])
            recall_at_constraint = tpr[valid][idx]
            thr_at = thr[valid][idx]
            if recall_at_constraint > best_recall:
                best_recall = recall_at_constraint
                best_weights = weights
                best_threshold = thr_at
    return best_weights, best_threshold, best_recall


def tune_threshold(y: np.ndarray, proba: np.ndarray, min_precision: float = 0.15) -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    best_thr = 0.5
    best_recall = -1
    best_precision = 0.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= min_precision and r > best_recall:
            best_recall = r
            best_precision = p
            best_thr = t
    metrics = compute_metrics(y, proba, best_thr)
    metrics.update({"threshold": best_thr, "precision_at_thr": best_precision, "recall_at_thr": best_recall})
    return best_thr, metrics


def compute_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, proba),
        "confusion": confusion_matrix(y, pred).tolist(),
    }


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y)
    for i in range(n_bins):
        mask = (proba >= bins[i]) & (proba < bins[i + 1]) if i < n_bins - 1 else (proba >= bins[i])
        if not np.any(mask):
            continue
        conf = proba[mask].mean()
        acc = y[mask].mean()
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return ece


def plot_curves(y: np.ndarray, proba: np.ndarray, out_dir: Path):
    fpr, tpr, _ = roc_curve(y, proba)
    precision, recall, _ = precision_recall_curve(y, proba)
    prob_true, prob_pred = calibration_curve(y, proba, n_bins=10)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.savefig(out_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(out_dir / "pr_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title("Calibration Curve")
    plt.savefig(out_dir / "calibration_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_patient_predictions(pids: List[str], y: np.ndarray, preds: Dict[str, np.ndarray], ensemble: np.ndarray, out_path: Path):
    df = pd.DataFrame({
        "patient_id": pids,
        "label": y,
        "proba_xgb": preds["xgboost"],
        "proba_rf": preds["random_forest"],
        "proba_sgd": preds["sgd_classifier"],
        "proba_ensemble": ensemble,
    })
    df.to_csv(out_path, index=False)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"optimization_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    models, scaler = load_models(config)
    X, y, patient_ids = load_data()

    preds = predict_all(models, scaler, X)
    base_weights = config.get("ensemble", {}).get("weights", {"xgboost": 0.4, "random_forest": 0.3, "sgd_classifier": 0.3})
    base_ensemble = weighted_ensemble(preds, base_weights)

    # Save raw predictions
    save_patient_predictions(patient_ids, y, preds, base_ensemble, out_dir / "patient_predictions.csv")

    # Optimize weights under FPR constraint
    best_weights, best_thr_fpr, best_recall_fpr = optimize_weights(preds, y, step=0.05)

    # Apply best weights
    tuned_ensemble = weighted_ensemble(preds, best_weights)

    # Tune threshold with precision floor
    tuned_thr, tuned_metrics = tune_threshold(y, tuned_ensemble, min_precision=0.15)

    # Calibration stats
    brier = brier_score_loss(y, tuned_ensemble)
    ece = expected_calibration_error(y, tuned_ensemble, n_bins=10)

    # Metrics for report
    base_metrics = compute_metrics(y, base_ensemble, threshold=0.5)
    tuned_final_metrics = compute_metrics(y, tuned_ensemble, threshold=tuned_thr)
    tuned_final_metrics.update({"brier": brier, "ece": ece})

    # Cost of recall (vs baseline threshold 0.5)
    base_precision = base_metrics["precision"]
    base_recall = base_metrics["recall"]
    tuned_precision = tuned_final_metrics["precision"]
    tuned_recall = tuned_final_metrics["recall"]
    delta_recall = tuned_recall - base_recall
    delta_precision = tuned_precision - base_precision
    cost_of_recall = abs(delta_precision / delta_recall) if delta_recall != 0 else np.inf

    # Plots
    plot_curves(y, tuned_ensemble, out_dir)

    # Save optimized config
    opt_config = {
        "ensemble": {
            "weights": {k: float(v) for k, v in best_weights.items()},
            "aggregation_method": "soft_voting",
        },
        "thresholds": {
            "critical": float(tuned_thr),
            "warning": 0.35,
            "normal": 0.35,
        },
        "notes": {
            "optimized_at": timestamp,
            "fpr_constraint": 0.15,
            "min_precision": 0.15,
        },
    }
    with open(out_dir / "optimized_ensemble_v4.yaml", "w") as f:
        yaml.safe_dump(opt_config, f)

    # Save report (JSON for downstream rendering)
    report = {
        "base_weights": base_weights,
        "best_weights": best_weights,
        "best_threshold_fpr": float(best_thr_fpr),
        "best_recall_fpr": float(best_recall_fpr),
        "tuned_threshold": float(tuned_thr),
        "metrics_base": base_metrics,
        "metrics_tuned": tuned_final_metrics,
        "cost_of_recall": float(cost_of_recall),
        "brier": float(brier),
        "ece": float(ece),
    }
    with open(out_dir / "optimization_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Optimization complete")
    print(f"Best weights: {best_weights}")
    print(f"Tuned threshold: {tuned_thr:.3f}")
    print(f"Recall (tuned): {tuned_final_metrics['recall']:.3f}, Precision: {tuned_final_metrics['precision']:.3f}")
    print(f"Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
