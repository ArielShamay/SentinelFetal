"""V5 Super-Features Training Pipeline

Adds clinical meta-features and rule proxy to MiniRocket embeddings.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    StratifiedGroupKFold = None  # type: ignore
    HAS_STRATIFIED_GROUP_KFOLD = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

import xgboost as xgb

from src.ml.features.clinical import compute_clinical_features, compute_rule_score
from src.models.minirocket_encoder import MiniRocketEncoder, MiniRocketEncoderError
from src.ml.training.train_v4_ensemble import CTUCHBDataLoader
CONFIG_PATH = PROJECT_ROOT / "config" / "ensemble_v5.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
OUTPUT_DIR = PROJECT_ROOT / "models" / "ensemble_v5"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    outer_folds: int
    random_seed: int
    smote_enabled: bool
    smote_k: int


DEFAULT_CONFIG: Dict[str, Any] = {
    "training": {
        "outer_folds": 10,
        "random_seed": 42,
        "smote": {"enabled": True, "k_neighbors": 5},
    },
    "models": {
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
        "sgd_classifier": {
            "loss": "log_loss",
            "penalty": "elasticnet",
            "alpha": 0.0005,
            "l1_ratio": 0.15,
            "max_iter": 1000,
            "class_weight": "balanced",
        },
    },
    "windowing": {"window_minutes": 20, "stride_minutes": 5},
    "feature_spec": "v5_super_features",
}


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r") as f:
            return yaml.safe_load(f)
    logger.warning("ensemble_v5.yaml not found; using defaults")
    return DEFAULT_CONFIG


def sliding_windows_with_uc(
    fhr: np.ndarray,
    uc: Optional[np.ndarray],
    window_samples: int,
    stride_samples: int,
    fs: float,
):
    if len(fhr) < window_samples:
        return
    stride = max(1, stride_samples)
    w_idx = 0
    for start in range(0, len(fhr) - window_samples + 1, stride):
        end = start + window_samples
        fhr_win = fhr[start:end]
        uc_win = uc[start:end] if uc is not None and len(uc) >= end else None
        start_min = (start / fs) / 60.0 if fs > 0 else 0.0
        yield w_idx, start_min, fhr_win, uc_win
        w_idx += 1


class NestedCVTrainerV5:
    def __init__(self, config: Dict[str, Any], mr_dim: int, clinical_dim: int):
        self.config = config
        training_cfg = config.get("training", {})
        smote_cfg = training_cfg.get("smote", {})
        self.cfg = TrainingConfig(
            outer_folds=training_cfg.get("outer_folds", 10),
            random_seed=training_cfg.get("random_seed", 42),
            smote_enabled=smote_cfg.get("enabled", True) and SMOTE_AVAILABLE,
            smote_k=smote_cfg.get("k_neighbors", 5),
        )
        self.model_configs = config.get("models", {})
        self.mr_dim = mr_dim
        self.clinical_dim = clinical_dim
        self.scaler_full: Optional[StandardScaler] = None
        self.oof_records: List[Dict[str, Any]] = []
        self.final_models: Dict[str, Any] = {}

    def _splitter(self, X, y, groups):
        if HAS_STRATIFIED_GROUP_KFOLD:
            return StratifiedGroupKFold(
                n_splits=self.cfg.outer_folds,
                shuffle=True,
                random_state=self.cfg.random_seed,
            ).split(X, y, groups)
        return GroupKFold(n_splits=self.cfg.outer_folds).split(X, groups)

    def _scale_features(self, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
        clinical_part = X[:, self.mr_dim : self.mr_dim + self.clinical_dim]
        scaled_clin = scaler.transform(clinical_part)
        return np.hstack([X[:, : self.mr_dim], scaled_clin])

    def _train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        models = {}
        xgb_cfg = self.model_configs.get("xgboost", {})
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 300),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
            objective=xgb_cfg.get("objective", "binary:logistic"),
            eval_metric=xgb_cfg.get("eval_metric", "logloss"),
            random_state=self.cfg.random_seed,
            use_label_encoder=False,
        )
        models["xgboost"].fit(X_train, y_train)

        rf_cfg = self.model_configs.get("random_forest", {})
        models["random_forest"] = RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 300),
            max_depth=rf_cfg.get("max_depth", 12),
            min_samples_split=rf_cfg.get("min_samples_split", 4),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            class_weight=rf_cfg.get("class_weight", "balanced"),
            random_state=self.cfg.random_seed,
            n_jobs=-1,
        )
        models["random_forest"].fit(X_train, y_train)

        sgd_cfg = self.model_configs.get("sgd_classifier", {})
        models["sgd_classifier"] = SGDClassifier(
            loss=sgd_cfg.get("loss", "log_loss"),
            penalty=sgd_cfg.get("penalty", "elasticnet"),
            alpha=sgd_cfg.get("alpha", 0.0005),
            l1_ratio=sgd_cfg.get("l1_ratio", 0.15),
            max_iter=sgd_cfg.get("max_iter", 1000),
            class_weight=sgd_cfg.get("class_weight", "balanced"),
            random_state=self.cfg.random_seed,
        )
        models["sgd_classifier"].fit(X_train, y_train)
        return models

    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.cfg.smote_enabled:
            return X, y
        try:
            smote = SMOTE(k_neighbors=self.cfg.smote_k, random_state=self.cfg.random_seed)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(f"SMOTE: {len(y)} -> {len(y_res)}")
            return X_res, y_res
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}; using original data")
            return X, y

    def train(self, X: np.ndarray, y: np.ndarray, patient_ids: List[str], window_indices: List[int]) -> Dict[str, Any]:
        groups = np.array(patient_ids)
        splitter = self._splitter(X, y, groups)

        all_true: List[int] = []
        proba_store = {"xgboost": [], "random_forest": [], "sgd_classifier": []}
        pred_store = {"xgboost": [], "random_forest": [], "sgd_classifier": []}

        for fold_idx, (train_idx, test_idx) in enumerate(splitter):
            logger.info(f"--- Fold {fold_idx + 1}/{self.cfg.outer_folds} ---")
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            scaler.fit(X_train_raw[:, self.mr_dim : self.mr_dim + self.clinical_dim])
            X_train = self._scale_features(scaler, X_train_raw)
            X_test = self._scale_features(scaler, X_test_raw)

            X_train_bal, y_train_bal = self._apply_smote(X_train, y_train)
            models = self._train_models(X_train_bal, y_train_bal)

            fold_probas = {}
            for name, model in models.items():
                proba = model.predict_proba(X_test)[:, 1]
                pred = (proba >= 0.5).astype(int)
                fold_probas[name] = proba
                proba_store[name].extend(proba.tolist())
                pred_store[name].extend(pred.tolist())
            all_true.extend(y_test.tolist())

            for local_idx, global_idx in enumerate(test_idx):
                self.oof_records.append(
                    {
                        "patient_id": patient_ids[global_idx],
                        "window_index": int(window_indices[global_idx]),
                        "true_label": int(y_test[local_idx]),
                        "xgb_prob": float(fold_probas["xgboost"][local_idx]),
                        "rf_prob": float(fold_probas["random_forest"][local_idx]),
                        "sgd_prob": float(fold_probas["sgd_classifier"][local_idx]),
                    }
                )

        results = self._compute_metrics(all_true, pred_store, proba_store)
        return results

    def _compute_metrics(self, y_true: List[int], preds: Dict[str, List[int]], probas: Dict[str, List[float]]):
        res: Dict[str, Any] = {"n_samples": len(y_true), "models": {}}
        y_true_arr = np.array(y_true)
        for name in preds.keys():
            y_pred = np.array(preds[name])
            y_proba = np.array(probas[name])
            res["models"][name] = {
                "accuracy": float(accuracy_score(y_true_arr, y_pred)),
                "precision": float(precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0)),
                "recall": float(recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0)),
                "f1": float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true_arr, y_proba)) if len(set(y_true_arr)) > 1 else 0.0,
                "confusion_matrix": confusion_matrix(y_true_arr, y_pred).tolist(),
            }
        return res

    def fit_final(self, X: np.ndarray, y: np.ndarray):
        scaler = StandardScaler()
        scaler.fit(X[:, self.mr_dim : self.mr_dim + self.clinical_dim])
        self.scaler_full = scaler
        X_proc = self._scale_features(scaler, X)
        X_bal, y_bal = self._apply_smote(X_proc, y)
        models = self._train_models(X_bal, y_bal)
        calibrated = {}
        for name, model in models.items():
            calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrator.fit(X_proc, y)
            calibrated[name] = calibrator
        self.final_models = calibrated

    def save_models(self, output_dir: Path):
        import pickle

        output_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.final_models.items():
            path = output_dir / f"{name}_v5.pkl"
            with path.open("wb") as f:
                pickle.dump(model, f)
        if self.scaler_full is not None:
            with (output_dir / "super_feature_scaler_v5.pkl").open("wb") as f:
                pickle.dump(self.scaler_full, f)


def main():
    start_time = datetime.now()
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data (CTU-CHB)...")
    loader = CTUCHBDataLoader(DATA_DIR)
    # Load labels/patient IDs, then fetch UC per record for clinical context
    fhr_signals, labels, patient_ids = loader.load_all_records()

    window_cfg = config.get("windowing", {})
    fs = 4.0
    window_minutes = window_cfg.get("window_minutes", 20)
    stride_minutes = window_cfg.get("stride_minutes", 5)
    window_samples = int(window_minutes * 60 * fs)
    stride_samples = int(stride_minutes * 60 * fs)

    try:
        encoder = MiniRocketEncoder()
    except MiniRocketEncoderError as e:
        logger.error(f"MiniRocket not available: {e}")
        sys.exit(1)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    pid_list: List[str] = []
    widx_list: List[int] = []

    for fhr, label, pid in zip(fhr_signals, labels, patient_ids):
        _, uc, _ = loader.load_record(pid)
        for w_idx, start_min, fhr_win, uc_win in sliding_windows_with_uc(fhr, uc, window_samples, stride_samples, fs):
            try:
                mr_feat = encoder.transform(fhr_win)
                raw_feat = mr_feat.features if hasattr(mr_feat, "features") else np.asarray(mr_feat)
                mr_vec = np.asarray(raw_feat).ravel()
            except Exception as e:
                logger.warning(f"MiniRocket transform failed for {pid} window {w_idx}: {e}")
                continue
            clin = compute_clinical_features(fhr_win, uc_win, fs=fs, start_time_min=start_min)
            rule_score = compute_rule_score(clin["baseline_fhr"], clin["stv_proxy"], clin["fhr_std"])
            clin_vec = np.array([
                clin["baseline_fhr"],
                clin["stv_proxy"],
                clin["fhr_std"],
                clin["uc_contractions"],
                clin["uc_rate"],
                clin["time_since_start_min"],
                clin["signal_quality"],
                float(rule_score),
            ], dtype=float)
            full_vec = np.concatenate([mr_vec, clin_vec])
            X_list.append(full_vec)
            y_list.append(label)
            pid_list.append(pid)
            widx_list.append(w_idx)

    X = np.vstack(X_list)
    y = np.array(y_list)
    logger.info(f"Feature matrix shape: {X.shape}")

    mr_dim = X_list[0].shape[0] - 8
    clinical_dim = 7 + 1  # clinical + rule score

    trainer = NestedCVTrainerV5(config, mr_dim=mr_dim, clinical_dim=clinical_dim)
    results = trainer.train(X, y, pid_list, widx_list)

    trainer.fit_final(X, y)
    trainer.save_models(OUTPUT_DIR)

    oof_path = OUTPUT_DIR / "validation_preds_v5.csv"
    with oof_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "window_index", "true_label", "xgb_prob", "rf_prob", "sgd_prob"])
        for rec in trainer.oof_records:
            writer.writerow([
                rec["patient_id"],
                rec["window_index"],
                rec["true_label"],
                rec["xgb_prob"],
                rec["rf_prob"],
                rec["sgd_prob"],
            ])
    logger.info(f"Saved OOF preds to {oof_path}")

    results_path = OUTPUT_DIR / "training_results_v5.json"
    results.update(
        {
            "feature_spec": config.get("feature_spec", "v5_super_features"),
            "random_seed": config.get("training", {}).get("random_seed", 42),
            "outer_folds": trainer.cfg.outer_folds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved training results to {results_path}")

    logger.info("Training V5 complete")


if __name__ == "__main__":
    main()
