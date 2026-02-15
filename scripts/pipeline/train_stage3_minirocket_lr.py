#!/usr/bin/env python3
"""
Stage 3 Training Script: MiniRocket + StandardScaler + LogisticRegression

This script trains the AI Baseline pipeline for SentinelFetal.
It reads windows from Stage 2, trains MiniRocket features, scales them,
and fits a LogisticRegression classifier.

Outputs:
- stage3_ai_pipeline.joblib: Trained pipeline components
- stage3_eval_report.json: Training metrics
- STAGE3_TRUTH.md: Truth document with training details
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Try importing MiniRocket from sktime
try:
    from sktime.transformations.panel.rocket import MiniRocket
except ImportError:
    print("ERROR: sktime not installed. Please run: pip install sktime")
    print("Or add 'sktime' to requirements.txt")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = PROJECT_ROOT / "processed_data_v1"
WINDOWS_INDEX_PATH = DATA_DIR / "windows_index.csv"
X_NORM_PATH = DATA_DIR / "X_norm_4hz.npy"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

OUTPUT_PIPELINE_PATH = DATA_DIR / "stage3_ai_pipeline.joblib"
OUTPUT_REPORT_PATH = DATA_DIR / "stage3_eval_report.json"
OUTPUT_TRUTH_PATH = DATA_DIR / "STAGE3_TRUTH.md"

# Label policy: pH < 7.10 = positive (adverse outcome)
PH_THRESHOLD = 7.10

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Hyperparameters
RANDOM_STATE = 42
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 5000
LR_CLASS_WEIGHT = "balanced"

# Hyperparameter Tuning
# C is the inverse of regularization strength - smaller C = more regularization
# Grid search will find optimal C to prevent overfitting with 10K+ MiniRocket features
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]  # Regularization strengths to try

# Cross-validation
N_FOLDS_CV = 5  # Number of folds for Group K-Fold CV

# Signal parameters (invariants)
FS = 4  # Hz
WINDOW_SAMPLES = 4800  # 20 minutes * 60 seconds * 4 Hz
N_CHANNELS = 2  # FHR, UC


def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load windows index, normalized data, and manifest."""
    print("Loading data...")
    
    windows_df = pd.read_csv(WINDOWS_INDEX_PATH)
    print(f"  - Windows index: {len(windows_df)} rows")
    
    X_norm = np.load(X_NORM_PATH)
    print(f"  - X_norm shape: {X_norm.shape}")
    
    manifest_df = pd.read_csv(MANIFEST_PATH)
    print(f"  - Manifest: {len(manifest_df)} records")
    
    return windows_df, X_norm, manifest_df


def prepare_windows(
    windows_df: pd.DataFrame,
    X_norm: np.ndarray,
    manifest_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare windows for training.
    
    CRITICAL CONSTRAINTS:
    - Model input: ONLY FHR + UC signals (2 channels)
    - Label: Derived ONLY from pH (pH < 7.10 = adverse)
    - pH and other outcome variables (Apgar, NICU, etc.) are NEVER used as features
    - All metadata stays in manifest for tracking, but NOT in model inputs
    
    Returns:
        X: Window data (n_windows, n_channels, n_timestamps) - ONLY FHR + UC
        y: Labels (n_windows,) - Binary, derived from pH threshold
        groups: Record IDs for GroupSplit (n_windows,)
        valid_windows_df: Filtered windows dataframe with metadata
    """
    print("\nPreparing windows...")
    print("  ⚠️  MODEL INPUT: FHR + UC signals ONLY (2 channels)")
    print("  ⚠️  LABEL: pH < 7.10 (no other outcome variables)")
    
    # Filter valid windows
    valid_mask = windows_df["valid_for_ai"] == True
    valid_windows_df = windows_df[valid_mask].copy()
    print(f"  - Valid windows (valid_for_ai=True): {len(valid_windows_df)}")
    
    # Merge with manifest to get pH labels
    valid_windows_df = valid_windows_df.merge(
        manifest_df[["record_id", "ph"]],
        on="record_id",
        how="left"
    )
    
    # Create binary labels: pH < 7.10 = 1 (adverse)
    valid_windows_df["y"] = (valid_windows_df["ph"] < PH_THRESHOLD).astype(int)
    n_positive = valid_windows_df["y"].sum()
    n_negative = len(valid_windows_df) - n_positive
    print(f"  - Labels: {n_positive} positive (pH<{PH_THRESHOLD}), {n_negative} negative")
    
    # Extract window data
    # X_norm shape: (n_records, max_samples, n_channels)
    # We need: (n_windows, n_channels, n_timestamps)
    # CRITICAL: n_channels = 2 (FHR, UC) - NO OTHER FEATURES!
    X_list = []
    for _, row in valid_windows_df.iterrows():
        array_idx = int(row["array_idx"])
        start = int(row["start"])
        end = int(row["end"])
        
        # Extract window: (n_timestamps, n_channels)
        # n_channels = 2: [FHR, UC]
        window = X_norm[array_idx, start:end, :]
        
        # Transpose to (n_channels, n_timestamps)
        window = window.T
        X_list.append(window)
    
    X = np.array(X_list)
    y = valid_windows_df["y"].values
    groups = valid_windows_df["record_id"].values
    assert X.shape[1] == 2, f"Expected 2 channels (FHR, UC), got {X.shape[1]}"
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - Unique records: {len(np.unique(groups))}")
    print(f"  ✓ Verified: Model input is FHR + UC only (2 channels)")
    
    return X, y, groups, valid_windows_df


def split_data(
    X: np.ndarray,
    y: np.ndarray, 
    groups: np.ndarray,
    valid_windows_df: pd.DataFrame,
) -> dict:
    """
    Split data into train/val/test by record_id (group-based).
    
    Returns dict with X_train, X_val, X_test, y_train, y_val, y_test,
    and corresponding dataframes.
    """
    print("\nSplitting data by record_id...")
    
    # First split: train vs (val+test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO + TEST_RATIO, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss1.split(X, y, groups))
    
    # Second split: val vs test (from temp)
    X_temp = X[temp_idx]
    y_temp = y[temp_idx]
    groups_temp = groups[temp_idx]
    
    # Relative test size from temp
    relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=RANDOM_STATE)
    val_idx_rel, test_idx_rel = next(gss2.split(X_temp, y_temp, groups_temp))
    
    # Map back to original indices
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]
    
    # Extract splits
    data = {
        "X_train": X[train_idx],
        "X_val": X[val_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "y_test": y[test_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    
    # Assign split labels to dataframe
    valid_windows_df = valid_windows_df.copy()
    valid_windows_df["split"] = "unknown"
    valid_windows_df.iloc[train_idx, valid_windows_df.columns.get_loc("split")] = "train"
    valid_windows_df.iloc[val_idx, valid_windows_df.columns.get_loc("split")] = "val"
    valid_windows_df.iloc[test_idx, valid_windows_df.columns.get_loc("split")] = "test"
    data["windows_df"] = valid_windows_df
    
    # Verify no leakage
    train_records = set(groups[train_idx])
    val_records = set(groups[val_idx])
    test_records = set(groups[test_idx])
    
    assert train_records.isdisjoint(val_records), "Leakage: train-val overlap!"
    assert train_records.isdisjoint(test_records), "Leakage: train-test overlap!"
    assert val_records.isdisjoint(test_records), "Leakage: val-test overlap!"
    
    print(f"  - Train: {len(train_idx)} windows, {len(train_records)} records")
    print(f"  - Val: {len(val_idx)} windows, {len(val_records)} records")
    print(f"  - Test: {len(test_idx)} windows, {len(test_records)} records")
    print("  - ✓ No leakage between splits")
    
    return data


def compute_record_level_metrics(
    y_true_windows: np.ndarray,
    y_proba_windows: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """
    Compute record-level metrics by aggregating window predictions per patient.
    
    Strategy: Take max probability per record (worst-case window).
    This gives a more realistic evaluation since windows are dependent.
    """
    df = pd.DataFrame({
        'record_id': groups,
        'y_true': y_true_windows,
        'y_proba': y_proba_windows,
    })
    
    # Aggregate per record: take max probability (worst-case)
    record_agg = df.groupby('record_id').agg({
        'y_true': 'max',  # If any window is positive, record is positive
        'y_proba': 'max',  # Take highest probability from all windows
    }).reset_index()
    
    y_true_records = record_agg['y_true'].values
    y_proba_records = record_agg['y_proba'].values
    
    roc_auc = roc_auc_score(y_true_records, y_proba_records)
    pr_auc = average_precision_score(y_true_records, y_proba_records)
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'n_records': len(y_true_records),
        'n_positive': int(y_true_records.sum()),
    }


def train_pipeline_with_cv(data: dict) -> dict:
    """
    Train pipeline with Group K-Fold CV, hyperparameter tuning, and record-level metrics.
    
    Process:
    1. Fit MiniRocket + StandardScaler on full train set (no CV needed for feature extraction)
    2. Use Group K-Fold CV on train set to select best C parameter
    3. Train final model with best C on full train set
    4. Evaluate on train/val/test with both window-level and record-level metrics
    
    Returns dict with trained components and comprehensive metrics.
    """
    print("\n" + "=" * 60)
    print("Training Pipeline with CV and Hyperparameter Tuning")
    print("=" * 60)
    
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    
    # Get groups for record-level evaluation
    windows_df = data["windows_df"]
    train_groups = windows_df.iloc[train_idx]["record_id"].values
    val_groups = windows_df.iloc[val_idx]["record_id"].values
    test_groups = windows_df.iloc[test_idx]["record_id"].values
    
    # =========================================================================
    # Step 1: MiniRocket Feature Extraction (fit on train, transform all)
    # =========================================================================
    print("\n[STEP 1/4] Fitting MiniRocket on train set...")
    minirocket = MiniRocket(random_state=RANDOM_STATE)
    minirocket.fit(X_train)
    
    print("  Transforming all sets...")
    features_train = minirocket.transform(X_train)
    features_val = minirocket.transform(X_val)
    features_test = minirocket.transform(X_test)
    print(f"  - Features shape: {features_train.shape}")
    
    # =========================================================================
    # Step 2: StandardScaler (fit on train features, transform all)
    # =========================================================================
    print("\n[STEP 2/4] Fitting StandardScaler on train features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(features_train)
    X_val_scaled = scaler.transform(features_val)
    X_test_scaled = scaler.transform(features_test)
    
    # =========================================================================
    # Step 3: Hyperparameter Tuning with Group K-Fold CV
    # =========================================================================
    print(f"\n[STEP 3/4] Hyperparameter Tuning (Group {N_FOLDS_CV}-Fold CV)")
    print(f"  - Testing C values: {C_GRID}")
    print("  - Evaluating record-level ROC-AUC (max aggregation per patient)")
    
    group_kfold = GroupKFold(n_splits=N_FOLDS_CV)
    cv_results = []
    
    for c_value in C_GRID:
        fold_scores = []
        
        for fold_idx, (train_cv_idx, val_cv_idx) in enumerate(group_kfold.split(X_train_scaled, y_train, train_groups), 1):
            # Split CV fold
            X_train_cv = X_train_scaled[train_cv_idx]
            X_val_cv = X_train_scaled[val_cv_idx]
            y_train_cv = y_train[train_cv_idx]
            y_val_cv = y_train[val_cv_idx]
            groups_val_cv = train_groups[val_cv_idx]
            
            # Train model
            lr_cv = LogisticRegression(
                C=c_value,
                solver=LR_SOLVER,
                max_iter=LR_MAX_ITER,
                class_weight=LR_CLASS_WEIGHT,
                random_state=RANDOM_STATE,
            )
            lr_cv.fit(X_train_cv, y_train_cv)
            
            # Predict on validation fold
            y_proba_cv = lr_cv.predict_proba(X_val_cv)[:, 1]
            
            # Compute record-level metrics (realistic evaluation)
            record_metrics_cv = compute_record_level_metrics(y_val_cv, y_proba_cv, groups_val_cv)
            fold_scores.append(record_metrics_cv['roc_auc'])
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        cv_results.append({
            'C': c_value,
            'mean_roc_auc': mean_score,
            'std_roc_auc': std_score,
            'fold_scores': fold_scores,
        })
        print(f"    C={c_value:7.4f}: ROC-AUC = {mean_score:.4f} ± {std_score:.4f}")
    
    # Select best C
    best_result = max(cv_results, key=lambda x: x['mean_roc_auc'])
    best_c = best_result['C']
    print(f"\n  \u2713 Best C: {best_c} (ROC-AUC = {best_result['mean_roc_auc']:.4f} ± {best_result['std_roc_auc']:.4f})")
    
    # =========================================================================
    # Step 4: Train Final Model with Best C
    # =========================================================================
    print(f"\n[STEP 4/4] Training final model with C={best_c}...")
    lr_final = LogisticRegression(
        C=best_c,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        class_weight=LR_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
    )
    lr_final.fit(X_train_scaled, y_train)
    print(f"  - Converged in {lr_final.n_iter_[0]} iterations")
    
    # =========================================================================
    # Step 5: Comprehensive Evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Final Evaluation (Train / Val / Test)")
    print("=" * 60)
    
    # Predict on all sets
    y_train_proba = lr_final.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = lr_final.predict_proba(X_val_scaled)[:, 1]
    y_test_proba = lr_final.predict_proba(X_test_scaled)[:, 1]
    
    # Window-level metrics
    train_window_metrics = {
        'roc_auc': float(roc_auc_score(y_train, y_train_proba)),
        'pr_auc': float(average_precision_score(y_train, y_train_proba)),
        'n_windows': len(y_train),
        'n_positive': int(y_train.sum()),
    }
    val_window_metrics = {
        'roc_auc': float(roc_auc_score(y_val, y_val_proba)),
        'pr_auc': float(average_precision_score(y_val, y_val_proba)),
        'n_windows': len(y_val),
        'n_positive': int(y_val.sum()),
    }
    test_window_metrics = {
        'roc_auc': float(roc_auc_score(y_test, y_test_proba)),
        'pr_auc': float(average_precision_score(y_test, y_test_proba)),
        'n_windows': len(y_test),
        'n_positive': int(y_test.sum()),
    }
    
    # Record-level metrics (realistic performance)
    train_record_metrics = compute_record_level_metrics(y_train, y_train_proba, train_groups)
    val_record_metrics = compute_record_level_metrics(y_val, y_val_proba, val_groups)
    test_record_metrics = compute_record_level_metrics(y_test, y_test_proba, test_groups)
    
    # Print results
    print("\nWindow-Level Metrics:")
    print(f"  Train: ROC-AUC={train_window_metrics['roc_auc']:.4f}, PR-AUC={train_window_metrics['pr_auc']:.4f}")
    print(f"  Val:   ROC-AUC={val_window_metrics['roc_auc']:.4f}, PR-AUC={val_window_metrics['pr_auc']:.4f}")
    print(f"  Test:  ROC-AUC={test_window_metrics['roc_auc']:.4f}, PR-AUC={test_window_metrics['pr_auc']:.4f}")
    
    print("\nRecord-Level Metrics (max aggregation per patient):")
    print(f"  Train: ROC-AUC={train_record_metrics['roc_auc']:.4f}, PR-AUC={train_record_metrics['pr_auc']:.4f}")
    print(f"  Val:   ROC-AUC={val_record_metrics['roc_auc']:.4f}, PR-AUC={val_record_metrics['pr_auc']:.4f}")
    print(f"  Test:  ROC-AUC={test_record_metrics['roc_auc']:.4f}, PR-AUC={test_record_metrics['pr_auc']:.4f}")
    
    return {
        "minirocket": minirocket,
        "scaler": scaler,
        "lr": lr_final,
        "best_c": best_c,
        "cv_results": cv_results,
        "metrics": {
            "window_level": {
                "train": train_window_metrics,
                "val": val_window_metrics,
                "test": test_window_metrics,
            },
            "record_level": {
                "train": train_record_metrics,
                "val": val_record_metrics,
                "test": test_record_metrics,
            },
        },
    }


def save_artifacts(trained: dict, data: dict) -> None:
    """Save trained pipeline and reports."""
    print("\n" + "=" * 60)
    print("Saving Artifacts")
    print("=" * 60)
    
    timestamp = datetime.now().isoformat()
    
    # Save pipeline
    pipeline_artifact = {
        "minirocket": trained["minirocket"],
        "scaler": trained["scaler"],
        "lr": trained["lr"],
        "model_version": "stage3_v2.0",  # Updated version with CV
        "trained_at": timestamp,
        "config": {
            "ph_threshold": PH_THRESHOLD,
            "random_state": RANDOM_STATE,
            "lr_solver": LR_SOLVER,
            "lr_max_iter": LR_MAX_ITER,
            "lr_c": trained["best_c"],  # Save optimal C from CV
            "lr_class_weight": LR_CLASS_WEIGHT,
            "fs": FS,
            "window_samples": WINDOW_SAMPLES,
            "n_channels": N_CHANNELS,
            "n_folds_cv": N_FOLDS_CV,
            "c_grid": C_GRID,
        },
    }
    joblib.dump(pipeline_artifact, OUTPUT_PIPELINE_PATH)
    print(f"  - Pipeline saved: {OUTPUT_PIPELINE_PATH}")
    
    # Save eval report
    report = {
        "timestamp": timestamp,
        "model_version": "stage3_v2.0",
        "config": pipeline_artifact["config"],
        "cv_results": trained["cv_results"],
        "best_c": trained["best_c"],
        "metrics": trained["metrics"],
        "split_info": {
            "n_train_windows": len(data["train_idx"]),
            "n_val_windows": len(data["val_idx"]),
            "n_test_windows": len(data["test_idx"]),
            "n_train_records": len(set(data["windows_df"].iloc[data["train_idx"]]["record_id"])),
            "n_val_records": len(set(data["windows_df"].iloc[data["val_idx"]]["record_id"])),
            "n_test_records": len(set(data["windows_df"].iloc[data["test_idx"]]["record_id"])),
        },
    }
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  - Report saved: {OUTPUT_REPORT_PATH}")
    
    # Save TRUTH.md
    # Format CV results
    cv_table = "\n".join([
        f"| {r['C']:.4f} | {r['mean_roc_auc']:.4f} | {r['std_roc_auc']:.4f} |"
        for r in trained["cv_results"]
    ])
    
    truth_content = f'''<div dir="rtl" style="text-align: right;">

# מסמך אמת (Truth Source) - Stage 3 v2.0
**נכון לתאריך:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 1. מטרה
הפקת `ai_score` לכל חלון באמצעות MiniRocket + StandardScaler + LogisticRegression על CPU.
**גרסה 2.0:** כוללת Group K-Fold CV, Hyperparameter Tuning, ומדדים ברמת רשומה.

## 2. קלט
- **מקור חלונות:** Stage 2 (`windows_index.csv`)
- **קובץ נתונים:** `X_norm_4hz.npy`
- **Input shape:** `(n_windows, 2, 4800)` — FHR + UC
- **תדר דגימה:** {FS}Hz
- **אורך חלון:** 20 דקות ({WINDOW_SAMPLES} samples)

## 3. Label Policy
- **pH Threshold:** {PH_THRESHOLD}
- **Label:** pH < {PH_THRESHOLD} → 1 (adverse), אחרת → 0

## 4. Pipeline
1. MiniRocket.fit(train) → transform(all)
2. StandardScaler.fit(train_features) → transform(all)
3. Group K-Fold CV לבחירת C אופטימלי
4. LogisticRegression.fit(train_scaled, y_train) עם C אופטימלי
5. ai_score = predict_proba[:, 1]

## 5. Split (GroupShuffleSplit לפי record_id)
| Split | Records | Windows |
|-------|---------|---------|
| Train | {report["split_info"]["n_train_records"]} | {report["split_info"]["n_train_windows"]} |
| Val | {report["split_info"]["n_val_records"]} | {report["split_info"]["n_val_windows"]} |
| Test | {report["split_info"]["n_test_records"]} | {report["split_info"]["n_test_windows"]} |

## 6. Hyperparameter Tuning (NEW)
**Method:** Group {N_FOLDS_CV}-Fold Cross-Validation
**Metric:** Record-level ROC-AUC (max aggregation per patient)

### CV Results:
| C (Regularization) | Mean ROC-AUC | Std ROC-AUC |
|--------------------|--------------|-------------|
{cv_table}

**Selected C:** {trained["best_c"]} (Best mean ROC-AUC)

## 7. Hyperparameters (Final Model)
| פרמטר | ערך |
|-------|-----|
| MiniRocket random_state | {RANDOM_STATE} |
| StandardScaler | defaults (z-score) |
| LR solver | {LR_SOLVER} |
| LR max_iter | {LR_MAX_ITER} |
| LR C (regularization) | {trained["best_c"]} ✓ |
| LR class_weight | {LR_CLASS_WEIGHT} |
| LR random_state | {RANDOM_STATE} |

## 8. Data Integrity
- **Model Input:** FHR + UC signals ONLY (2 channels, normalized [0,1])
- **Label Source:** pH < {PH_THRESHOLD} ONLY
- **Prohibited Features:** pH value, Apgar scores, NICU days, Seizures, HIE, diagnoses, or ANY outcome variables
- **Metadata:** All outcome variables stored in manifest.csv for tracking ONLY, never as features

## 9. Split Validation
- **Method:** GroupShuffleSplit by record_id
- **Critical:** NO windows from same patient in both train and val/test
- **Verified:** No data leakage between splits

## 10. Metrics

### Window-Level Metrics (per-window predictions):
| | Train ROC | Val ROC | Test ROC | Train PR | Val PR | Test PR |
|---|-----------|---------|----------|----------|--------|---------|
| | {trained["metrics"]["window_level"]["train"]["roc_auc"]:.4f} | {trained["metrics"]["window_level"]["val"]["roc_auc"]:.4f} | {trained["metrics"]["window_level"]["test"]["roc_auc"]:.4f} | {trained["metrics"]["window_level"]["train"]["pr_auc"]:.4f} | {trained["metrics"]["window_level"]["val"]["pr_auc"]:.4f} | {trained["metrics"]["window_level"]["test"]["pr_auc"]:.4f} |

### Record-Level Metrics (NEW - max aggregation per patient):
| | Train ROC | Val ROC | Test ROC | Train PR | Val PR | Test PR |
|---|-----------|---------|----------|----------|--------|---------|
| | {trained["metrics"]["record_level"]["train"]["roc_auc"]:.4f} | {trained["metrics"]["record_level"]["val"]["roc_auc"]:.4f} | {trained["metrics"]["record_level"]["test"]["roc_auc"]:.4f} | {trained["metrics"]["record_level"]["train"]["pr_auc"]:.4f} | {trained["metrics"]["record_level"]["val"]["pr_auc"]:.4f} | {trained["metrics"]["record_level"]["test"]["pr_auc"]:.4f} |

**⚠️ Note:** 
- Window-level metrics may be optimistic due to dependent windows from same patient
- **Record-level metrics are more realistic** - aggregated by max probability per patient
- Use record-level metrics for production decision-making

## 11. Artifacts
- `stage3_ai_pipeline.joblib` — Pipeline מאומן (v2.0)
- `stage3_eval_report.json` — דוח הערכה (כולל CV results)
- `ai_scores.parquet` — סקורים לכל חלון (נוצר ע"י eval script)

## 12. Inference Contract
Stage 5 מקבל מ-Stage 3:
- `ai_score` (float, [0, 1])
- `model_version` (str, "stage3_v2.0")
- `window_start`, `window_end` (int, samples)

## 13. Changes from v1.0
- ✓ Added Group K-Fold Cross-Validation
- ✓ Added Hyperparameter Tuning (C grid search)
- ✓ Added Record-Level Metrics (max aggregation per patient)
- ✓ Added Test Set Evaluation
- ✓ Used optimal C={trained["best_c"]} instead of default C=1.0

</div>
'''
    with open(OUTPUT_TRUTH_PATH, "w") as f:
        f.write(truth_content)
    print(f"  - Truth doc saved: {OUTPUT_TRUTH_PATH}")


def main():
    """Main training flow."""
    print("=" * 60)
    print("Stage 3 Training v2.0: MiniRocket + LR with CV & Tuning")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Load data
    windows_df, X_norm, manifest_df = load_data()
    
    # Prepare windows
    X, y, groups, valid_windows_df = prepare_windows(windows_df, X_norm, manifest_df)
    
    # Split data
    data = split_data(X, y, groups, valid_windows_df)
    
    # Train pipeline with CV and hyperparameter tuning
    trained = train_pipeline_with_cv(data)
    
    # Save artifacts
    save_artifacts(trained, data)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Finished at: {datetime.now().isoformat()}")
    print(f"\nBest C: {trained['best_c']}")
    print(f"Test ROC-AUC (Record-Level): {trained['metrics']['record_level']['test']['roc_auc']:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
