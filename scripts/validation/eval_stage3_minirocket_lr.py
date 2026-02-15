#!/usr/bin/env python3
"""
Stage 3 Evaluation Script: Evaluate trained pipeline and generate ai_scores.parquet

This script loads the trained Stage 3 pipeline and evaluates it on all splits,
generating ai_scores.parquet with scores for each window.

Outputs:
- ai_scores.parquet: Scores for each window with metadata
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = PROJECT_ROOT / "processed_data_v1"
WINDOWS_INDEX_PATH = DATA_DIR / "windows_index.csv"
X_NORM_PATH = DATA_DIR / "X_norm_4hz.npy"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
PIPELINE_PATH = DATA_DIR / "stage3_ai_pipeline.joblib"

OUTPUT_SCORES_PATH = DATA_DIR / "ai_scores.parquet"

# Label policy (must match training)
PH_THRESHOLD = 7.10

# Split (must match training)
from sklearn.model_selection import GroupShuffleSplit
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_pipeline() -> dict:
    """Load trained pipeline."""
    print("Loading trained pipeline...")
    if not PIPELINE_PATH.exists():
        print(f"ERROR: Pipeline not found at {PIPELINE_PATH}")
        print("Please run train_stage3_minirocket_lr.py first.")
        sys.exit(1)
    
    pipeline = joblib.load(PIPELINE_PATH)
    print(f"  - Model version: {pipeline['model_version']}")
    print(f"  - Trained at: {pipeline['trained_at']}")
    return pipeline


def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load windows index, normalized data, and manifest."""
    print("\nLoading data...")
    
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
    Prepare windows for evaluation.
    
    Returns:
        X: Window data (n_windows, n_channels, n_timestamps)
        y: Labels (n_windows,)
        groups: Record IDs for GroupSplit (n_windows,)
        valid_windows_df: Filtered windows dataframe
    """
    print("\nPreparing windows...")
    
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
    
    # Create binary labels
    valid_windows_df["y_true"] = (valid_windows_df["ph"] < PH_THRESHOLD).astype(int)
    
    # Extract window data
    X_list = []
    for _, row in valid_windows_df.iterrows():
        array_idx = int(row["array_idx"])
        start = int(row["start"])
        end = int(row["end"])
        
        window = X_norm[array_idx, start:end, :]
        window = window.T  # (n_channels, n_timestamps)
        X_list.append(window)
    
    X = np.array(X_list)
    y = valid_windows_df["y_true"].values
    groups = valid_windows_df["record_id"].values
    
    print(f"  - X shape: {X.shape}")
    
    return X, y, groups, valid_windows_df


def assign_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    valid_windows_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign train/val/test splits (must match training exactly).
    """
    print("\nAssigning splits...")
    
    # Recreate exact same split as training
    gss1 = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO + TEST_RATIO, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss1.split(X, y, groups))
    
    X_temp = X[temp_idx]
    y_temp = y[temp_idx]
    groups_temp = groups[temp_idx]
    
    relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=RANDOM_STATE)
    val_idx_rel, test_idx_rel = next(gss2.split(X_temp, y_temp, groups_temp))
    
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]
    
    # Assign splits
    valid_windows_df = valid_windows_df.copy()
    valid_windows_df["split"] = "unknown"
    valid_windows_df.iloc[train_idx, valid_windows_df.columns.get_loc("split")] = "train"
    valid_windows_df.iloc[val_idx, valid_windows_df.columns.get_loc("split")] = "val"
    valid_windows_df.iloc[test_idx, valid_windows_df.columns.get_loc("split")] = "test"
    
    print(f"  - Train: {len(train_idx)} windows")
    print(f"  - Val: {len(val_idx)} windows")
    print(f"  - Test: {len(test_idx)} windows")
    
    return valid_windows_df


def compute_scores(
    pipeline: dict,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute ai_scores for all windows using the trained pipeline.
    """
    print("\nComputing ai_scores...")
    
    minirocket = pipeline["minirocket"]
    scaler = pipeline["scaler"]
    lr = pipeline["lr"]
    
    # Transform
    print("  - Applying MiniRocket transform...")
    features = minirocket.transform(X)
    print(f"    Features shape: {features.shape}")
    
    print("  - Applying StandardScaler transform...")
    features_scaled = scaler.transform(features)
    
    print("  - Computing predict_proba...")
    ai_scores = lr.predict_proba(features_scaled)[:, 1]
    
    print(f"  - ai_scores range: [{ai_scores.min():.4f}, {ai_scores.max():.4f}]")
    print(f"  - ai_scores mean: {ai_scores.mean():.4f}")
    
    return ai_scores


def compute_metrics(
    valid_windows_df: pd.DataFrame,
) -> dict:
    """Compute metrics for each split."""
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    
    metrics = {}
    
    for split in ["train", "val", "test"]:
        split_df = valid_windows_df[valid_windows_df["split"] == split]
        if len(split_df) == 0:
            continue
            
        y_true = split_df["y_true"].values
        y_score = split_df["ai_score"].values
        
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        
        print(f"\n{split.upper()} Set:")
        print(f"  - Windows: {len(split_df)} ({n_pos} pos, {n_neg} neg)")
        print(f"  - ROC-AUC: {roc_auc:.4f}")
        print(f"  - PR-AUC: {pr_auc:.4f}")
        
        # Threshold analysis
        thresholds = [0.3, 0.5, 0.7]
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"  - @t={t}: Precision={precision:.3f}, Recall={recall:.3f}")
        
        metrics[split] = {
            "n_windows": len(split_df),
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        }
    
    return metrics


def save_scores(
    valid_windows_df: pd.DataFrame,
    pipeline: dict,
) -> None:
    """Save ai_scores.parquet."""
    print("\n" + "=" * 60)
    print("Saving ai_scores.parquet")
    print("=" * 60)
    
    # Select columns for output
    output_cols = [
        "record_id",
        "array_idx",
        "window_idx",
        "start",
        "end",
        "start_time_sec",
        "end_time_sec",
        "inv_rate_fhr",
        "inv_rate_uc",
        "quality_class",
        "valid_for_ai",
        "valid_for_rules",
        "touches_padding",
        "ph",
        "y_true",
        "ai_score",
        "split",
    ]
    
    # Filter to available columns
    available_cols = [c for c in output_cols if c in valid_windows_df.columns]
    output_df = valid_windows_df[available_cols].copy()
    
    # Add model metadata
    output_df["model_version"] = pipeline["model_version"]
    output_df["scored_at"] = datetime.now().isoformat()
    
    # Rename for clarity
    output_df = output_df.rename(columns={
        "start": "window_start",
        "end": "window_end",
    })
    
    # Save
    output_df.to_parquet(OUTPUT_SCORES_PATH, index=False)
    print(f"  - Saved: {OUTPUT_SCORES_PATH}")
    print(f"  - Rows: {len(output_df)}")
    
    # Show sample
    print("\nSample (5 rows):")
    sample_cols = ["record_id", "window_start", "window_end", "ai_score", "y_true", "split"]
    sample_cols = [c for c in sample_cols if c in output_df.columns]
    print(output_df[sample_cols].head().to_string())


def main():
    """Main evaluation flow."""
    print("=" * 60)
    print("Stage 3 Evaluation: Generate ai_scores.parquet")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Load data
    windows_df, X_norm, manifest_df = load_data()
    
    # Prepare windows
    X, y, groups, valid_windows_df = prepare_windows(windows_df, X_norm, manifest_df)
    
    # Assign splits
    valid_windows_df = assign_splits(X, y, groups, valid_windows_df)
    
    # Compute scores
    ai_scores = compute_scores(pipeline, X)
    valid_windows_df["ai_score"] = ai_scores
    
    # Compute metrics
    metrics = compute_metrics(valid_windows_df)
    
    # Save scores
    save_scores(valid_windows_df, pipeline)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Finished at: {datetime.now().isoformat()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
