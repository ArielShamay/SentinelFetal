"""
Demo Model Training Script for SentinelFetal.

This script trains the XGBClassifierWrapper on the prepared dataset
(data/processed/X.npy, y.npy) and saves the model to models/xgb_demo.json.

Usage:
    python src/training/train_demo.py
    python src/training/train_demo.py --data-dir data/processed --output models/xgb_demo.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.models.classifier import XGBClassifierWrapper, ClassifierConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_demo_model(
    data_dir: str = "data/processed",
    output_path: str = "models/xgb_demo.json",
    n_folds: int = 3
) -> dict:
    """
    Train a demo XGBoost model on the prepared dataset.
    
    Args:
        data_dir: Directory containing X.npy and y.npy.
        output_path: Path to save the trained model.
        n_folds: Number of cross-validation folds.
        
    Returns:
        Dictionary with training results.
    """
    logger.info("=" * 60)
    logger.info("SentinelFetal Demo Model Training")
    logger.info("=" * 60)
    
    # Load data
    data_path = Path(data_dir)
    X_path = data_path / "X.npy"
    y_path = data_path / "y.npy"
    
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Please run prepare_data.py first.\n"
            f"Expected: {X_path}, {y_path}"
        )
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    logger.info(f"Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Handle case where not all 3 classes are present (demo dataset)
    # XGBoost expects contiguous class labels starting from 0
    if len(unique) < 3:
        logger.warning(f"Only {len(unique)} classes present in demo data. Relabeling for training.")
        # Create mapping: existing labels -> 0, 1, 2, ...
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique)}
        y_original = y.copy()
        y = np.array([label_map[label] for label in y])
        logger.info(f"Label mapping: {label_map}")
        logger.info(f"New class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        # Store reverse mapping for inference
        reverse_map = {v: k for k, v in label_map.items()}
    else:
        reverse_map = {0: 0, 1: 1, 2: 2}
    
    # Recompute unique/counts after potential relabeling
    unique, counts = np.unique(y, return_counts=True)
    
    # Check if we have enough samples for k-fold
    min_class_count = min(counts)
    if min_class_count < n_folds:
        logger.warning(
            f"Minimum class count ({min_class_count}) is less than n_folds ({n_folds}). "
            f"Reducing n_folds to {min_class_count}."
        )
        n_folds = max(2, min_class_count)
    
    # Create classifier
    config = ClassifierConfig(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        n_folds=n_folds,
        random_state=42
    )
    
    classifier = XGBClassifierWrapper(config)
    
    # For demo with limited classes, skip CV to avoid label mismatch issues
    if len(unique) < 3:
        logger.warning("Limited classes - training without cross-validation for demo")
        result = classifier.train(X, y, validate=False)
    else:
        # Train with cross-validation (the train method does CV + final training)
        logger.info(f"Training with {n_folds}-fold cross-validation...")
        result = classifier.train(X, y, validate=True)
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Mean CV F1: {result.mean_cv_score:.4f} (+/- {result.std_cv_score:.4f})")
    logger.info(f"Best CV F1: {result.best_score:.4f}")
    logger.info("Per-fold F1 scores: " + 
                ", ".join([f"{score:.4f}" for score in result.cv_scores]))
    logger.info("=" * 60)
    
    # Model is already trained on all data by train()
    
    # Save model
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(output_file))
    logger.info(f"Model saved to: {output_file}")
    
    return {
        "mean_cv_score": result.mean_cv_score,
        "std_cv_score": result.std_cv_score,
        "best_score": result.best_score,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "model_path": str(output_file)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train demo XGBoost model for SentinelFetal"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing X.npy and y.npy"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/xgb_demo.json",
        help="Output path for the trained model"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds"
    )
    
    args = parser.parse_args()
    
    try:
        results = train_demo_model(
            data_dir=args.data_dir,
            output_path=args.output,
            n_folds=args.n_folds
        )
        logger.info("Training completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
