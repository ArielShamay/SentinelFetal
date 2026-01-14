"""
XGBoost Classifier for SentinelFetal.

This module implements the classifier that predicts fetal categories
based on the 1035-dimensional hybrid feature vectors.

Classes:
    0: Category 1 - Normal
    1: Category 2 - Intermediate (Suspicious)
    2: Category 3 - Pathological

The classifier uses XGBoost with Stratified K-Fold cross-validation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        accuracy_score,
        balanced_accuracy_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    """Configuration for XGBoost classifier."""
    
    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Regularization
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    
    # Training parameters
    n_folds: int = 5
    random_state: int = 42
    
    # Class weighting (for imbalanced data)
    scale_pos_weight: Optional[float] = None
    use_class_weights: bool = True
    
    # GPU acceleration
    tree_method: str = "auto"  # "auto", "hist", "gpu_hist"


@dataclass
class TrainingResult:
    """Results from classifier training."""
    
    cv_scores: list[float]
    mean_cv_score: float
    std_cv_score: float
    best_score: float
    feature_importance: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None


class XGBClassifierWrapper:
    """
    XGBoost classifier wrapper for SentinelFetal.
    
    This class provides:
        - Training with Stratified K-Fold cross-validation
        - Prediction with probability outputs
        - Model saving/loading
        - Feature importance analysis
    
    Example:
        >>> classifier = XGBClassifierWrapper()
        >>> result = classifier.train(X_train, y_train)
        >>> print(f"CV Score: {result.mean_cv_score:.3f}")
        >>> predictions = classifier.predict(X_test)
    """
    
    CLASS_NAMES = ['Normal (Cat 1)', 'Intermediate (Cat 2)', 'Pathological (Cat 3)']
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Initialize the classifier.
        
        Args:
            config: Classifier configuration. Uses defaults if None.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required. Install with: pip install xgboost"
            )
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            )
        
        self.config = config or ClassifierConfig()
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False
        self._feature_importance: Optional[np.ndarray] = None
    
    def _create_model(self, class_weights: Optional[dict] = None) -> xgb.XGBClassifier:
        """
        Create a new XGBoost classifier instance.
        
        Args:
            class_weights: Optional class weight dictionary.
            
        Returns:
            Configured XGBClassifier instance.
        """
        params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'min_child_weight': self.config.min_child_weight,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'gamma': self.config.gamma,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': self.config.random_state,
            'tree_method': self.config.tree_method,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'verbosity': 0
        }
        
        if self.config.scale_pos_weight is not None:
            params['scale_pos_weight'] = self.config.scale_pos_weight
        
        if class_weights is not None:
            params['sample_weight'] = class_weights
        
        return xgb.XGBClassifier(**params)
    
    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute sample weights for handling class imbalance.
        
        Uses inverse frequency weighting.
        
        Args:
            y: Label array.
            
        Returns:
            Array of sample weights.
        """
        unique, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(unique)
        
        # Compute inverse frequency weights
        weights = n_samples / (n_classes * counts)
        weight_dict = dict(zip(unique, weights))
        
        # Map to sample weights
        sample_weights = np.array([weight_dict[label] for label in y])
        
        return sample_weights
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validate: bool = True
    ) -> TrainingResult:
        """
        Train the classifier with Stratified K-Fold cross-validation.
        
        Args:
            X: Feature matrix of shape (n_samples, 1035).
            y: Label array of shape (n_samples,).
            validate: If True, perform cross-validation.
            
        Returns:
            TrainingResult with CV scores and metrics.
            
        Example:
            >>> result = classifier.train(X, y)
            >>> print(f"Mean CV F1: {result.mean_cv_score:.3f} ± {result.std_cv_score:.3f}")
        """
        logger.info(f"Training XGBoost classifier on {X.shape[0]} samples...")
        logger.info(f"Feature dimension: {X.shape[1]}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Compute class weights if enabled
        sample_weights = None
        if self.config.use_class_weights:
            sample_weights = self._compute_class_weights(y)
            logger.info("Using class weights for imbalanced data")
        
        cv_scores = []
        confusion_matrices = []
        
        if validate:
            # Stratified K-Fold cross-validation
            skf = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            
            logger.info(f"Performing {self.config.n_folds}-fold cross-validation...")
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                fold_weights = None
                if sample_weights is not None:
                    fold_weights = sample_weights[train_idx]
                
                model = self._create_model()
                model.fit(X_train, y_train, sample_weight=fold_weights)
                
                y_pred = model.predict(X_val)
                
                # Use macro F1 for multi-class
                f1 = f1_score(y_val, y_pred, average='macro')
                cv_scores.append(f1)
                
                cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
                confusion_matrices.append(cm)
                
                logger.info(f"  Fold {fold}: F1 = {f1:.3f}")
        
        # Train final model on all data
        logger.info("Training final model on all data...")
        self.model = self._create_model()
        self.model.fit(X, y, sample_weight=sample_weights)
        self.is_trained = True
        
        # Store feature importance
        self._feature_importance = self.model.feature_importances_
        
        # Final predictions for confusion matrix
        y_pred_final = self.model.predict(X)
        final_cm = confusion_matrix(y, y_pred_final, labels=[0, 1, 2])
        final_report = classification_report(
            y, y_pred_final,
            target_names=self.CLASS_NAMES,
            digits=3
        )
        
        # Aggregate results
        if cv_scores:
            mean_cv = np.mean(cv_scores)
            std_cv = np.std(cv_scores)
            best_cv = np.max(cv_scores)
        else:
            mean_cv = std_cv = best_cv = 0.0
        
        result = TrainingResult(
            cv_scores=cv_scores,
            mean_cv_score=mean_cv,
            std_cv_score=std_cv,
            best_score=best_cv,
            feature_importance=self._feature_importance,
            confusion_matrix=final_cm,
            classification_report=final_report
        )
        
        logger.info(f"Training complete!")
        logger.info(f"Mean CV F1: {mean_cv:.3f} ± {std_cv:.3f}")
        logger.info(f"\nConfusion Matrix:\n{final_cm}")
        logger.info(f"\nClassification Report:\n{final_report}")
        
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Feature matrix of shape (n_samples, 1035).
            
        Returns:
            Array of predicted labels (0, 1, or 2).
            
        Raises:
            RuntimeError: If model is not trained.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Feature matrix of shape (n_samples, 1035).
            
        Returns:
            Array of shape (n_samples, 3) with class probabilities.
            
        Raises:
            RuntimeError: If model is not trained.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Array of feature importance scores, or None if not trained.
        """
        return self._feature_importance
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model (without extension).
            
        Raises:
            RuntimeError: If model is not trained.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = path.with_suffix('.json')
        self.model.save_model(str(model_path))
        
        # Save config
        config_path = path.with_suffix('.config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the model file (without extension).
        """
        path = Path(path)
        model_path = path.with_suffix('.json')
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        self.is_trained = True
        
        # Load config if available
        config_path = path.with_suffix('.config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = ClassifierConfig(**config_dict)
        
        logger.info(f"Model loaded from {model_path}")


# Convenience function for simple training
def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[ClassifierConfig] = None
) -> tuple[XGBClassifierWrapper, TrainingResult]:
    """
    Train a classifier with default settings.
    
    Args:
        X: Feature matrix.
        y: Label array.
        config: Optional classifier configuration.
        
    Returns:
        Tuple of (trained classifier, training result).
    """
    classifier = XGBClassifierWrapper(config)
    result = classifier.train(X, y)
    return classifier, result
