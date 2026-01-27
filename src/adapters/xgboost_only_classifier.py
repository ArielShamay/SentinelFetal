"""
SentinelFetal V6 - XGBoost-Only Classifier
==========================================
Simplified classifier that replaces the 3-model ensemble with a single XGBoost model.

This module provides:
- XGBoostOnlyClassifier: Core classifier wrapping xgboost_v5.pkl
- Feature padding for MiniRocket → XGBoost compatibility
- Rule Engine safety override (MAX aggregation)
- Compatible output structure with EnsemblePrediction

Author: SentinelFetal ML Team
Version: 6.0.0
"""

import pickle
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ensemble_v5" / "xgboost_v5.pkl"
ALT_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_v5.pkl"

# Feature dimensions
MINIROCKET_FEATURES = 9996
CLINICAL_FEATURES = 8
TOTAL_FEATURES = MINIROCKET_FEATURES + CLINICAL_FEATURES  # 10,004


@dataclass
class XGBoostPrediction:
    """Structured output from XGBoost-only inference."""
    # Core results
    risk_score: float              # Raw XGBoost probability (0-1)
    category: int                  # 1=Normal, 2=Suspicious, 3=Pathological
    category_name: str
    confidence: float              # Confidence in prediction

    # Safety override tracking
    rule_engine_applied: bool = False
    rule_engine_severity: Optional[float] = None
    final_risk_score: float = 0.0

    # Metadata
    inference_time_ms: float = 0.0
    model_name: str = "xgboost_v5"
    feature_dim: int = TOTAL_FEATURES

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'risk_score': self.risk_score,
            'final_risk_score': self.final_risk_score,
            'category': self.category,
            'category_name': self.category_name,
            'confidence': self.confidence,
            'rule_engine_applied': self.rule_engine_applied,
            'rule_engine_severity': self.rule_engine_severity,
            'inference_time_ms': self.inference_time_ms,
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            # Compatibility with EnsemblePrediction
            'model_probabilities': {'xgboost_v5': self.risk_score},
            'model_predictions': {'xgboost_v5': self.category},
            'models_loaded': ['xgboost_v5']
        }


def pad_minirocket_features(minirocket_features: np.ndarray) -> np.ndarray:
    """
    Pad MiniRocket features with zeros for clinical features.

    The xgboost_v5.pkl model was trained on V5 "super features":
    - MiniRocket: 9,996 features
    - Clinical features: 8 features (baseline_fhr, stv, etc.)
    - Total: 10,004 features

    This function pads MiniRocket output with zeros for the clinical features,
    allowing inference without computing clinical metrics.

    Args:
        minirocket_features: MiniRocket output of shape (9996,) or (N, 9996)

    Returns:
        Padded features of shape (10004,) or (N, 10004)
    """
    features = np.asarray(minirocket_features, dtype=np.float32)

    if features.ndim == 1:
        # Single sample
        if features.shape[0] == TOTAL_FEATURES:
            return features  # Already padded
        elif features.shape[0] == MINIROCKET_FEATURES:
            padded = np.zeros(TOTAL_FEATURES, dtype=np.float32)
            padded[:MINIROCKET_FEATURES] = features
            return padded
        else:
            logger.warning(f"Unexpected feature dim {features.shape[0]}, padding to {TOTAL_FEATURES}")
            padded = np.zeros(TOTAL_FEATURES, dtype=np.float32)
            padded[:min(features.shape[0], TOTAL_FEATURES)] = features[:TOTAL_FEATURES]
            return padded
    else:
        # Batch
        n_samples = features.shape[0]
        if features.shape[1] == TOTAL_FEATURES:
            return features
        elif features.shape[1] == MINIROCKET_FEATURES:
            padded = np.zeros((n_samples, TOTAL_FEATURES), dtype=np.float32)
            padded[:, :MINIROCKET_FEATURES] = features
            return padded
        else:
            logger.warning(f"Unexpected feature dim {features.shape[1]}, padding to {TOTAL_FEATURES}")
            padded = np.zeros((n_samples, TOTAL_FEATURES), dtype=np.float32)
            padded[:, :min(features.shape[1], TOTAL_FEATURES)] = features[:, :TOTAL_FEATURES]
            return padded


class XGBoostOnlyClassifier:
    """
    V6 XGBoost-Only Classifier

    Simplified classifier that uses only XGBoost for prediction, replacing
    the 3-model ensemble (XGBoost + RandomForest + SGD).

    Key Features:
    - Single XGBoost model for faster inference
    - Automatic feature padding for MiniRocket compatibility
    - Safety override: Final_Risk = MAX(AI_Risk, Rule_Engine_Severity)
    - Compatible output format with EnsemblePrediction

    Usage:
        >>> classifier = XGBoostOnlyClassifier()
        >>> result = classifier.predict(minirocket_features, rule_engine_severity=0.3)
        >>> print(result.category)  # 1, 2, or 3
    """

    DEFAULT_THRESHOLDS = {
        'critical': 0.60,
        'warning': 0.35,
    }

    def __init__(
        self,
        model_path: Optional[Path] = None,
        thresholds: Optional[Dict[str, float]] = None,
        auto_load: bool = True
    ):
        """
        Initialize the XGBoost-Only Classifier.

        Args:
            model_path: Path to xgboost_v5.pkl. Uses default if not provided.
            thresholds: Risk thresholds dict with 'critical' and 'warning' keys.
            auto_load: Whether to load model on initialization.
        """
        self.model_path = Path(model_path) if model_path else None
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()

        self.model = None
        self.is_loaded = False
        self.model_info: Dict[str, Any] = {}

        if auto_load:
            self._load_model()

    def _find_model_path(self) -> Optional[Path]:
        """Find the XGBoost model file."""
        if self.model_path and self.model_path.exists():
            return self.model_path

        # Try default paths
        for path in [DEFAULT_MODEL_PATH, ALT_MODEL_PATH]:
            if path.exists():
                return path

        return None

    def _load_model(self) -> bool:
        """Load the XGBoost model from disk."""
        model_path = self._find_model_path()

        if model_path is None:
            logger.warning("XGBoost model not found. Classifier will return default predictions.")
            return False

        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            self.model_path = model_path
            self.is_loaded = True

            # Collect model info
            self.model_info = {
                'path': str(model_path),
                'type': type(self.model).__name__,
            }

            # Check if CalibratedClassifierCV
            if hasattr(self.model, 'calibrated_classifiers_'):
                self.model_info['calibrated'] = True
                self.model_info['n_calibrators'] = len(self.model.calibrated_classifiers_)
                base = self.model.calibrated_classifiers_[0].estimator
                if hasattr(base, 'n_features_in_'):
                    self.model_info['n_features_in'] = base.n_features_in_
            else:
                self.model_info['calibrated'] = False
                if hasattr(self.model, 'n_features_in_'):
                    self.model_info['n_features_in'] = self.model.n_features_in_

            if hasattr(self.model, 'classes_'):
                self.model_info['classes'] = list(self.model.classes_)

            logger.info(f"Loaded XGBoost model from {model_path}")
            logger.info(f"Model info: {self.model_info}")

            return True

        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False

    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess features for inference.

        Handles:
        - Type conversion to float32
        - Shape normalization to 2D
        - NaN/Inf replacement
        - Feature padding if needed
        """
        features = np.asarray(features, dtype=np.float32)

        # Ensure 2D shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad features if needed
        if features.shape[1] < TOTAL_FEATURES:
            features = pad_minirocket_features(features)

        return features

    def _get_probability(self, features: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Get probability from the model.

        Args:
            features: Preprocessed feature matrix (N, 10004)

        Returns:
            Tuple of (risk_score, full_proba_array)
        """
        if not self.is_loaded:
            return 0.5, np.array([[0.5, 0.5]])

        try:
            proba = self.model.predict_proba(features)
            # Probability of positive/pathological class
            if proba.shape[1] > 1:
                risk_score = float(proba[0, 1])
            else:
                risk_score = float(proba[0, 0])
            return risk_score, proba
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5, np.array([[0.5, 0.5]])

    def _risk_to_category(self, risk_score: float) -> Tuple[int, str]:
        """
        Convert risk score to category.

        Args:
            risk_score: Probability-based risk score (0-1)

        Returns:
            Tuple of (category_int, category_name)
        """
        critical = self.thresholds.get('critical', 0.60)
        warning = self.thresholds.get('warning', 0.35)

        if risk_score > critical:
            return 3, "Pathological"
        elif risk_score > warning:
            return 2, "Suspicious"
        else:
            return 1, "Normal"

    def predict(
        self,
        features: np.ndarray,
        rule_engine_severity: Optional[float] = None
    ) -> XGBoostPrediction:
        """
        Run XGBoost inference on a feature vector.

        Args:
            features: Feature vector (MiniRocket output: 9,996 or 10,004 floats)
            rule_engine_severity: Optional Rule Engine severity score (0-1)
                                  If provided, final risk = MAX(ai_risk, rule_engine)

        Returns:
            XGBoostPrediction with all outputs and metadata
        """
        start_time = time.perf_counter()

        # Handle unloaded state
        if not self.is_loaded:
            return self._default_prediction(rule_engine_severity)

        # Preprocess features (includes padding)
        X = self._preprocess_features(features)

        # Get probability
        risk_score, _ = self._get_probability(X)

        # Safety override: MAX(AI_Risk, Rule_Engine_Severity)
        final_risk = risk_score
        rule_engine_applied = False

        if rule_engine_severity is not None:
            if rule_engine_severity > risk_score:
                final_risk = rule_engine_severity
                rule_engine_applied = True
                logger.debug(f"Rule Engine override: {risk_score:.3f} -> {final_risk:.3f}")

        # Convert to category
        category, category_name = self._risk_to_category(final_risk)

        # Calculate confidence (distance from decision boundary)
        # Higher confidence when further from thresholds
        confidence = self._calculate_confidence(risk_score)

        # Calculate inference time
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        return XGBoostPrediction(
            risk_score=risk_score,
            category=category,
            category_name=category_name,
            confidence=confidence,
            rule_engine_applied=rule_engine_applied,
            rule_engine_severity=rule_engine_severity,
            final_risk_score=final_risk,
            inference_time_ms=inference_time_ms,
            model_name="xgboost_v5",
            feature_dim=TOTAL_FEATURES
        )

    def _calculate_confidence(self, risk_score: float) -> float:
        """
        Calculate confidence based on distance from decision boundaries.

        Confidence is higher when:
        - Risk is very low (clearly normal)
        - Risk is very high (clearly pathological)

        Confidence is lower when:
        - Risk is near thresholds (uncertain)
        """
        critical = self.thresholds.get('critical', 0.60)
        warning = self.thresholds.get('warning', 0.35)

        # Distance from nearest threshold
        dist_to_warning = abs(risk_score - warning)
        dist_to_critical = abs(risk_score - critical)
        min_dist = min(dist_to_warning, dist_to_critical)

        # Normalize to 0.5-1.0 range (never fully uncertain)
        confidence = 0.5 + min(min_dist, 0.35) * (0.5 / 0.35)
        return float(np.clip(confidence, 0.5, 1.0))

    def _default_prediction(self, rule_engine_severity: Optional[float] = None) -> XGBoostPrediction:
        """Return default prediction when model is not loaded."""
        if rule_engine_severity is not None:
            final_risk = rule_engine_severity
        else:
            final_risk = 0.5  # Conservative middle ground

        category, category_name = self._risk_to_category(final_risk)

        return XGBoostPrediction(
            risk_score=0.5,
            category=category,
            category_name=category_name,
            confidence=0.5,
            rule_engine_applied=rule_engine_severity is not None,
            rule_engine_severity=rule_engine_severity,
            final_risk_score=final_risk,
            inference_time_ms=0.0,
            model_name="xgboost_v5_default",
            feature_dim=TOTAL_FEATURES
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get raw probabilities from the model.

        Args:
            features: Feature matrix (N, 9996 or 10004)

        Returns:
            Probability array of shape (N, 2) for [P(Normal/Suspicious), P(Pathological)]
        """
        if not self.is_loaded:
            n_samples = features.shape[0] if features.ndim > 1 else 1
            return np.full((n_samples, 2), 0.5)

        X = self._preprocess_features(features)
        return self.model.predict_proba(X)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'is_loaded': self.is_loaded,
            'model_path': str(self.model_path) if self.model_path else None,
            'thresholds': self.thresholds,
            'model_info': self.model_info,
            'version': '6.0.0'
        }


# =============================================================================
# Singleton Pattern for Global Access
# =============================================================================

_xgboost_classifier_instance: Optional[XGBoostOnlyClassifier] = None


def get_xgboost_classifier() -> XGBoostOnlyClassifier:
    """
    Get or create the global XGBoostOnlyClassifier instance.

    Returns:
        XGBoostOnlyClassifier singleton
    """
    global _xgboost_classifier_instance

    if _xgboost_classifier_instance is None:
        _xgboost_classifier_instance = XGBoostOnlyClassifier()

    return _xgboost_classifier_instance


def reset_xgboost_classifier():
    """Reset the global XGBoostOnlyClassifier instance (for testing)."""
    global _xgboost_classifier_instance
    _xgboost_classifier_instance = None


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("=" * 60)
    print("SentinelFetal V6 XGBoost-Only Classifier")
    print("=" * 60)

    classifier = get_xgboost_classifier()

    print(f"\nModel Status:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    if classifier.is_loaded:
        # Test inference with random features (simulating MiniRocket output)
        print("\n--- Test Inference ---")
        test_features = np.random.randn(MINIROCKET_FEATURES)  # 9,996 features

        result = classifier.predict(test_features, rule_engine_severity=0.3)

        print(f"Risk Score: {result.risk_score:.3f}")
        print(f"Final Risk (with Rule Engine): {result.final_risk_score:.3f}")
        print(f"Category: {result.category} - {result.category_name}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Rule Engine Override Applied: {result.rule_engine_applied}")
        print(f"Inference Time: {result.inference_time_ms:.2f} ms")
    else:
        print("\nModel not loaded. Place xgboost_v5.pkl in models/ensemble_v5/")

    sys.exit(0)
