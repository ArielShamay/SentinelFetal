"""
XGBoost V6 Classifier Adapter
=============================
Adapter implementing IClassifier protocol for V6 simplified pipeline.

This adapter wraps XGBoostOnlyClassifier and provides:
- Standard sklearn-style predict() and predict_proba() interface
- Integration with the analysis pipeline
- Support for rule engine safety override
- Drop-in replacement for EnsembleClassifierAdapter

Author: SentinelFetal ML Team
Version: 6.0.0
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from src.interfaces.protocols import IClassifier
from .xgboost_only_classifier import (
    XGBoostOnlyClassifier,
    get_xgboost_classifier,
    TOTAL_FEATURES,
)

logger = logging.getLogger(__name__)


class XGBoostV6Adapter:
    """
    Adapter to make XGBoostOnlyClassifier compatible with IClassifier protocol.

    This adapter wraps the V6 XGBoost-only classifier and provides:
    - Standard sklearn-style predict() and predict_proba() interface
    - Integration with the analysis pipeline
    - Support for rule engine safety override

    Usage:
        >>> adapter = XGBoostV6Adapter()
        >>> predictions = adapter.predict(X)
        >>> probabilities = adapter.predict_proba(X)

    With Rule Engine Override:
        >>> adapter.set_rule_engine_severity(0.75)  # High severity
        >>> predictions = adapter.predict(X)
    """

    def __init__(
        self,
        classifier: Optional[XGBoostOnlyClassifier] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize the adapter.

        Args:
            classifier: Optional XGBoostOnlyClassifier instance.
                        Uses singleton if not provided.
            model_path: Optional path to model file.
                        Only used if classifier is not provided.
        """
        if classifier is not None:
            self._classifier = classifier
        elif model_path is not None:
            self._classifier = XGBoostOnlyClassifier(model_path=Path(model_path))
        else:
            self._classifier = get_xgboost_classifier()

        self._rule_engine_severity: Optional[float] = None

        logger.info(f"XGBoostV6Adapter initialized. Model loaded: {self._classifier.is_loaded}")

    def set_rule_engine_severity(self, severity: Optional[float]):
        """
        Set the rule engine severity for safety override.

        This should be called before predict() when rule engine results
        are available. The classifier will use:
            Final_Risk = MAX(AI_Risk, Rule_Engine_Severity)

        Args:
            severity: Rule engine severity score (0-1), or None to disable
        """
        self._rule_engine_severity = severity

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict categories for feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
               Accepts 9,996 (MiniRocket) or 10,004 (full) features.

        Returns:
            Array of predicted categories (0, 1, or 2 for 0-indexed).
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []

        for i in range(X.shape[0]):
            result = self._classifier.predict(
                X[i],
                rule_engine_severity=self._rule_engine_severity
            )
            # Convert category (1,2,3) to 0-indexed (0,1,2)
            predictions.append(result.category - 1)

        return np.array(predictions, dtype=np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 3) with class probabilities.
            Columns: [P(Normal), P(Suspicious), P(Pathological)]
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        probabilities = []

        for i in range(X.shape[0]):
            result = self._classifier.predict(
                X[i],
                rule_engine_severity=self._rule_engine_severity
            )

            # Convert binary risk score to 3-class probabilities
            # risk_score represents P(Pathological)
            risk = result.final_risk_score

            # Map to 3 classes using thresholds
            thresholds = self._classifier.thresholds
            critical = thresholds.get('critical', 0.60)
            warning = thresholds.get('warning', 0.35)

            if risk > critical:
                # High risk: mostly pathological
                proba = [0.1, 0.2, 0.7 * risk / critical + 0.3]
            elif risk > warning:
                # Medium risk: mostly suspicious
                ratio = (risk - warning) / (critical - warning)
                proba = [0.1, 0.5 + 0.3 * ratio, 0.4 - 0.2 * ratio]
            else:
                # Low risk: mostly normal
                ratio = risk / warning if warning > 0 else 0
                proba = [0.8 - 0.3 * ratio, 0.15 + 0.25 * ratio, 0.05 + 0.05 * ratio]

            # Normalize to sum to 1
            proba = np.array(proba)
            proba = proba / proba.sum()
            probabilities.append(proba)

        return np.array(probabilities, dtype=np.float32)

    def save_model(self, path: str) -> None:
        """
        Save model to file.

        Note: The XGBoost model is loaded from disk; this method is provided
        for interface compatibility.
        """
        logger.info(f"XGBoost V6 model is at {self._classifier.model_path}")
        logger.info(f"Request to save to {path} noted (no action taken)")

    def load_model(self, path: str) -> None:
        """
        Load model from file.

        Note: Creates a new XGBoostOnlyClassifier with the specified path.
        """
        self._classifier = XGBoostOnlyClassifier(model_path=Path(path))
        logger.info(f"Loaded model from {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = self._classifier.get_model_info()
        info['adapter'] = 'XGBoostV6Adapter'
        info['expected_features'] = TOTAL_FEATURES
        return info

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._classifier.is_loaded


# Factory function for easy integration
def create_xgboost_v6_classifier(model_path: Optional[str] = None) -> XGBoostV6Adapter:
    """
    Factory function to create an XGBoostV6Adapter.

    Args:
        model_path: Optional path to model file.

    Returns:
        Configured XGBoostV6Adapter instance.
    """
    return XGBoostV6Adapter(model_path=model_path)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("=" * 60)
    print("SentinelFetal V6 XGBoost Adapter Test")
    print("=" * 60)

    adapter = create_xgboost_v6_classifier()

    print(f"\nModel Info:")
    info = adapter.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    if adapter.is_loaded:
        # Test inference
        print("\n--- Test Inference ---")
        # Simulate MiniRocket output (9,996 features)
        test_features = np.random.randn(9996)

        # Single prediction
        pred = adapter.predict(test_features)
        print(f"Single prediction: {pred[0]} (0=Normal, 1=Suspicious, 2=Pathological)")

        # Probabilities
        proba = adapter.predict_proba(test_features)
        print(f"Probabilities: Normal={proba[0,0]:.3f}, Suspicious={proba[0,1]:.3f}, Pathological={proba[0,2]:.3f}")

        # With rule engine
        print("\n--- With Rule Engine Override ---")
        adapter.set_rule_engine_severity(0.75)
        pred_override = adapter.predict(test_features)
        print(f"With rule override (0.75): {pred_override[0]}")

        # Batch prediction
        print("\n--- Batch Prediction ---")
        batch_features = np.random.randn(5, 9996)
        batch_preds = adapter.predict(batch_features)
        print(f"Batch predictions: {batch_preds}")
    else:
        print("\nModel not loaded. Place xgboost_v5.pkl in models/ensemble_v5/")

    sys.exit(0)
