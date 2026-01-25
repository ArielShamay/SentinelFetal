"""
Ensemble Classifier Adapter for V4.0 Hybrid Ensemble.
======================================================
Adapts the EnsembleManager to the IClassifier protocol for pipeline integration.

This adapter:
- Implements the IClassifier protocol
- Wraps the EnsembleManager for pipeline use
- Supports safety override via Rule Engine severity integration
- Provides backwards-compatible predict/predict_proba interface
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

from src.interfaces.protocols import IClassifier
from src.ml.ensemble_manager import EnsembleManager, get_ensemble_manager

logger = logging.getLogger(__name__)


class EnsembleClassifierAdapter:
    """
    Adapter to make EnsembleManager compatible with IClassifier protocol.
    
    This adapter wraps the V4.0 Hybrid Ensemble and provides:
    - Standard sklearn-style predict() and predict_proba() interface
    - Integration with the analysis pipeline
    - Support for rule engine safety override
    
    Usage:
        >>> adapter = EnsembleClassifierAdapter()
        >>> predictions = adapter.predict(X)
        >>> probabilities = adapter.predict_proba(X)
        
    With Rule Engine Override:
        >>> adapter.set_rule_engine_severity(0.75)  # High severity
        >>> predictions = adapter.predict(X)
    """
    
    def __init__(self, ensemble_manager: Optional[EnsembleManager] = None):
        """
        Initialize the adapter.
        
        Args:
            ensemble_manager: Optional EnsembleManager instance.
                             Uses singleton if not provided.
        """
        self._ensemble = ensemble_manager or get_ensemble_manager()
        self._rule_engine_severity: Optional[float] = None
        
        logger.info(f"EnsembleClassifierAdapter initialized. Models loaded: {self._ensemble.is_loaded}")
    
    def set_rule_engine_severity(self, severity: Optional[float]):
        """
        Set the rule engine severity for safety override.
        
        This should be called before predict() when rule engine results
        are available. The ensemble will use:
            Final_Risk = MAX(Ensemble_Risk, Rule_Engine_Severity)
        
        Args:
            severity: Rule engine severity score (0-1), or None to disable
        """
        self._rule_engine_severity = severity
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict categories for feature matrix.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Array of predicted categories (0, 1, or 2 for 0-indexed).
        """
        X = np.asarray(X, dtype=np.float32)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        
        for i in range(X.shape[0]):
            result = self._ensemble.predict(
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
            result = self._ensemble.predict(
                X[i],
                rule_engine_severity=self._rule_engine_severity
            )
            
            # Convert binary risk score to 3-class probabilities
            # risk_score represents P(Pathological)
            risk = result.final_risk_score
            
            # Map to 3 classes using thresholds
            thresholds = self._ensemble.thresholds
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
                ratio = risk / warning
                proba = [0.8 - 0.3 * ratio, 0.15 + 0.25 * ratio, 0.05 + 0.05 * ratio]
            
            # Normalize to sum to 1
            proba = np.array(proba)
            proba = proba / proba.sum()
            probabilities.append(proba)
        
        return np.array(probabilities, dtype=np.float32)
    
    def save_model(self, path: str) -> None:
        """
        Save model to file.
        
        Note: The EnsembleManager models are saved during training.
        This method is provided for interface compatibility.
        """
        logger.info(f"EnsembleManager models are saved during training to models/ensemble_v4/")
        logger.info(f"Request to save to {path} noted (no action taken)")
    
    def load_model(self, path: str) -> None:
        """
        Load model from file.
        
        Note: The EnsembleManager loads models automatically on initialization.
        This method is provided for interface compatibility.
        """
        logger.info(f"EnsembleManager loads models from models/ensemble_v4/ on init")
        logger.info(f"Request to load from {path} noted (no action taken)")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded ensemble."""
        return self._ensemble.get_model_info()
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded and ready for inference."""
        return self._ensemble.is_loaded


# Factory function for easy integration
def create_ensemble_classifier() -> EnsembleClassifierAdapter:
    """
    Factory function to create an EnsembleClassifierAdapter.
    
    Returns:
        Configured EnsembleClassifierAdapter instance.
    """
    return EnsembleClassifierAdapter()
