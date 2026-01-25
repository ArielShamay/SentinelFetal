"""
SentinelFetal V4.0 — Hybrid Ensemble Manager
=============================================
Production inference engine implementing weighted soft voting across
XGBoost, Random Forest, and SGD Classifier.

SAFETY-CRITICAL COMPONENT
- All outputs are pessimistically aggregated with Rule Engine
- Probability calibration ensures reliable confidence scores
- Designed for < 100ms inference latency on CPU

Author: SentinelFetal ML Team
Version: 4.0.0
"""

import pickle
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "ensemble_v4.yaml"
MODELS_DIR = PROJECT_ROOT / "models" / "ensemble_v4"


@dataclass
class EnsemblePrediction:
    """Structured output from ensemble inference."""
    # Aggregated results
    risk_score: float              # Weighted ensemble probability (0-1)
    category: int                  # 1=Normal, 2=Suspicious, 3=Pathological
    category_name: str
    confidence: float              # Confidence in prediction
    
    # Individual model outputs
    model_probabilities: Dict[str, float] = field(default_factory=dict)
    model_predictions: Dict[str, int] = field(default_factory=dict)
    
    # Safety override tracking
    rule_engine_applied: bool = False
    rule_engine_severity: Optional[float] = None
    final_risk_score: float = 0.0
    
    # Metadata
    inference_time_ms: float = 0.0
    models_loaded: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'risk_score': self.risk_score,
            'final_risk_score': self.final_risk_score,
            'category': self.category,
            'category_name': self.category_name,
            'confidence': self.confidence,
            'model_probabilities': self.model_probabilities,
            'model_predictions': self.model_predictions,
            'rule_engine_applied': self.rule_engine_applied,
            'rule_engine_severity': self.rule_engine_severity,
            'inference_time_ms': self.inference_time_ms,
            'models_loaded': self.models_loaded
        }


class EnsembleManager:
    """
    V4.0 Hybrid Ensemble ML Engine
    
    Implements weighted soft voting across three calibrated classifiers:
    - XGBoost (weight: 0.4)
    - Random Forest (weight: 0.3)
    - SGD Classifier (weight: 0.3)
    
    Key Features:
    - Probability calibration for reliable confidence scores
    - Safety override: Final_Risk = MAX(Ensemble_Risk, Rule_Engine_Severity)
    - < 100ms inference latency constraint
    - Full audit trail and explainability support
    
    Usage:
        manager = EnsembleManager()
        result = manager.predict(features, rule_engine_severity=0.3)
    """
    
    # Default weights (loaded from config)
    DEFAULT_WEIGHTS = {
        'xgboost': 0.4,
        'random_forest': 0.3,
        'sgd_classifier': 0.3
    }
    
    # Risk thresholds (loaded from config)
    DEFAULT_THRESHOLDS = {
        'critical': 0.60,
        'warning': 0.35,
        'normal': 0.35
    }
    
    def __init__(self, config_path: Optional[Path] = None, models_dir: Optional[Path] = None):
        """
        Initialize the Ensemble Manager.
        
        Args:
            config_path: Path to ensemble_v4.yaml configuration
            models_dir: Directory containing trained model artifacts
        """
        self.config_path = config_path or CONFIG_PATH
        self.models_dir = models_dir or MODELS_DIR
        
        # Load configuration
        self.config = self._load_config()
        
        # Model weights (configurable, not hardcoded)
        self.weights = self._get_weights()
        
        # Risk thresholds
        self.thresholds = self._get_thresholds()
        
        # Model containers
        self.models = {}
        self.scaler = None
        self.is_loaded = False
        self.loaded_models = []
        
        # Feature expectations
        self.expected_feature_dim = self.config.get('inference', {}).get('feature_dim', 10000)
        
        # Load models
        self._load_models()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config not found at {self.config_path}, using defaults")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded ensemble config from {self.config_path}")
            return config or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _get_weights(self) -> Dict[str, float]:
        """Get model weights from config or defaults."""
        weights = self.config.get('ensemble', {}).get('weights', self.DEFAULT_WEIGHTS)
        
        # Validate weights sum to 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            weights = {k: v / total for k, v in weights.items()}
        
        logger.info(f"Ensemble weights: {weights}")
        return weights
    
    def _get_thresholds(self) -> Dict[str, float]:
        """Get risk thresholds from config or defaults."""
        return self.config.get('thresholds', self.DEFAULT_THRESHOLDS)
    
    def _load_models(self) -> bool:
        """Load all trained model artifacts."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            logger.info("Run 'python src/ml/training/train_v4_ensemble.py' to train models first.")
            return False
        
        artifacts = self.config.get('artifacts', {})
        
        # Load XGBoost
        xgb_path = PROJECT_ROOT / artifacts.get('xgboost_path', 'models/ensemble_v4/xgboost_calibrated.pkl')
        if xgb_path.exists():
            try:
                with open(xgb_path, 'rb') as f:
                    self.models['xgboost'] = pickle.load(f)
                self.loaded_models.append('xgboost')
                logger.info(f"Loaded XGBoost from {xgb_path}")
            except Exception as e:
                logger.error(f"Failed to load XGBoost: {e}")
        
        # Load Random Forest
        rf_path = PROJECT_ROOT / artifacts.get('random_forest_path', 'models/ensemble_v4/random_forest_calibrated.pkl')
        if rf_path.exists():
            try:
                with open(rf_path, 'rb') as f:
                    self.models['random_forest'] = pickle.load(f)
                self.loaded_models.append('random_forest')
                logger.info(f"Loaded Random Forest from {rf_path}")
            except Exception as e:
                logger.error(f"Failed to load Random Forest: {e}")
        
        # Load SGD Classifier
        sgd_path = PROJECT_ROOT / artifacts.get('sgd_path', 'models/ensemble_v4/sgd_calibrated.pkl')
        if sgd_path.exists():
            try:
                with open(sgd_path, 'rb') as f:
                    self.models['sgd_classifier'] = pickle.load(f)
                self.loaded_models.append('sgd_classifier')
                logger.info(f"Loaded SGD Classifier from {sgd_path}")
            except Exception as e:
                logger.error(f"Failed to load SGD Classifier: {e}")
        
        # Load feature scaler
        scaler_path = PROJECT_ROOT / artifacts.get('scaler_path', 'models/ensemble_v4/feature_scaler.pkl')
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded feature scaler from {scaler_path}")
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}")
        
        self.is_loaded = len(self.models) > 0
        
        if self.is_loaded:
            logger.info(f"Ensemble ready with {len(self.models)} models: {self.loaded_models}")
        else:
            logger.warning("No models loaded. Ensemble will return default predictions.")
        
        return self.is_loaded
    
    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess feature vector for inference.
        
        Args:
            features: Raw feature vector (MiniRocket output or manual features)
            
        Returns:
            Preprocessed feature vector
        """
        features = np.asarray(features, dtype=np.float32)
        
        # Ensure 2D shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply scaling if available
        if self.scaler is not None:
            try:
                features = self.scaler.transform(features)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}")
        
        return features
    
    def _get_model_probability(self, model_name: str, features: np.ndarray) -> Tuple[float, int]:
        """
        Get probability and prediction from a single model.
        
        Args:
            model_name: Name of the model ('xgboost', 'random_forest', 'sgd_classifier')
            features: Preprocessed feature vector
            
        Returns:
            Tuple of (probability of positive class, predicted class)
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not loaded, returning default")
            return 0.5, 0
        
        model = self.models[model_name]
        
        try:
            # Get probability
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)
                # Probability of pathological class (positive)
                prob_positive = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
            else:
                # Fallback for models without predict_proba
                prob_positive = 0.5
            
            # Get prediction
            pred = int(model.predict(features)[0])
            
            return prob_positive, pred
            
        except Exception as e:
            logger.error(f"Inference error for {model_name}: {e}")
            return 0.5, 0
    
    def _weighted_soft_vote(self, probabilities: Dict[str, float]) -> float:
        """
        Apply weighted soft voting to get ensemble risk score.
        
        Formula:
            Ensemble_Risk = 0.4 * P_xgb + 0.3 * P_rf + 0.3 * P_sgd
        
        Args:
            probabilities: Dictionary of model name -> probability
            
        Returns:
            Weighted ensemble risk score (0-1)
        """
        risk_score = 0.0
        total_weight = 0.0
        
        for model_name, weight in self.weights.items():
            if model_name in probabilities:
                risk_score += weight * probabilities[model_name]
                total_weight += weight
        
        # Normalize if not all models available
        if total_weight > 0 and total_weight < 1.0:
            risk_score /= total_weight
        
        return float(np.clip(risk_score, 0.0, 1.0))
    
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
    ) -> EnsemblePrediction:
        """
        Run ensemble inference on a feature vector.
        
        Args:
            features: Feature vector (MiniRocket output: 10,000 floats)
            rule_engine_severity: Optional Rule Engine severity score (0-1)
                                  If provided, final risk = MAX(ensemble, rule_engine)
        
        Returns:
            EnsemblePrediction with all outputs and metadata
        """
        start_time = time.perf_counter()
        
        # Handle unloaded state
        if not self.is_loaded:
            return self._default_prediction(rule_engine_severity)
        
        # Preprocess features
        X = self._preprocess_features(features)
        
        # Get individual model probabilities
        model_probabilities = {}
        model_predictions = {}
        
        for model_name in ['xgboost', 'random_forest', 'sgd_classifier']:
            if model_name in self.models:
                prob, pred = self._get_model_probability(model_name, X)
                model_probabilities[model_name] = prob
                model_predictions[model_name] = pred
        
        # Weighted soft voting
        ensemble_risk = self._weighted_soft_vote(model_probabilities)
        
        # Safety override: MAX(Ensemble_Risk, Rule_Engine_Severity)
        final_risk = ensemble_risk
        rule_engine_applied = False
        
        if rule_engine_severity is not None:
            if rule_engine_severity > ensemble_risk:
                final_risk = rule_engine_severity
                rule_engine_applied = True
                logger.debug(f"Rule Engine override: {ensemble_risk:.3f} -> {final_risk:.3f}")
        
        # Convert to category
        category, category_name = self._risk_to_category(final_risk)
        
        # Calculate confidence (agreement between models)
        if model_probabilities:
            probs = list(model_probabilities.values())
            confidence = 1.0 - np.std(probs)  # Higher agreement = higher confidence
            confidence = float(np.clip(confidence, 0.5, 1.0))
        else:
            confidence = 0.5
        
        # Calculate inference time
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Verify latency constraint
        if inference_time_ms > 100:
            logger.warning(f"Inference latency {inference_time_ms:.1f}ms exceeds 100ms constraint")
        
        return EnsemblePrediction(
            risk_score=ensemble_risk,
            category=category,
            category_name=category_name,
            confidence=confidence,
            model_probabilities=model_probabilities,
            model_predictions=model_predictions,
            rule_engine_applied=rule_engine_applied,
            rule_engine_severity=rule_engine_severity,
            final_risk_score=final_risk,
            inference_time_ms=inference_time_ms,
            models_loaded=self.loaded_models
        )
    
    def _default_prediction(self, rule_engine_severity: Optional[float] = None) -> EnsemblePrediction:
        """Return default prediction when models are not loaded."""
        # Use rule engine if available, otherwise conservative default
        if rule_engine_severity is not None:
            final_risk = rule_engine_severity
        else:
            final_risk = 0.5  # Conservative middle ground
        
        category, category_name = self._risk_to_category(final_risk)
        
        return EnsemblePrediction(
            risk_score=0.5,
            category=category,
            category_name=category_name,
            confidence=0.5,
            model_probabilities={},
            model_predictions={},
            rule_engine_applied=rule_engine_severity is not None,
            rule_engine_severity=rule_engine_severity,
            final_risk_score=final_risk,
            inference_time_ms=0.0,
            models_loaded=[]
        )
    
    def predict_batch(
        self, 
        features_batch: np.ndarray,
        rule_engine_severities: Optional[List[float]] = None
    ) -> List[EnsemblePrediction]:
        """
        Run batch inference.
        
        Args:
            features_batch: Batch of feature vectors (N x feature_dim)
            rule_engine_severities: Optional list of rule engine severities
            
        Returns:
            List of EnsemblePrediction objects
        """
        results = []
        n_samples = features_batch.shape[0]
        
        if rule_engine_severities is None:
            rule_engine_severities = [None] * n_samples
        
        for i in range(n_samples):
            result = self.predict(features_batch[i], rule_engine_severities[i])
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'is_loaded': self.is_loaded,
            'models_loaded': self.loaded_models,
            'weights': self.weights,
            'thresholds': self.thresholds,
            'config_path': str(self.config_path),
            'models_dir': str(self.models_dir),
            'version': self.config.get('version', {}).get('engine', '4.0.0')
        }


# =============================================================================
# Singleton Pattern for Global Access
# =============================================================================

_ensemble_manager_instance: Optional[EnsembleManager] = None


def get_ensemble_manager() -> EnsembleManager:
    """
    Get or create the global EnsembleManager instance.
    
    Returns:
        EnsembleManager singleton
    """
    global _ensemble_manager_instance
    
    if _ensemble_manager_instance is None:
        _ensemble_manager_instance = EnsembleManager()
    
    return _ensemble_manager_instance


def reset_ensemble_manager():
    """Reset the global EnsembleManager instance (for testing)."""
    global _ensemble_manager_instance
    _ensemble_manager_instance = None


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("SentinelFetal V4.0 Ensemble Manager")
    print("=" * 60)
    
    manager = get_ensemble_manager()
    
    print(f"\nModel Status:")
    info = manager.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    if manager.is_loaded:
        # Test inference with random features
        print("\n--- Test Inference ---")
        test_features = np.random.randn(32)  # Simplified test
        
        result = manager.predict(test_features, rule_engine_severity=0.3)
        
        print(f"Risk Score: {result.risk_score:.3f}")
        print(f"Final Risk (with Rule Engine): {result.final_risk_score:.3f}")
        print(f"Category: {result.category} - {result.category_name}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Rule Engine Override Applied: {result.rule_engine_applied}")
        print(f"Inference Time: {result.inference_time_ms:.2f} ms")
        print(f"Model Probabilities: {result.model_probabilities}")
    else:
        print("\nModels not trained. Run training script first:")
        print("  python src/ml/training/train_v4_ensemble.py")
    
    sys.exit(0)
