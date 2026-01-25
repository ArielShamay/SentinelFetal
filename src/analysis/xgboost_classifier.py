"""
XGBoost CTG Classifier - Production Runtime Module
===================================================
Loads the trained XGBoost pipeline and provides real-time classification.

Updated to support REAL DATA pipeline (v2.0-realdata) format.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'ctg_xgboost_pipeline.pkl'


class XGBoostCTGClassifier:
    """
    Production classifier using trained XGBoost model.
    
    Supports both legacy (synthetic) and new (real data) pipeline formats.
    
    Provides:
    - Category prediction (1=Normal, 2=Suspicious, 3=Pathological)
    - Probability scores for confidence display
    - Feature importance for explainability
    """
    
    # Updated for 1-indexed categories (FIGO standard)
    CATEGORY_NAMES = {
        1: 'Category I (Normal)',
        2: 'Category II (Suspicious)',
        3: 'Category III (Pathological)'
    }
    
    CATEGORY_COLORS = {
        1: 'green',
        2: 'yellow', 
        3: 'red'
    }
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the trained pipeline. Uses default if not specified.
        """
        self.model_path = model_path or MODEL_PATH
        self.pipeline = None
        self.model = None
        self.scaler = None
        self.feature_extractor = None  # Legacy support
        self.feature_names = None
        self.is_loaded = False
        self.metrics = {}
        self.is_realdata_model = False
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the trained model pipeline."""
        if not self.model_path.exists():
            logger.warning(f"Model file not found: {self.model_path}")
            logger.info("Run 'python scripts/train_xgboost_realdata.py' to train the model first.")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            
            self.model = self.pipeline['model']
            
            # Support both old and new pipeline formats
            if 'scaler' in self.pipeline:
                # New real-data format (v2.0-realdata)
                self.scaler = self.pipeline['scaler']
                self.feature_names = self.pipeline.get('feature_names', [])
                self.is_realdata_model = True
                self.metrics = self.pipeline.get('cv_results', {})
            elif 'feature_extractor' in self.pipeline:
                # Legacy synthetic format
                self.feature_extractor = self.pipeline['feature_extractor']
                self.scaler = getattr(self.feature_extractor, 'scaler', None)
                self.is_realdata_model = False
                self.metrics = self.pipeline.get('metrics', {})
            
            self.is_loaded = True
            
            version = self.pipeline.get('version', 'unknown')
            logger.info(f"XGBoost model loaded from {self.model_path}")
            logger.info(f"   Model version: {version}")
            
            if self.is_realdata_model:
                logger.info(f"   Data source: {self.pipeline.get('data_source', 'Unknown')}")
                logger.info(f"   CV Accuracy: {self.metrics.get('mean_accuracy', 0):.1%}")
            else:
                logger.info(f"   Training accuracy: {self.metrics.get('accuracy', 0):.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _extract_features_manual(self, fhr: np.ndarray, uc: np.ndarray) -> np.ndarray:
        """
        Extract features manually (for real-data model).
        Must match the feature extraction in train_xgboost_realdata.py
        """
        # Handle invalid FHR values
        valid_mask = (fhr > 50) & (fhr < 210)
        fhr_valid = fhr[valid_mask] if valid_mask.any() else fhr
        
        sampling_rate = 4  # Default CTG sampling rate
        
        features = []
        
        # FHR Statistical Features (9)
        features.append(np.mean(fhr_valid))
        features.append(np.std(fhr_valid))
        features.append(np.min(fhr_valid))
        features.append(np.max(fhr_valid))
        features.append(np.max(fhr_valid) - np.min(fhr_valid))
        features.append(np.median(fhr_valid))
        features.append(np.percentile(fhr_valid, 25))
        features.append(np.percentile(fhr_valid, 75))
        features.append(np.percentile(fhr_valid, 75) - np.percentile(fhr_valid, 25))
        
        # Baseline & Variability (4)
        baseline = np.median(fhr_valid)  # Simplified baseline
        features.append(baseline)
        features.append(np.std(fhr_valid))  # variability
        features.append(np.mean(np.abs(np.diff(fhr_valid))) if len(fhr_valid) > 1 else 0)  # STV
        features.append(np.std(fhr_valid))  # LTV (simplified)
        
        # Pathological indicators (4)
        features.append(np.sum(fhr_valid < 110) / len(fhr_valid))  # time below 110
        features.append(np.sum(fhr_valid > 160) / len(fhr_valid))  # time above 160
        features.append(0)  # brady episodes (simplified)
        features.append(0)  # tachy episodes (simplified)
        
        # Deceleration features (5)
        features.append(0)  # decel_count
        features.append(0)  # decel_mean_depth
        features.append(0)  # decel_max_depth
        features.append(0)  # decel_total_duration
        features.append(0)  # late_decel_score
        
        # Acceleration features (2)
        features.append(0)  # accel_count
        features.append(0)  # accel_mean_amplitude
        
        # UC features (4)
        uc_valid = uc[uc > 0] if (uc > 0).any() else uc
        features.append(np.mean(uc_valid))
        features.append(np.std(uc_valid))
        features.append(0)  # uc_frequency (simplified)
        features.append(1.0)  # uc_regularity
        
        # Cross-correlation (2)
        min_len = min(len(fhr_valid), len(uc))
        if min_len > 10:
            corr = np.corrcoef(fhr_valid[:min_len], uc[:min_len])[0, 1]
            features.append(corr if not np.isnan(corr) else 0)
        else:
            features.append(0)
        features.append(0)  # lag correlation (simplified)
        
        # Signal quality (2)
        features.append(np.sum(valid_mask) / len(fhr))
        features.append(0)  # gaps
        
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def _prepare_sample_legacy(self, fhr: np.ndarray, uc: np.ndarray) -> np.ndarray:
        """
        Prepare a single sample for classification (legacy format).
        """
        fhr = np.asarray(fhr, dtype=np.float32)
        uc = np.asarray(uc, dtype=np.float32)
        
        min_len = min(len(fhr), len(uc))
        fhr = fhr[:min_len]
        uc = uc[:min_len]
        
        sample = np.stack([fhr, uc], axis=0)
        return sample[np.newaxis, ...]
    
    def classify(self, fhr: np.ndarray, uc: np.ndarray) -> Dict[str, Any]:
        """
        Classify a CTG trace.
        
        Args:
            fhr: FHR signal array (at least 60 seconds of data recommended)
            uc: UC (contraction) signal array
            
        Returns:
            Dictionary with:
            - category: int (1, 2, or 3 for real-data model; 0,1,2 for legacy)
            - category_name: str
            - confidence: float (0-1)
            - probabilities: dict mapping category to probability
            - color: str (green/yellow/red)
        """
        if not self.is_loaded:
            logger.warning("Model not loaded, returning default classification")
            return self._default_result()
        
        try:
            # Ensure numpy arrays
            fhr = np.asarray(fhr, dtype=np.float32)
            uc = np.asarray(uc, dtype=np.float32)
            
            if self.is_realdata_model:
                # New real-data pipeline
                X_features = self._extract_features_manual(fhr, uc)
                X_scaled = self.scaler.transform(X_features)
                
                # Model predicts 0-indexed (0,1,2), convert to 1-indexed (1,2,3)
                category_0idx = int(self.model.predict(X_scaled)[0])
                category = category_0idx + 1  # Convert to 1-indexed
                probabilities = self.model.predict_proba(X_scaled)[0]
                
                confidence = float(probabilities[category_0idx])
                
                return {
                    'category': category,
                    'category_name': self.CATEGORY_NAMES.get(category, f'Category {category}'),
                    'confidence': confidence,
                    'probabilities': {
                        1: float(probabilities[0]),
                        2: float(probabilities[1]),
                        3: float(probabilities[2]) if len(probabilities) > 2 else 0.0
                    },
                    'color': self.CATEGORY_COLORS.get(category, 'gray')
                }
            else:
                # Legacy synthetic pipeline
                X_raw = self._prepare_sample_legacy(fhr, uc)
                X_features = self.feature_extractor.transform(X_raw)
                
                category = int(self.model.predict(X_features)[0])
                probabilities = self.model.predict_proba(X_features)[0]
                confidence = float(probabilities[category])
                
                return {
                    'category': category,
                    'category_name': self.CATEGORY_NAMES.get(category + 1, f'Category {category}'),
                    'confidence': confidence,
                    'probabilities': {
                        0: float(probabilities[0]),
                        1: float(probabilities[1]),
                        2: float(probabilities[2]) if len(probabilities) > 2 else 0.0
                    },
                    'color': self.CATEGORY_COLORS.get(category + 1, 'gray')
                }
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
            return self._default_result()
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result when model unavailable."""
        return {
            'category': 1,
            'category_name': self.CATEGORY_NAMES[1],
            'confidence': 0.5,
            'probabilities': {1: 0.5, 2: 0.3, 3: 0.2},
            'color': 'green'
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the model."""
        if not self.is_loaded or not hasattr(self.model, 'feature_importances_'):
            return None
        
        feature_names = [
            'fhr_mean', 'fhr_std', 'fhr_min', 'fhr_max',
            'fhr_q25', 'fhr_q75', 'fhr_median',
            'stv', 'stv_std',
            'below_110', 'below_100', 'above_160',
            'uc_mean', 'uc_std', 'uc_max',
            'fhr_uc_corr'
        ]
        
        importances = self.model.feature_importances_
        
        # Match names to importances
        return dict(zip(feature_names[:len(importances)], importances.tolist()))


# Singleton instance for easy import
_classifier_instance: Optional[XGBoostCTGClassifier] = None


def get_classifier() -> XGBoostCTGClassifier:
    """Get or create the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = XGBoostCTGClassifier()
    return _classifier_instance


def classify_ctg(fhr: np.ndarray, uc: np.ndarray) -> Dict[str, Any]:
    """
    Convenience function to classify CTG data.
    
    Args:
        fhr: FHR signal array
        uc: UC signal array
        
    Returns:
        Classification result dictionary
    """
    return get_classifier().classify(fhr, uc)
