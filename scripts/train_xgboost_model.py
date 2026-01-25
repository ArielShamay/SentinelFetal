"""
XGBoost Training Script with Stratified Cross-Validation
=========================================================
Senior ML Engineer approach for CTG classification with imbalanced data handling.

This script:
1. Generates synthetic training data using patient_generator
2. Extracts features using MiniRocket
3. Trains XGBoost with proper class weighting
4. Uses Stratified 10-Fold CV for robust validation
5. Saves the complete pipeline for production use
"""

import sys
import os
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info(f"✅ XGBoost version: {xgb.__version__}")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("⚠️ XGBoost not installed. Will use fallback classifier.")

# Try to import MiniRocket
try:
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    MINIROCKET_AVAILABLE = True
    logger.info("✅ MiniRocket available")
except ImportError:
    MINIROCKET_AVAILABLE = False
    logger.warning("⚠️ MiniRocket not available. Will use manual feature extraction.")


class SyntheticDataGenerator:
    """Generates labeled synthetic CTG data for training."""
    
    # Event configurations with clinical significance
    EVENT_CONFIGS = {
        'normal': {
            'events': [],
            'label': 0,  # Category I (Normal)
            'description': 'Normal baseline with good variability',
            'fhr_pattern': 'normal'
        },
        'early_decel': {
            'events': ['EARLY_DECELERATION'],
            'label': 0,  # Category I - Early decels are usually benign
            'description': 'Early deceleration (head compression)',
            'fhr_pattern': 'early_decel'
        },
        'late_decel_single': {
            'events': ['LATE_DECELERATION'],
            'label': 1,  # Category II (Indeterminate)
            'description': 'Single late deceleration',
            'fhr_pattern': 'late_decel'
        },
        'late_decel_recurrent': {
            'events': ['LATE_DECELERATION'] * 3,
            'label': 2,  # Category III (Pathological)
            'description': 'Recurrent late decelerations (>=50% contractions)',
            'fhr_pattern': 'late_decel_recurrent'
        },
        'variable_decel': {
            'events': ['VARIABLE_DECELERATION'],
            'label': 1,  # Category II
            'description': 'Variable deceleration (cord compression)',
            'fhr_pattern': 'variable_decel'
        },
        'prolonged_decel': {
            'events': ['PROLONGED_DECELERATION'],
            'label': 2,  # Category III
            'description': 'Prolonged deceleration (>2 min)',
            'fhr_pattern': 'prolonged_decel'
        },
        'bradycardia': {
            'events': ['BRADYCARDIA'],
            'label': 2,  # Category III
            'description': 'Fetal bradycardia (<110 bpm for >10 min)',
            'fhr_pattern': 'bradycardia'
        },
        'tachycardia': {
            'events': ['TACHYCARDIA'],
            'label': 1,  # Category II
            'description': 'Fetal tachycardia (>160 bpm)',
            'fhr_pattern': 'tachycardia'
        },
        'reduced_variability': {
            'events': ['REDUCED_VARIABILITY'],
            'label': 1,  # Category II
            'description': 'Reduced baseline variability (<5 bpm)',
            'fhr_pattern': 'reduced_variability'
        },
        'acceleration': {
            'events': ['ACCELERATION'],
            'label': 0,  # Category I - Accelerations are reassuring
            'description': 'Fetal accelerations (reactive pattern)',
            'fhr_pattern': 'acceleration'
        },
    }
    
    def __init__(self, duration_seconds: int = 600, sample_rate: int = 4):
        """
        Initialize generator.
        
        Args:
            duration_seconds: Length of each sample in seconds (default 10 min)
            sample_rate: Samples per second (default 4 Hz)
        """
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.samples_per_trace = duration_seconds * sample_rate
        
    def _generate_contraction(self, t: float, period: float = 180, duration: float = 60) -> float:
        """Generate a single contraction waveform."""
        cycle_pos = t % period
        if cycle_pos < duration:
            # Gaussian-like contraction shape
            peak_time = duration / 2
            sigma = duration / 4
            intensity = 80 * np.exp(-((cycle_pos - peak_time) ** 2) / (2 * sigma ** 2))
            return intensity + np.random.normal(0, 2)
        return np.random.normal(10, 2)  # Baseline tonus
    
    def _generate_fhr_normal(self, t: float, baseline: float, variability: float) -> float:
        """Generate normal FHR with good variability."""
        # Baseline with random walk variability
        noise = np.random.normal(0, variability / 2)
        slow_wave = 5 * np.sin(2 * np.pi * t / 60)  # ~1 min cycle
        return baseline + noise + slow_wave
    
    def _apply_deceleration(self, fhr: float, t: float, decel_start: float, 
                           decel_type: str, uc_value: float) -> float:
        """Apply deceleration pattern to FHR."""
        if decel_type == 'early_decel':
            # Early decel: mirrors contraction, nadir with peak
            if uc_value > 30:
                depth = min(30, (uc_value - 30) * 0.5)
                return fhr - depth
        elif decel_type == 'late_decel':
            # Late decel: delayed by 20-30 seconds
            delay = 25
            if t > decel_start + delay and t < decel_start + delay + 90:
                progress = (t - decel_start - delay) / 90
                depth = 40 * np.sin(np.pi * progress)
                return fhr - depth
        elif decel_type == 'variable_decel':
            # Variable: abrupt onset/offset
            if t > decel_start and t < decel_start + 60:
                return fhr - 50 + np.random.normal(0, 5)
        elif decel_type == 'prolonged_decel':
            # Prolonged: >2 min duration
            if t > decel_start and t < decel_start + 180:
                return fhr - 45
        return fhr
        
    def generate_single_trace(self, pattern: str, baseline: float = 140.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single FHR/UC trace with specified pattern."""
        fhr_trace = []
        uc_trace = []
        
        variability = 10.0 if pattern not in ['reduced_variability'] else 2.0
        
        # Adjust baseline for certain conditions
        if pattern == 'bradycardia':
            baseline = 95  # Below 110 for >10 min
        elif pattern == 'tachycardia':
            baseline = 175  # Above 160
        
        contraction_period = 180  # 3 min between contractions
        contraction_times = [i * contraction_period + 60 for i in range(self.duration_seconds // contraction_period)]
        
        for t in range(self.duration_seconds):
            # Generate UC
            uc_val = self._generate_contraction(t, contraction_period)
            
            # Generate base FHR
            fhr_val = self._generate_fhr_normal(t, baseline, variability)
            
            # Apply patterns
            if pattern == 'acceleration':
                # Random accelerations
                if any(abs(t - ct - 30) < 15 for ct in contraction_times):
                    fhr_val += 20
            elif pattern in ['early_decel', 'late_decel', 'variable_decel']:
                for ct in contraction_times:
                    if pattern == 'early_decel' and abs(t - ct - 30) < 45:
                        fhr_val = self._apply_deceleration(fhr_val, t, ct, pattern, uc_val)
                    elif pattern == 'late_decel' and t > ct:
                        fhr_val = self._apply_deceleration(fhr_val, t, ct, pattern, uc_val)
                    elif pattern == 'variable_decel' and abs(t - ct - 20) < 30:
                        fhr_val = self._apply_deceleration(fhr_val, t, ct, pattern, uc_val)
            elif pattern == 'late_decel_recurrent':
                # Apply late decel to most contractions
                for ct in contraction_times:
                    if np.random.random() > 0.2:  # 80% of contractions
                        fhr_val = self._apply_deceleration(fhr_val, t, ct, 'late_decel', uc_val)
            elif pattern == 'prolonged_decel':
                # One prolonged decel in the middle
                decel_start = self.duration_seconds // 2 - 90
                fhr_val = self._apply_deceleration(fhr_val, t, decel_start, pattern, uc_val)
            
            # Clamp FHR to physiological range
            fhr_val = np.clip(fhr_val, 60, 200)
            uc_val = np.clip(uc_val, 0, 100)
            
            # Generate samples at sample_rate Hz
            for _ in range(self.sample_rate):
                fhr_trace.append(fhr_val + np.random.normal(0, 1))
                uc_trace.append(uc_val + np.random.normal(0, 0.5))
        
        return np.array(fhr_trace), np.array(uc_trace)
    
    def generate_dataset(self, samples_per_class: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate a complete dataset with specified class distribution.
        """
        X_list = []
        y_list = []
        descriptions = []
        
        for scenario_name, n_samples in samples_per_class.items():
            config = self.EVENT_CONFIGS.get(scenario_name)
            if not config:
                logger.warning(f"Unknown scenario: {scenario_name}")
                continue
                
            logger.info(f"Generating {n_samples} samples for '{scenario_name}' (Label: {config['label']})")
            
            for i in range(n_samples):
                # Vary baseline slightly for diversity
                baseline = np.random.uniform(130, 150)
                fhr, uc = self.generate_single_trace(config['fhr_pattern'], baseline)
                
                # Stack FHR and UC as 2 channels
                sample = np.stack([fhr, uc], axis=0)  # Shape: (2, n_timepoints)
                X_list.append(sample)
                y_list.append(config['label'])
                descriptions.append(config['description'])
                
        X = np.array(X_list)  # Shape: (n_samples, 2, n_timepoints)
        y = np.array(y_list)
        
        return X, y, descriptions


class FeatureExtractor:
    """Extract features from CTG traces using MiniRocket or manual methods."""
    
    def __init__(self, use_minirocket: bool = True):
        self.use_minirocket = use_minirocket and MINIROCKET_AVAILABLE
        self.minirocket = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _extract_manual_features(self, X: np.ndarray) -> np.ndarray:
        """Extract manual statistical features when MiniRocket unavailable."""
        features_list = []
        
        for sample in X:
            fhr = sample[0]  # FHR channel
            uc = sample[1]   # UC channel
            
            features = [
                # FHR statistics
                np.mean(fhr),
                np.std(fhr),
                np.min(fhr),
                np.max(fhr),
                np.percentile(fhr, 25),
                np.percentile(fhr, 75),
                np.median(fhr),
                
                # Variability measures
                np.mean(np.abs(np.diff(fhr))),  # Short-term variability
                np.std(np.diff(fhr)),
                
                # Deceleration indicators
                np.sum(fhr < 110) / len(fhr),  # Fraction below 110
                np.sum(fhr < 100) / len(fhr),  # Fraction below 100
                np.sum(fhr > 160) / len(fhr),  # Fraction above 160
                
                # UC statistics
                np.mean(uc),
                np.std(uc),
                np.max(uc),
                
                # Cross-correlation (FHR response to UC)
                np.corrcoef(fhr[:-10], uc[10:])[0, 1] if len(fhr) > 10 else 0,
            ]
            
            features_list.append(features)
            
        return np.array(features_list)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the feature extractor and transform data."""
        if self.use_minirocket:
            logger.info("Using MiniRocket for feature extraction...")
            # MiniRocket expects shape (n_samples, n_timepoints, n_channels)
            X_reshaped = X.transpose(0, 2, 1)  # (n_samples, 2, time) -> (n_samples, time, 2)
            self.minirocket = MiniRocketMultivariate()
            features = self.minirocket.fit_transform(X_reshaped)
            features = np.array(features)
        else:
            logger.info("Using manual feature extraction...")
            features = self._extract_manual_features(X)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        self.is_fitted = True
        
        return features_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted extractor."""
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before transform")
            
        if self.use_minirocket:
            X_reshaped = X.transpose(0, 2, 1)
            features = self.minirocket.transform(X_reshaped)
            features = np.array(features)
        else:
            features = self._extract_manual_features(X)
            
        return self.scaler.transform(features)


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced data."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {}
    
    for cls, count in zip(classes, counts):
        # Weight = total / (n_classes * count)
        weights[cls] = total / (len(classes) * count)
        
    logger.info(f"Class distribution: {dict(zip(classes, counts))}")
    logger.info(f"Class weights: {weights}")
    
    return weights


def train_xgboost_model(
    X: np.ndarray, 
    y: np.ndarray, 
    n_folds: int = 10
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train XGBoost with Stratified K-Fold Cross-Validation.
    
    Args:
        X: Feature matrix
        y: Labels
        n_folds: Number of CV folds
        
    Returns:
        Trained model and metrics dictionary
    """
    class_weights = calculate_class_weights(y)
    
    # Calculate scale_pos_weight for binary or sample_weight for multiclass
    n_classes = len(np.unique(y))
    
    if XGBOOST_AVAILABLE:
        logger.info("Training XGBoost classifier...")
        
        if n_classes == 2:
            # Binary classification
            scale_pos_weight = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Multiclass classification
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=n_classes,
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
    else:
        # Fallback to sklearn
        from sklearn.ensemble import GradientBoostingClassifier
        logger.info("Using sklearn GradientBoostingClassifier as fallback...")
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    
    # Stratified K-Fold Cross-Validation
    logger.info(f"Performing {n_folds}-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Get cross-validated predictions
    if XGBOOST_AVAILABLE and n_classes > 2:
        # For multiclass XGBoost, we need sample weights
        sample_weights = np.array([class_weights[label] for label in y])
        y_pred_cv = cross_val_predict(model, X, y, cv=skf, method='predict')
        y_proba_cv = cross_val_predict(model, X, y, cv=skf, method='predict_proba')
    else:
        y_pred_cv = cross_val_predict(model, X, y, cv=skf, method='predict')
        y_proba_cv = cross_val_predict(model, X, y, cv=skf, method='predict_proba')
    
    # Calculate metrics
    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("="*60)
    
    # Classification report
    class_names = ['Cat I (Normal)', 'Cat II (Indeterminate)', 'Cat III (Pathological)'][:n_classes]
    report = classification_report(y, y_pred_cv, target_names=class_names, digits=3)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred_cv)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Per-class metrics
    metrics = {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'accuracy': np.mean(y_pred_cv == y),
        'n_samples': len(y),
        'n_features': X.shape[1],
        'n_folds': n_folds,
        'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
    }
    
    # Calculate ROC-AUC if possible
    try:
        if n_classes == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_proba_cv[:, 1])
        else:
            metrics['roc_auc'] = roc_auc_score(y, y_proba_cv, multi_class='ovr')
        logger.info(f"\nROC-AUC Score: {metrics['roc_auc']:.3f}")
    except Exception as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
    
    # Train final model on all data
    logger.info("\nTraining final model on complete dataset...")
    if XGBOOST_AVAILABLE and n_classes > 2:
        model.fit(X, y, sample_weight=sample_weights)
    else:
        model.fit(X, y)
    
    return model, metrics


def save_pipeline(
    model: Any,
    feature_extractor: FeatureExtractor,
    metrics: Dict[str, Any],
    output_path: Path
):
    """Save the complete pipeline for production use."""
    pipeline_data = {
        'model': model,
        'feature_extractor': feature_extractor,
        'metrics': metrics,
        'created_at': datetime.now().isoformat(),
        'version': '2.0.0-xgboost'
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline_data, f)
    
    logger.info(f"✅ Pipeline saved to: {output_path}")


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("XGBoost CTG Classification Training Pipeline")
    logger.info("="*60)
    
    # Configuration
    OUTPUT_DIR = PROJECT_ROOT / 'models'
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Generate synthetic training data with class imbalance
    # Mimics real-world distribution where pathological cases are rare
    samples_config = {
        'normal': 500,              # Most common
        'acceleration': 200,        # Common, reassuring
        'early_decel': 150,         # Relatively common, benign
        'variable_decel': 100,      # Less common
        'tachycardia': 80,          # Less common
        'reduced_variability': 80,  # Less common
        'late_decel_single': 60,    # Rare
        'late_decel_recurrent': 40, # Very rare, dangerous
        'prolonged_decel': 30,      # Very rare, dangerous
        'bradycardia': 30,          # Very rare, dangerous
    }
    
    logger.info("\n📊 Phase 1: Generating Synthetic Training Data...")
    generator = SyntheticDataGenerator(duration_seconds=600, sample_rate=4)
    X_raw, y, descriptions = generator.generate_dataset(samples_config)
    
    logger.info(f"Generated dataset shape: {X_raw.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    # Extract features
    logger.info("\n🔬 Phase 2: Extracting Features...")
    feature_extractor = FeatureExtractor(use_minirocket=MINIROCKET_AVAILABLE)
    X_features = feature_extractor.fit_transform(X_raw)
    
    logger.info(f"Feature matrix shape: {X_features.shape}")
    
    # Train model
    logger.info("\n🎯 Phase 3: Training XGBoost with Cross-Validation...")
    model, metrics = train_xgboost_model(X_features, y, n_folds=10)
    
    # Save pipeline
    logger.info("\n💾 Phase 4: Saving Pipeline...")
    pipeline_path = OUTPUT_DIR / 'ctg_xgboost_pipeline.pkl'
    save_pipeline(model, feature_extractor, metrics, pipeline_path)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Model: {'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting (fallback)'}")
    logger.info(f"Features: {'MiniRocket' if MINIROCKET_AVAILABLE else 'Manual extraction'}")
    logger.info(f"Samples: {len(y)}")
    logger.info(f"Features: {X_features.shape[1]}")
    logger.info(f"Accuracy: {metrics['accuracy']:.1%}")
    logger.info(f"Pipeline saved to: {pipeline_path}")
    
    return model, metrics


if __name__ == '__main__':
    main()
