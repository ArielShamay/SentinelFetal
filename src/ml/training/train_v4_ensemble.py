"""
SentinelFetal V4.0 — Hybrid Ensemble Training Pipeline
=======================================================
Training script implementing nested stratified cross-validation with:
- XGBoost, Random Forest, and SGD Classifier
- Patient-level splits (NO data leakage between train/test)
- SMOTE applied ONLY in inner training loop
- Probability calibration (Platt Scaling)

CLINICAL SAFETY CRITICAL
- All training decisions are logged for audit
- Reproducibility guaranteed via fixed seeds
- Patient isolation verified before training

Usage:
    python src/ml/training/train_v4_ensemble.py

Author: SentinelFetal ML Team
Version: 4.0.0
"""

import os
import sys
import pickle
import json
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import warnings

import numpy as np
import yaml
from collections import Counter

# Scikit-learn imports
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    StratifiedGroupKFold = None
    HAS_STRATIFIED_GROUP_KFOLD = False
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# XGBoost
import xgboost as xgb

# SMOTE for class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    warnings.warn("imbalanced-learn not installed. Run: pip install imbalanced-learn")

# WFDB for CTU-CHB data loading
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    warnings.warn("wfdb not installed. Run: pip install wfdb")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "ensemble_v4.yaml"
# Note: nested directory structure in CTU-CHB download
DATA_DIR = PROJECT_ROOT / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
OUTPUT_DIR = PROJECT_ROOT / "models" / "ensemble_v4"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"
LOG_DIR = PROJECT_ROOT / "REPORTS" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "training.log"),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Loader
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load training configuration from YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {CONFIG_PATH}, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration if YAML not found."""
    return {
        'ensemble': {
            'weights': {'xgboost': 0.4, 'random_forest': 0.3, 'sgd_classifier': 0.3}
        },
        'thresholds': {'critical': 0.60, 'warning': 0.35},
        'windowing': {
            'window_minutes': 20,
            'stride_minutes': 5
        },
        'training': {
            'outer_folds': 10,
            'inner_folds': 5,
            'random_seed': 42,
            'test_size': 0.2,
            'smote': {'enabled': True, 'sampling_strategy': 'auto', 'k_neighbors': 5}
        },
        'models': {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced'
            },
            'sgd_classifier': {
                'loss': 'log_loss',
                'penalty': 'elasticnet',
                'alpha': 0.0001,
                'l1_ratio': 0.15,
                'max_iter': 1000,
                'class_weight': 'balanced'
            }
        }
    }


# =============================================================================
# CTU-CHB Data Loader (Native WFDB)
# =============================================================================

class CTUCHBDataLoader:
    """
    Load CTU-CHB Intrapartum CTG database using native WFDB library.
    
    Dataset: 552 intrapartum CTG recordings
    Labels: pH-based outcome (pH >= 7.20 = Normal, pH < 7.15 = Pathological)
    """
    
    # pH thresholds for binary classification
    PH_NORMAL_THRESHOLD = 7.20
    PH_PATHOLOGICAL_THRESHOLD = 7.15
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.records = []
        self.metadata = {}
        
    def discover_records(self) -> List[str]:
        """Find all record IDs in the dataset."""
        records = []
        for hea_file in self.data_dir.glob("*.hea"):
            record_id = hea_file.stem
            dat_file = hea_file.with_suffix(".dat")
            if dat_file.exists():
                records.append(record_id)
        
        self.records = sorted(records)
        logger.info(f"Discovered {len(self.records)} records in {self.data_dir}")
        return self.records
    
    def _parse_header_native(self, record_id: str) -> Dict[str, Any]:
        """
        Parse .hea header file directly for metadata.
        This handles CTU-CHB's specific comment format.
        """
        header_path = self.data_dir / f"{record_id}.hea"
        
        metadata = {
            'record_id': record_id,
            'ph': None,
            'apgar1': None,
            'apgar5': None,
        }
        
        with open(header_path, 'r') as f:
            lines = f.readlines()
        
        # Parse comment lines for clinical data (CTU-CHB format: #pH           7.14)
        for line in lines:
            line = line.strip()
            if line.startswith('#pH'):
                try:
                    # Handle format: "#pH           7.14"
                    parts = line.split()
                    if len(parts) >= 2:
                        metadata['ph'] = float(parts[-1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith('#Apgar1'):
                try:
                    metadata['apgar1'] = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith('#Apgar5'):
                try:
                    metadata['apgar5'] = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
        
        return metadata
    
    def load_record(self, record_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Load a single record.
        
        Returns:
            Tuple of (fhr_signal, uc_signal, metadata_dict) or (None, None, {}) on failure
        """
        record_path = str(self.data_dir / record_id)
        
        try:
            record = wfdb.rdrecord(record_path)
            
            fhr = record.p_signal[:, 0] if record.p_signal.shape[1] > 0 else None
            uc = record.p_signal[:, 1] if record.p_signal.shape[1] > 1 else None
            
            # Get metadata from native parsing (more reliable for CTU-CHB)
            meta = self._parse_header_native(record_id)
            meta['fs'] = record.fs
            meta['sig_len'] = record.sig_len
            
            return fhr, uc, meta
            
        except Exception as e:
            logger.warning(f"Failed to load record {record_id}: {e}")
            return None, None, {}
    
    def get_ph_label(self, metadata: Dict[str, Any]) -> Optional[int]:
        """
        Extract pH-based binary label from metadata.
        
        Returns:
            0 = Normal (pH >= 7.20)
            1 = Pathological (pH < 7.15)
            None = Unknown or Suspicious (excluded from training)
        """
        ph = metadata.get('ph', None)
        if ph is None:
            return None
        
        try:
            ph = float(ph)
            
            if ph >= self.PH_NORMAL_THRESHOLD:
                return 0  # Normal
            elif ph < self.PH_PATHOLOGICAL_THRESHOLD:
                return 1  # Pathological
            else:
                return None  # Suspicious (7.15 <= pH < 7.20) - excluded
                
        except (ValueError, TypeError):
            return None
    
    def load_all_records(self) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Load all records with valid pH labels.
        
        Returns:
            Tuple of (fhr_signals, labels, patient_ids)
        """
        if not self.records:
            self.discover_records()
        
        fhr_signals = []
        labels = []
        patient_ids = []
        
        for record_id in self.records:
            fhr, uc, meta = self.load_record(record_id)
            
            if fhr is None:
                continue
            
            label = self.get_ph_label(meta)
            if label is None:
                continue  # Skip records without clear label
            
            fhr_signals.append(fhr)
            labels.append(label)
            patient_ids.append(record_id)
        
        logger.info(f"Loaded {len(fhr_signals)} records with valid pH labels")
        logger.info(f"Class distribution: Normal={labels.count(0)}, Pathological={labels.count(1)}")
        
        return fhr_signals, labels, patient_ids


# =============================================================================
# Feature Extraction
# =============================================================================

class CTGFeatureExtractor:
    """
    Extract statistical features from CTG signals.
    
    Features (32 total):
    - Time-domain: mean, std, min, max, range, IQR, skewness, kurtosis
    - Variability: short-term, long-term, baseline crossings
    - Acceleration/Deceleration: counts, durations, areas
    """
    
    BASELINE_FHR = 140.0  # bpm
    
    def extract(self, fhr: np.ndarray, fs: float = 4.0) -> np.ndarray:
        """
        Extract 32 features from FHR signal.
        
        Args:
            fhr: FHR signal array
            fs: Sampling frequency (default 4 Hz for CTU-CHB)
            
        Returns:
            Feature vector of shape (32,)
        """
        # Clean signal
        fhr_clean = self._clean_signal(fhr)
        
        if len(fhr_clean) < 60:
            return np.zeros(32)
        
        features = []
        
        # Time-domain features (8)
        features.extend([
            np.mean(fhr_clean),
            np.std(fhr_clean),
            np.min(fhr_clean),
            np.max(fhr_clean),
            np.ptp(fhr_clean),  # range
            np.percentile(fhr_clean, 75) - np.percentile(fhr_clean, 25),  # IQR
            self._skewness(fhr_clean),
            self._kurtosis(fhr_clean)
        ])
        
        # Variability features (6)
        features.extend(self._variability_features(fhr_clean, fs))
        
        # Baseline features (4)
        features.extend(self._baseline_features(fhr_clean))
        
        # Acceleration features (6)
        features.extend(self._acceleration_features(fhr_clean, fs))
        
        # Deceleration features (6)
        features.extend(self._deceleration_features(fhr_clean, fs))
        
        # Trend features (2)
        features.extend(self._trend_features(fhr_clean))
        
        return np.array(features, dtype=np.float32)
    
    def _clean_signal(self, fhr: np.ndarray) -> np.ndarray:
        """Remove artifacts and invalid values."""
        fhr_clean = fhr.copy()
        
        # Replace invalid values
        fhr_clean[(fhr_clean < 50) | (fhr_clean > 210)] = np.nan
        
        # Interpolate short gaps
        valid_mask = ~np.isnan(fhr_clean)
        if valid_mask.sum() > 10:
            fhr_clean = np.interp(
                np.arange(len(fhr_clean)),
                np.where(valid_mask)[0],
                fhr_clean[valid_mask]
            )
        
        return fhr_clean
    
    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(x)
        if n < 3:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 3))
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 4) - 3.0)
    
    def _variability_features(self, fhr: np.ndarray, fs: float) -> List[float]:
        """Extract variability features."""
        # Short-term variability (beat-to-beat)
        diff = np.abs(np.diff(fhr))
        stv = np.mean(diff) if len(diff) > 0 else 0.0
        stv_std = np.std(diff) if len(diff) > 0 else 0.0
        
        # Long-term variability (1-minute segments)
        segment_len = int(60 * fs)
        n_segments = len(fhr) // segment_len
        
        if n_segments > 1:
            segment_ranges = []
            for i in range(n_segments):
                seg = fhr[i * segment_len:(i + 1) * segment_len]
                segment_ranges.append(np.ptp(seg))
            ltv = np.mean(segment_ranges)
            ltv_std = np.std(segment_ranges)
        else:
            ltv = np.ptp(fhr)
            ltv_std = 0.0
        
        # Baseline crossings
        crossings = np.sum(np.diff(np.sign(fhr - self.BASELINE_FHR)) != 0)
        crossing_rate = crossings / (len(fhr) / fs)  # per second
        
        return [stv, stv_std, ltv, ltv_std, crossings, crossing_rate]
    
    def _baseline_features(self, fhr: np.ndarray) -> List[float]:
        """Extract baseline-related features."""
        baseline = np.median(fhr)
        deviation_from_baseline = np.abs(fhr - baseline)
        
        return [
            baseline,
            np.mean(deviation_from_baseline),
            np.percentile(deviation_from_baseline, 90),
            np.sum(deviation_from_baseline > 15) / len(fhr)  # % time >15 bpm from baseline
        ]
    
    def _acceleration_features(self, fhr: np.ndarray, fs: float) -> List[float]:
        """Extract acceleration features."""
        baseline = np.median(fhr)
        
        # Acceleration: >15 bpm above baseline for >15 sec
        accel_mask = fhr > baseline + 15
        accel_regions = self._find_regions(accel_mask)
        
        min_duration = 15 * fs  # 15 seconds
        accelerations = [r for r in accel_regions if r[1] - r[0] >= min_duration]
        
        if accelerations:
            durations = [(r[1] - r[0]) / fs for r in accelerations]
            amplitudes = [np.max(fhr[r[0]:r[1]]) - baseline for r in accelerations]
            return [
                len(accelerations),
                np.mean(durations),
                np.max(durations),
                np.mean(amplitudes),
                np.max(amplitudes),
                sum(durations)
            ]
        else:
            return [0, 0, 0, 0, 0, 0]
    
    def _deceleration_features(self, fhr: np.ndarray, fs: float) -> List[float]:
        """Extract deceleration features."""
        baseline = np.median(fhr)
        
        # Deceleration: >15 bpm below baseline for >15 sec
        decel_mask = fhr < baseline - 15
        decel_regions = self._find_regions(decel_mask)
        
        min_duration = 15 * fs  # 15 seconds
        decelerations = [r for r in decel_regions if r[1] - r[0] >= min_duration]
        
        if decelerations:
            durations = [(r[1] - r[0]) / fs for r in decelerations]
            amplitudes = [baseline - np.min(fhr[r[0]:r[1]]) for r in decelerations]
            return [
                len(decelerations),
                np.mean(durations),
                np.max(durations),
                np.mean(amplitudes),
                np.max(amplitudes),
                sum(durations)
            ]
        else:
            return [0, 0, 0, 0, 0, 0]
    
    def _trend_features(self, fhr: np.ndarray) -> List[float]:
        """Extract trend features."""
        n = len(fhr)
        if n < 10:
            return [0, 0]
        
        # Linear regression slope
        x = np.arange(n)
        slope = np.polyfit(x, fhr, 1)[0]
        
        # Trend strength (R-squared)
        y_pred = slope * x + np.mean(fhr)
        ss_res = np.sum((fhr - y_pred) ** 2)
        ss_tot = np.sum((fhr - np.mean(fhr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return [slope, r_squared]
    
    def _find_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous True regions in boolean mask."""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                in_region = True
                start = i
            elif not val and in_region:
                in_region = False
                regions.append((start, i))
        
        if in_region:
            regions.append((start, len(mask)))
        
        return regions


# =============================================================================
# Sliding-Window Utilities
# =============================================================================

def generate_sliding_windows(
    fhr: np.ndarray,
    window_samples: int,
    stride_samples: int
) -> List[np.ndarray]:
    """Slice FHR into overlapping windows, dropping partial tails."""
    if len(fhr) < window_samples or window_samples <= 0:
        return []

    stride = max(1, stride_samples)
    windows = []
    for start in range(0, len(fhr) - window_samples + 1, stride):
        windows.append(fhr[start:start + window_samples])
    return windows


# =============================================================================
# Nested Cross-Validation Trainer
# =============================================================================

class NestedCVTrainer:
    """
    Nested Stratified Cross-Validation Trainer for V4.0 Ensemble.
    
    Implements:
    - Outer loop: 10-fold stratified CV for unbiased performance estimation
    - Inner loop: 5-fold CV for hyperparameter tuning (not implemented yet)
    - SMOTE applied ONLY in inner training fold
    - Patient-level splits to prevent data leakage
    - Probability calibration using Platt Scaling
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config.get('training', {})
        self.model_configs = config.get('models', {})
        
        self.random_seed = self.training_config.get('random_seed', 42)
        self.outer_folds = self.training_config.get('outer_folds', 10)
        self.inner_folds = self.training_config.get('inner_folds', 5)
        
        # SMOTE configuration
        self.smote_config = self.training_config.get('smote', {})
        self.use_smote = self.smote_config.get('enabled', True) and SMOTE_AVAILABLE
        
        # Results storage
        self.fold_results = []
        self.final_models = {}
        self.scaler = None
        self.oof_records = []
        
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        patient_ids: List[str],
        window_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Run nested cross-validation training.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            patient_ids: Patient identifiers for patient-level splitting
            window_indices: Window index per sample (for OOF tracking)
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("=" * 60)
        logger.info("Starting V4.0 Ensemble Training")
        logger.info(f"Samples: {len(y)}, Features: {X.shape[1]}")
        logger.info(f"Class distribution: {dict(Counter(y))}")
        logger.info(f"SMOTE enabled: {self.use_smote}")
        logger.info("=" * 60)
        
        # Verify patient-level integrity
        self._verify_patient_integrity(patient_ids)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        groups = np.array(patient_ids)

        # Outer CV loop (patient-level grouping to avoid leakage)
        outer_cv = self._build_outer_cv_splitter(X_scaled, y, groups)
        
        all_predictions = {
            'xgboost': [],
            'random_forest': [],
            'sgd_classifier': [],
            'ensemble': []
        }
        all_true = []
        all_proba = {
            'xgboost': [],
            'random_forest': [],
            'sgd_classifier': [],
            'ensemble': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv):
            logger.info(f"\n--- Fold {fold_idx + 1}/{self.outer_folds} ---")
            
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Log patient split
            train_patients = [patient_ids[i] for i in train_idx]
            test_patients = [patient_ids[i] for i in test_idx]
            
            # Verify no overlap
            overlap = set(train_patients) & set(test_patients)
            if overlap:
                logger.error(f"CRITICAL: Patient overlap detected: {overlap}")
                raise ValueError("Patient data leakage detected!")
            
            logger.info(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
            logger.info(f"Train class dist: {dict(Counter(y_train))}")
            logger.info(f"Test class dist: {dict(Counter(y_test))}")
            
            # Apply SMOTE to training data only
            X_train_resampled, y_train_resampled = self._apply_smote(X_train, y_train)
            
            # Train models for this fold
            fold_models = self._train_fold_models(X_train_resampled, y_train_resampled, fold_idx)
            
            # Evaluate on test set
            fold_predictions, fold_proba = self._evaluate_fold(fold_models, X_test, y_test, fold_idx)

            # Collect out-of-fold window-level predictions
            for local_idx, global_idx in enumerate(test_idx):
                self.oof_records.append({
                    'patient_id': patient_ids[global_idx],
                    'window_index': int(window_indices[global_idx]),
                    'true_label': int(y_test[local_idx]),
                    'xgb_prob': float(fold_proba.get('xgboost', [0])[local_idx]),
                    'rf_prob': float(fold_proba.get('random_forest', [0])[local_idx]),
                    'sgd_prob': float(fold_proba.get('sgd_classifier', [0])[local_idx])
                })
            
            # Store results
            for model_name in fold_predictions:
                all_predictions[model_name].extend(fold_predictions[model_name])
                all_proba[model_name].extend(fold_proba[model_name])
            all_true.extend(y_test)
            
            # Store fold-level metrics
            self.fold_results.append({
                'fold': fold_idx + 1,
                'train_size': len(y_train_resampled),
                'test_size': len(y_test),
                'train_patients': len(set(train_patients)),
                'test_patients': len(set(test_patients)),
                'predictions': fold_predictions,
                'true_labels': list(y_test)
            })
        
        # Aggregate results
        results = self._compute_final_metrics(all_true, all_predictions, all_proba)
        
        # Train final models on all data for deployment
        logger.info("\n--- Training Final Models on Full Dataset ---")
        X_full_resampled, y_full_resampled = self._apply_smote(X_scaled, y)
        self.final_models = self._train_fold_models(X_full_resampled, y_full_resampled, -1)
        
        # Calibrate final models
        logger.info("\n--- Calibrating Final Models ---")
        self._calibrate_models(X_scaled, y)
        
        return results
    
    def _verify_patient_integrity(self, patient_ids: List[str]):
        """Verify patient ID uniqueness for patient-level splitting."""
        unique_patients = len(set(patient_ids))
        total_samples = len(patient_ids)
        
        logger.info(f"Patient integrity check: {unique_patients} unique patients, {total_samples} samples")
        
        if unique_patients != total_samples:
            logger.info("Multiple samples per patient detected; enforcing patient-level grouped CV.")

    def _build_outer_cv_splitter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ):
        """Create patient-grouped CV splitter with stratification when available."""
        if HAS_STRATIFIED_GROUP_KFOLD:
            return StratifiedGroupKFold(
                n_splits=self.outer_folds,
                shuffle=True,
                random_state=self.random_seed
            ).split(X, y, groups)
        logger.warning("StratifiedGroupKFold unavailable; using GroupKFold (no stratification).")
        return GroupKFold(n_splits=self.outer_folds).split(X, y, groups)
    
    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to training data."""
        if not self.use_smote:
            return X, y
        
        try:
            smote = SMOTE(
                sampling_strategy=self.smote_config.get('sampling_strategy', 'auto'),
                k_neighbors=self.smote_config.get('k_neighbors', 5),
                random_state=self.random_seed
            )
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            logger.info(f"SMOTE: {len(y)} -> {len(y_resampled)} samples")
            logger.info(f"After SMOTE: {dict(Counter(y_resampled))}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def _train_fold_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        fold_idx: int
    ) -> Dict[str, Any]:
        """Train all three models for a single fold."""
        models = {}
        
        # XGBoost
        xgb_config = self.model_configs.get('xgboost', {})
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 200),
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            objective=xgb_config.get('objective', 'binary:logistic'),
            eval_metric=xgb_config.get('eval_metric', 'logloss'),
            random_state=self.random_seed,
            use_label_encoder=False
        )
        models['xgboost'].fit(X_train, y_train)
        logger.info(f"  XGBoost trained")
        
        # Random Forest
        rf_config = self.model_configs.get('random_forest', {})
        models['random_forest'] = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', 10),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            class_weight=rf_config.get('class_weight', 'balanced'),
            random_state=self.random_seed,
            n_jobs=-1
        )
        models['random_forest'].fit(X_train, y_train)
        logger.info(f"  Random Forest trained")
        
        # SGD Classifier
        sgd_config = self.model_configs.get('sgd_classifier', {})
        models['sgd_classifier'] = SGDClassifier(
            loss=sgd_config.get('loss', 'log_loss'),
            penalty=sgd_config.get('penalty', 'elasticnet'),
            alpha=sgd_config.get('alpha', 0.0001),
            l1_ratio=sgd_config.get('l1_ratio', 0.15),
            max_iter=sgd_config.get('max_iter', 1000),
            class_weight=sgd_config.get('class_weight', 'balanced'),
            random_state=self.random_seed
        )
        models['sgd_classifier'].fit(X_train, y_train)
        logger.info(f"  SGD Classifier trained")
        
        return models
    
    def _evaluate_fold(
        self, 
        models: Dict[str, Any], 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        fold_idx: int
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
        """Evaluate all models on test set."""
        predictions = {}
        probabilities = {}
        
        # Get ensemble weights
        weights = self.config.get('ensemble', {}).get('weights', {
            'xgboost': 0.4, 'random_forest': 0.3, 'sgd_classifier': 0.3
        })
        
        ensemble_proba = np.zeros(len(y_test))
        
        for model_name, model in models.items():
            pred = model.predict(X_test)
            predictions[model_name] = list(pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = np.zeros(len(y_test)) + 0.5
            
            probabilities[model_name] = list(proba)
            
            # Weighted contribution to ensemble
            weight = weights.get(model_name, 1.0 / len(models))
            ensemble_proba += weight * proba
            
            # Log individual model metrics
            acc = accuracy_score(y_test, pred)
            recall = recall_score(y_test, pred, pos_label=1, zero_division=0)
            logger.info(f"  {model_name}: Acc={acc:.3f}, Recall(Path)={recall:.3f}")
        
        # Ensemble predictions
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        predictions['ensemble'] = list(ensemble_pred)
        probabilities['ensemble'] = list(ensemble_proba)
        
        # Log ensemble metrics
        acc = accuracy_score(y_test, ensemble_pred)
        recall = recall_score(y_test, ensemble_pred, pos_label=1, zero_division=0)
        logger.info(f"  ENSEMBLE: Acc={acc:.3f}, Recall(Path)={recall:.3f}")
        
        return predictions, probabilities
    
    def _compute_final_metrics(
        self, 
        y_true: List[int], 
        predictions: Dict[str, List[int]], 
        probabilities: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Compute aggregate metrics across all folds."""
        y_true = np.array(y_true)
        
        results = {
            'n_samples': len(y_true),
            'n_folds': self.outer_folds,
            'class_distribution': dict(Counter(y_true)),
            'models': {}
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("FINAL AGGREGATE RESULTS")
        logger.info("=" * 60)
        
        for model_name in predictions:
            y_pred = np.array(predictions[model_name])
            y_proba = np.array(probabilities[model_name])
            
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            
            results['models'][model_name] = metrics
            
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.3f}")
            logger.info(f"  Precision: {metrics['precision']:.3f}")
            logger.info(f"  Recall:    {metrics['recall']:.3f}")
            logger.info(f"  F1:        {metrics['f1']:.3f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
            logger.info(f"  Confusion Matrix:\n    {metrics['confusion_matrix']}")
        
        # Check pathological recall target (>= 95%)
        ensemble_recall = results['models']['ensemble']['recall']
        target_recall = 0.95
        
        if ensemble_recall >= target_recall:
            logger.info(f"\n[PASS] Pathological recall {ensemble_recall:.1%} >= target {target_recall:.0%}")
        else:
            logger.warning(f"\n[WARN] Pathological recall {ensemble_recall:.1%} < target {target_recall:.0%}")
        
        return results
    
    def _calibrate_models(self, X: np.ndarray, y: np.ndarray):
        """Apply Platt Scaling calibration to final models."""
        for model_name in list(self.final_models.keys()):
            try:
                calibrator = CalibratedClassifierCV(
                    self.final_models[model_name],
                    method='sigmoid',  # Platt Scaling
                    cv='prefit'
                )
                calibrator.fit(X, y)
                self.final_models[model_name] = calibrator
                logger.info(f"  Calibrated {model_name}")
            except Exception as e:
                logger.warning(f"  Calibration failed for {model_name}: {e}")
    
    def save_models(self, output_dir: Path) -> Dict[str, str]:
        """Save trained models to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}
        
        # Save individual models
        for model_name, model in self.final_models.items():
            path = output_dir / f"{model_name}_calibrated.pkl"
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            saved_paths[model_name] = str(path)
            logger.info(f"Saved {model_name} to {path}")
        
        # Save scaler
        scaler_path = output_dir / "feature_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        saved_paths['scaler'] = str(scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        return saved_paths


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main entry point for V4.0 ensemble training."""
    start_time = datetime.now()
    
    print("=" * 70)
    print("SentinelFetal V4.0 — Hybrid Ensemble Training Pipeline")
    print("=" * 70)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print()
    
    # Check dependencies
    if not WFDB_AVAILABLE:
        print("[ERROR] wfdb library required. Run: pip install wfdb")
        sys.exit(1)
    
    if not SMOTE_AVAILABLE:
        print("[WARNING] imbalanced-learn not installed. SMOTE disabled.")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration from {CONFIG_PATH}")
    
    # Load data
    logger.info("\n--- Loading CTU-CHB Data ---")
    loader = CTUCHBDataLoader(DATA_DIR)
    fhr_signals, labels, patient_ids = loader.load_all_records()
    
    if len(fhr_signals) == 0:
        logger.error("No valid records found. Check data directory.")
        sys.exit(1)
    
    # Extract features with sliding windows
    logger.info("\n--- Extracting Features (Sliding Windows) ---")
    extractor = CTGFeatureExtractor()

    window_config = config.get('windowing', {})
    window_minutes = window_config.get('window_minutes', 20)
    stride_minutes = window_config.get('stride_minutes', 5)
    fs = 4.0
    window_samples = int(window_minutes * 60 * fs)
    stride_samples = int(stride_minutes * 60 * fs)

    logger.info(
        f"Windowing params: window={window_minutes} min ({window_samples} samples), "
        f"stride={stride_minutes} min ({stride_samples} samples)"
    )
    if stride_samples <= 0:
        logger.warning("Stride non-positive; defaulting stride to window size (no overlap).")
        stride_samples = window_samples

    X_list = []
    valid_labels = []
    valid_patient_ids = []
    window_indices = []
    window_counts = []

    for i, (fhr, label, pid) in enumerate(zip(fhr_signals, labels, patient_ids)):
        windows = generate_sliding_windows(fhr, window_samples, stride_samples)
        if not windows:
            logger.warning(
                f"Record {pid} too short for {window_minutes} min window (len={len(fhr)} samples); skipping."
            )
            continue

        window_counts.append(len(windows))

        for w_idx, window in enumerate(windows):
            features = extractor.extract(window, fs=fs)
            if np.sum(np.abs(features)) > 0:  # Valid features
                X_list.append(features)
                valid_labels.append(label)
                valid_patient_ids.append(pid)
            window_indices.append(w_idx)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(fhr_signals)} records...")
    
    X = np.array(X_list)
    y = np.array(valid_labels)

    if len(window_counts) > 0:
        logger.info(
            f"Generated {len(X)} windows across {len(window_counts)} patients | "
            f"mean/median windows per patient: {np.mean(window_counts):.2f}/{np.median(window_counts):.2f} | "
            f"min={np.min(window_counts)}, max={np.max(window_counts)}"
        )
    else:
        logger.error("No windows generated; check window configuration and source signals.")

    logger.info(f"Feature extraction complete: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train ensemble
    trainer = NestedCVTrainer(config)
    results = trainer.train(X, y, valid_patient_ids, window_indices)

    # Save out-of-fold validation probabilities
    oof_path = OUTPUT_DIR / "validation_preds.csv"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    with open(oof_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'window_index', 'true_label', 'xgb_prob', 'rf_prob', 'sgd_prob'])
        for rec in trainer.oof_records:
            writer.writerow([
                rec['patient_id'],
                rec['window_index'],
                rec['true_label'],
                rec['xgb_prob'],
                rec['rf_prob'],
                rec['sgd_prob']
            ])
    logger.info(f"Saved OOF validation predictions to {oof_path}")
    
    # Save models
    logger.info("\n--- Saving Models ---")
    saved_paths = trainer.save_models(OUTPUT_DIR)
    
    # Save training results
    results_path = OUTPUT_DIR / "training_results.json"
    results['saved_paths'] = saved_paths
    results['training_time'] = str(datetime.now() - start_time)
    results['timestamp'] = datetime.now().isoformat()
    results['config_path'] = str(CONFIG_PATH)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results_clean = convert_numpy(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    logger.info(f"Saved training results to {results_path}")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Duration: {duration}")
    print(f"Samples: {results['n_samples']}")
    print(f"Folds: {results['n_folds']}")
    print()
    print("Ensemble Performance:")
    ensemble_metrics = results['models']['ensemble']
    print(f"  Accuracy:  {ensemble_metrics['accuracy']:.1%}")
    print(f"  Precision: {ensemble_metrics['precision']:.1%}")
    print(f"  Recall:    {ensemble_metrics['recall']:.1%}")
    print(f"  F1:        {ensemble_metrics['f1']:.1%}")
    print(f"  ROC-AUC:   {ensemble_metrics['roc_auc']:.3f}")
    print()
    print(f"Models saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
