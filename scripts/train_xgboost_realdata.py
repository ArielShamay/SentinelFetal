#!/usr/bin/env python3
"""
SentinelFetal XGBoost Training Pipeline - REAL CLINICAL DATA VERSION
=====================================================================
UPGRADE FROM SYNTHETIC: Now using CTU-CHB Intrapartum CTG Database (552 recordings)

This script replaces the synthetic data training with real clinical data from:
- CTU-CHB Intrapartum Cardiotocography Database (PhysioNet)
- 552 real intrapartum CTG recordings with pH-based outcome labels
- Gold standard: Umbilical cord arterial pH

Label Mapping (based on ACOG/FIGO guidelines):
- pH >= 7.20: Normal (Category 1)
- pH 7.15-7.19: Suspicious (Category 2)
- pH < 7.15: Pathological (Category 3)

Features:
- Patient-level 10-fold stratified CV (NO data leakage)
- SMOTE applied only to training folds
- Final model trained on ALL 552 patients
- Automatic verification after training
"""

import os
import sys
import re
import struct
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)

# Optional: SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imblearn not installed - will use class weights instead of SMOTE")

import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# =============================================================================
# CTU-CHB WFDB Data Loader (Native Python - no wfdb dependency required)
# =============================================================================

@dataclass
class CTGRecord:
    """Single CTG recording with signals and metadata."""
    record_id: str
    fhr: np.ndarray  # Fetal Heart Rate signal (4 Hz)
    uc: np.ndarray   # Uterine Contractions signal (4 Hz)
    ph: float        # Umbilical cord arterial pH (outcome)
    sampling_rate: int = 4
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def category(self) -> int:
        """Map pH to FIGO category (1=Normal, 2=Suspicious, 3=Pathological)."""
        if self.ph >= 7.20:
            return 1  # Normal
        elif self.ph >= 7.15:
            return 2  # Suspicious
        else:
            return 3  # Pathological
    
    @property
    def duration_minutes(self) -> float:
        """Recording duration in minutes."""
        return len(self.fhr) / (self.sampling_rate * 60)


class CTUCHBDataLoader:
    """
    Native loader for CTU-CHB Intrapartum CTG Database.
    Reads WFDB format without external dependencies.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.records_file = self.data_dir / "RECORDS"
        
    def get_record_list(self) -> List[str]:
        """Get list of all record IDs from RECORDS file."""
        if not self.records_file.exists():
            raise FileNotFoundError(f"RECORDS file not found at {self.records_file}")
        
        with open(self.records_file, 'r') as f:
            records = [line.strip() for line in f if line.strip()]
        return records
    
    def _parse_header(self, record_id: str) -> Dict[str, Any]:
        """Parse .hea header file for metadata and signal info."""
        header_path = self.data_dir / f"{record_id}.hea"
        
        metadata = {
            'record_id': record_id,
            'n_signals': 0,
            'n_samples': 0,
            'sampling_rate': 4,
            'signals': [],
            'ph': None,
            'apgar1': None,
            'apgar5': None,
            'gest_weeks': None,
            'weight': None,
        }
        
        with open(header_path, 'r') as f:
            lines = f.readlines()
        
        # Parse first line: record_name n_signals sampling_rate n_samples
        first_line = lines[0].strip().split()
        metadata['n_signals'] = int(first_line[1])
        metadata['sampling_rate'] = int(first_line[2])
        metadata['n_samples'] = int(first_line[3])
        
        # Parse signal lines (lines 1 and 2 for FHR and UC)
        for i in range(1, min(3, len(lines))):
            if lines[i].startswith('#'):
                break
            parts = lines[i].strip().split()
            if len(parts) >= 9:
                metadata['signals'].append({
                    'name': parts[8],  # FHR or UC
                    'gain': float(parts[2].split('(')[0].split('/')[0]) if '/' in parts[2] else 1.0,
                    'baseline': int(parts[4]),
                    'adc_res': int(parts[3]),
                })
        
        # Parse comment lines for clinical data
        for line in lines:
            line = line.strip()
            if line.startswith('#pH'):
                try:
                    metadata['ph'] = float(line.split()[-1])
                except:
                    pass
            elif line.startswith('#Apgar1'):
                try:
                    metadata['apgar1'] = int(line.split()[-1])
                except:
                    pass
            elif line.startswith('#Apgar5'):
                try:
                    metadata['apgar5'] = int(line.split()[-1])
                except:
                    pass
            elif line.startswith('#Gest. weeks'):
                try:
                    metadata['gest_weeks'] = int(line.split()[-1])
                except:
                    pass
            elif line.startswith('#Weight'):
                try:
                    metadata['weight'] = int(line.split()[-1])
                except:
                    pass
        
        return metadata
    
    def _read_signal_data(self, record_id: str, metadata: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Read binary .dat file and extract FHR and UC signals."""
        dat_path = self.data_dir / f"{record_id}.dat"
        
        n_samples = metadata['n_samples']
        n_signals = metadata['n_signals']
        
        # Read raw binary data (16-bit signed integers, little-endian)
        with open(dat_path, 'rb') as f:
            raw_data = f.read()
        
        # Unpack as int16, interleaved format
        n_values = len(raw_data) // 2
        values = struct.unpack(f'<{n_values}h', raw_data)
        
        # Reshape to (n_samples, n_signals)
        data = np.array(values).reshape(-1, n_signals)
        
        # Apply gain and baseline correction
        fhr_raw = data[:, 0].astype(float)
        uc_raw = data[:, 1].astype(float)
        
        # FHR: gain is typically 100, so divide to get bpm
        # UC: keep as relative units
        fhr = fhr_raw / 100.0  # Convert to bpm
        uc = uc_raw / 100.0    # Normalize
        
        return fhr, uc
    
    def load_record(self, record_id: str) -> Optional[CTGRecord]:
        """Load a single CTG record with signals and metadata."""
        try:
            metadata = self._parse_header(record_id)
            
            # Skip records without pH (can't label them)
            if metadata['ph'] is None:
                logger.warning(f"Record {record_id}: No pH value, skipping")
                return None
            
            fhr, uc = self._read_signal_data(record_id, metadata)
            
            return CTGRecord(
                record_id=record_id,
                fhr=fhr,
                uc=uc,
                ph=metadata['ph'],
                sampling_rate=metadata['sampling_rate'],
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to load record {record_id}: {e}")
            return None
    
    def load_all_records(self, max_records: int = None) -> List[CTGRecord]:
        """Load all valid CTG records from the database."""
        record_ids = self.get_record_list()
        if max_records:
            record_ids = record_ids[:max_records]
        
        records = []
        for record_id in record_ids:
            record = self.load_record(record_id)
            if record is not None:
                records.append(record)
        
        logger.info(f"Loaded {len(records)} valid records from {len(record_ids)} total")
        return records


# =============================================================================
# Feature Extraction for CTG Signals
# =============================================================================

class CTGFeatureExtractor:
    """
    Extract clinically meaningful features from FHR and UC signals.
    Based on FIGO/ACOG CTG interpretation guidelines.
    """
    
    FEATURE_NAMES = [
        # FHR Statistical Features
        'fhr_mean', 'fhr_std', 'fhr_min', 'fhr_max', 'fhr_range',
        'fhr_median', 'fhr_q25', 'fhr_q75', 'fhr_iqr',
        
        # FHR Baseline & Variability
        'fhr_baseline', 'fhr_variability', 'fhr_stv', 'fhr_ltv',
        
        # FHR Pathological Indicators
        'fhr_time_below_110', 'fhr_time_above_160',
        'fhr_bradycardia_episodes', 'fhr_tachycardia_episodes',
        
        # Deceleration Indicators
        'decel_count', 'decel_mean_depth', 'decel_max_depth',
        'decel_total_duration', 'late_decel_score',
        
        # Acceleration Indicators
        'accel_count', 'accel_mean_amplitude',
        
        # UC Features
        'uc_mean', 'uc_std', 'uc_frequency', 'uc_regularity',
        
        # Cross-correlation (FHR-UC coupling)
        'fhr_uc_correlation', 'fhr_uc_lag_correlation',
        
        # Signal Quality
        'fhr_valid_ratio', 'signal_gaps',
    ]
    
    def __init__(self, sampling_rate: int = 4):
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def extract_features(self, record: CTGRecord) -> np.ndarray:
        """Extract feature vector from a single CTG record."""
        fhr = record.fhr.copy()
        uc = record.uc.copy()
        
        # Handle invalid FHR values (0 or out of range)
        valid_mask = (fhr > 50) & (fhr < 210)
        fhr_valid = fhr[valid_mask] if valid_mask.any() else fhr
        
        features = []
        
        # === FHR Statistical Features ===
        features.append(np.mean(fhr_valid))                    # fhr_mean
        features.append(np.std(fhr_valid))                     # fhr_std
        features.append(np.min(fhr_valid))                     # fhr_min
        features.append(np.max(fhr_valid))                     # fhr_max
        features.append(np.max(fhr_valid) - np.min(fhr_valid)) # fhr_range
        features.append(np.median(fhr_valid))                  # fhr_median
        features.append(np.percentile(fhr_valid, 25))          # fhr_q25
        features.append(np.percentile(fhr_valid, 75))          # fhr_q75
        features.append(np.percentile(fhr_valid, 75) - np.percentile(fhr_valid, 25))  # fhr_iqr
        
        # === Baseline & Variability ===
        baseline = self._compute_baseline(fhr_valid)
        features.append(baseline)                              # fhr_baseline
        
        variability = self._compute_variability(fhr_valid, baseline)
        features.append(variability)                           # fhr_variability
        
        stv = self._compute_stv(fhr_valid)
        features.append(stv)                                   # fhr_stv (short-term)
        
        ltv = self._compute_ltv(fhr_valid)
        features.append(ltv)                                   # fhr_ltv (long-term)
        
        # === Pathological Time Percentages ===
        time_below_110 = np.sum(fhr_valid < 110) / len(fhr_valid)
        features.append(time_below_110)                        # fhr_time_below_110
        
        time_above_160 = np.sum(fhr_valid > 160) / len(fhr_valid)
        features.append(time_above_160)                        # fhr_time_above_160
        
        brady_episodes = self._count_episodes(fhr_valid, threshold=110, below=True)
        features.append(brady_episodes)                        # fhr_bradycardia_episodes
        
        tachy_episodes = self._count_episodes(fhr_valid, threshold=160, below=False)
        features.append(tachy_episodes)                        # fhr_tachycardia_episodes
        
        # === Deceleration Features ===
        decels = self._detect_decelerations(fhr_valid, baseline)
        features.append(decels['count'])                       # decel_count
        features.append(decels['mean_depth'])                  # decel_mean_depth
        features.append(decels['max_depth'])                   # decel_max_depth
        features.append(decels['total_duration'])              # decel_total_duration
        
        late_score = self._compute_late_decel_score(fhr, uc, baseline)
        features.append(late_score)                            # late_decel_score
        
        # === Acceleration Features ===
        accels = self._detect_accelerations(fhr_valid, baseline)
        features.append(accels['count'])                       # accel_count
        features.append(accels['mean_amplitude'])              # accel_mean_amplitude
        
        # === UC Features ===
        uc_valid = uc[uc > 0] if (uc > 0).any() else uc
        features.append(np.mean(uc_valid))                     # uc_mean
        features.append(np.std(uc_valid))                      # uc_std
        
        uc_freq = self._compute_uc_frequency(uc)
        features.append(uc_freq)                               # uc_frequency
        
        uc_reg = self._compute_uc_regularity(uc)
        features.append(uc_reg)                                # uc_regularity
        
        # === Cross-correlation ===
        corr = self._compute_fhr_uc_correlation(fhr_valid, uc[:len(fhr_valid)] if len(uc) >= len(fhr_valid) else uc)
        features.append(corr)                                  # fhr_uc_correlation
        
        lag_corr = self._compute_lag_correlation(fhr, uc)
        features.append(lag_corr)                              # fhr_uc_lag_correlation
        
        # === Signal Quality ===
        valid_ratio = np.sum(valid_mask) / len(fhr)
        features.append(valid_ratio)                           # fhr_valid_ratio
        
        gaps = self._count_signal_gaps(fhr)
        features.append(gaps)                                  # signal_gaps
        
        return np.array(features, dtype=np.float32)
    
    def _compute_baseline(self, fhr: np.ndarray) -> float:
        """Compute FHR baseline using moving window mode."""
        window_size = self.sampling_rate * 60  # 1 minute
        if len(fhr) < window_size:
            return np.median(fhr)
        
        # Use overlapping windows
        baselines = []
        for i in range(0, len(fhr) - window_size, window_size // 2):
            segment = fhr[i:i + window_size]
            # Mode approximation using histogram
            hist, edges = np.histogram(segment, bins=20)
            baselines.append(edges[np.argmax(hist)])
        
        return np.median(baselines) if baselines else np.median(fhr)
    
    def _compute_variability(self, fhr: np.ndarray, baseline: float) -> float:
        """Compute FHR variability (amplitude range in 1-min segments)."""
        window_size = self.sampling_rate * 60
        if len(fhr) < window_size:
            return np.std(fhr)
        
        variabilities = []
        for i in range(0, len(fhr) - window_size, window_size):
            segment = fhr[i:i + window_size]
            variabilities.append(np.max(segment) - np.min(segment))
        
        return np.mean(variabilities) if variabilities else np.std(fhr)
    
    def _compute_stv(self, fhr: np.ndarray) -> float:
        """Short-term variability (beat-to-beat)."""
        if len(fhr) < 2:
            return 0.0
        diffs = np.abs(np.diff(fhr))
        return np.mean(diffs)
    
    def _compute_ltv(self, fhr: np.ndarray) -> float:
        """Long-term variability (minute-to-minute)."""
        window = self.sampling_rate * 60
        if len(fhr) < 2 * window:
            return np.std(fhr)
        
        minute_means = []
        for i in range(0, len(fhr) - window, window):
            minute_means.append(np.mean(fhr[i:i + window]))
        
        return np.std(minute_means) if len(minute_means) > 1 else 0.0
    
    def _count_episodes(self, fhr: np.ndarray, threshold: float, below: bool, 
                       min_duration_sec: int = 60) -> int:
        """Count episodes where FHR crosses threshold for minimum duration."""
        min_samples = min_duration_sec * self.sampling_rate
        
        if below:
            mask = fhr < threshold
        else:
            mask = fhr > threshold
        
        # Find contiguous regions
        episodes = 0
        in_episode = False
        episode_length = 0
        
        for val in mask:
            if val:
                episode_length += 1
                if not in_episode and episode_length >= min_samples:
                    episodes += 1
                    in_episode = True
            else:
                in_episode = False
                episode_length = 0
        
        return episodes
    
    def _detect_decelerations(self, fhr: np.ndarray, baseline: float) -> Dict:
        """Detect FHR decelerations (drops > 15 bpm below baseline)."""
        threshold = baseline - 15
        decel_mask = fhr < threshold
        
        # Find contiguous deceleration regions
        decels = []
        in_decel = False
        start = 0
        
        for i, is_decel in enumerate(decel_mask):
            if is_decel and not in_decel:
                start = i
                in_decel = True
            elif not is_decel and in_decel:
                if i - start >= self.sampling_rate * 15:  # Min 15 seconds
                    depth = baseline - np.min(fhr[start:i])
                    decels.append({'start': start, 'end': i, 'depth': depth})
                in_decel = False
        
        return {
            'count': len(decels),
            'mean_depth': np.mean([d['depth'] for d in decels]) if decels else 0,
            'max_depth': max([d['depth'] for d in decels]) if decels else 0,
            'total_duration': sum([d['end'] - d['start'] for d in decels]) / self.sampling_rate if decels else 0
        }
    
    def _compute_late_decel_score(self, fhr: np.ndarray, uc: np.ndarray, baseline: float) -> float:
        """
        Score for late decelerations (FHR nadir occurs after UC peak).
        Higher score = more concerning.
        """
        # Simplified: check correlation with lag
        min_len = min(len(fhr), len(uc))
        if min_len < self.sampling_rate * 60:
            return 0.0
        
        fhr_seg = fhr[:min_len]
        uc_seg = uc[:min_len]
        
        # Detect UC peaks and check if FHR drops follow
        late_score = 0.0
        window = self.sampling_rate * 30  # 30 second lag window
        
        # Find significant UC peaks
        uc_threshold = np.mean(uc_seg) + np.std(uc_seg)
        
        for i in range(window, min_len - window):
            # Check for UC peak
            if uc_seg[i] > uc_threshold and uc_seg[i] == max(uc_seg[max(0,i-10):i+10]):
                # Check for FHR drop 15-60 seconds later
                fhr_after = fhr_seg[i:i+window]
                if len(fhr_after) > 0 and np.min(fhr_after) < baseline - 15:
                    late_score += 1
        
        return late_score / max(1, min_len / (self.sampling_rate * 60))  # Normalize per minute
    
    def _detect_accelerations(self, fhr: np.ndarray, baseline: float) -> Dict:
        """Detect FHR accelerations (rises > 15 bpm above baseline for > 15 sec)."""
        threshold = baseline + 15
        accel_mask = fhr > threshold
        
        accels = []
        in_accel = False
        start = 0
        
        for i, is_accel in enumerate(accel_mask):
            if is_accel and not in_accel:
                start = i
                in_accel = True
            elif not is_accel and in_accel:
                if i - start >= self.sampling_rate * 15:
                    amplitude = np.max(fhr[start:i]) - baseline
                    accels.append({'amplitude': amplitude})
                in_accel = False
        
        return {
            'count': len(accels),
            'mean_amplitude': np.mean([a['amplitude'] for a in accels]) if accels else 0
        }
    
    def _compute_uc_frequency(self, uc: np.ndarray) -> float:
        """Estimate contractions per 10 minutes."""
        if len(uc) < self.sampling_rate * 60:
            return 0.0
        
        # Simple peak counting
        threshold = np.mean(uc) + 0.5 * np.std(uc)
        peaks = 0
        in_peak = False
        
        for val in uc:
            if val > threshold and not in_peak:
                peaks += 1
                in_peak = True
            elif val < threshold:
                in_peak = False
        
        duration_min = len(uc) / (self.sampling_rate * 60)
        return (peaks / duration_min) * 10 if duration_min > 0 else 0
    
    def _compute_uc_regularity(self, uc: np.ndarray) -> float:
        """Measure regularity of contractions (lower = more regular)."""
        threshold = np.mean(uc) + 0.5 * np.std(uc)
        
        # Find peak intervals
        in_peak = False
        last_peak = 0
        intervals = []
        
        for i, val in enumerate(uc):
            if val > threshold and not in_peak:
                if last_peak > 0:
                    intervals.append(i - last_peak)
                last_peak = i
                in_peak = True
            elif val < threshold:
                in_peak = False
        
        if len(intervals) < 2:
            return 1.0  # No regularity data
        
        return np.std(intervals) / np.mean(intervals)  # CV of intervals
    
    def _compute_fhr_uc_correlation(self, fhr: np.ndarray, uc: np.ndarray) -> float:
        """Correlation between FHR and UC signals."""
        min_len = min(len(fhr), len(uc))
        if min_len < 10:
            return 0.0
        
        corr_val = np.corrcoef(fhr[:min_len], uc[:min_len])[0, 1]
        return corr_val if not np.isnan(corr_val) else 0.0
    
    def _compute_lag_correlation(self, fhr: np.ndarray, uc: np.ndarray, 
                                 max_lag_sec: int = 60) -> float:
        """Max correlation with lag (for late decel detection)."""
        min_len = min(len(fhr), len(uc))
        max_lag = max_lag_sec * self.sampling_rate
        
        if min_len < max_lag * 2:
            return 0.0
        
        best_corr = 0.0
        for lag in range(0, max_lag, self.sampling_rate * 5):  # 5-sec steps
            if lag < min_len:
                corr = np.corrcoef(fhr[lag:min_len], uc[:min_len-lag])[0, 1]
                if not np.isnan(corr):
                    best_corr = max(best_corr, abs(corr))
        
        return best_corr
    
    def _count_signal_gaps(self, fhr: np.ndarray) -> int:
        """Count number of signal dropout gaps (consecutive zeros)."""
        min_gap = self.sampling_rate * 5  # 5 second gaps
        
        gaps = 0
        zero_run = 0
        
        for val in fhr:
            if val == 0 or val < 50:
                zero_run += 1
            else:
                if zero_run >= min_gap:
                    gaps += 1
                zero_run = 0
        
        return gaps
    
    def fit_transform(self, records: List[CTGRecord]) -> np.ndarray:
        """Extract features from all records and fit scaler."""
        logger.info(f"Extracting features from {len(records)} records...")
        
        features = []
        for record in records:
            feat = self.extract_features(record)
            features.append(feat)
        
        X = np.array(features)
        
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit and transform scaler
        X_scaled = self.scaler.fit_transform(X)
        self._is_fitted = True
        
        return X_scaled
    
    def transform(self, records: List[CTGRecord]) -> np.ndarray:
        """Extract and scale features (scaler must be fitted)."""
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        features = []
        for record in records:
            feat = self.extract_features(record)
            features.append(feat)
        
        X = np.array(features)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return self.scaler.transform(X)


# =============================================================================
# XGBoost Training with Patient-Level Cross-Validation
# =============================================================================

def train_with_cross_validation(
    records: List[CTGRecord],
    n_folds: int = 10,
    use_smote: bool = True
) -> Tuple[xgb.XGBClassifier, CTGFeatureExtractor, Dict]:
    """
    Train XGBoost with strict patient-level stratified cross-validation.
    
    CRITICAL: Split by patient (record), NOT by timepoints, to prevent data leakage.
    """
    logger.info("=" * 70)
    logger.info("STARTING 10-FOLD STRATIFIED CROSS-VALIDATION")
    logger.info("=" * 70)
    
    # Extract features from all records
    extractor = CTGFeatureExtractor()
    X_all = extractor.fit_transform(records)
    y_all = np.array([r.category for r in records])
    patient_ids = np.array([r.record_id for r in records])
    
    logger.info(f"Dataset: {len(records)} patients, {X_all.shape[1]} features")
    logger.info(f"Class distribution: Cat1={sum(y_all==1)}, Cat2={sum(y_all==2)}, Cat3={sum(y_all==3)}")
    
    # Stratified K-Fold at PATIENT level
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]
        
        # Apply SMOTE only to training data (if available and enabled)
        if use_smote and SMOTE_AVAILABLE and len(np.unique(y_train)) > 1:
            try:
                # Only oversample if minority class has enough samples
                min_class_count = min(np.bincount(y_train)[1:])  # Skip class 0
                if min_class_count >= 2:
                    smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    logger.info(f"  SMOTE: {len(y_train)} -> {len(y_train_resampled)} samples")
                else:
                    X_train_resampled, y_train_resampled = X_train, y_train
            except Exception as e:
                logger.warning(f"  SMOTE failed: {e}, using original data")
                X_train_resampled, y_train_resampled = X_train, y_train
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Compute class weights for imbalanced data
        class_counts = np.bincount(y_train_resampled, minlength=4)[1:]  # Classes 1,2,3
        total = len(y_train_resampled)
        class_weights = {i+1: total / (3 * count) if count > 0 else 1.0 for i, count in enumerate(class_counts)}
        sample_weights = np.array([class_weights[y] for y in y_train_resampled])
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # Adjust labels to 0-indexed for XGBoost
        y_train_adj = y_train_resampled - 1
        y_val_adj = y_val - 1
        
        model.fit(X_train_resampled, y_train_adj, sample_weight=sample_weights)
        
        # Validate on held-out fold (RAW data, no SMOTE)
        y_pred_adj = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        # Convert back to 1-indexed
        y_pred = y_pred_adj + 1
        
        # Metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        try:
            # Multi-class ROC-AUC
            roc_auc = roc_auc_score(y_val - 1, y_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = 0.0
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'n_train': len(y_train_resampled),
            'n_val': len(y_val)
        })
        
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        
        logger.info(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # Aggregate results
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    mean_f1 = np.mean([r['f1_score'] for r in fold_results])
    mean_auc = np.mean([r['roc_auc'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    std_auc = np.std([r['roc_auc'] for r in fold_results])
    
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-VALIDATION RESULTS (MEAN ± STD)")
    logger.info("=" * 70)
    logger.info(f"  Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"  F1-Score:  {mean_f1:.4f}")
    logger.info(f"  ROC-AUC:   {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Confusion matrix on all validation predictions
    logger.info("\nAggregate Confusion Matrix (all folds):")
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[1, 2, 3])
    logger.info(f"  {'':>12} Pred_1  Pred_2  Pred_3")
    logger.info(f"  {'True_1':>12} {cm[0,0]:>6}  {cm[0,1]:>6}  {cm[0,2]:>6}")
    logger.info(f"  {'True_2':>12} {cm[1,0]:>6}  {cm[1,1]:>6}  {cm[1,2]:>6}")
    logger.info(f"  {'True_3':>12} {cm[2,0]:>6}  {cm[2,1]:>6}  {cm[2,2]:>6}")
    
    cv_results = {
        'n_folds': n_folds,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_f1': mean_f1,
        'mean_roc_auc': mean_auc,
        'std_roc_auc': std_auc,
        'fold_results': fold_results,
        'confusion_matrix': cm.tolist()
    }
    
    return None, extractor, cv_results


def train_final_model(
    records: List[CTGRecord],
    extractor: CTGFeatureExtractor
) -> xgb.XGBClassifier:
    """
    Train final production model on ALL data (after CV validation).
    This is the "battle-tested" model that goes to production.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING FINAL PRODUCTION MODEL ON ALL DATA")
    logger.info("=" * 70)
    
    # Re-extract features with the same fitted scaler
    X_all = extractor.transform(records)
    y_all = np.array([r.category for r in records])
    
    # Apply SMOTE if available
    if SMOTE_AVAILABLE:
        try:
            min_class_count = min(np.bincount(y_all)[1:])
            if min_class_count >= 2:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
                X_resampled, y_resampled = smote.fit_resample(X_all, y_all)
                logger.info(f"SMOTE applied: {len(y_all)} -> {len(y_resampled)} samples")
            else:
                X_resampled, y_resampled = X_all, y_all
        except:
            X_resampled, y_resampled = X_all, y_all
    else:
        X_resampled, y_resampled = X_all, y_all
    
    # Compute class weights
    class_counts = np.bincount(y_resampled, minlength=4)[1:]
    total = len(y_resampled)
    class_weights = {i+1: total / (3 * count) if count > 0 else 1.0 for i, count in enumerate(class_counts)}
    sample_weights = np.array([class_weights[y] for y in y_resampled])
    
    # Train final model with same hyperparameters
    final_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # 0-indexed labels
    y_adj = y_resampled - 1
    final_model.fit(X_resampled, y_adj, sample_weight=sample_weights)
    
    # Quick sanity check on training data
    y_pred = final_model.predict(X_all) + 1
    train_acc = accuracy_score(y_all, y_pred)
    logger.info(f"Final model training accuracy: {train_acc:.4f}")
    
    return final_model


def save_pipeline(
    model: xgb.XGBClassifier,
    extractor: CTGFeatureExtractor,
    cv_results: Dict,
    output_path: Path
):
    """Save complete pipeline (model + scaler + metadata)."""
    pipeline = {
        'model': model,
        'scaler': extractor.scaler,
        'feature_names': CTGFeatureExtractor.FEATURE_NAMES,
        'cv_results': cv_results,
        'version': '2.0-realdata',
        'n_patients': cv_results.get('n_patients', 0),
        'data_source': 'CTU-CHB Intrapartum CTG Database',
        'label_mapping': {
            1: 'Normal (pH >= 7.20)',
            2: 'Suspicious (pH 7.15-7.19)',
            3: 'Pathological (pH < 7.15)'
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    logger.info(f"Pipeline saved to: {output_path}")


def run_verification_test():
    """Run the integration test to verify model works with pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING VERIFICATION TEST")
    logger.info("=" * 70)
    
    test_script = PROJECT_ROOT / "tests" / "trace_execution_simple.py"
    
    if test_script.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode == 0:
            logger.info("✅ Verification test PASSED!")
            print(result.stdout)
        else:
            logger.error("❌ Verification test FAILED!")
            print(result.stderr)
            return False
    else:
        logger.warning(f"Verification script not found: {test_script}")
    
    return True


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("🎯 SentinelFetal XGBoost Training Pipeline")
    print("📊 REAL CLINICAL DATA: CTU-CHB Intrapartum CTG Database")
    print("=" * 70 + "\n")
    
    # 1. Load all CTG records
    logger.info("Step 1: Loading CTU-CHB Database...")
    loader = CTUCHBDataLoader(DATA_DIR)
    records = loader.load_all_records()
    
    if len(records) < 50:
        logger.error(f"Only {len(records)} records loaded. Check data directory: {DATA_DIR}")
        sys.exit(1)
    
    # Show class distribution
    categories = [r.category for r in records]
    logger.info(f"\nClass Distribution (pH-based labels):")
    logger.info(f"  Category 1 (Normal, pH≥7.20):      {categories.count(1):>4} ({100*categories.count(1)/len(records):.1f}%)")
    logger.info(f"  Category 2 (Suspicious, pH 7.15-7.19): {categories.count(2):>4} ({100*categories.count(2)/len(records):.1f}%)")
    logger.info(f"  Category 3 (Pathological, pH<7.15):    {categories.count(3):>4} ({100*categories.count(3)/len(records):.1f}%)")
    
    # 2. Run cross-validation
    logger.info("\nStep 2: Running 10-Fold Stratified Cross-Validation...")
    _, extractor, cv_results = train_with_cross_validation(records, n_folds=10, use_smote=True)
    cv_results['n_patients'] = len(records)
    
    # 3. Train final model on all data
    logger.info("\nStep 3: Training Final Model on All Data...")
    
    # Re-fit extractor on all data for final model
    final_extractor = CTGFeatureExtractor()
    _ = final_extractor.fit_transform(records)
    
    final_model = train_final_model(records, final_extractor)
    
    # 4. Save pipeline
    logger.info("\nStep 4: Saving Pipeline...")
    output_path = MODELS_DIR / "ctg_xgboost_pipeline.pkl"
    save_pipeline(final_model, final_extractor, cv_results, output_path)
    
    # 5. Run verification
    logger.info("\nStep 5: Verification...")
    run_verification_test()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Model saved to: {output_path}")
    print(f"📊 Trained on: {len(records)} real clinical CTG recordings")
    print(f"🔬 Cross-Validation Results:")
    print(f"   • Accuracy: {cv_results['mean_accuracy']:.2%} ± {cv_results['std_accuracy']:.2%}")
    print(f"   • ROC-AUC:  {cv_results['mean_roc_auc']:.3f} ± {cv_results['std_roc_auc']:.3f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
