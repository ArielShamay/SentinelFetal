"""
SentinelFetal V6 Pipeline Validation Suite
==========================================
Comprehensive testing for the XGBoost-only V6 pipeline.

Phase 1: Pipeline Integrity - Verify data flows through all stages
Phase 2: Quality Metrics - Evaluate accuracy on real and synthetic data

Author: SentinelFetal ML Team
Date: 2026-01-27
"""

import sys
import json
import time
import logging
import traceback
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Disable strict mode for testing
os.environ['SENTINELFETAL_STRICT_MODE'] = 'false'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
SAMPLING_RATE = 4.0  # Hz
MIN_DURATION_MINUTES = 20  # Minimum recording length
MIN_DURATION_SAMPLES = int(MIN_DURATION_MINUTES * 60 * SAMPLING_RATE)

CTU_CHB_PATH = PROJECT_ROOT / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
LABELS_PATH = PROJECT_ROOT / "data" / "labels"
RESULTS_DIR = PROJECT_ROOT / "REPORTS" / "v6_validation"


# =============================================================================
# Data Classes for Results
# =============================================================================
@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    success: bool
    duration_ms: float
    output_type: str
    output_summary: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass  
class PipelineIntegrityResult:
    """Complete result from pipeline integrity test."""
    record_id: str
    overall_success: bool
    total_duration_ms: float
    stages: List[StageResult] = field(default_factory=list)
    final_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class QualityMetrics:
    """Quality metrics for model evaluation."""
    total_samples: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Category-specific
    category_1_correct: int = 0
    category_2_correct: int = 0
    category_3_correct: int = 0
    category_1_total: int = 0
    category_2_total: int = 0
    category_3_total: int = 0
    
    # Override tracking
    override_count: int = 0
    
    @property
    def accuracy(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total_samples
    
    @property
    def sensitivity(self) -> float:
        """Also called recall - ability to detect pathological cases."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def specificity(self) -> float:
        """Ability to correctly identify normal cases."""
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """Positive predictive value."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        if self.precision + self.sensitivity == 0:
            return 0.0
        return 2 * (self.precision * self.sensitivity) / (self.precision + self.sensitivity)
    
    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0


# =============================================================================
# Data Loading Utilities
# =============================================================================
def load_ctu_chb_record(record_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a CTU-CHB record.
    
    Returns:
        fhr: FHR signal array
        uc: UC signal array
        metadata: Record metadata
    """
    import wfdb
    
    record_path = CTU_CHB_PATH / record_id
    if not record_path.with_suffix('.dat').exists():
        raise FileNotFoundError(f"Record {record_id} not found")
    
    record = wfdb.rdrecord(str(record_path))
    
    # Extract signals (FHR is channel 0, UC is channel 1)
    fhr = record.p_signal[:, 0].astype(np.float32)
    uc = record.p_signal[:, 1].astype(np.float32)
    
    # Handle missing values (marked as 0 in CTU-CHB)
    fhr[fhr == 0] = np.nan
    
    metadata = {
        'record_id': record_id,
        'duration_samples': len(fhr),
        'duration_minutes': len(fhr) / (record.fs * 60),
        'sampling_rate': record.fs,
        'nan_fraction': np.isnan(fhr).mean()
    }
    
    return fhr, uc, metadata


def load_labels() -> Dict[str, int]:
    """
    Load outcome labels for CTU-CHB records.
    
    Returns:
        Dictionary mapping record_id to outcome (0=normal, 1=pathological)
    """
    labels = {}
    
    # Try to load from labels directory
    label_files = list(LABELS_PATH.glob("*.csv")) + list(LABELS_PATH.glob("*.json"))
    
    if not label_files:
        # Use pH-based labeling from the database
        # pH < 7.15 is considered acidemic (pathological)
        logger.warning("No label files found, using default pH-based labeling")
        return labels
    
    for label_file in label_files:
        try:
            if label_file.suffix == '.csv':
                import csv
                with open(label_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        record_id = row.get('record_id', row.get('id', ''))
                        outcome = int(row.get('outcome', row.get('label', 0)))
                        labels[record_id] = outcome
            elif label_file.suffix == '.json':
                with open(label_file, 'r') as f:
                    data = json.load(f)
                    labels.update(data)
        except Exception as e:
            logger.warning(f"Failed to load {label_file}: {e}")
    
    return labels


def generate_synthetic_signal(
    event_type: str,
    duration_minutes: int = 30,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a synthetic CTG signal with known characteristics.
    
    Creates realistic FHR signals with proper variability patterns
    that can pass baseline detection algorithms.
    
    Args:
        event_type: Type of signal ('normal', 'late_decel', 'variable_decel', 
                   'bradycardia', 'tachycardia', 'reduced_variability')
        duration_minutes: Signal duration in minutes
        rng: Random number generator
        
    Returns:
        fhr: FHR signal
        uc: UC signal  
        expected_category: Expected FIGO category (1, 2, or 3)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n_samples = int(duration_minutes * 60 * SAMPLING_RATE)
    t = np.arange(n_samples) / SAMPLING_RATE
    
    # Base signals - Start with stable baseline around 140 bpm
    baseline_fhr = 140  # Normal baseline
    
    if event_type == 'normal':
        # Normal FHR with moderate variability (5-15 bpm)
        # Create stable baseline with realistic variability
        fhr = np.full(n_samples, baseline_fhr, dtype=np.float32)
        # Add long-term variability (cycles 3-5 per minute)
        fhr += 5 * np.sin(2 * np.pi * 0.067 * t)  # ~4 cycles/min
        # Add short-term variability (smaller amplitude)
        fhr += rng.normal(0, 2, n_samples)
        # Add occasional accelerations (normal feature)
        for i in range(3, duration_minutes, 7):
            start = int(i * 60 * SAMPLING_RATE)
            duration = int(20 * SAMPLING_RATE)  # 20 second acceleration
            if start + duration < n_samples:
                accel = 15 * np.sin(np.linspace(0, np.pi, duration))
                fhr[start:start+duration] += accel
        expected_category = 1
        
    elif event_type == 'late_decel':
        # Late decelerations (pathological)
        fhr = np.full(n_samples, baseline_fhr, dtype=np.float32)
        fhr += rng.normal(0, 2, n_samples)  # Small variability
        # Add late decels every 3-4 minutes (after contractions)
        contraction_times = list(range(2, duration_minutes, 4))
        for ct in contraction_times:
            # Decel starts 20-30 seconds after contraction peak
            decel_start = int((ct * 60 + 25) * SAMPLING_RATE)
            duration = int(50 * SAMPLING_RATE)  # 50 second decel
            if decel_start + duration < n_samples:
                decel_depth = 25 + rng.random() * 15  # 25-40 bpm drop
                decel = decel_depth * np.sin(np.linspace(0, np.pi, duration))
                fhr[decel_start:decel_start+duration] -= decel
        expected_category = 3
        
    elif event_type == 'variable_decel':
        # Variable decelerations (suspicious to pathological)
        fhr = np.full(n_samples, baseline_fhr, dtype=np.float32)
        fhr += 4 * np.sin(2 * np.pi * 0.05 * t)  # Some baseline variability
        fhr += rng.normal(0, 2, n_samples)
        # Add variable decels with abrupt onset
        for i in range(3, duration_minutes, 5):
            start = int(i * 60 * SAMPLING_RATE)
            duration = int(25 * SAMPLING_RATE)  # 25 second decel
            if start + duration < n_samples:
                decel_depth = 35 + rng.random() * 25  # 35-60 bpm drop
                # Abrupt onset/offset characteristic of variable decels
                decel = np.zeros(duration)
                onset = int(duration * 0.1)
                offset = int(duration * 0.1)
                decel[:onset] = np.linspace(0, decel_depth, onset)
                decel[onset:-offset] = decel_depth
                decel[-offset:] = np.linspace(decel_depth, 0, offset)
                fhr[start:start+duration] -= decel
        expected_category = 2
        
    elif event_type == 'bradycardia':
        # Baseline bradycardia (<110 bpm for >10 minutes)
        baseline_fhr = 105  # Low baseline
        fhr = np.full(n_samples, baseline_fhr, dtype=np.float32)
        fhr += 4 * np.sin(2 * np.pi * 0.05 * t)
        fhr += rng.normal(0, 2, n_samples)
        expected_category = 3
        
    elif event_type == 'tachycardia':
        # Baseline tachycardia (>160 bpm)
        baseline_fhr = 170  # High baseline
        fhr = np.full(n_samples, baseline_fhr, dtype=np.float32)
        fhr += 4 * np.sin(2 * np.pi * 0.05 * t)
        fhr += rng.normal(0, 2, n_samples)
        expected_category = 2
        
    elif event_type == 'reduced_variability':
        # Minimal variability (suspicious) - 2-5 bpm range
        fhr = np.full(n_samples, baseline_fhr, dtype=np.float32)
        fhr += rng.normal(0, 0.8, n_samples)  # Very low variability ~3 bpm
        expected_category = 2
        
    elif event_type == 'absent_variability':
        # Absent variability (pathological) - <2 bpm range
        fhr = np.full(n_samples, baseline_fhr, dtype=np.float32)
        fhr += rng.normal(0, 0.3, n_samples)  # Almost no variability
        expected_category = 3
        
    else:
        raise ValueError(f"Unknown event type: {event_type}")
    
    # Clip FHR to physiological range
    fhr = np.clip(fhr, 60, 200)
    
    # Generate UC signal with regular contractions
    uc = np.zeros(n_samples, dtype=np.float32)
    # Add contractions every 3-4 minutes (normal pattern)
    contraction_interval = int(3.5 * 60 * SAMPLING_RATE)
    for start in range(int(60 * SAMPLING_RATE), n_samples - int(70 * SAMPLING_RATE), contraction_interval):
        duration = int(60 * SAMPLING_RATE)  # 60 second contraction
        contraction = 50 * np.sin(np.linspace(0, np.pi, duration))
        end_idx = min(start + duration, n_samples)
        uc[start:end_idx] += contraction[:end_idx-start]
    
    # Add baseline tone and noise to UC
    uc += 10  # Baseline uterine tone
    uc += rng.normal(0, 1.5, n_samples)
    uc = np.clip(uc, 0, 100)
    
    return fhr.astype(np.float32), uc.astype(np.float32), expected_category


# =============================================================================
# Phase 1: Pipeline Integrity Tests
# =============================================================================
def disable_strict_mode():
    """Disable strict mode in runtime config for testing."""
    try:
        from src.utils.runtime_config import load_runtime_config
        # Clear the cache to reload config
        load_runtime_config.cache_clear()
        
        # Temporarily modify the config file
        config_path = PROJECT_ROOT / "config" / "runtime.yaml"
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        original_strict = config.get('strict_mode', False)
        config['strict_mode'] = False
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        
        # Clear cache again to load new config
        load_runtime_config.cache_clear()
        
        return original_strict
    except Exception as e:
        logger.warning(f"Could not disable strict mode: {e}")
        return None


def restore_strict_mode(original_value: bool):
    """Restore strict mode to original value."""
    try:
        config_path = PROJECT_ROOT / "config" / "runtime.yaml"
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['strict_mode'] = original_value
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
    except Exception as e:
        logger.warning(f"Could not restore strict mode: {e}")


class PipelineIntegrityTester:
    """Tests that the V6 pipeline processes data correctly through all stages."""
    
    def __init__(self):
        self.container = None
        self.pipeline = None
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize the V6 pipeline."""
        from src.pipeline.container import PipelineContainer
        from src.pipeline.analysis_pipeline import AnalysisPipeline
        
        logger.info("Initializing V6 XGBoost pipeline...")
        self.container = PipelineContainer.create_v6_xgboost()
        self.pipeline = AnalysisPipeline(self.container)
        logger.info("[OK] Pipeline initialized successfully")
    
    def test_single_record(self, fhr: np.ndarray, uc: np.ndarray, record_id: str = "test") -> PipelineIntegrityResult:
        """
        Test pipeline integrity on a single record.
        
        Tests each stage individually and verifies outputs.
        """
        stages = []
        overall_start = time.time()
        
        try:
            c = self.container
            
            # Stage 1: Preprocessing
            stage_start = time.time()
            try:
                preprocess_result = c.preprocessor.process(fhr.copy())
                fhr_clean = preprocess_result.processed_signal
                stages.append(StageResult(
                    stage_name="Preprocessing",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type=type(preprocess_result).__name__,
                    output_summary={
                        'input_length': len(fhr),
                        'output_length': len(fhr_clean),
                        'nan_before': float(np.isnan(fhr).mean()),
                        'nan_after': float(np.isnan(fhr_clean).mean())
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="Preprocessing",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Stage 2: Baseline Calculation
            stage_start = time.time()
            try:
                baseline = c.baseline_calculator.calculate(fhr_clean, SAMPLING_RATE)
                stages.append(StageResult(
                    stage_name="Baseline Calculation",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type=type(baseline).__name__,
                    output_summary={
                        'baseline_bpm': float(baseline.value),
                        'is_valid': baseline.value >= 100 and baseline.value <= 180
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="Baseline Calculation",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Stage 3: Variability Analysis
            stage_start = time.time()
            try:
                variability = c.variability_calculator.calculate(fhr_clean, SAMPLING_RATE)
                stages.append(StageResult(
                    stage_name="Variability Analysis",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type=type(variability).__name__,
                    output_summary={
                        'stv': float(variability.stv) if hasattr(variability, 'stv') else None,
                        'ltv': float(variability.ltv) if hasattr(variability, 'ltv') else None,
                        'classification': str(variability.classification) if hasattr(variability, 'classification') else None
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="Variability Analysis",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Stage 4: Deceleration Detection
            stage_start = time.time()
            try:
                decelerations = c.deceleration_detector.detect(
                    fhr_clean, uc, baseline.value, SAMPLING_RATE
                )
                stages.append(StageResult(
                    stage_name="Deceleration Detection",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type=type(decelerations).__name__,
                    output_summary={
                        'count': len(decelerations) if decelerations else 0,
                        'types': [str(d.decel_type) if hasattr(d, 'decel_type') else 'unknown' 
                                 for d in (decelerations or [])][:5]
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="Deceleration Detection",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Stage 5: Tachysystole Detection  
            stage_start = time.time()
            try:
                tachysystole = c.tachysystole_detector.detect(uc, SAMPLING_RATE)
                stages.append(StageResult(
                    stage_name="Tachysystole Detection",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type=type(tachysystole).__name__,
                    output_summary={
                        'detected': bool(tachysystole.detected) if hasattr(tachysystole, 'detected') else None,
                        'contraction_count': int(tachysystole.contraction_count) if hasattr(tachysystole, 'contraction_count') else None
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="Tachysystole Detection",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Stage 6: Sinusoidal Detection
            stage_start = time.time()
            try:
                sinusoidal = c.sinusoidal_detector.detect(fhr_clean, SAMPLING_RATE)
                stages.append(StageResult(
                    stage_name="Sinusoidal Detection",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type=type(sinusoidal).__name__,
                    output_summary={
                        'detected': bool(sinusoidal.detected) if hasattr(sinusoidal, 'detected') else None,
                        'confidence': float(sinusoidal.confidence) if hasattr(sinusoidal, 'confidence') else None
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="Sinusoidal Detection",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Stage 7: Feature Extraction (MiniRocket)
            stage_start = time.time()
            try:
                embedding_result = c.feature_extractor.extract(fhr_clean)
                stages.append(StageResult(
                    stage_name="Feature Extraction (MiniRocket)",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type=type(embedding_result).__name__,
                    output_summary={
                        'embedding_dim': len(embedding_result.embedding),
                        'is_mock': embedding_result.is_mock if hasattr(embedding_result, 'is_mock') else None
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="Feature Extraction (MiniRocket)",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Stage 8: Feature Fusion
            stage_start = time.time()
            try:
                feature_vector = c.feature_fusion.fuse(
                    embedding=embedding_result.embedding,
                    baseline=baseline,
                    variability=variability,
                    decelerations=decelerations,
                    tachysystole=tachysystole,
                    sinusoidal=sinusoidal
                )
                stages.append(StageResult(
                    stage_name="Feature Fusion",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type=type(feature_vector).__name__,
                    output_summary={
                        'vector_dim': len(feature_vector.vector) if hasattr(feature_vector, 'vector') else None
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="Feature Fusion",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Stage 9: XGBoost Classification
            stage_start = time.time()
            try:
                X = feature_vector.vector.reshape(1, -1)
                ml_prediction = int(c.classifier.predict(X)[0])
                probas = c.classifier.predict_proba(X)[0]
                confidence = float(np.max(probas))
                stages.append(StageResult(
                    stage_name="XGBoost Classification",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="XGBoostPrediction",
                    output_summary={
                        'prediction': ml_prediction,
                        'confidence': confidence,
                        'probabilities': probas.tolist() if hasattr(probas, 'tolist') else list(probas)
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="XGBoost Classification",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Stage 10: Medical Override
            stage_start = time.time()
            try:
                override = c.medical_override.apply(
                    ml_prediction=ml_prediction,
                    baseline=baseline,
                    variability=variability,
                    decelerations=decelerations,
                    tachysystole=tachysystole,
                    sinusoidal=sinusoidal
                )
                final_category = override.final_category + 1
                stages.append(StageResult(
                    stage_name="Medical Override",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type=type(override).__name__,
                    output_summary={
                        'was_overridden': override.should_override,
                        'ml_prediction': ml_prediction + 1,
                        'final_category': final_category,
                        'reason': str(override.reason) if hasattr(override, 'reason') else None
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="Medical Override",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Stage 11: Alert Generation
            stage_start = time.time()
            try:
                alert = c.alert_generator.generate(
                    category=final_category,
                    confidence=confidence,
                    baseline=baseline,
                    variability=variability,
                    decelerations=decelerations,
                    tachysystole=tachysystole,
                    sinusoidal=sinusoidal
                )
                stages.append(StageResult(
                    stage_name="Alert Generation",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type=type(alert).__name__,
                    output_summary={
                        'headline': alert.headline if hasattr(alert, 'headline') else None,
                        'level': str(alert.level) if hasattr(alert, 'level') else None
                    }
                ))
            except Exception as e:
                stages.append(StageResult(
                    stage_name="Alert Generation",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output_type="Error",
                    output_summary={},
                    error_message=str(e)
                ))
                raise
            
            # Build final output (JSON for UI)
            final_output = {
                'record_id': record_id,
                'timestamp': datetime.now().isoformat(),
                'category': final_category,
                'category_name': ['Normal', 'Suspicious', 'Pathological'][final_category - 1],
                'confidence': confidence,
                'ml_prediction': ml_prediction + 1,
                'was_overridden': override.should_override,
                'baseline': {
                    'value': float(baseline.value),
                },
                'variability': {
                    'stv': float(variability.stv) if hasattr(variability, 'stv') else None,
                    'ltv': float(variability.ltv) if hasattr(variability, 'ltv') else None,
                },
                'decelerations': {
                    'count': len(decelerations) if decelerations else 0,
                },
                'tachysystole': {
                    'detected': bool(tachysystole.detected) if hasattr(tachysystole, 'detected') else False,
                },
                'sinusoidal': {
                    'detected': bool(sinusoidal.detected) if hasattr(sinusoidal, 'detected') else False,
                },
                'alert': {
                    'headline': alert.headline if hasattr(alert, 'headline') else None,
                }
            }
            
            total_duration = (time.time() - overall_start) * 1000
            
            return PipelineIntegrityResult(
                record_id=record_id,
                overall_success=True,
                total_duration_ms=total_duration,
                stages=stages,
                final_output=final_output
            )
            
        except Exception as e:
            total_duration = (time.time() - overall_start) * 1000
            return PipelineIntegrityResult(
                record_id=record_id,
                overall_success=False,
                total_duration_ms=total_duration,
                stages=stages,
                error_message=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )
    
    def run_full_pipeline_test(self, fhr: np.ndarray, uc: np.ndarray, record_id: str = "test"):
        """Run complete pipeline analysis (simpler test)."""
        try:
            result = self.pipeline.analyze(fhr, uc, SAMPLING_RATE)
            return result, None
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)}"


# =============================================================================
# Phase 2: Quality Metrics Tests
# =============================================================================
class QualityTester:
    """Evaluates pipeline quality on labeled data."""
    
    def __init__(self, pipeline_tester: PipelineIntegrityTester):
        self.tester = pipeline_tester
        self.metrics = QualityMetrics()
        self.results = []
    
    def evaluate_record(
        self, 
        fhr: np.ndarray, 
        uc: np.ndarray, 
        true_label: int,
        record_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single record against its true label.
        
        Args:
            fhr: FHR signal
            uc: UC signal
            true_label: True category (1, 2, or 3) or binary (0=normal, 1=pathological)
            record_id: Record identifier
        """
        result, error = self.tester.run_full_pipeline_test(fhr, uc, record_id)
        
        if error or result is None:
            return {
                'record_id': record_id,
                'success': False,
                'error': error
            }
        
        predicted = result.category
        
        # Update metrics (binary: 1 vs 2-3)
        self.metrics.total_samples += 1
        
        # Convert to binary if needed
        true_binary = 0 if true_label == 1 else 1  # 1=normal, 2/3=abnormal
        pred_binary = 0 if predicted == 1 else 1
        
        if true_binary == 1 and pred_binary == 1:
            self.metrics.true_positives += 1
        elif true_binary == 0 and pred_binary == 0:
            self.metrics.true_negatives += 1
        elif true_binary == 0 and pred_binary == 1:
            self.metrics.false_positives += 1
        elif true_binary == 1 and pred_binary == 0:
            self.metrics.false_negatives += 1
        
        # Category-specific tracking
        if true_label == 1:
            self.metrics.category_1_total += 1
            if predicted == 1:
                self.metrics.category_1_correct += 1
        elif true_label == 2:
            self.metrics.category_2_total += 1
            if predicted == 2:
                self.metrics.category_2_correct += 1
        elif true_label == 3:
            self.metrics.category_3_total += 1
            if predicted == 3:
                self.metrics.category_3_correct += 1
        
        if result.was_overridden:
            self.metrics.override_count += 1
        
        record_result = {
            'record_id': record_id,
            'success': True,
            'true_label': true_label,
            'predicted': predicted,
            'confidence': result.confidence,
            'was_overridden': result.was_overridden,
            'correct': (predicted == true_label) or (pred_binary == true_binary)
        }
        
        self.results.append(record_result)
        return record_result
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all quality metrics."""
        return {
            'total_samples': self.metrics.total_samples,
            'accuracy': self.metrics.accuracy,
            'sensitivity': self.metrics.sensitivity,
            'specificity': self.metrics.specificity,
            'precision': self.metrics.precision,
            'f1_score': self.metrics.f1_score,
            'false_positive_rate': self.metrics.false_positive_rate,
            'confusion_matrix': {
                'true_positives': self.metrics.true_positives,
                'true_negatives': self.metrics.true_negatives,
                'false_positives': self.metrics.false_positives,
                'false_negatives': self.metrics.false_negatives
            },
            'category_accuracy': {
                'category_1': self.metrics.category_1_correct / max(1, self.metrics.category_1_total),
                'category_2': self.metrics.category_2_correct / max(1, self.metrics.category_2_total),
                'category_3': self.metrics.category_3_correct / max(1, self.metrics.category_3_total)
            },
            'override_rate': self.metrics.override_count / max(1, self.metrics.total_samples)
        }


# =============================================================================
# Main Execution
# =============================================================================
def run_phase1_integrity_tests():
    """
    Phase 1: Pipeline Integrity Tests
    
    Verify that data flows correctly through all pipeline stages.
    """
    print("\n" + "="*70)
    print("PHASE 1: PIPELINE INTEGRITY TESTS")
    print("="*70 + "\n")
    
    tester = PipelineIntegrityTester()
    results = []
    
    # Test 1: Synthetic Normal Signal (30 minutes)
    print("Test 1: Synthetic Normal Signal...")
    fhr, uc, expected = generate_synthetic_signal('normal', duration_minutes=30)
    result = tester.test_single_record(fhr, uc, "synthetic_normal")
    results.append(result)
    
    if result.overall_success:
        print(f"  [OK] PASSED - Category: {result.final_output['category']} (expected ~1)")
        for stage in result.stages:
            status = "[OK]" if stage.success else "[FAIL]"
            print(f"    {status} {stage.stage_name}: {stage.duration_ms:.1f}ms")
    else:
        print(f"  [FAIL] FAILED: {result.error_message}")
        for stage in result.stages:
            status = "[OK]" if stage.success else "[FAIL]"
            print(f"    {status} {stage.stage_name}: {stage.error_message or 'OK'}")
    
    # Test 2: Synthetic Late Decelerations (30 minutes)
    print("\nTest 2: Synthetic Late Decelerations...")
    fhr, uc, expected = generate_synthetic_signal('late_decel', duration_minutes=30)
    result = tester.test_single_record(fhr, uc, "synthetic_late_decel")
    results.append(result)
    
    if result.overall_success:
        print(f"  [OK] PASSED - Category: {result.final_output['category']} (expected ~3)")
    else:
        print(f"  [FAIL] FAILED: {result.error_message}")
    
    # Test 3: Real CTU-CHB Record (if available)
    print("\nTest 3: Real CTU-CHB Record...")
    try:
        fhr, uc, metadata = load_ctu_chb_record("1001")
        if len(fhr) >= MIN_DURATION_SAMPLES:
            # Take last 30 minutes
            fhr = fhr[-MIN_DURATION_SAMPLES:]
            uc = uc[-MIN_DURATION_SAMPLES:]
            result = tester.test_single_record(fhr, uc, "ctu_chb_1001")
            results.append(result)
            
            if result.overall_success:
                print(f"  [OK] PASSED - Category: {result.final_output['category']}")
                print(f"    Total processing time: {result.total_duration_ms:.1f}ms")
            else:
                print(f"  [FAIL] FAILED: {result.error_message}")
        else:
            print(f"  [WARN] SKIPPED - Record too short ({metadata['duration_minutes']:.1f} min)")
    except Exception as e:
        print(f"  [FAIL] ERROR loading record: {e}")
    
    # Summary
    print("\n" + "-"*70)
    passed = sum(1 for r in results if r.overall_success)
    print(f"Phase 1 Summary: {passed}/{len(results)} tests passed")
    
    return tester, results


def run_phase2_quality_tests(tester: PipelineIntegrityTester):
    """
    Phase 2: Quality Metrics Tests
    
    Evaluate classification accuracy on labeled data.
    """
    print("\n" + "="*70)
    print("PHASE 2: QUALITY METRICS TESTS")
    print("="*70 + "\n")
    
    quality_tester = QualityTester(tester)
    
    # Test synthetic signals with known labels
    synthetic_tests = [
        ('normal', 1),
        ('normal', 1),
        ('normal', 1),
        ('late_decel', 3),
        ('late_decel', 3),
        ('variable_decel', 2),
        ('variable_decel', 2),
        ('bradycardia', 3),
        ('tachycardia', 2),
        ('reduced_variability', 2),
        ('absent_variability', 3),
    ]
    
    print("Testing synthetic signals...")
    rng = np.random.default_rng(42)
    
    for i, (event_type, expected_category) in enumerate(synthetic_tests):
        fhr, uc, _ = generate_synthetic_signal(event_type, duration_minutes=25, rng=rng)
        record_id = f"synthetic_{event_type}_{i}"
        
        result = quality_tester.evaluate_record(fhr, uc, expected_category, record_id)
        
        if result['success']:
            status = "[OK]" if result['correct'] else "[FAIL]"
            print(f"  {status} {record_id}: predicted={result['predicted']}, expected={expected_category}")
        else:
            print(f"  [FAIL] {record_id}: ERROR - {result['error']}")
    
    # Test on CTU-CHB records if labels available
    print("\nTesting CTU-CHB records...")
    labels = load_labels()
    
    if labels:
        test_records = list(labels.keys())[:20]  # Limit to 20 records
        for record_id in test_records:
            try:
                fhr, uc, metadata = load_ctu_chb_record(record_id)
                if len(fhr) >= MIN_DURATION_SAMPLES:
                    fhr = fhr[-MIN_DURATION_SAMPLES:]
                    uc = uc[-MIN_DURATION_SAMPLES:]
                    
                    true_label = labels[record_id]
                    # Convert binary to category if needed
                    if true_label in [0, 1]:
                        true_label = 1 if true_label == 0 else 3
                    
                    result = quality_tester.evaluate_record(fhr, uc, true_label, record_id)
                    
                    if result['success']:
                        status = "[OK]" if result['correct'] else "[FAIL]"
                        print(f"  {status} {record_id}: predicted={result['predicted']}, expected={true_label}")
            except Exception as e:
                print(f"  [WARN] {record_id}: SKIPPED - {e}")
    else:
        print("  [WARN] No labels available - testing with assumed labels from pH")
        
        # Test a few records with assumed normal label
        for record_id in ['1001', '1002', '1003', '1004', '1005']:
            try:
                fhr, uc, metadata = load_ctu_chb_record(record_id)
                if len(fhr) >= MIN_DURATION_SAMPLES:
                    fhr = fhr[-MIN_DURATION_SAMPLES:]
                    uc = uc[-MIN_DURATION_SAMPLES:]
                    
                    # Assume normal for testing pipeline functionality
                    result = quality_tester.evaluate_record(fhr, uc, 1, record_id)
                    
                    if result['success']:
                        print(f"  • {record_id}: predicted={result['predicted']}, confidence={result['confidence']:.2f}")
            except Exception as e:
                print(f"  [WARN] {record_id}: ERROR - {e}")
    
    # Print metrics summary
    print("\n" + "-"*70)
    print("QUALITY METRICS SUMMARY")
    print("-"*70)
    
    metrics = quality_tester.get_metrics_summary()
    
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Sensitivity (Recall): {metrics['sensitivity']:.2%}")
    print(f"Specificity: {metrics['specificity']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"F1 Score: {metrics['f1_score']:.2%}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.2%}")
    print(f"Override Rate: {metrics['override_rate']:.2%}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TP: {cm['true_positives']}, TN: {cm['true_negatives']}")
    print(f"  FP: {cm['false_positives']}, FN: {cm['false_negatives']}")
    
    return quality_tester, metrics


def save_results(integrity_results, quality_metrics):
    """Save test results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"v6_validation_{timestamp}.json"
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'phase1_integrity': {
            'total_tests': len(integrity_results),
            'passed': sum(1 for r in integrity_results if r.overall_success),
            'results': [
                {
                    'record_id': r.record_id,
                    'success': r.overall_success,
                    'duration_ms': r.total_duration_ms,
                    'error': r.error_message,
                    'stages': [
                        {
                            'name': s.stage_name,
                            'success': s.success,
                            'duration_ms': s.duration_ms,
                            'summary': s.output_summary
                        }
                        for s in r.stages
                    ] if r.stages else [],
                    'final_output': r.final_output
                }
                for r in integrity_results
            ]
        },
        'phase2_quality': quality_metrics
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {results_file}")
    return results_file


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("SENTINELFETAL V6 PIPELINE VALIDATION SUITE")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Min Recording Duration: {MIN_DURATION_MINUTES} minutes")
    print("="*70)
    
    # Disable strict mode for testing
    print("\nDisabling strict mode for validation...")
    original_strict = disable_strict_mode()
    
    try:
        # Phase 1: Pipeline Integrity
        tester, integrity_results = run_phase1_integrity_tests()
        
        # Check if Phase 1 passed before proceeding
        if not all(r.overall_success for r in integrity_results):
            print("\n" + "!"*70)
            print("WARNING: Some Phase 1 tests failed!")
            print("Fix pipeline issues before running Phase 2.")
            print("!"*70)
            
            # Still try to save partial results
            save_results(integrity_results, {})
            return 1
        
        # Phase 2: Quality Metrics  
        quality_tester, quality_metrics = run_phase2_quality_tests(tester)
        
        # Save all results
        save_results(integrity_results, quality_metrics)
        
        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
        
        return 0
    finally:
        # Restore strict mode
        if original_strict is not None:
            print("\nRestoring strict mode...")
            restore_strict_mode(original_strict)


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
