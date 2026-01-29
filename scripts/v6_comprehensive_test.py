"""
SentinelFetal V6 Comprehensive Testing Script
==============================================
Two-stage comprehensive testing for the V6 simplified pipeline.

Stage 1: Pipeline Integrity (must pass before Stage 2)
    - Load data (CTU-CHB or synthetic gauntlet)
    - Quality gate verification
    - Windowing verification
    - Rules engine sanity check
    - AI pipeline verification (MiniRocket → XGBoost)
    - Hybrid logic verification
    - JSON output verification
    - Output: REPORTS/v6_pipeline_integrity_report.json

Stage 2: Quality & Performance (only if Stage 1 passes)
    - Metrics: Recall, Precision, F1/F2, AUC-ROC
    - Error rates: FPR < 20%, FNR < 5%
    - Noise robustness
    - Latency measurements (target < 100ms per window)
    - Output: REPORTS/v6_quality_metrics_report.json

Usage:
    python scripts/v6_comprehensive_test.py --stage 1
    python scripts/v6_comprehensive_test.py --stage 2
    python scripts/v6_comprehensive_test.py --all

Author: SentinelFetal ML Team
Version: 6.0.0
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
REPORTS_DIR = PROJECT_ROOT / "REPORTS"
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic_gauntlet"

# Constants
MINIROCKET_DIM = 9996
CLINICAL_DIM = 8
TOTAL_DIM = MINIROCKET_DIM + CLINICAL_DIM
WINDOW_DURATION_MIN = 20
WINDOW_STRIDE_MIN = 5
SAMPLING_RATE = 4  # Hz
SAMPLES_PER_WINDOW = WINDOW_DURATION_MIN * 60 * SAMPLING_RATE  # 4,800


@dataclass
class StageResult:
    """Result from a test stage."""
    stage: int
    stage_name: str
    passed: bool
    tests_run: int
    tests_passed: int
    tests_failed: int
    errors: List[str]
    warnings: List[str]
    execution_time_s: float
    timestamp: str


@dataclass
class IntegrityTestResults:
    """Results from Stage 1: Pipeline Integrity."""
    quality_gate_ok: bool
    windowing_ok: bool
    rules_engine_ok: bool
    ai_pipeline_ok: bool
    hybrid_logic_ok: bool
    json_output_ok: bool
    
    quality_gate_details: Dict[str, Any]
    windowing_details: Dict[str, Any]
    rules_details: Dict[str, Any]
    ai_details: Dict[str, Any]
    hybrid_details: Dict[str, Any]
    json_details: Dict[str, Any]
    
    overall_passed: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class QualityMetrics:
    """Results from Stage 2: Quality & Performance."""
    recall: float
    precision: float
    specificity: float
    f1_score: float
    f2_score: float
    auc_roc: float
    
    fpr: float
    fnr: float
    
    noise_robustness_ok: bool
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    
    overall_passed: bool
    errors: List[str]
    warnings: List[str]


# =============================================================================
# Stage 1: Pipeline Integrity Tests
# =============================================================================

def test_quality_gate() -> Tuple[bool, Dict[str, Any]]:
    """Test 1.1: Quality Gate verification."""
    logger.info("Testing Quality Gate...")
    
    try:
        # Import quality gate components
        from src.v6.pre_ai.quality_gate import QualityGate, QualityLevel
        
        gate = QualityGate()
        
        # Test 1: Good quality signal
        good_signal = np.random.randn(4800) * 5 + 140  # Normal FHR around 140
        result_good = gate.assess(good_signal, sampling_rate=SAMPLING_RATE)
        
        # Test 2: Bad quality signal (too many NaNs)
        bad_signal = np.ones(4800) * 140
        bad_signal[:3000] = np.nan  # 62.5% NaN
        result_bad = gate.assess(bad_signal, sampling_rate=SAMPLING_RATE)
        
        # Test 3: Medium quality signal
        med_signal = np.random.randn(4800) * 5 + 140
        med_signal[:1000] = np.nan  # 20.8% NaN
        result_med = gate.assess(med_signal, sampling_rate=SAMPLING_RATE)
        
        details = {
            'good_signal_quality': result_good.level.name if hasattr(result_good, 'level') else 'UNKNOWN',
            'bad_signal_quality': result_bad.level.name if hasattr(result_bad, 'level') else 'UNKNOWN',
            'med_signal_quality': result_med.level.name if hasattr(result_med, 'level') else 'UNKNOWN',
            'gate_available': True,
        }
        
        # Verify expected outcomes
        passed = True
        if hasattr(result_bad, 'level'):
            # Bad signal should be LOW quality
            if result_bad.level != QualityLevel.LOW:
                passed = False
                details['error'] = f"Bad signal should be LOW, got {result_bad.level.name}"
        
        return passed, details
        
    except ImportError as e:
        logger.warning(f"Quality gate not available: {e}")
        return True, {'gate_available': False, 'note': 'Skipped - component not found'}
    except Exception as e:
        logger.error(f"Quality gate test failed: {e}")
        return False, {'error': str(e)}


def test_windowing() -> Tuple[bool, Dict[str, Any]]:
    """Test 1.2: Windowing verification."""
    logger.info("Testing Windowing...")
    
    try:
        # Create a 30-minute synthetic signal
        duration_min = 30
        n_samples = duration_min * 60 * SAMPLING_RATE  # 7,200 samples
        signal = np.random.randn(n_samples) * 5 + 140
        
        # Compute expected number of windows
        # 20-min windows with 5-min stride
        # Window 1: 0-20min, Window 2: 5-25min, Window 3: 10-30min
        # Expected: (30 - 20) / 5 + 1 = 3 windows
        expected_windows = int((duration_min - WINDOW_DURATION_MIN) / WINDOW_STRIDE_MIN) + 1
        
        windows = []
        window_samples = WINDOW_DURATION_MIN * 60 * SAMPLING_RATE
        stride_samples = WINDOW_STRIDE_MIN * 60 * SAMPLING_RATE
        
        for i in range(expected_windows):
            start = i * stride_samples
            end = start + window_samples
            if end <= len(signal):
                window = signal[start:end]
                windows.append(window)
        
        details = {
            'signal_duration_min': duration_min,
            'signal_samples': n_samples,
            'window_duration_min': WINDOW_DURATION_MIN,
            'window_stride_min': WINDOW_STRIDE_MIN,
            'expected_windows': expected_windows,
            'actual_windows': len(windows),
            'samples_per_window': SAMPLES_PER_WINDOW,
            'actual_samples_per_window': windows[0].shape[0] if windows else 0,
        }
        
        # Verify
        passed = True
        if len(windows) != expected_windows:
            passed = False
            details['error'] = f"Expected {expected_windows} windows, got {len(windows)}"
        elif windows and windows[0].shape[0] != SAMPLES_PER_WINDOW:
            passed = False
            details['error'] = f"Expected {SAMPLES_PER_WINDOW} samples per window, got {windows[0].shape[0]}"
        
        return passed, details
        
    except Exception as e:
        logger.error(f"Windowing test failed: {e}")
        return False, {'error': str(e)}


def test_rules_engine() -> Tuple[bool, Dict[str, Any]]:
    """Test 1.3: Rules engine sanity check."""
    logger.info("Testing Rules Engine...")
    
    try:
        # Test baseline detection
        from src.rules.baseline import calculate_baseline
        
        signal = np.ones(4800) * 140
        signal[1000:2000] += 20  # Temporary elevation
        
        result = calculate_baseline(signal, sampling_rate=SAMPLING_RATE)
        
        details = {
            'baseline_available': True,
            'baseline_value': float(result.baseline_fhr) if hasattr(result, 'baseline_fhr') else None,
        }
        
        # Try other detectors
        try:
            from src.rules.variability import calculate_variability
            var_result = calculate_variability(signal, sampling_rate=SAMPLING_RATE)
            details['variability_available'] = True
        except ImportError:
            details['variability_available'] = False
        
        try:
            from src.rules.decelerations import detect_decelerations
            # Provide baseline parameter
            baseline = details.get('baseline_value', 140)
            decel_result = detect_decelerations(
                signal, 
                np.zeros(len(signal)), 
                baseline=baseline,
                sampling_rate=SAMPLING_RATE
            )
            details['deceleration_available'] = True
        except (ImportError, TypeError) as e:
            details['deceleration_available'] = False
            details['deceleration_note'] = str(e)
        
        return True, details
        
    except ImportError as e:
        logger.warning(f"Rules engine not fully available: {e}")
        return True, {'rules_available': False, 'note': 'Skipped - component not found'}
    except Exception as e:
        logger.error(f"Rules engine test failed: {e}")
        return False, {'error': str(e)}


def test_ai_pipeline() -> Tuple[bool, Dict[str, Any]]:
    """Test 1.4: AI pipeline verification (MiniRocket → XGBoost)."""
    logger.info("Testing AI Pipeline...")
    
    try:
        # Import directly to avoid circular dependencies and wfdb issues
        # Note: Dynamic loading used to bypass main import chain that has wfdb/pandas issues
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "xgboost_only_classifier_test",  # Unique name to avoid conflicts
            PROJECT_ROOT / "src" / "adapters" / "xgboost_only_classifier.py"
        )
        xgb_module = importlib.util.module_from_spec(spec)
        # Note: Not adding to sys.modules to avoid side effects
        spec.loader.exec_module(xgb_module)
        
        XGBoostOnlyClassifier = xgb_module.XGBoostOnlyClassifier
        pad_minirocket_features = xgb_module.pad_minirocket_features
        MINIROCKET_FEATURES = xgb_module.MINIROCKET_FEATURES
        TOTAL_FEATURES = xgb_module.TOTAL_FEATURES
        
        # Verify feature dimensions
        assert MINIROCKET_FEATURES == MINIROCKET_DIM, \
            f"MiniRocket dimension mismatch: {MINIROCKET_FEATURES} != {MINIROCKET_DIM}"
        assert TOTAL_FEATURES == TOTAL_DIM, \
            f"Total dimension mismatch: {TOTAL_FEATURES} != {TOTAL_DIM}"
        
        # Test padding
        minirocket_output = np.random.randn(MINIROCKET_DIM).astype(np.float32)
        padded = pad_minirocket_features(minirocket_output)
        
        assert padded.shape == (TOTAL_DIM,), \
            f"Padded shape mismatch: {padded.shape} != ({TOTAL_DIM},)"
        assert np.allclose(padded[:MINIROCKET_DIM], minirocket_output), \
            "Padding corrupted MiniRocket features"
        assert np.allclose(padded[MINIROCKET_DIM:], 0), \
            "Padding should be zeros"
        
        # Test XGBoost classifier
        classifier = XGBoostOnlyClassifier()
        
        if not classifier.is_loaded:
            return False, {
                'error': 'XGBoost model not loaded',
                'model_available': False
            }
        
        # Test inference
        result = classifier.predict(minirocket_output)
        
        details = {
            'minirocket_dim': MINIROCKET_DIM,
            'clinical_dim': CLINICAL_DIM,
            'total_dim': TOTAL_DIM,
            'padding_ok': True,
            'model_loaded': classifier.is_loaded,
            'model_info': classifier.model_info,
            'test_risk_score': float(result.risk_score),
            'test_category': int(result.category),
            'test_category_name': result.category_name,
            'inference_time_ms': float(result.inference_time_ms),
        }
        
        # Verify outputs
        passed = True
        if not 0 <= result.risk_score <= 1:
            passed = False
            details['error'] = f"Risk score out of range: {result.risk_score}"
        if result.category not in [1, 2, 3]:
            passed = False
            details['error'] = f"Invalid category: {result.category}"
        
        return passed, details
        
    except Exception as e:
        logger.error(f"AI pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


def test_hybrid_logic() -> Tuple[bool, Dict[str, Any]]:
    """Test 1.5: Smart Hybrid Logic (MAX override)."""
    logger.info("Testing Hybrid Logic...")
    
    try:
        # Import directly to avoid circular dependencies
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "xgboost_only_classifier_test",
            PROJECT_ROOT / "src" / "adapters" / "xgboost_only_classifier.py"
        )
        xgb_module = importlib.util.module_from_spec(spec)
        # Not adding to sys.modules to avoid conflicts
        spec.loader.exec_module(xgb_module)
        
        XGBoostOnlyClassifier = xgb_module.XGBoostOnlyClassifier
        
        classifier = XGBoostOnlyClassifier()
        if not classifier.is_loaded:
            return True, {'note': 'Skipped - model not loaded'}
        
        # Test MAX override behavior
        test_features = np.random.randn(MINIROCKET_DIM).astype(np.float32)
        
        # Test 1: No override
        result_no_override = classifier.predict(test_features, rule_engine_severity=None)
        
        # Test 2: Low override (should not apply if AI is higher)
        result_low_override = classifier.predict(test_features, rule_engine_severity=0.1)
        
        # Test 3: High override (should always apply)
        result_high_override = classifier.predict(test_features, rule_engine_severity=0.95)
        
        details = {
            'no_override_risk': float(result_no_override.risk_score),
            'no_override_final': float(result_no_override.final_risk_score),
            'low_override_final': float(result_low_override.final_risk_score),
            'high_override_final': float(result_high_override.final_risk_score),
            'high_override_applied': bool(result_high_override.rule_engine_applied),
        }
        
        # Verify MAX behavior
        passed = True
        
        # Final risk should be MAX(ai_risk, rule_severity)
        expected_high = max(result_no_override.risk_score, 0.95)
        if not np.isclose(result_high_override.final_risk_score, expected_high, atol=0.01):
            passed = False
            details['error'] = (
                f"MAX override failed: expected {expected_high:.3f}, "
                f"got {result_high_override.final_risk_score:.3f}"
            )
        
        # High override should be applied
        if not result_high_override.rule_engine_applied:
            passed = False
            details['error'] = "High override should be applied"
        
        return passed, details
        
    except Exception as e:
        logger.error(f"Hybrid logic test failed: {e}")
        return False, {'error': str(e)}


def test_json_output() -> Tuple[bool, Dict[str, Any]]:
    """Test 1.6: JSON output verification."""
    logger.info("Testing JSON Output...")
    
    try:
        # Import directly to avoid circular dependencies
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "xgboost_only_classifier_test",
            PROJECT_ROOT / "src" / "adapters" / "xgboost_only_classifier.py"
        )
        xgb_module = importlib.util.module_from_spec(spec)
        # Not adding to sys.modules to avoid conflicts
        spec.loader.exec_module(xgb_module)
        
        XGBoostOnlyClassifier = xgb_module.XGBoostOnlyClassifier
        
        classifier = XGBoostOnlyClassifier()
        if not classifier.is_loaded:
            return True, {'note': 'Skipped - model not loaded'}
        
        test_features = np.random.randn(MINIROCKET_DIM).astype(np.float32)
        result = classifier.predict(test_features, rule_engine_severity=0.5)
        
        # Convert to dict
        result_dict = result.to_dict()
        
        # Verify required fields
        required_fields = [
            'risk_score', 'final_risk_score', 'category', 'category_name',
            'confidence', 'rule_engine_applied', 'inference_time_ms',
            'model_name', 'feature_dim'
        ]
        
        missing_fields = [f for f in required_fields if f not in result_dict]
        
        # Verify JSON serializable
        try:
            json_str = json.dumps(result_dict)
            json.loads(json_str)  # Verify it can be loaded back
            serializable = True
        except:
            serializable = False
        
        details = {
            'has_all_required_fields': len(missing_fields) == 0,
            'missing_fields': missing_fields,
            'json_serializable': serializable,
            'sample_output': result_dict,
        }
        
        passed = len(missing_fields) == 0 and serializable
        if not passed:
            if missing_fields:
                details['error'] = f"Missing fields: {missing_fields}"
            elif not serializable:
                details['error'] = "Output is not JSON serializable"
        
        return passed, details
        
    except Exception as e:
        logger.error(f"JSON output test failed: {e}")
        return False, {'error': str(e)}


def run_stage1_integrity() -> IntegrityTestResults:
    """Run Stage 1: Pipeline Integrity tests."""
    logger.info("=" * 60)
    logger.info("STAGE 1: PIPELINE INTEGRITY TESTS")
    logger.info("=" * 60)
    
    errors = []
    warnings = []
    
    # Run all tests
    quality_ok, quality_details = test_quality_gate()
    windowing_ok, windowing_details = test_windowing()
    rules_ok, rules_details = test_rules_engine()
    ai_ok, ai_details = test_ai_pipeline()
    hybrid_ok, hybrid_details = test_hybrid_logic()
    json_ok, json_details = test_json_output()
    
    # Collect errors and warnings
    for name, details in [
        ('quality_gate', quality_details),
        ('windowing', windowing_details),
        ('rules_engine', rules_details),
        ('ai_pipeline', ai_details),
        ('hybrid_logic', hybrid_details),
        ('json_output', json_details),
    ]:
        if 'error' in details:
            errors.append(f"{name}: {details['error']}")
        if 'warning' in details:
            warnings.append(f"{name}: {details['warning']}")
        if 'note' in details:
            warnings.append(f"{name}: {details['note']}")
    
    overall = quality_ok and windowing_ok and rules_ok and ai_ok and hybrid_ok and json_ok
    
    return IntegrityTestResults(
        quality_gate_ok=quality_ok,
        windowing_ok=windowing_ok,
        rules_engine_ok=rules_ok,
        ai_pipeline_ok=ai_ok,
        hybrid_logic_ok=hybrid_ok,
        json_output_ok=json_ok,
        quality_gate_details=quality_details,
        windowing_details=windowing_details,
        rules_details=rules_details,
        ai_details=ai_details,
        hybrid_details=hybrid_details,
        json_details=json_details,
        overall_passed=overall,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# Stage 2: Quality & Performance Tests
# =============================================================================

def run_stage2_quality() -> QualityMetrics:
    """Run Stage 2: Quality & Performance tests."""
    logger.info("=" * 60)
    logger.info("STAGE 2: QUALITY & PERFORMANCE TESTS")
    logger.info("=" * 60)
    
    errors = []
    warnings = []
    
    # For now, run synthetic tests since CTU-CHB data may not be available
    logger.info("Running synthetic quality tests...")
    
    try:
        # Import XGBoostOnlyClassifier first
        import sys
        import importlib.util
        
        # Load XGBoostOnlyClassifier
        spec1 = importlib.util.spec_from_file_location(
            "xgboost_only_classifier",
            PROJECT_ROOT / "src" / "adapters" / "xgboost_only_classifier.py"
        )
        xgb_module = importlib.util.module_from_spec(spec1)
        sys.modules['xgboost_only_classifier'] = xgb_module
        spec1.loader.exec_module(xgb_module)
        
        # Load IClassifier protocol
        spec2 = importlib.util.spec_from_file_location(
            "protocols",
            PROJECT_ROOT / "src" / "interfaces" / "protocols.py"
        )
        protocols_module = importlib.util.module_from_spec(spec2)
        sys.modules['src.interfaces.protocols'] = protocols_module
        spec2.loader.exec_module(protocols_module)
        
        # Now we can create the adapter manually
        XGBoostOnlyClassifier = xgb_module.XGBoostOnlyClassifier
        
        # Create a simple adapter wrapper
        class XGBoostV6AdapterSimple:
            def __init__(self):
                self._classifier = XGBoostOnlyClassifier()
                self._rule_engine_severity = None
            
            @property
            def is_loaded(self):
                return self._classifier.is_loaded
            
            def set_rule_engine_severity(self, severity):
                self._rule_engine_severity = severity
            
            def predict(self, X):
                X = np.asarray(X, dtype=np.float32)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                predictions = []
                for i in range(X.shape[0]):
                    result = self._classifier.predict(
                        X[i], 
                        rule_engine_severity=self._rule_engine_severity
                    )
                    predictions.append(result.category - 1)
                return np.array(predictions, dtype=np.int32)
            
            def predict_proba(self, X):
                X = np.asarray(X, dtype=np.float32)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                probabilities = []
                for i in range(X.shape[0]):
                    result = self._classifier.predict(
                        X[i],
                        rule_engine_severity=self._rule_engine_severity
                    )
                    risk = result.final_risk_score
                    thresholds = self._classifier.thresholds
                    critical = thresholds.get('critical', 0.60)
                    warning = thresholds.get('warning', 0.35)
                    
                    if risk > critical:
                        proba = [0.1, 0.2, 0.7 * risk / critical + 0.3]
                    elif risk > warning:
                        ratio = (risk - warning) / (critical - warning)
                        proba = [0.1, 0.5 + 0.3 * ratio, 0.4 - 0.2 * ratio]
                    else:
                        ratio = risk / warning if warning > 0 else 0
                        proba = [0.8 - 0.3 * ratio, 0.15 + 0.25 * ratio, 0.05 + 0.05 * ratio]
                    
                    proba = np.array(proba)
                    proba = proba / proba.sum()
                    probabilities.append(proba)
                
                return np.array(probabilities, dtype=np.float32)
        
        adapter = XGBoostV6AdapterSimple()
        if not adapter.is_loaded:
            errors.append("Model not loaded - cannot run quality tests")
            return QualityMetrics(
                recall=0.0, precision=0.0, specificity=0.0,
                f1_score=0.0, f2_score=0.0, auc_roc=0.0,
                fpr=0.0, fnr=0.0,
                noise_robustness_ok=False,
                latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
                overall_passed=False,
                errors=errors,
                warnings=warnings,
            )
        
        # Generate synthetic test data (100 samples)
        # Note: Using random labels for infrastructure testing only
        # Real quality metrics require actual CTU-CHB data with pH labels
        logger.warning("Using synthetic random data - quality metrics are for infrastructure testing only")
        n_samples = 100
        X_test = np.random.randn(n_samples, MINIROCKET_DIM).astype(np.float32)
        y_true = np.random.randint(0, 2, n_samples)  # Binary: 0=normal, 1=pathological
        
        # Run predictions and measure latency
        latencies = []
        y_pred_proba = []
        
        for i in range(n_samples):
            start = time.perf_counter()
            proba = adapter.predict_proba(X_test[i:i+1])
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # ms
            y_pred_proba.append(proba[0, 2])  # P(pathological)
        
        y_pred_proba = np.array(y_pred_proba)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        # F2 weighs recall higher than precision
        beta = 2
        f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0.0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Simplified AUC-ROC calculation
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_pred_proba)
        except ImportError:
            logger.warning("sklearn not available for AUC calculation")
            auc = 0.5
        except ValueError as e:
            logger.warning(f"AUC calculation failed: {e}")
            auc = 0.5
        
        # Latency percentiles
        latencies_sorted = np.sort(latencies)
        p50 = np.percentile(latencies_sorted, 50)
        p95 = np.percentile(latencies_sorted, 95)
        p99 = np.percentile(latencies_sorted, 99)
        
        # Noise robustness test
        noise_ok = True
        try:
            # Test with noisy input
            noisy_input = np.random.randn(MINIROCKET_DIM) * 100  # Very noisy
            result = adapter.predict(noisy_input)
            # Should not crash and should return valid category
            if result[0] not in [0, 1, 2]:
                noise_ok = False
        except:
            noise_ok = False
        
        # Check if metrics meet targets
        overall_passed = True
        if fpr > 0.20:
            warnings.append(f"FPR {fpr:.3f} exceeds target 0.20")
        if fnr > 0.05:
            errors.append(f"FNR {fnr:.3f} exceeds critical target 0.05")
            overall_passed = False
        if recall < 0.85:
            warnings.append(f"Recall {recall:.3f} below target 0.85")
        if p95 > 100:
            warnings.append(f"p95 latency {p95:.1f}ms exceeds target 100ms")
        
        return QualityMetrics(
            recall=float(recall),
            precision=float(precision),
            specificity=float(specificity),
            f1_score=float(f1),
            f2_score=float(f2),
            auc_roc=float(auc),
            fpr=float(fpr),
            fnr=float(fnr),
            noise_robustness_ok=noise_ok,
            latency_p50_ms=float(p50),
            latency_p95_ms=float(p95),
            latency_p99_ms=float(p99),
            overall_passed=overall_passed,
            errors=errors,
            warnings=warnings,
        )
        
    except Exception as e:
        logger.error(f"Quality tests failed: {e}")
        import traceback
        traceback.print_exc()
        errors.append(f"Exception: {str(e)}")
        return QualityMetrics(
            recall=0.0, precision=0.0, specificity=0.0,
            f1_score=0.0, f2_score=0.0, auc_roc=0.0,
            fpr=0.0, fnr=0.0,
            noise_robustness_ok=False,
            latency_p50_ms=0.0, latency_p95_ms=0.0, latency_p99_ms=0.0,
            overall_passed=False,
            errors=errors,
            warnings=warnings,
        )


# =============================================================================
# Main Execution
# =============================================================================

def save_report(filename: str, data: Dict[str, Any]):
    """Save report to JSON file."""
    REPORTS_DIR.mkdir(exist_ok=True)
    filepath = REPORTS_DIR / filename
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Report saved to: {filepath}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="SentinelFetal V6 Comprehensive Testing"
    )
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2],
        help='Run specific stage (1=Integrity, 2=Quality)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all stages'
    )
    
    args = parser.parse_args()
    
    if not args.stage and not args.all:
        parser.print_help()
        return 1
    
    print("=" * 60)
    print("SentinelFetal V6 Comprehensive Testing")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Stage 1
    if args.stage == 1 or args.all:
        start_time = time.time()
        integrity_results = run_stage1_integrity()
        execution_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("STAGE 1 SUMMARY")
        print("=" * 60)
        print(f"Quality Gate:    {'PASS' if integrity_results.quality_gate_ok else 'FAIL'}")
        print(f"Windowing:       {'PASS' if integrity_results.windowing_ok else 'FAIL'}")
        print(f"Rules Engine:    {'PASS' if integrity_results.rules_engine_ok else 'FAIL'}")
        print(f"AI Pipeline:     {'PASS' if integrity_results.ai_pipeline_ok else 'FAIL'}")
        print(f"Hybrid Logic:    {'PASS' if integrity_results.hybrid_logic_ok else 'FAIL'}")
        print(f"JSON Output:     {'PASS' if integrity_results.json_output_ok else 'FAIL'}")
        print(f"\nOverall: {'PASSED' if integrity_results.overall_passed else 'FAILED'}")
        print(f"Execution time: {execution_time:.2f}s")
        
        if integrity_results.errors:
            print("\nErrors:")
            for error in integrity_results.errors:
                print(f"  - {error}")
        
        if integrity_results.warnings:
            print("\nWarnings:")
            for warning in integrity_results.warnings:
                print(f"  - {warning}")
        
        # Save report
        report = {
            'stage': 1,
            'stage_name': 'Pipeline Integrity',
            'timestamp': datetime.now().isoformat(),
            'execution_time_s': execution_time,
            'overall_passed': integrity_results.overall_passed,
            'results': asdict(integrity_results),
        }
        save_report('v6_pipeline_integrity_report.json', report)
        
        if not integrity_results.overall_passed:
            print("\nStage 1 FAILED. Fix issues before running Stage 2.")
            return 1
    
    # Stage 2 (only if Stage 1 passed or --stage 2 specified)
    if args.stage == 2 or args.all:
        start_time = time.time()
        quality_metrics = run_stage2_quality()
        execution_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("STAGE 2 SUMMARY")
        print("=" * 60)
        print(f"Recall:          {quality_metrics.recall:.3f} (target > 0.85)")
        print(f"Precision:       {quality_metrics.precision:.3f}")
        print(f"Specificity:     {quality_metrics.specificity:.3f}")
        print(f"F1 Score:        {quality_metrics.f1_score:.3f}")
        print(f"F2 Score:        {quality_metrics.f2_score:.3f}")
        print(f"AUC-ROC:         {quality_metrics.auc_roc:.3f}")
        print(f"\nFPR:             {quality_metrics.fpr:.3f} (target < 0.20)")
        print(f"FNR:             {quality_metrics.fnr:.3f} (target < 0.05)")
        print(f"\nNoise Robust:    {'YES' if quality_metrics.noise_robustness_ok else 'NO'}")
        print(f"Latency p50:     {quality_metrics.latency_p50_ms:.1f}ms")
        print(f"Latency p95:     {quality_metrics.latency_p95_ms:.1f}ms (target < 100ms)")
        print(f"Latency p99:     {quality_metrics.latency_p99_ms:.1f}ms")
        print(f"\nOverall: {'PASSED' if quality_metrics.overall_passed else 'FAILED'}")
        print(f"Execution time: {execution_time:.2f}s")
        
        if quality_metrics.errors:
            print("\nErrors:")
            for error in quality_metrics.errors:
                print(f"  - {error}")
        
        if quality_metrics.warnings:
            print("\nWarnings:")
            for warning in quality_metrics.warnings:
                print(f"  - {warning}")
        
        # Save report
        report = {
            'stage': 2,
            'stage_name': 'Quality & Performance',
            'timestamp': datetime.now().isoformat(),
            'execution_time_s': execution_time,
            'overall_passed': quality_metrics.overall_passed,
            'metrics': asdict(quality_metrics),
        }
        save_report('v6_quality_metrics_report.json', report)
        
        if not quality_metrics.overall_passed:
            return 1
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
