#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 Verification Script.

This script verifies that Stage 4 components (calibration and persistence)
are working correctly.

Usage:
    python src/calibration/verify_stage4.py
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.calibration.calibrator import ThresholdCalibrator, load_thresholds
from src.analysis.persistence import PersistenceManager
from src.config import load_dynamic_thresholds

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_calibrator():
    """Test the ThresholdCalibrator."""
    logger.info("Testing ThresholdCalibrator...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_neg = 1000
    n_pos = 200
    
    # Negative samples: low scores
    neg_scores = np.random.beta(2, 5, n_neg)  # Skewed toward 0
    
    # Positive samples: high scores
    pos_scores = np.random.beta(5, 2, n_pos)  # Skewed toward 1
    
    scores = np.concatenate([neg_scores, pos_scores])
    labels = np.array([0] * n_neg + [1] * n_pos)
    
    # Calibrate
    calibrator = ThresholdCalibrator(
        t_low_percentile=95.0,
        t_high_percentile=99.0
    )
    thresholds = calibrator.calibrate(scores, labels)
    
    # Validate
    assert 0 <= thresholds['t_low'] <= 1, "t_low out of range"
    assert 0 <= thresholds['t_high'] <= 1, "t_high out of range"
    assert thresholds['t_low'] < thresholds['t_high'], "t_low >= t_high"
    
    logger.info(f"  ✓ Thresholds: t_low={thresholds['t_low']:.4f}, t_high={thresholds['t_high']:.4f}")
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / 'test_thresholds.yaml'
        calibrator.save_thresholds(yaml_path)
        
        loaded = load_thresholds(yaml_path)
        assert loaded['thresholds']['t_low'] == thresholds['t_low']
        assert loaded['thresholds']['t_high'] == thresholds['t_high']
    
    logger.info("  ✓ Save/Load works correctly")
    return True


def test_persistence():
    """Test the PersistenceManager."""
    logger.info("Testing PersistenceManager (K=2, N=3)...")
    
    manager = PersistenceManager(K=2, N=3)
    
    # Test Case 1: Gradual buildup
    assert manager.update("test1", False) == False  # 0/3
    assert manager.update("test1", True) == False   # 1/3
    assert manager.update("test1", True) == True    # 2/3 ✓
    logger.info("  ✓ Test 1: Gradual buildup triggers correctly")
    
    # Test Case 2: Persistence after trigger
    assert manager.update("test1", False) == True   # 2/3 still ✓
    assert manager.update("test1", False) == False  # 1/3 drops
    logger.info("  ✓ Test 2: Persistence maintains then drops correctly")
    
    # Test Case 3: Multiple monitors
    assert manager.update("test2", True) == False   # 1/3
    assert manager.update("test2", True) == True    # 2/3 ✓
    assert manager.get_state("test1").alert_active == False
    assert manager.get_state("test2").alert_active == True
    logger.info("  ✓ Test 3: Multiple monitors tracked independently")
    
    # Test Case 4: Reset
    manager.reset("test2")
    assert manager.get_state("test2") is None
    logger.info("  ✓ Test 4: Reset works correctly")
    
    return True


def test_config_loading():
    """Test dynamic threshold loading from config."""
    logger.info("Testing config threshold loading...")
    
    # Test loading non-existent file (should use defaults)
    thresholds = load_dynamic_thresholds("nonexistent.yaml")
    assert thresholds.t_low == 0.45  # Default
    assert thresholds.t_high == 0.72  # Default
    logger.info("  ✓ Fallback to defaults works")
    
    # Test loading from a real file
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / 'test_thresholds.yaml'
        
        # Create test calibrator and save
        calibrator = ThresholdCalibrator(
            t_low_percentile=90.0,
            t_high_percentile=95.0,
            persistence_k=3,
            persistence_n=5
        )
        
        np.random.seed(123)
        scores = np.random.random(100)
        labels = np.zeros(100)
        
        calibrator.calibrate(scores, labels)
        calibrator.save_thresholds(yaml_path)
        
        # Load
        loaded = load_dynamic_thresholds(str(yaml_path))
        assert loaded.K == 3
        assert loaded.N == 5
        assert 0 <= loaded.t_low < loaded.t_high <= 1
    
    logger.info("  ✓ Loading from YAML works correctly")
    return True


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("STAGE 4 VERIFICATION")
    print("="*60 + "\n")
    
    tests = [
        ("Calibrator", test_calibrator),
        ("Persistence", test_persistence),
        ("Config Loading", test_config_loading)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            logger.error(f"✗ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            logger.error(f"✗ {name} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if failed > 0:
        print(f"⚠  {failed} tests failed")
        print("="*60 + "\n")
        sys.exit(1)
    else:
        print("✓ All tests passed!")
        print("="*60 + "\n")
        sys.exit(0)


if __name__ == '__main__':
    main()
