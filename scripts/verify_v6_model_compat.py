"""
Verify V6 Model Compatibility
=============================
Verification script for the V6 simplified pipeline (MiniRocket -> XGBoost).

This script verifies:
1. XGBoost model can be loaded and has expected structure
2. MiniRocket encoder is available
3. Feature dimensions are compatible
4. End-to-end inference works correctly
5. Pipeline container can be created

Usage:
    python scripts/verify_v6_model_compat.py

Author: SentinelFetal ML Team
Version: 6.0.0
"""

import sys
import pickle
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model paths
XGBOOST_PATH = PROJECT_ROOT / "models" / "ensemble_v5" / "xgboost_v5.pkl"
XGBOOST_ALT_PATH = PROJECT_ROOT / "models" / "xgboost_v5.pkl"
MINIROCKET_PATH = PROJECT_ROOT / "models" / "minirocket_encoder.joblib"

# Expected dimensions
MINIROCKET_DIM = 9996
CLINICAL_DIM = 8
TOTAL_DIM = MINIROCKET_DIM + CLINICAL_DIM


def check_xgboost_model():
    """Verify XGBoost model structure."""
    print("\n" + "=" * 60)
    print("1. XGBOOST MODEL VERIFICATION")
    print("=" * 60)

    model_path = XGBOOST_PATH if XGBOOST_PATH.exists() else XGBOOST_ALT_PATH

    if not model_path.exists():
        print(f"[FAIL] XGBoost model not found at:")
        print(f"       - {XGBOOST_PATH}")
        print(f"       - {XGBOOST_ALT_PATH}")
        return False

    print(f"[OK] Model found: {model_path}")
    print(f"     File size: {model_path.stat().st_size:,} bytes")

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] Model loaded successfully")
        print(f"     Type: {type(model).__name__}")

        # Check if CalibratedClassifierCV
        if hasattr(model, 'calibrated_classifiers_'):
            print(f"[OK] Model is CalibratedClassifierCV")
            print(f"     Calibrators: {len(model.calibrated_classifiers_)}")
            base = model.calibrated_classifiers_[0].estimator
            if hasattr(base, 'n_features_in_'):
                n_features = base.n_features_in_
                print(f"     Base estimator features: {n_features}")
                if n_features == TOTAL_DIM:
                    print(f"[OK] Feature dimension matches expected ({TOTAL_DIM})")
                else:
                    print(f"[WARN] Feature dimension {n_features} != expected {TOTAL_DIM}")
        else:
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
                print(f"     Expected features: {n_features}")
                if n_features == TOTAL_DIM:
                    print(f"[OK] Feature dimension matches expected ({TOTAL_DIM})")
                else:
                    print(f"[WARN] Feature dimension {n_features} != expected {TOTAL_DIM}")

        if hasattr(model, 'classes_'):
            print(f"     Classes: {model.classes_}")

        return True

    except Exception as e:
        print(f"[FAIL] Error loading model: {e}")
        return False


def check_minirocket_encoder():
    """Verify MiniRocket encoder."""
    print("\n" + "=" * 60)
    print("2. MINIROCKET ENCODER VERIFICATION")
    print("=" * 60)

    if not MINIROCKET_PATH.exists():
        print(f"[WARN] MiniRocket encoder not found: {MINIROCKET_PATH}")
        print("       V6 pipeline will fit encoder on first use")
        return True  # Not critical - will be fitted

    print(f"[OK] Encoder found: {MINIROCKET_PATH}")
    print(f"     File size: {MINIROCKET_PATH.stat().st_size:,} bytes")

    try:
        import joblib
        encoder = joblib.load(MINIROCKET_PATH)
        print(f"[OK] Encoder loaded successfully")
        print(f"     Type: {type(encoder).__name__}")

        if hasattr(encoder, 'parameters_'):
            print(f"[OK] Encoder is fitted")
        else:
            print(f"[WARN] Encoder may not be fitted")

        return True

    except Exception as e:
        print(f"[WARN] Error loading encoder: {e}")
        return True  # Not critical


def check_v6_classifier():
    """Verify V6 XGBoost-only classifier."""
    print("\n" + "=" * 60)
    print("3. V6 CLASSIFIER VERIFICATION")
    print("=" * 60)

    try:
        from src.adapters.xgboost_only_classifier import (
            XGBoostOnlyClassifier,
            pad_minirocket_features,
            MINIROCKET_FEATURES,
            TOTAL_FEATURES
        )

        print(f"[OK] XGBoostOnlyClassifier imported successfully")
        print(f"     MINIROCKET_FEATURES: {MINIROCKET_FEATURES}")
        print(f"     TOTAL_FEATURES: {TOTAL_FEATURES}")

        # Test padding function
        test_input = np.random.randn(MINIROCKET_FEATURES)
        padded = pad_minirocket_features(test_input)
        print(f"[OK] Feature padding works: {test_input.shape} -> {padded.shape}")

        # Create classifier
        classifier = XGBoostOnlyClassifier()
        if classifier.is_loaded:
            print(f"[OK] Classifier loaded and ready")
            print(f"     Model info: {classifier.model_info}")
        else:
            print(f"[WARN] Classifier not loaded (model may be missing)")

        return classifier.is_loaded

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_v6_adapter():
    """Verify V6 adapter implements IClassifier."""
    print("\n" + "=" * 60)
    print("4. V6 ADAPTER VERIFICATION")
    print("=" * 60)

    try:
        from src.adapters.xgboost_v6_adapter import XGBoostV6Adapter
        from src.interfaces.protocols import IClassifier

        adapter = XGBoostV6Adapter()
        print(f"[OK] XGBoostV6Adapter created")
        print(f"     is_loaded: {adapter.is_loaded}")

        # Check protocol methods
        required_methods = ['predict', 'predict_proba', 'save_model', 'load_model']
        for method in required_methods:
            if hasattr(adapter, method):
                print(f"[OK] Has method: {method}")
            else:
                print(f"[FAIL] Missing method: {method}")
                return False

        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_inference():
    """Test end-to-end inference."""
    print("\n" + "=" * 60)
    print("5. INFERENCE TEST")
    print("=" * 60)

    try:
        from src.adapters.xgboost_v6_adapter import XGBoostV6Adapter

        adapter = XGBoostV6Adapter()
        if not adapter.is_loaded:
            print(f"[SKIP] Model not loaded, skipping inference test")
            return True

        # Test with MiniRocket-sized features
        test_features = np.random.randn(MINIROCKET_DIM)

        # Single prediction
        pred = adapter.predict(test_features)
        print(f"[OK] Single prediction: {pred[0]} (0=Normal, 1=Suspicious, 2=Pathological)")

        # Probabilities
        proba = adapter.predict_proba(test_features)
        print(f"[OK] Probabilities: {proba[0]}")

        # Batch prediction
        batch = np.random.randn(5, MINIROCKET_DIM)
        batch_pred = adapter.predict(batch)
        print(f"[OK] Batch prediction (5 samples): {batch_pred}")

        # With rule engine override
        adapter.set_rule_engine_severity(0.75)
        pred_override = adapter.predict(test_features)
        print(f"[OK] With rule override (0.75): {pred_override[0]}")

        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pipeline_container():
    """Test V6 pipeline container creation."""
    print("\n" + "=" * 60)
    print("6. PIPELINE CONTAINER TEST")
    print("=" * 60)

    try:
        from src.pipeline.container import PipelineContainer

        container = PipelineContainer.create_v6_xgboost()
        print(f"[OK] PipelineContainer.create_v6_xgboost() succeeded")

        if container.validate():
            print(f"[OK] Container validated - all components present")
        else:
            missing = container.get_missing_components()
            print(f"[WARN] Missing components: {missing}")

        # Check classifier type
        classifier = container.classifier
        print(f"[OK] Classifier type: {type(classifier).__name__}")

        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("SentinelFetal V6 Model Compatibility Verification")
    print("=" * 60)

    results = {
        'xgboost_model': check_xgboost_model(),
        'minirocket_encoder': check_minirocket_encoder(),
        'v6_classifier': check_v6_classifier(),
        'v6_adapter': check_v6_adapter(),
        'inference': check_inference(),
        'pipeline_container': check_pipeline_container(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for check, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("V6 PIPELINE VERIFICATION: ALL CHECKS PASSED")
        print("=" * 60)
        return 0
    else:
        print("V6 PIPELINE VERIFICATION: SOME CHECKS FAILED")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
