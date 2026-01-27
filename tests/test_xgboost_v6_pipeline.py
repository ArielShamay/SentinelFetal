"""
SentinelFetal V6.0 — XGBoost-Only Pipeline Tests
=================================================
Verifies the V6 simplified pipeline (MiniRocket -> XGBoost).

Tests:
1. XGBoostOnlyClassifier loads model correctly
2. Feature padding works for MiniRocket output
3. Prediction returns correct structure
4. Rule Engine safety override works
5. XGBoostV6Adapter implements IClassifier protocol
6. Pipeline container supports create_v6_xgboost()

Author: SentinelFetal ML Team
Version: 6.0.0
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Feature dimensions
MINIROCKET_DIM = 9996
CLINICAL_DIM = 8
TOTAL_DIM = MINIROCKET_DIM + CLINICAL_DIM


def test_feature_padding():
    """Test 1: Feature padding function works correctly."""
    print("\n--- Test 1: Feature Padding ---")

    from src.adapters.xgboost_only_classifier import (
        pad_minirocket_features,
        MINIROCKET_FEATURES,
        TOTAL_FEATURES
    )

    assert MINIROCKET_FEATURES == MINIROCKET_DIM, f"Expected {MINIROCKET_DIM}, got {MINIROCKET_FEATURES}"
    assert TOTAL_FEATURES == TOTAL_DIM, f"Expected {TOTAL_DIM}, got {TOTAL_FEATURES}"

    # Test 1D padding
    input_1d = np.random.randn(MINIROCKET_DIM)
    output_1d = pad_minirocket_features(input_1d)
    assert output_1d.shape == (TOTAL_DIM,), f"Expected ({TOTAL_DIM},), got {output_1d.shape}"
    assert np.allclose(output_1d[:MINIROCKET_DIM], input_1d), "First 9996 features should match"
    assert np.allclose(output_1d[MINIROCKET_DIM:], 0), "Padding should be zeros"
    print(f"  [PASS] 1D padding: {input_1d.shape} -> {output_1d.shape}")

    # Test 2D padding (batch)
    input_2d = np.random.randn(5, MINIROCKET_DIM)
    output_2d = pad_minirocket_features(input_2d)
    assert output_2d.shape == (5, TOTAL_DIM), f"Expected (5, {TOTAL_DIM}), got {output_2d.shape}"
    print(f"  [PASS] 2D padding: {input_2d.shape} -> {output_2d.shape}")

    # Test already padded input
    already_padded = np.random.randn(TOTAL_DIM)
    output_padded = pad_minirocket_features(already_padded)
    assert output_padded.shape == (TOTAL_DIM,), "Already padded input should pass through"
    print(f"  [PASS] Already padded input passes through")


def test_xgboost_classifier_loads():
    """Test 2: XGBoostOnlyClassifier loads model."""
    print("\n--- Test 2: XGBoostOnlyClassifier Loading ---")

    from src.adapters.xgboost_only_classifier import (
        XGBoostOnlyClassifier,
        get_xgboost_classifier,
        reset_xgboost_classifier
    )

    # Reset to ensure fresh load
    reset_xgboost_classifier()

    classifier = get_xgboost_classifier()

    # Check model info
    info = classifier.get_model_info()
    print(f"  Model info: {info}")

    if classifier.is_loaded:
        print(f"  [PASS] Model loaded successfully")
        print(f"  [PASS] Model path: {classifier.model_path}")
        assert 'version' in info, "Should have version info"
    else:
        print(f"  [SKIP] Model not found (will use defaults)")

    return classifier.is_loaded


def test_xgboost_prediction_structure():
    """Test 3: XGBoost prediction returns correct structure."""
    print("\n--- Test 3: Prediction Output Structure ---")

    from src.adapters.xgboost_only_classifier import (
        XGBoostOnlyClassifier,
        XGBoostPrediction
    )

    classifier = XGBoostOnlyClassifier()

    # Generate test features (MiniRocket size)
    test_features = np.random.randn(MINIROCKET_DIM)

    result = classifier.predict(test_features)

    # Verify it's the right type
    assert isinstance(result, XGBoostPrediction), f"Expected XGBoostPrediction, got {type(result)}"

    # Verify structure
    assert hasattr(result, 'risk_score'), "Missing risk_score"
    assert hasattr(result, 'category'), "Missing category"
    assert hasattr(result, 'category_name'), "Missing category_name"
    assert hasattr(result, 'confidence'), "Missing confidence"
    assert hasattr(result, 'final_risk_score'), "Missing final_risk_score"
    assert hasattr(result, 'rule_engine_applied'), "Missing rule_engine_applied"
    assert hasattr(result, 'inference_time_ms'), "Missing inference_time_ms"

    # Verify values
    assert 0 <= result.risk_score <= 1, f"Risk score out of bounds: {result.risk_score}"
    assert result.category in [1, 2, 3], f"Invalid category: {result.category}"
    assert result.category_name in ['Normal', 'Suspicious', 'Pathological']
    assert 0 <= result.confidence <= 1, f"Confidence out of bounds: {result.confidence}"

    print(f"  [PASS] Risk Score: {result.risk_score:.3f}")
    print(f"  [PASS] Category: {result.category} ({result.category_name})")
    print(f"  [PASS] Confidence: {result.confidence:.3f}")
    print(f"  [PASS] Inference time: {result.inference_time_ms:.2f}ms")

    # Test to_dict()
    result_dict = result.to_dict()
    assert 'risk_score' in result_dict
    assert 'category' in result_dict
    assert 'model_probabilities' in result_dict  # Compatibility with EnsemblePrediction
    print(f"  [PASS] to_dict() works correctly")


def test_rule_engine_override():
    """Test 4: Rule Engine safety override works."""
    print("\n--- Test 4: Rule Engine Safety Override ---")

    from src.adapters.xgboost_only_classifier import XGBoostOnlyClassifier

    classifier = XGBoostOnlyClassifier()

    # Test with low risk score and high rule engine severity
    test_features = np.random.randn(MINIROCKET_DIM)

    # First, predict without rule engine
    result_no_override = classifier.predict(test_features, rule_engine_severity=None)
    print(f"  Without override: risk={result_no_override.risk_score:.3f}, final={result_no_override.final_risk_score:.3f}")

    # Now with high rule engine severity
    result_with_override = classifier.predict(test_features, rule_engine_severity=0.95)

    # Final risk should be MAX(ai_risk, rule_severity)
    assert result_with_override.final_risk_score >= 0.95, \
        f"Final risk should be >= 0.95, got {result_with_override.final_risk_score}"
    assert result_with_override.rule_engine_applied == True, "Rule engine should be applied"
    assert result_with_override.rule_engine_severity == 0.95

    print(f"  With override (0.95): final={result_with_override.final_risk_score:.3f}")
    print(f"  [PASS] Rule engine override applied correctly")

    # Test that low severity doesn't override higher AI risk
    result_low_override = classifier.predict(test_features, rule_engine_severity=0.01)
    # If AI risk is higher than 0.01, override should NOT apply
    if result_no_override.risk_score > 0.01:
        assert result_low_override.rule_engine_applied == False, \
            "Low severity should not override higher AI risk"
        print(f"  [PASS] Low severity doesn't override higher AI risk")


def test_v6_adapter_protocol():
    """Test 5: XGBoostV6Adapter implements IClassifier protocol."""
    print("\n--- Test 5: V6 Adapter Protocol ---")

    from src.adapters.xgboost_v6_adapter import XGBoostV6Adapter

    adapter = XGBoostV6Adapter()

    # Check required methods exist
    required_methods = ['predict', 'predict_proba', 'save_model', 'load_model']
    for method in required_methods:
        assert hasattr(adapter, method), f"Missing method: {method}"
        print(f"  [PASS] Has method: {method}")

    # Test predict() returns numpy array
    test_features = np.random.randn(MINIROCKET_DIM)
    predictions = adapter.predict(test_features)
    assert isinstance(predictions, np.ndarray), "predict() should return numpy array"
    assert predictions.dtype in [np.int32, np.int64], f"predict() should return int array, got {predictions.dtype}"
    assert predictions[0] in [0, 1, 2], f"Predictions should be 0, 1, or 2, got {predictions[0]}"
    print(f"  [PASS] predict() returns: {predictions}")

    # Test predict_proba() returns numpy array with correct shape
    proba = adapter.predict_proba(test_features)
    assert isinstance(proba, np.ndarray), "predict_proba() should return numpy array"
    assert proba.shape == (1, 3), f"predict_proba() should return (1, 3), got {proba.shape}"
    assert np.isclose(proba.sum(), 1.0, atol=0.01), f"Probabilities should sum to 1, got {proba.sum()}"
    print(f"  [PASS] predict_proba() returns: {proba}")

    # Test batch prediction
    batch_features = np.random.randn(5, MINIROCKET_DIM)
    batch_pred = adapter.predict(batch_features)
    assert batch_pred.shape == (5,), f"Batch prediction should be (5,), got {batch_pred.shape}"
    print(f"  [PASS] Batch prediction shape: {batch_pred.shape}")

    batch_proba = adapter.predict_proba(batch_features)
    assert batch_proba.shape == (5, 3), f"Batch proba should be (5, 3), got {batch_proba.shape}"
    print(f"  [PASS] Batch proba shape: {batch_proba.shape}")


def test_adapter_rule_engine():
    """Test 6: V6 Adapter rule engine integration."""
    print("\n--- Test 6: V6 Adapter Rule Engine ---")

    from src.adapters.xgboost_v6_adapter import XGBoostV6Adapter

    adapter = XGBoostV6Adapter()
    test_features = np.random.randn(MINIROCKET_DIM)

    # Without rule engine
    pred_normal = adapter.predict(test_features)

    # With high severity
    adapter.set_rule_engine_severity(0.85)
    pred_override = adapter.predict(test_features)

    print(f"  Without override: {pred_normal[0]}")
    print(f"  With override (0.85): {pred_override[0]}")

    # High severity should result in category 2 or 3 (Suspicious or Pathological)
    assert pred_override[0] >= 1, "High severity should not result in Normal category"
    print(f"  [PASS] Rule engine override affects predictions")


def test_pipeline_container_v6():
    """Test 7: Pipeline container supports create_v6_xgboost()."""
    print("\n--- Test 7: Pipeline Container V6 ---")

    from src.pipeline.container import PipelineContainer

    # Test factory method exists
    assert hasattr(PipelineContainer, 'create_v6_xgboost'), "Missing create_v6_xgboost method"
    print(f"  [PASS] create_v6_xgboost() method exists")

    # Create container
    container = PipelineContainer.create_v6_xgboost()
    print(f"  [PASS] Container created successfully")

    # Validate all components present
    if container.validate():
        print(f"  [PASS] All components present")
    else:
        missing = container.get_missing_components()
        print(f"  [WARN] Missing components: {missing}")

    # Check classifier type
    from src.adapters.xgboost_v6_adapter import XGBoostV6Adapter
    assert isinstance(container.classifier, XGBoostV6Adapter), \
        f"Classifier should be XGBoostV6Adapter, got {type(container.classifier)}"
    print(f"  [PASS] Classifier is XGBoostV6Adapter")


def test_adapter_get_model_info():
    """Test 8: V6 Adapter get_model_info()."""
    print("\n--- Test 8: Adapter Model Info ---")

    from src.adapters.xgboost_v6_adapter import XGBoostV6Adapter

    adapter = XGBoostV6Adapter()
    info = adapter.get_model_info()

    assert 'is_loaded' in info, "Should have is_loaded"
    assert 'adapter' in info, "Should have adapter info"
    assert info['adapter'] == 'XGBoostV6Adapter'
    assert 'expected_features' in info, "Should have expected_features"
    assert info['expected_features'] == TOTAL_DIM

    print(f"  Model info: {info}")
    print(f"  [PASS] get_model_info() returns expected structure")


def run_all_tests():
    """Run all tests and summarize results."""
    print("=" * 60)
    print("SentinelFetal V6 XGBoost-Only Pipeline Tests")
    print("=" * 60)

    tests = [
        ("Feature Padding", test_feature_padding),
        ("XGBoost Classifier Loading", test_xgboost_classifier_loads),
        ("Prediction Structure", test_xgboost_prediction_structure),
        ("Rule Engine Override", test_rule_engine_override),
        ("V6 Adapter Protocol", test_v6_adapter_protocol),
        ("Adapter Rule Engine", test_adapter_rule_engine),
        ("Pipeline Container V6", test_pipeline_container_v6),
        ("Adapter Model Info", test_adapter_get_model_info),
    ]

    results = {}
    for name, test_func in tests:
        try:
            test_func()
            results[name] = True
        except Exception as e:
            results[name] = False
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"{status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All V6 pipeline tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
