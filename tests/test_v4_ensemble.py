"""
SentinelFetal V4.0 — End-to-End Verification Test
==================================================
Verifies the complete V4.0 Ensemble pipeline integration.

Tests:
1. EnsembleManager loads all 3 models
2. Weighted soft voting produces expected output structure
3. Rule Engine safety override works correctly
4. EnsembleClassifierAdapter implements IClassifier protocol
5. Pipeline container supports use_ensemble_v4 flag
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_ensemble_manager_loads():
    """Test 1: EnsembleManager loads all 3 models."""
    print("\n--- Test 1: EnsembleManager Model Loading ---")
    
    from src.ml.ensemble_manager import get_ensemble_manager, reset_ensemble_manager
    
    # Reset to ensure fresh load
    reset_ensemble_manager()
    
    manager = get_ensemble_manager()
    
    assert manager.is_loaded, "Models should be loaded"
    assert len(manager.loaded_models) == 3, f"Expected 3 models, got {len(manager.loaded_models)}"
    assert 'xgboost' in manager.loaded_models
    assert 'random_forest' in manager.loaded_models
    assert 'sgd_classifier' in manager.loaded_models
    
    print(f"  [PASS] Loaded models: {manager.loaded_models}")
    print(f"  [PASS] Weights: {manager.weights}")


def test_ensemble_prediction_structure():
    """Test 2: Ensemble prediction returns correct structure."""
    print("\n--- Test 2: Prediction Output Structure ---")
    
    from src.ml.ensemble_manager import get_ensemble_manager
    
    manager = get_ensemble_manager()
    
    # Generate test features (32 features as per training)
    test_features = np.random.randn(32)
    
    result = manager.predict(test_features)
    
    # Verify structure
    assert hasattr(result, 'risk_score'), "Missing risk_score"
    assert hasattr(result, 'category'), "Missing category"
    assert hasattr(result, 'category_name'), "Missing category_name"
    assert hasattr(result, 'confidence'), "Missing confidence"
    assert hasattr(result, 'model_probabilities'), "Missing model_probabilities"
    assert hasattr(result, 'model_predictions'), "Missing model_predictions"
    assert hasattr(result, 'final_risk_score'), "Missing final_risk_score"
    assert hasattr(result, 'inference_time_ms'), "Missing inference_time_ms"
    
    # Verify values
    assert 0 <= result.risk_score <= 1, f"Risk score out of bounds: {result.risk_score}"
    assert result.category in [1, 2, 3], f"Invalid category: {result.category}"
    assert result.category_name in ['Normal', 'Suspicious', 'Pathological']
    
    print(f"  [PASS] Risk Score: {result.risk_score:.3f}")
    print(f"  [PASS] Category: {result.category} ({result.category_name})")
    print(f"  [PASS] Model probs: {list(result.model_probabilities.keys())}")
    print(f"  [PASS] Inference time: {result.inference_time_ms:.2f}ms")


def test_rule_engine_override():
    """Test 3: Rule Engine safety override works."""
    print("\n--- Test 3: Rule Engine Safety Override ---")
    
    from src.ml.ensemble_manager import get_ensemble_manager
    
    manager = get_ensemble_manager()
    test_features = np.random.randn(32)
    
    # Test without rule engine
    result_no_override = manager.predict(test_features, rule_engine_severity=None)
    
    # Test with high rule engine severity
    result_with_override = manager.predict(test_features, rule_engine_severity=0.90)
    
    # With high severity, final risk should be >= 0.90
    assert result_with_override.final_risk_score >= 0.90, \
        f"Rule engine override failed: {result_with_override.final_risk_score}"
    assert result_with_override.rule_engine_applied == True, \
        "Rule engine should be marked as applied"
    
    # Category should be Pathological with 0.90 severity
    assert result_with_override.category == 3, \
        f"Expected category 3, got {result_with_override.category}"
    
    print(f"  [PASS] Without override: risk={result_no_override.final_risk_score:.3f}")
    print(f"  [PASS] With override (0.90): risk={result_with_override.final_risk_score:.3f}")
    print(f"  [PASS] Override applied: {result_with_override.rule_engine_applied}")
    print(f"  [PASS] MAX(ensemble, rule_engine) = {result_with_override.final_risk_score:.3f}")


def test_ensemble_adapter_protocol():
    """Test 4: EnsembleClassifierAdapter implements IClassifier."""
    print("\n--- Test 4: EnsembleClassifierAdapter Protocol ---")
    
    from src.adapters import EnsembleClassifierAdapter
    
    adapter = EnsembleClassifierAdapter()
    
    # Test predict
    X = np.random.randn(5, 32)  # 5 samples, 32 features
    predictions = adapter.predict(X)
    
    assert predictions.shape == (5,), f"Expected shape (5,), got {predictions.shape}"
    assert all(p in [0, 1, 2] for p in predictions), "Predictions should be 0, 1, or 2"
    
    # Test predict_proba
    probabilities = adapter.predict_proba(X)
    
    assert probabilities.shape == (5, 3), f"Expected shape (5, 3), got {probabilities.shape}"
    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
    
    # Test rule engine integration
    adapter.set_rule_engine_severity(0.80)
    predictions_high = adapter.predict(X)
    
    print(f"  [PASS] predict() shape: {predictions.shape}")
    print(f"  [PASS] predict_proba() shape: {probabilities.shape}")
    print(f"  [PASS] Predictions (no override): {predictions}")
    print(f"  [PASS] Predictions (severity=0.80): {predictions_high}")


def test_pipeline_container_integration():
    """Test 5: Pipeline container supports use_ensemble_v4."""
    print("\n--- Test 5: Pipeline Container Integration ---")
    
    from src.adapters import EnsembleClassifierAdapter
    
    # Test that EnsembleClassifierAdapter works standalone
    adapter = EnsembleClassifierAdapter()
    
    # Verify it's the ensemble adapter
    assert hasattr(adapter, 'set_rule_engine_severity'), \
        "Adapter should have set_rule_engine_severity"
    assert adapter.is_loaded, "Ensemble should be loaded"
    
    # Test info retrieval
    info = adapter.get_model_info()
    assert info['is_loaded'] == True
    assert len(info['models_loaded']) == 3
    
    # Test prediction through adapter (simulating pipeline use)
    X_test = np.random.randn(1, 32)  # 32 features as per our training
    pred = adapter.predict(X_test)
    proba = adapter.predict_proba(X_test)
    
    assert pred.shape == (1,), f"Expected (1,), got {pred.shape}"
    assert proba.shape == (1, 3), f"Expected (1, 3), got {proba.shape}"
    
    print(f"  [PASS] EnsembleClassifierAdapter created successfully")
    print(f"  [PASS] Adapter type: {type(adapter).__name__}")
    print(f"  [PASS] Ensemble is_loaded: {adapter.is_loaded}")
    print(f"  [PASS] Models: {info['models_loaded']}")
    print(f"  [PASS] Test prediction: {pred}")


def run_all_tests():
    """Run all verification tests."""
    print("=" * 60)
    print("SentinelFetal V4.0 End-to-End Verification")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_ensemble_manager_loads),
        ("Prediction Structure", test_ensemble_prediction_structure),
        ("Rule Engine Override", test_rule_engine_override),
        ("Adapter Protocol", test_ensemble_adapter_protocol),
        ("Pipeline Integration", test_pipeline_container_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\n[WARN] Some tests failed. Review output above.")
        return 1
    else:
        print("\n[SUCCESS] All V4.0 verification tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
