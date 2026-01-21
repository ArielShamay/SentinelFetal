"""
Quick test script to verify the modular pipeline and simulation work correctly.
"""
import sys
import io
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

print("=" * 60)
print("SentinelFetal System Verification")
print("=" * 60)
print()

# Test 1: Import modular pipeline
print("[1] Testing modular pipeline imports...")
try:
    from src.pipeline import PipelineContainer, AnalysisPipeline
    from src.interfaces import IClassifier, IBaselineCalculator
    from src.adapters import BaselineAdapter, ClassifierAdapter
    print("    [OK] Modular pipeline imports successful")
except Exception as e:
    print(f"    [FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Create container
print("[2] Testing pipeline container creation...")
try:
    container = PipelineContainer.create_default(use_mock_moment=True)
    assert container.validate(), "Container validation failed"
    print("    [OK] Container created and validated")
except Exception as e:
    print(f"    [FAIL] Container error: {e}")
    sys.exit(1)

# Test 3: Create pipeline
print("[3] Testing pipeline creation...")
try:
    pipeline = AnalysisPipeline(container)
    print("    [OK] Pipeline created")
except Exception as e:
    print(f"    [FAIL] Pipeline error: {e}")
    sys.exit(1)

# Test 4: Generate synthetic data and analyze
print("[4] Testing analysis with synthetic data...")
try:
    np.random.seed(42)
    n_samples = 2400  # 10 minutes at 4Hz
    
    # Normal FHR with variability
    fhr = 140 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    fhr += np.random.normal(0, 5, n_samples)
    
    # Simple UC pattern
    uc = 10 + 70 * np.abs(np.sin(np.linspace(0, 2*np.pi, n_samples)))
    
    result = pipeline.analyze(fhr, uc)
    
    print(f"    Category: {result.category}")
    print(f"    Confidence: {result.confidence:.2f}")
    print(f"    Baseline: {result.baseline.value:.1f} bpm")
    print(f"    Variability: {result.variability.value:.1f} bpm ({result.variability.category.value})")
    print(f"    Decelerations: {len(result.decelerations)}")
    print(f"    Tachysystole: {result.tachysystole.detected}")
    print(f"    Sinusoidal: {result.sinusoidal.detected}")
    print(f"    Alert Headline: {result.alert.headline}")
    print("    [OK] Analysis completed successfully")
except Exception as e:
    print(f"    [FAIL] Analysis error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test simulation components
print("[5] Testing simulation components...")
try:
    from src.simulation import (
        SimulationOrchestrator, OrchestratorConfig,
        PipelineAdapter, PipelineAdapterConfig,
        PatientGenerator, PatientConfig,
    )
    
    # Create a patient generator with proper config
    patient_config = PatientConfig(patient_id="test_patient", bed_number=1)
    patient_gen = PatientGenerator(patient_config)
    
    # Generate some data (100 ticks of 4 samples each)
    for _ in range(100):
        patient_gen.generate_tick(4)
    
    data = patient_gen.get_buffer_data()
    
    print(f"    Patient FHR samples: {len(data['fhr'])}")
    print(f"    Patient UC samples: {len(data['uc'])}")
    print(f"    FHR range: {data['fhr'].min():.1f} - {data['fhr'].max():.1f} bpm")
    print("    [OK] Simulation components working")
except Exception as e:
    print(f"    [FAIL] Simulation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test Pipeline Adapter
print("[6] Testing pipeline adapter for simulation...")
try:
    adapter_config = PipelineAdapterConfig(
        use_real_moment=False,  # Use mock for testing
        model_path="models/xgb_demo.json"
    )
    adapter = PipelineAdapter(adapter_config)
    
    # Create enough data for processing
    np.random.seed(42)
    n_samples = 2400
    test_fhr = 140 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    test_fhr += np.random.normal(0, 5, n_samples)
    test_uc = 10 + 70 * np.abs(np.sin(np.linspace(0, 2*np.pi, n_samples)))
    
    test_data = {
        'fhr': test_fhr,
        'uc': test_uc,
        'timestamps': np.arange(n_samples) / 4.0
    }
    
    result = adapter.process_patient("test_patient", test_data, run_moment=True)
    
    print(f"    Category: {result['category']}")
    print(f"    Confidence: {result['confidence']:.2f}")
    print(f"    Insufficient data: {result.get('insufficient_data', False)}")
    print("    [OK] Pipeline adapter working")
except Exception as e:
    print(f"    [FAIL] Pipeline adapter error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("[SUCCESS] All systems verified and working correctly!")
print("=" * 60)
print()
print("You can now run the simulation dashboard:")
print("  streamlit run src/ui/simulation_app.py")
print()
