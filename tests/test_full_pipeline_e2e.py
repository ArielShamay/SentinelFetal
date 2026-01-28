"""
Full End-to-End Pipeline Test with Synthetic Data
=================================================
Test all stages from signal generation to final classification
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def run_e2e_test():
    print('='*70)
    print('FULL PIPELINE TEST - END TO END')
    print('='*70)

    # Stage 1: Import all components
    print('\n[1] Loading components...')
    from src.simulation.generators.patient_generator import PatientGenerator, PatientConfig
    from src.simulation.events.event_types import EventType, LateDecelerationParams, SinusoidalParams
    from src.adapters import (
        PreprocessorAdapter, BaselineAdapter, VariabilityAdapter, 
        DecelerationAdapter, TachysystoleAdapter, SinusoidalAdapter,
        OverrideAdapter, AlertAdapter, MiniRocketAdapter, ClassifierAdapter, FusionAdapter
    )
    from src.pipeline.container import PipelineContainer
    from src.pipeline.analysis_pipeline import AnalysisPipeline
    print('   All components loaded!')

    # Stage 2: Create synthetic signals - Normal case
    print('\n[2] Generating NORMAL synthetic signal (30 minutes)...')
    pcfg = PatientConfig(
        patient_id='test_normal', 
        bed_number=1, 
        baseline_fhr=140.0, 
        baseline_variability=10.0,
        contractions_per_10min=4.0
    )
    patient = PatientGenerator(pcfg)

    # Generate 30 minutes = 1800 seconds = 7200 samples at 4Hz
    fhr_list, uc_list = [], []
    for _ in range(1800):  # 1800 ticks = 1800 seconds
        data = patient.generate_tick(4)  # 4 samples per tick = 1 second
        fhr_list.extend(data['fhr'].tolist())
        uc_list.extend(data['uc'].tolist())

    fhr_normal = np.array(fhr_list)
    uc_normal = np.array(uc_list)
    print(f'   Signal length: {len(fhr_normal)} samples ({len(fhr_normal)/4/60:.1f} minutes)')
    print(f'   FHR range: [{fhr_normal.min():.1f}, {fhr_normal.max():.1f}] bpm')
    print(f'   FHR mean: {fhr_normal.mean():.1f} bpm, std: {fhr_normal.std():.1f} bpm')

    # Stage 3: Create pipeline
    print('\n[3] Creating analysis pipeline...')
    from src.adapters.xgboost_only_classifier import XGBoostOnlyClassifier
    
    # Create classifier adapter that conforms to IClassifier protocol
    class V6ClassifierAdapter:
        def __init__(self):
            self._clf = XGBoostOnlyClassifier()
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict categories as array for pipeline compatibility."""
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            results = []
            for i in range(X.shape[0]):
                pred = self._clf.predict(X[i])
                results.append(pred.category - 1)  # Convert 1-3 to 0-2
            return np.array(results)
        
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict probabilities."""
            return self._clf.predict_proba(X)
        
        def save_model(self, path: str) -> None:
            pass
        
        def load_model(self, path: str) -> None:
            pass
    
    container = PipelineContainer(
        preprocessor=PreprocessorAdapter(),
        baseline_calculator=BaselineAdapter(),
        variability_calculator=VariabilityAdapter(),
        deceleration_detector=DecelerationAdapter(),
        tachysystole_detector=TachysystoleAdapter(),
        sinusoidal_detector=SinusoidalAdapter(),
        feature_extractor=MiniRocketAdapter(model_path='models/minirocket_encoder.joblib', auto_fit=True),
        feature_fusion=FusionAdapter(),
        classifier=V6ClassifierAdapter(),  # V6 classifier adapted for pipeline
        medical_override=OverrideAdapter(),
        alert_generator=AlertAdapter(),
    )
    pipeline = AnalysisPipeline(container)
    print('   Pipeline created!')

    # Stage 4: Run analysis on normal signal
    print('\n[4] Running analysis on NORMAL signal...')
    result_normal = pipeline.analyze(fhr_normal, uc_normal)
    print(f'   Baseline: {result_normal.baseline.value} bpm')
    print(f'   Variability: {result_normal.variability.value:.1f} bpm ({result_normal.variability.category.name})')
    print(f'   Decelerations found: {len(result_normal.decelerations)}')
    print(f'   ML Prediction: Category {result_normal.ml_prediction}')
    print(f'   Override applied: {result_normal.was_overridden}')
    print(f'   FINAL CATEGORY: {result_normal.category}')
    print(f'   Alert: {result_normal.alert.headline}')

    # Stage 5: Generate pathological signal - Late decelerations
    print('\n[5] Generating PATHOLOGICAL signal (late decelerations)...')
    patient2 = PatientGenerator(PatientConfig(
        patient_id='test_late_decel', 
        bed_number=2, 
        baseline_fhr=140.0, 
        baseline_variability=10.0,
        contractions_per_10min=4.0
    ))
    # Inject late deceleration event
    patient2.inject_event(EventType.LATE_DECELERATION, LateDecelerationParams.severe())

    fhr_list2, uc_list2 = [], []
    for _ in range(1800):
        data = patient2.generate_tick(4)
        fhr_list2.extend(data['fhr'].tolist())
        uc_list2.extend(data['uc'].tolist())

    fhr_late = np.array(fhr_list2)
    uc_late = np.array(uc_list2)
    print(f'   FHR range: [{fhr_late.min():.1f}, {fhr_late.max():.1f}] bpm')
    print(f'   FHR mean: {fhr_late.mean():.1f} bpm, std: {fhr_late.std():.1f} bpm')

    # Stage 6: Run analysis on pathological signal
    print('\n[6] Running analysis on PATHOLOGICAL signal...')
    result_late = pipeline.analyze(fhr_late, uc_late)
    print(f'   Baseline: {result_late.baseline.value} bpm')
    print(f'   Variability: {result_late.variability.value:.1f} bpm ({result_late.variability.category.name})')
    print(f'   Decelerations found: {len(result_late.decelerations)}')
    if result_late.decelerations:
        for d in result_late.decelerations[:3]:
            print(f'      - Type: {d.decel_type}, Depth: {d.depth:.1f} bpm')
    print(f'   ML Prediction: Category {result_late.ml_prediction}')
    print(f'   Override applied: {result_late.was_overridden}')
    print(f'   FINAL CATEGORY: {result_late.category}')
    print(f'   Alert: {result_late.alert.headline}')

    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'Normal signal -> Category {result_normal.category} (expected: 1-2)')
    print(f'Late decels   -> Category {result_late.category} (expected: 2-3)')
    print('')
    if result_normal.category in [1, 2] and result_late.category in [2, 3]:
        print('PASS: Pipeline correctly differentiates normal from pathological!')
        return True
    else:
        print('WARNING: Check detection thresholds')
        return False

if __name__ == '__main__':
    run_e2e_test()
