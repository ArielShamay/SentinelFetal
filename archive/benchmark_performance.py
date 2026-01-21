"""
SentinelFetal Performance Benchmark Script
הרץ עם: python benchmark_performance.py
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import CTUDataLoader
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig
from src.rules.baseline import calculate_baseline
from src.rules.variability import calculate_variability
from src.rules.decelerations import detect_decelerations
from src.rules.tachysystole import detect_tachysystole
from src.rules.sinusoidal import detect_sinusoidal_pattern
from src.models.moment_encoder import MomentFeatureExtractor

DATA_DIR = "data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0"

def benchmark():
    print("=" * 60)
    print("SentinelFetal Performance Benchmark")
    print("=" * 60)
    
    # Initialize
    try:
        loader = CTUDataLoader(DATA_DIR)
        preprocessor = CTGPreprocessor(PreprocessingConfig())
    except Exception as e:
        print(f"Initialization Error: {e}")
        return
    
    # Test MOMENT timing (mock vs real if available)
    print("\n[1] MOMENT Encoder Benchmark:")
    
    # Mock mode
    moment_mock = MomentFeatureExtractor(use_mock=True)
    test_signal = np.random.randn(2400) * 10 + 140
    
    mock_times = []
    for _ in range(10):
        start = time.time()
        _ = moment_mock.extract(test_signal)
        mock_times.append(time.time() - start)
    print(f"    Mock MOMENT: {np.mean(mock_times)*1000:.1f}ms ± {np.std(mock_times)*1000:.1f}ms")
    
    # Real MOMENT (if available)
    try:
        # Note: We need to handle the case where momentfm isn't installed
        # The MomentFeatureExtractor usually handles this internal to __init__, 
        # but let's be safe
        moment_real = MomentFeatureExtractor(use_mock=False)
        if not moment_real.use_mock:
            real_times = []
            for _ in range(5):
                start = time.time()
                _ = moment_real.extract(test_signal)
                real_times.append(time.time() - start)
            print(f"    Real MOMENT: {np.mean(real_times)*1000:.1f}ms ± {np.std(real_times)*1000:.1f}ms")
        else:
            print("    Real MOMENT: Not available (library missing, using mock)")
    except Exception as e:
        print(f"    Real MOMENT: Error/Not available - {e}")
    
    # Rule Engine timing
    print("\n[2] Rule Engine Benchmark (per 10-min window):")
    
    try:
        record = loader.load_record("1001")
        prep_result = preprocessor.process(record.fhr1)
        fhr = prep_result.processed_signal[:2400]
        uc = record.uc[:2400]
        
        # Baseline
        start = time.time()
        for _ in range(10):
            baseline = calculate_baseline(fhr)
        baseline_time = (time.time() - start) / 10
        print(f"    Baseline:      {baseline_time*1000:.1f}ms")
        
        # Variability
        start = time.time()
        for _ in range(10):
            var = calculate_variability(fhr)
        var_time = (time.time() - start) / 10
        print(f"    Variability:   {var_time*1000:.1f}ms")
        
        # Decelerations
        start = time.time()
        for _ in range(10):
            decels = detect_decelerations(fhr, uc, baseline.value)
        decel_time = (time.time() - start) / 10
        print(f"    Decelerations: {decel_time*1000:.1f}ms")
        
        # Tachysystole
        start = time.time()
        for _ in range(10):
            tachy = detect_tachysystole(uc)
        tachy_time = (time.time() - start) / 10
        print(f"    Tachysystole:  {tachy_time*1000:.1f}ms")
        
        # Sinusoidal
        start = time.time()
        for _ in range(10):
            sinus = detect_sinusoidal_pattern(fhr)
        sinus_time = (time.time() - start) / 10
        print(f"    Sinusoidal:    {sinus_time*1000:.1f}ms")
        
        total_rules = baseline_time + var_time + decel_time + tachy_time + sinus_time
        print(f"    ─────────────────────────")
        print(f"    Total Rules:   {total_rules*1000:.1f}ms")
    
    except Exception as e:
        print(f"    Error running rules: {e}")

    # pH Distribution
    print("\n[3] Dataset pH Distribution:")
    try:
        ph_values = loader.get_all_ph_values()
        known_ph = {k: v for k, v in ph_values.items() if v is not None}
        
        if known_ph:
            pathological = sum(1 for ph in known_ph.values() if ph < 7.15)
            intermediate = sum(1 for ph in known_ph.values() if 7.15 <= ph < 7.20)
            normal = sum(1 for ph in known_ph.values() if ph >= 7.20)
            
            print(f"    Total records: {len(ph_values)}")
            print(f"    With pH data:  {len(known_ph)}")
            print(f"    Category 3 (pH < 7.15):     {pathological} ({100*pathological/len(known_ph):.1f}%)")
            print(f"    Category 2 (7.15-7.20):     {intermediate} ({100*intermediate/len(known_ph):.1f}%)")
            print(f"    Category 1 (pH >= 7.20):    {normal} ({100*normal/len(known_ph):.1f}%)")
        else:
            print("    No pH data found in dataset headers.")
            
    except Exception as e:
        print(f"    Error processing pH data: {e}")
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

if __name__ == "__main__":
    benchmark()
