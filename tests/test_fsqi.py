import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.signal_quality import calculate_fsqi

def test_fsqi_rejection():
    print("Testing FSQI Noise Rejection...")
    # 1. Generate clean signal (Should pass)
    t = np.linspace(0, 10, 40) # 10 seconds at 4Hz
    clean_fhr = np.full_like(t, 140.0)
    result_clean = calculate_fsqi(clean_fhr, 4.0)
    print(f"Clean Signal Score: {result_clean.score} (Should be >= 0.7)")
    
    # 2. Generate noisy signal (Should fail)
    # White noise around 140
    np.random.seed(42)
    noise = np.random.normal(0, 20, len(t))
    noisy_fhr = clean_fhr + noise
    result_noisy = calculate_fsqi(noisy_fhr, 4.0)
    print(f"Noisy Signal Score: {result_noisy.score} (Should be < 0.7)")
    print(f"  Valid Ratio: {result_noisy.valid_ratio}")
    print(f"  Physio Ratio: {result_noisy.physiological_ratio}")
    print(f"  Noise Score: {result_noisy.noise_score}")
    print(f"  Stability Score: {result_noisy.stability_score}")

    if result_noisy.score < 0.7:
        print("PASS: Noise rejected.")
    else:
        print("FAIL: Noise accepted.")

if __name__ == "__main__":
    test_fsqi_rejection()
