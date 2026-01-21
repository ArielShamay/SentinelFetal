
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import numpy as np
from src.data.signal_quality import calculate_fsqi, SignalQuality, apply_quality_gate
from src.rules.decelerations import calculate_descent_time

class TestFixVerification(unittest.TestCase):
    
    def test_fsqi_noise_rejection(self):
        """Test that high frequency noise is rejected."""
        # Generate clean signal
        t = np.linspace(0, 20, 80) # 20 seconds at 4Hz
        clean = 140.0 + 5.0 * np.sin(2 * np.pi * 0.1 * t)
        
        # Generate noisy signal (white noise)
        np.random.seed(42)
        noise = np.random.normal(0, 20, len(t))
        noisy_signal = clean + noise
        
        # Calculate FSQI
        result = calculate_fsqi(noisy_signal, sampling_rate=4.0)
        
        print(f"\nFSQI Score for noisy signal: {result.score} (Quality: {result.quality.value})")
        print(f"Noise score: {result.noise_score}")
        print(f"Stability score: {result.stability_score}")
        
        # Should be REJECTED (Low Quality)
        # Current bug expectation: It might pass if not strict enough
        self.assertTrue(result.score < 0.4, f"Noisy signal should be rejected! Score: {result.score}")
        self.assertFalse(result.should_classify, "Should classify flag should be False for noisy signal")

    def test_descent_time_calculation(self):
        """Test descent time calculation for Late Deceleration."""
        # Generate a 'V' shape deceleration
        # Baseline 140
        # Descent starts at t=10s, Nadir at t=50s (40s descent -> Late)
        # Recovery ends at t=90s
        sampling_rate = 4.0
        duration = 100 # seconds
        t = np.linspace(0, duration, int(duration * sampling_rate))
        fhr = np.full_like(t, 140.0)
        
        # Create gradual descent
        start_idx = int(10 * sampling_rate)
        nadir_idx = int(50 * sampling_rate)
        end_idx = int(90 * sampling_rate)
        
        # Linear descent
        fhr[start_idx:nadir_idx] = np.linspace(140.0, 100.0, nadir_idx - start_idx)
        # Linear recovery
        fhr[nadir_idx:end_idx] = np.linspace(100.0, 140.0, end_idx - nadir_idx)
        
        # In the pipeline, decel_start is usually detected when it crosses a threshold (e.g. baseline - 10bpm = 130)
        # 140 -> 100 over 40s (1 bpm/s). 130 is reached at 10s + 10s = 20s.
        threshold_crossing_idx = int(20 * sampling_rate)
        
        # Calculate descent time
        # It should pass threshold_crossing_idx and scan back to find start_idx (approx)
        dt = calculate_descent_time(fhr, threshold_crossing_idx, nadir_idx, sampling_rate)
        
        print(f"\nCalculated Descent Time: {dt:.2f}s (Expected ~40s)")
        
        # Should be >= 30s
        self.assertGreaterEqual(dt, 30.0, f"Descent time {dt} should be >= 30s")

if __name__ == '__main__':
    unittest.main()
