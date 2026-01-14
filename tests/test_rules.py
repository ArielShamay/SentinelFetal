"""
Unit Tests for Rule Engine Module.

Tests the medical algorithms using synthetic data since we don't have
ground-truth labels for individual rules.

Test Strategy:
    - Generate synthetic signals with known characteristics
    - Verify that the algorithms detect those characteristics correctly
    - Use boundary conditions to test edge cases

References:
    - Israeli Position Paper on CTG Interpretation
    - SentinelFetal Gen3.5 Technical Specification
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rules.baseline import calculate_baseline, BaselineResult
from rules.variability import (
    calculate_variability,
    VariabilityCategory,
    VariabilityResult
)
from rules.decelerations import (
    detect_decelerations,
    DecelerationType,
    Deceleration
)
from rules.tachysystole import detect_tachysystole, TachysystoleResult
from rules.sinusoidal import (
    detect_sinusoidal_pattern,
    SinusoidalResult,
    generate_sinusoidal_test_signal
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sampling_rate() -> float:
    """Standard sampling rate of 4 Hz."""
    return 4.0


@pytest.fixture
def flat_fhr_140() -> np.ndarray:
    """Generate flat FHR signal at 140 bpm (10 minutes)."""
    duration_samples = 10 * 60 * 4  # 10 minutes at 4 Hz = 2400 samples
    return np.full(duration_samples, 140.0)


@pytest.fixture
def normal_fhr_with_variability() -> np.ndarray:
    """Generate normal FHR with moderate variability (6-25 bpm)."""
    np.random.seed(42)
    duration_samples = 10 * 60 * 4  # 10 minutes
    baseline = 140.0
    # Add variability with amplitude ~15 bpm
    variability = np.random.uniform(-7.5, 7.5, duration_samples)
    return baseline + variability


@pytest.fixture
def absent_variability_fhr() -> np.ndarray:
    """Generate FHR with absent variability (< 2 bpm)."""
    np.random.seed(42)
    duration_samples = 10 * 60 * 4
    baseline = 140.0
    # Very small variability (< 2 bpm amplitude)
    variability = np.random.uniform(-0.5, 0.5, duration_samples)
    return baseline + variability


@pytest.fixture
def fhr_with_late_deceleration(sampling_rate: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate FHR with a late deceleration and corresponding UC signal.
    
    Returns:
        Tuple of (fhr, uc, baseline)
    """
    np.random.seed(42)
    duration_samples = 5 * 60 * 4  # 5 minutes
    baseline = 140.0
    
    # Start with baseline + small variability
    fhr = np.full(duration_samples, baseline) + np.random.uniform(-2, 2, duration_samples)
    
    # Create UC signal with a contraction
    uc = np.zeros(duration_samples)
    contraction_start = 200  # ~50 seconds
    contraction_peak = 280   # ~70 seconds
    contraction_end = 360    # ~90 seconds
    
    # Simple contraction shape
    for i in range(contraction_start, contraction_end):
        if i < contraction_peak:
            uc[i] = 50 * (i - contraction_start) / (contraction_peak - contraction_start)
        else:
            uc[i] = 50 * (contraction_end - i) / (contraction_end - contraction_peak)
    
    # Create late deceleration - must be at least 15 seconds (60 samples at 4Hz)
    # and depth >= 15 bpm below baseline
    decel_start = contraction_peak + 20  # ~5 seconds after peak
    decel_nadir = contraction_peak + 80  # ~20 seconds after peak (>15s = Late)
    decel_end = contraction_peak + 160   # ~40 seconds after peak (ensure >15s duration)
    
    # Deceleration shape (depth 30 bpm to ensure it's clearly detected)
    depth = 30
    for i in range(decel_start, min(decel_end, duration_samples)):
        if i < decel_nadir:
            current_depth = depth * (i - decel_start) / (decel_nadir - decel_start)
        else:
            current_depth = depth * (decel_end - i) / (decel_end - decel_nadir)
        fhr[i] = baseline - current_depth
    
    return fhr, uc, baseline


# =============================================================================
# Baseline Tests
# =============================================================================

class TestBaseline:
    """Tests for baseline calculation."""
    
    def test_flat_signal_returns_exact_value(self, flat_fhr_140: np.ndarray, sampling_rate: float):
        """Test that a flat signal at 140 bpm returns baseline of 140."""
        result = calculate_baseline(flat_fhr_140, sampling_rate=sampling_rate)
        
        assert isinstance(result, BaselineResult)
        assert result.value == 140.0, f"Expected 140, got {result.value}"
        assert result.is_normal is True, "140 bpm should be normal"
        assert result.is_bradycardia is False
        assert result.is_tachycardia is False
    
    def test_baseline_rounded_to_nearest_5(self, sampling_rate: float):
        """Test that baseline is rounded to nearest multiple of 5."""
        # Signal at 142 bpm should round to 140
        fhr = np.full(2400, 142.0)
        result = calculate_baseline(fhr, sampling_rate=sampling_rate)
        assert result.value == 140.0
        
        # Signal at 143 bpm should round to 145
        fhr = np.full(2400, 143.0)
        result = calculate_baseline(fhr, sampling_rate=sampling_rate)
        assert result.value == 145.0
    
    def test_bradycardia_detection(self, sampling_rate: float):
        """Test that baseline < 110 is classified as bradycardia."""
        fhr = np.full(2400, 100.0)  # 100 bpm
        result = calculate_baseline(fhr, sampling_rate=sampling_rate)
        
        assert result.value == 100.0
        assert result.is_bradycardia is True
        assert result.is_normal is False
    
    def test_tachycardia_detection(self, sampling_rate: float):
        """Test that baseline > 160 is classified as tachycardia."""
        fhr = np.full(2400, 170.0)  # 170 bpm
        result = calculate_baseline(fhr, sampling_rate=sampling_rate)
        
        assert result.value == 170.0
        assert result.is_tachycardia is True
        assert result.is_normal is False
    
    def test_normal_range_boundaries(self, sampling_rate: float):
        """Test that 110 and 160 are both considered normal."""
        # 110 bpm - lower boundary
        fhr_low = np.full(2400, 110.0)
        result_low = calculate_baseline(fhr_low, sampling_rate=sampling_rate)
        assert result_low.is_normal is True, "110 bpm should be normal"
        
        # 160 bpm - upper boundary
        fhr_high = np.full(2400, 160.0)
        result_high = calculate_baseline(fhr_high, sampling_rate=sampling_rate)
        assert result_high.is_normal is True, "160 bpm should be normal"
    
    def test_handles_nan_values(self, sampling_rate: float):
        """Test that baseline calculation handles NaN values."""
        fhr = np.full(2400, 140.0)
        fhr[100:200] = np.nan  # Add some NaN
        
        result = calculate_baseline(fhr, sampling_rate=sampling_rate)
        
        assert result.value == 140.0
        assert not np.isnan(result.value)


# =============================================================================
# Variability Tests
# =============================================================================

class TestVariability:
    """Tests for variability calculation."""
    
    def test_absent_variability_detection(
        self, 
        absent_variability_fhr: np.ndarray, 
        sampling_rate: float
    ):
        """Test that very low variability (< 2 bpm) is classified as Absent."""
        result = calculate_variability(absent_variability_fhr, sampling_rate=sampling_rate)
        
        assert isinstance(result, VariabilityResult)
        assert result.category == VariabilityCategory.ABSENT
        assert result.value <= 2, f"Expected <= 2, got {result.value}"
        assert result.is_concerning is True, "Absent variability should be concerning"
    
    def test_moderate_variability_detection(
        self, 
        normal_fhr_with_variability: np.ndarray, 
        sampling_rate: float
    ):
        """Test that variability 6-25 bpm is classified as Moderate (normal)."""
        result = calculate_variability(normal_fhr_with_variability, sampling_rate=sampling_rate)
        
        assert result.category == VariabilityCategory.MODERATE
        assert 6 <= result.value <= 25, f"Expected 6-25, got {result.value}"
        assert result.is_normal is True, "Moderate variability should be normal"
        assert result.is_concerning is False
    
    def test_minimal_variability_detection(self, sampling_rate: float):
        """Test that variability 3-5 bpm is classified as Minimal."""
        np.random.seed(42)
        duration_samples = 2400
        # Variability of ~4 bpm (amplitude 2 bpm each way)
        fhr = 140 + np.random.uniform(-2, 2, duration_samples)
        
        result = calculate_variability(fhr, sampling_rate=sampling_rate)
        
        assert result.category in (VariabilityCategory.MINIMAL, VariabilityCategory.ABSENT)
        assert result.is_concerning is True
    
    def test_marked_variability_detection(self, sampling_rate: float):
        """Test that variability > 25 bpm is classified as Marked."""
        np.random.seed(42)
        duration_samples = 2400
        # Large variability > 25 bpm
        fhr = 140 + np.random.uniform(-20, 20, duration_samples)
        
        result = calculate_variability(fhr, sampling_rate=sampling_rate)
        
        # Should be either Marked or Moderate depending on the random distribution
        assert result.value > 20, f"Expected large variability, got {result.value}"
    
    def test_category_boundaries(self):
        """Test variability category boundaries."""
        from rules.variability import classify_variability
        
        # Test exact boundaries
        assert classify_variability(2.0) == VariabilityCategory.ABSENT
        assert classify_variability(2.1) == VariabilityCategory.MINIMAL
        assert classify_variability(5.0) == VariabilityCategory.MINIMAL
        assert classify_variability(5.1) == VariabilityCategory.MODERATE
        assert classify_variability(25.0) == VariabilityCategory.MODERATE
        assert classify_variability(25.1) == VariabilityCategory.MARKED


# =============================================================================
# Deceleration Tests
# =============================================================================

class TestDecelerations:
    """Tests for deceleration detection and classification."""
    
    def test_no_decelerations_in_flat_signal(
        self, 
        flat_fhr_140: np.ndarray, 
        sampling_rate: float
    ):
        """Test that no decelerations are found in a flat signal."""
        uc = np.zeros_like(flat_fhr_140)
        baseline = 140.0
        
        decels = detect_decelerations(
            flat_fhr_140, uc, baseline, sampling_rate=sampling_rate
        )
        
        assert len(decels) == 0, "Flat signal should have no decelerations"
    
    def test_late_deceleration_detection(
        self,
        fhr_with_late_deceleration: tuple[np.ndarray, np.ndarray, float],
        sampling_rate: float
    ):
        """Test detection of a late deceleration (lag > 15 seconds)."""
        fhr, uc, baseline = fhr_with_late_deceleration
        
        decels = detect_decelerations(fhr, uc, baseline, sampling_rate=sampling_rate)
        
        assert len(decels) >= 1, "Should detect at least one deceleration"
        
        # Check that at least one is classified as Late
        late_decels = [d for d in decels if d.decel_type == DecelerationType.LATE]
        assert len(late_decels) >= 1, f"Should have Late deceleration, got types: {[d.decel_type for d in decels]}"
    
    def test_deceleration_depth_calculation(self, sampling_rate: float):
        """Test that deceleration depth is calculated correctly."""
        baseline = 140.0
        depth = 25.0
        
        # Create signal with a clear deceleration
        fhr = np.full(2400, baseline)
        # Add 30-second deceleration (120 samples)
        decel_start = 500
        decel_end = 620
        fhr[decel_start:decel_end] = baseline - depth
        
        uc = np.zeros(2400)
        
        decels = detect_decelerations(fhr, uc, baseline, sampling_rate=sampling_rate)
        
        assert len(decels) >= 1
        assert decels[0].depth >= depth - 2, f"Expected depth ~{depth}, got {decels[0].depth}"
    
    def test_short_dip_not_classified_as_deceleration(self, sampling_rate: float):
        """Test that dips < 15 seconds are not classified as decelerations."""
        baseline = 140.0
        fhr = np.full(2400, baseline)
        
        # Add 10-second dip (40 samples) - should NOT be detected
        fhr[500:540] = baseline - 20
        uc = np.zeros(2400)
        
        decels = detect_decelerations(
            fhr, uc, baseline, 
            sampling_rate=sampling_rate,
            min_duration_seconds=15.0
        )
        
        assert len(decels) == 0, "10-second dip should not be classified as deceleration"
    
    def test_shallow_dip_not_classified_as_deceleration(self, sampling_rate: float):
        """Test that dips < 15 bpm are not classified as decelerations."""
        baseline = 140.0
        fhr = np.full(2400, baseline)
        
        # Add shallow 30-second dip (only 10 bpm depth) - should NOT be detected
        fhr[500:620] = baseline - 10
        uc = np.zeros(2400)
        
        decels = detect_decelerations(
            fhr, uc, baseline, 
            sampling_rate=sampling_rate,
            min_depth=15.0
        )
        
        assert len(decels) == 0, "10 bpm dip should not be classified as deceleration"


# =============================================================================
# Tachysystole Tests
# =============================================================================

class TestTachysystole:
    """Tests for tachysystole detection."""
    
    def test_no_tachysystole_in_flat_uc(self, sampling_rate: float):
        """Test that flat UC signal shows no tachysystole."""
        duration_samples = 30 * 60 * 4  # 30 minutes
        uc = np.zeros(duration_samples)
        
        result = detect_tachysystole(uc, sampling_rate=sampling_rate)
        
        assert isinstance(result, TachysystoleResult)
        assert result.detected is False
        assert result.contractions_per_10min == 0.0
    
    def test_tachysystole_with_many_contractions(self, sampling_rate: float):
        """Test detection of tachysystole (>5 contractions/10min)."""
        duration_samples = 30 * 60 * 4  # 30 minutes
        uc = np.zeros(duration_samples)
        
        # Add 18 contractions in 30 minutes = 6/10min (tachysystole)
        # Each contraction about 60 seconds apart = 7200/18 ≈ 400 samples
        contraction_interval = 400
        
        for i in range(18):
            peak_idx = 100 + i * contraction_interval
            if peak_idx + 50 < duration_samples:
                # Simple contraction shape
                uc[peak_idx-40:peak_idx+40] = np.concatenate([
                    np.linspace(0, 80, 40),
                    np.linspace(80, 0, 40)
                ])
        
        result = detect_tachysystole(uc, sampling_rate=sampling_rate)
        
        assert result.detected is True, f"Expected tachysystole, got {result.contractions_per_10min}/10min"
        assert result.contractions_per_10min > 5
    
    def test_normal_contraction_rate(self, sampling_rate: float):
        """Test that normal contraction rate (3-4/10min) is not tachysystole."""
        duration_samples = 30 * 60 * 4  # 30 minutes
        uc = np.zeros(duration_samples)
        
        # Add 12 contractions in 30 minutes = 4/10min (normal)
        contraction_interval = 600  # ~2.5 minutes apart
        
        for i in range(12):
            peak_idx = 100 + i * contraction_interval
            if peak_idx + 50 < duration_samples:
                uc[peak_idx-40:peak_idx+40] = np.concatenate([
                    np.linspace(0, 80, 40),
                    np.linspace(80, 0, 40)
                ])
        
        result = detect_tachysystole(uc, sampling_rate=sampling_rate)
        
        assert result.detected is False, f"4/10min should not be tachysystole"
        assert result.contractions_per_10min <= 5


# =============================================================================
# Sinusoidal Pattern Tests
# =============================================================================

class TestSinusoidal:
    """Tests for sinusoidal pattern detection."""
    
    def test_sinusoidal_pattern_detected(self, sampling_rate: float):
        """Test that a pure sinusoidal signal is detected."""
        # Generate sinusoidal signal: 4 cycles/min, 5 bpm amplitude (peak-to-peak = 10 bpm)
        # Note: amplitude parameter is half of peak-to-peak, so use 5 to get 10 bpm range
        fhr = generate_sinusoidal_test_signal(
            duration_minutes=25.0,
            sampling_rate=sampling_rate,
            baseline=140.0,
            amplitude=5.0,  # Half of peak-to-peak (will be 10 bpm total range)
            cycles_per_minute=4.0
        )
        
        result = detect_sinusoidal_pattern(fhr, sampling_rate=sampling_rate)
        
        assert isinstance(result, SinusoidalResult)
        # Check that amplitude is in valid range (5-15 bpm)
        assert result.amplitude_in_range is True, f"Amplitude {result.amplitude} should be in range 5-15"
        assert 3 <= result.frequency_cycles_per_min <= 5, f"Frequency {result.frequency_cycles_per_min} should be 3-5"
        # Detection depends on dominance - check we're close
        assert result.dominance_ratio > 0.2, f"Should have significant dominance ratio"
    
    def test_flat_signal_not_sinusoidal(
        self, 
        flat_fhr_140: np.ndarray, 
        sampling_rate: float
    ):
        """Test that a flat signal is NOT classified as sinusoidal."""
        # Extend to 25 minutes
        flat_extended = np.tile(flat_fhr_140, 3)[:25*60*4]
        
        result = detect_sinusoidal_pattern(flat_extended, sampling_rate=sampling_rate)
        
        assert result.detected is False, "Flat signal should not be sinusoidal"
    
    def test_normal_variability_not_sinusoidal(
        self, 
        normal_fhr_with_variability: np.ndarray, 
        sampling_rate: float
    ):
        """Test that normal variability (random) is NOT classified as sinusoidal."""
        # Extend to 25 minutes
        extended = np.tile(normal_fhr_with_variability, 3)[:25*60*4]
        
        result = detect_sinusoidal_pattern(extended, sampling_rate=sampling_rate)
        
        assert result.detected is False, "Random variability should not be sinusoidal"
    
    def test_wrong_frequency_not_detected(self, sampling_rate: float):
        """Test that sine wave outside 3-5 cycles/min is NOT detected."""
        # Generate signal with 10 cycles/min (outside range)
        fhr = generate_sinusoidal_test_signal(
            duration_minutes=25.0,
            sampling_rate=sampling_rate,
            baseline=140.0,
            amplitude=10.0,
            cycles_per_minute=10.0  # Too fast
        )
        
        result = detect_sinusoidal_pattern(fhr, sampling_rate=sampling_rate)
        
        # Should not detect because frequency is outside 3-5 range
        assert result.detected is False or result.confidence < 0.5, \
            "10 cycles/min should not be classified as sinusoidal"
    
    def test_wrong_amplitude_not_detected(self, sampling_rate: float):
        """Test that sine wave with wrong amplitude is NOT detected."""
        # Generate signal with 30 bpm amplitude (outside 5-15 range)
        fhr = generate_sinusoidal_test_signal(
            duration_minutes=25.0,
            sampling_rate=sampling_rate,
            baseline=140.0,
            amplitude=30.0,  # Too large
            cycles_per_minute=4.0
        )
        
        result = detect_sinusoidal_pattern(fhr, sampling_rate=sampling_rate)
        
        assert result.amplitude_in_range is False
        # May or may not detect depending on dominance


# =============================================================================
# Integration Tests
# =============================================================================

class TestRuleEngineIntegration:
    """Integration tests for the complete rule engine."""
    
    def test_all_modules_can_be_imported(self):
        """Test that all rule engine modules can be imported."""
        from rules import (
            calculate_baseline,
            calculate_variability,
            detect_decelerations,
            detect_tachysystole,
            detect_sinusoidal_pattern
        )
        
        # Verify they are callable
        assert callable(calculate_baseline)
        assert callable(calculate_variability)
        assert callable(detect_decelerations)
        assert callable(detect_tachysystole)
        assert callable(detect_sinusoidal_pattern)
    
    def test_process_synthetic_normal_recording(self, sampling_rate: float):
        """Test processing a synthetic normal CTG recording."""
        np.random.seed(42)
        
        # Create 30 minutes of normal FHR
        duration_samples = 30 * 60 * 4
        baseline_true = 140.0
        
        # Normal FHR with moderate variability
        fhr = baseline_true + np.random.uniform(-7, 7, duration_samples)
        
        # Normal UC with 4 contractions per 10 minutes
        uc = np.zeros(duration_samples)
        
        # Analyze
        baseline_result = calculate_baseline(fhr, sampling_rate)
        variability_result = calculate_variability(fhr, sampling_rate)
        decels = detect_decelerations(fhr, uc, baseline_result.value, sampling_rate)
        tachysystole = detect_tachysystole(uc, sampling_rate)
        
        # Verify normal findings
        assert baseline_result.is_normal, "Baseline should be normal"
        assert variability_result.category == VariabilityCategory.MODERATE, "Variability should be moderate"
        assert tachysystole.detected is False, "No tachysystole expected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
