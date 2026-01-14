"""
Unit tests for CTG preprocessing module.

Tests the 10-second rule and gap filling logic.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocess import CTGPreprocessor, PreprocessingConfig


class TestGapFilling:
    """Tests for gap filling with the 10-second rule."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor with default config (4Hz, 10-second rule)."""
        config = PreprocessingConfig(
            sampling_rate=4.0,
            max_gap_seconds=10.0,
            fhr_min=50.0,
            fhr_max=240.0
        )
        return CTGPreprocessor(config)
    
    def test_5_second_gap_is_filled(self, preprocessor):
        """Test that a 5-second gap (20 samples @ 4Hz) is filled."""
        # Create signal with 5-second gap (20 samples)
        fhr = np.array([140.0] * 50 + [np.nan] * 20 + [140.0] * 50, dtype=float)
        
        result = preprocessor.process(fhr)
        
        # Gap should be filled - no NaN in the gap region
        gap_region = result.processed_signal[50:70]
        assert not np.any(np.isnan(gap_region)), \
            "5-second gap should be completely filled"
        
        # Filled mask should be True for gap region
        assert np.all(result.filled_mask[50:70]), \
            "Filled mask should be True for 5-second gap"
        
        # Values should be interpolated (all 140 since before=after=140)
        np.testing.assert_array_almost_equal(
            gap_region, 
            np.full(20, 140.0),
            err_msg="Interpolated values should be 140"
        )
    
    def test_15_second_gap_remains_nan(self, preprocessor):
        """Test that a 15-second gap (60 samples @ 4Hz) remains as NaN."""
        # Create signal with 15-second gap (60 samples)
        fhr = np.array([140.0] * 50 + [np.nan] * 60 + [140.0] * 50, dtype=float)
        
        result = preprocessor.process(fhr)
        
        # Gap should NOT be filled - all NaN in the gap region
        gap_region = result.processed_signal[50:110]
        assert np.all(np.isnan(gap_region)), \
            "15-second gap should remain as NaN"
        
        # Unfilled mask should be True for gap region
        assert np.all(result.unfilled_mask[50:110]), \
            "Unfilled mask should be True for 15-second gap"
        
        # Filled mask should be False for gap region
        assert not np.any(result.filled_mask[50:110]), \
            "Filled mask should be False for 15-second gap"
    
    def test_exactly_10_second_gap_is_filled(self, preprocessor):
        """Test that exactly 10-second gap (40 samples) is filled."""
        # Create signal with exactly 10-second gap (40 samples)
        fhr = np.array([130.0] * 50 + [np.nan] * 40 + [150.0] * 50, dtype=float)
        
        result = preprocessor.process(fhr)
        
        # Gap should be filled
        gap_region = result.processed_signal[50:90]
        assert not np.any(np.isnan(gap_region)), \
            "Exactly 10-second gap should be filled"
        
        # Values should be linearly interpolated from 130 to 150
        expected_start = 130.0
        expected_end = 150.0
        
        # Check that values increase from start to end
        assert gap_region[0] > expected_start, \
            "First interpolated value should be > 130"
        assert gap_region[-1] < expected_end, \
            "Last interpolated value should be < 150"
    
    def test_11_second_gap_remains_nan(self, preprocessor):
        """Test that 11-second gap (44 samples) remains as NaN."""
        # Create signal with 11-second gap (44 samples) - just over limit
        fhr = np.array([140.0] * 50 + [np.nan] * 44 + [140.0] * 50, dtype=float)
        
        result = preprocessor.process(fhr)
        
        # Gap should NOT be filled
        gap_region = result.processed_signal[50:94]
        assert np.all(np.isnan(gap_region)), \
            "11-second gap (44 samples) should remain as NaN"


class TestOutOfRangeHandling:
    """Tests for out-of-range value handling."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor with default config."""
        return CTGPreprocessor(PreprocessingConfig())
    
    def test_values_below_50_become_nan(self, preprocessor):
        """Test that FHR values below 50 BPM become NaN."""
        fhr = np.array([140.0, 140.0, 45.0, 140.0, 30.0, 140.0], dtype=float)
        
        result = preprocessor.process(fhr)
        
        # Original out-of-range positions should be marked in nan_mask
        assert result.nan_mask[2] == True, "Value 45 should be marked as invalid"
        assert result.nan_mask[4] == True, "Value 30 should be marked as invalid"
    
    def test_values_above_240_become_nan(self, preprocessor):
        """Test that FHR values above 240 BPM become NaN."""
        fhr = np.array([140.0, 250.0, 140.0, 300.0, 140.0], dtype=float)
        
        result = preprocessor.process(fhr)
        
        # Original out-of-range positions should be marked in nan_mask
        assert result.nan_mask[1] == True, "Value 250 should be marked as invalid"
        assert result.nan_mask[3] == True, "Value 300 should be marked as invalid"
    
    def test_zero_values_become_nan(self, preprocessor):
        """Test that zero values (common placeholder) become NaN."""
        fhr = np.array([140.0, 0.0, 140.0, 0.0, 0.0, 140.0], dtype=float)
        
        result = preprocessor.process(fhr)
        
        assert result.nan_mask[1] == True, "Zero should be marked as invalid"
        assert result.nan_mask[3] == True, "Zero should be marked as invalid"
        assert result.nan_mask[4] == True, "Zero should be marked as invalid"
    
    def test_small_out_of_range_gap_is_filled(self, preprocessor):
        """Test that small gaps created by out-of-range values are filled."""
        # 3 consecutive out-of-range values (0.75 seconds @ 4Hz) - should be filled
        fhr = np.array([140.0] * 10 + [0.0, 0.0, 0.0] + [140.0] * 10, dtype=float)
        
        result = preprocessor.process(fhr)
        
        # The gap should be filled since it's only 0.75 seconds
        gap_region = result.processed_signal[10:13]
        assert not np.any(np.isnan(gap_region)), \
            "Small out-of-range gap should be filled"


class TestLinearInterpolation:
    """Tests for linear interpolation quality."""
    
    @pytest.fixture
    def preprocessor(self):
        return CTGPreprocessor(PreprocessingConfig())
    
    def test_interpolation_is_linear(self, preprocessor):
        """Test that gap filling uses linear interpolation."""
        # Gap between 100 and 200 BPM
        fhr = np.array([100.0] + [np.nan] * 10 + [200.0], dtype=float)
        
        result = preprocessor.process(fhr)
        
        # Check linear increase
        gap_values = result.processed_signal[1:11]
        
        # Calculate expected linear values
        expected = np.linspace(100, 200, 12)[1:11]  # Exclude endpoints
        
        np.testing.assert_array_almost_equal(
            gap_values, 
            expected,
            decimal=1,
            err_msg="Interpolation should be linear"
        )


class TestEdgeCases:
    """Tests for edge cases."""
    
    @pytest.fixture
    def preprocessor(self):
        return CTGPreprocessor(PreprocessingConfig())
    
    def test_all_valid_signal(self, preprocessor):
        """Test processing of completely valid signal."""
        fhr = np.array([140.0] * 100, dtype=float)
        
        result = preprocessor.process(fhr)
        
        assert not np.any(result.nan_mask), "No values should be invalid"
        assert not np.any(result.filled_mask), "No gaps should need filling"
        np.testing.assert_array_equal(result.processed_signal, fhr)
    
    def test_gap_at_start(self, preprocessor):
        """Test handling of gap at signal start."""
        fhr = np.array([np.nan] * 5 + [140.0] * 50, dtype=float)
        
        result = preprocessor.process(fhr)
        
        # Gap at start should be filled with forward fill
        assert not np.any(np.isnan(result.processed_signal[:5])), \
            "Gap at start should be filled"
    
    def test_gap_at_end(self, preprocessor):
        """Test handling of gap at signal end."""
        fhr = np.array([140.0] * 50 + [np.nan] * 5, dtype=float)
        
        result = preprocessor.process(fhr)
        
        # Gap at end should be filled with backward fill
        assert not np.any(np.isnan(result.processed_signal[-5:])), \
            "Gap at end should be filled"
    
    def test_multiple_gaps(self, preprocessor):
        """Test handling of multiple gaps in signal."""
        # Two 5-second gaps (should be filled) and one 15-second gap (should not)
        fhr = np.concatenate([
            np.full(20, 140.0),      # Valid
            np.full(20, np.nan),     # 5-second gap - fill
            np.full(20, 140.0),      # Valid
            np.full(60, np.nan),     # 15-second gap - don't fill
            np.full(20, 140.0),      # Valid
            np.full(20, np.nan),     # 5-second gap - fill
            np.full(20, 140.0),      # Valid
        ])
        
        result = preprocessor.process(fhr)
        
        # First gap (indices 20-40) should be filled
        assert not np.any(np.isnan(result.processed_signal[20:40])), \
            "First 5-second gap should be filled"
        
        # Second gap (indices 60-120) should NOT be filled
        assert np.all(np.isnan(result.processed_signal[60:120])), \
            "15-second gap should remain NaN"
        
        # Third gap (indices 140-160) should be filled
        assert not np.any(np.isnan(result.processed_signal[140:160])), \
            "Third 5-second gap should be filled"


class TestStatistics:
    """Tests for preprocessing statistics."""
    
    def test_stats_calculation(self):
        """Test that statistics are calculated correctly."""
        preprocessor = CTGPreprocessor(PreprocessingConfig())
        
        # 100 samples, 10 invalid (NaN), at 4Hz = 25 seconds
        fhr = np.array([140.0] * 45 + [np.nan] * 10 + [140.0] * 45, dtype=float)
        
        result = preprocessor.process(fhr)
        
        assert result.stats['total_samples'] == 100
        assert result.stats['duration_seconds'] == 25.0
        assert result.stats['invalid_samples'] == 10
        assert result.stats['invalid_percent'] == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
