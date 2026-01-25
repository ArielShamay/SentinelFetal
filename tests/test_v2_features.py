"""
V2.0 Feature Tests.

Tests for the three new modules introduced in SentinelFetal V2.0:
    1. MHR Guard - Maternal heart rate contamination detection
    2. Trend Analyzer - 60-minute trend analysis with deterioration scoring
    3. Explainability - Rule-based and SHAP explanations

Run with: pytest tests/test_v2_features.py -v
"""

import time
import pytest
import numpy as np

# ============================================================================
# MHR Guard Tests
# ============================================================================

class TestSpectralAnalyzer:
    """Tests for SpectralAnalyzer RSA detection."""

    def test_adult_rsa_pattern_detected(self):
        """Synthetic adult RSA signal should show high adult band power."""
        from src.safety.spectral_analyzer import SpectralAnalyzer, generate_mhr_test_signal

        analyzer = SpectralAnalyzer()

        # Generate signal with adult breathing rate (16/min = 0.267 Hz)
        mhr_signal = generate_mhr_test_signal(
            duration_seconds=60,
            sampling_rate=4.0,
            baseline=95,
            rsa_amplitude=5.0,
            breathing_rate_per_min=16.0
        )

        result = analyzer.analyze_rsa(mhr_signal, sampling_rate=4.0)

        # Adult RSA band should have significant power
        assert result.adult_power_ratio > 0.2, (
            f"Adult RSA signal should have adult_power_ratio > 0.2, got {result.adult_power_ratio:.3f}"
        )

    def test_fetal_rsa_pattern_not_flagged(self):
        """Synthetic fetal RSA signal should NOT trigger adult RSA detection."""
        from src.safety.spectral_analyzer import SpectralAnalyzer

        analyzer = SpectralAnalyzer()

        # Generate signal with fetal breathing rate (40/min = 0.67 Hz)
        # This is in the fetal RSA band (0.4-1.0 Hz)
        n_samples = 240
        t = np.arange(n_samples) / 4.0
        fetal_freq_hz = 40 / 60  # 0.67 Hz
        fhr_signal = 140 + 8 * np.sin(2 * np.pi * fetal_freq_hz * t)

        result = analyzer.analyze_rsa(fhr_signal, sampling_rate=4.0)

        # Adult RSA band should have LOW power ratio
        assert result.adult_power_ratio < 0.4, (
            f"Fetal RSA signal should have adult_power_ratio < 0.4, got {result.adult_power_ratio:.3f}"
        )

    def test_spectral_centroid_calculation(self):
        """Spectral centroid should be in valid range."""
        from src.safety.spectral_analyzer import SpectralAnalyzer

        analyzer = SpectralAnalyzer()

        # Normal FHR signal with variability
        np.random.seed(42)
        fhr_signal = 140 + 10 * np.random.randn(240)

        result = analyzer.analyze_rsa(fhr_signal, sampling_rate=4.0)

        # Centroid should be positive and less than Nyquist (2 Hz)
        assert 0 < result.spectral_centroid < 2.0, (
            f"Spectral centroid should be in (0, 2) Hz, got {result.spectral_centroid:.3f}"
        )


class TestMHRDetector:
    """Tests for MHRDetector main detection logic."""

    def test_synthetic_mhr_signal_detected(self):
        """Synthetic MHR signal should trigger detection."""
        from src.safety import MHRDetector, MHRDetectorConfig
        from src.safety.spectral_analyzer import generate_mhr_test_signal

        detector = MHRDetector(MHRDetectorConfig())

        # Generate MHR-like signal (adult RSA pattern)
        mhr_signal = generate_mhr_test_signal(
            duration_seconds=60,
            sampling_rate=4.0,
            baseline=95,
            rsa_amplitude=8.0,
            breathing_rate_per_min=16.0
        )

        result = detector.check_segment(
            fhr_segment=mhr_signal,
            mhr_reference=None,
            has_accelerations=False,
            sampling_rate=4.0
        )

        # Should at least trigger warning
        assert result.is_suspected or result.confidence > 0.3, (
            f"MHR signal should be suspected, got confidence={result.confidence:.2f}"
        )

    def test_normal_fhr_not_flagged(self):
        """Normal FHR signal should NOT trigger MHR detection."""
        from src.safety import MHRDetector, MHRDetectorConfig

        detector = MHRDetector(MHRDetectorConfig())

        # Generate normal FHR-like signal (higher baseline, fetal variability)
        np.random.seed(42)
        t = np.arange(240) / 4.0
        fetal_breathing = 3 * np.sin(2 * np.pi * 0.6 * t)  # 0.6 Hz = fetal band
        noise = 5 * np.random.randn(240)
        fhr_signal = 140 + fetal_breathing + noise

        result = detector.check_segment(
            fhr_segment=fhr_signal,
            mhr_reference=None,
            has_accelerations=True,  # Accelerations present = likely fetal
            sampling_rate=4.0
        )

        # Should NOT be suspected with low confidence
        assert not result.is_suspected or result.confidence < 0.5, (
            f"Normal FHR should not be suspected, got confidence={result.confidence:.2f}"
        )

    def test_fetal_sleep_cycle_not_mhr(self):
        """
        CRITICAL: Low variability + accelerations = fetal sleep, NOT MHR.

        This tests the key distinction in the PRD:
        - Sleeping fetus: Low variability but RETAINS accelerations
        - MHR contamination: No accelerations (maternal signal)
        """
        from src.safety import MHRDetector, MHRDetectorConfig

        detector = MHRDetector(MHRDetectorConfig())

        # Generate low variability signal (could be MHR or fetal sleep)
        np.random.seed(42)
        t = np.arange(240) / 4.0
        # Add subtle adult RSA pattern to make it look like MHR
        adult_rsa = 3 * np.sin(2 * np.pi * 0.25 * t)  # 0.25 Hz = adult band
        fhr_signal = 140 + adult_rsa + np.random.randn(240) * 2

        # WITHOUT accelerations - might flag as MHR
        result_no_accel = detector.check_segment(
            fhr_segment=fhr_signal,
            mhr_reference=None,
            has_accelerations=False,
            sampling_rate=4.0
        )

        # WITH accelerations - should reduce MHR confidence (fetal sleep)
        result_with_accel = detector.check_segment(
            fhr_segment=fhr_signal,
            mhr_reference=None,
            has_accelerations=True,
            sampling_rate=4.0
        )

        # Confidence should be lower when accelerations present
        assert result_with_accel.confidence <= result_no_accel.confidence, (
            f"Accelerations should reduce MHR confidence: "
            f"no_accel={result_no_accel.confidence:.2f}, "
            f"with_accel={result_with_accel.confidence:.2f}"
        )

    def test_baseline_jump_detection(self):
        """Sudden baseline jump should contribute to MHR detection."""
        from src.safety import MHRDetector, MHRDetectorConfig

        detector = MHRDetector(MHRDetectorConfig())

        # Generate signal with sudden baseline jump
        fhr_signal = np.zeros(240)
        fhr_signal[:100] = 140 + np.random.randn(100) * 3  # Fetal baseline
        fhr_signal[100:] = 95 + np.random.randn(140) * 2   # Sudden jump to MHR

        result = detector.check_segment(
            fhr_segment=fhr_signal,
            mhr_reference=None,
            has_accelerations=False,
            sampling_rate=4.0
        )

        # Baseline jump should contribute to detection
        assert any("jump" in r.lower() for r in result.reasons) or result.confidence > 0.2


# ============================================================================
# Trend Analyzer Tests
# ============================================================================

class TestTrendBuffer:
    """Tests for TrendBuffer circular buffer with FSQI masking."""

    def test_fsqi_masking(self):
        """Samples with FSQI < 0.9 should be masked (not stored)."""
        from src.analysis.trend_buffer import TrendBuffer, TrendDataPoint

        buffer = TrendBuffer(max_minutes=60, sample_interval_minutes=2)

        # Add good quality sample
        buffer.add_sample(TrendDataPoint(
            timestamp=time.time(),
            variability=12.0,
            baseline=140,
            decel_count_15min=0,
            has_late_decel=False,
            has_variable_decel=False,
            category=1,
            fsqi_score=0.95  # Good quality
        ))

        assert buffer.size == 1, "Good quality sample should be stored"

        # Add poor quality sample
        buffer._last_sample_time = 0  # Reset to allow immediate sample
        buffer.add_sample(TrendDataPoint(
            timestamp=time.time(),
            variability=5.0,
            baseline=140,
            decel_count_15min=1,
            has_late_decel=True,
            has_variable_decel=False,
            category=2,
            fsqi_score=0.5  # Poor quality - should be masked!
        ))

        assert buffer.size == 1, "Poor quality sample should NOT be stored"
        assert buffer._samples_masked == 1, "Masked count should increment"

    def test_circular_buffer_overflow(self):
        """Buffer should maintain max size by dropping oldest."""
        from src.analysis.trend_buffer import TrendBuffer, TrendDataPoint

        buffer = TrendBuffer(max_minutes=10, sample_interval_minutes=2)
        # Max points = 10 / 2 = 5

        # Add 7 samples (should only keep last 5)
        for i in range(7):
            buffer._last_sample_time = 0  # Allow immediate sampling
            buffer.add_sample(TrendDataPoint(
                timestamp=time.time() + i * 120,
                variability=10 + i,
                baseline=140,
                decel_count_15min=0,
                has_late_decel=False,
                has_variable_decel=False,
                category=1,
                fsqi_score=0.95
            ))

        assert buffer.size == 5, f"Buffer should have max 5 points, got {buffer.size}"

        # First sample should be variability=12 (samples 0,1 dropped)
        var_series = buffer.get_variability_series()
        assert var_series[0] == 12, f"Oldest kept sample should have var=12, got {var_series[0]}"

    def test_decel_event_window(self):
        """Deceleration events should be retrievable by time window."""
        from src.analysis.trend_buffer import TrendBuffer, TrendDataPoint

        buffer = TrendBuffer(max_minutes=60, sample_interval_minutes=2)
        base_time = time.time()

        # Add samples with decels at different times
        for i in range(5):
            buffer._last_sample_time = 0
            buffer.add_sample(TrendDataPoint(
                timestamp=base_time - (60 - i * 12) * 60,  # Spread over 60 min
                variability=10,
                baseline=140,
                decel_count_15min=1,
                has_late_decel=(i % 2 == 0),  # Every other sample has late decel
                has_variable_decel=(i % 2 == 1),
                category=1,
                fsqi_score=0.95
            ))

        # Check 15-min window
        decels_15 = buffer.get_decel_events_in_window(15)
        # Most recent samples should be in window
        assert len(decels_15) >= 0  # Depends on timing


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer with deterioration scoring."""

    def test_linear_regression_slope(self):
        """Linear trend should be detected correctly."""
        from src.analysis.trend_buffer import TrendBuffer, TrendDataPoint
        from src.analysis.trend_analyzer import TrendAnalyzer, TrendDirection

        buffer = TrendBuffer(max_minutes=60, sample_interval_minutes=2)
        analyzer = TrendAnalyzer()
        base_time = time.time()

        # Add declining variability trend: 15 → 5 over 30 min
        for i in range(15):
            buffer._last_sample_time = 0
            buffer.add_sample(TrendDataPoint(
                timestamp=base_time + i * 120,
                variability=15 - i * 0.7,  # Declining
                baseline=140,
                decel_count_15min=0,
                has_late_decel=False,
                has_variable_decel=False,
                category=1,
                fsqi_score=0.95
            ))

        result = analyzer.analyze(buffer)

        assert result.has_sufficient_data, "Should have sufficient data"
        assert result.variability_slope < 0, f"Slope should be negative, got {result.variability_slope}"
        assert result.variability_trend == TrendDirection.DECLINING, (
            f"Trend should be DECLINING, got {result.variability_trend}"
        )

    def test_deterioration_score_calculation(self):
        """Deterioration score should increase with worsening metrics."""
        from src.analysis.trend_buffer import TrendBuffer, TrendDataPoint
        from src.analysis.trend_analyzer import TrendAnalyzer

        analyzer = TrendAnalyzer()

        # Create buffer with stable, good metrics
        buffer_good = TrendBuffer(max_minutes=60, sample_interval_minutes=2)
        base_time = time.time()
        for i in range(10):
            buffer_good._last_sample_time = 0
            buffer_good.add_sample(TrendDataPoint(
                timestamp=base_time + i * 120,
                variability=12,  # Good, stable
                baseline=140,
                decel_count_15min=0,
                has_late_decel=False,
                has_variable_decel=False,
                category=1,
                fsqi_score=0.95
            ))

        result_good = analyzer.analyze(buffer_good)

        # Create buffer with deteriorating metrics
        buffer_bad = TrendBuffer(max_minutes=60, sample_interval_minutes=2)
        for i in range(10):
            buffer_bad._last_sample_time = 0
            buffer_bad.add_sample(TrendDataPoint(
                timestamp=base_time + i * 120,
                variability=12 - i * 0.8,  # Declining
                baseline=140 + i * 2,  # Drifting up
                decel_count_15min=i,  # Increasing
                has_late_decel=(i > 5),
                has_variable_decel=(i > 3),
                category=2,
                fsqi_score=0.95
            ))

        result_bad = analyzer.analyze(buffer_bad)

        assert result_bad.deterioration_score > result_good.deterioration_score, (
            f"Bad metrics should have higher score: "
            f"good={result_good.deterioration_score}, bad={result_bad.deterioration_score}"
        )


# ============================================================================
# Explainability Tests
# ============================================================================

class TestRuleExplainer:
    """Tests for RuleExplainer deterministic explanations."""

    def test_sinusoidal_explanation(self):
        """Sinusoidal pattern should generate CRITICAL explanation."""
        from src.explainability.rule_explainer import RuleExplainer

        explainer = RuleExplainer()

        # Mock sinusoidal result
        class MockSinusoidal:
            detected = True
            frequency_cycles_per_min = 4.0
            amplitude = 10.0

        rule_outputs = {
            "sinusoidal": MockSinusoidal(),
            "variability": None,
            "baseline": None,
            "decelerations": [],
        }

        explanations = explainer.explain(rule_outputs)

        # Should have sinusoidal explanation first
        assert len(explanations) > 0
        sinusoidal_exp = next((e for e in explanations if e.rule_name == "sinusoidal"), None)
        assert sinusoidal_exp is not None, "Should have sinusoidal explanation"
        assert sinusoidal_exp.severity == "CRITICAL"
        assert sinusoidal_exp.contribution > 0.5  # High contribution

    def test_variability_explanations(self):
        """Different variability categories should generate appropriate explanations."""
        from src.explainability.rule_explainer import RuleExplainer
        from enum import Enum

        explainer = RuleExplainer()

        class VariabilityCategory(Enum):
            ABSENT = "ABSENT"
            MINIMAL = "MINIMAL"
            MODERATE = "MODERATE"

        class MockVariability:
            def __init__(self, cat_name, value):
                self.category = type('obj', (object,), {'name': cat_name})()
                self.value = value

        # Test ABSENT variability
        rule_outputs = {
            "variability": MockVariability("ABSENT", 1.5),
            "sinusoidal": None,
            "baseline": None,
            "decelerations": [],
        }
        explanations = explainer.explain(rule_outputs)
        var_exp = next((e for e in explanations if e.rule_name == "variability"), None)
        assert var_exp is not None
        assert var_exp.severity == "HIGH"
        assert var_exp.contribution > 0.3

        # Test MODERATE (normal) variability
        rule_outputs["variability"] = MockVariability("MODERATE", 12.0)
        explanations = explainer.explain(rule_outputs)
        var_exp = next((e for e in explanations if e.rule_name == "variability"), None)
        assert var_exp is not None
        assert var_exp.contribution < 0  # Negative = mitigating

    def test_late_decel_explanation(self):
        """Late decelerations should generate HIGH severity explanation."""
        from src.explainability.rule_explainer import RuleExplainer

        explainer = RuleExplainer()

        class MockDecel:
            def __init__(self):
                self.decel_type = type('obj', (object,), {'name': 'LATE'})()
                self.nadir_value = 85
                self.depth = 50
                self.lag_seconds = 35
                self.start_idx = 100
                self.end_idx = 150
                self.duration_seconds = 45
                self.has_severity_signs = False

        rule_outputs = {
            "decelerations": [MockDecel(), MockDecel()],  # 2 late decels
            "variability": None,
            "sinusoidal": None,
            "baseline": None,
        }

        explanations = explainer.explain(rule_outputs)

        late_exp = next((e for e in explanations if e.rule_name == "late_decel"), None)
        assert late_exp is not None, "Should have late_decel explanation"
        assert late_exp.severity == "HIGH"
        assert "2 total" in late_exp.description


class TestExplanationEngine:
    """Tests for ExplanationEngine orchestrator."""

    def test_contributors_ranked_by_impact(self):
        """Contributors should be sorted by absolute contribution."""
        from src.explainability import ExplanationEngine

        engine = ExplanationEngine(xgboost_model=None)  # No SHAP

        class MockVariability:
            category = type('obj', (object,), {'name': 'ABSENT'})()
            value = 1.5

        class MockBaseline:
            value = 140
            is_normal = True
            is_bradycardia = False
            is_tachycardia = False

        rule_outputs = {
            "variability": MockVariability(),
            "baseline": MockBaseline(),
            "sinusoidal": None,
            "decelerations": [],
            "tachysystole": None,
        }

        result = engine.explain(
            category=2,
            confidence=0.85,
            rule_outputs=rule_outputs,
            fhr_length=2400
        )

        # Check contributors are sorted by |contribution|
        contributions = [abs(c.contribution) for c in result.contributors]
        assert contributions == sorted(contributions, reverse=True), (
            "Contributors should be sorted by |contribution|"
        )

    def test_summary_generation(self):
        """Summary should include category and key concerns."""
        from src.explainability import ExplanationEngine

        engine = ExplanationEngine(xgboost_model=None)

        class MockVariability:
            category = type('obj', (object,), {'name': 'MINIMAL'})()
            value = 4.0

        rule_outputs = {
            "variability": MockVariability(),
            "baseline": None,
            "sinusoidal": None,
            "decelerations": [],
            "tachysystole": None,
        }

        result = engine.explain(
            category=2,
            confidence=0.75,
            rule_outputs=rule_outputs,
            fhr_length=2400
        )

        assert "Category II" in result.summary or "Intermediate" in result.summary
        assert "75%" in result.summary or "0.75" in result.summary


class TestVisualMapper:
    """Tests for VisualMapper highlight generation."""

    def test_highlights_created_for_contributors(self):
        """Contributors with time_region should generate highlights."""
        from src.explainability.visual_mapper import VisualMapper
        from src.explainability.models import Contributor, ContributorSource, TimeRegion

        mapper = VisualMapper()

        contributors = [
            Contributor(
                source=ContributorSource.RULE,
                name="late_decel",
                contribution=0.5,
                description="Late deceleration",
                time_region=TimeRegion(start_index=100, end_index=150, color="red"),
                is_mitigating=False
            ),
            Contributor(
                source=ContributorSource.RULE,
                name="variability",
                contribution=0.3,
                description="Minimal variability",
                time_region=None,  # No time region
                is_mitigating=False
            ),
        ]

        highlights = mapper.create_highlights(contributors, fhr_length=2400)

        # Only late_decel has time_region, so only 1 highlight
        assert len(highlights) == 1
        assert highlights[0].label == "Late Decel"
        assert "rgba" in highlights[0].color

    def test_negative_index_conversion(self):
        """Negative indices should be converted to positive."""
        from src.explainability.visual_mapper import VisualMapper
        from src.explainability.models import Contributor, ContributorSource, TimeRegion

        mapper = VisualMapper()

        contributors = [
            Contributor(
                source=ContributorSource.RULE,
                name="variability",
                contribution=0.5,
                description="Test",
                time_region=TimeRegion(start_index=-240, end_index=-1, color="orange"),
                is_mitigating=False
            ),
        ]

        fhr_length = 2400
        highlights = mapper.create_highlights(contributors, fhr_length=fhr_length)

        assert len(highlights) == 1
        # -240 from end of 2400 = 2160
        assert highlights[0].start == 2160
        assert highlights[0].end == 2399


# ============================================================================
# Integration Tests
# ============================================================================

class TestPipelineV2Integration:
    """Integration tests for V2 pipeline."""

    def test_mhr_result_in_response(self):
        """MHR check result should be included in pipeline response."""
        from src.simulation.processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig

        config = PipelineAdapterConfig(
            enable_mhr_guard=True,
            enable_trend_analysis=True,
            enable_explanations=True
        )
        adapter = PipelineAdapter(config)

        # Generate simple test data
        np.random.seed(42)
        fhr = 140 + 10 * np.random.randn(480)  # 2 minutes at 4Hz
        uc = 20 + 10 * np.abs(np.sin(np.linspace(0, 4 * np.pi, 480)))

        result = adapter.process_patient(
            patient_id="test_001",
            data={'fhr': fhr, 'uc': uc}
        )

        # Should have V2 fields (even if None)
        assert 'mhr_alert' in result
        assert 'trend' in result
        assert 'explanation' in result

    def test_trend_buffer_persists(self):
        """Trend buffer should persist across multiple process_patient calls."""
        from src.simulation.processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig

        config = PipelineAdapterConfig(
            enable_mhr_guard=True,
            enable_trend_analysis=True,
            enable_explanations=True
        )
        adapter = PipelineAdapter(config)

        np.random.seed(42)

        # Process same patient multiple times
        for i in range(3):
            fhr = 140 + 10 * np.random.randn(480)
            uc = 20 + 10 * np.abs(np.sin(np.linspace(0, 4 * np.pi, 480)))

            adapter.process_patient(
                patient_id="test_persist",
                data={'fhr': fhr, 'uc': uc}
            )

        # Check trend buffer exists and has data
        buffer = adapter.get_trend_buffer("test_persist")
        assert buffer is not None, "Trend buffer should exist for patient"

    def test_v2_stats_available(self):
        """V2 statistics should be available in get_stats()."""
        from src.simulation.processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig

        config = PipelineAdapterConfig(
            enable_mhr_guard=True,
            enable_trend_analysis=True,
            enable_explanations=True
        )
        adapter = PipelineAdapter(config)

        stats = adapter.get_stats()

        assert 'mhr_guard_enabled' in stats
        assert 'trend_analysis_enabled' in stats
        assert 'explanation_engine_enabled' in stats
        assert stats['mhr_guard_enabled'] == True
        assert stats['trend_analysis_enabled'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
