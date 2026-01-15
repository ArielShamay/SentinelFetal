"""
Unit Tests for Simulation Core Components.

Tests the fundamental building blocks of the real-time simulator:
- RingBuffer: Fixed-size circular buffer
- Event Types: Enum and parameter dataclasses
- FHR Generator: Heart rate signal generation
- UC Generator: Uterine contraction signal generation
- Patient Generator: Combined generation and coordination

Test Strategy:
- Verify component instantiation
- Check signal generation produces valid physiological values
- Confirm buffer behavior (append, wrap-around, retrieval)
- Validate event injection affects signal output

References:
    - SentinelFetal Real-Time Simulator SPEC Part 1 & 2
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.core.ring_buffer import RingBuffer
from src.simulation.events.event_types import (
    EventType,
    EventSeverity,
    LateDecelerationParams,
    VariableDecelerationParams,
    BradycardiaParams,
    VariabilityParams,
    InjectedEvent,
)
from src.simulation.generators.fhr_generator import FHRGenerator, FHRGeneratorConfig
from src.simulation.generators.uc_generator import UCGenerator, UCGeneratorConfig
from src.simulation.generators.patient_generator import PatientGenerator, PatientConfig


# =============================================================================
# RingBuffer Tests
# =============================================================================

class TestRingBuffer:
    """Tests for RingBuffer class."""
    
    def test_buffer_creation(self):
        """Test buffer can be created with default config."""
        buffer = RingBuffer()
        assert buffer.max_samples == 2400  # 10 min at 4 Hz
        assert buffer.sampling_rate == 4.0
        assert buffer.size == 0
        assert buffer.is_empty
        assert not buffer.is_full
    
    def test_buffer_append_single(self):
        """Test appending single samples."""
        buffer = RingBuffer(max_samples=100)
        
        buffer.append(140.0, 20.0, 0.0)
        assert buffer.size == 1
        
        buffer.append(142.0, 22.0, 0.25)
        assert buffer.size == 2
        
        latest = buffer.get_latest()
        assert latest['fhr'] == 142.0
        assert latest['uc'] == 22.0
        assert latest['timestamp'] == 0.25
    
    def test_buffer_append_batch(self):
        """Test appending batch of samples."""
        buffer = RingBuffer(max_samples=100)
        
        fhr = np.array([140, 141, 142, 143], dtype=float)
        uc = np.array([20, 21, 22, 23], dtype=float)
        ts = np.array([0.0, 0.25, 0.5, 0.75])
        
        buffer.append_batch(fhr, uc, ts)
        assert buffer.size == 4
        assert buffer.duration_seconds == 1.0
    
    def test_buffer_wrap_around(self):
        """Test buffer correctly wraps when full."""
        buffer = RingBuffer(max_samples=10)
        
        # Add 15 samples (should keep only last 10)
        for i in range(15):
            buffer.append(float(100 + i), float(10 + i), float(i))
        
        assert buffer.size == 10
        assert buffer.is_full
        
        # Check that oldest samples were discarded
        data = buffer.get_window()
        assert data['fhr'][0] == 105.0  # First kept sample
        assert data['fhr'][-1] == 114.0  # Last sample
    
    def test_buffer_get_window_duration(self):
        """Test retrieving specific duration from buffer."""
        buffer = RingBuffer(max_samples=100, sampling_rate=4.0)
        
        # Add 10 seconds of data (40 samples)
        fhr = np.full(40, 140.0)
        uc = np.full(40, 20.0)
        ts = np.arange(40) * 0.25
        
        buffer.append_batch(fhr, uc, ts)
        
        # Get last 5 seconds (20 samples)
        data = buffer.get_window(duration_seconds=5.0)
        assert len(data['fhr']) == 20
        assert data['duration_seconds'] == 5.0
    
    def test_buffer_clear(self):
        """Test buffer clear."""
        buffer = RingBuffer(max_samples=100)
        buffer.append(140.0, 20.0, 0.0)
        buffer.append(140.0, 20.0, 0.25)
        
        buffer.clear()
        assert buffer.size == 0
        assert buffer.is_empty


# =============================================================================
# Event Types Tests
# =============================================================================

class TestEventTypes:
    """Tests for event type definitions."""
    
    def test_event_type_enum(self):
        """Test EventType enum has expected values."""
        assert EventType.LATE_DECELERATION is not None
        assert EventType.VARIABLE_DECELERATION is not None
        assert EventType.BRADYCARDIA is not None
        assert EventType.SINUSOIDAL_PATTERN is not None
        assert EventType.TACHYSYSTOLE is not None
    
    def test_late_decel_params_factory(self):
        """Test LateDecelerationParams factory methods."""
        mild = LateDecelerationParams.mild()
        assert mild.depth_bpm == 20.0
        assert mild.severity == EventSeverity.MILD
        
        severe = LateDecelerationParams.severe()
        assert severe.depth_bpm == 50.0
        assert severe.severity == EventSeverity.SEVERE
        assert severe.recurrence_rate == 0.8
    
    def test_variable_decel_severity_signs(self):
        """Test VariableDecelerationParams severity signs."""
        severe = VariableDecelerationParams.severe()
        assert severe.drops_below_70 is True
        assert severe.slow_recovery is True
    
    def test_injected_event_lifecycle(self):
        """Test InjectedEvent creation and properties."""
        params = LateDecelerationParams.moderate()
        event = InjectedEvent(
            event_type=EventType.LATE_DECELERATION,
            params=params,
            patient_id='P1',
            start_time=100.0,
            end_time=400.0,
            is_active=True
        )
        
        assert event.duration_seconds == 300.0
        assert event.is_expired(350.0) is False
        assert event.is_expired(450.0) is True
        assert event.progress(250.0) == 0.5


# =============================================================================
# FHR Generator Tests
# =============================================================================

class TestFHRGenerator:
    """Tests for FHR signal generator."""
    
    def test_fhr_generator_creation(self):
        """Test FHR generator can be created."""
        generator = FHRGenerator()
        assert generator.config.baseline_fhr == 140.0
        assert generator.config.sampling_rate == 4.0
    
    def test_fhr_generates_samples(self):
        """Test FHR generator produces samples."""
        generator = FHRGenerator()
        fhr = generator.generate_samples(4)
        
        assert len(fhr) == 4
        assert fhr.dtype == np.float64
    
    def test_fhr_values_physiological(self):
        """Test FHR values are within physiological range (50-240)."""
        generator = FHRGenerator()
        
        # Generate 10 seconds of data
        fhr = generator.generate_samples(40)
        
        assert np.all(fhr >= 50.0)
        assert np.all(fhr <= 240.0)
    
    def test_fhr_near_baseline(self):
        """Test FHR values are near configured baseline."""
        config = FHRGeneratorConfig(baseline_fhr=145.0)
        generator = FHRGenerator(config)
        
        # Generate 10 seconds of data
        fhr = generator.generate_samples(40)
        
        # Mean should be near baseline (within variability + noise)
        mean_fhr = np.mean(fhr)
        assert abs(mean_fhr - 145.0) < 20.0
    
    def test_fhr_has_variability(self):
        """Test FHR signal has variability (not flat)."""
        generator = FHRGenerator()
        fhr = generator.generate_samples(40)
        
        # Standard deviation should be > 0 (signal varies)
        assert np.std(fhr) > 0.5
    
    def test_fhr_state_continuity(self):
        """Test FHR generator maintains state across calls."""
        generator = FHRGenerator()
        
        fhr1 = generator.generate_samples(4)
        time1 = generator.current_time
        
        fhr2 = generator.generate_samples(4)
        time2 = generator.current_time
        
        # Time should advance
        assert time2 > time1


# =============================================================================
# UC Generator Tests
# =============================================================================

class TestUCGenerator:
    """Tests for UC signal generator."""
    
    def test_uc_generator_creation(self):
        """Test UC generator can be created."""
        generator = UCGenerator()
        assert generator.config.contractions_per_10min == 4.0
        assert generator.config.sampling_rate == 4.0
    
    def test_uc_generates_samples(self):
        """Test UC generator produces samples."""
        generator = UCGenerator()
        uc, peaks = generator.generate_samples(4)
        
        assert len(uc) == 4
        assert len(peaks) == 4
        assert uc.dtype == np.float64
        assert peaks.dtype == bool
    
    def test_uc_values_in_range(self):
        """Test UC values are within valid range (0-100)."""
        generator = UCGenerator()
        
        # Generate 10 seconds of data
        uc, _ = generator.generate_samples(40)
        
        assert np.all(uc >= 0)
        assert np.all(uc <= 100)
    
    def test_uc_has_baseline_tonus(self):
        """Test UC signal has baseline tonus when no contraction."""
        config = UCGeneratorConfig(baseline_tonus=12.0)
        generator = UCGenerator(config)
        
        # Generate short segment (unlikely to have contraction)
        uc, _ = generator.generate_samples(4)
        
        # Should be near baseline tonus (with noise)
        assert np.mean(uc) < 30  # Not in a contraction


# =============================================================================
# Patient Generator Tests
# =============================================================================

class TestPatientGenerator:
    """Tests for combined patient generator."""
    
    def test_patient_generator_creation(self):
        """Test patient generator can be created."""
        config = PatientConfig(
            patient_id='P1',
            bed_number=1,
            name='Test Patient'
        )
        patient = PatientGenerator(config)
        
        assert patient.patient_id == 'P1'
        assert patient.config.bed_number == 1
        assert patient.latest_category == 1  # Default normal
    
    def test_patient_generate_tick(self):
        """Test patient generator produces data tick."""
        config = PatientConfig(patient_id='P1', bed_number=1)
        patient = PatientGenerator(config)
        
        data = patient.generate_tick(4)  # 1 second
        
        assert 'fhr' in data
        assert 'uc' in data
        assert 'timestamps' in data
        assert 'contraction_peaks' in data
        
        assert len(data['fhr']) == 4
        assert len(data['uc']) == 4
    
    def test_patient_buffer_fills(self):
        """Test patient buffer accumulates data from ticks."""
        config = PatientConfig(patient_id='P1', bed_number=1)
        patient = PatientGenerator(config)
        
        # Generate 10 ticks (10 seconds at 4 samples each)
        for _ in range(10):
            patient.generate_tick(4)
        
        assert patient.buffer.size == 40
        assert patient.buffer.duration_seconds == 10.0
    
    def test_patient_fhr_values_physiological(self):
        """Test patient FHR values are within physiological range."""
        config = PatientConfig(patient_id='P1', bed_number=1)
        patient = PatientGenerator(config)
        
        # Generate 10 seconds
        for _ in range(10):
            data = patient.generate_tick(4)
            
            # Check each tick
            assert np.all(data['fhr'] >= 50.0), f"FHR below 50: {data['fhr'].min()}"
            assert np.all(data['fhr'] <= 240.0), f"FHR above 240: {data['fhr'].max()}"
    
    def test_patient_event_injection(self):
        """Test patient can receive injected events."""
        config = PatientConfig(patient_id='P1', bed_number=1)
        patient = PatientGenerator(config)
        
        # Inject a bradycardia event
        params = BradycardiaParams.mild()
        event = patient.inject_event(
            EventType.BRADYCARDIA,
            params,
            duration_seconds=60.0
        )
        
        assert event is not None
        assert len(patient.get_active_events()) == 1
        assert event.event_type == EventType.BRADYCARDIA
    
    def test_patient_get_buffer_data(self):
        """Test patient returns buffer data with metadata."""
        config = PatientConfig(
            patient_id='P1',
            bed_number=3,
            name='Test Patient'
        )
        patient = PatientGenerator(config)
        
        # Generate some data
        patient.generate_tick(4)
        
        data = patient.get_buffer_data()
        
        assert data['patient_id'] == 'P1'
        assert data['bed_number'] == 3
        assert data['name'] == 'Test Patient'
        assert 'fhr' in data
        assert 'uc' in data
    
    def test_patient_get_status(self):
        """Test patient status summary."""
        config = PatientConfig(patient_id='P1', bed_number=1)
        patient = PatientGenerator(config)
        
        patient.generate_tick(4)
        status = patient.get_status()
        
        assert status['patient_id'] == 'P1'
        assert status['bed_number'] == 1
        assert status['category'] == 1
        assert 'buffer_size' in status
        assert 'simulation_time' in status
    
    def test_patient_reset(self):
        """Test patient reset clears all state."""
        config = PatientConfig(patient_id='P1', bed_number=1)
        patient = PatientGenerator(config)
        
        # Generate data and inject event
        patient.generate_tick(4)
        patient.inject_event(EventType.BRADYCARDIA, BradycardiaParams.mild())
        
        # Reset
        patient.reset()
        
        assert patient.buffer.size == 0
        assert len(patient.get_active_events()) == 0
        assert patient.simulation_time == 0.0
        assert patient.latest_category == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestSimulationIntegration:
    """Integration tests for simulation components working together."""
    
    def test_multiple_patients(self):
        """Test creating and running multiple patients."""
        patients = []
        for i in range(4):
            config = PatientConfig(
                patient_id=f'P{i+1}',
                bed_number=i + 1,
                baseline_fhr=135 + i * 5  # Vary baselines
            )
            patients.append(PatientGenerator(config))
        
        # Generate 10 ticks for all patients
        for _ in range(10):
            for patient in patients:
                data = patient.generate_tick(4)
                # Verify each patient produces valid data
                assert len(data['fhr']) == 4
                assert np.all(data['fhr'] >= 50)
                assert np.all(data['fhr'] <= 240)
    
    def test_event_affects_signal(self):
        """Test that injecting an event modifies the signal."""
        config = PatientConfig(
            patient_id='P1',
            bed_number=1,
            baseline_fhr=140.0
        )
        patient = PatientGenerator(config)
        
        # Generate baseline data
        for _ in range(5):
            patient.generate_tick(4)
        
        baseline_data = patient.get_buffer_data()
        baseline_mean = np.mean(baseline_data['fhr'])
        
        # Clear and inject severe bradycardia
        patient.reset()
        patient.inject_event(
            EventType.BRADYCARDIA,
            BradycardiaParams.severe(),  # Target ~80 bpm
            duration_seconds=120.0
        )
        
        # Generate more data with event active
        for _ in range(60):  # 60 seconds to let event fully develop
            patient.generate_tick(4)
        
        event_data = patient.get_buffer_data()
        event_mean = np.mean(event_data['fhr'])
        
        # Event should lower the mean FHR significantly
        # (baseline ~140, bradycardia ~80, so event mean should be lower)
        assert event_mean < baseline_mean
    
    def test_long_simulation(self):
        """Test simulation can run for extended period without issues."""
        config = PatientConfig(patient_id='P1', bed_number=1)
        patient = PatientGenerator(config)
        
        # Simulate 5 minutes (300 seconds = 1200 samples)
        for _ in range(300):
            data = patient.generate_tick(4)
            
            # Quick sanity checks
            assert len(data['fhr']) == 4
            assert np.all(np.isfinite(data['fhr']))
        
        # Buffer should be at capacity (10 min = 2400 samples max)
        # but we only generated 5 min = 1200 samples
        assert patient.buffer.size == 1200
        assert abs(patient.simulation_time - 300.0) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
