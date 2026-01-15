"""
Event Type Definitions - All injectable clinical events.

Defines enums and parameter dataclasses for CTG simulation events
including decelerations, baseline changes, variability changes,
sinusoidal patterns, and tachysystole.

References:
    - Israeli Position Paper on CTG Interpretation
    - SentinelFetal Real-Time Simulator SPEC Part 1, Section 3.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any


class EventType(Enum):
    """
    All injectable event types for CTG simulation.
    
    Categories:
        Decelerations: LATE, VARIABLE, PROLONGED, EARLY
        Baseline: BRADYCARDIA, TACHYCARDIA
        Variability: ABSENT, MINIMAL, MARKED
        Patterns: SINUSOIDAL
        Uterine: TACHYSYSTOLE
    """
    # Decelerations
    LATE_DECELERATION = auto()
    VARIABLE_DECELERATION = auto()
    PROLONGED_DECELERATION = auto()
    EARLY_DECELERATION = auto()
    
    # Baseline changes
    BRADYCARDIA = auto()
    TACHYCARDIA = auto()
    
    # Variability changes
    ABSENT_VARIABILITY = auto()
    MINIMAL_VARIABILITY = auto()
    MARKED_VARIABILITY = auto()
    
    # Special patterns
    SINUSOIDAL_PATTERN = auto()
    
    # Uterine activity
    TACHYSYSTOLE = auto()


class EventSeverity(Enum):
    """Severity levels for events."""
    MILD = auto()
    MODERATE = auto()
    SEVERE = auto()


@dataclass
class EventParameters:
    """
    Base parameters for all events.
    
    Attributes:
        duration_seconds: How long the event lasts (default: 5 minutes).
        severity: Severity level (MILD/MODERATE/SEVERE).
        recurrence_rate: For decelerations, fraction of contractions affected.
    """
    duration_seconds: float = 300.0  # 5 minutes default
    severity: EventSeverity = EventSeverity.MODERATE
    recurrence_rate: float = 0.6  # 60% of contractions


# =============================================================================
# Deceleration Parameters
# =============================================================================

@dataclass
class LateDecelerationParams(EventParameters):
    """
    Late deceleration parameters.
    
    Late decelerations are gradual decreases in FHR that begin after
    the contraction peak, with nadir occurring after the peak.
    
    Attributes:
        depth_bpm: Drop below baseline in bpm.
        lag_seconds: Time from contraction peak to deceleration nadir.
        recovery_seconds: Time to return to baseline.
    """
    depth_bpm: float = 30.0
    lag_seconds: float = 20.0
    recovery_seconds: float = 30.0
    
    @classmethod
    def mild(cls) -> LateDecelerationParams:
        """Create mild late deceleration parameters."""
        return cls(
            depth_bpm=20.0,
            lag_seconds=15.0,
            recovery_seconds=20.0,
            severity=EventSeverity.MILD,
            recurrence_rate=0.3
        )
    
    @classmethod
    def moderate(cls) -> LateDecelerationParams:
        """Create moderate late deceleration parameters."""
        return cls(
            depth_bpm=30.0,
            lag_seconds=20.0,
            recovery_seconds=30.0,
            severity=EventSeverity.MODERATE,
            recurrence_rate=0.6
        )
    
    @classmethod
    def severe(cls) -> LateDecelerationParams:
        """Create severe late deceleration parameters."""
        return cls(
            depth_bpm=50.0,
            lag_seconds=25.0,
            recovery_seconds=45.0,
            severity=EventSeverity.SEVERE,
            recurrence_rate=0.8
        )


@dataclass
class VariableDecelerationParams(EventParameters):
    """
    Variable deceleration parameters.
    
    Variable decelerations have abrupt onset and variable timing
    relative to contractions.
    
    Attributes:
        depth_bpm: Drop below baseline in bpm.
        duration_decel_seconds: Duration of each deceleration.
        has_shoulders: Whether to add acceleration shoulders.
        
    Severity Signs (Category 3 indicators):
        drops_below_70: FHR drops below 70 bpm.
        absent_internal_variability: No variability within decel.
        slow_recovery: Recovery takes > 60 seconds.
        overshoot: FHR rises > 10 bpm above baseline after recovery.
        biphasic_w_shape: W-shaped pattern.
    """
    depth_bpm: float = 40.0
    duration_decel_seconds: float = 45.0
    has_shoulders: bool = True
    
    # Severity signs (Category 3 indicators)
    drops_below_70: bool = False
    absent_internal_variability: bool = False
    slow_recovery: bool = False
    overshoot: bool = False
    biphasic_w_shape: bool = False
    
    @classmethod
    def mild(cls) -> VariableDecelerationParams:
        """Create mild variable deceleration parameters."""
        return cls(
            depth_bpm=25.0,
            duration_decel_seconds=30.0,
            severity=EventSeverity.MILD,
            recurrence_rate=0.3
        )
    
    @classmethod
    def moderate(cls) -> VariableDecelerationParams:
        """Create moderate variable deceleration parameters."""
        return cls(
            depth_bpm=40.0,
            duration_decel_seconds=45.0,
            severity=EventSeverity.MODERATE,
            recurrence_rate=0.5
        )
    
    @classmethod
    def severe(cls) -> VariableDecelerationParams:
        """Create severe variable deceleration with severity signs."""
        return cls(
            depth_bpm=60.0,
            duration_decel_seconds=60.0,
            drops_below_70=True,
            slow_recovery=True,
            severity=EventSeverity.SEVERE,
            recurrence_rate=0.7
        )


@dataclass
class ProlongedDecelerationParams(EventParameters):
    """
    Prolonged deceleration parameters.
    
    Prolonged decelerations last > 2 minutes but < 10 minutes.
    """
    depth_bpm: float = 40.0
    duration_seconds: float = 180.0  # 3 minutes
    
    @classmethod
    def moderate(cls) -> ProlongedDecelerationParams:
        """Create moderate prolonged deceleration."""
        return cls(
            depth_bpm=40.0,
            duration_seconds=180.0,
            severity=EventSeverity.MODERATE
        )
    
    @classmethod
    def severe(cls) -> ProlongedDecelerationParams:
        """Create severe prolonged deceleration."""
        return cls(
            depth_bpm=60.0,
            duration_seconds=300.0,
            severity=EventSeverity.SEVERE
        )


@dataclass
class EarlyDecelerationParams(EventParameters):
    """
    Early deceleration parameters.
    
    Early decelerations are gradual, symmetrical, and coincide
    with contractions (nadir at contraction peak).
    """
    depth_bpm: float = 20.0
    
    @classmethod
    def typical(cls) -> EarlyDecelerationParams:
        """Create typical early deceleration parameters."""
        return cls(
            depth_bpm=20.0,
            severity=EventSeverity.MILD,
            recurrence_rate=0.5
        )


# =============================================================================
# Baseline Change Parameters
# =============================================================================

@dataclass
class BradycardiaParams(EventParameters):
    """
    Bradycardia episode parameters.
    
    Bradycardia: FHR < 110 bpm for ≥ 10 minutes.
    
    Attributes:
        target_fhr: Target FHR to reach in bpm.
        onset_type: 'gradual' or 'sudden'.
    """
    target_fhr: float = 100.0
    onset_type: str = 'gradual'
    duration_seconds: float = 180.0
    
    @classmethod
    def mild(cls) -> BradycardiaParams:
        """Create mild bradycardia (100-110 bpm)."""
        return cls(
            target_fhr=105.0,
            duration_seconds=120.0,
            severity=EventSeverity.MILD
        )
    
    @classmethod
    def moderate(cls) -> BradycardiaParams:
        """Create moderate bradycardia (90-100 bpm)."""
        return cls(
            target_fhr=95.0,
            duration_seconds=180.0,
            severity=EventSeverity.MODERATE
        )
    
    @classmethod
    def severe(cls) -> BradycardiaParams:
        """Create severe bradycardia (< 80 bpm)."""
        return cls(
            target_fhr=80.0,
            duration_seconds=300.0,
            severity=EventSeverity.SEVERE
        )


@dataclass
class TachycardiaParams(EventParameters):
    """
    Tachycardia episode parameters.
    
    Tachycardia: FHR > 160 bpm for ≥ 10 minutes.
    
    Attributes:
        target_fhr: Target FHR to reach in bpm.
    """
    target_fhr: float = 170.0
    duration_seconds: float = 300.0
    
    @classmethod
    def mild(cls) -> TachycardiaParams:
        """Create mild tachycardia (160-170 bpm)."""
        return cls(
            target_fhr=165.0,
            duration_seconds=180.0,
            severity=EventSeverity.MILD
        )
    
    @classmethod
    def moderate(cls) -> TachycardiaParams:
        """Create moderate tachycardia (170-180 bpm)."""
        return cls(
            target_fhr=175.0,
            duration_seconds=300.0,
            severity=EventSeverity.MODERATE
        )
    
    @classmethod
    def severe(cls) -> TachycardiaParams:
        """Create severe tachycardia (> 180 bpm)."""
        return cls(
            target_fhr=185.0,
            duration_seconds=600.0,
            severity=EventSeverity.SEVERE
        )


# =============================================================================
# Variability Change Parameters
# =============================================================================

@dataclass
class VariabilityParams(EventParameters):
    """
    Variability change parameters.
    
    Changes baseline variability to simulate:
        - Absent variability (≤ 2 bpm) - SEVERE finding
        - Minimal variability (3-5 bpm) - Concerning
        - Marked variability (> 25 bpm) - May indicate hypoxia
    
    Attributes:
        target_variability_bpm: Target variability amplitude in bpm.
    """
    target_variability_bpm: float = 2.0
    duration_seconds: float = 300.0
    
    @classmethod
    def absent(cls) -> VariabilityParams:
        """Create absent variability (≤ 2 bpm) - SEVERE."""
        return cls(
            target_variability_bpm=1.0,
            duration_seconds=300.0,
            severity=EventSeverity.SEVERE
        )
    
    @classmethod
    def minimal(cls) -> VariabilityParams:
        """Create minimal variability (3-5 bpm)."""
        return cls(
            target_variability_bpm=4.0,
            duration_seconds=600.0,
            severity=EventSeverity.MODERATE
        )
    
    @classmethod
    def marked(cls) -> VariabilityParams:
        """Create marked variability (> 25 bpm)."""
        return cls(
            target_variability_bpm=30.0,
            duration_seconds=300.0,
            severity=EventSeverity.MODERATE
        )


# =============================================================================
# Special Pattern Parameters
# =============================================================================

@dataclass
class SinusoidalParams(EventParameters):
    """
    Sinusoidal pattern parameters.
    
    Sinusoidal pattern: Smooth, sine wave-like oscillation.
    ALWAYS Category 3 (Pathological) - indicates severe fetal anemia.
    
    Characteristics:
        - Frequency: 3-5 cycles per minute
        - Amplitude: 5-15 bpm
        - Duration: > 20 minutes
        - Absent short-term variability (smooth waves)
    
    Attributes:
        frequency_cycles_per_min: Oscillation frequency.
        amplitude_bpm: Peak-to-peak amplitude.
    """
    frequency_cycles_per_min: float = 4.0
    amplitude_bpm: float = 10.0
    duration_seconds: float = 1200.0  # Must be > 20 min
    severity: EventSeverity = EventSeverity.SEVERE  # Always severe
    
    @classmethod
    def typical(cls) -> SinusoidalParams:
        """Create typical sinusoidal pattern."""
        return cls(
            frequency_cycles_per_min=4.0,
            amplitude_bpm=10.0,
            duration_seconds=1200.0,
            severity=EventSeverity.SEVERE
        )


# =============================================================================
# Uterine Activity Parameters
# =============================================================================

@dataclass
class TachysystoleParams(EventParameters):
    """
    Tachysystole parameters.
    
    Tachysystole: > 5 contractions per 10-minute window.
    Can reduce blood flow to fetus and may cause hypoxia.
    
    Attributes:
        contractions_per_10min: Target contraction rate.
    """
    contractions_per_10min: int = 6
    duration_seconds: float = 600.0  # 10 minutes
    
    @classmethod
    def mild(cls) -> TachysystoleParams:
        """Create mild tachysystole (6 contractions/10min)."""
        return cls(
            contractions_per_10min=6,
            duration_seconds=600.0,
            severity=EventSeverity.MILD
        )
    
    @classmethod
    def severe(cls) -> TachysystoleParams:
        """Create severe tachysystole (8+ contractions/10min)."""
        return cls(
            contractions_per_10min=8,
            duration_seconds=600.0,
            severity=EventSeverity.SEVERE
        )


# =============================================================================
# Injected Event Tracking
# =============================================================================

@dataclass
class InjectedEvent:
    """
    Represents an actively injected event.
    
    Tracks the lifecycle of an injected clinical event from start to end.
    
    Attributes:
        event_type: Type of the event (from EventType enum).
        params: Event-specific parameters.
        patient_id: ID of the patient receiving this event.
        start_time: Simulation time when event started (seconds).
        end_time: Simulation time when event ends (seconds).
        is_active: Whether the event is currently active.
    """
    event_type: EventType
    params: EventParameters
    patient_id: str
    start_time: float
    end_time: float
    is_active: bool = True
    
    @property
    def remaining_seconds(self) -> float:
        """Calculate remaining duration."""
        return max(0.0, self.end_time - self.start_time)
    
    @property
    def duration_seconds(self) -> float:
        """Total duration of the event."""
        return self.end_time - self.start_time
    
    def is_expired(self, current_time: float) -> bool:
        """Check if event has expired."""
        return current_time > self.end_time
    
    def progress(self, current_time: float) -> float:
        """Get progress through event (0.0 to 1.0)."""
        if current_time <= self.start_time:
            return 0.0
        if current_time >= self.end_time:
            return 1.0
        elapsed = current_time - self.start_time
        duration = self.end_time - self.start_time
        return elapsed / duration if duration > 0 else 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_type': self.event_type.name,
            'patient_id': self.patient_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration_seconds,
            'severity': self.params.severity.name if hasattr(self.params, 'severity') else None,
            'is_active': self.is_active
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"InjectedEvent({self.event_type.name}, "
            f"patient={self.patient_id}, "
            f"duration={self.duration_seconds:.0f}s, "
            f"active={self.is_active})"
        )
