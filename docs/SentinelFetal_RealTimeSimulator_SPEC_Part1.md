# SentinelFetal Real-Time Simulator
## Technical Specification Document (SPEC) - Part 1

**Version:** 1.0  
**Date:** January 2026  
**Project:** SentinelFetal Gen3.5 - Real-Time Simulation Module

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT APPLICATION                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      SimulationControlPanel                          │    │
│  │  [Speed] [Pause/Resume] [Reset] [Event Injection Controls]          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      SimulationOrchestrator                          │    │
│  │  - Manages all patient generators                                    │    │
│  │  - Schedules MOMENT processing                                       │    │
│  │  - Coordinates event injection                                       │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│         ┌───────────────────────┼───────────────────────┐                   │
│         ▼                       ▼                       ▼                   │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐           │
│  │  Patient 1  │         │  Patient 2  │   ...   │  Patient 8  │           │
│  │  Generator  │         │  Generator  │         │  Generator  │           │
│  └──────┬──────┘         └──────┬──────┘         └──────┬──────┘           │
│         │                       │                       │                   │
│         └───────────────────────┼───────────────────────┘                   │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      ProcessingPipeline                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │    │
│  │  │Preprocess│→ │Rule Eng. │→ │ MOMENT   │→ │Classifier│→ Alert      │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      LiveDashboard                                   │    │
│  │  [Patient Overview] [Selected Patient CTG] [Alerts] [Findings]      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Processing Timeline

```
Time →  0s      1s      2s      3s      4s      5s      6s      7s      8s
        │       │       │       │       │       │       │       │       │
Gen:    ████████████████████████████████████████████████████████████████████
        (All 8 generators produce 4 samples each second continuously)
        
Rules:  ▓▓▓     ▓▓▓     ▓▓▓     ▓▓▓     ▓▓▓     ▓▓▓     ▓▓▓     ▓▓▓     ▓▓▓
        (Rule engine processes all 8 patients every second - FAST)
        
MOMENT: P1              P2              P3              P4
        (One patient every 4 seconds - staggered to avoid CPU spikes)
        
UI:     ░░░     ░░░     ░░░     ░░░     ░░░     ░░░     ░░░     ░░░     ░░░
        (Dashboard updates every second with latest data)
```

---

## 2. Directory Structure

### 2.1 New Files to Create

```
src/
├── simulation/                          # NEW MODULE
│   ├── __init__.py                      # Module exports
│   ├── config.py                        # Simulation constants
│   │
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── fhr_generator.py             # FHR signal generation
│   │   ├── uc_generator.py              # Uterine contractions
│   │   └── patient_generator.py         # Combined patient generator
│   │
│   ├── events/
│   │   ├── __init__.py
│   │   ├── event_types.py               # Event definitions (Enum + dataclasses)
│   │   └── event_injector.py            # Event injection logic
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ring_buffer.py               # Circular buffer for signals
│   │   ├── patient_state.py             # Patient state management
│   │   └── orchestrator.py              # Main simulation coordinator
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   └── pipeline_adapter.py          # Adapts to existing pipeline
│   │
│   └── logging/
│       ├── __init__.py
│       └── event_logger.py              # Lightweight event logging
│
├── ui/
│   ├── simulation_app.py                # NEW: Simulation dashboard
│   └── components/
│       ├── __init__.py
│       ├── control_panel.py             # NEW: Simulation controls
│       ├── patient_overview.py          # NEW: Multi-patient grid
│       └── event_injector_ui.py         # NEW: Event injection UI
```

---

## 3. Core Components

### 3.1 Ring Buffer (`src/simulation/core/ring_buffer.py`)

```python
"""
Ring Buffer - Fixed-size circular buffer for CTG signals.

Purpose: Store last 10 minutes of data per patient without memory growth.
Memory: ~77KB for 8 patients (2400 samples × 8 × 4 bytes × 2 channels)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import deque


@dataclass
class RingBuffer:
    """
    Fixed-size circular buffer for CTG signals.
    
    Attributes:
        max_samples: Maximum samples to store (2400 = 10 min at 4Hz)
        sampling_rate: Sampling frequency in Hz (4.0)
    """
    max_samples: int = 2400
    sampling_rate: float = 4.0
    
    _fhr: deque = field(default_factory=lambda: deque(maxlen=2400))
    _uc: deque = field(default_factory=lambda: deque(maxlen=2400))
    _timestamps: deque = field(default_factory=lambda: deque(maxlen=2400))
    
    def __post_init__(self):
        """Initialize deques with correct maxlen."""
        self._fhr = deque(maxlen=self.max_samples)
        self._uc = deque(maxlen=self.max_samples)
        self._timestamps = deque(maxlen=self.max_samples)
    
    def append(self, fhr: float, uc: float, timestamp: float) -> None:
        """Add a single sample to the buffer."""
        self._fhr.append(fhr)
        self._uc.append(uc)
        self._timestamps.append(timestamp)
    
    def append_batch(self, fhr: np.ndarray, uc: np.ndarray, 
                     timestamps: np.ndarray) -> None:
        """Add multiple samples to the buffer."""
        for f, u, t in zip(fhr, uc, timestamps):
            self.append(f, u, t)
    
    def get_window(self, duration_seconds: Optional[float] = None) -> dict:
        """
        Get data from the buffer.
        
        Args:
            duration_seconds: Seconds of data to return (None = all)
            
        Returns:
            dict with 'fhr', 'uc', 'timestamps' as numpy arrays
        """
        if duration_seconds is None:
            samples = self.max_samples
        else:
            samples = min(int(duration_seconds * self.sampling_rate), 
                         len(self._fhr))
        
        return {
            'fhr': np.array(list(self._fhr)[-samples:]),
            'uc': np.array(list(self._uc)[-samples:]),
            'timestamps': np.array(list(self._timestamps)[-samples:]),
            'duration_seconds': samples / self.sampling_rate
        }
    
    def get_last_n_minutes(self, minutes: float) -> dict:
        """Convenience method for getting last N minutes."""
        return self.get_window(minutes * 60)
    
    def clear(self) -> None:
        """Clear all data from buffer."""
        self._fhr.clear()
        self._uc.clear()
        self._timestamps.clear()
    
    @property
    def size(self) -> int:
        """Current number of samples in buffer."""
        return len(self._fhr)
    
    @property
    def duration_seconds(self) -> float:
        """Current duration of data in buffer."""
        return self.size / self.sampling_rate
    
    @property
    def is_full(self) -> bool:
        """Whether buffer has reached max capacity."""
        return self.size >= self.max_samples
```

### 3.2 Event Types (`src/simulation/events/event_types.py`)

```python
"""
Event Type Definitions - All injectable clinical events.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class EventType(Enum):
    """All injectable event types."""
    LATE_DECELERATION = auto()
    VARIABLE_DECELERATION = auto()
    PROLONGED_DECELERATION = auto()
    EARLY_DECELERATION = auto()
    BRADYCARDIA = auto()
    TACHYCARDIA = auto()
    ABSENT_VARIABILITY = auto()
    MINIMAL_VARIABILITY = auto()
    MARKED_VARIABILITY = auto()
    SINUSOIDAL_PATTERN = auto()
    TACHYSYSTOLE = auto()


class EventSeverity(Enum):
    """Severity levels for events."""
    MILD = auto()
    MODERATE = auto()
    SEVERE = auto()


@dataclass
class EventParameters:
    """Base parameters for all events."""
    duration_seconds: float = 300.0  # 5 minutes default
    severity: EventSeverity = EventSeverity.MODERATE
    recurrence_rate: float = 0.6  # 60% of contractions


@dataclass
class LateDecelerationParams(EventParameters):
    """Late deceleration parameters."""
    depth_bpm: float = 30.0
    lag_seconds: float = 20.0
    recovery_seconds: float = 30.0
    
    @classmethod
    def mild(cls) -> 'LateDecelerationParams':
        return cls(depth_bpm=20, lag_seconds=15, recovery_seconds=20,
                   severity=EventSeverity.MILD, recurrence_rate=0.3)
    
    @classmethod
    def moderate(cls) -> 'LateDecelerationParams':
        return cls(depth_bpm=30, lag_seconds=20, recovery_seconds=30,
                   severity=EventSeverity.MODERATE, recurrence_rate=0.6)
    
    @classmethod
    def severe(cls) -> 'LateDecelerationParams':
        return cls(depth_bpm=50, lag_seconds=25, recovery_seconds=45,
                   severity=EventSeverity.SEVERE, recurrence_rate=0.8)


@dataclass
class VariableDecelerationParams(EventParameters):
    """Variable deceleration parameters."""
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
    def mild(cls) -> 'VariableDecelerationParams':
        return cls(depth_bpm=25, duration_decel_seconds=30,
                   severity=EventSeverity.MILD, recurrence_rate=0.3)
    
    @classmethod
    def moderate(cls) -> 'VariableDecelerationParams':
        return cls(depth_bpm=40, duration_decel_seconds=45,
                   severity=EventSeverity.MODERATE, recurrence_rate=0.5)
    
    @classmethod
    def severe(cls) -> 'VariableDecelerationParams':
        return cls(depth_bpm=60, duration_decel_seconds=60,
                   drops_below_70=True, slow_recovery=True,
                   severity=EventSeverity.SEVERE, recurrence_rate=0.7)


@dataclass
class BradycardiaParams(EventParameters):
    """Bradycardia episode parameters."""
    target_fhr: float = 100.0
    onset_type: str = 'gradual'  # 'gradual' or 'sudden'
    duration_seconds: float = 180.0
    
    @classmethod
    def mild(cls) -> 'BradycardiaParams':
        return cls(target_fhr=105, duration_seconds=120)
    
    @classmethod
    def severe(cls) -> 'BradycardiaParams':
        return cls(target_fhr=80, duration_seconds=300)


@dataclass
class TachycardiaParams(EventParameters):
    """Tachycardia episode parameters."""
    target_fhr: float = 170.0
    duration_seconds: float = 300.0
    
    @classmethod
    def mild(cls) -> 'TachycardiaParams':
        return cls(target_fhr=165, duration_seconds=180)
    
    @classmethod
    def moderate(cls) -> 'TachycardiaParams':
        return cls(target_fhr=175, duration_seconds=300)


@dataclass
class VariabilityParams(EventParameters):
    """Variability change parameters."""
    target_variability_bpm: float = 2.0
    duration_seconds: float = 300.0
    
    @classmethod
    def absent(cls) -> 'VariabilityParams':
        return cls(target_variability_bpm=1.0, duration_seconds=300)
    
    @classmethod
    def minimal(cls) -> 'VariabilityParams':
        return cls(target_variability_bpm=4.0, duration_seconds=600)
    
    @classmethod
    def marked(cls) -> 'VariabilityParams':
        return cls(target_variability_bpm=30.0, duration_seconds=300)


@dataclass
class SinusoidalParams(EventParameters):
    """Sinusoidal pattern parameters."""
    frequency_cycles_per_min: float = 4.0  # 3-5 cycles/min
    amplitude_bpm: float = 10.0  # 5-15 bpm
    duration_seconds: float = 1200.0  # Must be >20 min


@dataclass
class TachysystoleParams(EventParameters):
    """Tachysystole parameters."""
    contractions_per_10min: int = 6  # >5 is tachysystole
    duration_seconds: float = 600.0


@dataclass
class InjectedEvent:
    """Represents an injected event."""
    event_type: EventType
    params: EventParameters
    patient_id: str
    start_time: float
    end_time: float
    is_active: bool = True
    
    @property
    def remaining_seconds(self) -> float:
        return max(0, self.end_time - self.start_time)
    
    def to_dict(self) -> dict:
        return {
            'event_type': self.event_type.name,
            'patient_id': self.patient_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'severity': self.params.severity.name if hasattr(self.params, 'severity') else None,
            'is_active': self.is_active
        }
```

### 3.3 FHR Generator (`src/simulation/generators/fhr_generator.py`)

```python
"""
FHR Signal Generator - Generates realistic fetal heart rate signals.
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass

from ..events.event_types import (
    InjectedEvent, EventType, LateDecelerationParams,
    VariableDecelerationParams, BradycardiaParams, TachycardiaParams,
    VariabilityParams, SinusoidalParams
)


@dataclass
class FHRGeneratorConfig:
    """Configuration for FHR generation."""
    sampling_rate: float = 4.0
    baseline_fhr: float = 140.0
    baseline_variability: float = 10.0
    high_freq_noise_std: float = 2.0
    min_fhr: float = 50.0
    max_fhr: float = 240.0


class FHRGenerator:
    """
    Generates realistic FHR signals with:
    - Configurable baseline (default 140 bpm)
    - Moderate variability (5-25 bpm oscillations)
    - Support for injected events
    """
    
    def __init__(self, config: Optional[FHRGeneratorConfig] = None):
        self.config = config or FHRGeneratorConfig()
        self._phase = 0.0
        self._time = 0.0
    
    def generate_samples(
        self,
        n_samples: int,
        active_events: Optional[List[InjectedEvent]] = None,
        contraction_peaks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate FHR samples.
        
        Args:
            n_samples: Number of samples to generate
            active_events: Currently active injected events
            contraction_peaks: Boolean array marking contraction peaks
            
        Returns:
            numpy array of FHR values
        """
        active_events = active_events or []
        dt = 1.0 / self.config.sampling_rate
        times = self._time + np.arange(n_samples) * dt
        
        # Start with baseline
        fhr = np.full(n_samples, self.config.baseline_fhr)
        
        # Add variability
        variability = self._generate_variability(n_samples, active_events)
        fhr += variability
        
        # Add noise
        noise = np.random.normal(0, self.config.high_freq_noise_std, n_samples)
        fhr += noise
        
        # Apply events
        for event in active_events:
            fhr = self._apply_event(fhr, times, event, contraction_peaks)
        
        # Clip to physiological range
        fhr = np.clip(fhr, self.config.min_fhr, self.config.max_fhr)
        
        # Update state
        self._time = times[-1] + dt
        self._phase += n_samples * dt * 0.1
        
        return fhr
    
    def _generate_variability(
        self,
        n_samples: int,
        active_events: List[InjectedEvent]
    ) -> np.ndarray:
        """Generate baseline variability oscillations."""
        target_var = self.config.baseline_variability
        
        # Check for variability-modifying events
        for event in active_events:
            if event.event_type in [EventType.ABSENT_VARIABILITY,
                                     EventType.MINIMAL_VARIABILITY,
                                     EventType.MARKED_VARIABILITY]:
                target_var = event.params.target_variability_bpm
                break
        
        dt = 1.0 / self.config.sampling_rate
        t = np.arange(n_samples) * dt
        
        # Primary variability: 2-6 cycles per minute
        freq1 = 0.05 + 0.02 * np.sin(self._phase * 0.1)
        var1 = np.sin(2 * np.pi * freq1 * t + self._phase) * (target_var * 0.6)
        
        # Secondary variability
        freq2 = 0.15
        var2 = np.sin(2 * np.pi * freq2 * t + self._phase * 2) * (target_var * 0.3)
        
        return var1 + var2
    
    def _apply_event(
        self,
        fhr: np.ndarray,
        times: np.ndarray,
        event: InjectedEvent,
        contraction_peaks: Optional[np.ndarray]
    ) -> np.ndarray:
        """Apply an injected event to the FHR signal."""
        
        if event.event_type == EventType.LATE_DECELERATION:
            return self._apply_late_deceleration(fhr, times, event, contraction_peaks)
        elif event.event_type == EventType.VARIABLE_DECELERATION:
            return self._apply_variable_deceleration(fhr, times, event, contraction_peaks)
        elif event.event_type == EventType.BRADYCARDIA:
            return self._apply_bradycardia(fhr, times, event)
        elif event.event_type == EventType.TACHYCARDIA:
            return self._apply_tachycardia(fhr, times, event)
        elif event.event_type == EventType.SINUSOIDAL_PATTERN:
            return self._apply_sinusoidal(fhr, times, event)
        
        return fhr
    
    def _apply_late_deceleration(
        self, fhr: np.ndarray, times: np.ndarray,
        event: InjectedEvent, contraction_peaks: Optional[np.ndarray]
    ) -> np.ndarray:
        """Apply late deceleration pattern."""
        if contraction_peaks is None:
            return fhr
        
        params: LateDecelerationParams = event.params
        
        for i, is_peak in enumerate(contraction_peaks):
            if not is_peak or np.random.random() > params.recurrence_rate:
                continue
            
            peak_time = times[i]
            nadir_time = peak_time + params.lag_seconds
            decel_duration = params.recovery_seconds * 2
            decel_mask = (times >= peak_time) & (times <= peak_time + decel_duration)
            
            if np.any(decel_mask):
                t_rel = times[decel_mask] - nadir_time
                sigma = params.recovery_seconds / 2
                decel_shape = np.exp(-0.5 * (t_rel / sigma) ** 2)
                fhr[decel_mask] -= params.depth_bpm * decel_shape
        
        return fhr
    
    def _apply_variable_deceleration(
        self, fhr: np.ndarray, times: np.ndarray,
        event: InjectedEvent, contraction_peaks: Optional[np.ndarray]
    ) -> np.ndarray:
        """Apply variable deceleration pattern."""
        if contraction_peaks is None:
            return fhr
        
        params: VariableDecelerationParams = event.params
        
        for i, is_peak in enumerate(contraction_peaks):
            if not is_peak or np.random.random() > params.recurrence_rate:
                continue
            
            peak_time = times[i]
            offset = np.random.uniform(-10, 10)
            decel_start = peak_time + offset
            decel_end = decel_start + params.duration_decel_seconds
            decel_mask = (times >= decel_start) & (times <= decel_end)
            
            if np.any(decel_mask):
                t_rel = times[decel_mask] - decel_start
                duration = params.duration_decel_seconds
                
                # Sharp trapezoid shape
                shape = np.ones_like(t_rel)
                descent_mask = t_rel < 5
                if np.any(descent_mask):
                    shape[descent_mask] = t_rel[descent_mask] / 5
                
                recovery_time = 10 if params.slow_recovery else 5
                ascent_mask = t_rel > (duration - recovery_time)
                if np.any(ascent_mask):
                    shape[ascent_mask] = (duration - t_rel[ascent_mask]) / recovery_time
                
                depth = params.depth_bpm
                if params.drops_below_70:
                    depth = max(depth, self.config.baseline_fhr - 65)
                
                fhr[decel_mask] -= depth * shape
        
        return fhr
    
    def _apply_bradycardia(
        self, fhr: np.ndarray, times: np.ndarray, event: InjectedEvent
    ) -> np.ndarray:
        """Apply bradycardia (baseline shift down)."""
        params: BradycardiaParams = event.params
        event_mask = (times >= event.start_time) & (times <= event.end_time)
        
        if np.any(event_mask):
            reduction = self.config.baseline_fhr - params.target_fhr
            t_rel = times[event_mask] - event.start_time
            duration = event.end_time - event.start_time
            ramp_time = min(30, duration / 4)
            
            factor = np.ones_like(t_rel)
            factor[t_rel < ramp_time] = t_rel[t_rel < ramp_time] / ramp_time
            factor[t_rel > duration - ramp_time] = (duration - t_rel[t_rel > duration - ramp_time]) / ramp_time
            
            fhr[event_mask] -= reduction * factor
        
        return fhr
    
    def _apply_tachycardia(
        self, fhr: np.ndarray, times: np.ndarray, event: InjectedEvent
    ) -> np.ndarray:
        """Apply tachycardia (baseline shift up)."""
        params: TachycardiaParams = event.params
        event_mask = (times >= event.start_time) & (times <= event.end_time)
        
        if np.any(event_mask):
            increase = params.target_fhr - self.config.baseline_fhr
            t_rel = times[event_mask] - event.start_time
            duration = event.end_time - event.start_time
            ramp_time = min(60, duration / 4)
            
            factor = np.ones_like(t_rel)
            factor[t_rel < ramp_time] = t_rel[t_rel < ramp_time] / ramp_time
            factor[t_rel > duration - ramp_time] = (duration - t_rel[t_rel > duration - ramp_time]) / ramp_time
            
            fhr[event_mask] += increase * factor
        
        return fhr
    
    def _apply_sinusoidal(
        self, fhr: np.ndarray, times: np.ndarray, event: InjectedEvent
    ) -> np.ndarray:
        """Apply sinusoidal pattern."""
        params: SinusoidalParams = event.params
        event_mask = (times >= event.start_time) & (times <= event.end_time)
        
        if np.any(event_mask):
            t_rel = times[event_mask] - event.start_time
            freq = params.frequency_cycles_per_min / 60
            sinusoidal = np.sin(2 * np.pi * freq * t_rel) * params.amplitude_bpm
            fhr[event_mask] = self.config.baseline_fhr + sinusoidal
        
        return fhr
    
    def reset(self) -> None:
        """Reset generator state."""
        self._phase = 0.0
        self._time = 0.0
```

### 3.4 UC Generator (`src/simulation/generators/uc_generator.py`)

```python
"""
UC Generator - Generates realistic uterine contraction signals.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks

from ..events.event_types import InjectedEvent, EventType


@dataclass
class UCGeneratorConfig:
    """Configuration for UC generation."""
    sampling_rate: float = 4.0
    contractions_per_10min: float = 4.0
    contraction_duration_sec: float = 60.0
    contraction_amplitude: float = 80.0
    baseline_tonus: float = 10.0
    noise_std: float = 3.0


class UCGenerator:
    """
    Generates realistic uterine contraction signals with:
    - Configurable contraction frequency
    - Realistic contraction shape
    - Support for tachysystole events
    """
    
    def __init__(self, config: Optional[UCGeneratorConfig] = None):
        self.config = config or UCGeneratorConfig()
        self._time = 0.0
        self._next_contraction_time = np.random.uniform(30, 90)
        self._contraction_peaks: List[float] = []
    
    def generate_samples(
        self,
        n_samples: int,
        active_events: Optional[List[InjectedEvent]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate UC samples.
        
        Returns:
            Tuple of (uc_signal, contraction_peak_mask)
        """
        active_events = active_events or []
        dt = 1.0 / self.config.sampling_rate
        times = self._time + np.arange(n_samples) * dt
        
        # Check for tachysystole
        contractions_per_10min = self.config.contractions_per_10min
        for event in active_events:
            if event.event_type == EventType.TACHYSYSTOLE:
                contractions_per_10min = event.params.contractions_per_10min
                break
        
        mean_interval = 600.0 / contractions_per_10min
        
        # Start with baseline
        uc = np.full(n_samples, self.config.baseline_tonus)
        contraction_peaks = np.zeros(n_samples, dtype=bool)
        
        # Generate contractions
        end_time = times[-1]
        
        while self._next_contraction_time <= end_time:
            peak_time = self._next_contraction_time
            self._contraction_peaks.append(peak_time)
            
            duration = self.config.contraction_duration_sec
            sigma = duration / 4
            
            cont_start = peak_time - duration / 2
            cont_end = peak_time + duration / 2
            cont_mask = (times >= cont_start) & (times <= cont_end)
            
            if np.any(cont_mask):
                t_rel = times[cont_mask] - peak_time
                shape = np.exp(-0.5 * (t_rel / sigma) ** 2)
                uc[cont_mask] += self.config.contraction_amplitude * shape
                
                peak_idx = np.argmin(np.abs(times - peak_time))
                if 0 <= peak_idx < n_samples:
                    contraction_peaks[peak_idx] = True
            
            interval = np.random.normal(mean_interval, mean_interval * 0.2)
            interval = max(interval, 60)
            self._next_contraction_time += interval
        
        # Add noise
        uc += np.random.normal(0, self.config.noise_std, n_samples)
        uc = np.clip(uc, 0, 100)
        
        self._time = times[-1] + dt
        self._contraction_peaks = [p for p in self._contraction_peaks 
                                   if p > self._time - 1200]
        
        return uc, contraction_peaks
    
    def reset(self) -> None:
        """Reset generator state."""
        self._time = 0.0
        self._next_contraction_time = np.random.uniform(30, 90)
        self._contraction_peaks = []
```

---

## 4. Integration Guide

### 4.1 Using Existing Components

The simulation module MUST use these existing components:

| Component | Import Path | Usage |
|-----------|-------------|-------|
| CTGPreprocessor | `src.data.preprocess` | Preprocess generated signals |
| calculate_baseline | `src.rules.baseline` | Calculate FHR baseline |
| calculate_variability | `src.rules.variability` | Calculate variability |
| detect_decelerations | `src.rules.decelerations` | Detect decelerations |
| detect_tachysystole | `src.rules.tachysystole` | Detect tachysystole |
| detect_sinusoidal_pattern | `src.rules.sinusoidal` | Detect sinusoidal |
| MomentFeatureExtractor | `src.models.moment_encoder` | Extract embeddings |
| build_feature_vector | `src.models.fusion` | Build feature vector |
| XGBClassifierWrapper | `src.models.classifier` | Classification |
| apply_medical_override | `src.analysis.override` | Safety rules |
| generate_alert | `src.analysis.alerts` | Generate alerts |

### 4.2 Configuration Constants

Use from `src/config.py`:

```python
from src.config import CTG, THRESHOLDS, COLORS

# Sampling rate
CTG.SAMPLING_RATE  # 4.0 Hz

# Clinical thresholds
THRESHOLDS.BASELINE_MIN  # 110 bpm
THRESHOLDS.BASELINE_MAX  # 160 bpm
THRESHOLDS.VARIABILITY_ABSENT_MAX  # 2 bpm
THRESHOLDS.VARIABILITY_MINIMAL_MAX  # 5 bpm

# UI Colors
COLORS.CATEGORY_1  # Green
COLORS.CATEGORY_2  # Orange  
COLORS.CATEGORY_3  # Red
```

---

*Continued in Part 2...*
