"""
Patient Generator - Combined FHR and UC generation for a single patient.

Coordinates the FHR and UC generators, manages the ring buffer,
and handles event injection for a simulated patient.

References:
    - SentinelFetal Real-Time Simulator SPEC Part 2, Section 5
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

from src.config import CTG
from .fhr_generator import FHRGenerator, FHRGeneratorConfig
from .uc_generator import UCGenerator, UCGeneratorConfig
from ..core.ring_buffer import RingBuffer
from ..events.event_types import (
    InjectedEvent,
    EventType,
    EventParameters,
)


@dataclass
class PatientConfig:
    """
    Configuration for a simulated patient.
    
    Attributes:
        patient_id: Unique identifier (e.g., 'P1', 'P2').
        bed_number: Bed/room number for display.
        name: Patient name for display (Hebrew).
        baseline_fhr: Initial baseline FHR in bpm.
        baseline_variability: Initial variability amplitude in bpm.
        contractions_per_10min: Initial contraction rate.
        buffer_duration_minutes: Ring buffer size in minutes.
        sampling_rate: Samples per second (from config).
    """
    patient_id: str
    bed_number: int
    name: str = "יולדת סימולציה"
    baseline_fhr: float = 140.0
    baseline_variability: float = 10.0
    contractions_per_10min: float = 4.0
    buffer_duration_minutes: float = 10.0
    sampling_rate: float = CTG.SAMPLING_RATE


class PatientGenerator:
    """
    Generates and manages CTG data for a single simulated patient.
    
    Responsibilities:
    - Coordinates FHR and UC signal generation
    - Manages ring buffer of recent data
    - Handles event injection and lifecycle
    - Tracks analysis results (category, alerts, findings)
    
    Thread Safety:
    - Event list modifications are protected by a lock
    - Buffer operations are atomic (append_batch)
    
    Example:
        >>> config = PatientConfig(patient_id='P1', bed_number=1)
        >>> patient = PatientGenerator(config)
        >>> data = patient.generate_tick(4)  # 1 second at 4 Hz
        >>> print(data['fhr'].shape)  # (4,)
        >>> print(patient.buffer.size)  # 4
    """
    
    def __init__(self, config: PatientConfig):
        """
        Initialize patient generator.
        
        Args:
            config: Patient configuration.
        """
        self.config = config
        self.patient_id = config.patient_id
        
        # Create FHR generator
        fhr_config = FHRGeneratorConfig(
            baseline_fhr=config.baseline_fhr,
            baseline_variability=config.baseline_variability,
            sampling_rate=config.sampling_rate
        )
        self._fhr_generator = FHRGenerator(fhr_config)
        
        # Create UC generator
        uc_config = UCGeneratorConfig(
            contractions_per_10min=config.contractions_per_10min,
            sampling_rate=config.sampling_rate
        )
        self._uc_generator = UCGenerator(uc_config)
        
        # Create ring buffer
        buffer_samples = int(
            config.buffer_duration_minutes * 60 * config.sampling_rate
        )
        self._buffer = RingBuffer(
            max_samples=buffer_samples,
            sampling_rate=config.sampling_rate
        )
        
        # Event management
        self._active_events: List[InjectedEvent] = []
        self._event_lock = threading.Lock()
        
        # Simulation state
        self._simulation_time = 0.0
        
        # Latest analysis results (updated by processing pipeline)
        self.latest_category: int = 1  # Default: Normal
        self.latest_alert: Optional[Any] = None
        self.latest_findings: Dict[str, Any] = {}
    
    def generate_tick(self, n_samples: int = 4) -> Dict[str, Any]:
        """
        Generate one tick of data (default: 1 second = 4 samples).
        
        Args:
            n_samples: Number of samples to generate.
            
        Returns:
            Dictionary with:
                - 'fhr': numpy array of FHR values
                - 'uc': numpy array of UC values
                - 'timestamps': numpy array of timestamps
                - 'contraction_peaks': boolean array of peak locations
        """
        dt = 1.0 / self.config.sampling_rate
        
        # Remove expired events
        self._update_active_events()
        
        # Get thread-safe copy of active events
        with self._event_lock:
            active_events = list(self._active_events)
        
        # Generate UC first (needed for deceleration timing)
        uc, contraction_peaks = self._uc_generator.generate_samples(
            n_samples, active_events
        )
        
        # Generate FHR (uses contraction peaks for deceleration timing)
        fhr = self._fhr_generator.generate_samples(
            n_samples, active_events, contraction_peaks
        )
        
        # Generate timestamps
        timestamps = self._simulation_time + np.arange(n_samples) * dt
        
        # Add to ring buffer
        self._buffer.append_batch(fhr, uc, timestamps)
        
        # Update simulation time
        self._simulation_time = timestamps[-1] + dt
        
        return {
            'fhr': fhr,
            'uc': uc,
            'timestamps': timestamps,
            'contraction_peaks': contraction_peaks
        }
    
    def inject_event(
        self,
        event_type: EventType,
        params: EventParameters,
        duration_seconds: Optional[float] = None
    ) -> InjectedEvent:
        """
        Inject a clinical event into this patient.
        
        Args:
            event_type: Type of event to inject.
            params: Event-specific parameters.
            duration_seconds: Override duration (uses params default if None).
            
        Returns:
            The created InjectedEvent object.
        """
        duration = duration_seconds or params.duration_seconds
        
        event = InjectedEvent(
            event_type=event_type,
            params=params,
            patient_id=self.patient_id,
            start_time=self._simulation_time,
            end_time=self._simulation_time + duration,
            is_active=True
        )
        
        with self._event_lock:
            self._active_events.append(event)
        
        return event
    
    def remove_event(self, event: InjectedEvent) -> bool:
        """
        Remove a specific event.
        
        Args:
            event: The event to remove.
            
        Returns:
            True if event was found and removed.
        """
        with self._event_lock:
            if event in self._active_events:
                self._active_events.remove(event)
                return True
        return False
    
    def clear_events(self) -> None:
        """Remove all active events."""
        with self._event_lock:
            self._active_events.clear()
    
    def _update_active_events(self) -> None:
        """Remove expired events from the active list."""
        with self._event_lock:
            self._active_events = [
                e for e in self._active_events
                if e.end_time > self._simulation_time
            ]
    
    def get_buffer_data(
        self, 
        duration_minutes: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get data from the ring buffer.
        
        Args:
            duration_minutes: Minutes of data to return (None = all).
            
        Returns:
            Dictionary with buffer data plus patient metadata.
        """
        if duration_minutes:
            data = self._buffer.get_last_n_minutes(duration_minutes)
        else:
            data = self._buffer.get_window()
        
        return {
            **data,
            'patient_id': self.patient_id,
            'bed_number': self.config.bed_number,
            'name': self.config.name,
            'simulation_time': self._simulation_time
        }
    
    def get_active_events(self) -> List[InjectedEvent]:
        """Get currently active events (thread-safe copy)."""
        with self._event_lock:
            return list(self._active_events)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get patient status summary.
        
        Returns:
            Dictionary with patient state and metadata.
        """
        return {
            'patient_id': self.patient_id,
            'bed_number': self.config.bed_number,
            'name': self.config.name,
            'category': self.latest_category,
            'active_events': [e.event_type.name for e in self.get_active_events()],
            'buffer_duration': self._buffer.duration_seconds,
            'buffer_size': self._buffer.size,
            'simulation_time': self._simulation_time
        }
    
    def reset(self) -> None:
        """Reset patient to initial state."""
        self._fhr_generator.reset()
        self._uc_generator.reset()
        self._buffer.clear()
        
        with self._event_lock:
            self._active_events.clear()
        
        self._simulation_time = 0.0
        self.latest_category = 1
        self.latest_alert = None
        self.latest_findings = {}
    
    @property
    def buffer(self) -> RingBuffer:
        """Access to the ring buffer."""
        return self._buffer
    
    @property
    def simulation_time(self) -> float:
        """Current simulation time in seconds."""
        return self._simulation_time
    
    @property
    def simulation_time_minutes(self) -> float:
        """Current simulation time in minutes."""
        return self._simulation_time / 60.0
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PatientGenerator(id={self.patient_id}, "
            f"bed={self.config.bed_number}, "
            f"time={self._simulation_time:.1f}s, "
            f"category={self.latest_category})"
        )
