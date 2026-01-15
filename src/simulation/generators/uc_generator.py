"""
UC Generator - Generates realistic uterine contraction signals.

Implements mathematical models for:
- Regular contraction patterns (3-5 per 10 minutes normally)
- Gaussian-shaped contractions
- Baseline uterine tonus
- Tachysystole events (> 5 contractions per 10 min)

References:
    - Israeli Position Paper on CTG Interpretation
    - SentinelFetal Real-Time Simulator SPEC Part 1, Section 3.4
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from src.config import CTG
from ..events.event_types import InjectedEvent, EventType, TachysystoleParams


@dataclass
class UCGeneratorConfig:
    """
    Configuration for UC generation.
    
    Attributes:
        sampling_rate: Samples per second (default: from config).
        contractions_per_10min: Normal contraction rate (default: 4).
        contraction_duration_sec: Typical contraction duration (default: 60).
        contraction_amplitude: Peak amplitude (0-100 scale, default: 80).
        baseline_tonus: Resting uterine tone (default: 10).
        noise_std: Noise standard deviation (default: 3).
    """
    sampling_rate: float = CTG.SAMPLING_RATE
    contractions_per_10min: float = 4.0
    contraction_duration_sec: float = 60.0
    contraction_amplitude: float = 80.0
    baseline_tonus: float = 10.0
    noise_std: float = 3.0


class UCGenerator:
    """
    Generates realistic uterine contraction signals.
    
    Features:
    - Configurable contraction frequency
    - Gaussian-shaped contractions for realistic appearance
    - Baseline uterine tonus
    - Support for tachysystole events
    - Tracks contraction peak times for deceleration timing
    
    Example:
        >>> config = UCGeneratorConfig(contractions_per_10min=4.0)
        >>> generator = UCGenerator(config)
        >>> uc, peaks = generator.generate_samples(4)  # 1 second at 4 Hz
        >>> print(uc.shape)  # (4,)
    """
    
    def __init__(self, config: Optional[UCGeneratorConfig] = None):
        """
        Initialize UC generator.
        
        Args:
            config: Generator configuration. Uses defaults if None.
        """
        self.config = config or UCGeneratorConfig()
        self._time = 0.0
        self._rng = np.random.default_rng()
        
        # Schedule next contraction
        self._next_contraction_time = self._rng.uniform(30, 90)
        
        # Track recent contraction peaks for deceleration coordination
        self._contraction_peaks: List[float] = []
    
    def generate_samples(
        self,
        n_samples: int,
        active_events: Optional[List[InjectedEvent]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate UC samples.
        
        Args:
            n_samples: Number of samples to generate.
            active_events: Currently active injected events.
            
        Returns:
            Tuple of:
                - uc: numpy array of UC values (0-100 scale)
                - contraction_peaks: boolean array marking contraction peaks
        """
        active_events = active_events or []
        dt = 1.0 / self.config.sampling_rate
        times = self._time + np.arange(n_samples) * dt
        
        # Check for tachysystole event
        contractions_per_10min = self.config.contractions_per_10min
        for event in active_events:
            if event.event_type == EventType.TACHYSYSTOLE:
                params: TachysystoleParams = event.params
                contractions_per_10min = params.contractions_per_10min
                break
        
        # Calculate mean interval between contractions
        # 600 seconds / contractions_per_10min = interval
        mean_interval = 600.0 / contractions_per_10min
        
        # Start with baseline tonus
        uc = np.full(n_samples, self.config.baseline_tonus, dtype=np.float64)
        contraction_peaks = np.zeros(n_samples, dtype=bool)
        
        # Generate contractions
        end_time = times[-1] if len(times) > 0 else self._time
        
        while self._next_contraction_time <= end_time:
            peak_time = self._next_contraction_time
            self._contraction_peaks.append(peak_time)
            
            duration = self.config.contraction_duration_sec
            sigma = duration / 4  # Gaussian spread
            
            # Contraction spans approximately 2 standard deviations each side
            cont_start = peak_time - duration / 2
            cont_end = peak_time + duration / 2
            cont_mask = (times >= cont_start) & (times <= cont_end)
            
            if np.any(cont_mask):
                t_rel = times[cont_mask] - peak_time
                # Gaussian shape
                shape = np.exp(-0.5 * (t_rel / sigma) ** 2)
                uc[cont_mask] += self.config.contraction_amplitude * shape
                
                # Mark peak location
                peak_idx = np.argmin(np.abs(times - peak_time))
                if 0 <= peak_idx < n_samples:
                    contraction_peaks[peak_idx] = True
            
            # Schedule next contraction with some randomness
            interval = self._rng.normal(mean_interval, mean_interval * 0.2)
            interval = max(interval, 60.0)  # Minimum 60 seconds between contractions
            self._next_contraction_time += interval
        
        # Add noise
        noise = self._rng.normal(0, self.config.noise_std, n_samples)
        uc += noise
        
        # Clip to valid range
        uc = np.clip(uc, 0, 100)
        
        # Update time
        self._time = times[-1] + dt if len(times) > 0 else self._time + n_samples * dt
        
        # Clean up old contraction peaks (keep last 20 minutes)
        self._contraction_peaks = [
            p for p in self._contraction_peaks 
            if p > self._time - 1200
        ]
        
        return uc, contraction_peaks
    
    def get_recent_peaks(self, duration_seconds: float = 600.0) -> List[float]:
        """
        Get recent contraction peak times.
        
        Args:
            duration_seconds: How far back to look (default: 10 min).
            
        Returns:
            List of peak times within the specified duration.
        """
        cutoff = self._time - duration_seconds
        return [p for p in self._contraction_peaks if p >= cutoff]
    
    def get_contraction_rate(self) -> float:
        """
        Calculate current contraction rate per 10 minutes.
        
        Returns:
            Contractions per 10 minutes based on recent activity.
        """
        # Count contractions in last 10 minutes
        recent = self.get_recent_peaks(600.0)
        return float(len(recent))
    
    def reset(self) -> None:
        """Reset generator state to initial conditions."""
        self._time = 0.0
        self._next_contraction_time = self._rng.uniform(30, 90)
        self._contraction_peaks = []
    
    @property
    def current_time(self) -> float:
        """Get current simulation time."""
        return self._time
    
    @property
    def next_contraction_time(self) -> float:
        """Get scheduled time for next contraction."""
        return self._next_contraction_time
