"""
FHR Signal Generator - Generates realistic fetal heart rate signals.

Implements mathematical models for:
- Baseline FHR with configurable setpoint
- Variability oscillations (2-6 cycles per minute)
- High-frequency noise
- Event injection (decelerations, baseline shifts, sinusoidal)

References:
    - Israeli Position Paper on CTG Interpretation
    - SentinelFetal Real-Time Simulator SPEC Part 1, Section 3.3
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from src.config import CTG, THRESHOLDS
from ..events.event_types import (
    InjectedEvent,
    EventType,
    LateDecelerationParams,
    VariableDecelerationParams,
    ProlongedDecelerationParams,
    EarlyDecelerationParams,
    BradycardiaParams,
    TachycardiaParams,
    VariabilityParams,
    SinusoidalParams,
)


@dataclass
class FHRGeneratorConfig:
    """
    Configuration for FHR generation.
    
    Attributes:
        sampling_rate: Samples per second (default: from config).
        baseline_fhr: Normal baseline FHR in bpm (default: 140).
        baseline_variability: Normal variability amplitude in bpm (default: 10).
        high_freq_noise_std: Standard deviation of high-freq noise (default: 2).
        min_fhr: Minimum physiological FHR (default: 50).
        max_fhr: Maximum physiological FHR (default: 240).
    """
    sampling_rate: float = CTG.SAMPLING_RATE
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
    
    The generator maintains internal state for phase continuity
    across multiple generate_samples() calls.
    
    Example:
        >>> config = FHRGeneratorConfig(baseline_fhr=145.0)
        >>> generator = FHRGenerator(config)
        >>> fhr = generator.generate_samples(4)  # 1 second at 4 Hz
        >>> print(fhr.shape)  # (4,)
    """
    
    def __init__(self, config: Optional[FHRGeneratorConfig] = None):
        """
        Initialize FHR generator.
        
        Args:
            config: Generator configuration. Uses defaults if None.
        """
        self.config = config or FHRGeneratorConfig()
        self._phase = 0.0
        self._time = 0.0
        self._rng = np.random.default_rng()
    
    def generate_samples(
        self,
        n_samples: int,
        active_events: Optional[List[InjectedEvent]] = None,
        contraction_peaks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate FHR samples.
        
        Args:
            n_samples: Number of samples to generate.
            active_events: Currently active injected events.
            contraction_peaks: Boolean array marking contraction peaks
                             (used for deceleration timing).
            
        Returns:
            Numpy array of FHR values in bpm.
        """
        active_events = active_events or []
        dt = 1.0 / self.config.sampling_rate
        times = self._time + np.arange(n_samples) * dt
        
        # Start with baseline
        fhr = np.full(n_samples, self.config.baseline_fhr, dtype=np.float64)
        
        # Add variability
        variability = self._generate_variability(n_samples, times, active_events)
        fhr += variability
        
        # Add high-frequency noise
        noise = self._rng.normal(0, self.config.high_freq_noise_std, n_samples)
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
        times: np.ndarray,
        active_events: List[InjectedEvent]
    ) -> np.ndarray:
        """
        Generate baseline variability oscillations.
        
        Normal variability: 6-25 bpm (Moderate)
        Uses multiple sine waves to create realistic irregular oscillations.
        
        Args:
            n_samples: Number of samples.
            times: Time array for samples.
            active_events: Active events (may modify variability).
            
        Returns:
            Variability component to add to baseline.
        """
        # Check for variability-modifying events
        target_var = self.config.baseline_variability
        
        for event in active_events:
            if event.event_type == EventType.ABSENT_VARIABILITY:
                target_var = event.params.target_variability_bpm
                break
            elif event.event_type == EventType.MINIMAL_VARIABILITY:
                target_var = event.params.target_variability_bpm
                break
            elif event.event_type == EventType.MARKED_VARIABILITY:
                target_var = event.params.target_variability_bpm
                break
        
        dt = 1.0 / self.config.sampling_rate
        t = np.arange(n_samples) * dt
        
        # Primary variability: 2-6 cycles per minute (0.033-0.1 Hz)
        # Frequency slowly varies to create irregular pattern
        freq1 = 0.05 + 0.02 * np.sin(self._phase * 0.1)
        var1 = np.sin(2 * np.pi * freq1 * t + self._phase) * (target_var * 0.6)
        
        # Secondary variability: faster component
        freq2 = 0.15
        var2 = np.sin(2 * np.pi * freq2 * t + self._phase * 2) * (target_var * 0.3)
        
        # Tertiary: very slow drift
        freq3 = 0.01
        var3 = np.sin(2 * np.pi * freq3 * t + self._phase * 0.5) * (target_var * 0.1)
        
        return var1 + var2 + var3
    
    def _apply_event(
        self,
        fhr: np.ndarray,
        times: np.ndarray,
        event: InjectedEvent,
        contraction_peaks: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Apply an injected event to the FHR signal.
        
        Dispatches to specific event handlers based on event type.
        """
        if event.event_type == EventType.LATE_DECELERATION:
            return self._apply_late_deceleration(fhr, times, event, contraction_peaks)
        elif event.event_type == EventType.VARIABLE_DECELERATION:
            return self._apply_variable_deceleration(fhr, times, event, contraction_peaks)
        elif event.event_type == EventType.EARLY_DECELERATION:
            return self._apply_early_deceleration(fhr, times, event, contraction_peaks)
        elif event.event_type == EventType.PROLONGED_DECELERATION:
            return self._apply_prolonged_deceleration(fhr, times, event)
        elif event.event_type == EventType.BRADYCARDIA:
            return self._apply_bradycardia(fhr, times, event)
        elif event.event_type == EventType.TACHYCARDIA:
            return self._apply_tachycardia(fhr, times, event)
        elif event.event_type == EventType.SINUSOIDAL_PATTERN:
            return self._apply_sinusoidal(fhr, times, event)
        # Variability events are handled in _generate_variability
        
        return fhr
    
    def _apply_late_deceleration(
        self,
        fhr: np.ndarray,
        times: np.ndarray,
        event: InjectedEvent,
        contraction_peaks: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Apply late deceleration pattern.
        
        Late decelerations start after contraction peak with nadir
        occurring after the peak. Uses Gaussian shape.
        """
        if contraction_peaks is None or not np.any(contraction_peaks):
            return fhr
        
        params: LateDecelerationParams = event.params
        forced_remaining = getattr(event, 'forced_contractions_remaining', None)
        
        for i, is_peak in enumerate(contraction_peaks):
            if not is_peak:
                continue

            if forced_remaining is not None:
                if forced_remaining <= 0:
                    continue
                forced_remaining -= 1
            else:
                # Random recurrence based on rate
                if self._rng.random() > params.recurrence_rate:
                    continue
            
            peak_time = times[i]
            nadir_time = peak_time + params.lag_seconds
            
            # Gaussian shape centered at nadir
            # Full deceleration spans approximately 2 * recovery_seconds
            sigma = params.recovery_seconds / 2.0
            
            # Calculate deceleration for all samples
            t_rel = times - nadir_time
            decel_shape = np.exp(-0.5 * (t_rel / sigma) ** 2)
            
            # Only apply significant portion of deceleration
            significant_mask = decel_shape > 0.01
            fhr[significant_mask] -= params.depth_bpm * decel_shape[significant_mask]
        
        if forced_remaining is not None:
            event.forced_contractions_remaining = forced_remaining
            if forced_remaining <= 0 and len(times) > 0:
                event.end_time = min(event.end_time, times[-1])

        return fhr
    
    def _apply_variable_deceleration(
        self,
        fhr: np.ndarray,
        times: np.ndarray,
        event: InjectedEvent,
        contraction_peaks: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Apply variable deceleration pattern.
        
        Variable decelerations have abrupt onset, variable timing,
        and may include severity signs.
        """
        if contraction_peaks is None or not np.any(contraction_peaks):
            return fhr
        
        params: VariableDecelerationParams = event.params
        forced_remaining = getattr(event, 'forced_contractions_remaining', None)
        
        for i, is_peak in enumerate(contraction_peaks):
            if not is_peak:
                continue

            if forced_remaining is not None:
                if forced_remaining <= 0:
                    continue
                forced_remaining -= 1
            else:
                if self._rng.random() > params.recurrence_rate:
                    continue
            
            peak_time = times[i]
            
            # Variable timing: offset from -10 to +10 seconds around peak
            offset = self._rng.uniform(-10, 10)
            decel_start = peak_time + offset
            decel_end = decel_start + params.duration_decel_seconds
            
            decel_mask = (times >= decel_start) & (times <= decel_end)
            
            if not np.any(decel_mask):
                continue
            
            t_rel = times[decel_mask] - decel_start
            duration = params.duration_decel_seconds
            
            # Sharp trapezoid shape (abrupt onset characteristic of variable)
            shape = np.ones_like(t_rel)
            
            # Fast descent; parameterized to keep variable decels steeper than late decels
            descent_time = getattr(params, "descent_time_seconds", 5.0)
            descent_mask = t_rel < descent_time
            if np.any(descent_mask):
                shape[descent_mask] = t_rel[descent_mask] / descent_time
            
            # Recovery phase
            recovery_time = 10.0 if params.slow_recovery else 5.0
            ascent_start = duration - recovery_time
            ascent_mask = t_rel > ascent_start
            if np.any(ascent_mask):
                shape[ascent_mask] = (duration - t_rel[ascent_mask]) / recovery_time
            
            # Calculate depth
            depth = params.depth_bpm
            if params.drops_below_70:
                # Ensure FHR drops below 70
                min_depth = self.config.baseline_fhr - 65.0
                depth = max(depth, min_depth)
            
            fhr[decel_mask] -= depth * shape
            
            # Add overshoot if specified
            if params.overshoot:
                overshoot_start = decel_end
                overshoot_end = decel_end + 15.0
                overshoot_mask = (times >= overshoot_start) & (times <= overshoot_end)
                if np.any(overshoot_mask):
                    t_os = times[overshoot_mask] - overshoot_start
                    os_shape = np.exp(-0.5 * (t_os / 5.0) ** 2)
                    fhr[overshoot_mask] += 15.0 * os_shape
        
        if forced_remaining is not None:
            event.forced_contractions_remaining = forced_remaining
            if forced_remaining <= 0 and len(times) > 0:
                event.end_time = min(event.end_time, times[-1])

        return fhr
    
    def _apply_early_deceleration(
        self,
        fhr: np.ndarray,
        times: np.ndarray,
        event: InjectedEvent,
        contraction_peaks: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Apply early deceleration pattern.
        
        Early decelerations are symmetric and coincide with contractions
        (nadir at contraction peak).
        """
        if contraction_peaks is None or not np.any(contraction_peaks):
            return fhr
        
        params: EarlyDecelerationParams = event.params
        forced_remaining = getattr(event, 'forced_contractions_remaining', None)
        
        for i, is_peak in enumerate(contraction_peaks):
            if not is_peak:
                continue

            if forced_remaining is not None:
                if forced_remaining <= 0:
                    continue
                forced_remaining -= 1
            else:
                if self._rng.random() > params.recurrence_rate:
                    continue
            
            peak_time = times[i]
            
            # Symmetric Gaussian centered at peak
            sigma = 15.0  # ~30 second duration
            t_rel = times - peak_time
            decel_shape = np.exp(-0.5 * (t_rel / sigma) ** 2)
            
            significant_mask = decel_shape > 0.01
            fhr[significant_mask] -= params.depth_bpm * decel_shape[significant_mask]
        
        if forced_remaining is not None:
            event.forced_contractions_remaining = forced_remaining
            if forced_remaining <= 0 and len(times) > 0:
                event.end_time = min(event.end_time, times[-1])

        return fhr
    
    def _apply_prolonged_deceleration(
        self,
        fhr: np.ndarray,
        times: np.ndarray,
        event: InjectedEvent
    ) -> np.ndarray:
        """
        Apply prolonged deceleration (> 2 minutes).
        
        Uses trapezoidal shape with gradual onset and recovery.
        """
        params: ProlongedDecelerationParams = event.params
        
        event_mask = (times >= event.start_time) & (times <= event.end_time)
        
        if not np.any(event_mask):
            return fhr
        
        t_rel = times[event_mask] - event.start_time
        duration = event.end_time - event.start_time
        
        # Trapezoidal shape
        shape = np.ones_like(t_rel)
        ramp_time = min(30.0, duration / 4)
        
        # Ramp up
        ramp_up_mask = t_rel < ramp_time
        shape[ramp_up_mask] = t_rel[ramp_up_mask] / ramp_time
        
        # Ramp down
        ramp_down_mask = t_rel > (duration - ramp_time)
        shape[ramp_down_mask] = (duration - t_rel[ramp_down_mask]) / ramp_time
        
        fhr[event_mask] -= params.depth_bpm * shape
        
        return fhr
    
    def _apply_bradycardia(
        self,
        fhr: np.ndarray,
        times: np.ndarray,
        event: InjectedEvent
    ) -> np.ndarray:
        """
        Apply bradycardia (baseline shift down).
        
        Gradually shifts baseline to target FHR.
        """
        params: BradycardiaParams = event.params
        event_mask = (times >= event.start_time) & (times <= event.end_time)
        
        if not np.any(event_mask):
            return fhr
        
        reduction = self.config.baseline_fhr - params.target_fhr
        t_rel = times[event_mask] - event.start_time
        duration = event.end_time - event.start_time
        ramp_time = min(30.0, duration / 4)
        
        # Trapezoidal ramp
        factor = np.ones_like(t_rel)
        factor[t_rel < ramp_time] = t_rel[t_rel < ramp_time] / ramp_time
        recovery_start = duration - ramp_time
        factor[t_rel > recovery_start] = (duration - t_rel[t_rel > recovery_start]) / ramp_time
        
        fhr[event_mask] -= reduction * factor
        
        return fhr
    
    def _apply_tachycardia(
        self,
        fhr: np.ndarray,
        times: np.ndarray,
        event: InjectedEvent
    ) -> np.ndarray:
        """
        Apply tachycardia (baseline shift up).
        
        Gradually shifts baseline to target FHR.
        """
        params: TachycardiaParams = event.params
        event_mask = (times >= event.start_time) & (times <= event.end_time)
        
        if not np.any(event_mask):
            return fhr
        
        increase = params.target_fhr - self.config.baseline_fhr
        t_rel = times[event_mask] - event.start_time
        duration = event.end_time - event.start_time
        ramp_time = min(60.0, duration / 4)
        
        # Trapezoidal ramp
        factor = np.ones_like(t_rel)
        factor[t_rel < ramp_time] = t_rel[t_rel < ramp_time] / ramp_time
        recovery_start = duration - ramp_time
        factor[t_rel > recovery_start] = (duration - t_rel[t_rel > recovery_start]) / ramp_time
        
        fhr[event_mask] += increase * factor
        
        return fhr
    
    def _apply_sinusoidal(
        self,
        fhr: np.ndarray,
        times: np.ndarray,
        event: InjectedEvent
    ) -> np.ndarray:
        """
        Apply sinusoidal pattern with realistic physiological noise.

        Replaces normal variability with smooth sine wave oscillation
        plus small physiological noise for clinical realism.
        SEVERE finding - always Category 3.

        Clinical Note:
            Real sinusoidal patterns are never mathematically perfect.
            Small beat-to-beat variations (1-2 bpm) are always present
            due to residual autonomic activity.
        """
        params: SinusoidalParams = event.params
        event_mask = (times >= event.start_time) & (times <= event.end_time)

        if not np.any(event_mask):
            return fhr

        t_rel = times[event_mask] - event.start_time
        freq = params.frequency_cycles_per_min / 60.0  # Convert to Hz

        # Sinusoidal pattern with realistic physiological noise
        sinusoidal = np.sin(2 * np.pi * freq * t_rel) * params.amplitude_bpm

        # Add small physiological noise (1-2 bpm standard deviation)
        # This makes the pattern look clinically realistic while
        # preserving the characteristic smooth oscillation
        noise = np.random.normal(0, 1.5, len(t_rel))

        fhr[event_mask] = self.config.baseline_fhr + sinusoidal + noise

        return fhr
    
    def reset(self) -> None:
        """Reset generator state to initial conditions."""
        self._phase = 0.0
        self._time = 0.0
    
    @property
    def current_time(self) -> float:
        """Get current simulation time."""
        return self._time
