"""
Sinusoidal Pattern Detection Module.

Implements sinusoidal pattern detection according to the Israeli Position Paper.

Definition:
    A sinusoidal pattern is a smooth, sine wave-like oscillating pattern in the
    baseline FHR with the following characteristics:
        - Frequency: 3-5 cycles per minute (0.05-0.083 Hz)
        - Amplitude: 5-15 bpm
        - Duration: > 20 minutes
        - Absent short-term variability (smooth waves)

Clinical Significance:
    SEVERE FINDING - Always classified as Category 3 (Pathological).
    Associated with:
        - Fetal anemia (Rh isoimmunization, fetomaternal hemorrhage)
        - Severe hypoxia
        - May indicate imminent fetal death
    
    REQUIRES IMMEDIATE ACTION - Override to Category 3 regardless of ML prediction.

Detection Algorithm:
    Uses FFT (Fast Fourier Transform) analysis to detect dominant frequency
    in the 3-5 cycles/minute range.

References:
    - Israeli Position Paper on CTG Interpretation
    - SentinelFetal Gen3.5 Technical Specification, Section 5.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.fft import fft, fftfreq

# Configure module logger
logger = logging.getLogger(__name__)


class SinusoidalDetectionError(Exception):
    """Raised when sinusoidal pattern detection fails."""
    pass


@dataclass
class SinusoidalResult:
    """
    Result of sinusoidal pattern detection.
    
    Attributes:
        detected: True if sinusoidal pattern is present (SEVERE FINDING).
        confidence: Confidence in the detection (0-1).
        dominant_frequency: Dominant frequency in the target range (Hz).
        frequency_cycles_per_min: Dominant frequency in cycles/minute.
        amplitude: Measured amplitude of oscillation (bpm).
        amplitude_in_range: True if amplitude is in 5-15 bpm range.
        dominance_ratio: Ratio of target frequency power to total power.
        analysis_duration_minutes: Duration of signal analyzed.
    """
    
    detected: bool
    confidence: float
    dominant_frequency: Optional[float]
    frequency_cycles_per_min: Optional[float]
    amplitude: float
    amplitude_in_range: bool
    dominance_ratio: float
    analysis_duration_minutes: float
    
    def __repr__(self) -> str:
        if self.detected:
            return (
                f"SinusoidalResult(DETECTED - SEVERE!, "
                f"freq={self.frequency_cycles_per_min:.1f} cycles/min, "
                f"amp={self.amplitude:.1f} bpm)"
            )
        return f"SinusoidalResult(Not detected, confidence={self.confidence:.2f})"


def detect_sinusoidal_pattern(
    fhr: np.ndarray,
    sampling_rate: float = 4.0,
    min_duration_minutes: float = 20.0,
    freq_min_cycles_per_min: float = 3.0,
    freq_max_cycles_per_min: float = 5.0,
    amp_min: float = 5.0,
    amp_max: float = 15.0,
    dominance_threshold: float = 0.3
) -> SinusoidalResult:
    """
    Detect sinusoidal pattern in FHR signal.
    
    Algorithm:
        1. Take the last `min_duration_minutes` of the signal
        2. Perform FFT to analyze frequency content
        3. Look for dominant peak in 3-5 cycles/minute range (0.05-0.083 Hz)
        4. Check if amplitude is in 5-15 bpm range
        5. If both criteria met → Sinusoidal pattern detected
    
    WARNING: Sinusoidal pattern is a SEVERE finding that ALWAYS results in
    Category 3 classification, regardless of other parameters.
    
    Args:
        fhr: FHR signal array in bpm.
        sampling_rate: Sampling frequency in Hz (default: 4.0).
        min_duration_minutes: Minimum duration to analyze (default: 20.0).
        freq_min_cycles_per_min: Minimum frequency in cycles/min (default: 3.0).
        freq_max_cycles_per_min: Maximum frequency in cycles/min (default: 5.0).
        amp_min: Minimum amplitude in bpm (default: 5.0).
        amp_max: Maximum amplitude in bpm (default: 15.0).
        dominance_threshold: Minimum ratio for frequency dominance (default: 0.3).
        
    Returns:
        SinusoidalResult with detection outcome and analysis metrics.
        
    Raises:
        SinusoidalDetectionError: If detection fails due to invalid input.
        
    Example:
        >>> result = detect_sinusoidal_pattern(fhr_signal)
        >>> if result.detected:
        ...     print("SEVERE: Sinusoidal pattern detected - Category 3!")
    """
    # Validate input
    if fhr is None or len(fhr) == 0:
        raise SinusoidalDetectionError("Input FHR signal is empty or None")
    
    # Calculate minimum required samples
    min_samples = int(min_duration_minutes * 60 * sampling_rate)  # 20 * 60 * 4 = 4800
    
    # Convert frequency range to Hz
    freq_min_hz = freq_min_cycles_per_min / 60  # 3/60 = 0.05 Hz
    freq_max_hz = freq_max_cycles_per_min / 60  # 5/60 = 0.083 Hz
    
    # Check signal length
    if len(fhr) < min_samples:
        actual_duration = len(fhr) / sampling_rate / 60
        logger.warning(
            f"Signal too short ({actual_duration:.1f} min) for sinusoidal detection "
            f"(requires {min_duration_minutes} min)"
        )
        return _create_not_detected_result(actual_duration)
    
    # Use the last min_duration_minutes of the signal
    segment = fhr[-min_samples:]
    actual_duration = min_duration_minutes
    
    # Handle NaN values - replace with mean (FFT doesn't handle NaN)
    valid_values = segment[~np.isnan(segment)]
    if len(valid_values) < min_samples * 0.5:
        logger.warning("Too many NaN values for reliable sinusoidal detection")
        return _create_not_detected_result(actual_duration, confidence=0.2)
    
    mean_fhr = float(np.mean(valid_values))
    segment_clean = np.nan_to_num(segment, nan=mean_fhr)
    
    # Calculate amplitude (peak-to-peak)
    amplitude = float(np.max(valid_values) - np.min(valid_values))
    amplitude_in_range = amp_min <= amplitude <= amp_max
    
    # Perform FFT analysis
    n = len(segment_clean)
    
    # Remove mean (DC component) for better frequency analysis
    segment_centered = segment_clean - mean_fhr
    
    # Compute FFT
    yf = np.abs(fft(segment_centered))
    xf = fftfreq(n, 1 / sampling_rate)
    
    # Only look at positive frequencies
    positive_mask = xf > 0
    xf_positive = xf[positive_mask]
    yf_positive = yf[positive_mask]
    
    # Find the target frequency range
    target_mask = (xf_positive >= freq_min_hz) & (xf_positive <= freq_max_hz)
    
    if not np.any(target_mask):
        logger.warning("No frequency bins in target range")
        return _create_not_detected_result(actual_duration)
    
    # Get power in target range
    target_frequencies = xf_positive[target_mask]
    target_power = yf_positive[target_mask]
    
    # Find dominant frequency in target range
    peak_idx = np.argmax(target_power)
    dominant_frequency = float(target_frequencies[peak_idx])
    peak_power = float(target_power[peak_idx])
    
    # Calculate dominance ratio (power in target range / total power)
    total_power = float(np.sum(yf_positive))
    target_range_power = float(np.sum(target_power))
    dominance_ratio = target_range_power / total_power if total_power > 0 else 0
    
    # Convert dominant frequency to cycles per minute
    frequency_cycles_per_min = dominant_frequency * 60
    
    # Determine if sinusoidal pattern is detected
    frequency_ok = freq_min_hz <= dominant_frequency <= freq_max_hz
    dominance_ok = dominance_ratio >= dominance_threshold
    
    detected = frequency_ok and dominance_ok and amplitude_in_range
    
    # Calculate confidence
    confidence = _calculate_confidence(
        dominance_ratio, dominance_threshold,
        amplitude, amp_min, amp_max
    )
    
    if detected:
        logger.warning(
            f"SINUSOIDAL PATTERN DETECTED - SEVERE FINDING! "
            f"Frequency: {frequency_cycles_per_min:.1f} cycles/min, "
            f"Amplitude: {amplitude:.1f} bpm, "
            f"Dominance: {dominance_ratio:.2f}"
        )
    else:
        logger.debug(
            f"Sinusoidal check: freq={frequency_cycles_per_min:.1f}/min, "
            f"amp={amplitude:.1f}, dominance={dominance_ratio:.2f} - Not detected"
        )
    
    return SinusoidalResult(
        detected=detected,
        confidence=round(confidence, 2),
        dominant_frequency=round(dominant_frequency, 4),
        frequency_cycles_per_min=round(frequency_cycles_per_min, 1),
        amplitude=round(amplitude, 1),
        amplitude_in_range=amplitude_in_range,
        dominance_ratio=round(dominance_ratio, 3),
        analysis_duration_minutes=round(actual_duration, 1)
    )


def _create_not_detected_result(
    duration: float,
    confidence: float = 0.0
) -> SinusoidalResult:
    """
    Create a result indicating no sinusoidal pattern detected.
    
    Args:
        duration: Duration of signal analyzed.
        confidence: Confidence level.
        
    Returns:
        SinusoidalResult with detected=False.
    """
    return SinusoidalResult(
        detected=False,
        confidence=confidence,
        dominant_frequency=None,
        frequency_cycles_per_min=None,
        amplitude=0.0,
        amplitude_in_range=False,
        dominance_ratio=0.0,
        analysis_duration_minutes=duration
    )


def _calculate_confidence(
    dominance_ratio: float,
    dominance_threshold: float,
    amplitude: float,
    amp_min: float,
    amp_max: float
) -> float:
    """
    Calculate confidence score for sinusoidal detection.
    
    Args:
        dominance_ratio: Measured frequency dominance.
        dominance_threshold: Required dominance threshold.
        amplitude: Measured amplitude.
        amp_min: Minimum valid amplitude.
        amp_max: Maximum valid amplitude.
        
    Returns:
        Confidence score between 0 and 1.
    """
    # Dominance component (how much of signal is in target frequency range)
    dominance_score = min(1.0, dominance_ratio / (dominance_threshold * 2))
    
    # Amplitude component (how close to ideal range)
    amp_center = (amp_min + amp_max) / 2
    amp_range = (amp_max - amp_min) / 2
    amp_deviation = abs(amplitude - amp_center) / amp_range if amp_range > 0 else 1
    amp_score = max(0, 1 - amp_deviation)
    
    # Weighted average
    confidence = 0.6 * dominance_score + 0.4 * amp_score
    
    return min(1.0, max(0.0, confidence))


def generate_sinusoidal_test_signal(
    duration_minutes: float = 25.0,
    sampling_rate: float = 4.0,
    baseline: float = 140.0,
    amplitude: float = 10.0,
    cycles_per_minute: float = 4.0
) -> np.ndarray:
    """
    Generate a synthetic sinusoidal FHR signal for testing.
    
    Creates a pure sine wave that should trigger sinusoidal detection.
    
    Args:
        duration_minutes: Signal duration in minutes.
        sampling_rate: Sampling frequency in Hz.
        baseline: Baseline FHR in bpm.
        amplitude: Amplitude of oscillation in bpm.
        cycles_per_minute: Frequency in cycles per minute.
        
    Returns:
        Synthetic FHR signal array.
        
    Example:
        >>> sinusoidal_signal = generate_sinusoidal_test_signal()
        >>> result = detect_sinusoidal_pattern(sinusoidal_signal)
        >>> assert result.detected, "Should detect sinusoidal pattern"
    """
    n_samples = int(duration_minutes * 60 * sampling_rate)
    t = np.arange(n_samples) / sampling_rate  # Time in seconds
    
    frequency_hz = cycles_per_minute / 60
    
    # Generate pure sine wave
    fhr = baseline + amplitude * np.sin(2 * np.pi * frequency_hz * t)
    
    return fhr
