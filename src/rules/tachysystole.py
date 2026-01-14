"""
Tachysystole Detection Module.

Implements tachysystole detection according to the Israeli Position Paper.

Definition:
    Tachysystole is excessive uterine activity, defined as:
    - Average of > 5 contractions per 10-minute window
    - Calculated over a 30-minute period

Clinical Significance:
    Tachysystole can reduce blood flow to the fetus and may cause fetal
    hypoxia. It requires intervention (e.g., tocolytics, position change).

References:
    - Israeli Position Paper on CTG Interpretation
    - SentinelFetal Gen3.5 Technical Specification, Section 5.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

# Configure module logger
logger = logging.getLogger(__name__)


class TachysystoleDetectionError(Exception):
    """Raised when tachysystole detection fails."""
    pass


@dataclass
class TachysystoleResult:
    """
    Result of tachysystole detection.
    
    Attributes:
        detected: True if tachysystole is present (>5 contractions/10min).
        contractions_per_10min: Average contraction rate per 10 minutes.
        total_contractions: Total contractions detected in analysis window.
        analysis_duration_minutes: Duration of the analysis window.
        contraction_indices: Indices of detected contraction peaks.
        confidence: Confidence in the detection (0-1).
    """
    
    detected: bool
    contractions_per_10min: float
    total_contractions: int
    analysis_duration_minutes: float
    contraction_indices: list[int]
    confidence: float = 1.0
    
    def __repr__(self) -> str:
        status = "DETECTED" if self.detected else "Not detected"
        return (
            f"TachysystoleResult({status}, "
            f"{self.contractions_per_10min:.1f} contractions/10min)"
        )


def detect_tachysystole(
    uc: np.ndarray,
    sampling_rate: float = 4.0,
    analysis_window_minutes: float = 30.0,
    threshold_per_10min: float = 5.0,
    min_contraction_distance_seconds: float = 60.0
) -> TachysystoleResult:
    """
    Detect tachysystole (excessive uterine activity).
    
    Algorithm:
        1. Take the last `analysis_window_minutes` of the UC signal
        2. Detect contractions using peak detection
        3. Calculate average contractions per 10 minutes
        4. If > threshold_per_10min → Tachysystole detected
    
    Args:
        uc: Uterine contraction signal array.
        sampling_rate: Sampling frequency in Hz (default: 4.0).
        analysis_window_minutes: Duration to analyze in minutes (default: 30.0).
        threshold_per_10min: Threshold for tachysystole (default: 5.0).
        min_contraction_distance_seconds: Minimum time between contractions (default: 60.0).
        
    Returns:
        TachysystoleResult with detection outcome and statistics.
        
    Raises:
        TachysystoleDetectionError: If detection fails due to invalid input.
        
    Example:
        >>> result = detect_tachysystole(uc_signal, sampling_rate=4.0)
        >>> if result.detected:
        ...     print(f"Tachysystole: {result.contractions_per_10min:.1f}/10min")
    """
    # Validate input
    if uc is None or len(uc) == 0:
        raise TachysystoleDetectionError("Input UC signal is empty or None")
    
    # Calculate window parameters
    window_samples = int(analysis_window_minutes * 60 * sampling_rate)
    min_distance_samples = int(min_contraction_distance_seconds * sampling_rate)
    
    # Use the last N minutes of the signal
    if len(uc) < window_samples:
        # Use entire signal if shorter than window
        analysis_window = uc.copy()
        actual_duration_minutes = len(uc) / sampling_rate / 60
        logger.warning(
            f"UC signal ({actual_duration_minutes:.1f} min) shorter than "
            f"analysis window ({analysis_window_minutes} min)"
        )
    else:
        analysis_window = uc[-window_samples:]
        actual_duration_minutes = analysis_window_minutes
    
    # Clean up UC signal
    uc_clean = np.nan_to_num(analysis_window, nan=0.0)
    
    # Check if UC signal has meaningful data
    if np.all(uc_clean == 0) or np.std(uc_clean) < 1e-6:
        logger.warning("UC signal appears flat or empty")
        return TachysystoleResult(
            detected=False,
            contractions_per_10min=0.0,
            total_contractions=0,
            analysis_duration_minutes=actual_duration_minutes,
            contraction_indices=[],
            confidence=0.0
        )
    
    # Detect contractions using peak detection
    # Use percentile threshold to adapt to signal amplitude
    threshold = np.percentile(uc_clean, 75)
    
    # Find peaks (contractions)
    peaks, properties = find_peaks(
        uc_clean,
        height=threshold,
        distance=min_distance_samples,
        prominence=np.std(uc_clean) * 0.5  # Require some prominence
    )
    
    total_contractions = len(peaks)
    
    # Calculate contractions per 10 minutes
    # actual_duration_minutes / 10 = number of 10-minute periods
    ten_min_periods = actual_duration_minutes / 10.0
    contractions_per_10min = total_contractions / ten_min_periods if ten_min_periods > 0 else 0.0
    
    # Determine if tachysystole is present
    detected = contractions_per_10min > threshold_per_10min
    
    # Calculate confidence based on signal quality
    signal_quality = min(1.0, np.std(uc_clean) / 20)  # Normalize by expected std
    confidence = signal_quality
    
    logger.info(
        f"Tachysystole analysis: {total_contractions} contractions in "
        f"{actual_duration_minutes:.1f} min = {contractions_per_10min:.1f}/10min"
    )
    
    return TachysystoleResult(
        detected=detected,
        contractions_per_10min=round(contractions_per_10min, 1),
        total_contractions=total_contractions,
        analysis_duration_minutes=round(actual_duration_minutes, 1),
        contraction_indices=peaks.tolist(),
        confidence=round(confidence, 2)
    )


def count_contractions(
    uc: np.ndarray,
    sampling_rate: float = 4.0,
    min_distance_seconds: float = 60.0
) -> int:
    """
    Count the number of contractions in a UC signal.
    
    Convenience function for getting contraction count without full analysis.
    
    Args:
        uc: Uterine contraction signal array.
        sampling_rate: Sampling frequency in Hz.
        min_distance_seconds: Minimum time between contractions.
        
    Returns:
        Number of detected contractions.
    """
    result = detect_tachysystole(
        uc,
        sampling_rate=sampling_rate,
        analysis_window_minutes=len(uc) / sampling_rate / 60,
        min_contraction_distance_seconds=min_distance_seconds
    )
    return result.total_contractions
