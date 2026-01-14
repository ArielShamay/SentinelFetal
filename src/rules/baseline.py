"""
Baseline FHR Calculator.

Implements baseline calculation according to the Israeli Position Paper.

Definition:
    Baseline FHR is the mean FHR rounded to increments of 5 bpm during a 
    10-minute segment, excluding:
        - Periodic or episodic changes
        - Periods of marked variability (>25 bpm)
        - Segments where the FHR differs by >25 bpm

The algorithm finds stable 2-minute segments where variability is <25 bpm,
then calculates the mean and rounds to the nearest 5 bpm.

Normal Range: 110-160 bpm

References:
    - Israeli Position Paper on CTG Interpretation
    - SentinelFetal Gen3.5 Technical Specification, Section 5.1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


class BaselineCalculationError(Exception):
    """Raised when baseline calculation fails."""
    pass


@dataclass
class BaselineResult:
    """
    Result of baseline FHR calculation.
    
    Attributes:
        value: Calculated baseline in bpm (rounded to nearest 5).
        is_normal: True if baseline is within normal range (110-160 bpm).
        is_bradycardia: True if baseline < 110 bpm.
        is_tachycardia: True if baseline > 160 bpm.
        stable_segment_found: True if a stable segment was found.
        segment_start_idx: Start index of the stable segment used.
        segment_variability: Variability in the stable segment.
        confidence: Confidence in the calculation (0-1).
    """
    
    value: float
    is_normal: bool
    is_bradycardia: bool
    is_tachycardia: bool
    stable_segment_found: bool
    segment_start_idx: Optional[int] = None
    segment_variability: Optional[float] = None
    confidence: float = 1.0
    
    def __repr__(self) -> str:
        status = "Normal" if self.is_normal else ("Bradycardia" if self.is_bradycardia else "Tachycardia")
        return f"BaselineResult(value={self.value} bpm, status={status})"


def calculate_baseline(
    fhr: np.ndarray,
    sampling_rate: float = 4.0,
    window_minutes: float = 2.0,
    variability_threshold: float = 25.0,
    step_seconds: float = 10.0
) -> BaselineResult:
    """
    Calculate baseline FHR according to the Israeli Position Paper.
    
    Algorithm:
        1. Slide a window of `window_minutes` across the signal
        2. For each window, calculate variability (max - min)
        3. Find the most stable segment (lowest variability < threshold)
        4. Calculate mean FHR in that segment
        5. Round to nearest multiple of 5
    
    Args:
        fhr: FHR signal array in bpm.
        sampling_rate: Sampling frequency in Hz (default: 4.0).
        window_minutes: Minimum stable segment duration in minutes (default: 2.0).
        variability_threshold: Maximum variability for stable segment in bpm (default: 25.0).
        step_seconds: Step size for sliding window in seconds (default: 10.0).
        
    Returns:
        BaselineResult with calculated baseline and classification.
        
    Raises:
        BaselineCalculationError: If calculation fails due to invalid input.
        
    Example:
        >>> result = calculate_baseline(fhr_signal, sampling_rate=4.0)
        >>> print(f"Baseline: {result.value} bpm")
        >>> if result.is_bradycardia:
        ...     print("Warning: Bradycardia detected")
    """
    # Validate input
    if fhr is None or len(fhr) == 0:
        raise BaselineCalculationError("Input FHR signal is empty or None")
    
    # Calculate window parameters
    window_samples = int(window_minutes * 60 * sampling_rate)  # e.g., 2 min * 60 * 4 = 480
    step_samples = int(step_seconds * sampling_rate)  # e.g., 10 * 4 = 40
    
    # Check if signal is long enough
    if len(fhr) < window_samples:
        logger.warning(
            f"Signal too short ({len(fhr)} samples) for {window_minutes}-minute window. "
            "Using global mean as fallback."
        )
        return _calculate_fallback_baseline(fhr)
    
    # Search for stable segment
    best_baseline: Optional[float] = None
    best_variability = float('inf')
    best_start_idx: Optional[int] = None
    
    for start in range(0, len(fhr) - window_samples + 1, step_samples):
        end = start + window_samples
        segment = fhr[start:end]
        
        # Filter out NaN values
        valid_segment = segment[~np.isnan(segment)]
        
        # Require at least 80% valid data in segment
        if len(valid_segment) < window_samples * 0.8:
            continue
        
        # Calculate variability (max - min) in segment
        variability = float(np.max(valid_segment) - np.min(valid_segment))
        
        # Check if this is a stable segment
        if variability < variability_threshold and variability < best_variability:
            best_variability = variability
            best_baseline = float(np.mean(valid_segment))
            best_start_idx = start
            
            logger.debug(
                f"Found stable segment at {start}: "
                f"mean={best_baseline:.1f}, variability={variability:.1f}"
            )
    
    # If no stable segment found, use fallback
    if best_baseline is None:
        logger.warning(
            f"No stable segment found (variability < {variability_threshold}). "
            "Using global mean as fallback."
        )
        return _calculate_fallback_baseline(fhr)
    
    # Round to nearest multiple of 5
    baseline_rounded = _round_to_nearest_5(best_baseline)
    
    # Classify baseline
    is_normal = 110 <= baseline_rounded <= 160
    is_bradycardia = baseline_rounded < 110
    is_tachycardia = baseline_rounded > 160
    
    logger.info(
        f"Baseline calculated: {baseline_rounded} bpm "
        f"(raw: {best_baseline:.1f}, variability: {best_variability:.1f})"
    )
    
    return BaselineResult(
        value=baseline_rounded,
        is_normal=is_normal,
        is_bradycardia=is_bradycardia,
        is_tachycardia=is_tachycardia,
        stable_segment_found=True,
        segment_start_idx=best_start_idx,
        segment_variability=best_variability,
        confidence=1.0 - (best_variability / variability_threshold)  # Higher variability = lower confidence
    )


def _calculate_fallback_baseline(fhr: np.ndarray) -> BaselineResult:
    """
    Calculate fallback baseline using global mean.
    
    Used when no stable segment is found or signal is too short.
    
    Args:
        fhr: FHR signal array.
        
    Returns:
        BaselineResult with reduced confidence.
    """
    valid_fhr = fhr[~np.isnan(fhr)]
    
    if len(valid_fhr) == 0:
        # Return default normal baseline if no valid data
        logger.error("No valid FHR data for baseline calculation")
        return BaselineResult(
            value=140.0,  # Default normal value
            is_normal=True,
            is_bradycardia=False,
            is_tachycardia=False,
            stable_segment_found=False,
            confidence=0.0
        )
    
    baseline_raw = float(np.mean(valid_fhr))
    baseline_rounded = _round_to_nearest_5(baseline_raw)
    
    return BaselineResult(
        value=baseline_rounded,
        is_normal=110 <= baseline_rounded <= 160,
        is_bradycardia=baseline_rounded < 110,
        is_tachycardia=baseline_rounded > 160,
        stable_segment_found=False,
        segment_variability=float(np.max(valid_fhr) - np.min(valid_fhr)),
        confidence=0.5  # Reduced confidence for fallback
    )


def _round_to_nearest_5(value: float) -> float:
    """
    Round a value to the nearest multiple of 5.
    
    Args:
        value: Value to round.
        
    Returns:
        Value rounded to nearest 5.
        
    Example:
        >>> _round_to_nearest_5(142.3)
        140.0
        >>> _round_to_nearest_5(143.5)
        145.0
    """
    return round(value / 5) * 5
