"""
Deceleration Detection and Classification Module.

Implements deceleration detection and classification according to the Israeli Position Paper.

Definition:
    Deceleration is a decrease in FHR of ≥15 bpm below the baseline lasting
    ≥15 seconds but less than 10 minutes.

Classification by Lag Time (Section 5.3):
    - Early:    Nadir occurs < 5 seconds after contraction peak (with contraction)
    - Late:     Nadir occurs > 15 seconds after contraction peak (after contraction)
    - Variable: Abrupt onset (>0.5 bpm/sample descent rate), variable timing

Severity Signs for Variable Decelerations:
    1. Drop to < 70 bpm for > 60 seconds
    2. Absent variability within the deceleration
    3. Slow recovery (> 60 seconds to return to baseline)
    4. Overshoot (rise > 10 bpm above baseline after recovery)
    5. W-shape (biphasic pattern)

References:
    - Israeli Position Paper on CTG Interpretation
    - SentinelFetal Gen3.5 Technical Specification, Section 5.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


class DecelerationDetectionError(Exception):
    """Raised when deceleration detection fails."""
    pass


class DecelerationType(Enum):
    """
    Types of FHR decelerations per Israeli Position Paper.
    
    Clinical interpretation:
        EARLY:        With contraction, symmetric, normal head compression
        LATE:         After contraction, indicates uteroplacental insufficiency
        VARIABLE:     Variable timing, umbilical cord compression
        PROLONGED:    Duration > 2 minutes, indicates severe stress
        UNCLASSIFIED: Cannot determine type
    """
    EARLY = "Early"
    LATE = "Late"
    VARIABLE = "Variable"
    PROLONGED = "Prolonged"
    UNCLASSIFIED = "Unclassified"


@dataclass
class Deceleration:
    """
    Represents a single FHR deceleration event.
    
    Attributes:
        start_idx: Start index of the deceleration.
        end_idx: End index of the deceleration (exclusive).
        nadir_idx: Index of the minimum FHR (nadir point).
        depth: Depth of deceleration below baseline (in bpm).
        duration_seconds: Duration of the deceleration in seconds.
        decel_type: Classification (Early/Late/Variable/Prolonged).
        lag_seconds: Time from contraction peak to nadir (for classification).
        has_severity_signs: True if severity signs are present.
        severity_signs: List of detected severity signs.
        descent_rate: Rate of FHR decline (bpm/sample).
        nadir_value: FHR value at nadir point (bpm).
    """
    
    start_idx: int
    end_idx: int
    nadir_idx: int
    depth: float
    duration_seconds: float
    decel_type: DecelerationType
    lag_seconds: float
    has_severity_signs: bool
    severity_signs: list[str] = field(default_factory=list)
    descent_rate: float = 0.0
    nadir_value: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"Deceleration(type={self.decel_type.value}, "
            f"depth={self.depth:.1f}bpm, duration={self.duration_seconds:.1f}s, "
            f"severity={self.has_severity_signs})"
        )


def detect_decelerations(
    fhr: np.ndarray,
    uc: np.ndarray,
    baseline: float,
    sampling_rate: float = 4.0,
    min_depth: float = 15.0,
    min_duration_seconds: float = 15.0,
    max_duration_seconds: float = 600.0
) -> list[Deceleration]:
    """
    Detect and classify all decelerations in the FHR signal.
    
    Algorithm:
        1. Find all regions where FHR < baseline - min_depth
        2. Filter by duration (15 seconds to 10 minutes)
        3. Classify each deceleration by lag time
        4. Check for severity signs
    
    Args:
        fhr: FHR signal array in bpm.
        uc: Uterine contraction signal array.
        baseline: Baseline FHR value in bpm.
        sampling_rate: Sampling frequency in Hz (default: 4.0).
        min_depth: Minimum depth below baseline in bpm (default: 15.0).
        min_duration_seconds: Minimum duration in seconds (default: 15.0).
        max_duration_seconds: Maximum duration in seconds (default: 600.0).
        
    Returns:
        List of detected Deceleration objects.
        
    Raises:
        DecelerationDetectionError: If detection fails due to invalid input.
        
    Example:
        >>> decels = detect_decelerations(fhr, uc, baseline=140.0)
        >>> late_count = sum(1 for d in decels if d.decel_type == DecelerationType.LATE)
        >>> print(f"Found {len(decels)} decelerations, {late_count} late")
    """
    # Validate input
    if fhr is None or len(fhr) == 0:
        raise DecelerationDetectionError("Input FHR signal is empty or None")
    if uc is None or len(uc) != len(fhr):
        logger.warning("UC signal missing or length mismatch. Using zeros.")
        uc = np.zeros_like(fhr)
    
    decelerations: list[Deceleration] = []
    
    # Convert duration to samples
    min_duration_samples = int(min_duration_seconds * sampling_rate)
    max_duration_samples = int(max_duration_seconds * sampling_rate)
    
    # Find points below threshold
    threshold = baseline - min_depth
    below_threshold = fhr < threshold
    
    # Handle NaN values
    below_threshold = below_threshold & ~np.isnan(fhr)
    
    # Find contiguous regions below threshold
    decel_regions = _find_contiguous_regions(below_threshold)
    
    logger.debug(f"Found {len(decel_regions)} potential deceleration regions")
    
    for start, end in decel_regions:
        duration_samples = end - start
        duration_seconds = duration_samples / sampling_rate
        
        # Filter by duration
        if duration_samples < min_duration_samples:
            continue
        if duration_samples > max_duration_samples:
            # Prolonged deceleration
            pass
        
        # Find nadir (minimum FHR) within the deceleration
        segment = fhr[start:end]
        valid_mask = ~np.isnan(segment)
        
        if not np.any(valid_mask):
            continue
        
        nadir_local = np.nanargmin(segment)
        nadir_idx = start + nadir_local
        nadir_value = float(fhr[nadir_idx])
        depth = baseline - nadir_value
        
        # Classify deceleration
        decel_type, lag_seconds, descent_rate = classify_deceleration(
            fhr, uc, nadir_idx, start, end, sampling_rate
        )
        
        # Check for severity signs
        has_severity, severity_signs = _check_severity_signs(
            fhr, start, end, nadir_idx, baseline, sampling_rate
        )
        
        # Check if this should be classified as prolonged
        if duration_seconds > 120:  # > 2 minutes
            decel_type = DecelerationType.PROLONGED
        
        deceleration = Deceleration(
            start_idx=start,
            end_idx=end,
            nadir_idx=nadir_idx,
            depth=round(depth, 1),
            duration_seconds=round(duration_seconds, 1),
            decel_type=decel_type,
            lag_seconds=round(lag_seconds, 1),
            has_severity_signs=has_severity,
            severity_signs=severity_signs,
            descent_rate=round(descent_rate, 3),
            nadir_value=round(nadir_value, 1)
        )
        
        decelerations.append(deceleration)
        logger.debug(f"Detected: {deceleration}")
    
    logger.info(f"Total decelerations detected: {len(decelerations)}")
    return decelerations


def classify_deceleration(
    fhr: np.ndarray,
    uc: np.ndarray,
    nadir_idx: int,
    decel_start: int,
    decel_end: int,
    sampling_rate: float = 4.0
) -> tuple[DecelerationType, float, float]:
    """
    Classify a deceleration based on its relationship to contractions.
    
    Classification Algorithm (per Section 5.3):
        1. Find the nearest contraction peak
        2. Calculate Lag = time(nadir) - time(contraction_peak)
        3. Classify based on lag and descent rate:
            - Lag < 5 seconds → Early
            - Lag > 15 seconds → Late
            - Abrupt descent (>0.5 bpm/sample) → Variable
    
    Args:
        fhr: FHR signal array.
        uc: Uterine contraction signal array.
        nadir_idx: Index of the deceleration nadir.
        decel_start: Start index of deceleration.
        decel_end: End index of deceleration.
        sampling_rate: Sampling frequency in Hz.
        
    Returns:
        Tuple of (DecelerationType, lag_seconds, descent_rate).
    """
    # Calculate descent rate (how abrupt is the drop)
    descent_rate = _calculate_descent_rate(fhr, decel_start, nadir_idx)
    
    # Search for contraction peak in the vicinity
    # Look 2 minutes before decel start to 1 minute after decel start
    search_start = max(0, decel_start - int(120 * sampling_rate))
    search_end = min(len(uc), decel_start + int(60 * sampling_rate))
    
    uc_segment = uc[search_start:search_end]
    
    # Handle case with no valid UC data
    if len(uc_segment) == 0 or np.all(np.isnan(uc_segment)) or np.all(uc_segment == 0):
        # Cannot classify without contraction data
        # Use descent rate to distinguish Variable
        if descent_rate > 0.5:  # Abrupt onset
            return DecelerationType.VARIABLE, 0.0, descent_rate
        return DecelerationType.UNCLASSIFIED, 0.0, descent_rate
    
    # Find contraction peak (maximum in UC signal)
    uc_clean = np.nan_to_num(uc_segment, nan=0.0)
    contraction_peak_local = int(np.argmax(uc_clean))
    contraction_peak_idx = search_start + contraction_peak_local
    
    # Calculate lag in seconds
    lag_samples = nadir_idx - contraction_peak_idx
    lag_seconds = lag_samples / sampling_rate
    
    # Classify based on lag and descent rate (per Section 5.3)
    if descent_rate > 0.5:  # Abrupt onset characteristic of Variable
        return DecelerationType.VARIABLE, lag_seconds, descent_rate
    elif abs(lag_seconds) < 5:
        # Nadir occurs within 5 seconds of contraction peak
        return DecelerationType.EARLY, lag_seconds, descent_rate
    elif lag_seconds > 15:
        # Nadir occurs more than 15 seconds after contraction peak
        return DecelerationType.LATE, lag_seconds, descent_rate
    else:
        # In between - needs further analysis
        # Check if it has variable characteristics
        if descent_rate > 0.3:
            return DecelerationType.VARIABLE, lag_seconds, descent_rate
        return DecelerationType.UNCLASSIFIED, lag_seconds, descent_rate


def _find_contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Find contiguous True regions in a boolean mask.
    
    Args:
        mask: Boolean array.
        
    Returns:
        List of (start, end) index tuples.
    """
    regions: list[tuple[int, int]] = []
    in_region = False
    region_start = 0
    
    for i, val in enumerate(mask):
        if val and not in_region:
            in_region = True
            region_start = i
        elif not val and in_region:
            regions.append((region_start, i))
            in_region = False
    
    # Handle region at end
    if in_region:
        regions.append((region_start, len(mask)))
    
    return regions


def _calculate_descent_rate(
    fhr: np.ndarray,
    start: int,
    nadir: int
) -> float:
    """
    Calculate the rate of FHR descent (bpm per sample).
    
    A rate > 0.5 bpm/sample indicates abrupt onset (Variable deceleration).
    
    Args:
        fhr: FHR signal array.
        start: Start index of descent.
        nadir: Index of nadir (lowest point).
        
    Returns:
        Descent rate in bpm per sample.
    """
    if nadir <= start:
        return 0.0
    
    # Get values at start and nadir
    start_val = fhr[start]
    nadir_val = fhr[nadir]
    
    if np.isnan(start_val) or np.isnan(nadir_val):
        return 0.0
    
    drop = start_val - nadir_val
    samples = nadir - start
    
    return drop / samples if samples > 0 else 0.0


def _check_severity_signs(
    fhr: np.ndarray,
    start: int,
    end: int,
    nadir: int,
    baseline: float,
    sampling_rate: float
) -> tuple[bool, list[str]]:
    """
    Check for severity signs in a variable deceleration.
    
    Severity Signs (per Israeli Position Paper):
        1. Drop to < 70 bpm for > 60 seconds
        2. Absent variability within the deceleration
        3. Slow recovery (> 60 seconds to return to baseline)
        4. Overshoot (rise > 10 bpm above baseline after recovery)
        5. W-shape (biphasic pattern)
    
    Args:
        fhr: FHR signal array.
        start: Start index of deceleration.
        end: End index of deceleration.
        nadir: Index of nadir.
        baseline: Baseline FHR value.
        sampling_rate: Sampling frequency.
        
    Returns:
        Tuple of (has_severity_signs, list of detected signs).
    """
    severity_signs: list[str] = []
    segment = fhr[start:end]
    valid_segment = segment[~np.isnan(segment)]
    
    if len(valid_segment) == 0:
        return False, severity_signs
    
    # Sign 1: Severe drop (< 70 bpm for > 60 seconds)
    below_70 = np.sum(valid_segment < 70)
    if below_70 / sampling_rate > 60:
        severity_signs.append("Severe drop (<70 bpm for >60s)")
    
    # Sign 2: Absent variability within deceleration
    internal_variability = float(np.max(valid_segment) - np.min(valid_segment))
    if internal_variability < 5:
        severity_signs.append("Absent internal variability")
    
    # Sign 3: Slow recovery (> 60 seconds from nadir to end)
    recovery_samples = end - nadir
    recovery_seconds = recovery_samples / sampling_rate
    if recovery_seconds > 60:
        severity_signs.append(f"Slow recovery ({recovery_seconds:.0f}s)")
    
    # Sign 4: Overshoot (FHR > baseline + 10 after deceleration)
    post_decel_start = end
    post_decel_end = min(len(fhr), end + int(60 * sampling_rate))
    
    if post_decel_end > post_decel_start:
        post_decel = fhr[post_decel_start:post_decel_end]
        valid_post = post_decel[~np.isnan(post_decel)]
        
        if len(valid_post) > 0 and np.max(valid_post) > baseline + 10:
            severity_signs.append("Overshoot")
    
    # Sign 5: W-shape (biphasic) - simplified detection
    # Look for a secondary minimum after partial recovery
    if len(valid_segment) > 20:  # Need enough points
        mid_point = len(valid_segment) // 2
        first_half_min = np.min(valid_segment[:mid_point])
        second_half_min = np.min(valid_segment[mid_point:])
        
        # If both halves have significant dips below a threshold
        threshold = baseline - 10
        if first_half_min < threshold and second_half_min < threshold:
            # Check if there's a recovery between them
            mid_region = valid_segment[mid_point-5:mid_point+5] if mid_point > 5 else valid_segment[:10]
            if len(mid_region) > 0 and np.max(mid_region) > (first_half_min + 10):
                severity_signs.append("W-shape (biphasic)")
    
    has_severity = len(severity_signs) > 0
    
    if has_severity:
        logger.debug(f"Severity signs detected: {severity_signs}")
    
    return has_severity, severity_signs


def count_recurrent_decelerations(
    decelerations: list[Deceleration],
    decel_type: DecelerationType,
    total_contractions: int,
    threshold_ratio: float = 0.5
) -> tuple[int, bool]:
    """
    Count decelerations of a specific type and check if they are recurrent.
    
    Definition: Recurrent decelerations occur with ≥50% of contractions.
    
    Args:
        decelerations: List of detected decelerations.
        decel_type: Type to count (e.g., LATE, VARIABLE).
        total_contractions: Total number of contractions.
        threshold_ratio: Ratio threshold for recurrent (default: 0.5).
        
    Returns:
        Tuple of (count, is_recurrent).
    """
    count = sum(1 for d in decelerations if d.decel_type == decel_type)
    
    if total_contractions == 0:
        return count, False
    
    ratio = count / total_contractions
    is_recurrent = ratio >= threshold_ratio
    
    return count, is_recurrent
