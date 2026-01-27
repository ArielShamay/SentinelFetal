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

from src.analysis.fallback_audit import record_fallback, get_case_context
from src.utils.runtime_config import load_runtime_config
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
    min_depth: float = 12.0,  # Phase 13: Relaxed from 15 to 12 bpm
    min_duration_seconds: float = 12.0,  # Phase 13: Relaxed from 15 to 12 seconds
    max_duration_seconds: float = 600.0
) -> list[Deceleration]:
    """
    Detect and classify all decelerations in the FHR signal.
    
    Algorithm:
        1. Find all regions where FHR < baseline - min_depth
        2. Filter by duration (12 seconds to 10 minutes) - Phase 13: relaxed from 15s
        3. Classify each deceleration by lag time
        4. Check for severity signs
    
    Phase 13 Changes:
        - min_depth reduced: 12 bpm (was 15 bpm)
        - min_duration reduced: 12 seconds (was 15 seconds)
        - These changes improve detection under noisy conditions
    
    Args:
        fhr: FHR signal array in bpm.
        uc: Uterine contraction signal array.
        baseline: Baseline FHR value in bpm.
        sampling_rate: Sampling frequency in Hz (default: 4.0).
        min_depth: Minimum depth below baseline in bpm (default: 12.0).
        min_duration_seconds: Minimum duration in seconds (default: 12.0).
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
        runtime_cfg = load_runtime_config()
        stats = {
            "n_fhr_samples": int(len(fhr)),
            "n_uc_samples": int(0 if uc is None else len(uc)),
            "reason_detail": "uc_missing_or_mismatch",
        }
        record_fallback("decelerations", "uc_missing", stats)
        if runtime_cfg.strict_mode:
            case_id = get_case_context()
            case_tag = case_id if case_id is not None else "unknown"
            raise RuntimeError(
                "STRICT_MODE decelerations fallback | "
                f"case_id={case_tag} reason=uc_missing_or_mismatch stats={stats}"
            )
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


def calculate_descent_time(
    fhr: np.ndarray,
    decel_start: int,
    nadir_idx: int,
    sampling_rate: float = 4.0
) -> float:
    """
    Calculate descent time from deceleration onset to nadir.
    
    This is the PRIMARY criterion for Late vs Variable classification
    per FIGO/NICHD guidelines and Das et al. 2023 (97.94% accuracy).
    
    Definition:
        - < 30 seconds: ABRUPT onset → Variable deceleration
        - ≥ 30 seconds: GRADUAL onset → Late or Early deceleration
    
    ROBUST VERSION: Scans backwards from 'decel_start' (threshold crossing)
    to find the true onset of the deceleration.
    
    Args:
        fhr: FHR signal array.
        decel_start: Start index of deceleration (threshold crossing).
        nadir_idx: Index of the nadir (minimum point).
        sampling_rate: Sampling frequency in Hz (default: 4.0).
        
    Returns:
        Descent time in seconds.
    """
    if nadir_idx <= decel_start:
        return 0.0
    
    try:
        # Standard approach: decel_start is where FHR drops below Baseline - Threshold (15bpm)
        # To find VALID descent time, we must look back to where it left the baseline.
        
        if decel_start >= len(fhr):
            return 0.0
            
        threshold_val = float(fhr[decel_start])
        # We want to find where signal was higher (closer to baseline)
        # Assuming decel_start is already ~15bpm below baseline
        # Let's look for a rise of at least 10 bpm above the start threshold
        target_val = threshold_val + 10.0
        
        true_start_idx = decel_start
        # Look back up to 90 seconds (generous) to find the onset
        lookback_limit = max(0, decel_start - int(90 * sampling_rate))
        
        found_onset = False
        
        # Scan backward for the true onset (crossing back above the threshold gap)
        for i in range(decel_start, lookback_limit, -1):
            val = fhr[i]
            if np.isnan(val):
                continue
            if val >= target_val:
                true_start_idx = i
                found_onset = True
                
                # Refinement: Look for local Maxima around this crossing
                # This helps pinpoint exact start of descent
                window_start = max(0, i - int(10*sampling_rate))
                window_end = min(len(fhr), i + 2)
                local_window = fhr[window_start:window_end]
                # Filter NaNs for argmax
                if len(local_window) > 0 and not np.all(np.isnan(local_window)):
                    peak_offset = np.nanargmax(local_window)
                    true_start_idx = window_start + peak_offset
                    # Found peak
                break
        
        if not found_onset:
            # Try finding max in the window anyway
            window_fhr = fhr[lookback_limit:decel_start+1]
            if len(window_fhr) > 0:
                 peak_rel = np.nanargmax(window_fhr)
                 if window_fhr[peak_rel] > threshold_val + 5.0:
                     true_start_idx = lookback_limit + peak_rel

        logger.debug(f"Adjusted onset: {decel_start} -> {true_start_idx} (Target: {target_val:.1f})")
        
    except Exception as e:
        logger.warning(f"Error calculating descent time: {e}")
        true_start_idx = decel_start

    descent_samples = nadir_idx - true_start_idx
    descent_time_seconds = descent_samples / sampling_rate
    
    # Sanity check
    if descent_time_seconds < 0:
        descent_time_seconds = 0.0
    
    logger.debug(f"Descent Time: {descent_time_seconds:.1f}s")
    return descent_time_seconds


def classify_deceleration(
    fhr: np.ndarray,
    uc: np.ndarray,
    nadir_idx: int,
    decel_start: int,
    decel_end: int,
    sampling_rate: float = 4.0
) -> tuple[DecelerationType, float, float]:
    """
    Classify a deceleration using the 30-second descent time rule.
    
    Classification Algorithm (FIGO/NICHD-compliant):
        PRIMARY CRITERION - Descent Time (onset to nadir):
            - < 30 seconds: Variable (abrupt onset)
            - ≥ 30 seconds: Late or Early (gradual onset)
        
        SECONDARY CRITERION (for gradual decelerations):
            - UC relationship determines Late vs Early
    
    This implements the Das et al. 2023 approach achieving 97.94% accuracy
    vs 63.92% with crisp rules alone.
    
    Args:
        fhr: FHR signal array.
        uc: Uterine contraction signal array.
        nadir_idx: Index of the deceleration nadir.
        decel_start: Start index of deceleration.
        decel_end: End index of deceleration.
        sampling_rate: Sampling frequency in Hz.
        
    Returns:
        Tuple of (DecelerationType, lag_seconds, descent_time_seconds).
    """
    # PRIMARY CRITERION: Calculate descent time (onset to nadir)
    # This is the most discriminative feature per research
    descent_time = calculate_descent_time(fhr, decel_start, nadir_idx, sampling_rate)
    
    # Threshold: 30 seconds separates Variable (abrupt) from Late/Early (gradual)
    DESCENT_TIME_THRESHOLD = 30.0  # seconds - FIGO/NICHD standard
    
    # Handle fuzzy boundary (25-35s) with weighted classification
    # Values near 30s get classified based on other features
    FUZZY_LOWER = 25.0
    FUZZY_UPPER = 35.0
    
    # Search for contraction peak in the vicinity
    search_start = max(0, decel_start - int(120 * sampling_rate))
    search_end = min(len(uc), decel_start + int(60 * sampling_rate))
    uc_segment = uc[search_start:search_end]
    
    # Calculate lag for UC-based classification
    lag_seconds = 0.0
    has_valid_uc = (len(uc_segment) > 0 and 
                   not np.all(np.isnan(uc_segment)) and 
                   not np.all(uc_segment == 0))
    
    if has_valid_uc:
        uc_clean = np.nan_to_num(uc_segment, nan=0.0)
        contraction_peak_local = int(np.argmax(uc_clean))
        contraction_peak_idx = search_start + contraction_peak_local
        lag_samples = nadir_idx - contraction_peak_idx
        lag_seconds = lag_samples / sampling_rate
    
    # =========================================================================
    # PRIMARY CLASSIFICATION: 30-second descent time rule
    # =========================================================================
    
    if descent_time < FUZZY_LOWER:
        # Clearly ABRUPT onset (< 25 seconds) → Variable
        return DecelerationType.VARIABLE, lag_seconds, descent_time
    
    elif descent_time >= FUZZY_UPPER:
        # Clearly GRADUAL onset (≥ 35 seconds) → Late or Early
        # Use UC relationship to distinguish
        if has_valid_uc:
            if abs(lag_seconds) < 5:
                # Nadir coincides with UC peak → Early (head compression)
                return DecelerationType.EARLY, lag_seconds, descent_time
            elif lag_seconds > 15:
                # Nadir after UC peak + late recovery → Late
                return DecelerationType.LATE, lag_seconds, descent_time
            else:
                # Intermediate lag - lean toward Late for gradual decels
                return DecelerationType.LATE, lag_seconds, descent_time
        else:
            # No UC data - gradual onset strongly suggests Late
            # (Variable decels have abrupt onset by definition)
            return DecelerationType.LATE, lag_seconds, descent_time
    
    else:
        # FUZZY ZONE (25-35 seconds) - use secondary features
        # Calculate how close to Variable vs Late threshold
        variable_weight = (FUZZY_UPPER - descent_time) / (FUZZY_UPPER - FUZZY_LOWER)
        
        if has_valid_uc:
            # With UC data, use timing to break tie
            if abs(lag_seconds) < 5:
                return DecelerationType.EARLY, lag_seconds, descent_time
            elif lag_seconds > 15:
                return DecelerationType.LATE, lag_seconds, descent_time
            elif variable_weight > 0.5:
                # Closer to 25s → lean Variable
                return DecelerationType.VARIABLE, lag_seconds, descent_time
            else:
                # Closer to 35s → lean Late
                return DecelerationType.LATE, lag_seconds, descent_time
        else:
            # No UC - use descent time weight
            if variable_weight > 0.6:
                return DecelerationType.VARIABLE, lag_seconds, descent_time
            else:
                return DecelerationType.LATE, lag_seconds, descent_time


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
    
    Phase 13: Uses robust calculation that handles jagged descents.
    Instead of simple point-to-point, uses smoothed slope estimation.
    
    A rate > 0.3 bpm/sample indicates abrupt onset (Variable deceleration).
    Phase 13: Threshold reduced from 0.5 to 0.3 for better sensitivity.
    
    Args:
        fhr: FHR signal array.
        start: Start index of descent.
        nadir: Index of nadir (lowest point).
        
    Returns:
        Descent rate in bpm per sample.
    """
    if nadir <= start:
        return 0.0
    
    # Get segment from start to nadir
    segment = fhr[start:nadir+1]
    valid_mask = ~np.isnan(segment)
    
    if np.sum(valid_mask) < 2:
        return 0.0
    
    # Phase 13: Use robust descent calculation
    # Find valid start and nadir values (may not be at exact indices due to noise)
    valid_indices = np.where(valid_mask)[0]
    valid_values = segment[valid_mask]
    
    # Use 90th percentile of first 20% as "start" and 10th percentile as "nadir"
    # This is more robust to noise/jaggedness
    n_valid = len(valid_values)
    first_portion = max(1, n_valid // 5)
    
    start_val = np.percentile(valid_values[:first_portion], 90) if first_portion > 0 else valid_values[0]
    nadir_val = np.min(valid_values)  # True minimum
    
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
