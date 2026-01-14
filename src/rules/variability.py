"""
FHR Variability Calculator.

Implements variability calculation according to the Israeli Position Paper.

Definition:
    Variability is the fluctuation in the baseline FHR that is irregular
    in amplitude and frequency. It is quantified as the amplitude range
    (max - min) within a 1-minute window.

Categories (per Israeli Position Paper):
    - Absent:   ≤ 2 bpm  (severe - associated with fetal acidemia)
    - Minimal:  3-5 bpm  (requires monitoring)
    - Moderate: 6-25 bpm (NORMAL - indicates intact fetal autonomic nervous system)
    - Marked:   > 25 bpm (elevated - may indicate fetal hypoxia or infection)

Clinical Significance:
    Moderate variability is the single most reliable indicator of fetal well-being.
    Absent variability, especially with late/variable decelerations, indicates
    high risk of fetal acidemia.

References:
    - Israeli Position Paper on CTG Interpretation
    - SentinelFetal Gen3.5 Technical Specification, Section 5.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.config import CTG, THRESHOLDS

# Configure module logger
logger = logging.getLogger(__name__)


class VariabilityCalculationError(Exception):
    """Raised when variability calculation fails."""
    pass


class VariabilityCategory(Enum):
    """
    Variability categories per Israeli Position Paper.
    
    Clinical interpretation:
        ABSENT:   Severe finding - associated with fetal acidemia
        MINIMAL:  Concerning - requires monitoring
        MODERATE: Normal - indicates intact autonomic nervous system
        MARKED:   Elevated - may indicate hypoxia or infection
        UNKNOWN:  Unable to calculate
    """
    ABSENT = "Absent"      # 0-2 bpm
    MINIMAL = "Minimal"    # 3-5 bpm
    MODERATE = "Moderate"  # 6-25 bpm (NORMAL)
    MARKED = "Marked"      # > 25 bpm
    UNKNOWN = "Unknown"    # Unable to calculate
    
    @classmethod
    def from_value(cls, value: float) -> VariabilityCategory:
        """
        Determine category from variability value.
        
        Args:
            value: Variability amplitude in bpm.
            
        Returns:
            Corresponding VariabilityCategory.
        """
        if value <= THRESHOLDS.VARIABILITY_ABSENT_MAX:
            return cls.ABSENT
        elif value <= THRESHOLDS.VARIABILITY_MINIMAL_MAX:
            return cls.MINIMAL
        elif value <= THRESHOLDS.VARIABILITY_MODERATE_MAX:
            return cls.MODERATE
        else:
            return cls.MARKED


@dataclass
class VariabilityResult:
    """
    Result of variability calculation.
    
    Attributes:
        value: Average variability amplitude in bpm.
        category: Variability category (Absent/Minimal/Moderate/Marked).
        is_normal: True if variability is Moderate (6-25 bpm).
        is_concerning: True if variability is Absent or Minimal.
        window_values: List of variability values for each 1-minute window.
        min_value: Minimum variability across all windows.
        max_value: Maximum variability across all windows.
        n_windows: Number of valid 1-minute windows analyzed.
    """
    
    value: float
    category: VariabilityCategory
    is_normal: bool
    is_concerning: bool
    window_values: list[float]
    min_value: float
    max_value: float
    n_windows: int
    
    def __repr__(self) -> str:
        return (
            f"VariabilityResult(value={self.value:.1f} bpm, "
            f"category={self.category.value})"
        )


def calculate_variability(
    fhr: np.ndarray,
    sampling_rate: float = 4.0,
    window_seconds: float = 60.0,
    overlap_ratio: float = 0.5,
    min_valid_ratio: float = 0.5
) -> VariabilityResult:
    """
    Calculate FHR variability according to the Israeli Position Paper.
    
    Algorithm:
        1. Divide signal into 1-minute windows (with 50% overlap)
        2. For each window, calculate amplitude (max - min)
        3. Average the amplitudes across all windows
        4. Classify according to categories
    
    Args:
        fhr: FHR signal array in bpm.
        sampling_rate: Sampling frequency in Hz (default: 4.0).
        window_seconds: Window duration in seconds (default: 60.0 = 1 minute).
        overlap_ratio: Overlap between consecutive windows (default: 0.5 = 50%).
        min_valid_ratio: Minimum ratio of valid (non-NaN) samples per window (default: 0.5).
        
    Returns:
        VariabilityResult with calculated variability and classification.
        
    Raises:
        VariabilityCalculationError: If calculation fails due to invalid input.
        
    Example:
        >>> result = calculate_variability(fhr_signal, sampling_rate=4.0)
        >>> print(f"Variability: {result.value:.1f} bpm ({result.category.value})")
        >>> if result.is_concerning:
        ...     print("Warning: Abnormal variability detected")
    """
    # Validate input
    if fhr is None or len(fhr) == 0:
        raise VariabilityCalculationError("Input FHR signal is empty or None")
    
    # Calculate window parameters
    window_samples = int(window_seconds * sampling_rate)  # 60 * 4 = 240
    step_samples = int(window_samples * (1 - overlap_ratio))  # 240 * 0.5 = 120
    
    # Check if signal is long enough
    if len(fhr) < window_samples:
        logger.warning(
            f"Signal too short ({len(fhr)} samples) for {window_seconds}s window. "
            "Using entire signal."
        )
        return _calculate_single_window_variability(fhr)
    
    # Calculate variability for each window
    window_variabilities: list[float] = []
    
    for start in range(0, len(fhr) - window_samples + 1, step_samples):
        end = start + window_samples
        segment = fhr[start:end]
        
        # Filter out NaN values
        valid_segment = segment[~np.isnan(segment)]
        
        # Require minimum valid data in segment
        if len(valid_segment) < window_samples * min_valid_ratio:
            continue
        
        # Calculate variability (amplitude = max - min)
        variability = float(np.max(valid_segment) - np.min(valid_segment))
        window_variabilities.append(variability)
    
    # Check if we have valid windows
    if not window_variabilities:
        logger.warning("No valid windows for variability calculation")
        return _create_unknown_result()
    
    # Calculate statistics
    avg_variability = float(np.mean(window_variabilities))
    min_variability = float(np.min(window_variabilities))
    max_variability = float(np.max(window_variabilities))
    
    # Classify
    category = VariabilityCategory.from_value(avg_variability)
    is_normal = category == VariabilityCategory.MODERATE
    is_concerning = category in (VariabilityCategory.ABSENT, VariabilityCategory.MINIMAL)
    
    logger.info(
        f"Variability calculated: {avg_variability:.1f} bpm ({category.value}), "
        f"from {len(window_variabilities)} windows"
    )
    
    return VariabilityResult(
        value=round(avg_variability, 1),
        category=category,
        is_normal=is_normal,
        is_concerning=is_concerning,
        window_values=window_variabilities,
        min_value=round(min_variability, 1),
        max_value=round(max_variability, 1),
        n_windows=len(window_variabilities)
    )


def _calculate_single_window_variability(fhr: np.ndarray) -> VariabilityResult:
    """
    Calculate variability from entire signal as single window.
    
    Used when signal is shorter than one window.
    
    Args:
        fhr: FHR signal array.
        
    Returns:
        VariabilityResult calculated from entire signal.
    """
    valid_fhr = fhr[~np.isnan(fhr)]
    
    if len(valid_fhr) == 0:
        return _create_unknown_result()
    
    variability = float(np.max(valid_fhr) - np.min(valid_fhr))
    category = VariabilityCategory.from_value(variability)
    
    return VariabilityResult(
        value=round(variability, 1),
        category=category,
        is_normal=category == VariabilityCategory.MODERATE,
        is_concerning=category in (VariabilityCategory.ABSENT, VariabilityCategory.MINIMAL),
        window_values=[variability],
        min_value=round(variability, 1),
        max_value=round(variability, 1),
        n_windows=1
    )


def _create_unknown_result() -> VariabilityResult:
    """
    Create a result indicating unknown/uncalculable variability.
    
    Returns:
        VariabilityResult with UNKNOWN category.
    """
    return VariabilityResult(
        value=0.0,
        category=VariabilityCategory.UNKNOWN,
        is_normal=False,
        is_concerning=True,  # Unknown is concerning by default
        window_values=[],
        min_value=0.0,
        max_value=0.0,
        n_windows=0
    )


def classify_variability(value: float) -> VariabilityCategory:
    """
    Classify a variability value into a category.
    
    Convenience function for external use.
    
    Args:
        value: Variability amplitude in bpm.
        
    Returns:
        VariabilityCategory based on value.
        
    Example:
        >>> category = classify_variability(8.5)
        >>> print(category)  # VariabilityCategory.MODERATE
    """
    return VariabilityCategory.from_value(value)
