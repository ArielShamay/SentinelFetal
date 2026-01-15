"""
Adapters for rule engine components.
These wrap existing functions to conform to Protocol interfaces.

Each adapter wraps an existing implementation and exposes it through
the standardized interface, allowing for dependency injection.
"""

from typing import List
import numpy as np

from src.interfaces.protocols import (
    IBaselineCalculator,
    IBaselineResult,
    IVariabilityCalculator,
    IVariabilityResult,
    IDecelerationDetector,
    IDeceleration,
    ITachysystoleDetector,
    ITachysystoleResult,
    ISinusoidalDetector,
    ISinusoidalResult,
)

# Import existing implementations
from src.rules.baseline import calculate_baseline, BaselineResult
from src.rules.variability import calculate_variability, VariabilityResult
from src.rules.decelerations import detect_decelerations, Deceleration
from src.rules.tachysystole import detect_tachysystole, TachysystoleResult
from src.rules.sinusoidal import detect_sinusoidal_pattern, SinusoidalResult


class BaselineAdapter(IBaselineCalculator):
    """
    Adapter for baseline calculation.
    
    Wraps the calculate_baseline function to conform to IBaselineCalculator.
    
    Args:
        window_minutes: Minimum stable segment duration in minutes (default: 2.0).
        variability_threshold: Maximum variability for stable segment in bpm (default: 25.0).
        
    Example:
        >>> adapter = BaselineAdapter(window_minutes=2.0)
        >>> result = adapter.calculate(fhr, sampling_rate=4.0)
        >>> print(f"Baseline: {result.value} bpm")
    """
    
    def __init__(
        self,
        window_minutes: float = 2.0,
        variability_threshold: float = 25.0
    ):
        self._window_minutes = window_minutes
        self._variability_threshold = variability_threshold
    
    def calculate(self, fhr: np.ndarray, sampling_rate: float) -> IBaselineResult:
        """Calculate baseline using existing implementation."""
        return calculate_baseline(
            fhr,
            sampling_rate,
            window_minutes=self._window_minutes,
            variability_threshold=self._variability_threshold
        )


class VariabilityAdapter(IVariabilityCalculator):
    """
    Adapter for variability calculation.
    
    Wraps the calculate_variability function to conform to IVariabilityCalculator.
    
    Args:
        window_seconds: Window size for variability calculation in seconds (default: 60.0).
        overlap_ratio: Overlap ratio between windows (default: 0.5).
        
    Example:
        >>> adapter = VariabilityAdapter(window_seconds=60.0)
        >>> result = adapter.calculate(fhr, sampling_rate=4.0)
        >>> print(f"Variability: {result.value} bpm ({result.category.value})")
    """
    
    def __init__(
        self,
        window_seconds: float = 60.0,
        overlap_ratio: float = 0.5
    ):
        self._window_seconds = window_seconds
        self._overlap_ratio = overlap_ratio
    
    def calculate(self, fhr: np.ndarray, sampling_rate: float) -> IVariabilityResult:
        """Calculate variability using existing implementation."""
        return calculate_variability(
            fhr,
            sampling_rate,
            window_seconds=self._window_seconds,
            overlap_ratio=self._overlap_ratio
        )


class DecelerationAdapter(IDecelerationDetector):
    """
    Adapter for deceleration detection.
    
    Wraps the detect_decelerations function to conform to IDecelerationDetector.
    
    Args:
        min_depth: Minimum depth in bpm to consider as deceleration (default: 15.0).
        min_duration_seconds: Minimum duration in seconds (default: 15.0).
        max_duration_seconds: Maximum duration in seconds (default: 600.0).
        
    Example:
        >>> adapter = DecelerationAdapter(min_depth=15.0)
        >>> decels = adapter.detect(fhr, uc, baseline=140.0, sampling_rate=4.0)
        >>> print(f"Found {len(decels)} decelerations")
    """
    
    def __init__(
        self,
        min_depth: float = 15.0,
        min_duration_seconds: float = 15.0,
        max_duration_seconds: float = 600.0
    ):
        self._min_depth = min_depth
        self._min_duration = min_duration_seconds
        self._max_duration = max_duration_seconds
    
    def detect(
        self,
        fhr: np.ndarray,
        uc: np.ndarray,
        baseline: float,
        sampling_rate: float
    ) -> List[IDeceleration]:
        """Detect decelerations using existing implementation."""
        return detect_decelerations(
            fhr, uc, baseline, sampling_rate,
            min_depth=self._min_depth,
            min_duration_seconds=self._min_duration,
            max_duration_seconds=self._max_duration
        )


class TachysystoleAdapter(ITachysystoleDetector):
    """
    Adapter for tachysystole detection.
    
    Wraps the detect_tachysystole function to conform to ITachysystoleDetector.
    
    Args:
        analysis_window_minutes: Duration to analyze in minutes (default: 30.0).
        threshold_per_10min: Threshold contractions per 10 min (default: 5).
        
    Example:
        >>> adapter = TachysystoleAdapter(threshold_per_10min=5)
        >>> result = adapter.detect(uc, sampling_rate=4.0)
        >>> if result.detected:
        ...     print("Tachysystole detected!")
    """
    
    def __init__(
        self,
        analysis_window_minutes: float = 30.0,
        threshold_per_10min: int = 5
    ):
        self._window_minutes = analysis_window_minutes
        self._threshold = threshold_per_10min
    
    def detect(self, uc: np.ndarray, sampling_rate: float) -> ITachysystoleResult:
        """Detect tachysystole using existing implementation."""
        return detect_tachysystole(
            uc, sampling_rate,
            analysis_window_minutes=self._window_minutes,
            threshold_per_10min=self._threshold
        )


class SinusoidalAdapter(ISinusoidalDetector):
    """
    Adapter for sinusoidal pattern detection.
    
    Wraps the detect_sinusoidal_pattern function to conform to ISinusoidalDetector.
    
    Args:
        min_duration_minutes: Minimum duration to consider pattern (default: 20.0).
        freq_min: Minimum frequency in cycles per minute (default: 3.0).
        freq_max: Maximum frequency in cycles per minute (default: 5.0).
        
    Example:
        >>> adapter = SinusoidalAdapter(min_duration_minutes=20.0)
        >>> result = adapter.detect(fhr, sampling_rate=4.0)
        >>> if result.detected:
        ...     print("SEVERE: Sinusoidal pattern detected!")
    """
    
    def __init__(
        self,
        min_duration_minutes: float = 20.0,
        freq_min: float = 3.0,
        freq_max: float = 5.0
    ):
        self._min_duration = min_duration_minutes
        self._freq_min = freq_min
        self._freq_max = freq_max
    
    def detect(self, fhr: np.ndarray, sampling_rate: float) -> ISinusoidalResult:
        """Detect sinusoidal pattern using existing implementation."""
        return detect_sinusoidal_pattern(
            fhr, sampling_rate,
            min_duration_minutes=self._min_duration,
            freq_min_cycles_per_min=self._freq_min,
            freq_max_cycles_per_min=self._freq_max
        )
