"""
CTG Signal Preprocessing Module.

Implements preprocessing pipeline for CTG (Cardiotocography) signals including:
    - Out-of-range value handling (FHR: 50-240 BPM)
    - Spike detection and removal (>30 BPM change between samples)
    - Gap detection and filling (10-second rule per Israeli Position Paper)
    - Signal smoothing (Median filter)

The 10-Second Rule:
    Gaps ≤ 10 seconds (40 samples @ 4Hz) are filled with linear interpolation.
    Gaps > 10 seconds remain as NaN to indicate signal loss.

Example:
    >>> from src.data.preprocess import CTGPreprocessor, PreprocessingConfig
    >>> config = PreprocessingConfig(sampling_rate=4.0, max_gap_seconds=10.0)
    >>> preprocessor = CTGPreprocessor(config)
    >>> result = preprocessor.process(fhr_signal)
    >>> print(f"Filled {result.stats['filled_percent']:.1f}% of gaps")

References:
    - Israeli Position Paper on CTG Interpretation
    - SentinelFetal Gen3.5 Technical Specification, Section 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

# Configure module logger
logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""
    pass


class InvalidSignalError(PreprocessingError):
    """Raised when input signal is invalid."""
    pass


@dataclass
class PreprocessingConfig:
    """
    Configuration parameters for CTG preprocessing.
    
    Attributes:
        sampling_rate: Signal sampling frequency in Hz (default: 4.0).
        fhr_min: Minimum valid FHR in BPM (default: 50.0).
        fhr_max: Maximum valid FHR in BPM (default: 240.0).
        max_gap_seconds: Maximum gap duration to fill in seconds (default: 10.0).
        smoothing_window: Window size for median filter in samples (default: 5).
        spike_threshold: Maximum valid BPM change between consecutive samples (default: 30.0).
        
    Note:
        The 10-second rule is based on the Israeli Position Paper guidelines.
        At 4Hz sampling, 10 seconds = 40 samples.
    """
    
    sampling_rate: float = 4.0
    fhr_min: float = 50.0
    fhr_max: float = 240.0
    max_gap_seconds: float = 10.0
    smoothing_window: int = 5
    spike_threshold: float = 30.0
    
    @property
    def max_gap_samples(self) -> int:
        """
        Calculate maximum gap size in samples.
        
        Returns:
            Maximum gap size (10 seconds @ 4Hz = 40 samples).
        """
        return int(self.max_gap_seconds * self.sampling_rate)
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {self.sampling_rate}")
        if self.fhr_min >= self.fhr_max:
            raise ValueError(f"fhr_min ({self.fhr_min}) must be less than fhr_max ({self.fhr_max})")
        if self.max_gap_seconds < 0:
            raise ValueError(f"max_gap_seconds must be non-negative, got {self.max_gap_seconds}")


@dataclass  
class PreprocessingResult:
    """
    Result container for CTG signal preprocessing.
    
    Attributes:
        processed_signal: Cleaned and gap-filled FHR signal.
        original_signal: Original input signal (unchanged).
        nan_mask: Boolean mask where True indicates original invalid values.
        filled_mask: Boolean mask where True indicates gaps that were filled.
        unfilled_mask: Boolean mask where True indicates gaps that remain (too large).
        stats: Dictionary containing preprocessing statistics.
    """
    
    processed_signal: np.ndarray
    original_signal: np.ndarray
    nan_mask: np.ndarray
    filled_mask: np.ndarray
    unfilled_mask: np.ndarray
    stats: dict = field(default_factory=dict)


class CTGPreprocessor:
    """
    Preprocessor for CTG (Cardiotocography) signals.
    
    Implements the preprocessing pipeline as specified in the Israeli Position Paper
    and SentinelFetal Gen3.5 Technical Specification.
    
    Processing Steps:
        1. Mark out-of-range values as NaN (FHR outside 50-240 BPM)
        2. Detect and remove spikes (>30 BPM change between samples)
        3. Detect gaps in the signal
        4. Fill small gaps (≤10 seconds) using linear interpolation
        5. Leave large gaps (>10 seconds) as NaN
        6. Optional: Apply median filter smoothing
        
    Attributes:
        config: PreprocessingConfig instance with parameters.
        
    Example:
        >>> config = PreprocessingConfig(sampling_rate=4.0, max_gap_seconds=10.0)
        >>> preprocessor = CTGPreprocessor(config)
        >>> result = preprocessor.process(fhr_array)
        >>> clean_fhr = result.processed_signal
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration. Uses defaults if None.
        """
        self.config = config or PreprocessingConfig()
        logger.debug(f"Initialized CTGPreprocessor with config: {self.config}")
        
    def process(
        self, 
        fhr: np.ndarray, 
        apply_smoothing: bool = False
    ) -> PreprocessingResult:
        """
        Process a FHR signal through the full preprocessing pipeline.
        
        Args:
            fhr: Raw FHR signal array (1D numpy array).
            apply_smoothing: Whether to apply median filter smoothing.
            
        Returns:
            PreprocessingResult with processed signal and metadata.
            
        Raises:
            InvalidSignalError: If input signal is empty or invalid.
        """
        # Validate input
        if fhr is None or len(fhr) == 0:
            raise InvalidSignalError("Input FHR signal is empty or None")
        
        original = fhr.copy()
        processed = fhr.copy().astype(float)
        
        # Step 1: Mark out-of-range values as NaN
        out_of_range_mask = self._mark_out_of_range(processed)
        
        # Step 2: Detect all NaN positions (original NaN + out-of-range + spikes)
        nan_mask = np.isnan(processed) | out_of_range_mask
        processed[nan_mask] = np.nan
        
        # Step 3: Detect and remove spikes
        spike_mask = self._detect_spikes(processed)
        nan_mask = nan_mask | spike_mask
        processed[spike_mask] = np.nan
        
        # Step 4: Fill gaps according to 10-second rule
        processed, filled_mask, unfilled_mask = self._fill_gaps(processed)
        
        # Step 5: Optional smoothing
        if apply_smoothing:
            processed = self._apply_median_filter(processed)
        
        # Calculate statistics
        stats = self._calculate_stats(original, processed, nan_mask, filled_mask)
        
        logger.debug(
            f"Preprocessing complete: {stats['filled_percent']:.1f}% filled, "
            f"{stats['remaining_nan_percent']:.1f}% remaining NaN"
        )
        
        return PreprocessingResult(
            processed_signal=processed,
            original_signal=original,
            nan_mask=nan_mask,
            filled_mask=filled_mask,
            unfilled_mask=unfilled_mask,
            stats=stats
        )
    
    def _mark_out_of_range(self, signal: np.ndarray) -> np.ndarray:
        """
        Mark values outside valid FHR range (50-240 BPM) as invalid.
        
        Per the Israeli Position Paper, physiologically valid FHR is 50-240 BPM.
        Values outside this range, including 0 (common placeholder), are invalid.
        
        Args:
            signal: FHR signal array.
            
        Returns:
            Boolean mask where True indicates out-of-range values.
        """
        out_of_range = (signal < self.config.fhr_min) | (signal > self.config.fhr_max)
        # Treat 0 as invalid (common placeholder for missing data)
        out_of_range = out_of_range | (signal == 0)
        
        n_invalid = np.sum(out_of_range)
        if n_invalid > 0:
            logger.debug(f"Marked {n_invalid} out-of-range values as invalid")
        
        return out_of_range
    
    def _detect_spikes(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect spikes (non-physiological sudden changes) in the signal.
        
        A change of more than 30 BPM between consecutive samples is not physiological
        and indicates noise/artifact.
        
        Args:
            signal: FHR signal array.
            
        Returns:
            Boolean mask where True indicates spike positions.
        """
        spike_mask = np.zeros(len(signal), dtype=bool)
        
        # Calculate differences between consecutive valid samples
        diff = np.abs(np.diff(signal))
        
        # Find spikes (changes > threshold)
        spike_indices = np.where(diff > self.config.spike_threshold)[0] + 1
        spike_mask[spike_indices] = True
        
        n_spikes = np.sum(spike_mask)
        if n_spikes > 0:
            logger.debug(f"Detected {n_spikes} spike artifacts")
        
        return spike_mask
    
    def _fill_gaps(
        self, 
        signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fill gaps in the signal using the 10-second rule.
        
        Per the Israeli Position Paper:
            - Gaps ≤ 10 seconds (40 samples @ 4Hz): Fill with linear interpolation
            - Gaps > 10 seconds: Leave as NaN (signal loss)
        
        Args:
            signal: Signal with NaN values marking gaps.
            
        Returns:
            Tuple of (filled_signal, filled_mask, unfilled_mask).
        """
        result = signal.copy()
        filled_mask = np.zeros(len(signal), dtype=bool)
        unfilled_mask = np.zeros(len(signal), dtype=bool)
        
        # Find gap regions
        nan_positions = np.isnan(result)
        
        if not np.any(nan_positions):
            return result, filled_mask, unfilled_mask
        
        # Find contiguous NaN regions
        gaps = self._find_gaps(nan_positions)
        
        max_gap = self.config.max_gap_samples
        
        for start, end in gaps:
            gap_length = end - start
            gap_duration_sec = gap_length / self.config.sampling_rate
            
            if gap_length <= max_gap:
                # Fill this gap with linear interpolation
                result, was_filled = self._interpolate_gap(result, start, end)
                if was_filled:
                    filled_mask[start:end] = True
                    logger.debug(f"Filled gap at {start}: {gap_duration_sec:.1f}s")
                else:
                    unfilled_mask[start:end] = True
            else:
                # Gap too large - mark as unfilled
                unfilled_mask[start:end] = True
                logger.debug(f"Left unfilled gap at {start}: {gap_duration_sec:.1f}s (>{self.config.max_gap_seconds}s)")
        
        return result, filled_mask, unfilled_mask
    
    def _find_gaps(self, nan_mask: np.ndarray) -> list[Tuple[int, int]]:
        """
        Find contiguous regions of NaN values in the signal.
        
        Args:
            nan_mask: Boolean array where True indicates NaN.
            
        Returns:
            List of (start, end) index tuples for each gap.
        """
        gaps: list[Tuple[int, int]] = []
        in_gap = False
        gap_start = 0
        
        for i, is_nan in enumerate(nan_mask):
            if is_nan and not in_gap:
                # Starting a new gap
                in_gap = True
                gap_start = i
            elif not is_nan and in_gap:
                # Ending a gap
                gaps.append((gap_start, i))
                in_gap = False
        
        # Handle gap at end of signal
        if in_gap:
            gaps.append((gap_start, len(nan_mask)))
            
        return gaps
    
    def _interpolate_gap(
        self, 
        signal: np.ndarray, 
        start: int, 
        end: int
    ) -> Tuple[np.ndarray, bool]:
        """
        Interpolate a single gap using linear interpolation.
        
        Uses anchor points before and after the gap. If only one anchor
        is available (gap at start/end), uses forward/backward fill.
        
        Args:
            signal: Signal array.
            start: Start index of gap (inclusive).
            end: End index of gap (exclusive).
            
        Returns:
            Tuple of (signal with gap filled, success boolean).
        """
        # Find valid values before and after gap
        before_idx = start - 1 if start > 0 else None
        after_idx = end if end < len(signal) else None
        
        # Need at least one anchor point
        if before_idx is None and after_idx is None:
            return signal, False
            
        # Check if anchor points have valid values
        before_val: Optional[float] = None
        after_val: Optional[float] = None
        
        if before_idx is not None and not np.isnan(signal[before_idx]):
            before_val = signal[before_idx]
        if after_idx is not None and after_idx < len(signal) and not np.isnan(signal[after_idx]):
            after_val = signal[after_idx]
        
        if before_val is None and after_val is None:
            return signal, False
        
        # Interpolate
        gap_indices = np.arange(start, end)
        
        if before_val is not None and after_val is not None:
            # Linear interpolation between two points
            slope = (after_val - before_val) / (after_idx - before_idx)
            for idx in gap_indices:
                signal[idx] = before_val + slope * (idx - before_idx)
        elif before_val is not None:
            # Forward fill (gap at end or no valid after value)
            signal[start:end] = before_val
        else:
            # Backward fill (gap at start or no valid before value)
            signal[start:end] = after_val
            
        return signal, True
    
    def _apply_median_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply median filter for smoothing while preserving sharp edges.
        
        Median filtering is preferred over moving average because it preserves
        the sharp edges of decelerations (important for clinical interpretation).
        
        Args:
            signal: Signal to smooth.
            
        Returns:
            Smoothed signal with NaN values preserved.
        """
        window = self.config.smoothing_window
        if window < 2:
            return signal
        
        # Ensure window is odd for symmetric filtering
        if window % 2 == 0:
            window += 1
            
        # Use pandas for NaN-aware rolling median
        smoothed = pd.Series(signal).rolling(
            window=window, 
            center=True, 
            min_periods=1
        ).median().values
        
        return smoothed
    
    def _calculate_stats(
        self, 
        original: np.ndarray,
        processed: np.ndarray,
        nan_mask: np.ndarray,
        filled_mask: np.ndarray
    ) -> dict[str, float | int]:
        """
        Calculate preprocessing statistics.
        
        Args:
            original: Original signal before processing.
            processed: Signal after processing.
            nan_mask: Mask of originally invalid values.
            filled_mask: Mask of filled gap positions.
            
        Returns:
            Dictionary with statistics including sample counts and percentages.
        """
        total_samples = len(original)
        
        return {
            'total_samples': total_samples,
            'duration_seconds': total_samples / self.config.sampling_rate,
            'invalid_samples': int(np.sum(nan_mask)),
            'invalid_percent': float(np.sum(nan_mask) / total_samples * 100),
            'filled_samples': int(np.sum(filled_mask)),
            'filled_percent': float(np.sum(filled_mask) / total_samples * 100),
            'remaining_nan': int(np.sum(np.isnan(processed))),
            'remaining_nan_percent': float(np.sum(np.isnan(processed)) / total_samples * 100),
            'mean_fhr': float(np.nanmean(processed)) if not np.all(np.isnan(processed)) else 0.0,
            'std_fhr': float(np.nanstd(processed)) if not np.all(np.isnan(processed)) else 0.0,
        }


def preprocess_fhr(
    fhr: np.ndarray,
    config: Optional[PreprocessingConfig] = None
) -> np.ndarray:
    """
    Convenience function to preprocess FHR signal.
    
    This is a simple wrapper around CTGPreprocessor for quick preprocessing
    without needing to instantiate the class.
    
    Args:
        fhr: Raw FHR signal array.
        config: Optional preprocessing configuration.
        
    Returns:
        Preprocessed FHR signal array.
        
    Example:
        >>> from src.data.preprocess import preprocess_fhr
        >>> clean_fhr = preprocess_fhr(raw_fhr)
    """
    preprocessor = CTGPreprocessor(config)
    result = preprocessor.process(fhr)
    return result.processed_signal
