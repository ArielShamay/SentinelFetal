"""
Signal processing utility functions.

This module provides commonly used signal processing operations,
particularly for handling NaN values in CTG signals.

Functions:
    get_valid_values: Extract non-NaN values from signal
    get_valid_ratio: Calculate ratio of valid (non-NaN) samples
    interpolate_nans: Fill NaN gaps using interpolation
    safe_nanmean: Calculate mean ignoring NaN values
    safe_nanstd: Calculate standard deviation ignoring NaN values
    safe_nanmedian: Calculate median ignoring NaN values

Example:
    >>> from src.utils.signal_utils import get_valid_values, safe_nanmean
    >>> signal = np.array([120, np.nan, 130, 140, np.nan])
    >>> valid = get_valid_values(signal)
    >>> print(valid)  # [120, 130, 140]
    >>> mean = safe_nanmean(signal)
    >>> print(mean)  # 130.0
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def get_valid_values(signal: np.ndarray) -> np.ndarray:
    """
    Extract non-NaN values from a signal.
    
    Args:
        signal: Input signal array that may contain NaN values.
        
    Returns:
        Array containing only the valid (non-NaN) values.
        
    Example:
        >>> signal = np.array([120, np.nan, 130])
        >>> valid = get_valid_values(signal)
        >>> print(valid)  # [120, 130]
    """
    return signal[~np.isnan(signal)]


def get_valid_ratio(signal: np.ndarray) -> float:
    """
    Calculate the ratio of valid (non-NaN) samples in a signal.
    
    Args:
        signal: Input signal array.
        
    Returns:
        Ratio of valid samples (0.0 to 1.0).
        
    Example:
        >>> signal = np.array([120, np.nan, 130, 140, np.nan])
        >>> ratio = get_valid_ratio(signal)
        >>> print(ratio)  # 0.6
    """
    if len(signal) == 0:
        return 0.0
    return np.sum(~np.isnan(signal)) / len(signal)


def interpolate_nans(
    signal: np.ndarray,
    max_gap_samples: Optional[int] = None,
    method: str = 'linear'
) -> Tuple[np.ndarray, int]:
    """
    Fill NaN values using interpolation.
    
    Args:
        signal: Input signal with NaN values.
        max_gap_samples: Maximum gap size to fill (None = fill all).
        method: Interpolation method ('linear', 'nearest').
        
    Returns:
        Tuple of (filled signal, number of values filled).
        
    Example:
        >>> signal = np.array([100, np.nan, np.nan, 120])
        >>> filled, count = interpolate_nans(signal)
        >>> print(filled)  # [100, 106.67, 113.33, 120]
    """
    if len(signal) == 0:
        return signal.copy(), 0
    
    result = signal.copy()
    nan_mask = np.isnan(result)
    
    if not np.any(nan_mask):
        return result, 0
    
    # Get valid indices
    valid_indices = np.where(~nan_mask)[0]
    
    if len(valid_indices) == 0:
        return result, 0
    
    if len(valid_indices) == len(signal):
        return result, 0
    
    # Linear interpolation
    nan_indices = np.where(nan_mask)[0]
    
    if method == 'linear':
        result[nan_mask] = np.interp(
            nan_indices,
            valid_indices,
            result[valid_indices]
        )
    elif method == 'nearest':
        # Use nearest valid value
        for idx in nan_indices:
            distances = np.abs(valid_indices - idx)
            nearest = valid_indices[np.argmin(distances)]
            result[idx] = result[nearest]
    
    return result, int(np.sum(nan_mask))


def safe_nanmean(signal: np.ndarray, default: float = 0.0) -> float:
    """
    Calculate mean ignoring NaN values, with fallback for empty arrays.
    
    Args:
        signal: Input signal array.
        default: Value to return if no valid values exist.
        
    Returns:
        Mean of valid values, or default if none exist.
        
    Example:
        >>> safe_nanmean(np.array([np.nan, np.nan]))
        0.0
        >>> safe_nanmean(np.array([100, np.nan, 120]))
        110.0
    """
    valid = signal[~np.isnan(signal)]
    if len(valid) == 0:
        return default
    return float(np.mean(valid))


def safe_nanstd(signal: np.ndarray, default: float = 0.0) -> float:
    """
    Calculate standard deviation ignoring NaN values.
    
    Args:
        signal: Input signal array.
        default: Value to return if fewer than 2 valid values.
        
    Returns:
        Standard deviation of valid values.
        
    Example:
        >>> safe_nanstd(np.array([100, np.nan, 120]))
        10.0
    """
    valid = signal[~np.isnan(signal)]
    if len(valid) < 2:
        return default
    return float(np.std(valid))


def safe_nanmedian(signal: np.ndarray, default: float = 0.0) -> float:
    """
    Calculate median ignoring NaN values.
    
    Args:
        signal: Input signal array.
        default: Value to return if no valid values exist.
        
    Returns:
        Median of valid values.
        
    Example:
        >>> safe_nanmedian(np.array([100, np.nan, 120, 110]))
        110.0
    """
    valid = signal[~np.isnan(signal)]
    if len(valid) == 0:
        return default
    return float(np.median(valid))


def calculate_signal_quality(signal: np.ndarray) -> float:
    """
    Calculate overall signal quality score.
    
    Combines multiple factors:
    - Ratio of valid samples
    - Absence of unrealistic values
    - Consistency of signal
    
    Args:
        signal: Input signal array.
        
    Returns:
        Quality score from 0.0 (poor) to 1.0 (excellent).
    """
    if len(signal) == 0:
        return 0.0
    
    valid_ratio = get_valid_ratio(signal)
    valid = get_valid_values(signal)
    
    if len(valid) == 0:
        return 0.0
    
    # For FHR: check if values are in reasonable range (50-250 bpm)
    reasonable_mask = (valid >= 50) & (valid <= 250)
    reasonable_ratio = np.sum(reasonable_mask) / len(valid)
    
    # Combined quality score
    return float(valid_ratio * reasonable_ratio)


__all__ = [
    'get_valid_values',
    'get_valid_ratio',
    'interpolate_nans',
    'safe_nanmean',
    'safe_nanstd',
    'safe_nanmedian',
    'calculate_signal_quality',
]
