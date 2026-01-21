# -*- coding: utf-8 -*-
"""
Fetal Signal Quality Index (FSQI) Module.

Implements signal quality assessment for CTG/FHR signals to filter noise
before classification. Research shows FSQI eliminates 94.92% of incorrectly
detected deceleration events.

Quality Assessment Criteria:
    1. Valid sample ratio (non-NaN, non-zero)
    2. Physiological range (50-250 bpm for FHR)
    3. Spectral noise analysis (high-frequency content)
    4. Signal stability (sudden jumps/artifacts)
    5. Baseline detectability

Quality Thresholds:
    - score >= 0.7: HIGH quality - proceed with classification
    - score 0.4-0.7: MEDIUM quality - classification with reduced confidence
    - score < 0.4: LOW quality - skip classification ("Signal Lost/Noise")

References:
    - FSQI Paper: github.com/Majy-Yuji/FSQI
    - SentinelFetal Improvement Plan (Das et al. 2023)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


class SignalQuality(Enum):
    """Signal quality classification."""
    HIGH = "high"           # Score >= 0.7: Full classification
    MEDIUM = "medium"       # Score 0.4-0.7: Classification with warning
    LOW = "low"             # Score < 0.4: Skip classification
    INSUFFICIENT = "insufficient"  # Not enough data


@dataclass
class SignalQualityResult:
    """
    Result of signal quality assessment.
    
    Attributes:
        score: Overall quality score (0.0 to 1.0).
        quality: Quality classification (HIGH/MEDIUM/LOW/INSUFFICIENT).
        valid_ratio: Ratio of valid (non-NaN, non-zero) samples.
        physiological_ratio: Ratio of samples in physiological range.
        noise_score: Spectral noise assessment (0=noisy, 1=clean).
        stability_score: Signal stability (0=unstable, 1=stable).
        message: Human-readable quality description.
        should_classify: Whether to proceed with classification.
    """
    score: float
    quality: SignalQuality
    valid_ratio: float
    physiological_ratio: float
    noise_score: float
    stability_score: float
    message: str
    should_classify: bool
    
    def __repr__(self) -> str:
        return (
            f"SignalQualityResult(score={self.score:.2f}, "
            f"quality={self.quality.value}, classify={self.should_classify})"
        )


# Quality thresholds
HIGH_QUALITY_THRESHOLD = 0.7
MEDIUM_QUALITY_THRESHOLD = 0.4

# Physiological ranges
FHR_MIN = 50    # bpm - below this is artifact
FHR_MAX = 250   # bpm - above this is artifact
FHR_NORMAL_MIN = 110  # bpm - normal range lower
FHR_NORMAL_MAX = 160  # bpm - normal range upper


def calculate_fsqi(
    fhr: np.ndarray,
    sampling_rate: float = 4.0,
    min_samples: int = 40  # Minimum 10 seconds of data
) -> SignalQualityResult:
    """
    Calculate Fetal Signal Quality Index (FSQI) for FHR signal.
    
    This is the main entry point for signal quality assessment.
    
    Args:
        fhr: FHR signal array in bpm.
        sampling_rate: Sampling rate in Hz (default: 4.0).
        min_samples: Minimum number of samples required.
        
    Returns:
        SignalQualityResult with quality assessment.
        
    Example:
        >>> quality = calculate_fsqi(fhr_signal)
        >>> if quality.should_classify:
        ...     # Proceed with deceleration detection
        ...     pass
        >>> else:
        ...     print(quality.message)  # "Signal Lost/Noise"
    """
    # Check minimum data requirement
    if fhr is None or len(fhr) < min_samples:
        return SignalQualityResult(
            score=0.0,
            quality=SignalQuality.INSUFFICIENT,
            valid_ratio=0.0,
            physiological_ratio=0.0,
            noise_score=0.0,
            stability_score=0.0,
            message="Insufficient data for quality assessment",
            should_classify=False
        )
    
    # Component 1: Valid sample ratio
    valid_ratio = _calculate_valid_ratio(fhr)
    
    # Component 2: Physiological range compliance
    physiological_ratio = _calculate_physiological_ratio(fhr)
    
    # Component 3: Spectral noise assessment
    noise_score = _calculate_noise_score(fhr, sampling_rate)
    
    # Component 4: Signal stability
    stability_score = _calculate_stability_score(fhr, sampling_rate)
    
    # Weighted combination (weights based on clinical importance)
    # Valid ratio is most important - if data is missing, everything fails
    weights = {
        'valid': 0.35,
        'physiological': 0.25,
        'noise': 0.20,
        'stability': 0.20
    }
    
    score = (
        weights['valid'] * valid_ratio +
        weights['physiological'] * physiological_ratio +
        weights['noise'] * noise_score +
        weights['stability'] * stability_score
    )
    
    # VETO RULES: Safety mechanism to reject specific failure modes
    # 1. If noise is severe (score < 0.3), signal is unusable regardless of other metrics
    if noise_score < 0.3:
        logger.warning(f"FSQI Veto: Severe noise detected (score={noise_score:.2f})")
        score = min(score, 0.3)  # Cap at 0.3 (LOW quality)
        
    # 2. If stability is very low (artifacts), reduce score
    if stability_score < 0.3:
        score = min(score, 0.5)  # Cap at 0.5 (MEDIUM quality)
    
    # Determine quality classification
    if score >= HIGH_QUALITY_THRESHOLD:
        quality = SignalQuality.HIGH
        message = "Good signal quality"
        should_classify = True
    elif score >= MEDIUM_QUALITY_THRESHOLD:
        quality = SignalQuality.MEDIUM
        message = "Moderate signal quality - classification may be less reliable"
        should_classify = True
    else:
        quality = SignalQuality.LOW
        message = "Signal Lost/Noise - classification skipped"
        should_classify = False
    
    logger.debug(
        f"FSQI: score={score:.2f}, valid={valid_ratio:.2f}, "
        f"physio={physiological_ratio:.2f}, noise={noise_score:.2f}, "
        f"stability={stability_score:.2f}"
    )
    
    return SignalQualityResult(
        score=round(score, 3),
        quality=quality,
        valid_ratio=round(valid_ratio, 3),
        physiological_ratio=round(physiological_ratio, 3),
        noise_score=round(noise_score, 3),
        stability_score=round(stability_score, 3),
        message=message,
        should_classify=should_classify
    )


def _calculate_valid_ratio(fhr: np.ndarray) -> float:
    """
    Calculate ratio of valid (non-NaN, non-zero) samples.
    
    Zero values in CTG typically indicate signal loss.
    """
    valid_mask = ~np.isnan(fhr) & (fhr != 0)
    return float(np.sum(valid_mask) / len(fhr)) if len(fhr) > 0 else 0.0


def _calculate_physiological_ratio(fhr: np.ndarray) -> float:
    """
    Calculate ratio of samples within physiological FHR range.
    
    FHR should be between 50-250 bpm. Values outside this range
    are almost certainly artifacts.
    """
    valid_fhr = fhr[~np.isnan(fhr) & (fhr != 0)]
    
    if len(valid_fhr) == 0:
        return 0.0
    
    in_range = (valid_fhr >= FHR_MIN) & (valid_fhr <= FHR_MAX)
    return float(np.sum(in_range) / len(valid_fhr))


def _calculate_noise_score(fhr: np.ndarray, sampling_rate: float) -> float:
    """
    Assess signal noise using spectral analysis.
    
    High-frequency content (>0.5 Hz) in FHR is typically noise/artifact.
    Normal FHR variability is in the 0.03-0.15 Hz range (3-6 cycles/min).
    
    Returns:
        Score from 0 (very noisy) to 1 (clean signal).
    """
    # Get valid samples
    valid_fhr = fhr[~np.isnan(fhr) & (fhr != 0)]
    
    if len(valid_fhr) < 32:  # Need enough samples for FFT
        return 0.5  # Unknown
    
    try:
        # Compute power spectral density
        freqs, psd = scipy_signal.welch(
            valid_fhr - np.mean(valid_fhr),  # Remove DC
            fs=sampling_rate,
            nperseg=min(256, len(valid_fhr) // 2)
        )
        
        # Define frequency bands
        # Signal band: 0.03 - 0.5 Hz (Physiological variability)
        low_freq_mask = (freqs >= 0.03) & (freqs <= 0.5) 
        # Noise band: > 0.5 Hz (High frequency artifact)
        high_freq_mask = (freqs > 0.5)
        
        low_power = np.sum(psd[low_freq_mask]) if np.any(low_freq_mask) else 0
        high_power = np.sum(psd[high_freq_mask]) if np.any(high_freq_mask) else 0
        
        if high_power == 0:
            return 1.0 # No high frequency noise
            
        total_power = low_power + high_power
        if total_power == 0:
            return 0.5
        
        # Signal-to-noise ratio
        # STRICTER: If significant high frequency power exists, score drops rapidly
        ratio = low_power / (high_power + 1e-10)
        
        # If noise power is > 20% of signal power, it's getting bad
        # ratio < 5 -> starts penalizing
        
        if ratio > 10.0:
            return 1.0
        elif ratio < 1.0: # More noise than signal
            return 0.0
        else:
            return ratio / 10.0
            
    except Exception as e:
        logger.warning(f"Noise calculation failed: {e}")
        return 0.5


def _calculate_stability_score(fhr: np.ndarray, sampling_rate: float) -> float:
    """
    Assess signal stability (sudden jumps indicate artifacts).
    
    Normal FHR doesn't change by >25 bpm between consecutive samples
    at 4 Hz sampling. Larger jumps indicate artifacts.
    
    Returns:
        Score from 0 (many artifacts) to 1 (stable signal).
    """
    valid_fhr = fhr[~np.isnan(fhr) & (fhr != 0)]
    
    if len(valid_fhr) < 2:
        return 0.0
    
    # Calculate sample-to-sample differences
    diffs = np.abs(np.diff(valid_fhr))
    
    # Threshold: 25 bpm per sample is ~100 bpm/sec at 4Hz (definitely artifact)
    artifact_threshold = 25.0  # bpm per sample
    
    # Count artifacts
    n_artifacts = np.sum(diffs > artifact_threshold)
    artifact_ratio = n_artifacts / len(diffs)
    
    # Convert to stability score
    stability_score = 1.0 - min(1.0, artifact_ratio * 5)  # 20% artifacts = score 0
    
    return float(max(0.0, stability_score))


def apply_quality_gate(
    fhr: np.ndarray,
    sampling_rate: float = 4.0,
    threshold: float = 0.4
) -> Tuple[bool, SignalQualityResult]:
    """
    Quality gate for preprocessing pipeline.
    
    Use this before any classification to filter bad signals.
    
    Args:
        fhr: FHR signal array.
        sampling_rate: Sampling rate in Hz.
        threshold: Minimum quality score to pass (default: 0.4).
        
    Returns:
        Tuple of (passes_gate, quality_result).
        
    Example:
        >>> passes, quality = apply_quality_gate(fhr_signal)
        >>> if passes:
        ...     decels = detect_decelerations(fhr_signal, ...)
        ... else:
        ...     print(f"Skipping: {quality.message}")
    """
    quality = calculate_fsqi(fhr, sampling_rate)
    passes = quality.score >= threshold
    
    return passes, quality


def denoise_coiflet4(
    signal: np.ndarray,
    level: int = 5,
    threshold_mode: str = 'soft'
) -> np.ndarray:
    """
    Denoise signal using Coiflet 4 wavelet (validated for fetal signals).
    
    Research shows Coiflet 4 + soft thresholding achieves:
    - SNR improvement: +25.2 dB (simulated), +7.3 dB (experimental)
    - Preserved FHR accuracy: 138.7 vs 140.2 bpm (p > 0.05)
    
    Args:
        signal: Input signal array.
        level: Decomposition level (5-6 recommended).
        threshold_mode: 'soft' or 'hard' thresholding.
        
    Returns:
        Denoised signal array.
    """
    try:
        import pywt
    except ImportError:
        logger.warning("pywavelets not available. Returning original signal.")
        return signal
    
    # Handle NaN values
    valid_mask = ~np.isnan(signal)
    if not np.any(valid_mask):
        return signal
    
    # Interpolate NaN values for wavelet transform
    signal_interp = signal.copy()
    if np.any(~valid_mask):
        indices = np.arange(len(signal))
        signal_interp = np.interp(indices, indices[valid_mask], signal[valid_mask])
    
    # Wavelet decomposition
    wavelet = 'coif4'
    coeffs = pywt.wavedec(signal_interp, wavelet, level=level)
    
    # Universal threshold (Donoho-Johnstone)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_interp)))
    
    # Apply thresholding to detail coefficients
    denoised_coeffs = [coeffs[0]]  # Keep approximation
    for i in range(1, len(coeffs)):
        if threshold_mode == 'soft':
            denoised = pywt.threshold(coeffs[i], threshold, mode='soft')
        else:
            denoised = pywt.threshold(coeffs[i], threshold, mode='hard')
        denoised_coeffs.append(denoised)
    
    # Reconstruct
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    
    # Trim to original length (wavelet transform may pad)
    denoised_signal = denoised_signal[:len(signal)]
    
    # Restore NaN positions
    denoised_signal[~valid_mask] = np.nan
    
    return denoised_signal


__all__ = [
    'calculate_fsqi',
    'apply_quality_gate',
    'denoise_coiflet4',
    'SignalQuality',
    'SignalQualityResult',
    'HIGH_QUALITY_THRESHOLD',
    'MEDIUM_QUALITY_THRESHOLD',
]
