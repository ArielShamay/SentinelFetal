"""
Spectral Analyzer for MHR Guard Module.

Analyzes the spectral content of FHR variability to detect
Respiratory Sinus Arrhythmia (RSA) patterns that distinguish
maternal from fetal heart rate signals.

Key Insight:
    - Adult RSA: 0.15-0.35 Hz (12-20 breaths/minute)
    - Fetal RSA: 0.4-1.0 Hz (faster breathing movements)

If the signal shows dominant power in the adult RSA band with
spectral centroid < 0.35 Hz, it may be maternal heart rate.

References:
    - SentinelFetal V2.0 PRD, Section: MHR Guard Module
    - Pattern reused from src/rules/sinusoidal.py FFT implementation
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

from src.safety.models import SpectralResult

logger = logging.getLogger(__name__)


class SpectralAnalyzerError(Exception):
    """Raised when spectral analysis fails."""
    pass


class SpectralAnalyzer:
    """
    Analyzes spectral content of FHR variability to detect RSA patterns.

    RSA (Respiratory Sinus Arrhythmia) creates characteristic frequency
    signatures in heart rate variability. Adults breathe slower than
    fetal breathing movements, creating distinct frequency bands.

    Example:
        >>> analyzer = SpectralAnalyzer()
        >>> result = analyzer.analyze_rsa(fhr_segment, sampling_rate=4.0)
        >>> if result.adult_power_ratio > 0.4:
        ...     print("Warning: Signal may be maternal heart rate")
    """

    # Frequency bands (Hz)
    ADULT_RSA_BAND = (0.15, 0.35)   # Adult breathing: 12-20 breaths/min
    FETAL_RSA_BAND = (0.4, 1.0)    # Fetal breathing movements: 24-60/min

    def __init__(self):
        """Initialize spectral analyzer."""
        pass

    def analyze_rsa(
        self,
        fhr: np.ndarray,
        sampling_rate: float = 4.0
    ) -> SpectralResult:
        """
        Compute Power Spectral Density and analyze RSA bands.

        Algorithm:
            1. Validate and clean input
            2. Detrend signal (remove baseline drift)
            3. Apply Hann window (reduce spectral leakage)
            4. Compute FFT → Power Spectral Density
            5. Calculate power in adult and fetal RSA bands
            6. Compute spectral centroid (center of mass)

        Args:
            fhr: FHR signal array in bpm.
            sampling_rate: Sampling frequency in Hz (default: 4.0).

        Returns:
            SpectralResult with band powers and centroid.

        Raises:
            SpectralAnalyzerError: If analysis fails.
        """
        # Validate input
        if fhr is None or len(fhr) < 20:
            raise SpectralAnalyzerError(
                f"FHR signal too short: {len(fhr) if fhr is not None else 0} samples"
            )

        # Handle NaN values - replace with mean
        valid_mask = ~np.isnan(fhr)
        if np.sum(valid_mask) < len(fhr) * 0.5:
            raise SpectralAnalyzerError("Too many NaN values (>50%)")

        fhr_clean = fhr.copy()
        if np.any(~valid_mask):
            mean_val = np.mean(fhr[valid_mask])
            fhr_clean[~valid_mask] = mean_val

        try:
            # Step 1: Detrend (remove baseline drift)
            detrended = signal.detrend(fhr_clean)

            # Step 2: Apply Hann window (reduce spectral leakage)
            windowed = detrended * np.hanning(len(detrended))

            # Step 3-4: FFT and PSD
            n = len(windowed)
            freqs = fftfreq(n, 1 / sampling_rate)
            fft_vals = np.abs(fft(windowed))
            psd = fft_vals ** 2

            # Only use positive frequencies
            positive_mask = freqs > 0
            freqs_pos = freqs[positive_mask]
            psd_pos = psd[positive_mask]

            # Avoid division by zero
            total_power = float(np.sum(psd_pos)) + 1e-10

            # Step 5: Calculate band powers
            adult_mask = (
                (freqs_pos >= self.ADULT_RSA_BAND[0]) &
                (freqs_pos <= self.ADULT_RSA_BAND[1])
            )
            fetal_mask = (
                (freqs_pos >= self.FETAL_RSA_BAND[0]) &
                (freqs_pos <= self.FETAL_RSA_BAND[1])
            )

            adult_power = float(np.sum(psd_pos[adult_mask]))
            fetal_power = float(np.sum(psd_pos[fetal_mask]))

            # Step 6: Spectral centroid (center of mass)
            spectral_centroid = float(np.sum(freqs_pos * psd_pos) / total_power)

            # Find dominant frequency
            dominant_idx = np.argmax(psd_pos)
            dominant_frequency = float(freqs_pos[dominant_idx])

            return SpectralResult(
                adult_band_power=adult_power,
                fetal_band_power=fetal_power,
                total_power=total_power,
                spectral_centroid=spectral_centroid,
                dominant_frequency=dominant_frequency
            )

        except Exception as e:
            logger.error(f"Spectral analysis failed: {e}")
            raise SpectralAnalyzerError(f"Analysis failed: {e}")

    def analyze_rsa_welch(
        self,
        fhr: np.ndarray,
        sampling_rate: float = 4.0,
        nperseg: int = 256
    ) -> SpectralResult:
        """
        Alternative analysis using Welch's method for smoother PSD.

        Welch's method provides better noise reduction through
        segment averaging, useful for longer signals.

        Args:
            fhr: FHR signal array in bpm.
            sampling_rate: Sampling frequency in Hz.
            nperseg: Samples per segment for Welch method.

        Returns:
            SpectralResult with band powers and centroid.
        """
        if fhr is None or len(fhr) < nperseg:
            raise SpectralAnalyzerError(
                f"Signal too short for Welch method: need {nperseg} samples"
            )

        # Handle NaN values
        valid_mask = ~np.isnan(fhr)
        if np.sum(valid_mask) < len(fhr) * 0.5:
            raise SpectralAnalyzerError("Too many NaN values (>50%)")

        fhr_clean = fhr.copy()
        if np.any(~valid_mask):
            mean_val = np.mean(fhr[valid_mask])
            fhr_clean[~valid_mask] = mean_val

        try:
            # Use Welch's method for PSD
            freqs, psd = signal.welch(
                fhr_clean,
                fs=sampling_rate,
                nperseg=min(nperseg, len(fhr_clean)),
                noverlap=nperseg // 2
            )

            total_power = float(np.sum(psd)) + 1e-10

            # Calculate band powers
            adult_mask = (
                (freqs >= self.ADULT_RSA_BAND[0]) &
                (freqs <= self.ADULT_RSA_BAND[1])
            )
            fetal_mask = (
                (freqs >= self.FETAL_RSA_BAND[0]) &
                (freqs <= self.FETAL_RSA_BAND[1])
            )

            adult_power = float(np.sum(psd[adult_mask]))
            fetal_power = float(np.sum(psd[fetal_mask]))

            # Spectral centroid
            spectral_centroid = float(np.sum(freqs * psd) / total_power)

            # Dominant frequency
            dominant_idx = np.argmax(psd)
            dominant_frequency = float(freqs[dominant_idx])

            return SpectralResult(
                adult_band_power=adult_power,
                fetal_band_power=fetal_power,
                total_power=total_power,
                spectral_centroid=spectral_centroid,
                dominant_frequency=dominant_frequency
            )

        except Exception as e:
            logger.error(f"Welch analysis failed: {e}")
            raise SpectralAnalyzerError(f"Welch analysis failed: {e}")


def generate_mhr_test_signal(
    duration_seconds: float = 60.0,
    sampling_rate: float = 4.0,
    baseline: float = 95.0,
    rsa_amplitude: float = 5.0,
    breathing_rate_per_min: float = 16.0
) -> np.ndarray:
    """
    Generate a synthetic maternal heart rate signal for testing.

    Creates a signal with adult RSA characteristics that should
    trigger MHR detection.

    Args:
        duration_seconds: Signal duration.
        sampling_rate: Sampling frequency in Hz.
        baseline: Baseline heart rate in bpm.
        rsa_amplitude: Amplitude of RSA oscillation.
        breathing_rate_per_min: Breathing rate (12-20 for adults).

    Returns:
        Synthetic MHR signal array.
    """
    n_samples = int(duration_seconds * sampling_rate)
    t = np.arange(n_samples) / sampling_rate

    # RSA frequency from breathing rate
    rsa_freq_hz = breathing_rate_per_min / 60

    # Generate sine wave with adult RSA characteristics
    signal_out = baseline + rsa_amplitude * np.sin(2 * np.pi * rsa_freq_hz * t)

    # Add small noise
    noise = np.random.normal(0, 0.5, n_samples)
    signal_out += noise

    return signal_out
