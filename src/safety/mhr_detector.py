"""
MHR Detector - Maternal Heart Rate Contamination Detection.

Detects when the fetal monitor may be picking up maternal heart rate
instead of fetal heart rate. This is a critical safety feature as
MHR contamination can mask fetal distress.

Detection Methods:
    1. Cross-correlation with MHR reference (if SpO2 pulse available)
    2. Spectral RSA analysis (adult vs fetal breathing patterns)
    3. Baseline jump detection (sudden signal source switch)

CRITICAL SAFETY FEATURE:
    When MHR is suspected with high confidence, the segment is BLOCKED
    from ML classification to prevent false reassurance.

FETAL SLEEP HANDLING:
    Low variability with accelerations present = likely fetal sleep, NOT MHR.
    The detector reduces confidence when accelerations are present.

References:
    - SentinelFetal V2.0 PRD, Section: MHR Guard Module
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from src.safety.models import (
    MHRAction,
    MHRCheckResult,
    MHRDetectorConfig,
    DetectionResult,
    BaselineJumpResult,
)
from src.safety.spectral_analyzer import SpectralAnalyzer, SpectralAnalyzerError

logger = logging.getLogger(__name__)


class MHRDetector:
    """
    Detects suspected Maternal Heart Rate signal contamination.

    Uses multiple detection strategies in priority order:
    1. Cross-correlation with MHR reference (most reliable)
    2. Spectral RSA analysis (good standalone detection)
    3. Baseline jump detection (supplementary evidence)

    Results are fused using weighted voting to produce a final
    confidence score and recommended action.

    CRITICAL: Accounts for fetal sleep cycles. If accelerations are
    present alongside low variability, confidence is reduced as this
    pattern indicates sleeping fetus, not MHR contamination.

    Example:
        >>> detector = MHRDetector()
        >>> result = detector.check_segment(
        ...     fhr_segment=fhr[-240:],
        ...     has_accelerations=True
        ... )
        >>> if result.recommended_action == MHRAction.BLOCK_SEGMENT:
        ...     # Do not use this segment for classification
        ...     return {"category": None, "error": "MHR suspected"}
    """

    def __init__(self, config: Optional[MHRDetectorConfig] = None):
        """
        Initialize MHR detector.

        Args:
            config: Configuration options. If None, uses defaults.
        """
        self.config = config or MHRDetectorConfig()
        self.spectral_analyzer = SpectralAnalyzer()

    def check_segment(
        self,
        fhr_segment: np.ndarray,
        mhr_reference: Optional[np.ndarray] = None,
        has_accelerations: bool = False,
        sampling_rate: float = 4.0
    ) -> MHRCheckResult:
        """
        Main entry point. Analyzes segment for MHR contamination.

        Algorithm:
            1. If MHR reference available → Cross-correlation method
            2. Always → Spectral (RSA) analysis
            3. Always → Baseline jump detection
            4. Fuse results with weighted voting
            5. Adjust for fetal sleep if accelerations present

        Args:
            fhr_segment: FHR signal segment (recommend 60 seconds = 240 samples at 4Hz).
            mhr_reference: Optional maternal HR from SpO2 (same length as fhr_segment).
            has_accelerations: True if accelerations detected in this window.
                              CRITICAL for fetal sleep detection.
            sampling_rate: Sampling frequency in Hz.

        Returns:
            MHRCheckResult with detection outcome and recommended action.
        """
        if fhr_segment is None or len(fhr_segment) < 20:
            return MHRCheckResult.no_suspicion()

        results: List[DetectionResult] = []

        # ─────────────────────────────────────────────────────────
        # METHOD 1: Cross-correlation with MHR reference
        # ─────────────────────────────────────────────────────────
        if mhr_reference is not None and len(mhr_reference) >= len(fhr_segment):
            try:
                correlation = self._compute_cross_correlation(
                    fhr_segment, mhr_reference[:len(fhr_segment)]
                )

                if correlation > self.config.cross_correlation_threshold:
                    results.append(DetectionResult(
                        method="cross_correlation",
                        is_suspected=True,
                        confidence=correlation,
                        reason=f"FHR-MHR correlation: {correlation:.2f} (high correlation)"
                    ))
                else:
                    results.append(DetectionResult(
                        method="cross_correlation",
                        is_suspected=False,
                        confidence=1.0 - correlation,
                        reason=f"FHR-MHR correlation: {correlation:.2f} (normal)"
                    ))
            except Exception as e:
                logger.warning(f"Cross-correlation failed: {e}")

        # ─────────────────────────────────────────────────────────
        # METHOD 2: Spectral Analysis (RSA pattern)
        # ─────────────────────────────────────────────────────────
        try:
            spectral_result = self.spectral_analyzer.analyze_rsa(
                fhr_segment, sampling_rate
            )

            adult_power_ratio = spectral_result.adult_power_ratio

            # Suspect MHR if adult RSA band dominates and centroid is low
            if (adult_power_ratio > self.config.adult_power_ratio_threshold and
                spectral_result.spectral_centroid < self.config.spectral_centroid_threshold):

                confidence = min(adult_power_ratio * 1.5, 0.95)
                results.append(DetectionResult(
                    method="spectral_rsa",
                    is_suspected=True,
                    confidence=confidence,
                    reason=(
                        f"Adult RSA pattern: {adult_power_ratio:.1%} power in "
                        f"0.15-0.35Hz band, centroid {spectral_result.spectral_centroid:.2f}Hz"
                    )
                ))
            else:
                results.append(DetectionResult(
                    method="spectral_rsa",
                    is_suspected=False,
                    confidence=1.0 - adult_power_ratio,
                    reason=f"Fetal RSA pattern (adult ratio: {adult_power_ratio:.1%})"
                ))

        except SpectralAnalyzerError as e:
            logger.warning(f"Spectral analysis failed: {e}")

        # ─────────────────────────────────────────────────────────
        # METHOD 3: Baseline Jump Detection
        # ─────────────────────────────────────────────────────────
        try:
            jump_result = self._detect_baseline_jump(fhr_segment, sampling_rate)

            if jump_result.jump_detected:
                results.append(DetectionResult(
                    method="baseline_jump",
                    is_suspected=True,
                    confidence=jump_result.confidence,
                    reason=(
                        f"Baseline jump: {jump_result.jump_magnitude:.0f} bpm "
                        f"in {jump_result.jump_duration:.1f}s"
                    )
                ))
        except Exception as e:
            logger.warning(f"Baseline jump detection failed: {e}")

        # ─────────────────────────────────────────────────────────
        # FUSION: Combine results with fetal sleep adjustment
        # ─────────────────────────────────────────────────────────
        return self._fuse_results(
            results,
            has_accelerations=has_accelerations,
            segment_length=len(fhr_segment)
        )

    def _compute_cross_correlation(
        self,
        fhr: np.ndarray,
        mhr: np.ndarray
    ) -> float:
        """
        Compute normalized cross-correlation between FHR and MHR.

        High correlation (>0.8) suggests FHR is tracking MHR.

        Args:
            fhr: FHR signal array.
            mhr: MHR reference array (same length).

        Returns:
            Correlation coefficient (0-1, absolute value).
        """
        # Handle NaN values
        valid_mask = ~np.isnan(fhr) & ~np.isnan(mhr)
        if np.sum(valid_mask) < 10:
            return 0.0

        fhr_valid = fhr[valid_mask]
        mhr_valid = mhr[valid_mask]

        # Z-score normalization
        fhr_std = np.std(fhr_valid)
        mhr_std = np.std(mhr_valid)

        if fhr_std < 1e-6 or mhr_std < 1e-6:
            return 0.0

        fhr_norm = (fhr_valid - np.mean(fhr_valid)) / fhr_std
        mhr_norm = (mhr_valid - np.mean(mhr_valid)) / mhr_std

        # Pearson correlation
        correlation = np.correlate(fhr_norm, mhr_norm, mode='valid')[0] / len(fhr_norm)

        return abs(float(correlation))

    def _detect_baseline_jump(
        self,
        fhr: np.ndarray,
        sampling_rate: float
    ) -> BaselineJumpResult:
        """
        Detect sudden baseline changes suggesting sensor switch.

        Algorithm:
            1. Compute rolling mean with 5-second window
            2. Compute first derivative of rolling mean
            3. If |derivative| > 4 bpm/sec (>20 bpm in 5 sec)
               AND signal stable after → jump detected

        A sudden jump followed by stability often indicates the
        sensor switched from fetal to maternal signal.

        Args:
            fhr: FHR signal array.
            sampling_rate: Sampling rate in Hz.

        Returns:
            BaselineJumpResult indicating if jump was detected.
        """
        window = int(5 * sampling_rate)  # 5 seconds

        if len(fhr) < window * 3:
            return BaselineJumpResult(jump_detected=False)

        # Handle NaN values
        fhr_clean = np.copy(fhr)
        nan_mask = np.isnan(fhr_clean)
        if np.any(nan_mask):
            valid_mean = np.nanmean(fhr_clean)
            fhr_clean[nan_mask] = valid_mean

        # Rolling mean
        rolling_mean = np.convolve(
            fhr_clean,
            np.ones(window) / window,
            mode='valid'
        )

        if len(rolling_mean) < 2:
            return BaselineJumpResult(jump_detected=False)

        # Derivative (bpm per second)
        derivative = np.diff(rolling_mean) * sampling_rate

        # Find large jumps (>4 bpm/sec = >20 bpm in 5 sec)
        jump_threshold = self.config.jump_magnitude_threshold / self.config.jump_duration_threshold
        jump_indices = np.where(np.abs(derivative) > jump_threshold)[0]

        if len(jump_indices) == 0:
            return BaselineJumpResult(jump_detected=False)

        # Check if signal is stable after the most recent jump
        last_jump = jump_indices[-1]
        post_jump = rolling_mean[last_jump:]

        if len(post_jump) > window:
            post_jump_std = np.std(post_jump)

            if post_jump_std < self.config.post_jump_stability_threshold:
                jump_magnitude = abs(derivative[last_jump]) * self.config.jump_duration_threshold
                confidence = min(0.9, abs(derivative[last_jump]) / (jump_threshold * 2))

                return BaselineJumpResult(
                    jump_detected=True,
                    jump_magnitude=jump_magnitude,
                    jump_duration=self.config.jump_duration_threshold,
                    confidence=confidence
                )

        return BaselineJumpResult(jump_detected=False)

    def _fuse_results(
        self,
        results: List[DetectionResult],
        has_accelerations: bool,
        segment_length: int
    ) -> MHRCheckResult:
        """
        Weighted voting fusion of detection methods.

        CRITICAL: Applies fetal sleep adjustment. If accelerations
        are present, confidence is reduced because:
        - Sleeping fetus: Low variability BUT retains accelerations
        - MHR contamination: No accelerations (maternal signal doesn't have them)

        Weights:
            - cross_correlation: 0.5 (most reliable if available)
            - spectral_rsa: 0.3 (good but can have false positives)
            - baseline_jump: 0.2 (supplementary)

        Args:
            results: List of detection results from each method.
            has_accelerations: True if accelerations present.
            segment_length: Length of analyzed segment.

        Returns:
            MHRCheckResult with fused confidence and recommended action.
        """
        if not results:
            return MHRCheckResult.no_suspicion()

        weights = {
            "cross_correlation": self.config.cross_correlation_weight,
            "spectral_rsa": self.config.spectral_rsa_weight,
            "baseline_jump": self.config.baseline_jump_weight
        }

        total_weight = 0.0
        weighted_confidence = 0.0
        reasons = []

        for r in results:
            if r.is_suspected:
                w = weights.get(r.method, 0.1)
                weighted_confidence += r.confidence * w
                total_weight += w
                reasons.append(r.reason)

        if total_weight == 0:
            return MHRCheckResult.no_suspicion()

        final_confidence = weighted_confidence / total_weight

        # ─────────────────────────────────────────────────────────
        # CRITICAL: Fetal Sleep Adjustment
        # ─────────────────────────────────────────────────────────
        # If accelerations are present, reduce MHR suspicion.
        # Sleeping fetus shows low variability but RETAINS accelerations.
        # MHR contamination typically shows NO accelerations.
        if has_accelerations and final_confidence > 0.3:
            original_confidence = final_confidence
            final_confidence *= self.config.acceleration_dampening
            reasons.append(
                f"Confidence reduced {original_confidence:.0%}→{final_confidence:.0%} "
                f"(accelerations present - may be fetal sleep)"
            )
            logger.info(
                f"MHR confidence dampened due to accelerations: "
                f"{original_confidence:.2f} → {final_confidence:.2f}"
            )

        # Determine recommended action
        action = self._determine_action(final_confidence)

        return MHRCheckResult(
            is_suspected=(final_confidence > self.config.warn_confidence_threshold),
            confidence=final_confidence,
            reasons=reasons,
            recommended_action=action,
            detection_timestamp=time.time(),
            segment_start_idx=0,
            segment_end_idx=segment_length
        )

    def _determine_action(self, confidence: float) -> MHRAction:
        """
        Determine recommended action based on confidence level.

        Args:
            confidence: MHR detection confidence (0-1).

        Returns:
            MHRAction indicating what to do.
        """
        if confidence > self.config.block_confidence_threshold:
            return MHRAction.BLOCK_SEGMENT
        elif confidence > self.config.warn_confidence_threshold:
            return MHRAction.WARN_USER
        else:
            return MHRAction.NONE
