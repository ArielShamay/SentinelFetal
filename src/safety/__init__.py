"""
Safety Module - MHR Guard.

Provides maternal heart rate contamination detection to prevent
misclassification when the fetal monitor picks up maternal signal.

Usage:
    from src.safety import MHRDetector, MHRCheckResult, MHRAction

    detector = MHRDetector()
    result = detector.check_segment(fhr_segment)

    if result.recommended_action == MHRAction.BLOCK_SEGMENT:
        # Do not use this segment for ML classification
        pass

References:
    - SentinelFetal V2.0 PRD, Section: MHR Guard Module
"""

from src.safety.models import (
    MHRAction,
    MHRCheckResult,
    MHRDetectorConfig,
    DetectionResult,
    SpectralResult,
    BaselineJumpResult,
)
from src.safety.spectral_analyzer import SpectralAnalyzer
from src.safety.mhr_detector import MHRDetector

__all__ = [
    "MHRDetector",
    "MHRDetectorConfig",
    "MHRAction",
    "MHRCheckResult",
    "DetectionResult",
    "SpectralResult",
    "BaselineJumpResult",
    "SpectralAnalyzer",
]
