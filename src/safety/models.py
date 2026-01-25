"""
MHR Guard Module - Data Models.

Defines dataclasses for Maternal Heart Rate detection results.

References:
    - SentinelFetal V2.0 PRD, Section: MHR Guard Module
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
import time


class MHRAction(Enum):
    """Recommended action based on MHR detection confidence."""
    NONE = auto()           # No MHR suspected - continue normal processing
    WARN_USER = auto()      # Alert but continue classification (moderate confidence)
    BLOCK_SEGMENT = auto()  # Exclude from ML classification (high confidence)


@dataclass
class DetectionResult:
    """Result from a single MHR detection method."""
    method: str             # "cross_correlation", "spectral_rsa", "baseline_jump"
    is_suspected: bool      # True if this method suspects MHR
    confidence: float       # 0.0 to 1.0
    reason: str             # Human-readable explanation


@dataclass
class SpectralResult:
    """
    Result of RSA (Respiratory Sinus Arrhythmia) spectral analysis.

    RSA creates characteristic frequency signatures:
    - Adult RSA: 0.15-0.35 Hz (12-20 breaths/minute)
    - Fetal RSA: 0.4-1.0 Hz (faster breathing movements)

    If adult RSA band dominates, signal may be maternal.
    """
    adult_band_power: float     # Power in 0.15-0.35 Hz band
    fetal_band_power: float     # Power in 0.4-1.0 Hz band
    total_power: float          # Total spectral power
    spectral_centroid: float    # Center of mass of spectrum (Hz)
    dominant_frequency: float   # Frequency with highest power (Hz)

    @property
    def adult_power_ratio(self) -> float:
        """Ratio of adult RSA power to total power."""
        if self.total_power <= 0:
            return 0.0
        return self.adult_band_power / self.total_power


@dataclass
class BaselineJumpResult:
    """
    Result of baseline jump detection.

    A sudden baseline jump (>20 bpm in <5 seconds) followed by
    stability may indicate sensor switching from fetal to maternal signal.
    """
    jump_detected: bool
    jump_magnitude: float = 0.0     # Magnitude of jump in bpm
    jump_duration: float = 0.0      # Duration of jump in seconds
    confidence: float = 0.0         # Detection confidence 0-1


@dataclass
class MHRCheckResult:
    """
    Final result of MHR (Maternal Heart Rate) detection.

    Combines results from multiple detection methods:
    1. Cross-correlation with MHR reference (if available)
    2. Spectral RSA analysis
    3. Baseline jump detection

    The fused result determines the recommended action.
    """
    is_suspected: bool              # True if MHR contamination suspected
    confidence: float               # Overall confidence 0.0 to 1.0
    reasons: List[str]              # List of contributing reasons
    recommended_action: MHRAction   # What to do with this segment
    detection_timestamp: float = field(default_factory=time.time)
    segment_start_idx: int = 0      # Start of analyzed segment
    segment_end_idx: int = 0        # End of analyzed segment

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_suspected": self.is_suspected,
            "confidence": round(self.confidence, 3),
            "reasons": self.reasons,
            "action": self.recommended_action.name,
            "timestamp": self.detection_timestamp,
            "segment_start": self.segment_start_idx,
            "segment_end": self.segment_end_idx
        }

    @classmethod
    def no_suspicion(cls) -> MHRCheckResult:
        """Create a result indicating no MHR suspicion."""
        return cls(
            is_suspected=False,
            confidence=0.0,
            reasons=[],
            recommended_action=MHRAction.NONE
        )


@dataclass
class MHRDetectorConfig:
    """Configuration for MHR detector."""
    # Cross-correlation thresholds
    cross_correlation_threshold: float = 0.8  # Correlation above this = suspected

    # Spectral analysis settings
    adult_rsa_band: tuple = (0.15, 0.35)      # Hz - adult breathing rate
    fetal_rsa_band: tuple = (0.4, 1.0)        # Hz - fetal breathing rate
    adult_power_ratio_threshold: float = 0.4   # Ratio above this = suspected
    spectral_centroid_threshold: float = 0.35  # Centroid below this = suspected

    # Baseline jump settings
    jump_magnitude_threshold: float = 20.0     # bpm - minimum jump to detect
    jump_duration_threshold: float = 5.0       # seconds - max time for jump
    post_jump_stability_threshold: float = 5.0 # bpm std - stability after jump

    # Fusion settings
    warn_confidence_threshold: float = 0.5     # Above this = WARN_USER
    block_confidence_threshold: float = 0.8    # Above this = BLOCK_SEGMENT

    # Method weights for fusion
    cross_correlation_weight: float = 0.5      # Most reliable if available
    spectral_rsa_weight: float = 0.3           # Good but can have false positives
    baseline_jump_weight: float = 0.2          # Supplementary evidence

    # Fetal sleep detection
    acceleration_dampening: float = 0.5        # Reduce confidence if accelerations present
