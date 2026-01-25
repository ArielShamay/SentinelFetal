"""
Trend Buffer Module.

Provides an efficient circular buffer for storing trend metrics over time.
Used by TrendAnalyzer to track 60-minute trends in variability, baseline,
and deceleration frequency.

CRITICAL: Implements FSQI masking - samples with poor signal quality
are excluded from trend analysis to prevent artifact contamination.

References:
    - SentinelFetal V2.0 PRD, Section: Trend Analyzer Module
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrendDataPoint:
    """
    Single point in trend history (sampled every 2 minutes).

    Stores aggregated metrics from a single analysis window,
    not raw signal data.
    """
    timestamp: float            # Unix timestamp
    variability: float          # Variability value in bpm
    baseline: float             # Baseline value in bpm
    decel_count_15min: int      # Decelerations in last 15 minutes
    has_late_decel: bool        # Late deceleration detected this window
    has_variable_decel: bool    # Variable deceleration detected this window
    category: int               # Classification category (1, 2, or 3)
    fsqi_score: float = 1.0     # Signal quality score (0-1)


class TrendBuffer:
    """
    Fixed-size circular buffer for trend metrics.

    Stores 60 minutes of data at 2-minute intervals = 30 points max.
    Automatically discards oldest data when full.

    CRITICAL: Implements FSQI masking - samples with score < 0.9
    are NOT added to the buffer, preventing artifact contamination.

    Example:
        >>> buffer = TrendBuffer(max_minutes=60, sample_interval_minutes=2)
        >>> buffer.add_sample(TrendDataPoint(
        ...     timestamp=time.time(),
        ...     variability=12.5,
        ...     baseline=140,
        ...     fsqi_score=0.95  # Will be added
        ... ))
        >>> print(buffer.get_minutes_of_data())
    """

    # Minimum FSQI score to include sample in buffer
    MIN_FSQI_THRESHOLD = 0.9

    def __init__(
        self,
        max_minutes: int = 60,
        sample_interval_minutes: int = 2
    ):
        """
        Initialize trend buffer.

        Args:
            max_minutes: Maximum history to retain (default: 60 minutes).
            sample_interval_minutes: Time between samples (default: 2 minutes).
        """
        self.max_minutes = max_minutes
        self.sample_interval_minutes = sample_interval_minutes
        self.max_points = max_minutes // sample_interval_minutes
        self._buffer: deque[TrendDataPoint] = deque(maxlen=self.max_points)
        self._last_sample_time: float = 0.0
        self.sample_interval_seconds = sample_interval_minutes * 60

        # Statistics
        self._total_samples_received = 0
        self._samples_masked = 0

    def should_sample(self, current_time: float) -> bool:
        """
        Check if enough time has passed for a new sample.

        Args:
            current_time: Current Unix timestamp.

        Returns:
            True if we should take a new sample.
        """
        return (current_time - self._last_sample_time) >= self.sample_interval_seconds

    def add_sample(self, data_point: TrendDataPoint) -> bool:
        """
        Add new sample to buffer.

        CRITICAL: Implements FSQI masking. Samples with poor signal
        quality (FSQI < 0.9) are silently dropped to prevent artifact
        contamination of trend analysis.

        Args:
            data_point: The trend data point to add.

        Returns:
            True if sample was added, False if masked due to poor FSQI.
        """
        self._total_samples_received += 1

        # CRITICAL: FSQI masking - do not store low-quality samples
        if data_point.fsqi_score < self.MIN_FSQI_THRESHOLD:
            self._samples_masked += 1
            logger.debug(
                f"Trend sample masked: FSQI={data_point.fsqi_score:.2f} "
                f"< {self.MIN_FSQI_THRESHOLD}"
            )
            return False

        self._buffer.append(data_point)
        self._last_sample_time = data_point.timestamp
        return True

    def get_variability_series(self) -> np.ndarray:
        """Get variability values for trend analysis."""
        if not self._buffer:
            return np.array([])
        return np.array([p.variability for p in self._buffer])

    def get_baseline_series(self) -> np.ndarray:
        """Get baseline values for drift detection."""
        if not self._buffer:
            return np.array([])
        return np.array([p.baseline for p in self._buffer])

    def get_category_series(self) -> np.ndarray:
        """Get category history."""
        if not self._buffer:
            return np.array([])
        return np.array([p.category for p in self._buffer])

    def get_timestamps(self) -> np.ndarray:
        """Get timestamp array for plotting."""
        if not self._buffer:
            return np.array([])
        return np.array([p.timestamp for p in self._buffer])

    def get_decel_events_in_window(self, window_minutes: int) -> List[TrendDataPoint]:
        """
        Get data points with decelerations in last N minutes.

        Args:
            window_minutes: Time window to look back.

        Returns:
            List of TrendDataPoint objects that had decelerations.
        """
        if not self._buffer:
            return []

        cutoff = time.time() - (window_minutes * 60)
        return [
            p for p in self._buffer
            if p.timestamp > cutoff and (p.has_late_decel or p.has_variable_decel)
        ]

    def get_late_decel_count_in_window(self, window_minutes: int) -> int:
        """Count late decelerations in time window."""
        cutoff = time.time() - (window_minutes * 60)
        return sum(
            1 for p in self._buffer
            if p.timestamp > cutoff and p.has_late_decel
        )

    def get_variable_decel_count_in_window(self, window_minutes: int) -> int:
        """Count variable decelerations in time window."""
        cutoff = time.time() - (window_minutes * 60)
        return sum(
            1 for p in self._buffer
            if p.timestamp > cutoff and p.has_variable_decel
        )

    def get_minutes_of_data(self) -> float:
        """
        Get how many minutes of history we have.

        Returns:
            Duration in minutes, or 0 if insufficient data.
        """
        if len(self._buffer) < 2:
            return 0.0
        return (self._buffer[-1].timestamp - self._buffer[0].timestamp) / 60

    def get_most_recent(self) -> Optional[TrendDataPoint]:
        """Get most recent data point, or None if empty."""
        if not self._buffer:
            return None
        return self._buffer[-1]

    def clear(self) -> None:
        """Clear all data from buffer."""
        self._buffer.clear()
        self._last_sample_time = 0.0

    @property
    def size(self) -> int:
        """Current number of samples in buffer."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """True if buffer is at maximum capacity."""
        return len(self._buffer) >= self.max_points

    @property
    def mask_rate(self) -> float:
        """Percentage of samples that were masked due to poor FSQI."""
        if self._total_samples_received == 0:
            return 0.0
        return self._samples_masked / self._total_samples_received

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "size": self.size,
            "max_points": self.max_points,
            "minutes_of_data": self.get_minutes_of_data(),
            "total_received": self._total_samples_received,
            "samples_masked": self._samples_masked,
            "mask_rate": f"{self.mask_rate:.1%}"
        }
