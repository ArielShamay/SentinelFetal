"""
Trend Analyzer Module.

Analyzes 60-minute trends in clinical metrics to detect gradual deterioration
that instant analysis misses. MiniRocket's PPV pooling erases temporal order;
this module restores clinical awareness of "getting worse over time."

Key Features:
    - Variability trend analysis (slope calculation)
    - Baseline drift detection
    - Deceleration frequency tracking
    - Composite deterioration score (0-100)

CRITICAL: Uses masked TrendBuffer to exclude signal artifacts from regression.

References:
    - SentinelFetal V2.0 PRD, Section: Trend Analyzer Module
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from src.analysis.trend_buffer import TrendBuffer

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Direction of metric trend over time."""
    DECLINING = "declining"     # Getting worse (e.g., variability decreasing)
    STABLE = "stable"           # No significant change
    IMPROVING = "improving"     # Getting better


class TrendAlertType(Enum):
    """Types of trend-based alerts."""
    VARIABILITY_DECLINING = "variability_declining"
    BASELINE_DRIFT = "baseline_drift"
    RECURRENT_DECELS = "recurrent_decels"
    RAPID_DETERIORATION = "rapid_deterioration"


@dataclass
class TrendAlert:
    """Alert generated from trend analysis."""
    type: TrendAlertType
    severity: str           # "LOW", "MEDIUM", "HIGH"
    message: str            # Human-readable message

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "severity": self.severity,
            "message": self.message
        }


@dataclass
class LinearTrendResult:
    """Result of linear regression on a metric series."""
    slope: float            # Units per 10 minutes
    r_squared: float        # Goodness of fit (0-1)
    confidence: float       # Confidence based on data and R²


@dataclass
class TrendAnalysisResult:
    """
    Complete result of 60-minute trend analysis.

    Provides insight into whether the fetus is deteriorating over time,
    which single-snapshot analysis cannot detect.
    """
    has_sufficient_data: bool = False
    minutes_of_data: float = 0.0

    # Variability trend
    variability_slope: float = 0.0      # bpm per 10 minutes
    variability_trend: TrendDirection = TrendDirection.STABLE
    variability_current: float = 0.0    # Most recent value

    # Baseline trend
    baseline_slope: float = 0.0         # bpm per 10 minutes
    baseline_trend: TrendDirection = TrendDirection.STABLE
    baseline_current: float = 0.0

    # Deceleration frequency (from trend buffer)
    decel_count_15min: int = 0
    decel_count_30min: int = 0
    decel_count_60min: int = 0

    # Late decel counts specifically
    late_decel_count_15min: int = 0
    late_decel_count_30min: int = 0

    # Deterioration score
    deterioration_score: int = 0        # 0-100

    # Generated alerts
    alerts: List[TrendAlert] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_data": self.has_sufficient_data,
            "minutes": round(self.minutes_of_data, 1),
            "variability_slope": round(self.variability_slope, 2),
            "variability_trend": self.variability_trend.value,
            "variability_current": round(self.variability_current, 1),
            "baseline_slope": round(self.baseline_slope, 2),
            "baseline_current": round(self.baseline_current, 1),
            "decel_count_15min": self.decel_count_15min,
            "decel_count_30min": self.decel_count_30min,
            "decel_count_60min": self.decel_count_60min,
            "late_decel_count_15min": self.late_decel_count_15min,
            "deterioration_score": self.deterioration_score,
            "alerts": [a.to_dict() for a in self.alerts]
        }


@dataclass
class TrendAnalyzerConfig:
    """Configuration for trend analyzer."""
    # Minimum data requirements
    min_data_minutes: float = 10.0      # Need at least 10 min for trend

    # Variability trend thresholds
    var_slope_declining_threshold: float = -0.5   # bpm per 10 min
    var_slope_improving_threshold: float = 0.5
    var_severe_slope_threshold: float = -1.0      # Rapid decline

    # Baseline drift thresholds
    baseline_drift_threshold: float = 5.0         # bpm per 10 min

    # Deceleration frequency thresholds
    recurrent_decels_threshold: int = 3           # per 15 min window

    # Deterioration score weights (must sum to 1.0)
    weight_variability: float = 0.4
    weight_decelerations: float = 0.3
    weight_baseline: float = 0.3

    # Deterioration score thresholds
    score_caution_threshold: int = 30
    score_warning_threshold: int = 50
    score_critical_threshold: int = 70


class TrendAnalyzer:
    """
    Analyzes trends in clinical metrics over 30-60 minutes.

    Detects gradual deterioration that instant analysis cannot see.
    MiniRocket's PPV pooling erases temporal information; this module
    restores clinical awareness of "getting worse."

    Example:
        >>> analyzer = TrendAnalyzer()
        >>> result = analyzer.analyze(patient_trend_buffer)
        >>> if result.deterioration_score > 70:
        ...     print("CRITICAL: Rapid deterioration detected!")
    """

    def __init__(self, config: Optional[TrendAnalyzerConfig] = None):
        """Initialize trend analyzer with configuration."""
        self.config = config or TrendAnalyzerConfig()

    def analyze(self, trend_buffer: TrendBuffer) -> TrendAnalysisResult:
        """
        Main analysis entry point.

        Algorithm:
            1. Check for sufficient data (>10 minutes)
            2. Compute variability trend (linear regression → slope)
            3. Compute baseline drift
            4. Count deceleration frequency in time windows
            5. Calculate composite deterioration score
            6. Generate alerts if thresholds exceeded

        Args:
            trend_buffer: Buffer containing historical trend data.

        Returns:
            TrendAnalysisResult with all metrics and alerts.
        """
        result = TrendAnalysisResult()

        # Check minimum data requirement
        minutes = trend_buffer.get_minutes_of_data()
        result.minutes_of_data = minutes

        if minutes < self.config.min_data_minutes:
            result.has_sufficient_data = False
            logger.debug(
                f"Insufficient trend data: {minutes:.1f} min "
                f"(need {self.config.min_data_minutes} min)"
            )
            return result

        result.has_sufficient_data = True

        # Get current values
        most_recent = trend_buffer.get_most_recent()
        if most_recent:
            result.variability_current = most_recent.variability
            result.baseline_current = most_recent.baseline

        # ─────────────────────────────────────────────────────────
        # ANALYSIS 1: Variability Trend
        # ─────────────────────────────────────────────────────────
        var_series = trend_buffer.get_variability_series()
        if len(var_series) >= 3:
            var_trend = self._compute_linear_trend(var_series)
            result.variability_slope = var_trend.slope
            result.variability_trend = self._classify_variability_trend(var_trend.slope)

            # Alert if declining rapidly
            if var_trend.slope < self.config.var_slope_declining_threshold:
                severity = "HIGH" if var_trend.slope < self.config.var_severe_slope_threshold else "MEDIUM"
                result.alerts.append(TrendAlert(
                    type=TrendAlertType.VARIABILITY_DECLINING,
                    severity=severity,
                    message=f"Variability declining: {var_trend.slope:.1f} bpm/10min"
                ))

        # ─────────────────────────────────────────────────────────
        # ANALYSIS 2: Baseline Drift
        # ─────────────────────────────────────────────────────────
        baseline_series = trend_buffer.get_baseline_series()
        if len(baseline_series) >= 3:
            baseline_trend = self._compute_linear_trend(baseline_series)
            result.baseline_slope = baseline_trend.slope

            # Alert if drifting significantly
            if abs(baseline_trend.slope) > self.config.baseline_drift_threshold:
                direction = "rising" if baseline_trend.slope > 0 else "falling"
                result.alerts.append(TrendAlert(
                    type=TrendAlertType.BASELINE_DRIFT,
                    severity="MEDIUM",
                    message=f"Baseline {direction}: {abs(baseline_trend.slope):.1f} bpm/10min"
                ))

        # ─────────────────────────────────────────────────────────
        # ANALYSIS 3: Deceleration Frequency
        # ─────────────────────────────────────────────────────────
        result.decel_count_15min = len(trend_buffer.get_decel_events_in_window(15))
        result.decel_count_30min = len(trend_buffer.get_decel_events_in_window(30))
        result.decel_count_60min = len(trend_buffer.get_decel_events_in_window(60))

        result.late_decel_count_15min = trend_buffer.get_late_decel_count_in_window(15)
        result.late_decel_count_30min = trend_buffer.get_late_decel_count_in_window(30)

        # Alert if recurrent decelerations
        if result.late_decel_count_15min >= self.config.recurrent_decels_threshold:
            result.alerts.append(TrendAlert(
                type=TrendAlertType.RECURRENT_DECELS,
                severity="HIGH",
                message=f"Recurrent late decelerations: {result.late_decel_count_15min} in last 15 min"
            ))
        elif result.decel_count_15min >= self.config.recurrent_decels_threshold:
            result.alerts.append(TrendAlert(
                type=TrendAlertType.RECURRENT_DECELS,
                severity="MEDIUM",
                message=f"Recurrent decelerations: {result.decel_count_15min} in last 15 min"
            ))

        # ─────────────────────────────────────────────────────────
        # ANALYSIS 4: Deterioration Score
        # ─────────────────────────────────────────────────────────
        result.deterioration_score = self._calculate_deterioration_score(
            var_slope=result.variability_slope,
            baseline_slope=result.baseline_slope,
            decel_count=result.late_decel_count_15min,
            current_variability=result.variability_current
        )

        # Alert on rapid deterioration
        if result.deterioration_score >= self.config.score_critical_threshold:
            result.alerts.append(TrendAlert(
                type=TrendAlertType.RAPID_DETERIORATION,
                severity="HIGH",
                message=f"Rapid deterioration: Score {result.deterioration_score}/100"
            ))

        return result

    def _compute_linear_trend(self, series: np.ndarray) -> LinearTrendResult:
        """
        Compute linear regression on metric series.

        Uses simple linear regression to find slope.
        Returns slope in units per 10 minutes (assuming 2-min samples).

        Args:
            series: Array of metric values over time.

        Returns:
            LinearTrendResult with slope, R², and confidence.
        """
        if len(series) < 3:
            return LinearTrendResult(slope=0.0, r_squared=0.0, confidence=0.0)

        # Time axis in 10-minute units (2 min intervals → divide by 5)
        x = np.arange(len(series)) * 2 / 10  # Convert to 10-min units

        # Simple linear regression: y = mx + b
        # slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
        x_mean = np.mean(x)
        y_mean = np.mean(series)

        numerator = np.sum((x - x_mean) * (series - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator < 1e-10:
            return LinearTrendResult(slope=0.0, r_squared=0.0, confidence=0.0)

        slope = numerator / denominator

        # Calculate R² for confidence
        y_pred = slope * (x - x_mean) + y_mean
        ss_res = np.sum((series - y_pred) ** 2)
        ss_tot = np.sum((series - y_mean) ** 2)

        r_squared = 0.0
        if ss_tot > 1e-10:
            r_squared = max(0.0, 1 - (ss_res / ss_tot))

        # Confidence based on R² and data points
        confidence = r_squared if len(series) >= 5 else r_squared * 0.5

        return LinearTrendResult(
            slope=float(slope),
            r_squared=float(r_squared),
            confidence=float(confidence)
        )

    def _classify_variability_trend(self, slope: float) -> TrendDirection:
        """Classify variability slope into trend direction."""
        if slope < self.config.var_slope_declining_threshold:
            return TrendDirection.DECLINING
        elif slope > self.config.var_slope_improving_threshold:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.STABLE

    def _calculate_deterioration_score(
        self,
        var_slope: float,
        baseline_slope: float,
        decel_count: int,
        current_variability: float
    ) -> int:
        """
        Calculate composite deterioration score (0-100).

        Formula:
            Score = 40% × Variability_Penalty +
                    30% × Decel_Penalty +
                    30% × Baseline_Penalty

        Interpretation:
            0-30:  Good - Stable or improving
            31-50: Caution - Minor concerns
            51-70: Warning - Deteriorating
            71-100: Critical - Rapid deterioration

        Args:
            var_slope: Variability slope (bpm per 10 min).
            baseline_slope: Baseline slope (bpm per 10 min).
            decel_count: Late decelerations in 15 min.
            current_variability: Current variability value.

        Returns:
            Integer score 0-100.
        """
        # Variability penalty: declining slope + low absolute value
        # -2 bpm/10min slope = max penalty
        var_slope_penalty = min(1.0, max(0.0, -var_slope) / 2.0)

        # Low variability level also contributes (<8 bpm is concerning)
        var_level_penalty = min(1.0, max(0.0, (8 - current_variability) / 8))

        # Combined variability penalty (60% slope, 40% level)
        var_penalty = 0.6 * var_slope_penalty + 0.4 * var_level_penalty

        # Deceleration penalty: count-based
        # 5+ decels in 15min = max penalty
        decel_penalty = min(1.0, decel_count / 5)

        # Baseline penalty: abnormal drift in either direction
        # 10 bpm/10min drift = max penalty
        baseline_penalty = min(1.0, abs(baseline_slope) / 10)

        # Weighted sum
        score = (
            self.config.weight_variability * var_penalty +
            self.config.weight_decelerations * decel_penalty +
            self.config.weight_baseline * baseline_penalty
        )

        # Scale to 0-100
        return min(100, max(0, int(score * 100)))


def get_deterioration_color(score: int) -> str:
    """
    Get color for deterioration score display.

    Args:
        score: Deterioration score 0-100.

    Returns:
        Hex color string.
    """
    if score <= 30:
        return "#28a745"  # Green - Good
    elif score <= 50:
        return "#ffc107"  # Yellow - Caution
    elif score <= 70:
        return "#fd7e14"  # Orange - Warning
    else:
        return "#dc3545"  # Red - Critical


def get_trend_arrow(direction: TrendDirection) -> str:
    """
    Get arrow emoji for trend direction.

    Args:
        direction: TrendDirection enum value.

    Returns:
        Arrow emoji string.
    """
    arrows = {
        TrendDirection.DECLINING: "↘",
        TrendDirection.STABLE: "→",
        TrendDirection.IMPROVING: "↗"
    }
    return arrows.get(direction, "→")
