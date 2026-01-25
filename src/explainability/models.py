"""
Explainability Module - Data Models.

Defines dataclasses for classification explanations and visual highlighting.

References:
    - SentinelFetal V2.0 PRD, Section: Explainability Module
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ContributorSource(Enum):
    """Source of the explanation contributor."""
    RULE = "rule"       # Deterministic rule-based
    ML = "ml"           # Machine learning feature attribution


@dataclass
class TimeRegion:
    """
    Region of the FHR signal that contributed to classification.

    Used for visual highlighting on the CTG graph.
    Indices can be negative (relative to end of signal).
    """
    start_index: int        # Start sample index (can be negative)
    end_index: int          # End sample index (can be negative)
    color: str              # Color name: "red", "orange", "yellow", "green", "blue"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "start": self.start_index,
            "end": self.end_index,
            "color": self.color
        }


@dataclass
class Contributor:
    """
    A factor that contributed to the classification decision.

    Positive contribution = pushes toward pathological.
    Negative contribution = mitigating factor (pushes toward normal).
    """
    source: ContributorSource   # "rule" or "ml"
    name: str                   # e.g., "late_decel", "variability", "baseline"
    contribution: float         # Positive = pathological, Negative = mitigating
    description: str            # Human-readable explanation
    time_region: Optional[TimeRegion] = None  # Where on the graph
    is_mitigating: bool = False  # True if this reduces severity

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source.value,
            "name": self.name,
            "contribution": round(self.contribution, 3),
            "description": self.description,
            "is_mitigating": self.is_mitigating,
            "time_region": self.time_region.to_dict() if self.time_region else None
        }


@dataclass
class HighlightRegion:
    """
    Visual highlight region for the CTG graph.

    Used by the UI to overlay colored regions on the FHR trace.
    """
    start: int              # Sample index (absolute, 0-based)
    end: int                # Sample index (absolute, 0-based)
    color: str              # RGBA string, e.g., "rgba(220, 53, 69, 0.3)"
    label: str              # Short label for the region
    is_pathological: bool   # True if this region contributed to pathological

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "color": self.color,
            "label": self.label,
            "is_pathological": self.is_pathological
        }


@dataclass
class RuleExplanation:
    """
    Explanation from a single clinical rule.

    These are deterministic and always available (no ML required).
    """
    rule_name: str              # e.g., "sinusoidal", "late_decel", "variability"
    contribution: float         # Positive = toward pathological
    description: str            # Human-readable explanation with values
    time_region: Optional[TimeRegion]  # Where this was detected
    severity: str               # "NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"


@dataclass
class SHAPExplanation:
    """
    Explanation from SHAP feature attribution.

    Provides insight into which ML features drove the prediction.
    """
    feature_index: int          # Index in feature vector
    feature_name: str           # Human-readable name
    shap_value: float           # SHAP contribution value
    clinical_category: str      # Clinical category: "Variability", "Baseline", etc.
    description: str            # Generated description


@dataclass
class ExplanationResult:
    """
    Complete explanation for a classification decision.

    Combines rule-based and (optionally) ML-based explanations,
    with visual highlighting information for the UI.
    """
    summary: str                            # Natural language summary
    contributors: List[Contributor] = field(default_factory=list)  # Ranked by |contribution|
    highlights: List[HighlightRegion] = field(default_factory=list)  # For graph overlay
    confidence: float = 0.0                 # Classification confidence

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": self.summary,
            "contributors": [c.to_dict() for c in self.contributors],
            "highlights": [h.to_dict() for h in self.highlights],
            "confidence": round(self.confidence, 3)
        }

    @property
    def top_contributors(self) -> List[Contributor]:
        """Get top 5 contributors by absolute contribution."""
        sorted_contribs = sorted(
            self.contributors,
            key=lambda c: abs(c.contribution),
            reverse=True
        )
        return sorted_contribs[:5]

    @property
    def pathological_factors(self) -> List[Contributor]:
        """Get factors pushing toward pathological (positive contribution)."""
        return [c for c in self.contributors if c.contribution > 0 and not c.is_mitigating]

    @property
    def mitigating_factors(self) -> List[Contributor]:
        """Get factors pushing toward normal (negative contribution)."""
        return [c for c in self.contributors if c.is_mitigating or c.contribution < 0]
