"""
Explanation Engine - Main Orchestrator.

Combines rule-based and ML-based explanations into a unified
explanation result with visual highlighting.

Priority Order:
    1. Rule-based explanations (deterministic, always available)
    2. SHAP explanations (on-demand, for ML component)
    3. Visual highlight mapping

References:
    - SentinelFetal V2.0 PRD, Section: Explainability Module
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.explainability.models import (
    Contributor,
    ExplanationResult,
)
from src.explainability.rule_explainer import RuleExplainer
from src.explainability.visual_mapper import VisualMapper

logger = logging.getLogger(__name__)


# Category names for summary generation
CATEGORY_NAMES = {
    0: "Normal (Category I)",
    1: "Normal (Category I)",
    2: "Intermediate (Category II)",
    3: "Pathological (Category III)",
}


class ExplanationEngine:
    """
    Generates comprehensive explanations for classification decisions.

    Orchestrates rule-based and ML-based explanation methods,
    combining them into a unified result with visual highlights.

    PRIORITY ORDER:
        1. Rule-based explanations (most interpretable, always available)
        2. SHAP feature attribution (for ML component, on-demand only)
        3. Visual highlight mapping (for graph overlay)

    Example:
        >>> engine = ExplanationEngine(xgboost_model=classifier.model)
        >>> result = engine.explain(
        ...     category=3,
        ...     confidence=0.94,
        ...     rule_outputs={"baseline": baseline, "variability": variability, ...},
        ...     fhr_length=2400
        ... )
        >>> print(result.summary)
        >>> for c in result.top_contributors:
        ...     print(f"  {c.name}: {c.description}")
    """

    def __init__(self, xgboost_model: Optional[Any] = None):
        """
        Initialize explanation engine.

        Args:
            xgboost_model: XGBoost model for SHAP explanations (optional).
        """
        self.rule_explainer = RuleExplainer()
        self.visual_mapper = VisualMapper()

        # SHAP explainer is optional
        self.shap_explainer = None
        if xgboost_model is not None:
            try:
                from src.explainability.shap_explainer import SHAPExplainer
                self.shap_explainer = SHAPExplainer(xgboost_model)
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer: {e}")

    def explain(
        self,
        category: int,
        confidence: float,
        rule_outputs: Dict[str, Any],
        fhr_length: int = 2400,
        ml_features: Optional[np.ndarray] = None,
        compute_shap: bool = False
    ) -> ExplanationResult:
        """
        Generate comprehensive explanation for a classification.

        Algorithm:
            1. Generate rule-based explanations (always)
            2. If compute_shap=True and ML features provided, add SHAP
            3. Rank all contributors by absolute impact
            4. Generate visual highlight regions
            5. Create natural language summary

        Args:
            category: Classification category (1, 2, or 3).
            confidence: Classification confidence (0-1).
            rule_outputs: Dictionary of rule engine results.
            fhr_length: Length of FHR signal for highlight mapping.
            ml_features: Feature vector for SHAP (optional).
            compute_shap: If True, compute SHAP explanations (slow!).

        Returns:
            ExplanationResult with summary, contributors, and highlights.
        """
        contributors: List[Contributor] = []

        # ─────────────────────────────────────────────────────────
        # STEP 1: Rule-Based Explanations (always)
        # ─────────────────────────────────────────────────────────
        rule_explanations = self.rule_explainer.explain(rule_outputs)
        rule_contributors = self.rule_explainer.to_contributors(rule_explanations)
        contributors.extend(rule_contributors)

        # ─────────────────────────────────────────────────────────
        # STEP 2: SHAP Explanations (on-demand only)
        # ─────────────────────────────────────────────────────────
        if (compute_shap and
            ml_features is not None and
            self.shap_explainer is not None and
            self.shap_explainer.is_available):

            try:
                shap_explanations = self.shap_explainer.explain(ml_features, top_n=5)

                # Filter to avoid duplicate coverage with rules
                filtered_shap = self._filter_covered_shap(
                    shap_explanations, rule_explanations
                )

                shap_contributors = self.shap_explainer.to_contributors(filtered_shap)

                # Only add top 3 SHAP features to avoid clutter
                contributors.extend(shap_contributors[:3])

            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        # ─────────────────────────────────────────────────────────
        # STEP 3: Rank Contributors
        # ─────────────────────────────────────────────────────────
        contributors.sort(key=lambda c: abs(c.contribution), reverse=True)

        # ─────────────────────────────────────────────────────────
        # STEP 4: Generate Visual Highlights
        # ─────────────────────────────────────────────────────────
        highlights = self.visual_mapper.create_highlights(
            contributors=contributors,
            fhr_length=fhr_length,
            sampling_rate=4.0
        )

        # Merge overlapping highlights for cleaner display
        highlights = self.visual_mapper.merge_overlapping(highlights)

        # ─────────────────────────────────────────────────────────
        # STEP 5: Generate Summary
        # ─────────────────────────────────────────────────────────
        summary = self._generate_summary(category, confidence, contributors)

        return ExplanationResult(
            summary=summary,
            contributors=contributors,
            highlights=highlights,
            confidence=confidence
        )

    def _filter_covered_shap(
        self,
        shap_explanations: List,
        rule_explanations: List
    ) -> List:
        """
        Filter SHAP features that overlap with rule explanations.

        Avoids redundant information (e.g., if we already explain
        "late deceleration" via rules, don't also show SHAP's
        "decel_late_count" feature).

        Args:
            shap_explanations: List of SHAP explanations.
            rule_explanations: List of rule explanations.

        Returns:
            Filtered list of SHAP explanations.
        """
        # Map rule names to SHAP feature prefixes
        rule_to_shap = {
            "variability": ["variability_"],
            "baseline": ["baseline_"],
            "late_decel": ["decel_late"],
            "variable_decel": ["decel_variable"],
            "sinusoidal": ["sinusoidal_"],
            "tachysystole": ["tachysystole_"],
            "accelerations": ["accel_"],
        }

        # Build set of covered SHAP prefixes
        covered_prefixes = set()
        for rule_exp in rule_explanations:
            prefixes = rule_to_shap.get(rule_exp.rule_name, [])
            covered_prefixes.update(prefixes)

        # Filter out covered SHAP explanations
        filtered = []
        for shap_exp in shap_explanations:
            is_covered = any(
                shap_exp.feature_name.startswith(prefix)
                for prefix in covered_prefixes
            )
            if not is_covered:
                filtered.append(shap_exp)

        return filtered

    def _generate_summary(
        self,
        category: int,
        confidence: float,
        contributors: List[Contributor]
    ) -> str:
        """
        Generate natural language summary of the classification.

        Args:
            category: Classification category.
            confidence: Confidence score.
            contributors: Ranked list of contributors.

        Returns:
            Multi-line summary string.
        """
        category_name = CATEGORY_NAMES.get(category, f"Category {category}")

        # Get top concerns (positive contribution)
        concerns = [c for c in contributors if c.contribution > 0 and not c.is_mitigating][:3]

        # Get mitigating factors (negative contribution)
        mitigating = [c for c in contributors if c.is_mitigating or c.contribution < 0][:2]

        # Build summary
        lines = [
            f"Classification: {category_name}",
            f"Confidence: {confidence:.0%}",
            ""
        ]

        if concerns:
            lines.append("Main Concerns:")
            for c in concerns:
                lines.append(f"  \u2022 {c.description}")
            lines.append("")

        if mitigating:
            lines.append("Mitigating Factors:")
            for c in mitigating:
                lines.append(f"  \u2022 {c.description}")

        return "\n".join(lines)

    @property
    def shap_available(self) -> bool:
        """True if SHAP explanations are available."""
        return (
            self.shap_explainer is not None and
            self.shap_explainer.is_available
        )


# =========================================================================
# Singleton Instance
# =========================================================================

_engine_instance: Optional[ExplanationEngine] = None


def get_explanation_engine() -> ExplanationEngine:
    """Get or create the singleton ExplanationEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ExplanationEngine()
    return _engine_instance
