"""
Rule Explainer Module.

Generates human-readable explanations from rule engine outputs.
These explanations are deterministic and always available (no ML required).

Provides clear, clinically meaningful descriptions of why each
rule contributed to the classification decision.

References:
    - SentinelFetal V2.0 PRD, Section: Explainability Module
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.explainability.models import (
    ContributorSource,
    Contributor,
    RuleExplanation,
    TimeRegion,
)

logger = logging.getLogger(__name__)


class RuleExplainer:
    """
    Converts rule engine outputs into human-readable explanations.

    These explanations are DETERMINISTIC and always available,
    providing the foundation for classification explainability.

    Example:
        >>> explainer = RuleExplainer()
        >>> explanations = explainer.explain({
        ...     "baseline": baseline_result,
        ...     "variability": variability_result,
        ...     "decelerations": deceleration_list,
        ... })
        >>> for exp in explanations:
        ...     print(f"{exp.rule_name}: {exp.description}")
    """

    def explain(self, rule_outputs: Dict[str, Any]) -> List[RuleExplanation]:
        """
        Process all rule outputs and generate explanations.

        Args:
            rule_outputs: Dictionary containing rule engine results:
                - "baseline": BaselineResult
                - "variability": VariabilityResult
                - "decelerations": List[Deceleration]
                - "tachysystole": TachysystoleResult
                - "sinusoidal": SinusoidalResult
                - "accelerations": List[Acceleration] (optional)

        Returns:
            List of RuleExplanation objects, sorted by contribution.
        """
        explanations: List[RuleExplanation] = []

        # ─────────────────────────────────────────────────────────
        # SINUSOIDAL (Critical - always first if present)
        # ─────────────────────────────────────────────────────────
        sinusoidal = rule_outputs.get("sinusoidal")
        if sinusoidal and self._get_attr(sinusoidal, "detected", False):
            freq = self._get_attr(sinusoidal, "frequency_cycles_per_min", 0)
            amp = self._get_attr(sinusoidal, "amplitude", 0)
            explanations.append(RuleExplanation(
                rule_name="sinusoidal",
                contribution=0.8,  # Very high contribution
                description=(
                    f"Sinusoidal pattern detected: {freq:.1f} cycles/min, "
                    f"amplitude {amp:.1f} bpm - SEVERE (indicates fetal anemia)"
                ),
                time_region=TimeRegion(
                    start_index=-4800,  # Last 20 minutes at 4Hz
                    end_index=-1,
                    color="red"
                ),
                severity="CRITICAL"
            ))

        # ─────────────────────────────────────────────────────────
        # VARIABILITY
        # ─────────────────────────────────────────────────────────
        variability = rule_outputs.get("variability")
        if variability:
            var_value = self._get_attr(variability, "value", 0)
            var_category = self._get_attr(variability, "category", None)
            cat_name = var_category.name if hasattr(var_category, "name") else str(var_category)

            if cat_name == "ABSENT":
                explanations.append(RuleExplanation(
                    rule_name="variability",
                    contribution=0.6,
                    description=(
                        f"Absent variability: {var_value:.1f} bpm "
                        f"(threshold: \u22642 bpm) - SEVERE (indicates fetal compromise)"
                    ),
                    time_region=TimeRegion(
                        start_index=-240,  # Last 60 seconds at 4Hz
                        end_index=-1,
                        color="red"
                    ),
                    severity="HIGH"
                ))
            elif cat_name == "MINIMAL":
                explanations.append(RuleExplanation(
                    rule_name="variability",
                    contribution=0.3,
                    description=(
                        f"Minimal variability: {var_value:.1f} bpm "
                        f"(normal: 6-25 bpm) - concerning"
                    ),
                    time_region=TimeRegion(
                        start_index=-240,
                        end_index=-1,
                        color="orange"
                    ),
                    severity="MEDIUM"
                ))
            elif cat_name == "MODERATE":
                explanations.append(RuleExplanation(
                    rule_name="variability",
                    contribution=-0.2,  # Negative = mitigating
                    description=f"Normal variability: {var_value:.1f} bpm (reassuring)",
                    time_region=None,
                    severity="NONE"
                ))
            elif cat_name == "MARKED":
                explanations.append(RuleExplanation(
                    rule_name="variability",
                    contribution=0.15,
                    description=(
                        f"Marked variability: {var_value:.1f} bpm "
                        f"(elevated, may indicate cord compression)"
                    ),
                    time_region=TimeRegion(
                        start_index=-240,
                        end_index=-1,
                        color="yellow"
                    ),
                    severity="LOW"
                ))

        # ─────────────────────────────────────────────────────────
        # DECELERATIONS
        # ─────────────────────────────────────────────────────────
        decelerations = rule_outputs.get("decelerations", [])

        # Count by type
        late_decels = []
        variable_decels = []
        early_decels = []
        prolonged_decels = []

        for d in decelerations:
            decel_type = self._get_attr(d, "decel_type", None)
            type_name = decel_type.name if hasattr(decel_type, "name") else str(decel_type)

            if type_name == "LATE":
                late_decels.append(d)
            elif type_name == "VARIABLE":
                variable_decels.append(d)
            elif type_name == "EARLY":
                early_decels.append(d)
            elif type_name == "PROLONGED":
                prolonged_decels.append(d)

        # Late decelerations (most concerning)
        if late_decels:
            most_recent = late_decels[-1]
            nadir = self._get_attr(most_recent, "nadir_value",
                                   self._get_attr(most_recent, "depth", 0))
            lag = self._get_attr(most_recent, "lag_seconds", 0)
            start_idx = self._get_attr(most_recent, "start_idx", -120)
            end_idx = self._get_attr(most_recent, "end_idx", -1)

            contribution = 0.4 + 0.1 * min(len(late_decels), 5)
            explanations.append(RuleExplanation(
                rule_name="late_decel",
                contribution=contribution,
                description=(
                    f"Late deceleration: FHR dropped to {nadir:.0f} bpm, "
                    f"nadir {lag:.0f}s after contraction peak "
                    f"({len(late_decels)} total) - indicates uteroplacental insufficiency"
                ),
                time_region=TimeRegion(
                    start_index=start_idx,
                    end_index=end_idx,
                    color="red"
                ),
                severity="HIGH"
            ))

        # Variable decelerations
        if variable_decels:
            most_recent = variable_decels[-1]
            depth = self._get_attr(most_recent, "depth", 0)
            duration = self._get_attr(most_recent, "duration_seconds", 0)
            start_idx = self._get_attr(most_recent, "start_idx", -120)
            end_idx = self._get_attr(most_recent, "end_idx", -1)
            has_severity = self._get_attr(most_recent, "has_severity_signs", False)

            contribution = 0.2 + 0.05 * min(len(variable_decels), 5)
            if has_severity:
                contribution += 0.15

            severity_text = " with severity signs" if has_severity else ""
            explanations.append(RuleExplanation(
                rule_name="variable_decel",
                contribution=contribution,
                description=(
                    f"Variable deceleration: depth {depth:.0f} bpm, "
                    f"duration {duration:.0f}s{severity_text} "
                    f"({len(variable_decels)} total) - indicates cord compression"
                ),
                time_region=TimeRegion(
                    start_index=start_idx,
                    end_index=end_idx,
                    color="orange"
                ),
                severity="HIGH" if has_severity else "MEDIUM"
            ))

        # Prolonged decelerations
        if prolonged_decels:
            most_recent = prolonged_decels[-1]
            depth = self._get_attr(most_recent, "depth", 0)
            duration = self._get_attr(most_recent, "duration_seconds", 0)

            explanations.append(RuleExplanation(
                rule_name="prolonged_decel",
                contribution=0.5,
                description=(
                    f"Prolonged deceleration: depth {depth:.0f} bpm, "
                    f"duration {duration:.0f}s - concerning"
                ),
                time_region=None,
                severity="HIGH"
            ))

        # Early decelerations (benign)
        if early_decels and not late_decels:
            explanations.append(RuleExplanation(
                rule_name="early_decel",
                contribution=-0.05,  # Slightly mitigating
                description=(
                    f"Early decelerations ({len(early_decels)} total) - "
                    f"benign, indicates head compression"
                ),
                time_region=None,
                severity="NONE"
            ))

        # ─────────────────────────────────────────────────────────
        # BASELINE
        # ─────────────────────────────────────────────────────────
        baseline = rule_outputs.get("baseline")
        if baseline:
            baseline_value = self._get_attr(baseline, "value", 140)
            is_bradycardia = self._get_attr(baseline, "is_bradycardia", False)
            is_tachycardia = self._get_attr(baseline, "is_tachycardia", False)
            is_normal = self._get_attr(baseline, "is_normal", True)

            if is_bradycardia:
                explanations.append(RuleExplanation(
                    rule_name="baseline",
                    contribution=0.3,
                    description=(
                        f"Bradycardic baseline: {baseline_value:.0f} bpm "
                        f"(normal: 110-160) - concerning"
                    ),
                    time_region=None,
                    severity="MEDIUM"
                ))
            elif is_tachycardia:
                explanations.append(RuleExplanation(
                    rule_name="baseline",
                    contribution=0.2,
                    description=(
                        f"Tachycardic baseline: {baseline_value:.0f} bpm "
                        f"(normal: 110-160) - may indicate infection or distress"
                    ),
                    time_region=None,
                    severity="MEDIUM"
                ))
            elif is_normal:
                explanations.append(RuleExplanation(
                    rule_name="baseline",
                    contribution=-0.1,
                    description=f"Normal baseline: {baseline_value:.0f} bpm (reassuring)",
                    time_region=None,
                    severity="NONE"
                ))

        # ─────────────────────────────────────────────────────────
        # TACHYSYSTOLE
        # ─────────────────────────────────────────────────────────
        tachysystole = rule_outputs.get("tachysystole")
        if tachysystole and self._get_attr(tachysystole, "detected", False):
            rate = self._get_attr(tachysystole, "contractions_per_10min", 0)
            explanations.append(RuleExplanation(
                rule_name="tachysystole",
                contribution=0.25,
                description=(
                    f"Tachysystole: {rate:.1f} contractions/10min "
                    f"(threshold: >5) - excessive uterine activity"
                ),
                time_region=None,
                severity="MEDIUM"
            ))

        # ─────────────────────────────────────────────────────────
        # ACCELERATIONS (Mitigating)
        # ─────────────────────────────────────────────────────────
        accelerations = rule_outputs.get("accelerations", [])
        if accelerations:
            most_recent = accelerations[-1] if accelerations else None
            start_idx = self._get_attr(most_recent, "start_idx", -60) if most_recent else -60
            end_idx = self._get_attr(most_recent, "end_idx", -1) if most_recent else -1

            explanations.append(RuleExplanation(
                rule_name="accelerations",
                contribution=-0.15 * min(len(accelerations), 3),  # Negative = good
                description=(
                    f"Accelerations present: {len(accelerations)} in analysis window "
                    f"(reassuring sign of fetal well-being)"
                ),
                time_region=TimeRegion(
                    start_index=start_idx,
                    end_index=end_idx,
                    color="green"
                ),
                severity="NONE"
            ))

        # Sort by absolute contribution (most impactful first)
        explanations.sort(key=lambda e: abs(e.contribution), reverse=True)

        return explanations

    def to_contributors(
        self,
        explanations: List[RuleExplanation]
    ) -> List[Contributor]:
        """
        Convert RuleExplanations to Contributor objects.

        Args:
            explanations: List of rule explanations.

        Returns:
            List of Contributor objects for the ExplanationResult.
        """
        contributors = []
        for exp in explanations:
            contributors.append(Contributor(
                source=ContributorSource.RULE,
                name=exp.rule_name,
                contribution=exp.contribution,
                description=exp.description,
                time_region=exp.time_region,
                is_mitigating=(exp.contribution < 0)
            ))
        return contributors

    def _get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Safely get attribute from object or dict."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)
