"""
Medical Override Logic for SentinelFetal.

This module implements the "Safety Net" - medical override rules that
can force a category classification based on critical clinical findings,
regardless of the ML model's prediction.

These rules are based on Section 7 of the Israeli Position Paper and
ensure patient safety by catching high-risk patterns that MUST trigger
immediate clinical attention.

CRITICAL SAFETY RULES (Override ML):
1. Sinusoidal pattern detected → Force Category 3 (Pathological)
2. Absent variability + Recurrent late/variable decels OR bradycardia → Force Category 3
3. If ML predicts Normal BUT variability is Absent → Force Category 2 (Safety Floor)

The philosophy: The ML model can UPGRADE a classification to more severe,
but critical clinical findings should NEVER be downgraded by ML predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List

from src.rules.baseline import BaselineResult
from src.rules.variability import VariabilityResult, VariabilityCategory
from src.rules.decelerations import Deceleration, DecelerationType
from src.rules.tachysystole import TachysystoleResult
from src.rules.sinusoidal import SinusoidalResult


logger = logging.getLogger(__name__)


class OverrideReason(Enum):
    """Reasons for medical override."""
    
    NONE = auto()
    SINUSOIDAL_PATTERN = auto()
    ABSENT_VARIABILITY_WITH_DECELS = auto()
    BRADYCARDIA = auto()
    RECURRENT_LATE_DECELS = auto()
    ABSENT_VARIABILITY_SAFETY_FLOOR = auto()


@dataclass
class MedicalOverride:
    """
    Result of medical override evaluation.
    
    Attributes:
        should_override: Whether the ML prediction should be overridden.
        final_category: The final category after override (0, 1, or 2).
        reason: The reason for override (if any).
        ml_prediction: The original ML model prediction.
        explanation: Human-readable explanation of the override.
    """
    
    should_override: bool
    final_category: int
    reason: OverrideReason
    ml_prediction: int
    explanation: str
    
    @property
    def category_name(self) -> str:
        """Get the category name."""
        names = {0: 'Normal (Category 1)', 1: 'Intermediate (Category 2)', 2: 'Pathological (Category 3)'}
        return names.get(self.final_category, 'Unknown')


def _has_recurrent_late_decels(decelerations: List[Deceleration]) -> bool:
    """
    Check for recurrent late decelerations.
    
    Recurrent = appearing in ≥50% of contractions over a 20-minute period.
    Simplified: ≥3 late decelerations in the list.
    
    Args:
        decelerations: List of detected decelerations.
        
    Returns:
        True if recurrent late decelerations are present.
    """
    late_count = sum(1 for d in decelerations 
                     if d.decel_type == DecelerationType.LATE or 
                     (isinstance(d.decel_type, str) and d.decel_type.lower() == 'late'))
    return late_count >= 3


def _has_recurrent_variable_decels(decelerations: List[Deceleration]) -> bool:
    """
    Check for recurrent variable decelerations.
    
    Recurrent = appearing in ≥50% of contractions over a 20-minute period.
    Simplified: ≥3 variable decelerations in the list.
    
    Args:
        decelerations: List of detected decelerations.
        
    Returns:
        True if recurrent variable decelerations are present.
    """
    variable_count = sum(1 for d in decelerations 
                         if d.decel_type == DecelerationType.VARIABLE or
                         (isinstance(d.decel_type, str) and d.decel_type.lower() == 'variable'))
    return variable_count >= 3


def _detect_bradycardia(baseline: BaselineResult) -> bool:
    """
    Detect bradycardia from baseline analysis.
    
    Bradycardia: FHR < 110 bpm for ≥10 minutes.
    
    Args:
        baseline: Baseline analysis result.
        
    Returns:
        True if bradycardia is present.
    """
    # Check if baseline is abnormally low (< 110 bpm)
    if baseline.value is not None and baseline.value < 110:
        return True
    return False


def apply_medical_override(
    ml_prediction: int,
    baseline: BaselineResult,
    variability: VariabilityResult,
    decelerations: List[Deceleration],
    tachysystole: TachysystoleResult,
    sinusoidal: SinusoidalResult
) -> MedicalOverride:
    """
    Apply medical override rules to ML prediction.
    
    This function implements the safety net logic from Section 7 of the
    Israeli Position Paper. It can UPGRADE a classification to more severe
    but will never DOWNGRADE a critical finding.
    
    HARD OVERRIDE RULES (Force Category 3):
    1. Sinusoidal pattern detected → Force Category 3
    2. Absent variability + (recurrent late OR recurrent variable OR bradycardia) → Force Category 3
    
    SAFETY FLOOR RULE (Force Category 2):
    3. If ML predicts Category 1 (Normal) BUT variability is Absent → Force Category 2
    
    Args:
        ml_prediction: The ML model's prediction (0, 1, or 2).
        baseline: Baseline analysis result.
        variability: Variability analysis result.
        decelerations: List of detected decelerations.
        tachysystole: Tachysystole detection result.
        sinusoidal: Sinusoidal pattern detection result.
        
    Returns:
        MedicalOverride with final category and explanation.
        
    Example:
        >>> override = apply_medical_override(
        ...     ml_prediction=0,  # ML says Normal
        ...     baseline=baseline_result,
        ...     variability=VariabilityResult(category='Absent', ...),
        ...     decelerations=decels,
        ...     tachysystole=tachy_result,
        ...     sinusoidal=SinusoidalResult(detected=True, ...)
        ... )
        >>> print(override.final_category)  # 2 (Pathological) due to sinusoidal
    """
    
    # ==========================================================================
    # RULE 1: Sinusoidal Pattern → Force Category 3 (Pathological)
    # ==========================================================================
    # Sinusoidal pattern is a critical finding that indicates severe fetal
    # compromise (often fetal anemia). MUST be Category 3 regardless of ML.
    
    if sinusoidal.detected:
        logger.warning(
            "MEDICAL OVERRIDE: Sinusoidal pattern detected → Category 3 (Pathological)"
        )
        return MedicalOverride(
            should_override=True,
            final_category=2,
            reason=OverrideReason.SINUSOIDAL_PATTERN,
            ml_prediction=ml_prediction,
            explanation=(
                "Sinusoidal pattern detected. This is a critical finding indicating "
                "potential severe fetal anemia or compromise. Classification forced "
                "to Category 3 (Pathological) regardless of ML prediction."
            )
        )

    # Bradycardia alone warrants elevation to Category 2 for safety.
    if _detect_bradycardia(baseline):
        logger.warning("MEDICAL OVERRIDE: Bradycardia detected → Category 2")
        return MedicalOverride(
            should_override=True,
            final_category=1,
            reason=OverrideReason.BRADYCARDIA,
            ml_prediction=ml_prediction,
            explanation=(
                "Baseline FHR < 110 bpm for ≥10 minutes consistent with bradycardia. "
                "Classification elevated to Category 2."
            )
        )

    # Recurrent late decelerations elevate to Category 2 even without absent variability.
    if _has_recurrent_late_decels(decelerations):
        logger.warning("MEDICAL OVERRIDE: Recurrent late decelerations → Category 2")
        return MedicalOverride(
            should_override=True,
            final_category=1,
            reason=OverrideReason.RECURRENT_LATE_DECELS,
            ml_prediction=ml_prediction,
            explanation=(
                "Recurrent late decelerations detected (≥3 events). "
                "Classification elevated to Category 2 to reflect increased risk."
            )
        )
    
    # ==========================================================================
    # RULE 2: Absent Variability + Ominous Signs → Force Category 3
    # ==========================================================================
    # Absent variability combined with recurrent decelerations or bradycardia
    # is highly predictive of fetal acidemia. MUST be Category 3.
    
    is_absent_variability = (
        variability.category == VariabilityCategory.ABSENT or 
        (isinstance(variability.category, str) and variability.category.lower() == 'absent')
    )
    
    if is_absent_variability:
        recurrent_late = _has_recurrent_late_decels(decelerations)
        recurrent_variable = _has_recurrent_variable_decels(decelerations)
        bradycardia = _detect_bradycardia(baseline)
        
        if recurrent_late or recurrent_variable or bradycardia:
            finding = []
            if recurrent_late:
                finding.append("recurrent late decelerations")
            if recurrent_variable:
                finding.append("recurrent variable decelerations")
            if bradycardia:
                finding.append("bradycardia")
            
            finding_str = " and ".join(finding)
            
            logger.warning(
                f"MEDICAL OVERRIDE: Absent variability + {finding_str} → Category 3"
            )
            return MedicalOverride(
                should_override=True,
                final_category=2,
                reason=OverrideReason.ABSENT_VARIABILITY_WITH_DECELS,
                ml_prediction=ml_prediction,
                explanation=(
                    f"Absent FHR variability with {finding_str}. This combination "
                    "is highly predictive of fetal acidemia. Classification forced "
                    "to Category 3 (Pathological) regardless of ML prediction."
                )
            )
    
    # ==========================================================================
    # RULE 3: Safety Floor - Absent Variability → At Least Category 2
    # ==========================================================================
    # If variability is absent, we should NEVER classify as Normal (Cat 1),
    # even if ML predicts it. Minimum classification is Category 2.
    
    if is_absent_variability and ml_prediction == 0:
        logger.warning(
            "MEDICAL OVERRIDE: Safety floor - Absent variability with Normal prediction "
            "→ Upgraded to Category 2 (Intermediate)"
        )
        return MedicalOverride(
            should_override=True,
            final_category=1,
            reason=OverrideReason.ABSENT_VARIABILITY_SAFETY_FLOOR,
            ml_prediction=ml_prediction,
            explanation=(
                "Safety floor activated: ML predicted Normal (Category 1) but "
                "FHR variability is Absent. Absent variability requires closer "
                "monitoring. Classification upgraded to Category 2 (Intermediate)."
            )
        )
    
    # ==========================================================================
    # NO OVERRIDE - Use ML Prediction
    # ==========================================================================
    
    return MedicalOverride(
        should_override=False,
        final_category=ml_prediction,
        reason=OverrideReason.NONE,
        ml_prediction=ml_prediction,
        explanation=(
            f"No medical override triggered. Using ML prediction: "
            f"Category {ml_prediction + 1}."
        )
    )


def get_final_classification(
    ml_prediction: int,
    baseline: BaselineResult,
    variability: VariabilityResult,
    decelerations: List[Deceleration],
    tachysystole: TachysystoleResult,
    sinusoidal: SinusoidalResult
) -> tuple[int, str]:
    """
    Get the final classification with explanation.
    
    Convenience function that returns just the category and explanation.
    
    Args:
        ml_prediction: The ML model's prediction (0, 1, or 2).
        baseline: Baseline analysis result.
        variability: Variability analysis result.
        decelerations: List of detected decelerations.
        tachysystole: Tachysystole detection result.
        sinusoidal: Sinusoidal pattern detection result.
        
    Returns:
        Tuple of (final_category, explanation).
    """
    override = apply_medical_override(
        ml_prediction=ml_prediction,
        baseline=baseline,
        variability=variability,
        decelerations=decelerations,
        tachysystole=tachysystole,
        sinusoidal=sinusoidal
    )
    return override.final_category, override.explanation


# =============================================================================
# Stage 5: Rule Score Calculation
# =============================================================================

@dataclass
class RuleScoreResult:
    """
    Result of rule score calculation for Stage 5 Tiering.
    
    Attributes:
        score: Aggregate rule score (0.0 - 1.0).
        is_severe: Whether any critical rule was triggered (Tier-3 eligible).
        rule_hits: List of rule names that were triggered.
        reason_codes: List of reason codes for explainability.
    """
    score: float
    is_severe: bool
    rule_hits: List[str]
    reason_codes: List[str]


def calculate_rule_score(
    baseline: BaselineResult,
    variability: VariabilityResult,
    decelerations: List[Deceleration],
    tachysystole: TachysystoleResult,
    sinusoidal: SinusoidalResult
) -> RuleScoreResult:
    """
    Calculate aggregate rule score for Stage 5 Tiering.
    
    This function computes a 0-1 score based on clinical findings.
    Higher scores indicate more concerning patterns.
    
    Scoring weights (based on clinical severity):
    - Sinusoidal pattern: 1.0 (maximum - always severe)
    - Absent variability: 0.4
    - Minimal variability: 0.15
    - Late decelerations: 0.3 per occurrence (max 0.6)
    - Variable decelerations: 0.2 per occurrence (max 0.4)
    - Prolonged decelerations: 0.25 per occurrence
    - Bradycardia: 0.35
    - Tachycardia: 0.2
    - Tachysystole: 0.2
    
    Args:
        baseline: Baseline analysis result.
        variability: Variability analysis result.
        decelerations: List of detected decelerations.
        tachysystole: Tachysystole detection result.
        sinusoidal: Sinusoidal pattern detection result.
        
    Returns:
        RuleScoreResult with aggregate score and details.
    """
    score = 0.0
    is_severe = False
    rule_hits: List[str] = []
    reason_codes: List[str] = []
    
    # --- Sinusoidal Pattern (Critical - Tier 3) ---
    if sinusoidal.detected:
        score = 1.0  # Maximum score
        is_severe = True
        rule_hits.append("SINUSOIDAL")
        reason_codes.append("RULE_SINUSOIDAL_PATTERN")
        # Return immediately - this is the most severe finding
        return RuleScoreResult(
            score=score,
            is_severe=is_severe,
            rule_hits=rule_hits,
            reason_codes=reason_codes
        )
    
    # --- Variability ---
    is_absent = (
        variability.category == VariabilityCategory.ABSENT or
        (isinstance(variability.category, str) and variability.category.lower() == 'absent')
    )
    is_minimal = (
        variability.category == VariabilityCategory.MINIMAL or
        (isinstance(variability.category, str) and variability.category.lower() == 'minimal')
    )
    
    if is_absent:
        score += 0.4
        rule_hits.append("ABSENT_VARIABILITY")
        reason_codes.append("RULE_ABSENT_VARIABILITY")
    elif is_minimal:
        score += 0.15
        rule_hits.append("MINIMAL_VARIABILITY")
        reason_codes.append("RULE_MINIMAL_VARIABILITY")
    
    # --- Decelerations ---
    late_count = sum(1 for d in decelerations 
                     if d.decel_type == DecelerationType.LATE or
                     (isinstance(d.decel_type, str) and d.decel_type.lower() == 'late'))
    variable_count = sum(1 for d in decelerations 
                         if d.decel_type == DecelerationType.VARIABLE or
                         (isinstance(d.decel_type, str) and d.decel_type.lower() == 'variable'))
    prolonged_count = sum(1 for d in decelerations 
                          if d.decel_type == DecelerationType.PROLONGED or
                          (isinstance(d.decel_type, str) and d.decel_type.lower() == 'prolonged'))
    
    if late_count > 0:
        score += min(late_count * 0.3, 0.6)  # Cap at 0.6
        rule_hits.append(f"LATE_DECELS_{late_count}")
        reason_codes.append("RULE_LATE_DECELERATION")
    
    if variable_count > 0:
        score += min(variable_count * 0.2, 0.4)  # Cap at 0.4
        rule_hits.append(f"VARIABLE_DECELS_{variable_count}")
        reason_codes.append("RULE_VARIABLE_DECELERATION")
    
    if prolonged_count > 0:
        score += prolonged_count * 0.25
        rule_hits.append(f"PROLONGED_DECELS_{prolonged_count}")
        reason_codes.append("RULE_PROLONGED_DECELERATION")
    
    # --- Baseline Abnormalities ---
    if baseline.is_bradycardia:
        score += 0.35
        rule_hits.append("BRADYCARDIA")
        reason_codes.append("RULE_BRADYCARDIA")
    
    if baseline.is_tachycardia:
        score += 0.2
        rule_hits.append("TACHYCARDIA")
        reason_codes.append("RULE_TACHYCARDIA")
    
    # --- Tachysystole ---
    if tachysystole.detected:
        score += 0.2
        rule_hits.append("TACHYSYSTOLE")
        reason_codes.append("RULE_TACHYSYSTOLE")
    
    # --- Check for Severe Combination (Tier-3 eligible) ---
    # Absent variability + any concerning finding = severe
    if is_absent and (late_count >= 3 or variable_count >= 3 or baseline.is_bradycardia):
        is_severe = True
        reason_codes.append("COMBINATION_ABSENT_VAR_WITH_DECELS_OR_BRADY")
    
    # Cap score at 1.0
    score = min(score, 1.0)
    
    return RuleScoreResult(
        score=round(score, 3),
        is_severe=is_severe,
        rule_hits=rule_hits,
        reason_codes=reason_codes
    )

