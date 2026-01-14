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
