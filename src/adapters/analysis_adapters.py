"""
Adapters for analysis components.
These wrap existing analysis functions to conform to Protocol interfaces.
"""

from typing import List

from src.interfaces.protocols import (
    IMedicalOverride,
    IOverrideResult,
    IAlertGenerator,
    IAlert,
    IBaselineResult,
    IVariabilityResult,
    IDeceleration,
    ITachysystoleResult,
    ISinusoidalResult,
)

from src.analysis.override import apply_medical_override
from src.analysis.alerts import generate_alert


class OverrideAdapter(IMedicalOverride):
    """
    Adapter for medical override logic.
    
    Wraps apply_medical_override to conform to IMedicalOverride interface.
    
    The medical override is a safety net that can force a category
    classification based on critical clinical findings, regardless
    of the ML model's prediction.
    
    Example:
        >>> adapter = OverrideAdapter()
        >>> result = adapter.apply(
        ...     ml_prediction=0,  # ML says Normal
        ...     baseline=baseline_result,
        ...     variability=var_result,
        ...     decelerations=decels,
        ...     tachysystole=tachy_result,
        ...     sinusoidal=sinus_result
        ... )
        >>> if result.should_override:
        ...     print(f"Override to Category {result.final_category + 1}")
    """
    
    def apply(
        self,
        ml_prediction: int,
        baseline: IBaselineResult,
        variability: IVariabilityResult,
        decelerations: List[IDeceleration],
        tachysystole: ITachysystoleResult,
        sinusoidal: ISinusoidalResult
    ) -> IOverrideResult:
        """Apply medical override using existing implementation."""
        return apply_medical_override(
            ml_prediction=ml_prediction,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal
        )


class AlertAdapter(IAlertGenerator):
    """
    Adapter for alert generation.
    
    Wraps generate_alert to conform to IAlertGenerator interface.
    
    Generates explanatory alerts in Hebrew for CTG monitoring,
    following the Israeli Position Paper terminology.
    
    Example:
        >>> adapter = AlertAdapter()
        >>> alert = adapter.generate(
        ...     category=3,
        ...     confidence=0.95,
        ...     baseline=baseline_result,
        ...     variability=var_result,
        ...     decelerations=decels,
        ...     tachysystole=tachy_result,
        ...     sinusoidal=sinus_result
        ... )
        >>> print(alert.headline)
    """
    
    def generate(
        self,
        category: int,
        confidence: float,
        baseline: IBaselineResult,
        variability: IVariabilityResult,
        decelerations: List[IDeceleration],
        tachysystole: ITachysystoleResult,
        sinusoidal: ISinusoidalResult
    ) -> IAlert:
        """Generate alert using existing implementation."""
        return generate_alert(
            category=category,
            confidence=confidence,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal
        )
