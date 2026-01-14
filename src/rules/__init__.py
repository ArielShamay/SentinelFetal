"""
Rule Engine Module for SentinelFetal Gen3.5.

This package implements the medical algorithms based on the Israeli Position Paper
for CTG (Cardiotocography) interpretation.

Modules:
    - baseline: Calculate baseline FHR (קצב בסיסי)
    - variability: Analyze FHR variability (שונות)
    - decelerations: Detect and classify decelerations (האטות)
    - tachysystole: Detect excessive uterine activity (Tachysystole)
    - sinusoidal: Detect sinusoidal pattern (תבנית סינוסואידלית)

The Rule Engine provides deterministic, explainable outputs that form the
"medical brain" component of the Hybrid AI architecture.

Example:
    >>> from src.rules import (
    ...     calculate_baseline,
    ...     calculate_variability,
    ...     detect_decelerations,
    ...     detect_tachysystole,
    ...     detect_sinusoidal_pattern
    ... )
    >>> baseline = calculate_baseline(fhr, sampling_rate=4.0)
    >>> variability = calculate_variability(fhr, sampling_rate=4.0)
"""

from .baseline import calculate_baseline, BaselineResult
from .variability import (
    calculate_variability,
    VariabilityCategory,
    VariabilityResult
)
from .decelerations import (
    detect_decelerations,
    classify_deceleration,
    Deceleration,
    DecelerationType
)
from .tachysystole import detect_tachysystole, TachysystoleResult
from .sinusoidal import detect_sinusoidal_pattern, SinusoidalResult

__all__ = [
    # Baseline
    "calculate_baseline",
    "BaselineResult",
    # Variability
    "calculate_variability",
    "VariabilityCategory",
    "VariabilityResult",
    # Decelerations
    "detect_decelerations",
    "classify_deceleration",
    "Deceleration",
    "DecelerationType",
    # Tachysystole
    "detect_tachysystole",
    "TachysystoleResult",
    # Sinusoidal
    "detect_sinusoidal_pattern",
    "SinusoidalResult",
]
