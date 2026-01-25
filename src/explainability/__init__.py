"""
Explainability Module.

Provides human-readable explanations for CTG classification decisions.

Features:
    - Rule-based explanations (deterministic, always available)
    - SHAP-based ML explanations (on-demand)
    - Visual highlighting for CTG graph regions

Usage:
    from src.explainability import ExplanationEngine, ExplanationResult

    engine = ExplanationEngine(xgboost_model=classifier.model)
    result = engine.explain(
        category=final_category,
        rule_outputs={"baseline": baseline, "variability": variability, ...},
        ml_features=feature_vector
    )

    print(result.summary)
    for contrib in result.top_contributors:
        print(f"  {contrib.name}: {contrib.description}")

References:
    - SentinelFetal V2.0 PRD, Section: Explainability Module
"""

from src.explainability.models import (
    ContributorSource,
    TimeRegion,
    Contributor,
    HighlightRegion,
    RuleExplanation,
    SHAPExplanation,
    ExplanationResult,
)
from src.explainability.rule_explainer import RuleExplainer
from src.explainability.visual_mapper import VisualMapper
from src.explainability.explanation_engine import ExplanationEngine

# SHAP explainer is optional - may not be installed
try:
    from src.explainability.shap_explainer import SHAPExplainer
except ImportError:
    SHAPExplainer = None

__all__ = [
    "ExplanationEngine",
    "RuleExplainer",
    "VisualMapper",
    "SHAPExplainer",
    "ContributorSource",
    "TimeRegion",
    "Contributor",
    "HighlightRegion",
    "RuleExplanation",
    "SHAPExplanation",
    "ExplanationResult",
]
