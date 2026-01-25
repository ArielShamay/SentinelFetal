"""
SHAP Explainer Module.

Provides SHAP-based explanations for XGBoost predictions.
Maps abstract ML features to clinical categories.

NOTE: This module is OPTIONAL and only runs ON-DEMAND to preserve latency.
SHAP computation can take 50-100ms, which would exceed the latency budget
if run on every prediction.

References:
    - SentinelFetal V2.0 PRD, Section: Explainability Module
"""

from __future__ import annotations

import logging
from typing import List, Optional, Any

import numpy as np

from src.explainability.models import (
    ContributorSource,
    Contributor,
    SHAPExplanation,
)

logger = logging.getLogger(__name__)


# Feature name to clinical category mapping
FEATURE_CATEGORIES = {
    "baseline": "Baseline Analysis",
    "variability": "Variability Analysis",
    "decel": "Deceleration Analysis",
    "accel": "Acceleration Analysis",
    "tachysystole": "Contraction Analysis",
    "sinusoidal": "Pattern Analysis",
    "minirocket": "Pattern Recognition (ML)",
}

# First N features are rule-based (after fusion)
RULE_FEATURE_NAMES = [
    "baseline_value",
    "variability_value",
    "variability_cat_absent",
    "variability_cat_minimal",
    "variability_cat_moderate",
    "variability_cat_marked",
    "decel_late_count",
    "decel_variable_count",
    "decel_recurrent",
    "tachysystole_detected",
    "sinusoidal_detected",
]


class SHAPExplainer:
    """
    Uses SHAP to explain XGBoost predictions.

    Maps abstract feature indices to human-readable clinical categories.

    IMPORTANT: Only call this on-demand (user clicks "Explain ML").
    SHAP computation is slow (~50-100ms) and should not run on every prediction.

    Example:
        >>> explainer = SHAPExplainer(xgboost_model)
        >>> if user_clicked_explain:
        ...     shap_explanations = explainer.explain(feature_vector)
        ...     for exp in shap_explanations[:3]:
        ...         print(f"{exp.feature_name}: {exp.shap_value:.3f}")
    """

    def __init__(self, model: Optional[Any] = None):
        """
        Initialize SHAP explainer.

        Args:
            model: XGBoost model instance (or compatible tree model).
        """
        self.model = model
        self._explainer = None
        self._shap_available = False

        if model is not None:
            self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """
        Initialize SHAP TreeExplainer.

        TreeExplainer is fast for tree-based models like XGBoost.
        Falls back gracefully if SHAP is not installed.
        """
        try:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
            self._shap_available = True
            logger.info("SHAP TreeExplainer initialized successfully")
        except ImportError:
            logger.warning("SHAP library not installed. ML explanations unavailable.")
            self._shap_available = False
        except Exception as e:
            logger.error(f"SHAP initialization failed: {e}")
            self._shap_available = False

    @property
    def is_available(self) -> bool:
        """True if SHAP explanations are available."""
        return self._shap_available and self._explainer is not None

    def explain(
        self,
        features: np.ndarray,
        top_n: int = 10
    ) -> List[SHAPExplanation]:
        """
        Generate SHAP explanations for feature vector.

        Algorithm:
            1. Compute SHAP values for all features
            2. Get top N features by absolute SHAP value
            3. Map feature indices to human-readable names
            4. Generate descriptions

        Args:
            features: Feature vector (1D array, will be reshaped).
            top_n: Number of top features to return.

        Returns:
            List of SHAPExplanation objects, sorted by |SHAP value|.
        """
        if not self.is_available:
            logger.warning("SHAP explainer not available")
            return []

        try:
            # Ensure 2D shape for SHAP
            X = features.reshape(1, -1)

            # Compute SHAP values
            shap_values = self._explainer.shap_values(X)

            # Handle multiclass output
            if isinstance(shap_values, list):
                # Use class 2 (pathological) or class 1 for binary
                if len(shap_values) >= 3:
                    shap_values = shap_values[2]  # Pathological class
                else:
                    shap_values = shap_values[-1]  # Last class

            shap_values = shap_values.flatten()

            # Get top features by absolute value
            top_indices = np.argsort(np.abs(shap_values))[-top_n:][::-1]

            # Build explanations
            explanations = []
            for idx in top_indices:
                feature_name = self._get_feature_name(idx)
                category = self._get_clinical_category(feature_name)
                shap_val = float(shap_values[idx])

                explanations.append(SHAPExplanation(
                    feature_index=int(idx),
                    feature_name=feature_name,
                    shap_value=shap_val,
                    clinical_category=category,
                    description=self._generate_description(feature_name, shap_val)
                ))

            return explanations

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return []

    def to_contributors(
        self,
        explanations: List[SHAPExplanation],
        exclude_covered: bool = True
    ) -> List[Contributor]:
        """
        Convert SHAP explanations to Contributor objects.

        Args:
            explanations: List of SHAP explanations.
            exclude_covered: If True, exclude features that overlap
                           with rule-based explanations.

        Returns:
            List of Contributor objects.
        """
        contributors = []

        for exp in explanations:
            # Skip MiniRocket features if they don't add insight
            if exp.feature_name.startswith("minirocket_") and abs(exp.shap_value) < 0.1:
                continue

            contributors.append(Contributor(
                source=ContributorSource.ML,
                name=exp.feature_name,
                contribution=exp.shap_value,
                description=exp.description,
                time_region=None,  # ML features don't map to time regions
                is_mitigating=(exp.shap_value < 0)
            ))

        return contributors

    def _get_feature_name(self, idx: int) -> str:
        """
        Map feature index to human-readable name.

        Args:
            idx: Feature index in the fused vector.

        Returns:
            Human-readable feature name.
        """
        if idx < len(RULE_FEATURE_NAMES):
            return RULE_FEATURE_NAMES[idx]
        else:
            # MiniRocket features
            return f"minirocket_{idx - len(RULE_FEATURE_NAMES)}"

    def _get_clinical_category(self, feature_name: str) -> str:
        """
        Map feature name to clinical category.

        Args:
            feature_name: Feature name.

        Returns:
            Clinical category string.
        """
        for prefix, category in FEATURE_CATEGORIES.items():
            if feature_name.startswith(prefix):
                return category
        return "Other"

    def _generate_description(self, feature_name: str, shap_value: float) -> str:
        """
        Generate human-readable description for SHAP explanation.

        Args:
            feature_name: Feature name.
            shap_value: SHAP contribution value.

        Returns:
            Description string.
        """
        direction = "increased" if shap_value > 0 else "decreased"
        impact = "pathological" if shap_value > 0 else "normal"

        # Clean up feature name for display
        display_name = feature_name.replace("_", " ").title()

        # Generate appropriate description based on feature type
        if "variability" in feature_name.lower():
            return f"Variability features {direction} pathological likelihood"
        elif "baseline" in feature_name.lower():
            return f"Baseline features {direction} pathological likelihood"
        elif "decel" in feature_name.lower():
            return f"Deceleration features {direction} pathological likelihood"
        elif "minirocket" in feature_name.lower():
            kernel_num = feature_name.split("_")[-1]
            return f"Pattern feature #{kernel_num} contributed to {impact} classification"
        else:
            return f"{display_name} {direction} likelihood of {impact} classification"


def check_shap_available() -> bool:
    """
    Check if SHAP library is installed and available.

    Returns:
        True if SHAP can be imported.
    """
    try:
        import shap
        return True
    except ImportError:
        return False
