"""
Feature Fusion Module.

Implements the fusion of MOMENT embeddings with Rule Engine features
to create the complete 1035-dimensional feature vector for classification.

Feature Vector Structure (1035 dimensions):
    Indices 0-1023:    MOMENT Embeddings (1024 dims)
    Index 1024:        Baseline (normalized / 160)
    Index 1025:        Variability Value (normalized / 25)
    Indices 1026-1029: Variability Category (One-Hot: Absent, Minimal, Moderate, Marked)
    Index 1030:        Late Decel Count (normalized / 10)
    Index 1031:        Variable Decel Count (normalized / 10)
    Index 1032:        Recurrent Decels Flag (0 or 1)
    Index 1033:        Tachysystole Flag (0 or 1)
    Index 1034:        Sinusoidal Flag (0 or 1)

This feature vector is the input to the final classifier (XGBoost or MLP)
that outputs the Category 1/2/3 prediction.

References:
    - SentinelFetal Gen3.5 Technical Specification, Section 6.1-6.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from ..rules import (
    BaselineResult,
    VariabilityResult,
    VariabilityCategory,
    Deceleration,
    DecelerationType,
    TachysystoleResult,
    SinusoidalResult,
)

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
FEATURE_VECTOR_DIM = 1035
EMBEDDING_DIM = 1024

# Feature indices (for reference and debugging)
FEATURE_INDICES = {
    'embedding': (0, 1024),           # 0-1023
    'baseline': 1024,                  # 1024
    'variability_value': 1025,         # 1025
    'variability_onehot': (1026, 1030), # 1026-1029
    'late_decel_count': 1030,          # 1030
    'variable_decel_count': 1031,      # 1031
    'recurrent_decels': 1032,          # 1032
    'tachysystole': 1033,              # 1033
    'sinusoidal': 1034,                # 1034
}


@dataclass
class FeatureVector:
    """
    Container for a fused feature vector with metadata.
    
    Attributes:
        vector: The 1035-dimensional feature vector.
        start_idx: Start index in the original signal (from embedding window).
        end_idx: End index in the original signal.
        start_time_sec: Start time in seconds.
        end_time_sec: End time in seconds.
        
        # Component values (for interpretability)
        baseline: Baseline FHR value.
        variability_value: Variability in bpm.
        variability_category: Variability category (Absent/Minimal/Moderate/Marked).
        late_decel_count: Number of late decelerations.
        variable_decel_count: Number of variable decelerations.
        recurrent_decels: Whether decelerations are recurrent.
        tachysystole: Whether tachysystole is present.
        sinusoidal: Whether sinusoidal pattern is detected.
    """
    
    vector: np.ndarray
    start_idx: int = 0
    end_idx: int = 0
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    
    # Component values for interpretability
    baseline: float = 0.0
    variability_value: float = 0.0
    variability_category: str = "Unknown"
    late_decel_count: int = 0
    variable_decel_count: int = 0
    recurrent_decels: bool = False
    tachysystole: bool = False
    sinusoidal: bool = False
    
    def __post_init__(self) -> None:
        """Validate vector dimension."""
        if self.vector.shape != (FEATURE_VECTOR_DIM,):
            raise ValueError(
                f"Feature vector must be {FEATURE_VECTOR_DIM}-dimensional, "
                f"got {self.vector.shape}"
            )
    
    def __repr__(self) -> str:
        return (
            f"FeatureVector(shape={self.vector.shape}, "
            f"baseline={self.baseline:.0f}, "
            f"var={self.variability_category}, "
            f"late_decels={self.late_decel_count})"
        )
    
    def get_rule_features(self) -> np.ndarray:
        """
        Extract only the rule-based features (indices 1024-1034).
        
        Returns:
            11-dimensional array of rule features.
        """
        return self.vector[EMBEDDING_DIM:]
    
    def get_embedding(self) -> np.ndarray:
        """
        Extract only the MOMENT embedding (indices 0-1023).
        
        Returns:
            1024-dimensional embedding array.
        """
        return self.vector[:EMBEDDING_DIM]


def build_feature_vector(
    embedding: np.ndarray,
    baseline: Union[float, BaselineResult],
    variability: Union[dict, VariabilityResult],
    decelerations: List[Deceleration],
    tachysystole: Union[dict, TachysystoleResult],
    sinusoidal: Union[dict, SinusoidalResult],
    total_contractions: int = 0,
    start_idx: int = 0,
    end_idx: int = 0,
    start_time_sec: float = 0.0,
    end_time_sec: float = 0.0,
) -> FeatureVector:
    """
    Build the complete 1035-dimensional feature vector.
    
    Combines MOMENT embeddings with rule-based features according to
    the specification in Section 6.2.
    
    Args:
        embedding: 1024-dimensional MOMENT embedding.
        baseline: Baseline FHR value or BaselineResult object.
        variability: Variability info (dict or VariabilityResult).
        decelerations: List of detected decelerations.
        tachysystole: Tachysystole detection result.
        sinusoidal: Sinusoidal pattern detection result.
        total_contractions: Total number of contractions in the window.
        start_idx: Start index of the window.
        end_idx: End index of the window.
        start_time_sec: Start time in seconds.
        end_time_sec: End time in seconds.
        
    Returns:
        FeatureVector containing the 1035-dim vector and metadata.
        
    Raises:
        ValueError: If embedding is not 1024-dimensional.
        
    Example:
        >>> feature_vec = build_feature_vector(
        ...     embedding=moment_embedding,
        ...     baseline=baseline_result,
        ...     variability=var_result,
        ...     decelerations=decel_list,
        ...     tachysystole=tachy_result,
        ...     sinusoidal=sinus_result
        ... )
        >>> print(f"Feature shape: {feature_vec.vector.shape}")  # (1035,)
        
    References:
        SentinelFetal Gen3.5 Technical Specification, Section 6.2
    """
    # Validate embedding
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Embedding must be {EMBEDDING_DIM}-dimensional, got {embedding.shape}"
        )
    
    features: List[float] = []
    
    # =========================================================================
    # 1. MOMENT Embedding (indices 0-1023)
    # =========================================================================
    features.extend(embedding.tolist())
    
    # =========================================================================
    # 2. Baseline (index 1024) - Normalized by 160
    # =========================================================================
    if isinstance(baseline, BaselineResult):
        baseline_value = baseline.value
    else:
        baseline_value = float(baseline)
    
    features.append(baseline_value / 160.0)
    
    # =========================================================================
    # 3. Variability Value (index 1025) - Normalized by 25
    # =========================================================================
    if isinstance(variability, VariabilityResult):
        var_value = variability.value
        var_category = variability.category
    else:
        var_value = variability.get('value', 10.0) or 10.0
        var_cat_str = variability.get('category', 'Moderate')
        var_category = VariabilityCategory[var_cat_str.upper()] if isinstance(var_cat_str, str) else var_cat_str
    
    features.append(var_value / 25.0)
    
    # =========================================================================
    # 4. Variability Category One-Hot (indices 1026-1029)
    # Order: [Absent, Minimal, Moderate, Marked]
    # =========================================================================
    category_order = [
        VariabilityCategory.ABSENT,
        VariabilityCategory.MINIMAL,
        VariabilityCategory.MODERATE,
        VariabilityCategory.MARKED
    ]
    
    for cat in category_order:
        features.append(1.0 if var_category == cat else 0.0)
    
    # =========================================================================
    # 5. Late Decel Count (index 1030) - Normalized by 10
    # =========================================================================
    late_count = sum(
        1 for d in decelerations 
        if d.decel_type == DecelerationType.LATE
    )
    features.append(late_count / 10.0)
    
    # =========================================================================
    # 6. Variable Decel Count (index 1031) - Normalized by 10
    # =========================================================================
    variable_count = sum(
        1 for d in decelerations 
        if d.decel_type == DecelerationType.VARIABLE
    )
    features.append(variable_count / 10.0)
    
    # =========================================================================
    # 7. Recurrent Decels Flag (index 1032)
    # Recurrent if >50% of contractions have associated decelerations
    # =========================================================================
    if total_contractions > 0:
        decel_ratio = len(decelerations) / total_contractions
        recurrent = decel_ratio > 0.5
    else:
        recurrent = False
    features.append(1.0 if recurrent else 0.0)
    
    # =========================================================================
    # 8. Tachysystole Flag (index 1033)
    # =========================================================================
    if isinstance(tachysystole, TachysystoleResult):
        tachy_detected = tachysystole.detected
    else:
        tachy_detected = tachysystole.get('detected', False)
    features.append(1.0 if tachy_detected else 0.0)
    
    # =========================================================================
    # 9. Sinusoidal Flag (index 1034)
    # =========================================================================
    if isinstance(sinusoidal, SinusoidalResult):
        sinus_detected = sinusoidal.detected
    else:
        sinus_detected = sinusoidal.get('detected', False)
    features.append(1.0 if sinus_detected else 0.0)
    
    # =========================================================================
    # Build final vector
    # =========================================================================
    feature_vector = np.array(features, dtype=np.float32)
    
    # Validate dimension
    assert feature_vector.shape == (FEATURE_VECTOR_DIM,), (
        f"Expected {FEATURE_VECTOR_DIM} features, got {len(features)}"
    )
    
    logger.debug(
        f"Built feature vector: baseline={baseline_value:.0f}, "
        f"var={var_category.name}, late_decels={late_count}, "
        f"tachy={tachy_detected}, sinus={sinus_detected}"
    )
    
    return FeatureVector(
        vector=feature_vector,
        start_idx=start_idx,
        end_idx=end_idx,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        baseline=baseline_value,
        variability_value=var_value,
        variability_category=var_category.name,
        late_decel_count=late_count,
        variable_decel_count=variable_count,
        recurrent_decels=recurrent,
        tachysystole=tachy_detected,
        sinusoidal=sinus_detected,
    )


def build_feature_matrix(
    feature_vectors: List[FeatureVector]
) -> np.ndarray:
    """
    Stack multiple feature vectors into a matrix for batch classification.
    
    Args:
        feature_vectors: List of FeatureVector objects.
        
    Returns:
        numpy array of shape (N, 1035) where N is the number of windows.
        
    Example:
        >>> vectors = [fv1, fv2, fv3]
        >>> matrix = build_feature_matrix(vectors)
        >>> print(matrix.shape)  # (3, 1035)
    """
    if not feature_vectors:
        return np.empty((0, FEATURE_VECTOR_DIM), dtype=np.float32)
    
    return np.vstack([fv.vector for fv in feature_vectors])


def get_feature_names() -> List[str]:
    """
    Get human-readable names for all 1035 features.
    
    Returns:
        List of feature names.
    """
    names = []
    
    # MOMENT embeddings
    for i in range(EMBEDDING_DIM):
        names.append(f"moment_emb_{i:04d}")
    
    # Rule-based features
    names.extend([
        "baseline_norm",
        "variability_value_norm",
        "var_cat_absent",
        "var_cat_minimal",
        "var_cat_moderate",
        "var_cat_marked",
        "late_decel_count_norm",
        "variable_decel_count_norm",
        "recurrent_decels_flag",
        "tachysystole_flag",
        "sinusoidal_flag",
    ])
    
    return names
