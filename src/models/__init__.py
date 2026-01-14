"""
SentinelFetal Models Module.

Contains:
    - MomentFeatureExtractor: MOMENT foundation model for embedding extraction
    - Feature fusion utilities for hybrid classification
"""

from .moment_encoder import (
    MomentFeatureExtractor,
    MomentEncoderError,
    EmbeddingResult,
    extract_embeddings_sliding_window,
    MOMENT_AVAILABLE,
)
from .fusion import (
    build_feature_vector,
    FeatureVector,
    FEATURE_VECTOR_DIM,
)

__all__ = [
    # MOMENT Encoder
    "MomentFeatureExtractor",
    "MomentEncoderError",
    "EmbeddingResult",
    "extract_embeddings_sliding_window",
    "MOMENT_AVAILABLE",
    # Fusion
    "build_feature_vector",
    "FeatureVector",
    "FEATURE_VECTOR_DIM",
]
