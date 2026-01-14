"""
SentinelFetal Models Module.

Contains:
    - MomentFeatureExtractor: MOMENT foundation model for embedding extraction
    - CTGClassifier: XGBoost classifier for CTG classification
    - Feature fusion utilities for hybrid classification

Usage:
    >>> from src.models import CTGClassifier, MomentFeatureExtractor
    >>> classifier = CTGClassifier()
    >>> classifier.load("models/xgb_demo.json")
    >>> encoder = MomentFeatureExtractor()
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
from .classifier import (
    XGBClassifierWrapper,
    ClassifierConfig,
    TrainingResult,
)

__all__ = [
    # MOMENT Encoder
    "MomentFeatureExtractor",
    "MomentEncoderError",
    "EmbeddingResult",
    "extract_embeddings_sliding_window",
    "MOMENT_AVAILABLE",
    # Classifier
    "XGBClassifierWrapper",
    "ClassifierConfig",
    "TrainingResult",
    # Fusion
    "build_feature_vector",
    "FeatureVector",
    "FEATURE_VECTOR_DIM",
]
