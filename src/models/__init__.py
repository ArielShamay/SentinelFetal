"""
SentinelFetal Models Module.

PRODUCTION MODULE - NO MOCK/DEMO FALLBACKS

Contains:
    - MomentFeatureExtractor: MOMENT foundation model for embedding extraction (PyTorch)
    - MomentONNXPredictor: Optimized ONNX/OpenVINO inference (V2)
    - get_moment_encoder: Factory function for best available backend
    - CTGClassifier: XGBoost classifier for CTG classification
    - Feature fusion utilities for hybrid classification

V2 Architecture (STRICT Production):
    Use get_moment_encoder() to automatically select the best backend:
    1. OpenVINO (best for Intel CPUs)
    2. ONNX Runtime (cross-platform)
    3. PyTorch MOMENT (fallback)
    
    NO MOCK MODE - If no backend is available, ModelNotFoundError is raised.

Exceptions:
    - ModelNotFoundError: Raised when no MOMENT model can be loaded
    - InferenceError: Raised when model inference fails
    - MomentEncoderError: Raised when MOMENT encoding fails

Usage:
    >>> from src.models import get_moment_encoder, XGBClassifierWrapper
    >>> encoder = get_moment_encoder()  # Auto-selects best backend (or raises error)
    >>> print(f"Using: {encoder.backend if hasattr(encoder, 'backend') else 'pytorch'}")
    >>> classifier = XGBClassifierWrapper()
    >>> classifier.load("models/sentinel_classifier.json")
"""

from .moment_encoder import (
    MomentFeatureExtractor,
    MomentEncoderError,
    EmbeddingResult,
    extract_embeddings_sliding_window,
    get_moment_encoder,
    get_encoder_info,
    MOMENT_AVAILABLE,
    ONNX_BACKEND_AVAILABLE,
)

# Import ONNX predictor if available
try:
    from .moment_onnx import (
        MomentONNXPredictor,
        MomentONNXConfig,
        MomentOpenVINO,  # Alias for compatibility
        ModelNotFoundError,
        InferenceError,
        convert_onnx_to_openvino,
    )
    _ONNX_EXPORTS = [
        "MomentONNXPredictor",
        "MomentONNXConfig",
        "MomentOpenVINO",
        "ModelNotFoundError",
        "InferenceError",
        "convert_onnx_to_openvino",
    ]
except ImportError:
    _ONNX_EXPORTS = []

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
    # MOMENT Encoder (PyTorch)
    "MomentFeatureExtractor",
    "MomentEncoderError",
    "EmbeddingResult",
    "extract_embeddings_sliding_window",
    "MOMENT_AVAILABLE",
    # ONNX Encoder (V2)
    *_ONNX_EXPORTS,
    "ONNX_BACKEND_AVAILABLE",
    # Factory function (recommended)
    "get_moment_encoder",
    "get_encoder_info",
    # Classifier
    "XGBClassifierWrapper",
    "ClassifierConfig",
    "TrainingResult",
    # Fusion
    "build_feature_vector",
    "FeatureVector",
    "FEATURE_VECTOR_DIM",
]
