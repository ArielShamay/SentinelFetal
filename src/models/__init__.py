"""
SentinelFetal Models Module.

Contains feature extractors and classifiers for CTG analysis.

Feature Extractors (choose one):
    - MiniRocketEncoder: RECOMMENDED - Fast, lightweight (84 kernels, ~1ms inference)
    - MomentFeatureExtractor: Legacy - Heavy (341M params, ~2-5s inference)

Classifier:
    - XGBClassifierWrapper: XGBoost classifier for CTG classification

Usage (RECOMMENDED - MiniRocket):
    >>> from src.models import get_minirocket_encoder
    >>> encoder = get_minirocket_encoder()
    >>> features = encoder.extract_features(fhr_signal)

Usage (Legacy - MOMENT):
    >>> from src.models import get_moment_encoder
    >>> encoder = get_moment_encoder()  # Requires torch + momentfm

Factory Functions:
    - get_minirocket_encoder(): Get MiniRocket encoder (recommended)
    - get_moment_encoder(): Get MOMENT encoder (legacy, requires torch)
"""

# MiniRocket - Recommended (10-20x faster than MOMENT)
try:
    from .minirocket_encoder import (
        MiniRocketEncoder,
        MiniRocketConfig,
        MiniRocketFeatureResult,
        MiniRocketEncoderError,
        get_minirocket_encoder,
        generate_synthetic_training_data,
        SKTIME_AVAILABLE,
    )
    _MINIROCKET_EXPORTS = [
        "MiniRocketEncoder",
        "MiniRocketConfig",
        "MiniRocketFeatureResult",
        "MiniRocketEncoderError",
        "get_minirocket_encoder",
        "generate_synthetic_training_data",
        "SKTIME_AVAILABLE",
    ]
except ImportError:
    _MINIROCKET_EXPORTS = []
    SKTIME_AVAILABLE = False

# MOMENT - Legacy (requires torch + momentfm)
try:
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
    _MOMENT_EXPORTS = [
        "MomentFeatureExtractor",
        "MomentEncoderError",
        "EmbeddingResult",
        "extract_embeddings_sliding_window",
        "get_moment_encoder",
        "get_encoder_info",
        "MOMENT_AVAILABLE",
        "ONNX_BACKEND_AVAILABLE",
    ]
except ImportError:
    _MOMENT_EXPORTS = []
    MOMENT_AVAILABLE = False
    ONNX_BACKEND_AVAILABLE = False

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
    # MiniRocket (RECOMMENDED)
    *_MINIROCKET_EXPORTS,
    # MOMENT (Legacy)
    *_MOMENT_EXPORTS,
    # ONNX Encoder (V2)
    *_ONNX_EXPORTS,
    # Classifier
    "XGBClassifierWrapper",
    "ClassifierConfig",
    "TrainingResult",
    # Fusion
    "build_feature_vector",
    "FeatureVector",
    "FEATURE_VECTOR_DIM",
]
