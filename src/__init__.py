"""
SentinelFetal - Real-time Fetal Distress Detection System.

A hybrid AI system for CTG analysis combining:
- MOMENT foundation model for time series embedding
- XGBoost classifier for category prediction
- Rule-based analysis per Israeli Position Paper
- Medical Override safety net

Architecture:
    This system uses a modular architecture with dependency injection.
    Any component can be replaced by implementing the corresponding Protocol.

Modules:
    config: Centralized configuration constants
    interfaces: Protocol definitions for component abstraction
    adapters: Adapter implementations wrapping existing code
    pipeline: Modular analysis pipeline with DI container
    data: CTU-UHB dataset loading and preprocessing
    rules: Clinical rule implementations (baseline, variability, etc.)
    models: ML models (MOMENT encoder, XGBoost classifier)
    analysis: Alert generation and medical override
    training: Model training utilities
    ui: Streamlit dashboard and visualizations
    utils: Reusable utility functions

Quick Start (Modular Pipeline):
    >>> from src.pipeline import PipelineContainer, AnalysisPipeline
    >>> container = PipelineContainer.create_default()
    >>> pipeline = AnalysisPipeline(container)
    >>> result = pipeline.analyze(fhr, uc)
    
Quick Start (Legacy - still works):
    >>> from src.data import CTUDataLoader, CTGRecord
    >>> from src.rules import calculate_baseline, calculate_variability
    >>> from src.analysis import generate_alert, apply_medical_override
    
Custom Component Example:
    >>> from src.interfaces import IClassifier
    >>> from src.pipeline import PipelineContainer, AnalysisPipeline
    >>> 
    >>> class MyClassifier(IClassifier):
    ...     def predict(self, X): ...
    ...     def predict_proba(self, X): ...
    ...     def save_model(self, path): ...
    ...     def load_model(self, path): ...
    >>> 
    >>> container = PipelineContainer.create_default()
    >>> container.classifier = MyClassifier()
    >>> pipeline = AnalysisPipeline(container)
"""

__version__ = "3.6.0"
__author__ = "Ariel Shamay"

# Expose main configuration
from src.config import CTG, THRESHOLDS, COLORS, PATHS, MODEL

# ============================================================================
# MODULAR PIPELINE (NEW)
# ============================================================================
from src.pipeline import PipelineContainer, AnalysisPipeline, AnalysisResult

# ============================================================================
# INTERFACES (for custom implementations)
# ============================================================================
from src.interfaces import (
    # Data Layer
    IDataLoader,
    ICTGRecord,
    IPreprocessor,
    IPreprocessingResult,
    # Rule Engine
    IBaselineCalculator,
    IBaselineResult,
    IVariabilityCalculator,
    IVariabilityResult,
    IDecelerationDetector,
    IDeceleration,
    ITachysystoleDetector,
    ITachysystoleResult,
    ISinusoidalDetector,
    ISinusoidalResult,
    # Model Layer
    IFeatureExtractor,
    IEmbeddingResult,
    IFeatureFusion,
    IFeatureVector,
    IClassifier,
    # Analysis Layer
    IMedicalOverride,
    IOverrideResult,
    IAlertGenerator,
    IAlert,
    # Pipeline
    IAnalysisPipeline,
    IAnalysisResult,
)

# ============================================================================
# ADAPTERS (default implementations)
# ============================================================================
from src.adapters import (
    PreprocessorAdapter,
    BaselineAdapter,
    VariabilityAdapter,
    DecelerationAdapter,
    TachysystoleAdapter,
    SinusoidalAdapter,
    MomentAdapter,
    ClassifierAdapter,
    FusionAdapter,
    OverrideAdapter,
    AlertAdapter,
    DataLoaderAdapter,
)

# ============================================================================
# LEGACY IMPORTS (backward compatibility)
# ============================================================================
from src.data import CTUDataLoader, CTGRecord, CTGPreprocessor, PreprocessingConfig
from src.rules import (
    calculate_baseline, BaselineResult,
    calculate_variability, VariabilityResult, VariabilityCategory,
    detect_decelerations, Deceleration, DecelerationType,
    detect_tachysystole, TachysystoleResult,
    detect_sinusoidal_pattern, SinusoidalResult
)
from src.models import (
    MomentFeatureExtractor, EmbeddingResult,
    build_feature_vector, FeatureVector,
    XGBClassifierWrapper,
)
from src.analysis import (
    apply_medical_override, MedicalOverride, OverrideReason,
    generate_alert, Alert,
)

# Version info
VERSION_INFO = {
    'version': __version__,
    'phase': 7,
    'status': 'Production Ready - Modular Architecture',
    'model': 'XGBoost + MOMENT',
    'architecture': 'Dependency Injection',
}

__all__ = [
    # Version
    '__version__',
    '__author__',
    'VERSION_INFO',
    
    # Configuration
    'CTG',
    'THRESHOLDS', 
    'COLORS',
    'PATHS',
    'MODEL',
    
    # Modular Pipeline
    'PipelineContainer',
    'AnalysisPipeline',
    'AnalysisResult',
    
    # Interfaces
    'IDataLoader',
    'ICTGRecord',
    'IPreprocessor',
    'IPreprocessingResult',
    'IBaselineCalculator',
    'IBaselineResult',
    'IVariabilityCalculator',
    'IVariabilityResult',
    'IDecelerationDetector',
    'IDeceleration',
    'ITachysystoleDetector',
    'ITachysystoleResult',
    'ISinusoidalDetector',
    'ISinusoidalResult',
    'IFeatureExtractor',
    'IEmbeddingResult',
    'IFeatureFusion',
    'IFeatureVector',
    'IClassifier',
    'IMedicalOverride',
    'IOverrideResult',
    'IAlertGenerator',
    'IAlert',
    'IAnalysisPipeline',
    'IAnalysisResult',
    
    # Adapters
    'PreprocessorAdapter',
    'BaselineAdapter',
    'VariabilityAdapter',
    'DecelerationAdapter',
    'TachysystoleAdapter',
    'SinusoidalAdapter',
    'MomentAdapter',
    'ClassifierAdapter',
    'FusionAdapter',
    'OverrideAdapter',
    'AlertAdapter',
    'DataLoaderAdapter',
    
    # Legacy (backward compatible)
    'CTUDataLoader',
    'CTGRecord',
    'CTGPreprocessor',
    'PreprocessingConfig',
    'calculate_baseline',
    'BaselineResult',
    'calculate_variability',
    'VariabilityResult',
    'VariabilityCategory',
    'detect_decelerations',
    'Deceleration',
    'DecelerationType',
    'detect_tachysystole',
    'TachysystoleResult',
    'detect_sinusoidal_pattern',
    'SinusoidalResult',
    'MomentFeatureExtractor',
    'EmbeddingResult',
    'build_feature_vector',
    'FeatureVector',
    'XGBClassifierWrapper',
    'apply_medical_override',
    'MedicalOverride',
    'OverrideReason',
    'generate_alert',
    'Alert',
]
