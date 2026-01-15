"""
Abstract interfaces (Protocols) for SentinelFetal components.
All concrete implementations must conform to these interfaces.

This module enables dependency injection and component abstraction,
allowing any component to be replaced without affecting the rest of the system.

Usage:
    >>> from src.interfaces import IClassifier, IBaselineCalculator
    >>> class MyCustomClassifier(IClassifier):
    ...     def predict(self, X): ...
    ...     def predict_proba(self, X): ...
"""

from .protocols import (
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

from .types import (
    FHRSignal,
    UCSignal,
    EmbeddingVector,
    FeatureVector,
    CategoryLabel,
)

__all__ = [
    # Data Layer
    'IDataLoader',
    'ICTGRecord',
    'IPreprocessor',
    'IPreprocessingResult',
    
    # Rule Engine
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
    
    # Model Layer
    'IFeatureExtractor',
    'IEmbeddingResult',
    'IFeatureFusion',
    'IFeatureVector',
    'IClassifier',
    
    # Analysis Layer
    'IMedicalOverride',
    'IOverrideResult',
    'IAlertGenerator',
    'IAlert',
    
    # Pipeline
    'IAnalysisPipeline',
    'IAnalysisResult',
    
    # Types
    'FHRSignal',
    'UCSignal',
    'EmbeddingVector',
    'FeatureVector',
    'CategoryLabel',
]
