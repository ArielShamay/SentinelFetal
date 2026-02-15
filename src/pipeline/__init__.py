"""
Analysis pipeline with dependency injection.

This module provides the modular pipeline architecture for SentinelFetal,
allowing any component to be replaced without affecting the rest of the system.

Usage:
    # Default configuration
    >>> container = PipelineContainer.create_default()
    >>> pipeline = AnalysisPipeline(container)
    >>> result = pipeline.analyze(fhr, uc)
    
    # Custom classifier
    >>> container = PipelineContainer.create_default()
    >>> container.classifier = MyCustomClassifier()
    >>> pipeline = AnalysisPipeline(container)
"""

# Temporarily comment out missing imports
# from .container import PipelineContainer
# from .analysis_pipeline import AnalysisPipeline, AnalysisResult

__all__ = ['PipelineContainer', 'AnalysisPipeline', 'AnalysisResult']
