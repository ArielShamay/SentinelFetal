"""
Main analysis pipeline using dependency injection.

This module contains the AnalysisPipeline class which orchestrates
the complete CTG analysis using injected components.
"""

from dataclasses import dataclass
from typing import Optional, List, Any
import numpy as np

from src.interfaces.protocols import (
    IAnalysisPipeline,
    IAnalysisResult,
    IAlert,
    IBaselineResult,
    IVariabilityResult,
    IDeceleration,
)
from .container import PipelineContainer
from src.config import CTG


@dataclass
class AnalysisResult:
    """
    Result of complete pipeline analysis.
    
    This dataclass contains all outputs from the analysis pipeline,
    including the final category, confidence, alert, and all
    intermediate results for interpretability.
    
    Attributes:
        category: Final category (1, 2, or 3) after medical override.
        confidence: Model confidence in the prediction (0-1).
        alert: Generated alert with Hebrew explanations.
        baseline: Baseline FHR calculation result.
        variability: Variability analysis result.
        decelerations: List of detected decelerations.
        tachysystole: Tachysystole detection result.
        sinusoidal: Sinusoidal pattern detection result.
        was_overridden: True if medical override was applied.
        ml_prediction: Original ML model prediction (before override).
    """
    
    category: int
    confidence: float
    alert: IAlert
    baseline: IBaselineResult
    variability: IVariabilityResult
    decelerations: List[IDeceleration]
    tachysystole: Any
    sinusoidal: Any
    was_overridden: bool
    ml_prediction: int


class AnalysisPipeline(IAnalysisPipeline):
    """
    Complete CTG analysis pipeline with dependency injection.
    
    This pipeline orchestrates the complete CTG analysis workflow:
    1. Preprocessing - Clean and fill gaps in FHR signal
    2. Rule Engine - Calculate baseline, variability, detect decelerations
    3. Feature Extraction - Extract MOMENT embeddings
    4. Feature Fusion - Combine embeddings with rule features
    5. Classification - Predict category using ML model
    6. Medical Override - Apply safety rules
    7. Alert Generation - Create explanatory alert
    
    All components are injected via the container, making them
    easily replaceable without changing this class.
    
    Usage:
        >>> container = PipelineContainer.create_default()
        >>> pipeline = AnalysisPipeline(container)
        >>> result = pipeline.analyze(fhr, uc)
        >>> print(f"Category: {result.category}")
        
    Custom Component:
        >>> container = PipelineContainer.create_default()
        >>> container.classifier = MyCustomClassifier()
        >>> pipeline = AnalysisPipeline(container)
    """
    
    def __init__(self, container: PipelineContainer):
        """
        Initialize pipeline with dependency container.
        
        Args:
            container: Container with all required components.
            
        Raises:
            ValueError: If container is missing required components.
        """
        if not container.validate():
            missing = container.get_missing_components()
            raise ValueError(
                f"Container missing required components: {', '.join(missing)}"
            )
        
        self._container = container
    
    @property
    def container(self) -> PipelineContainer:
        """Get the pipeline container for component access/modification."""
        return self._container
    
    def analyze(
        self,
        fhr: np.ndarray,
        uc: np.ndarray,
        sampling_rate: float = CTG.SAMPLING_RATE
    ) -> AnalysisResult:
        """
        Run complete analysis on CTG signals.
        
        This method executes the full analysis pipeline:
        1. Preprocess the FHR signal (gap filling, spike removal)
        2. Run rule engine (baseline, variability, decelerations)
        3. Extract MOMENT embeddings
        4. Fuse features into feature vector
        5. Classify using ML model
        6. Apply medical override safety rules
        7. Generate explanatory alert
        
        Args:
            fhr: FHR signal array in bpm.
            uc: UC signal array.
            sampling_rate: Signal sampling rate in Hz (default: 4.0).
            
        Returns:
            AnalysisResult with category, alert, and all findings.
            
        Example:
            >>> result = pipeline.analyze(fhr, uc)
            >>> print(f"Category: {result.category}")
            >>> print(f"Baseline: {result.baseline.value} bpm")
            >>> print(f"Alert: {result.alert.headline}")
        """
        c = self._container
        
        # Step 1: Preprocessing
        preprocess_result = c.preprocessor.process(fhr.copy())
        fhr_clean = preprocess_result.processed_signal
        
        # Step 2: Rule Engine
        baseline = c.baseline_calculator.calculate(fhr_clean, sampling_rate)
        variability = c.variability_calculator.calculate(fhr_clean, sampling_rate)
        decelerations = c.deceleration_detector.detect(
            fhr_clean, uc, baseline.value, sampling_rate
        )
        tachysystole = c.tachysystole_detector.detect(uc, sampling_rate)
        sinusoidal = c.sinusoidal_detector.detect(fhr_clean, sampling_rate)
        
        # Step 3: Feature Extraction
        embedding_result = c.feature_extractor.extract(fhr_clean)
        
        # Step 4: Feature Fusion
        feature_vector = c.feature_fusion.fuse(
            embedding=embedding_result.embedding,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal
        )
        
        # Step 5: Classification
        X = feature_vector.vector.reshape(1, -1)
        ml_prediction = int(c.classifier.predict(X)[0])
        probas = c.classifier.predict_proba(X)[0]
        confidence = float(np.max(probas))
        
        # Step 6: Medical Override
        override = c.medical_override.apply(
            ml_prediction=ml_prediction,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal
        )
        
        final_category = override.final_category + 1  # Convert to 1-indexed
        
        # Step 7: Alert Generation
        alert = c.alert_generator.generate(
            category=final_category,
            confidence=confidence,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal
        )
        
        return AnalysisResult(
            category=final_category,
            confidence=confidence,
            alert=alert,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal,
            was_overridden=override.should_override,
            ml_prediction=ml_prediction + 1  # Convert to 1-indexed
        )
