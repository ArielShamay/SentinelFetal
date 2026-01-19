"""
Dependency injection container for pipeline components.
Allows swapping implementations without changing client code.

The container holds references to all components needed by the pipeline,
and provides factory methods for common configurations.

Example:
    # Use defaults
    >>> container = PipelineContainer.create_default()
    
    # Override specific component
    >>> container = PipelineContainer.create_default()
    >>> container.classifier = MyCustomClassifier()
    
    # Create with all custom components
    >>> container = PipelineContainer(
    ...     preprocessor=MyPreprocessor(),
    ...     classifier=MyClassifier(),
    ...     ...
    ... )
"""

from dataclasses import dataclass
from typing import Optional

from src.interfaces.protocols import (
    IPreprocessor,
    IBaselineCalculator,
    IVariabilityCalculator,
    IDecelerationDetector,
    ITachysystoleDetector,
    ISinusoidalDetector,
    IFeatureExtractor,
    IFeatureFusion,
    IClassifier,
    IMedicalOverride,
    IAlertGenerator,
)

from src.adapters import (
    PreprocessorAdapter,
    BaselineAdapter,
    VariabilityAdapter,
    DecelerationAdapter,
    TachysystoleAdapter,
    SinusoidalAdapter,
    MomentAdapter,
    FusionAdapter,
    ClassifierAdapter,
    OverrideAdapter,
    AlertAdapter,
)


@dataclass
class PipelineContainer:
    """
    Dependency injection container for the analysis pipeline.
    
    This container holds references to all components needed by the
    analysis pipeline. Each component implements a Protocol interface,
    allowing for easy swapping of implementations.
    
    Attributes:
        preprocessor: Signal preprocessor (IPreprocessor).
        baseline_calculator: Baseline FHR calculator (IBaselineCalculator).
        variability_calculator: Variability analyzer (IVariabilityCalculator).
        deceleration_detector: Deceleration detector (IDecelerationDetector).
        tachysystole_detector: Tachysystole detector (ITachysystoleDetector).
        sinusoidal_detector: Sinusoidal pattern detector (ISinusoidalDetector).
        feature_extractor: MOMENT/embedding extractor (IFeatureExtractor).
        feature_fusion: Feature fusion module (IFeatureFusion).
        classifier: Classification model (IClassifier).
        medical_override: Medical override logic (IMedicalOverride).
        alert_generator: Alert generator (IAlertGenerator).
        
    Usage:
        >>> container = PipelineContainer.create_default()
        >>> container.classifier = MyCustomClassifier()
        >>> pipeline = AnalysisPipeline(container)
    """
    
    # Data layer
    preprocessor: Optional[IPreprocessor] = None
    
    # Rule engine
    baseline_calculator: Optional[IBaselineCalculator] = None
    variability_calculator: Optional[IVariabilityCalculator] = None
    deceleration_detector: Optional[IDecelerationDetector] = None
    tachysystole_detector: Optional[ITachysystoleDetector] = None
    sinusoidal_detector: Optional[ISinusoidalDetector] = None
    
    # Model layer
    feature_extractor: Optional[IFeatureExtractor] = None
    feature_fusion: Optional[IFeatureFusion] = None
    classifier: Optional[IClassifier] = None
    
    # Analysis layer
    medical_override: Optional[IMedicalOverride] = None
    alert_generator: Optional[IAlertGenerator] = None
    
    @classmethod
    def create_default(
        cls,
        use_mock_moment: bool = False,
        model_path: str = "models/sentinel_classifier.json"
    ) -> 'PipelineContainer':
        """
        Create container with default implementations.
        
        This factory method creates a container with all the standard
        SentinelFetal implementations configured with default parameters.
        
        Args:
            use_mock_moment: Use mock MOMENT (for testing without GPU).
            model_path: Path to trained XGBoost model file.
            
        Returns:
            Configured PipelineContainer ready for use.
            
        Example:
            >>> container = PipelineContainer.create_default(use_mock_moment=True)
            >>> pipeline = AnalysisPipeline(container)
        """
        return cls(
            # Data layer
            preprocessor=PreprocessorAdapter(),
            
            # Rule engine
            baseline_calculator=BaselineAdapter(),
            variability_calculator=VariabilityAdapter(),
            deceleration_detector=DecelerationAdapter(),
            tachysystole_detector=TachysystoleAdapter(),
            sinusoidal_detector=SinusoidalAdapter(),
            
            # Model layer
            feature_extractor=MomentAdapter(use_mock=use_mock_moment),
            feature_fusion=FusionAdapter(),
            classifier=ClassifierAdapter(model_path=model_path),
            
            # Analysis layer
            medical_override=OverrideAdapter(),
            alert_generator=AlertAdapter()
        )
    
    def validate(self) -> bool:
        """
        Validate that all required components are set.
        
        Returns:
            True if all components are configured, False otherwise.
        """
        required = [
            self.preprocessor,
            self.baseline_calculator,
            self.variability_calculator,
            self.deceleration_detector,
            self.tachysystole_detector,
            self.sinusoidal_detector,
            self.feature_extractor,
            self.feature_fusion,
            self.classifier,
            self.medical_override,
            self.alert_generator
        ]
        return all(component is not None for component in required)
    
    def get_missing_components(self) -> list:
        """
        Get list of missing component names.
        
        Returns:
            List of component names that are None.
        """
        components = {
            'preprocessor': self.preprocessor,
            'baseline_calculator': self.baseline_calculator,
            'variability_calculator': self.variability_calculator,
            'deceleration_detector': self.deceleration_detector,
            'tachysystole_detector': self.tachysystole_detector,
            'sinusoidal_detector': self.sinusoidal_detector,
            'feature_extractor': self.feature_extractor,
            'feature_fusion': self.feature_fusion,
            'classifier': self.classifier,
            'medical_override': self.medical_override,
            'alert_generator': self.alert_generator
        }
        return [name for name, component in components.items() if component is None]
