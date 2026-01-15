"""
Tests for modular pipeline architecture.

This module tests the dependency injection container and modular pipeline,
including component replacement and backward compatibility.
"""

import pytest
import numpy as np

from src.pipeline import PipelineContainer, AnalysisPipeline
from src.interfaces.protocols import IClassifier, IFeatureExtractor, IEmbeddingResult
from src.adapters import ClassifierAdapter


class MockClassifier(IClassifier):
    """Mock classifier for testing component replacement."""
    
    def __init__(self, fixed_prediction: int = 1):
        self._prediction = fixed_prediction
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._prediction] * len(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = np.zeros((len(X), 3))
        probas[:, self._prediction] = 1.0
        return probas
    
    def save_model(self, path: str) -> None:
        pass
    
    def load_model(self, path: str) -> None:
        pass


class MockEmbeddingResult:
    """Mock embedding result."""
    
    def __init__(self, embedding: np.ndarray):
        self._embedding = embedding
    
    @property
    def embedding(self) -> np.ndarray:
        return self._embedding
    
    @property
    def is_mock(self) -> bool:
        return True


class MockFeatureExtractor(IFeatureExtractor):
    """Mock feature extractor for testing."""
    
    def __init__(self, fixed_embedding: np.ndarray = None):
        self._embedding = fixed_embedding if fixed_embedding is not None else np.zeros(1024)
    
    def extract(self, signal: np.ndarray) -> IEmbeddingResult:
        return MockEmbeddingResult(self._embedding)


class TestPipelineContainer:
    """Tests for dependency injection container."""
    
    def test_create_default(self):
        """Test default container creation."""
        container = PipelineContainer.create_default(use_mock_moment=True)
        assert container.validate(), "Container should be valid with default components"
    
    def test_component_replacement(self):
        """Test that components can be replaced."""
        container = PipelineContainer.create_default(use_mock_moment=True)
        
        # Replace classifier
        mock_classifier = MockClassifier(fixed_prediction=2)
        container.classifier = mock_classifier
        
        assert container.classifier is mock_classifier
        assert container.validate()
    
    def test_validation_fails_without_components(self):
        """Test validation fails with missing components."""
        container = PipelineContainer()
        assert not container.validate()
    
    def test_get_missing_components(self):
        """Test getting list of missing components."""
        container = PipelineContainer()
        missing = container.get_missing_components()
        
        assert 'preprocessor' in missing
        assert 'classifier' in missing
        assert 'baseline_calculator' in missing
        assert len(missing) == 11  # All components should be missing
    
    def test_partial_configuration(self):
        """Test container with some components missing."""
        container = PipelineContainer()
        container.preprocessor = None  # Explicitly None
        
        missing = container.get_missing_components()
        assert 'preprocessor' in missing


class TestAnalysisPipeline:
    """Tests for the modular analysis pipeline."""
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample FHR and UC signals."""
        np.random.seed(42)
        n_samples = 2400  # 10 minutes at 4Hz
        
        # Normal FHR with variability
        fhr = 140 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))
        fhr += np.random.normal(0, 2, n_samples)
        
        # Simple UC pattern
        uc = 10 + 70 * np.abs(np.sin(np.linspace(0, 2*np.pi, n_samples)))
        
        return fhr, uc
    
    def test_pipeline_creation(self):
        """Test pipeline can be created with valid container."""
        container = PipelineContainer.create_default(use_mock_moment=True)
        pipeline = AnalysisPipeline(container)
        
        assert pipeline is not None
        assert pipeline.container is container
    
    def test_pipeline_requires_valid_container(self):
        """Test pipeline raises error with invalid container."""
        container = PipelineContainer()  # Empty container
        
        with pytest.raises(ValueError) as exc_info:
            AnalysisPipeline(container)
        
        assert "missing required components" in str(exc_info.value).lower()
    
    def test_pipeline_with_default_components(self, sample_signals):
        """Test pipeline runs with default components."""
        fhr, uc = sample_signals
        
        container = PipelineContainer.create_default(use_mock_moment=True)
        pipeline = AnalysisPipeline(container)
        
        result = pipeline.analyze(fhr, uc)
        
        assert result.category in [1, 2, 3]
        assert 0 <= result.confidence <= 1
        assert result.alert is not None
        assert result.baseline is not None
        assert result.variability is not None
    
    def test_pipeline_with_custom_classifier(self, sample_signals):
        """Test pipeline with replaced classifier."""
        fhr, uc = sample_signals
        
        container = PipelineContainer.create_default(use_mock_moment=True)
        container.classifier = MockClassifier(fixed_prediction=2)  # Force Category 3
        
        pipeline = AnalysisPipeline(container)
        result = pipeline.analyze(fhr, uc)
        
        # ML prediction should be Category 3 (prediction 2 + 1)
        # Note: Medical override may still change the final category
        assert result.ml_prediction == 3
    
    def test_pipeline_with_custom_feature_extractor(self, sample_signals):
        """Test pipeline with replaced feature extractor."""
        fhr, uc = sample_signals
        
        # Create custom embedding
        custom_embedding = np.ones(1024) * 0.5
        
        container = PipelineContainer.create_default(use_mock_moment=True)
        container.feature_extractor = MockFeatureExtractor(custom_embedding)
        
        pipeline = AnalysisPipeline(container)
        result = pipeline.analyze(fhr, uc)
        
        # Pipeline should still work with custom extractor
        assert result.category in [1, 2, 3]
    
    def test_pipeline_result_contains_all_fields(self, sample_signals):
        """Test that analysis result contains all expected fields."""
        fhr, uc = sample_signals
        
        container = PipelineContainer.create_default(use_mock_moment=True)
        pipeline = AnalysisPipeline(container)
        
        result = pipeline.analyze(fhr, uc)
        
        # Check all fields are present
        assert hasattr(result, 'category')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'alert')
        assert hasattr(result, 'baseline')
        assert hasattr(result, 'variability')
        assert hasattr(result, 'decelerations')
        assert hasattr(result, 'tachysystole')
        assert hasattr(result, 'sinusoidal')
        assert hasattr(result, 'was_overridden')
        assert hasattr(result, 'ml_prediction')


class TestBackwardCompatibility:
    """Test that old direct imports still work."""
    
    @pytest.fixture
    def sample_fhr(self):
        """Create sample FHR signal."""
        np.random.seed(42)
        n_samples = 2400
        fhr = 140 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))
        fhr += np.random.normal(0, 2, n_samples)
        return fhr
    
    def test_direct_rule_imports(self, sample_fhr):
        """Test that direct rule imports still work."""
        from src.rules.baseline import calculate_baseline
        from src.rules.variability import calculate_variability
        
        baseline = calculate_baseline(sample_fhr, 4.0)
        variability = calculate_variability(sample_fhr, 4.0)
        
        assert baseline.value > 0
        assert variability.value > 0
    
    def test_main_module_imports(self):
        """Test that imports from main module work."""
        from src import (
            CTG, THRESHOLDS, COLORS,
            PipelineContainer, AnalysisPipeline,
            IClassifier, IBaselineCalculator,
            BaselineAdapter, ClassifierAdapter,
            calculate_baseline, VariabilityResult
        )
        
        # Just check imports work
        assert CTG is not None
        assert PipelineContainer is not None
        assert IClassifier is not None
        assert BaselineAdapter is not None
        assert calculate_baseline is not None
    
    def test_adapter_compatibility(self, sample_fhr):
        """Test that adapters produce compatible results."""
        from src.adapters import BaselineAdapter, VariabilityAdapter
        from src.rules import calculate_baseline, calculate_variability
        
        # Direct function call
        baseline_direct = calculate_baseline(sample_fhr, 4.0)
        variability_direct = calculate_variability(sample_fhr, 4.0)
        
        # Via adapter
        baseline_adapter = BaselineAdapter()
        variability_adapter = VariabilityAdapter()
        
        baseline_via_adapter = baseline_adapter.calculate(sample_fhr, 4.0)
        variability_via_adapter = variability_adapter.calculate(sample_fhr, 4.0)
        
        # Results should be the same
        assert baseline_direct.value == baseline_via_adapter.value
        assert variability_direct.value == variability_via_adapter.value


class TestProtocolCompliance:
    """Test that implementations comply with Protocol interfaces."""
    
    def test_baseline_adapter_implements_protocol(self):
        """Test BaselineAdapter implements IBaselineCalculator."""
        from src.interfaces.protocols import IBaselineCalculator
        from src.adapters import BaselineAdapter
        
        adapter = BaselineAdapter()
        assert isinstance(adapter, IBaselineCalculator)
    
    def test_classifier_adapter_implements_protocol(self):
        """Test ClassifierAdapter implements IClassifier."""
        from src.interfaces.protocols import IClassifier
        from src.adapters import ClassifierAdapter
        
        adapter = ClassifierAdapter()
        assert isinstance(adapter, IClassifier)
    
    def test_mock_classifier_implements_protocol(self):
        """Test MockClassifier implements IClassifier."""
        from src.interfaces.protocols import IClassifier
        
        mock = MockClassifier()
        assert isinstance(mock, IClassifier)


class TestInterfaceTypes:
    """Test interface type definitions."""
    
    def test_type_aliases_exist(self):
        """Test that type aliases are exported."""
        from src.interfaces import (
            FHRSignal,
            UCSignal,
            EmbeddingVector,
            FeatureVector,
            CategoryLabel
        )
        
        # These should all be numpy array types
        assert FHRSignal is not None
        assert UCSignal is not None
        assert EmbeddingVector is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
