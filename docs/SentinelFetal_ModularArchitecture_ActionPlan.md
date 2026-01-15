# SentinelFetal - תוכנית שינוי ארכיטקטורה למודולרית
## Action Plan for Modular Architecture Refactoring

**מטרה:** להפוך את הקוד למבנה מודולרי שבו כל רכיב ניתן להחלפה ללא השפעה על שאר המערכת.

**עיקרון מנחה:** אין לשנות את הלוגיקה או הפונקציונליות - רק את המבנה והממשקים.

---

## שלב 0: הכנה ובדיקות

### 0.1 גיבוי ובדיקות קיימות
```bash
# וודא שכל הבדיקות עוברות לפני השינוי
pytest tests/ -v

# צור branch חדש
git checkout -b refactor/modular-architecture

# שמור snapshot של מצב נוכחי
git add -A && git commit -m "Pre-refactor snapshot"
```

### 0.2 הרץ את כל הבדיקות וודא 100% pass
```bash
pytest tests/ -v --tb=short
```

**אם יש בדיקות שנכשלות - תקן אותן קודם!**

---

## שלב 1: יצירת שכבת Interfaces (Protocols)

### 1.1 צור קובץ `src/interfaces/__init__.py`
```python
"""
Abstract interfaces (Protocols) for SentinelFetal components.
All concrete implementations must conform to these interfaces.
"""
```

### 1.2 צור קובץ `src/interfaces/protocols.py`

```python
"""
Protocol definitions for dependency injection and component abstraction.
These define the contracts that components must follow.
"""

from typing import Protocol, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from abc import abstractmethod


# ============================================================================
# DATA LAYER PROTOCOLS
# ============================================================================

class IDataLoader(Protocol):
    """Interface for loading CTG data from any source."""
    
    def list_records(self) -> List[str]:
        """List available record IDs."""
        ...
    
    def load_record(self, record_id: str) -> 'ICTGRecord':
        """Load a single record by ID."""
        ...
    
    def get_outcome_label(self, record_id: str) -> int:
        """Get outcome label (0/1/2) for a record."""
        ...


class ICTGRecord(Protocol):
    """Interface for a CTG recording."""
    
    @property
    def record_id(self) -> str: ...
    
    @property
    def fhr(self) -> np.ndarray: ...
    
    @property
    def uc(self) -> np.ndarray: ...
    
    @property
    def sampling_rate(self) -> float: ...
    
    @property
    def duration_seconds(self) -> float: ...


class IPreprocessor(Protocol):
    """Interface for signal preprocessing."""
    
    def process(self, fhr: np.ndarray, apply_smoothing: bool = False) -> 'IPreprocessingResult':
        """Process FHR signal and return result."""
        ...


class IPreprocessingResult(Protocol):
    """Interface for preprocessing output."""
    
    @property
    def processed_signal(self) -> np.ndarray: ...
    
    @property
    def original_signal(self) -> np.ndarray: ...
    
    @property
    def stats(self) -> Dict[str, Any]: ...


# ============================================================================
# RULE ENGINE PROTOCOLS
# ============================================================================

class IBaselineCalculator(Protocol):
    """Interface for baseline calculation."""
    
    def calculate(self, fhr: np.ndarray, sampling_rate: float) -> 'IBaselineResult':
        """Calculate baseline FHR."""
        ...


class IBaselineResult(Protocol):
    """Interface for baseline result."""
    
    @property
    def value(self) -> float: ...
    
    @property
    def is_normal(self) -> bool: ...
    
    @property
    def is_bradycardia(self) -> bool: ...
    
    @property
    def is_tachycardia(self) -> bool: ...


class IVariabilityCalculator(Protocol):
    """Interface for variability analysis."""
    
    def calculate(self, fhr: np.ndarray, sampling_rate: float) -> 'IVariabilityResult':
        """Calculate variability."""
        ...


class IVariabilityResult(Protocol):
    """Interface for variability result."""
    
    @property
    def value(self) -> float: ...
    
    @property
    def category(self) -> Any: ...  # VariabilityCategory enum
    
    @property
    def is_normal(self) -> bool: ...
    
    @property
    def is_concerning(self) -> bool: ...


class IDecelerationDetector(Protocol):
    """Interface for deceleration detection."""
    
    def detect(
        self,
        fhr: np.ndarray,
        uc: np.ndarray,
        baseline: float,
        sampling_rate: float
    ) -> List['IDeceleration']:
        """Detect decelerations in signal."""
        ...


class IDeceleration(Protocol):
    """Interface for a single deceleration."""
    
    @property
    def start_idx(self) -> int: ...
    
    @property
    def end_idx(self) -> int: ...
    
    @property
    def depth(self) -> float: ...
    
    @property
    def duration_seconds(self) -> float: ...
    
    @property
    def decel_type(self) -> Any: ...  # DecelerationType enum
    
    @property
    def has_severity_signs(self) -> bool: ...


class ITachysystoleDetector(Protocol):
    """Interface for tachysystole detection."""
    
    def detect(self, uc: np.ndarray, sampling_rate: float) -> 'ITachysystoleResult':
        """Detect tachysystole."""
        ...


class ITachysystoleResult(Protocol):
    """Interface for tachysystole result."""
    
    @property
    def detected(self) -> bool: ...
    
    @property
    def contractions_per_10min(self) -> float: ...


class ISinusoidalDetector(Protocol):
    """Interface for sinusoidal pattern detection."""
    
    def detect(self, fhr: np.ndarray, sampling_rate: float) -> 'ISinusoidalResult':
        """Detect sinusoidal pattern."""
        ...


class ISinusoidalResult(Protocol):
    """Interface for sinusoidal result."""
    
    @property
    def detected(self) -> bool: ...
    
    @property
    def confidence(self) -> float: ...


# ============================================================================
# MODEL LAYER PROTOCOLS
# ============================================================================

class IFeatureExtractor(Protocol):
    """Interface for feature extraction (MOMENT or other)."""
    
    def extract(self, signal: np.ndarray) -> 'IEmbeddingResult':
        """Extract features/embeddings from signal."""
        ...


class IEmbeddingResult(Protocol):
    """Interface for embedding result."""
    
    @property
    def embedding(self) -> np.ndarray: ...
    
    @property
    def is_mock(self) -> bool: ...


class IFeatureFusion(Protocol):
    """Interface for combining embeddings with rule features."""
    
    def fuse(
        self,
        embedding: np.ndarray,
        baseline: IBaselineResult,
        variability: IVariabilityResult,
        decelerations: List[IDeceleration],
        tachysystole: ITachysystoleResult,
        sinusoidal: ISinusoidalResult
    ) -> 'IFeatureVector':
        """Fuse features into single vector."""
        ...


class IFeatureVector(Protocol):
    """Interface for fused feature vector."""
    
    @property
    def vector(self) -> np.ndarray: ...
    
    def get_rule_features(self) -> np.ndarray: ...
    
    def get_embedding(self) -> np.ndarray: ...


class IClassifier(Protocol):
    """Interface for classification model."""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict categories."""
        ...
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        ...
    
    def save_model(self, path: str) -> None:
        """Save model to file."""
        ...
    
    def load_model(self, path: str) -> None:
        """Load model from file."""
        ...


# ============================================================================
# ANALYSIS LAYER PROTOCOLS
# ============================================================================

class IMedicalOverride(Protocol):
    """Interface for medical safety override."""
    
    def apply(
        self,
        ml_prediction: int,
        baseline: IBaselineResult,
        variability: IVariabilityResult,
        decelerations: List[IDeceleration],
        tachysystole: ITachysystoleResult,
        sinusoidal: ISinusoidalResult
    ) -> 'IOverrideResult':
        """Apply medical override rules."""
        ...


class IOverrideResult(Protocol):
    """Interface for override result."""
    
    @property
    def should_override(self) -> bool: ...
    
    @property
    def final_category(self) -> int: ...
    
    @property
    def reason(self) -> Any: ...  # OverrideReason enum


class IAlertGenerator(Protocol):
    """Interface for alert generation."""
    
    def generate(
        self,
        category: int,
        confidence: float,
        baseline: IBaselineResult,
        variability: IVariabilityResult,
        decelerations: List[IDeceleration],
        tachysystole: ITachysystoleResult,
        sinusoidal: ISinusoidalResult
    ) -> 'IAlert':
        """Generate alert with explanations."""
        ...


class IAlert(Protocol):
    """Interface for an alert."""
    
    @property
    def category(self) -> int: ...
    
    @property
    def headline(self) -> str: ...
    
    @property
    def explanation(self) -> str: ...
    
    @property
    def findings(self) -> List[str]: ...
    
    @property
    def recommendations(self) -> List[str]: ...


# ============================================================================
# PIPELINE PROTOCOL
# ============================================================================

class IAnalysisPipeline(Protocol):
    """Interface for the complete analysis pipeline."""
    
    def analyze(
        self,
        fhr: np.ndarray,
        uc: np.ndarray
    ) -> 'IAnalysisResult':
        """Run complete analysis on CTG signals."""
        ...


class IAnalysisResult(Protocol):
    """Interface for complete analysis result."""
    
    @property
    def category(self) -> int: ...
    
    @property
    def confidence(self) -> float: ...
    
    @property
    def alert(self) -> IAlert: ...
    
    @property
    def baseline(self) -> IBaselineResult: ...
    
    @property
    def variability(self) -> IVariabilityResult: ...
    
    @property
    def decelerations(self) -> List[IDeceleration]: ...
```

### 1.3 צור קובץ `src/interfaces/types.py`

```python
"""
Type aliases and common types used across the system.
"""

from typing import TypeVar, Generic, Callable, Any
import numpy as np

# Generic type variables
T = TypeVar('T')
ResultT = TypeVar('ResultT')

# Type aliases for clarity
FHRSignal = np.ndarray  # Shape: (n_samples,), dtype: float
UCSignal = np.ndarray   # Shape: (n_samples,), dtype: float
EmbeddingVector = np.ndarray  # Shape: (1024,), dtype: float32
FeatureVector = np.ndarray  # Shape: (1035,), dtype: float32
CategoryLabel = int  # 0, 1, or 2
```

---

## שלב 2: יצירת Adapters לממשקים

### 2.1 צור קובץ `src/adapters/__init__.py`
```python
"""
Adapters that wrap existing implementations to conform to interfaces.
These enable dependency injection without changing existing code.
"""

from .rule_adapters import (
    BaselineAdapter,
    VariabilityAdapter,
    DecelerationAdapter,
    TachysystoleAdapter,
    SinusoidalAdapter
)
from .model_adapters import (
    MomentAdapter,
    ClassifierAdapter,
    FusionAdapter
)
from .analysis_adapters import (
    OverrideAdapter,
    AlertAdapter
)
from .data_adapters import (
    DataLoaderAdapter,
    PreprocessorAdapter
)

__all__ = [
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
    'PreprocessorAdapter'
]
```

### 2.2 צור קובץ `src/adapters/rule_adapters.py`

```python
"""
Adapters for rule engine components.
These wrap existing functions to conform to Protocol interfaces.
"""

from typing import List
import numpy as np

from src.interfaces.protocols import (
    IBaselineCalculator, IBaselineResult,
    IVariabilityCalculator, IVariabilityResult,
    IDecelerationDetector, IDeceleration,
    ITachysystoleDetector, ITachysystoleResult,
    ISinusoidalDetector, ISinusoidalResult
)

# Import existing implementations
from src.rules.baseline import calculate_baseline, BaselineResult
from src.rules.variability import calculate_variability, VariabilityResult
from src.rules.decelerations import detect_decelerations, Deceleration
from src.rules.tachysystole import detect_tachysystole, TachysystoleResult
from src.rules.sinusoidal import detect_sinusoidal_pattern, SinusoidalResult


class BaselineAdapter(IBaselineCalculator):
    """Adapter for baseline calculation."""
    
    def __init__(self, window_minutes: float = 2.0, variability_threshold: float = 25.0):
        self._window_minutes = window_minutes
        self._variability_threshold = variability_threshold
    
    def calculate(self, fhr: np.ndarray, sampling_rate: float) -> IBaselineResult:
        """Calculate baseline using existing implementation."""
        return calculate_baseline(
            fhr, 
            sampling_rate,
            window_minutes=self._window_minutes,
            variability_threshold=self._variability_threshold
        )


class VariabilityAdapter(IVariabilityCalculator):
    """Adapter for variability calculation."""
    
    def __init__(self, window_seconds: float = 60.0, overlap_ratio: float = 0.5):
        self._window_seconds = window_seconds
        self._overlap_ratio = overlap_ratio
    
    def calculate(self, fhr: np.ndarray, sampling_rate: float) -> IVariabilityResult:
        """Calculate variability using existing implementation."""
        return calculate_variability(
            fhr,
            sampling_rate,
            window_seconds=self._window_seconds,
            overlap_ratio=self._overlap_ratio
        )


class DecelerationAdapter(IDecelerationDetector):
    """Adapter for deceleration detection."""
    
    def __init__(
        self,
        min_depth: float = 15.0,
        min_duration_seconds: float = 15.0,
        max_duration_seconds: float = 600.0
    ):
        self._min_depth = min_depth
        self._min_duration = min_duration_seconds
        self._max_duration = max_duration_seconds
    
    def detect(
        self,
        fhr: np.ndarray,
        uc: np.ndarray,
        baseline: float,
        sampling_rate: float
    ) -> List[IDeceleration]:
        """Detect decelerations using existing implementation."""
        return detect_decelerations(
            fhr, uc, baseline, sampling_rate,
            min_depth=self._min_depth,
            min_duration_seconds=self._min_duration,
            max_duration_seconds=self._max_duration
        )


class TachysystoleAdapter(ITachysystoleDetector):
    """Adapter for tachysystole detection."""
    
    def __init__(
        self,
        analysis_window_minutes: float = 30.0,
        threshold_per_10min: int = 5
    ):
        self._window_minutes = analysis_window_minutes
        self._threshold = threshold_per_10min
    
    def detect(self, uc: np.ndarray, sampling_rate: float) -> ITachysystoleResult:
        """Detect tachysystole using existing implementation."""
        return detect_tachysystole(
            uc, sampling_rate,
            analysis_window_minutes=self._window_minutes,
            threshold_per_10min=self._threshold
        )


class SinusoidalAdapter(ISinusoidalDetector):
    """Adapter for sinusoidal pattern detection."""
    
    def __init__(
        self,
        min_duration_minutes: float = 20.0,
        freq_min: float = 3.0,
        freq_max: float = 5.0
    ):
        self._min_duration = min_duration_minutes
        self._freq_min = freq_min
        self._freq_max = freq_max
    
    def detect(self, fhr: np.ndarray, sampling_rate: float) -> ISinusoidalResult:
        """Detect sinusoidal pattern using existing implementation."""
        return detect_sinusoidal_pattern(
            fhr, sampling_rate,
            min_duration_minutes=self._min_duration,
            freq_min_cycles_per_min=self._freq_min,
            freq_max_cycles_per_min=self._freq_max
        )
```

### 2.3 צור קובץ `src/adapters/model_adapters.py`

```python
"""
Adapters for model components.
"""

from typing import List
import numpy as np

from src.interfaces.protocols import (
    IFeatureExtractor, IEmbeddingResult,
    IClassifier,
    IFeatureFusion, IFeatureVector,
    IBaselineResult, IVariabilityResult, IDeceleration,
    ITachysystoleResult, ISinusoidalResult
)

from src.models.moment_encoder import MomentFeatureExtractor
from src.models.classifier import XGBClassifierWrapper
from src.models.fusion import build_feature_vector, FeatureVector


class MomentAdapter(IFeatureExtractor):
    """Adapter for MOMENT feature extractor."""
    
    def __init__(self, use_mock: bool = False, device: str = 'cpu'):
        self._extractor = MomentFeatureExtractor(use_mock=use_mock, device=device)
    
    def extract(self, signal: np.ndarray) -> IEmbeddingResult:
        """Extract embeddings using MOMENT."""
        return self._extractor.extract(signal)


class ClassifierAdapter(IClassifier):
    """Adapter for XGBoost classifier."""
    
    def __init__(self, model_path: str = None):
        self._classifier = XGBClassifierWrapper()
        if model_path:
            self._classifier.load_model(model_path)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict_proba(X)
    
    def save_model(self, path: str) -> None:
        self._classifier.save_model(path)
    
    def load_model(self, path: str) -> None:
        self._classifier.load_model(path)


class FusionAdapter(IFeatureFusion):
    """Adapter for feature fusion."""
    
    def fuse(
        self,
        embedding: np.ndarray,
        baseline: IBaselineResult,
        variability: IVariabilityResult,
        decelerations: List[IDeceleration],
        tachysystole: ITachysystoleResult,
        sinusoidal: ISinusoidalResult
    ) -> IFeatureVector:
        """Fuse features using existing implementation."""
        return build_feature_vector(
            embedding=embedding,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal,
            start_idx=0,
            end_idx=0,
            start_time_sec=0,
            end_time_sec=0
        )
```

### 2.4 צור קובץ `src/adapters/analysis_adapters.py`

```python
"""
Adapters for analysis components.
"""

from typing import List

from src.interfaces.protocols import (
    IMedicalOverride, IOverrideResult,
    IAlertGenerator, IAlert,
    IBaselineResult, IVariabilityResult, IDeceleration,
    ITachysystoleResult, ISinusoidalResult
)

from src.analysis.override import apply_medical_override
from src.analysis.alerts import generate_alert


class OverrideAdapter(IMedicalOverride):
    """Adapter for medical override."""
    
    def apply(
        self,
        ml_prediction: int,
        baseline: IBaselineResult,
        variability: IVariabilityResult,
        decelerations: List[IDeceleration],
        tachysystole: ITachysystoleResult,
        sinusoidal: ISinusoidalResult
    ) -> IOverrideResult:
        """Apply medical override using existing implementation."""
        return apply_medical_override(
            ml_prediction=ml_prediction,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal
        )


class AlertAdapter(IAlertGenerator):
    """Adapter for alert generation."""
    
    def generate(
        self,
        category: int,
        confidence: float,
        baseline: IBaselineResult,
        variability: IVariabilityResult,
        decelerations: List[IDeceleration],
        tachysystole: ITachysystoleResult,
        sinusoidal: ISinusoidalResult
    ) -> IAlert:
        """Generate alert using existing implementation."""
        return generate_alert(
            category=category,
            confidence=confidence,
            baseline=baseline,
            variability=variability,
            decelerations=decelerations,
            tachysystole=tachysystole,
            sinusoidal=sinusoidal
        )
```

### 2.5 צור קובץ `src/adapters/data_adapters.py`

```python
"""
Adapters for data components.
"""

from typing import List
import numpy as np

from src.interfaces.protocols import (
    IDataLoader, ICTGRecord,
    IPreprocessor, IPreprocessingResult
)

from src.data.loader import CTUDataLoader, CTGRecord
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig


class DataLoaderAdapter(IDataLoader):
    """Adapter for data loading."""
    
    def __init__(self, data_path: str = None):
        self._loader = CTUDataLoader(data_path) if data_path else CTUDataLoader()
    
    def list_records(self) -> List[str]:
        return self._loader.list_records()
    
    def load_record(self, record_id: str) -> ICTGRecord:
        return self._loader.load_record(record_id)
    
    def get_outcome_label(self, record_id: str) -> int:
        return self._loader.get_outcome_label(record_id)


class PreprocessorAdapter(IPreprocessor):
    """Adapter for preprocessing."""
    
    def __init__(self, config: PreprocessingConfig = None):
        self._preprocessor = CTGPreprocessor(config or PreprocessingConfig())
    
    def process(self, fhr: np.ndarray, apply_smoothing: bool = False) -> IPreprocessingResult:
        return self._preprocessor.process(fhr, apply_smoothing)
```

---

## שלב 3: יצירת Pipeline Container עם Dependency Injection

### 3.1 צור קובץ `src/pipeline/__init__.py`
```python
"""
Analysis pipeline with dependency injection.
"""

from .container import PipelineContainer
from .analysis_pipeline import AnalysisPipeline

__all__ = ['PipelineContainer', 'AnalysisPipeline']
```

### 3.2 צור קובץ `src/pipeline/container.py`

```python
"""
Dependency injection container for pipeline components.
Allows swapping implementations without changing client code.
"""

from dataclasses import dataclass, field
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
    IAlertGenerator
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
    AlertAdapter
)


@dataclass
class PipelineContainer:
    """
    Dependency injection container for the analysis pipeline.
    
    Usage:
        # Use defaults
        container = PipelineContainer.create_default()
        
        # Override specific component
        container = PipelineContainer.create_default()
        container.classifier = MyCustomClassifier()
        
        # Create with all custom components
        container = PipelineContainer(
            preprocessor=MyPreprocessor(),
            classifier=MyClassifier(),
            ...
        )
    """
    
    # Data layer
    preprocessor: IPreprocessor = None
    
    # Rule engine
    baseline_calculator: IBaselineCalculator = None
    variability_calculator: IVariabilityCalculator = None
    deceleration_detector: IDecelerationDetector = None
    tachysystole_detector: ITachysystoleDetector = None
    sinusoidal_detector: ISinusoidalDetector = None
    
    # Model layer
    feature_extractor: IFeatureExtractor = None
    feature_fusion: IFeatureFusion = None
    classifier: IClassifier = None
    
    # Analysis layer
    medical_override: IMedicalOverride = None
    alert_generator: IAlertGenerator = None
    
    @classmethod
    def create_default(
        cls,
        use_mock_moment: bool = False,
        model_path: str = "models/xgb_demo.json"
    ) -> 'PipelineContainer':
        """
        Create container with default implementations.
        
        Args:
            use_mock_moment: Use mock MOMENT (for testing)
            model_path: Path to trained XGBoost model
            
        Returns:
            Configured PipelineContainer
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
        """Validate that all required components are set."""
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
```

### 3.3 צור קובץ `src/pipeline/analysis_pipeline.py`

```python
"""
Main analysis pipeline using dependency injection.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from src.interfaces.protocols import (
    IAnalysisPipeline, IAnalysisResult, IAlert,
    IBaselineResult, IVariabilityResult, IDeceleration
)
from .container import PipelineContainer
from src.config import CTG


@dataclass
class AnalysisResult:
    """Result of pipeline analysis."""
    category: int
    confidence: float
    alert: IAlert
    baseline: IBaselineResult
    variability: IVariabilityResult
    decelerations: List[IDeceleration]
    tachysystole: any
    sinusoidal: any
    was_overridden: bool
    ml_prediction: int


class AnalysisPipeline(IAnalysisPipeline):
    """
    Complete CTG analysis pipeline with dependency injection.
    
    All components are injected via the container, making them
    easily replaceable without changing this class.
    
    Usage:
        container = PipelineContainer.create_default()
        pipeline = AnalysisPipeline(container)
        result = pipeline.analyze(fhr, uc)
    """
    
    def __init__(self, container: PipelineContainer):
        """
        Initialize pipeline with dependency container.
        
        Args:
            container: Container with all required components
            
        Raises:
            ValueError: If container is missing required components
        """
        if not container.validate():
            raise ValueError("Container missing required components")
        
        self._container = container
    
    def analyze(
        self,
        fhr: np.ndarray,
        uc: np.ndarray,
        sampling_rate: float = CTG.SAMPLING_RATE
    ) -> AnalysisResult:
        """
        Run complete analysis on CTG signals.
        
        Args:
            fhr: FHR signal array
            uc: UC signal array
            sampling_rate: Signal sampling rate (default: 4 Hz)
            
        Returns:
            AnalysisResult with category, alert, and findings
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
            ml_prediction=ml_prediction + 1
        )
```

---

## שלב 4: עדכון הקבצים הקיימים

### 4.1 עדכן `src/__init__.py`

```python
"""
SentinelFetal - Fetal Distress Detection System

Main exports for external usage.
"""

# Configuration
from .config import CTG, THRESHOLDS, COLORS, MODEL, PATHS, HEBREW

# Pipeline (new modular interface)
from .pipeline import PipelineContainer, AnalysisPipeline

# Interfaces (for custom implementations)
from .interfaces.protocols import (
    IPreprocessor,
    IBaselineCalculator,
    IVariabilityCalculator,
    IDecelerationDetector,
    ITachysystoleDetector,
    ISinusoidalDetector,
    IFeatureExtractor,
    IClassifier,
    IMedicalOverride,
    IAlertGenerator
)

# Adapters (default implementations)
from .adapters import (
    PreprocessorAdapter,
    BaselineAdapter,
    VariabilityAdapter,
    DecelerationAdapter,
    TachysystoleAdapter,
    SinusoidalAdapter,
    MomentAdapter,
    ClassifierAdapter,
    OverrideAdapter,
    AlertAdapter
)

# Legacy direct imports (for backward compatibility)
from .data import CTGRecord, CTUDataLoader, CTGPreprocessor
from .rules import (
    calculate_baseline, BaselineResult,
    calculate_variability, VariabilityResult, VariabilityCategory,
    detect_decelerations, Deceleration, DecelerationType,
    detect_tachysystole, TachysystoleResult,
    detect_sinusoidal_pattern, SinusoidalResult
)
from .models import (
    MomentFeatureExtractor, EmbeddingResult,
    build_feature_vector, FeatureVector,
    XGBClassifierWrapper
)
from .analysis import (
    apply_medical_override, MedicalOverride, OverrideReason,
    generate_alert, Alert
)

__all__ = [
    # Configuration
    'CTG', 'THRESHOLDS', 'COLORS', 'MODEL', 'PATHS', 'HEBREW',
    
    # Modular Pipeline
    'PipelineContainer', 'AnalysisPipeline',
    
    # Interfaces
    'IPreprocessor', 'IBaselineCalculator', 'IVariabilityCalculator',
    'IDecelerationDetector', 'ITachysystoleDetector', 'ISinusoidalDetector',
    'IFeatureExtractor', 'IClassifier', 'IMedicalOverride', 'IAlertGenerator',
    
    # Adapters
    'PreprocessorAdapter', 'BaselineAdapter', 'VariabilityAdapter',
    'DecelerationAdapter', 'TachysystoleAdapter', 'SinusoidalAdapter',
    'MomentAdapter', 'ClassifierAdapter', 'OverrideAdapter', 'AlertAdapter',
    
    # Legacy (backward compatible)
    'CTGRecord', 'CTUDataLoader', 'CTGPreprocessor',
    'calculate_baseline', 'BaselineResult',
    'calculate_variability', 'VariabilityResult', 'VariabilityCategory',
    'detect_decelerations', 'Deceleration', 'DecelerationType',
    'detect_tachysystole', 'TachysystoleResult',
    'detect_sinusoidal_pattern', 'SinusoidalResult',
    'MomentFeatureExtractor', 'EmbeddingResult',
    'build_feature_vector', 'FeatureVector',
    'XGBClassifierWrapper',
    'apply_medical_override', 'MedicalOverride', 'OverrideReason',
    'generate_alert', 'Alert'
]
```

---

## שלב 5: עדכון מודול הסימולציה לשימוש בממשקים

### 5.1 עדכן `src/simulation/processing/pipeline_adapter.py`

```python
"""
Pipeline Adapter - Uses modular pipeline for simulation.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.pipeline import PipelineContainer, AnalysisPipeline
from src.config import CTG


@dataclass
class PipelineAdapterConfig:
    """Configuration for pipeline adapter."""
    use_real_moment: bool = True
    model_path: str = "models/xgb_demo.json"
    sampling_rate: float = CTG.SAMPLING_RATE


class PipelineAdapter:
    """
    Bridges simulation to analysis pipeline using dependency injection.
    
    Now uses the modular PipelineContainer instead of direct imports.
    """
    
    def __init__(self, config: Optional[PipelineAdapterConfig] = None):
        self.config = config or PipelineAdapterConfig()
        
        # Create pipeline with dependency injection
        self._container = PipelineContainer.create_default(
            use_mock_moment=not self.config.use_real_moment,
            model_path=self.config.model_path
        )
        self._pipeline = AnalysisPipeline(self._container)
        
        # Embedding cache
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def process_patient(
        self,
        patient_id: str,
        data: Dict[str, Any],
        run_moment: bool = True
    ) -> Dict[str, Any]:
        """Process patient data through the modular pipeline."""
        fhr = data['fhr']
        uc = data['uc']
        
        if len(fhr) < 240:
            return {
                'category': 1,
                'alert': None,
                'findings': {},
                'confidence': 0.0,
                'insufficient_data': True
            }
        
        # Use the modular pipeline
        result = self._pipeline.analyze(fhr, uc, self.config.sampling_rate)
        
        # Build findings dict
        findings = {
            'baseline': {
                'value': result.baseline.value,
                'is_normal': result.baseline.is_normal,
                'is_bradycardia': result.baseline.is_bradycardia,
                'is_tachycardia': result.baseline.is_tachycardia
            },
            'variability': {
                'value': result.variability.value,
                'category': result.variability.category.name,
                'is_normal': result.variability.is_normal
            },
            'decelerations': {
                'total': len(result.decelerations),
                'late': sum(1 for d in result.decelerations if d.decel_type.name == 'LATE'),
                'variable': sum(1 for d in result.decelerations if d.decel_type.name == 'VARIABLE')
            },
            'override_applied': result.was_overridden
        }
        
        return {
            'category': result.category,
            'alert': result.alert,
            'findings': findings,
            'confidence': result.confidence,
            'ml_prediction': result.ml_prediction,
            'was_overridden': result.was_overridden
        }
    
    # Expose container for custom component injection
    @property
    def container(self) -> PipelineContainer:
        """Get the pipeline container for component customization."""
        return self._container
    
    def set_classifier(self, classifier) -> None:
        """Replace the classifier component."""
        self._container.classifier = classifier
        self._pipeline = AnalysisPipeline(self._container)
    
    def set_feature_extractor(self, extractor) -> None:
        """Replace the feature extractor component."""
        self._container.feature_extractor = extractor
        self._pipeline = AnalysisPipeline(self._container)
```

---

## שלב 6: בדיקות

### 6.1 צור קובץ `tests/test_modular_pipeline.py`

```python
"""
Tests for modular pipeline architecture.
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


class TestPipelineContainer:
    """Tests for dependency injection container."""
    
    def test_create_default(self):
        """Test default container creation."""
        container = PipelineContainer.create_default(use_mock_moment=True)
        assert container.validate()
    
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
    
    def test_pipeline_with_default_components(self, sample_signals):
        """Test pipeline runs with default components."""
        fhr, uc = sample_signals
        
        container = PipelineContainer.create_default(use_mock_moment=True)
        pipeline = AnalysisPipeline(container)
        
        result = pipeline.analyze(fhr, uc)
        
        assert result.category in [1, 2, 3]
        assert 0 <= result.confidence <= 1
        assert result.alert is not None
    
    def test_pipeline_with_custom_classifier(self, sample_signals):
        """Test pipeline with replaced classifier."""
        fhr, uc = sample_signals
        
        container = PipelineContainer.create_default(use_mock_moment=True)
        container.classifier = MockClassifier(fixed_prediction=2)  # Force Category 3
        
        pipeline = AnalysisPipeline(container)
        result = pipeline.analyze(fhr, uc)
        
        # Should be Category 3 (prediction 2 + 1)
        assert result.ml_prediction == 3
    
    def test_backward_compatibility(self, sample_signals):
        """Test that old direct imports still work."""
        fhr, uc = sample_signals
        
        # Old way should still work
        from src.rules.baseline import calculate_baseline
        from src.rules.variability import calculate_variability
        
        baseline = calculate_baseline(fhr, 4.0)
        variability = calculate_variability(fhr, 4.0)
        
        assert baseline.value > 0
        assert variability.value > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### 6.2 הרץ את כל הבדיקות

```bash
# Run all tests including new modular tests
pytest tests/ -v

# Run only modular pipeline tests
pytest tests/test_modular_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## שלב 7: מבנה קבצים סופי

```
src/
├── __init__.py                    # Updated with new exports
├── config.py                      # No changes
│
├── interfaces/                    # NEW - Abstract interfaces
│   ├── __init__.py
│   ├── protocols.py              # Protocol definitions
│   └── types.py                  # Type aliases
│
├── adapters/                      # NEW - Adapters for DI
│   ├── __init__.py
│   ├── rule_adapters.py          # Rule engine adapters
│   ├── model_adapters.py         # Model adapters
│   ├── analysis_adapters.py      # Analysis adapters
│   └── data_adapters.py          # Data adapters
│
├── pipeline/                      # NEW - Modular pipeline
│   ├── __init__.py
│   ├── container.py              # Dependency injection container
│   └── analysis_pipeline.py      # Main pipeline
│
├── data/                          # No changes to internals
│   ├── __init__.py
│   ├── loader.py
│   └── preprocess.py
│
├── rules/                         # No changes to internals
│   ├── __init__.py
│   ├── baseline.py
│   ├── variability.py
│   ├── decelerations.py
│   ├── tachysystole.py
│   └── sinusoidal.py
│
├── models/                        # No changes to internals
│   ├── __init__.py
│   ├── moment_encoder.py
│   ├── fusion.py
│   └── classifier.py
│
├── analysis/                      # No changes to internals
│   ├── __init__.py
│   ├── alerts.py
│   └── override.py
│
├── simulation/                    # Updated to use modular pipeline
│   └── processing/
│       └── pipeline_adapter.py   # Updated
│
├── ui/                            # No changes needed
│   ├── app.py
│   └── simulation_app.py
│
└── utils/                         # No changes
    └── signal_utils.py
```

---

## שלב 8: סיכום ובדיקה סופית

### 8.1 בדיקה שהכל עובד
```bash
# Run all tests
pytest tests/ -v

# Test the dashboards still work
streamlit run src/ui/app.py
streamlit run src/ui/simulation_app.py
```

### 8.2 Commit final changes
```bash
git add -A
git commit -m "Refactor: Modular architecture with dependency injection

- Added interfaces/ module with Protocol definitions
- Added adapters/ module wrapping existing implementations  
- Added pipeline/ module with DI container
- Updated simulation to use modular pipeline
- Maintained full backward compatibility
- All existing tests pass"
```

---

## דוגמאות שימוש אחרי השינוי

### דוגמה 1: שימוש רגיל (ללא שינוי)
```python
# Old way still works
from src.rules.baseline import calculate_baseline
baseline = calculate_baseline(fhr, 4.0)
```

### דוגמה 2: שימוש במודולרי
```python
from src.pipeline import PipelineContainer, AnalysisPipeline

container = PipelineContainer.create_default()
pipeline = AnalysisPipeline(container)
result = pipeline.analyze(fhr, uc)
```

### דוגמה 3: החלפת מודל
```python
from src.pipeline import PipelineContainer, AnalysisPipeline
from src.interfaces.protocols import IClassifier

class MyCustomClassifier(IClassifier):
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def save_model(self, path): ...
    def load_model(self, path): ...

container = PipelineContainer.create_default()
container.classifier = MyCustomClassifier()
pipeline = AnalysisPipeline(container)
```

### דוגמה 4: החלפת MOMENT
```python
from src.interfaces.protocols import IFeatureExtractor

class MyFeatureExtractor(IFeatureExtractor):
    def extract(self, signal):
        # Use different model
        ...

container = PipelineContainer.create_default()
container.feature_extractor = MyFeatureExtractor()
```

---

*End of Action Plan | SentinelFetal Modular Architecture Refactoring*
