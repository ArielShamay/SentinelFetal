"""
Protocol definitions for dependency injection and component abstraction.
These define the contracts that components must follow.

Using Python's Protocol (structural subtyping) allows any class that
implements the required methods to be used, without explicit inheritance.

Example:
    >>> class MyClassifier(IClassifier):
    ...     def predict(self, X): return np.array([1])
    ...     def predict_proba(self, X): return np.array([[0.1, 0.8, 0.1]])
    ...     def save_model(self, path): pass
    ...     def load_model(self, path): pass
    >>> 
    >>> container.classifier = MyClassifier()  # Works!
"""

from typing import Protocol, List, Dict, Any, Optional, Tuple, runtime_checkable
import numpy as np


# ============================================================================
# DATA LAYER PROTOCOLS
# ============================================================================

@runtime_checkable
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


@runtime_checkable
class ICTGRecord(Protocol):
    """Interface for a CTG recording."""
    
    @property
    def record_id(self) -> str:
        """Unique identifier for the record."""
        ...
    
    @property
    def fhr(self) -> np.ndarray:
        """Primary fetal heart rate signal."""
        ...
    
    @property
    def uc(self) -> np.ndarray:
        """Uterine contraction signal."""
        ...
    
    @property
    def sampling_rate(self) -> float:
        """Sampling frequency in Hz."""
        ...
    
    @property
    def duration_seconds(self) -> float:
        """Total duration of recording in seconds."""
        ...


@runtime_checkable
class IPreprocessor(Protocol):
    """Interface for signal preprocessing."""
    
    def process(self, fhr: np.ndarray, apply_smoothing: bool = False) -> 'IPreprocessingResult':
        """
        Process FHR signal and return result.
        
        Args:
            fhr: Raw FHR signal array.
            apply_smoothing: Whether to apply median filter smoothing.
            
        Returns:
            Preprocessing result with cleaned signal and statistics.
        """
        ...


@runtime_checkable
class IPreprocessingResult(Protocol):
    """Interface for preprocessing output."""
    
    @property
    def processed_signal(self) -> np.ndarray:
        """Cleaned and gap-filled signal."""
        ...
    
    @property
    def original_signal(self) -> np.ndarray:
        """Original unmodified signal."""
        ...
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Preprocessing statistics."""
        ...


# ============================================================================
# RULE ENGINE PROTOCOLS
# ============================================================================

@runtime_checkable
class IBaselineCalculator(Protocol):
    """Interface for baseline calculation."""
    
    def calculate(self, fhr: np.ndarray, sampling_rate: float) -> 'IBaselineResult':
        """
        Calculate baseline FHR.
        
        Args:
            fhr: FHR signal array in bpm.
            sampling_rate: Sampling frequency in Hz.
            
        Returns:
            Baseline calculation result.
        """
        ...


@runtime_checkable
class IBaselineResult(Protocol):
    """Interface for baseline result."""
    
    @property
    def value(self) -> float:
        """Calculated baseline in bpm."""
        ...
    
    @property
    def is_normal(self) -> bool:
        """True if baseline is within normal range (110-160 bpm)."""
        ...
    
    @property
    def is_bradycardia(self) -> bool:
        """True if baseline < 110 bpm."""
        ...
    
    @property
    def is_tachycardia(self) -> bool:
        """True if baseline > 160 bpm."""
        ...


@runtime_checkable
class IVariabilityCalculator(Protocol):
    """Interface for variability analysis."""
    
    def calculate(self, fhr: np.ndarray, sampling_rate: float) -> 'IVariabilityResult':
        """
        Calculate FHR variability.
        
        Args:
            fhr: FHR signal array in bpm.
            sampling_rate: Sampling frequency in Hz.
            
        Returns:
            Variability calculation result.
        """
        ...


@runtime_checkable
class IVariabilityResult(Protocol):
    """Interface for variability result."""
    
    @property
    def value(self) -> float:
        """Average variability amplitude in bpm."""
        ...
    
    @property
    def category(self) -> Any:
        """Variability category (Absent/Minimal/Moderate/Marked)."""
        ...
    
    @property
    def is_normal(self) -> bool:
        """True if variability is Moderate (6-25 bpm)."""
        ...
    
    @property
    def is_concerning(self) -> bool:
        """True if variability is Absent or Minimal."""
        ...


@runtime_checkable
class IDecelerationDetector(Protocol):
    """Interface for deceleration detection."""
    
    def detect(
        self,
        fhr: np.ndarray,
        uc: np.ndarray,
        baseline: float,
        sampling_rate: float
    ) -> List['IDeceleration']:
        """
        Detect decelerations in signal.
        
        Args:
            fhr: FHR signal array in bpm.
            uc: Uterine contraction signal array.
            baseline: Calculated baseline FHR in bpm.
            sampling_rate: Sampling frequency in Hz.
            
        Returns:
            List of detected decelerations.
        """
        ...


@runtime_checkable
class IDeceleration(Protocol):
    """Interface for a single deceleration."""
    
    @property
    def start_idx(self) -> int:
        """Start index of the deceleration."""
        ...
    
    @property
    def end_idx(self) -> int:
        """End index of the deceleration."""
        ...
    
    @property
    def depth(self) -> float:
        """Depth of deceleration below baseline in bpm."""
        ...
    
    @property
    def duration_seconds(self) -> float:
        """Duration of the deceleration in seconds."""
        ...
    
    @property
    def decel_type(self) -> Any:
        """Classification (Early/Late/Variable/Prolonged)."""
        ...
    
    @property
    def has_severity_signs(self) -> bool:
        """True if severity signs are present."""
        ...


@runtime_checkable
class ITachysystoleDetector(Protocol):
    """Interface for tachysystole detection."""
    
    def detect(self, uc: np.ndarray, sampling_rate: float) -> 'ITachysystoleResult':
        """
        Detect tachysystole (excessive uterine activity).
        
        Args:
            uc: Uterine contraction signal array.
            sampling_rate: Sampling frequency in Hz.
            
        Returns:
            Tachysystole detection result.
        """
        ...


@runtime_checkable
class ITachysystoleResult(Protocol):
    """Interface for tachysystole result."""
    
    @property
    def detected(self) -> bool:
        """True if tachysystole is present (>5 contractions/10min)."""
        ...
    
    @property
    def contractions_per_10min(self) -> float:
        """Average contraction rate per 10 minutes."""
        ...


@runtime_checkable
class ISinusoidalDetector(Protocol):
    """Interface for sinusoidal pattern detection."""
    
    def detect(self, fhr: np.ndarray, sampling_rate: float) -> 'ISinusoidalResult':
        """
        Detect sinusoidal pattern.
        
        Args:
            fhr: FHR signal array in bpm.
            sampling_rate: Sampling frequency in Hz.
            
        Returns:
            Sinusoidal detection result.
        """
        ...


@runtime_checkable
class ISinusoidalResult(Protocol):
    """Interface for sinusoidal result."""
    
    @property
    def detected(self) -> bool:
        """True if sinusoidal pattern is present (SEVERE FINDING)."""
        ...
    
    @property
    def confidence(self) -> float:
        """Confidence in the detection (0-1)."""
        ...


# ============================================================================
# MODEL LAYER PROTOCOLS
# ============================================================================

@runtime_checkable
class IFeatureExtractor(Protocol):
    """Interface for feature extraction (MOMENT or other)."""
    
    def extract(self, signal: np.ndarray) -> 'IEmbeddingResult':
        """
        Extract features/embeddings from signal.
        
        Args:
            signal: Input signal array.
            
        Returns:
            Embedding extraction result.
        """
        ...


@runtime_checkable
class IEmbeddingResult(Protocol):
    """Interface for embedding result."""
    
    @property
    def embedding(self) -> np.ndarray:
        """1024-dimensional embedding vector."""
        ...
    
    @property
    def is_mock(self) -> bool:
        """True if this was generated by mock mode."""
        ...


@runtime_checkable
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
        """
        Fuse features into single vector.
        
        Args:
            embedding: MOMENT embedding vector.
            baseline: Baseline calculation result.
            variability: Variability calculation result.
            decelerations: List of detected decelerations.
            tachysystole: Tachysystole detection result.
            sinusoidal: Sinusoidal detection result.
            
        Returns:
            Fused feature vector.
        """
        ...


@runtime_checkable
class IFeatureVector(Protocol):
    """Interface for fused feature vector."""
    
    @property
    def vector(self) -> np.ndarray:
        """The 1035-dimensional feature vector."""
        ...
    
    def get_rule_features(self) -> np.ndarray:
        """Extract only the rule-based features (indices 1024-1034)."""
        ...
    
    def get_embedding(self) -> np.ndarray:
        """Extract only the MOMENT embedding (indices 0-1023)."""
        ...


@runtime_checkable
class IClassifier(Protocol):
    """Interface for classification model."""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict categories.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Array of predicted categories (0, 1, or 2).
        """
        ...
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Array of shape (n_samples, 3) with class probabilities.
        """
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

@runtime_checkable
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
        """
        Apply medical override rules.
        
        Args:
            ml_prediction: ML model's predicted category (0, 1, or 2).
            baseline: Baseline calculation result.
            variability: Variability calculation result.
            decelerations: List of detected decelerations.
            tachysystole: Tachysystole detection result.
            sinusoidal: Sinusoidal detection result.
            
        Returns:
            Override result with final category.
        """
        ...


@runtime_checkable
class IOverrideResult(Protocol):
    """Interface for override result."""
    
    @property
    def should_override(self) -> bool:
        """Whether the ML prediction should be overridden."""
        ...
    
    @property
    def final_category(self) -> int:
        """The final category after override (0, 1, or 2)."""
        ...
    
    @property
    def reason(self) -> Any:
        """Override reason enum."""
        ...


@runtime_checkable
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
        """
        Generate alert with explanations.
        
        Args:
            category: Final category (1, 2, or 3).
            confidence: Model confidence (0-1).
            baseline: Baseline calculation result.
            variability: Variability calculation result.
            decelerations: List of detected decelerations.
            tachysystole: Tachysystole detection result.
            sinusoidal: Sinusoidal detection result.
            
        Returns:
            Generated alert with Hebrew explanations.
        """
        ...


@runtime_checkable
class IAlert(Protocol):
    """Interface for an alert."""
    
    @property
    def category(self) -> int:
        """Classification category (1, 2, or 3)."""
        ...
    
    @property
    def headline(self) -> str:
        """Short headline describing the alert."""
        ...
    
    @property
    def explanation(self) -> str:
        """Detailed explanation in Hebrew."""
        ...
    
    @property
    def findings(self) -> List[str]:
        """List of medical findings supporting the decision."""
        ...
    
    @property
    def recommendations(self) -> List[str]:
        """List of recommendations for the medical team."""
        ...


# ============================================================================
# PIPELINE PROTOCOL
# ============================================================================

@runtime_checkable
class IAnalysisPipeline(Protocol):
    """Interface for the complete analysis pipeline."""
    
    def analyze(
        self,
        fhr: np.ndarray,
        uc: np.ndarray
    ) -> 'IAnalysisResult':
        """
        Run complete analysis on CTG signals.
        
        Args:
            fhr: Fetal heart rate signal array.
            uc: Uterine contraction signal array.
            
        Returns:
            Complete analysis result.
        """
        ...


@runtime_checkable
class IAnalysisResult(Protocol):
    """Interface for complete analysis result."""
    
    @property
    def category(self) -> int:
        """Final category (1, 2, or 3)."""
        ...
    
    @property
    def confidence(self) -> float:
        """Model confidence (0-1)."""
        ...
    
    @property
    def alert(self) -> IAlert:
        """Generated alert with explanations."""
        ...
    
    @property
    def baseline(self) -> IBaselineResult:
        """Baseline calculation result."""
        ...
    
    @property
    def variability(self) -> IVariabilityResult:
        """Variability calculation result."""
        ...
    
    @property
    def decelerations(self) -> List[IDeceleration]:
        """List of detected decelerations."""
        ...
