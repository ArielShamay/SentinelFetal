"""
Pipeline Adapter - Bridges simulation to existing SentinelFetal pipeline.

This adapter connects the real-time simulator's RingBuffer output to the
existing Gen3.5 analysis pipeline, running the complete processing chain:

    Preprocess → Rule Engine → MOMENT → Fusion → Classifier → Override → Alert

CRITICAL: Uses the REAL MOMENT model (use_mock=False) by default.

References:
    - SentinelFetal Real-Time Simulator SPEC Part 2, Section 7
    - SentinelFetal Gen3.5 Technical Specification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

# Import from existing modules
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig
from src.rules.baseline import calculate_baseline
from src.rules.variability import calculate_variability
from src.rules.decelerations import detect_decelerations
from src.rules.tachysystole import detect_tachysystole
from src.rules.sinusoidal import detect_sinusoidal_pattern
from src.models.moment_encoder import MomentFeatureExtractor
from src.models.fusion import build_feature_vector
from src.models.classifier import XGBClassifierWrapper
from src.analysis.override import apply_medical_override
from src.analysis.alerts import generate_alert


logger = logging.getLogger(__name__)


@dataclass
class PipelineAdapterConfig:
    """
    Configuration for pipeline adapter.
    
    Attributes:
        use_real_moment: Whether to use real MOMENT model (default: True).
        model_path: Path to the XGBoost classifier model.
        sampling_rate: Signal sampling rate in Hz.
        min_data_seconds: Minimum data required for processing.
    """
    use_real_moment: bool = True  # CRITICAL: Use real MOMENT model
    model_path: str = "models/xgb_demo.json"
    sampling_rate: float = 4.0
    min_data_seconds: float = 60.0  # 1 minute minimum


class PipelineAdapter:
    """
    Bridges simulation with existing SentinelFetal Gen3.5 pipeline.
    
    This adapter:
    - Uses the REAL MOMENT model (not mock) by default
    - Runs the complete analysis chain
    - Caches embeddings for efficiency when MOMENT isn't run every tick
    - Applies medical override rules for safety
    
    Example:
        >>> adapter = PipelineAdapter()  # use_mock=False by default
        >>> results = adapter.process_patient('P1', patient_data, run_moment=True)
        >>> print(f"Category: {results['category']}")
    """
    
    def __init__(self, config: Optional[PipelineAdapterConfig] = None):
        """
        Initialize the pipeline adapter.
        
        Args:
            config: Configuration options. If None, uses defaults with
                   real MOMENT model enabled.
        """
        self.config = config or PipelineAdapterConfig()
        
        # Initialize preprocessor
        self._preprocessor = CTGPreprocessor(PreprocessingConfig(
            sampling_rate=self.config.sampling_rate
        ))
        
        # Initialize MOMENT - CRITICAL: use_mock=False for real model
        logger.info(
            f"Initializing MOMENT encoder with use_mock="
            f"{not self.config.use_real_moment}"
        )
        self._moment = MomentFeatureExtractor(
            use_mock=not self.config.use_real_moment
        )
        
        # Log MOMENT mode
        if self._moment.use_mock:
            logger.warning(
                "MOMENT is running in MOCK mode. For real embeddings, "
                "install momentfm: pip install momentfm"
            )
        else:
            logger.info("MOMENT is running with REAL model")
        
        # Initialize classifier
        self._classifier = XGBClassifierWrapper()
        self._classifier_loaded = False
        
        try:
            model_path = Path(self.config.model_path)
            if model_path.exists():
                self._classifier.load_model(str(model_path))
                self._classifier_loaded = True
                logger.info(f"Classifier loaded from {model_path}")
            else:
                logger.warning(
                    f"Classifier model not found at {model_path}. "
                    "Using rule-based fallback."
                )
        except Exception as e:
            logger.warning(f"Could not load classifier: {e}. Using fallback.")
        
        # Embedding cache for efficiency
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Processing statistics
        self._process_count = 0
        self._moment_calls = 0
    
    def process_patient(
        self,
        patient_id: str,
        data: Dict[str, Any],
        run_moment: bool = True
    ) -> Dict[str, Any]:
        """
        Process patient data through the full Gen3.5 pipeline.
        
        Pipeline Steps:
            1. Preprocessing (spike removal, gap filling, smoothing)
            2. Rule Engine (baseline, variability, decelerations, etc.)
            3. MOMENT Embedding (if run_moment=True)
            4. Feature Vector Fusion
            5. Classification (XGBoost)
            6. Medical Override (safety net)
            7. Alert Generation
        
        Args:
            patient_id: Patient identifier for caching.
            data: Dictionary with 'fhr', 'uc', 'timestamps' arrays.
            run_moment: Whether to run MOMENT (expensive, ~100-300ms).
                       If False, uses cached embedding if available.
        
        Returns:
            Dictionary containing:
                - category: Final category (1, 2, or 3)
                - alert: Generated Alert object
                - findings: Detailed findings from each step
                - confidence: Model confidence score
                - ml_prediction: Raw ML prediction before override
                - was_overridden: Whether override was applied
                - insufficient_data: True if not enough data
        """
        fhr = data.get('fhr', np.array([]))
        uc = data.get('uc', np.array([]))
        
        # Check minimum data requirement
        min_samples = int(self.config.min_data_seconds * self.config.sampling_rate)
        if len(fhr) < min_samples:
            return {
                'category': 1,
                'alert': None,
                'findings': {},
                'confidence': 0.0,
                'insufficient_data': True
            }
        
        self._process_count += 1
        
        try:
            # ================================================================
            # Step 1: Preprocessing
            # ================================================================
            preprocess_result = self._preprocessor.process(fhr.copy())
            fhr_clean = preprocess_result.processed_signal
            
            # ================================================================
            # Step 2: Rule Engine
            # ================================================================
            baseline_result = calculate_baseline(
                fhr_clean, self.config.sampling_rate
            )
            
            variability_result = calculate_variability(
                fhr_clean, self.config.sampling_rate
            )
            
            decelerations = detect_decelerations(
                fhr_clean, uc, baseline_result.value, self.config.sampling_rate
            )
            
            tachysystole_result = detect_tachysystole(
                uc, self.config.sampling_rate
            )
            
            sinusoidal_result = detect_sinusoidal_pattern(
                fhr_clean, self.config.sampling_rate
            )
            
            # ================================================================
            # Step 3: MOMENT Embedding
            # ================================================================
            if run_moment:
                # extract() returns numpy array directly (not EmbeddingResult)
                embedding = self._moment.extract(fhr_clean)
                self._embedding_cache[patient_id] = embedding
                self._moment_calls += 1
            else:
                # Use cached embedding if available
                embedding = self._embedding_cache.get(patient_id)
                if embedding is None:
                    # Generate a placeholder if no cache
                    # This should only happen on first run without MOMENT
                    embedding = np.zeros(1024, dtype=np.float32)
                    logger.debug(f"No cached embedding for {patient_id}")
            
            # ================================================================
            # Step 4: Feature Vector Fusion
            # ================================================================
            feature_vector = build_feature_vector(
                embedding=embedding,
                baseline=baseline_result,
                variability=variability_result,
                decelerations=decelerations,
                tachysystole=tachysystole_result,
                sinusoidal=sinusoidal_result,
                start_idx=0,
                end_idx=len(fhr_clean),
                start_time_sec=0,
                end_time_sec=len(fhr_clean) / self.config.sampling_rate
            )
            
            # ================================================================
            # Step 5: Classification
            # ================================================================
            if self._classifier_loaded:
                X = feature_vector.vector.reshape(1, -1)
                ml_prediction = int(self._classifier.predict(X)[0])
                probas = self._classifier.predict_proba(X)[0]
                confidence = float(np.max(probas))
            else:
                ml_prediction = self._rule_based_fallback(
                    variability_result,
                    decelerations,
                    baseline_result,
                    sinusoidal_result
                )
                confidence = 0.7
            
            # ================================================================
            # Step 6: Medical Override (Safety Net)
            # ================================================================
            override_result = apply_medical_override(
                ml_prediction=ml_prediction,
                baseline=baseline_result,
                variability=variability_result,
                decelerations=decelerations,
                tachysystole=tachysystole_result,
                sinusoidal=sinusoidal_result
            )
            
            # Convert from 0-indexed to 1-indexed category
            final_category = override_result.final_category + 1
            
            # ================================================================
            # Step 7: Generate Alert
            # ================================================================
            alert = generate_alert(
                category=final_category,
                confidence=confidence,
                baseline=baseline_result,
                variability=variability_result,
                decelerations=decelerations,
                tachysystole=tachysystole_result,
                sinusoidal=sinusoidal_result
            )
            
            # ================================================================
            # Compile Findings
            # ================================================================
            findings = self._compile_findings(
                baseline_result,
                variability_result,
                decelerations,
                tachysystole_result,
                sinusoidal_result,
                override_result
            )
            
            return {
                'category': final_category,
                'alert': alert,
                'findings': findings,
                'confidence': confidence,
                'ml_prediction': ml_prediction + 1,  # 1-indexed
                'was_overridden': override_result.should_override,
                'insufficient_data': False
            }
            
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}")
            # Return safe default on error
            return {
                'category': 2,  # Intermediate - be cautious
                'alert': None,
                'findings': {'error': str(e)},
                'confidence': 0.0,
                'processing_error': True
            }
    
    def _compile_findings(
        self,
        baseline_result,
        variability_result,
        decelerations,
        tachysystole_result,
        sinusoidal_result,
        override_result
    ) -> Dict[str, Any]:
        """Compile all findings into a structured dictionary."""
        # Count deceleration types
        late_count = sum(
            1 for d in decelerations
            if hasattr(d.decel_type, 'name') and d.decel_type.name == 'LATE'
        )
        variable_count = sum(
            1 for d in decelerations
            if hasattr(d.decel_type, 'name') and d.decel_type.name == 'VARIABLE'
        )
        with_severity = sum(
            1 for d in decelerations if d.has_severity_signs
        )
        
        return {
            'baseline': {
                'value': baseline_result.value,
                'is_normal': baseline_result.is_normal,
                'is_bradycardia': baseline_result.is_bradycardia,
                'is_tachycardia': baseline_result.is_tachycardia
            },
            'variability': {
                'value': variability_result.value,
                'category': variability_result.category.name,
                'is_normal': variability_result.is_normal
            },
            'decelerations': {
                'total': len(decelerations),
                'late': late_count,
                'variable': variable_count,
                'with_severity': with_severity
            },
            'tachysystole': {
                'detected': tachysystole_result.detected,
                'rate': tachysystole_result.contractions_per_10min
            },
            'sinusoidal': {
                'detected': sinusoidal_result.detected,
                'confidence': sinusoidal_result.confidence
            },
            'override_applied': override_result.should_override,
            'override_reason': (
                override_result.reason.name
                if override_result.should_override else None
            )
        }
    
    def _rule_based_fallback(
        self,
        variability,
        decelerations,
        baseline,
        sinusoidal
    ) -> int:
        """
        Fallback classification when classifier is unavailable.
        
        Uses conservative rules based on Israeli Position Paper.
        Returns 0-indexed category (0=Normal, 1=Intermediate, 2=Pathological).
        """
        # Sinusoidal is always Category 3
        if sinusoidal.detected:
            return 2
        
        # Absent variability with concerning findings
        if variability.category.name == 'ABSENT':
            late_count = sum(
                1 for d in decelerations
                if hasattr(d.decel_type, 'name') and d.decel_type.name == 'LATE'
            )
            var_count = sum(
                1 for d in decelerations
                if hasattr(d.decel_type, 'name') and d.decel_type.name == 'VARIABLE'
            )
            
            if late_count >= 2 or var_count >= 2:
                return 2  # Pathological
            if baseline.is_bradycardia:
                return 2  # Pathological
            
            return 1  # At least Intermediate
        
        # Perfect normal pattern
        if (baseline.is_normal and
            variability.category.name == 'MODERATE' and
            len(decelerations) == 0):
            return 0  # Normal
        
        # Default to intermediate
        return 1
    
    def clear_cache(self, patient_id: Optional[str] = None) -> None:
        """
        Clear embedding cache.
        
        Args:
            patient_id: If provided, clear only this patient's cache.
                       If None, clear all caches.
        """
        if patient_id:
            self._embedding_cache.pop(patient_id, None)
        else:
            self._embedding_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'total_processes': self._process_count,
            'moment_calls': self._moment_calls,
            'cached_patients': len(self._embedding_cache),
            'classifier_loaded': self._classifier_loaded,
            'using_real_moment': not self._moment.use_mock
        }
    
    @property
    def is_moment_real(self) -> bool:
        """Check if using real MOMENT model."""
        return not self._moment.use_mock
