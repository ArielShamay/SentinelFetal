"""
Pipeline Adapter - Bridges simulation to existing SentinelFetal pipeline.

This adapter connects the real-time simulator's RingBuffer output to the
existing Gen3.5 analysis pipeline, running the complete processing chain:

    Preprocess → Rule Engine → MiniRocket → Fusion → Classifier → Override → Alert

CRITICAL: Uses lightweight MiniRocket embeddings by default (MOMENT disabled).

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
from src.data.signal_quality import apply_quality_gate  # Added FSQI
from src.rules.baseline import calculate_baseline
from src.rules.variability import calculate_variability
from src.rules.decelerations import detect_decelerations
from src.rules.tachysystole import detect_tachysystole
from src.rules.sinusoidal import detect_sinusoidal_pattern
from src.models.minirocket_encoder import MiniRocketEncoder, MiniRocketEncoderError
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
        use_real_moment: Legacy flag; MiniRocket is always used now.
        model_path: Path to the XGBoost classifier model.
        sampling_rate: Signal sampling rate in Hz.
        min_data_seconds: Minimum data required for processing.
    """
    # Legacy flag retained for compatibility; MiniRocket is now the default engine
    use_real_moment: bool = False
    model_path: str = "models/sentinel_classifier.json"
    sampling_rate: float = 4.0
    min_data_seconds: float = 60.0  # 1 minute minimum


class PipelineAdapter:
    """
    Bridges simulation with existing SentinelFetal Gen3.5 pipeline.
    
    This adapter:
    - Uses MiniRocket embeddings by default (MOMENT removed)
    - Runs the complete analysis chain
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
            config: Configuration options. If None, uses defaults.
        """
        self.config = config or PipelineAdapterConfig()
        
        # Initialize preprocessor
        self._preprocessor = CTGPreprocessor(PreprocessingConfig(
            sampling_rate=self.config.sampling_rate
        ))
        
        # Initialize MiniRocket (lightweight encoder)
        self._encoder_available = False
        try:
            logger.info("Initializing MiniRocket encoder (lightweight)")
            self._encoder = MiniRocketEncoder()
            self._encoder_available = True
            logger.info("MiniRocket encoder initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MiniRocket encoder: {e}")
            self._encoder_available = False
        
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
        
        # Processing statistics
        self._process_count = 0
    
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
            run_moment: Legacy flag retained for compatibility; ignored when
                   using MiniRocket.
        
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
            # Step 1: Preprocessing & Quality Check
            # ================================================================
            preprocess_result = self._preprocessor.process(fhr.copy())
            fhr_clean = preprocess_result.processed_signal
            
            # CRITICAL FSQI GATE
            passes_gate, quality_result = apply_quality_gate(
                fhr_clean, self.config.sampling_rate
            )
            
            if not passes_gate:
                # Signal Rejected!
                logger.info(f"Signal rejected by FSQI: {quality_result.message} (Score: {quality_result.score:.2f})")
                return {
                    'category': 2, # Fallback/Uncertain
                    'alert': None, 
                    'findings': {
                        'error': 'Signal Quality Too Low', 
                        'quality_score': quality_result.score,
                        'message': quality_result.message
                    },
                    'confidence': 0.0,
                    'insufficient_data': True # Treat as insufficient
                }
            
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
            # Step 3: MiniRocket Features (lightweight, default engine)
            # ================================================================
            features = None
            if self._encoder_available:
                try:
                    feature_result = self._encoder.extract_features(
                        fhr_clean,
                        self.config.sampling_rate
                    )
                    features = feature_result.features
                except MiniRocketEncoderError as e:
                    logger.warning(f"MiniRocket encoding failed: {e}")
                except Exception as e:
                    logger.warning(f"MiniRocket unexpected error: {e}")
            
            if features is None:
                # Fallback: zero vector to keep pipeline running
                features = np.zeros(9996, dtype=np.float32)
            
            # ================================================================
            # Step 4: Feature Vector Fusion
            # ================================================================
            feature_vector = build_feature_vector(
                embedding=features,
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
        """No-op cache clearer (MiniRocket has no per-patient cache)."""
        return
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'total_processes': self._process_count,
            'encoder_available': self._encoder_available,
            'classifier_loaded': self._classifier_loaded,
            'using_minirocket': self._encoder_available
        }
    
    @property
    def is_minirocket_ready(self) -> bool:
        """Check if MiniRocket encoder is available."""
        return self._encoder_available
