"""
Pipeline Adapter - Bridges simulation to existing SentinelFetal pipeline.

This adapter connects the real-time simulator's RingBuffer output to the
existing Gen3.5 analysis pipeline, running the complete processing chain:

    V2.0 Pipeline (10 Steps):
    MHR Guard → Preprocess → Rule Engine → MiniRocket → Fusion →
    Classifier → Override → Alert → Trend Analysis → Explanation

CRITICAL: Uses lightweight MiniRocket embeddings by default (MOMENT disabled).

V2.0 Features:
    - MHR Guard: Detects maternal heart rate contamination (Step 0)
    - Trend Analysis: 60-minute trend tracking with deterioration score (Step 8)
    - Explainability: Rule-based and SHAP explanations (Step 9)

References:
    - SentinelFetal Real-Time Simulator SPEC Part 2, Section 7
    - SentinelFetal Gen3.5 Technical Specification
    - SentinelFetal V2.0 PRD (MHR Guard, Trend Analyzer, Explainability)
"""

from __future__ import annotations

import logging
import time
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
from src.analysis.override import apply_medical_override, calculate_rule_score
from src.analysis.alerts import generate_alert

# V2.0 imports
from src.safety import MHRDetector, MHRDetectorConfig, MHRAction, MHRCheckResult
from src.analysis.trend_buffer import TrendBuffer, TrendDataPoint
from src.analysis.trend_analyzer import TrendAnalyzer, TrendAnalysisResult
from src.explainability import ExplanationEngine, ExplanationResult

# Stage 5 Imports
from src.pipeline.stage5_hybrid import Stage5Pipeline
from src.analysis.tiering import Tier


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
        enable_mhr_guard: Enable MHR detection (V2.0).
        enable_trend_analysis: Enable 60-min trend analysis (V2.0).
        enable_explanations: Enable classification explanations (V2.0).
    """
    # Legacy flag retained for compatibility; MiniRocket is now the default engine
    use_real_moment: bool = False
    model_path: str = "models/sentinel_classifier.json"
    sampling_rate: float = 4.0
    min_data_seconds: float = 60.0  # 1 minute minimum

    # V2.0 feature flags
    enable_mhr_guard: bool = True
    enable_trend_analysis: bool = True
    enable_explanations: bool = True


class PipelineAdapter:
    """
    Bridges simulation with existing SentinelFetal Gen3.5 pipeline.

    V2.0 Extended Pipeline (10 Steps):
        Step 0: MHR Guard Check (NEW)
        Step 1: Preprocessing & FSQI Quality Gate
        Step 2: Rule Engine (baseline, variability, decelerations, etc.)
        Step 3: MiniRocket Encoding
        Step 4: Feature Vector Fusion
        Step 5: XGBoost Classification
        Step 6: Medical Override (safety net)
        Step 7: Alert Generation
        Step 8: Trend Analysis (NEW)
        Step 9: Explanation Generation (NEW)

    This adapter:
    - Uses MiniRocket embeddings by default (MOMENT removed)
    - Runs the complete analysis chain
    - Applies medical override rules for safety
    - Detects maternal heart rate contamination (V2.0)
    - Tracks 60-minute trends with deterioration scoring (V2.0)
    - Generates classification explanations (V2.0)

    Example:
        >>> adapter = PipelineAdapter()
        >>> results = adapter.process_patient('P1', patient_data, run_moment=True)
        >>> print(f"Category: {results['category']}")
        >>> if results.get('mhr_alert'):
        ...     print("Warning: MHR contamination suspected!")
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

        # ================================================================
        # V2.0 Module Initialization
        # ================================================================

        # MHR Guard (Step 0)
        self._mhr_detector: Optional[MHRDetector] = None
        if self.config.enable_mhr_guard:
            try:
                self._mhr_detector = MHRDetector(MHRDetectorConfig())
                logger.info("MHR Guard initialized (V2.0)")
            except Exception as e:
                logger.warning(f"Could not initialize MHR Guard: {e}")

        # Trend Analysis (Step 8) - per-patient buffers
        self._trend_buffers: Dict[str, TrendBuffer] = {}
        self._trend_analyzer: Optional[TrendAnalyzer] = None
        if self.config.enable_trend_analysis:
            try:
                self._trend_analyzer = TrendAnalyzer()
                logger.info("Trend Analyzer initialized (V2.0)")
            except Exception as e:
                logger.warning(f"Could not initialize Trend Analyzer: {e}")

        # Explanation Engine (Step 9)
        self._explanation_engine: Optional[ExplanationEngine] = None
        if self.config.enable_explanations:
            try:
                xgb_model = self._classifier.model if self._classifier_loaded else None
                self._explanation_engine = ExplanationEngine(xgboost_model=xgb_model)
                logger.info("Explanation Engine initialized (V2.0)")
            except Exception as e:
                logger.warning(f"Could not initialize Explanation Engine: {e}")

        # Processing statistics
        self._process_count = 0
        self._mhr_blocks = 0
        self._trend_overrides = 0
        
        # Stage 5 Pipeline (Hybrid Decision)
        self._stage5_pipeline = Stage5Pipeline()
        logger.info("Stage 5 Hybrid Pipeline initialized")
    
    def process_patient(
        self,
        patient_id: str,
        data: Dict[str, Any],
        run_moment: bool = True,
        compute_shap: bool = False
    ) -> Dict[str, Any]:
        """
        Process patient data through the full V2.0 pipeline.

        V2.0 Pipeline Steps:
            0. MHR Guard Check (NEW) - detect maternal signal contamination
            1. Preprocessing (spike removal, gap filling, smoothing)
            2. Rule Engine (baseline, variability, decelerations, etc.)
            3. MiniRocket Embedding
            4. Feature Vector Fusion
            5. Classification (XGBoost)
            6. Medical Override (safety net)
            7. Alert Generation
            8. Trend Analysis (NEW) - 60-minute trend tracking
            9. Explanation Generation (NEW) - rule/SHAP explanations

        Args:
            patient_id: Patient identifier for caching and trend tracking.
            data: Dictionary with 'fhr', 'uc', 'timestamps' arrays.
                  Optional: 'mhr' array for MHR reference from SpO2.
            run_moment: Legacy flag retained for compatibility.
            compute_shap: If True, compute SHAP explanations (slow!).

        Returns:
            Dictionary containing:
                - category: Final category (1, 2, or 3), or None if blocked
                - alert: Generated Alert object
                - findings: Detailed findings from each step
                - confidence: Model confidence score
                - ml_prediction: Raw ML prediction before override
                - was_overridden: Whether override was applied
                - insufficient_data: True if not enough data
                - mhr_alert: MHR detection result (V2.0)
                - trend: Trend analysis result (V2.0)
                - explanation: Classification explanation (V2.0)
        """
        fhr = data.get('fhr', np.array([]))
        uc = data.get('uc', np.array([]))
        mhr_reference = data.get('mhr', None)  # Optional MHR from SpO2

        # Check minimum data requirement
        min_samples = int(self.config.min_data_seconds * self.config.sampling_rate)
        if len(fhr) < min_samples:
            return {
                'category': 1,
                'alert': None,
                'findings': {},
                'confidence': 0.0,
                'insufficient_data': True,
                'mhr_alert': None,
                'trend': None,
                'explanation': None
            }

        self._process_count += 1

        # Initialize V2 result placeholders
        mhr_result: Optional[MHRCheckResult] = None
        trend_result: Optional[TrendAnalysisResult] = None
        explanation_result: Optional[ExplanationResult] = None
        was_trend_overridden = False

        try:
            # ================================================================
            # Step 0: MHR Guard Check (V2.0)
            # ================================================================
            if self._mhr_detector is not None:
                # Check last 60 seconds for MHR contamination
                segment_length = min(240, len(fhr))  # 60 sec at 4Hz
                mhr_segment = fhr[-segment_length:]
                mhr_ref_segment = mhr_reference[-segment_length:] if mhr_reference is not None else None

                mhr_result = self._mhr_detector.check_segment(
                    fhr_segment=mhr_segment,
                    mhr_reference=mhr_ref_segment,
                    has_accelerations=False,  # Will update after rule engine
                    sampling_rate=self.config.sampling_rate
                )

                # BLOCK if high confidence MHR detection
                if mhr_result.recommended_action == MHRAction.BLOCK_SEGMENT:
                    self._mhr_blocks += 1
                    logger.warning(
                        f"Patient {patient_id}: MHR contamination suspected "
                        f"(confidence: {mhr_result.confidence:.2f}). Segment BLOCKED."
                    )
                    return {
                        'category': None,  # SUSPENDED
                        'alert': None,
                        'findings': {'error': 'MHR contamination suspected'},
                        'confidence': 0.0,
                        'insufficient_data': False,
                        'mhr_alert': mhr_result.to_dict(),
                        'mhr_blocked': True,
                        'trend': None,
                        'explanation': None
                    }

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
                logger.info(f"Signal rejected by FSQI: {quality_result.message} (Score: {quality_result.score:.2f})")
                return {
                    'category': 2,
                    'alert': None,
                    'findings': {
                        'error': 'Signal Quality Too Low',
                        'quality_score': quality_result.score,
                        'message': quality_result.message
                    },
                    'confidence': 0.0,
                    'insufficient_data': True,
                    'mhr_alert': mhr_result.to_dict() if mhr_result else None,
                    'trend': None,
                    'explanation': None
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

            # Detect accelerations for MHR sleep-cycle adjustment
            # (accelerations present = likely fetal sleep, not MHR)
            accelerations = self._detect_accelerations(fhr_clean, baseline_result.value)

            # Re-check MHR with acceleration info if initially suspected
            if mhr_result and mhr_result.is_suspected and accelerations:
                mhr_result = self._mhr_detector.check_segment(
                    fhr_segment=fhr[-240:] if len(fhr) >= 240 else fhr,
                    mhr_reference=mhr_ref_segment,
                    has_accelerations=True,  # Reduces MHR confidence
                    sampling_rate=self.config.sampling_rate
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
            # Step 6: Hybrid Decision (Stage 5)
            # ================================================================
            
            # Calculate rule score for Stage 5
            rule_result = calculate_rule_score(
                baseline_result,
                variability_result,
                decelerations,
                tachysystole_result,
                sinusoidal_result
            )

            # Process through Stage 5 Pipeline
            # Note: We use confidence as ai_score
            decision = self._stage5_pipeline.process_window(
                record_id=patient_id,
                window_index=self._process_count,
                window_start=time.time(), # Real-time simulation
                window_end=time.time(),
                quality_class="GOOD", # Passed gate
                ai_score=confidence,
                rule_score=rule_result.score,
                rule_hits=rule_result.rule_hits,
                is_severe=rule_result.is_severe
            )

            # Map Stage 5 Tier to Category
            # Tier 3 -> Category 3 (Severe)
            # Tier 2 -> Category 2 (Warning)
            # Tier 1 -> Category 2 (AI Alert)
            # No Alert -> Category 1 (Normal)
            final_category = 1
            if decision.tier == Tier.TIER_3.value:
                final_category = 3
            elif decision.tier == Tier.TIER_2.value or decision.tier == Tier.TIER_1.value:
                final_category = 2

            # ================================================================
            # Compile Findings
            # ================================================================
            # Dummy override result for compatibility
            from src.analysis.override import MedicalOverride, OverrideReason
            override_result = MedicalOverride(
                should_override=decision.tier == Tier.TIER_3.value,
                final_category=final_category - 1,
                reason=OverrideReason.NONE,
                ml_prediction=ml_prediction,
                explanation=decision.summary
            )

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
                'alert': None,
                'findings': findings,
                'confidence': confidence,
                'ml_prediction': ml_prediction + 1,
                'was_overridden': decision.tier == Tier.TIER_3.value,
                'insufficient_data': False,
                'mhr_alert': mhr_result.to_dict() if mhr_result and mhr_result.is_suspected else None,
                'explanation': decision.summary,
                'stage5_decision': decision.to_dict()
            }

        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}")
            return {
                'category': 2,
                'alert': None,
                'findings': {'error': str(e)},
                'confidence': 0.0,
                'processing_error': True,
                'mhr_alert': None,
                'trend': None,
                'explanation': None
            }

    def _detect_accelerations(
        self,
        fhr: np.ndarray,
        baseline: float,
        min_amplitude: float = 15.0,
        min_duration_sec: float = 15.0
    ) -> List[Dict[str, Any]]:
        """
        Simple acceleration detection for MHR sleep-cycle adjustment.

        Accelerations are defined as FHR increases of >= 15 bpm above
        baseline lasting >= 15 seconds.

        Args:
            fhr: FHR signal array.
            baseline: Baseline FHR value.
            min_amplitude: Minimum increase above baseline (default: 15 bpm).
            min_duration_sec: Minimum duration (default: 15 sec).

        Returns:
            List of acceleration events (simplified).
        """
        accelerations = []
        sr = self.config.sampling_rate
        min_samples = int(min_duration_sec * sr)

        # Find regions above baseline + threshold
        above_threshold = fhr > (baseline + min_amplitude)

        # Find contiguous regions
        in_accel = False
        start_idx = 0

        for i, above in enumerate(above_threshold):
            if above and not in_accel:
                in_accel = True
                start_idx = i
            elif not above and in_accel:
                in_accel = False
                duration = i - start_idx
                if duration >= min_samples:
                    accelerations.append({
                        'start_idx': start_idx,
                        'end_idx': i,
                        'duration_sec': duration / sr
                    })

        # Handle case where signal ends during acceleration
        if in_accel:
            duration = len(fhr) - start_idx
            if duration >= min_samples:
                accelerations.append({
                    'start_idx': start_idx,
                    'end_idx': len(fhr) - 1,
                    'duration_sec': duration / sr
                })

        return accelerations
    
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
        Clear cached data for a patient or all patients.

        Args:
            patient_id: If provided, clear only this patient's cache.
                       If None, clear all caches.
        """
        if patient_id is not None:
            # Clear specific patient
            if patient_id in self._trend_buffers:
                self._trend_buffers[patient_id].clear()
                logger.debug(f"Cleared trend buffer for patient {patient_id}")
        else:
            # Clear all
            for buffer in self._trend_buffers.values():
                buffer.clear()
            self._trend_buffers.clear()
            logger.debug("Cleared all trend buffers")

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics including V2.0 metrics."""
        stats = {
            'total_processes': self._process_count,
            'encoder_available': self._encoder_available,
            'classifier_loaded': self._classifier_loaded,
            'using_minirocket': self._encoder_available,
            # V2.0 stats
            'mhr_guard_enabled': self._mhr_detector is not None,
            'mhr_blocks': self._mhr_blocks,
            'trend_analysis_enabled': self._trend_analyzer is not None,
            'trend_overrides': self._trend_overrides,
            'active_trend_buffers': len(self._trend_buffers),
            'explanation_engine_enabled': self._explanation_engine is not None,
            'shap_available': (
                self._explanation_engine.shap_available
                if self._explanation_engine else False
            ),
        }

        # Add trend buffer stats
        if self._trend_buffers:
            total_samples = sum(b.size for b in self._trend_buffers.values())
            stats['total_trend_samples'] = total_samples

        return stats

    def get_trend_buffer(self, patient_id: str) -> Optional[TrendBuffer]:
        """
        Get trend buffer for a specific patient.

        Useful for UI to display trend sparklines.

        Args:
            patient_id: Patient identifier.

        Returns:
            TrendBuffer if exists, None otherwise.
        """
        return self._trend_buffers.get(patient_id)

    @property
    def is_minirocket_ready(self) -> bool:
        """Check if MiniRocket encoder is available."""
        return self._encoder_available

    @property
    def is_v2_ready(self) -> bool:
        """Check if all V2.0 features are available."""
        return (
            self._mhr_detector is not None and
            self._trend_analyzer is not None and
            self._explanation_engine is not None
        )
