"""
Trend Analyzer - Combines XGBoost Classification with Clinical Rule Engine
=========================================================================

Now integrated with ExplanationEngine for full explainability support.
Supports configurable thresholds for sensitivity/specificity tuning.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from .xgboost_classifier import get_classifier, XGBoostCTGClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False

# Explainability integration
try:
    from src.explainability.explanation_engine import get_explanation_engine
    from src.explainability.models import ExplanationResult
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    ExplanationResult = None

# Rules imports for building rule outputs
try:
    from src.rules import (
        calculate_baseline, 
        calculate_variability, 
        detect_decelerations,
        detect_tachysystole, 
        detect_sinusoidal_pattern
    )
    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False
    
logger = logging.getLogger(__name__)

# Default thresholds (can be overridden by config file)
DEFAULT_THRESHOLDS = {
    'bradycardia_baseline': 100,      # Was 110 - less aggressive
    'bradycardia_duration': 180,      # 3 minutes
    'tachycardia_baseline': 170,      # Was 160 - less aggressive
    'absent_variability': 3,          # Was 5 - stricter definition
    'late_decel_drop_pct': 0.15,      # Was 0.10 - less sensitive
    'variable_decel_count': 5,        # Was 3 - require more
    'min_confidence': 0.60,           # Don't trust low-confidence ML
}


def load_thresholds() -> Dict[str, Any]:
    """Load thresholds from config file or use defaults."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'classification_thresholds.yaml'
    
    if YAML_AVAILABLE and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded thresholds from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load thresholds config: {e}")
    
    return DEFAULT_THRESHOLDS


class TrendAnalyzer:
    """
    Analyzes CTG trends and provides clinical categorization.
    
    Combines:
    1. XGBoost ML classification for pattern recognition
    2. Clinical rule engine for NICHD compliance
    3. Trend scoring for deterioration detection
    4. ExplanationEngine for human-readable explanations + visual highlights
    
    Now with configurable thresholds for sensitivity/specificity optimization.
    """
    
    def __init__(self):
        self.classifier = get_classifier() if CLASSIFIER_AVAILABLE else None
        self.explanation_engine = get_explanation_engine() if EXPLAINABILITY_AVAILABLE else None
        self.history: List[Dict[str, Any]] = []
        self.max_history = 60  # Keep 60 analysis results (e.g., 60 minutes)
        
        # Load configurable thresholds
        self.config = load_thresholds()
        self._extract_thresholds()
    
    def _extract_thresholds(self):
        """Extract threshold values from config."""
        # Clinical override thresholds
        overrides = self.config.get('clinical_overrides', {})
        
        brady = overrides.get('bradycardia', {})
        self.brady_threshold = brady.get('baseline_threshold', DEFAULT_THRESHOLDS['bradycardia_baseline'])
        self.brady_severe_threshold = brady.get('severe_threshold', 100)
        self.brady_duration = brady.get('duration_seconds', DEFAULT_THRESHOLDS['bradycardia_duration'])
        
        tachy = overrides.get('tachycardia', {})
        self.tachy_threshold = tachy.get('baseline_threshold', DEFAULT_THRESHOLDS['tachycardia_baseline'])
        
        absent_var = overrides.get('absent_variability', {})
        self.absent_var_threshold = absent_var.get('threshold_bpm', DEFAULT_THRESHOLDS['absent_variability'])
        self.absent_var_critical_threshold = absent_var.get('critical_threshold_bpm', 3)
        self.absent_var_requires_decels = absent_var.get('requires_decels', False)
        
        late_dec = overrides.get('late_decelerations', {})
        self.late_decel_min_count = late_dec.get('min_count', 2)
        
        var_dec = overrides.get('variable_decelerations', {})
        self.var_decel_min_count = var_dec.get('min_count', 3)
        
        # Reduced variability (new)
        reduced_var = overrides.get('reduced_variability', {})
        self.reduced_var_threshold = reduced_var.get('threshold_bpm', 10)
        
        # High variability - for sensitivity
        high_var = overrides.get('high_variability', {})
        self.high_var_enabled = high_var.get('enabled', False)
        self.high_var_threshold = high_var.get('threshold_bpm', 25)
        
        # XGBoost thresholds
        xgb = self.config.get('xgboost', {})
        self.min_confidence = xgb.get('min_confidence', DEFAULT_THRESHOLDS['min_confidence'])
        
        # Signal quality
        quality = self.config.get('signal_quality', {})
        self.max_valid_variability = quality.get('max_valid_variability', 50)
        
        # Temporal Confirmation thresholds (NEW)
        temporal = self.config.get('temporal_confirmation', {})
        self.temporal_enabled = temporal.get('enabled', True)
        self.brady_confirmed_duration = temporal.get('brady_confirmed_duration_sec', 30)
        self.brady_prolonged_duration = temporal.get('brady_prolonged_duration_sec', 60)
        self.brady_recurrent_count = temporal.get('brady_recurrent_count', 2)
        
        fp_filter = temporal.get('fp_filter', {})
        self.fp_filter_enabled = fp_filter.get('enabled', True)
        self.fp_pct_normal_threshold = fp_filter.get('pct_normal_threshold', 70)
        self.fp_max_brady_duration = fp_filter.get('max_brady_duration_sec', 45)
        
        critical = temporal.get('critical_patterns', {})
        self.critical_brady_bpm = critical.get('severe_baseline_brady_bpm', 100)
        self.critical_tachy_bpm = critical.get('severe_baseline_tachy_bpm', 170)
        self.critical_prolonged_sec = critical.get('prolonged_decel_sec', 60)
        self.critical_absent_var_brady = critical.get('absent_var_with_brady_threshold', 3)
        
        logger.info(f"Thresholds loaded: brady<{self.brady_threshold}, absent_var<{self.absent_var_threshold}, reduced_var<{self.reduced_var_threshold}")
        logger.info(f"Temporal Confirmation: enabled={self.temporal_enabled}, brady_confirmed>{self.brady_confirmed_duration}s")
        
    def analyze(
        self, 
        fhr: np.ndarray, 
        uc: np.ndarray,
        baseline: float = 140.0,
        variability: float = 10.0
    ) -> Dict[str, Any]:
        """
        Perform comprehensive CTG analysis.
        
        Args:
            fhr: FHR signal array
            uc: UC signal array  
            baseline: Current baseline FHR
            variability: Current variability
            
        Returns:
            Analysis result with category, trend, explanation, and highlight_regions
        """
        # Get ML classification
        ml_result = self._get_ml_classification(fhr, uc)
        
        # Apply clinical rules (may override ML)
        clinical_result = self._apply_clinical_rules(
            fhr, uc, baseline, variability, ml_result
        )
        
        # Calculate trend score
        trend_score = self._calculate_trend_score(clinical_result)
        
        # Generate rich explanation with ExplanationEngine
        explanation, highlight_regions = self._generate_rich_explanation(
            fhr, uc, baseline, variability, 
            clinical_result, ml_result
        )
        
        # Build final result
        result = {
            'category': clinical_result['category'],
            'category_name': clinical_result['category_name'],
            'confidence': ml_result['confidence'],
            'probabilities': ml_result['probabilities'],
            'trend_score': trend_score,
            'trend_direction': self._get_trend_direction(),
            'explanation': explanation,
            'highlight_regions': highlight_regions,  # NEW: Visual highlights
            'ml_category': ml_result['category'],
            'clinical_overrides': clinical_result.get('overrides', [])
        }
        
        # Store in history
        self._update_history(result)
        
        return result
    
    def _generate_rich_explanation(
        self,
        fhr: np.ndarray,
        uc: np.ndarray,
        baseline: float,
        variability: float,
        clinical_result: Dict[str, Any],
        ml_result: Dict[str, Any]
    ) -> tuple:
        """
        Generate rich explanation using ExplanationEngine.
        
        Returns:
            Tuple of (explanation_dict, highlight_regions_list)
        """
        highlight_regions = []
        
        # Try using ExplanationEngine if available
        if self.explanation_engine and RULES_AVAILABLE:
            try:
                # Build rule outputs from current analysis
                rule_outputs = self._build_rule_outputs(fhr, uc, baseline, variability)
                
                # Get ML probabilities for SHAP
                ml_proba = ml_result.get('probabilities', {0: 0.5, 1: 0.3, 2: 0.2})
                
                # Call ExplanationEngine with correct interface
                exp_result = self.explanation_engine.explain(
                    category=clinical_result['category'],
                    confidence=ml_result['confidence'],
                    rule_outputs=rule_outputs,
                    fhr_length=len(fhr),
                    ml_features=None,  # Skip SHAP for now
                    compute_shap=False
                )
                
                # Convert to dict format expected by Frontend
                # Frontend expects: primary_reason, contributing_factors, confidence (0-100)
                confidence_pct = int(ml_result['confidence'] * 100)
                explanation = {
                    'category': clinical_result['category'],
                    'category_name': clinical_result['category_name'],
                    'primary_reason': self._extract_primary_reason(exp_result),
                    'contributing_factors': [
                        c.description for c in exp_result.contributors[:5]
                    ],
                    'summary': exp_result.summary,
                    'factors': [
                        {
                            'factor': c.name,
                            'value': c.description,
                            'assessment': c.source.value,
                            'concern': 'high' if c.contribution > 0.5 else 'medium' if c.contribution > 0.2 else 'low'
                        }
                        for c in exp_result.contributors[:5]  # Top 5
                    ],
                    'confidence': confidence_pct,
                    'recommendation': self._get_recommendation(clinical_result['category'])
                }
                
                # Convert highlights to serializable format
                # Frontend expects: start_idx, end_idx, severity, label, color
                highlight_regions = [
                    {
                        'start_idx': h.start,
                        'end_idx': h.end,
                        'color': h.color,
                        'label': h.label,
                        'severity': 'critical' if h.is_pathological else 'warning'
                    }
                    for h in exp_result.highlights
                ]
                
                logger.debug(f"ExplanationEngine generated {len(highlight_regions)} highlight regions")
                return explanation, highlight_regions
                
            except Exception as e:
                logger.warning(f"ExplanationEngine failed, falling back: {e}")
        
        # Fallback to simple explanation
        explanation = self._generate_explanation(
            clinical_result, ml_result, baseline, variability
        )
        return explanation, highlight_regions
    
    def _build_rule_outputs(
        self,
        fhr: np.ndarray,
        uc: np.ndarray,
        baseline_val: float,
        variability_val: float
    ) -> Dict[str, Any]:
        """
        Build rule outputs dictionary for ExplanationEngine.
        
        Runs the rule engine functions to get proper typed results.
        """
        rule_outputs = {}
        
        try:
            # Run actual rules if available
            if RULES_AVAILABLE:
                baseline_result = calculate_baseline(fhr, sampling_rate=4.0)
                variability_result = calculate_variability(fhr, sampling_rate=4.0)
                decelerations = detect_decelerations(fhr, uc, baseline_val, sampling_rate=4.0)
                tachysystole = detect_tachysystole(uc, sampling_rate=4.0)
                sinusoidal = detect_sinusoidal_pattern(fhr, sampling_rate=4.0)
                
                rule_outputs = {
                    'baseline': baseline_result,
                    'variability': variability_result,
                    'decelerations': decelerations,
                    'tachysystole': tachysystole,
                    'sinusoidal': sinusoidal,
                }
            else:
                # Create mock rule outputs
                rule_outputs = {
                    'baseline': type('MockBaseline', (), {'value': baseline_val, 'is_normal': 110 <= baseline_val <= 160})(),
                    'variability': type('MockVar', (), {'value': variability_val, 'category': type('C', (), {'name': 'MODERATE' if 6 <= variability_val <= 25 else 'MINIMAL'})()})(),
                    'decelerations': [],
                    'tachysystole': type('MockTachy', (), {'detected': False})(),
                    'sinusoidal': type('MockSinus', (), {'detected': False})(),
                }
        except Exception as e:
            logger.warning(f"Failed to build rule outputs: {e}")
            rule_outputs = {}
        
        return rule_outputs
    
    def _extract_primary_reason(self, exp_result) -> str:
        """Extract primary reason from ExplanationResult."""
        if exp_result.contributors:
            top = exp_result.contributors[0]
            return top.description
        return "Analysis complete"
    
    def _get_ml_classification(self, fhr: np.ndarray, uc: np.ndarray) -> Dict[str, Any]:
        """Get classification from XGBoost model."""
        if self.classifier and self.classifier.is_loaded:
            return self.classifier.classify(fhr, uc)
        else:
            # Default fallback
            return {
                'category': 0,
                'category_name': 'Category I (Normal)',
                'confidence': 0.5,
                'probabilities': {0: 0.5, 1: 0.3, 2: 0.2},
                'color': 'green'
            }
    
    def _apply_clinical_rules(
        self,
        fhr: np.ndarray,
        uc: np.ndarray,
        baseline: float,
        variability: float,
        ml_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply NICHD clinical rules with Temporal Confirmation.
        
        Strategy (based on Israeli Position Paper 2023 + data analysis):
        1. CRITICAL patterns: Immediate alert (no confirmation needed)
        2. BORDERLINE patterns: Require temporal confirmation
        3. FP FILTER: High % normal + short episodes = likely false positive
        
        Note: Real-data model uses 1-indexed categories:
        - Category 1 = Normal
        - Category 2 = Suspicious
        - Category 3 = Pathological
        """
        overrides = []
        category = ml_result['category']
        category_name = ml_result['category_name']
        confidence = ml_result.get('confidence', 0.5)
        sample_rate = 4  # 4 Hz sampling
        
        # Signal quality check - high variability may indicate poor signal
        if variability > self.max_valid_variability:
            return {
                'category': 2,
                'category_name': 'Category II (Suspicious)',
                'overrides': [f'Signal quality concern: variability {variability:.1f} bpm too high']
            }
        
        # ================================================================
        # TEMPORAL CONFIRMATION ANALYSIS
        # ================================================================
        if self.temporal_enabled:
            temporal_result = self._temporal_confirmation_analysis(fhr, sample_rate, baseline, variability)
            
            # CRITICAL patterns - immediate Category 3
            if temporal_result['is_critical']:
                return {
                    'category': 3,
                    'category_name': 'Category III (Pathological)',
                    'overrides': temporal_result['critical_reasons']
                }
            
            # FP Filter - if flagged as likely FP, don't escalate
            if temporal_result['is_likely_fp']:
                # Return ML result or Category 1 (don't upgrade)
                return {
                    'category': max(1, ml_result['category']) if ml_result['category'] <= 1 else 1,
                    'category_name': 'Category I (Normal)',
                    'overrides': [f"Transient pattern filtered: {', '.join(temporal_result['alerts'])}"]
                }
            
            # Confirmed patterns - upgrade to appropriate category
            if temporal_result['confirmed_alerts']:
                # Check severity based on confirmation level
                if len(temporal_result['confirmed_alerts']) >= 2 or temporal_result['has_confirmed_brady']:
                    category = max(category, 2)
                    category_name = 'Category II (Suspicious)'
                    overrides.extend(temporal_result['confirmed_alerts'])
        
        # ================================================================
        # ADDITIONAL SAFETY CHECKS (legacy rules for backup)
        # ================================================================
        
        # Severe bradycardia baseline (backup check)
        if baseline < self.critical_brady_bpm:
            return {
                'category': 3,
                'category_name': 'Category III (Pathological)',
                'overrides': [f'CRITICAL: Severe bradycardia baseline {baseline:.0f} bpm']
            }
        
        # Severe tachycardia (backup check)
        if baseline > self.critical_tachy_bpm:
            return {
                'category': 3,
                'category_name': 'Category III (Pathological)',
                'overrides': [f'CRITICAL: Severe tachycardia baseline {baseline:.0f} bpm']
            }
        
        # Critically absent variability (backup check)
        if variability <= self.absent_var_critical_threshold:
            pct_brady = np.sum(fhr < 110) / len(fhr) * 100 if len(fhr) > 0 else 0
            if pct_brady > 10:  # Absent var + brady = Category 3 per Position Paper
                return {
                    'category': 3,
                    'category_name': 'Category III (Pathological)',
                    'overrides': [f'CRITICAL: Absent variability {variability:.1f} bpm with bradycardia']
                }
        
        # Moderate bradycardia baseline (110-100 range)
        if baseline < self.brady_threshold:
            category = max(category, 2)
            if category == 2:
                category_name = 'Category II (Suspicious)'
            overrides.append(f'Bradycardia baseline {baseline:.0f} bpm')
        
        # Tachycardia (160-170 range)
        if baseline > self.tachy_threshold:
            category = max(category, 2)
            if category == 2:
                category_name = 'Category II (Suspicious)'
            overrides.append(f'Tachycardia baseline {baseline:.0f} bpm')
        
        # Absent/Minimal variability (without brady = Category II only)
        if variability < self.absent_var_threshold:
            category = max(category, 2)
            if category == 2:
                category_name = 'Category II (Suspicious)'
            overrides.append(f'Minimal variability {variability:.1f} bpm')
        
        # High variability - can indicate problems (NEW for sensitivity)
        elif self.high_var_enabled and variability > self.high_var_threshold:
            category = max(category, 2)
            if category == 2:
                category_name = 'Category II (Suspicious)'
            overrides.append(f'High variability {variability:.1f} bpm')
        
        # Reduced variability with baseline deviation
        elif variability < self.reduced_var_threshold:
            if baseline < 110 or baseline > 160:
                category = max(category, 2)
                if category == 2:
                    category_name = 'Category II (Suspicious)'
                overrides.append(f'Reduced variability {variability:.1f} bpm with baseline deviation')
        
        # Recurrent variable decels
        if self._detect_recurrent_variables(fhr, uc):
            category = max(category, 2)
            if category == 2:
                category_name = 'Category II (Suspicious)'
            overrides.append('Recurrent variable decelerations')
        
        # Reassuring signs - DISABLED for high sensitivity mode
        # In high sensitivity mode, we don't want to downgrade based on reassuring signs
        # because we might miss pathological cases that appear normal
        # 
        # if variability >= 6 and variability <= 25:
        #     if self._detect_accelerations(fhr):
        #         if ml_result['category'] == 1 and category <= 2 and 'CRITICAL' not in ' '.join(overrides):
        #             category = 1
        #             category_name = 'Category I (Normal)'
        #             overrides = ['Reassuring: good variability + accelerations']
        
        return {
            'category': category,
            'category_name': category_name,
            'overrides': overrides if overrides else ['ML classification']
        }
    
    def _temporal_confirmation_analysis(
        self, 
        fhr: np.ndarray, 
        sample_rate: int,
        baseline: float,
        variability: float
    ) -> Dict[str, Any]:
        """
        Analyze patterns with temporal confirmation.
        
        Based on:
        1. Israeli Position Paper (2023) timing guidelines
        2. CTU-CHB database pattern duration analysis
        
        Returns dict with:
        - is_critical: bool - requires immediate Category 3
        - is_likely_fp: bool - likely false positive, don't escalate
        - alerts: list - all detected alerts
        - confirmed_alerts: list - alerts that passed temporal confirmation
        - has_confirmed_brady: bool - has confirmed bradycardia episode
        - critical_reasons: list - reasons for critical flag
        """
        result = {
            'is_critical': False,
            'is_likely_fp': False,
            'alerts': [],
            'confirmed_alerts': [],
            'has_confirmed_brady': False,
            'critical_reasons': []
        }
        
        if len(fhr) < 60:  # Need at least 1 minute of data
            return result
        
        # === CALCULATE KEY METRICS ===
        pct_normal = np.sum((fhr >= 110) & (fhr <= 160)) / len(fhr) * 100
        pct_brady = np.sum(fhr < 110) / len(fhr) * 100
        
        # === BRADY EPISODE DETECTION ===
        brady_mask = fhr < 110
        in_brady = False
        start = 0
        brady_durations = []
        
        for i, is_brady in enumerate(brady_mask):
            if is_brady and not in_brady:
                in_brady, start = True, i
            elif not is_brady and in_brady:
                in_brady = False
                duration = (i - start) / sample_rate
                brady_durations.append(duration)
        
        max_brady_dur = max(brady_durations) if brady_durations else 0
        brady_confirmed = [d for d in brady_durations if d >= self.brady_confirmed_duration]
        brady_prolonged = [d for d in brady_durations if d >= self.brady_prolonged_duration]
        
        # === CRITICAL PATTERN CHECK ===
        
        # Prolonged bradycardia (>60 sec) - per Position Paper
        if len(brady_prolonged) >= 1:
            result['is_critical'] = True
            result['critical_reasons'].append(f'Prolonged deceleration: {max_brady_dur:.0f} sec at <110 bpm')
        
        # Severe baseline bradycardia
        if baseline < self.critical_brady_bpm:
            result['is_critical'] = True
            result['critical_reasons'].append(f'Severe baseline bradycardia: {baseline:.0f} bpm')
        
        # Severe tachycardia
        if baseline > self.critical_tachy_bpm:
            result['is_critical'] = True
            result['critical_reasons'].append(f'Severe tachycardia: {baseline:.0f} bpm')
        
        # Absent variability + bradycardia (Category 3 per Position Paper)
        if variability < self.critical_absent_var_brady and pct_brady > 10:
            result['is_critical'] = True
            result['critical_reasons'].append(f'Absent variability ({variability:.1f} bpm) with bradycardia ({pct_brady:.0f}%)')
        
        if result['is_critical']:
            return result  # Skip FP filter for critical cases
        
        # === BUILD ALERTS LIST ===
        if baseline < 110:
            result['alerts'].append('baseline_brady')
        if baseline > 160:
            result['alerts'].append('baseline_tachy')
        if variability < 5:
            result['alerts'].append('absent_var')
        elif variability < 10:
            result['alerts'].append('reduced_var')
        if pct_brady > 5:
            result['alerts'].append('brady_episodes')
        
        if len(result['alerts']) == 0:
            return result  # Normal - no alerts
        
        # === TEMPORAL CONFIRMATION ===
        confirmed_count = 0
        
        # Confirmed bradycardia (>30 sec episodes)
        if len(brady_confirmed) >= self.brady_recurrent_count:
            result['confirmed_alerts'].append(f'Recurrent confirmed bradycardia ({len(brady_confirmed)} episodes >{self.brady_confirmed_duration}s)')
            result['has_confirmed_brady'] = True
            confirmed_count += 1
        elif len(brady_confirmed) >= 1:
            result['confirmed_alerts'].append(f'Confirmed bradycardia episode ({max(brady_confirmed):.0f}s)')
            result['has_confirmed_brady'] = True
            confirmed_count += 1
        
        # Confirmed tachycardia (always confirms - usually real)
        if 'baseline_tachy' in result['alerts']:
            result['confirmed_alerts'].append(f'Tachycardia baseline {baseline:.0f} bpm')
            confirmed_count += 1
        
        # Confirmed absent variability (if not mostly normal)
        if 'absent_var' in result['alerts'] and pct_normal < 60:
            result['confirmed_alerts'].append(f'Absent variability {variability:.1f} bpm (sustained)')
            confirmed_count += 1
        
        # === FP FILTER ===
        # If high % normal AND all brady episodes are short → likely FP
        # BUT: Never filter cases with very low variability (absent variability is pathological!)
        if self.fp_filter_enabled and confirmed_count == 0:
            # Protect low-variability cases - absent variability is a serious finding
            if variability >= 5:  # Only apply filter if variability is NOT absent
                if pct_normal > self.fp_pct_normal_threshold and max_brady_dur < self.fp_max_brady_duration:
                    result['is_likely_fp'] = True
                    logger.debug(f"FP filter triggered: pct_normal={pct_normal:.1f}%, max_brady={max_brady_dur:.1f}s")
        
        return result
    
    def _detect_late_decels(self, fhr: np.ndarray, uc: np.ndarray) -> bool:
        """
        Detect late decelerations (FHR nadir after UC peak).
        
        More conservative detection to reduce false positives:
        - Requires actual correlation with UC peaks
        - Must see multiple episodes
        """
        if len(fhr) < 240 or len(uc) < 240:  # Need at least 4 minutes
            return False
        
        # Simple check: Need significant and consistent drops
        fhr_mean = np.mean(fhr)
        fhr_std = np.std(fhr)
        
        # Drops must be significant (>2 std below mean)
        significant_drops = fhr < (fhr_mean - 2 * fhr_std)
        
        # Must be substantial portion AND have variation (not flat signal)
        if fhr_std < 5:  # Very low variability - might be signal issue
            return False
            
        # Conservative: need 20% with significant drops AND baseline must be reasonable
        if fhr_mean < 100 or fhr_mean > 180:  # Baseline out of range
            return True  # This is pathological regardless
            
        return np.sum(significant_drops) > len(fhr) * 0.20
    
    def _detect_recurrent_variables(self, fhr: np.ndarray, uc: np.ndarray) -> bool:
        """Detect recurrent variable decelerations."""
        if len(fhr) < 300:  # Need at least 5 minutes
            return False
            
        # Count rapid drops (>15 bpm in <30 seconds)
        rapid_drops = 0
        window = 30  # samples
        
        for i in range(window, len(fhr)):
            if fhr[i-window] - fhr[i] > 15:
                rapid_drops += 1
        
        # Uses configurable threshold (default 5, was 3)
        return rapid_drops >= self.var_decel_min_count
    
    def _detect_accelerations(self, fhr: np.ndarray) -> bool:
        """Detect presence of accelerations (reassuring sign)."""
        if len(fhr) < 60:
            return False
            
        baseline = np.median(fhr)
        # Acceleration = >15 bpm above baseline for >15 seconds
        above_baseline = fhr > (baseline + 15)
        
        # Count consecutive segments above baseline
        consecutive = 0
        max_consecutive = 0
        
        for val in above_baseline:
            if val:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
                
        return max_consecutive >= 15  # At least 15 seconds of acceleration
    
    def _calculate_trend_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate deterioration trend score (0-100).
        
        Higher score = more concern for deterioration.
        """
        base_score = result['category'] * 30  # 0, 30, or 60
        
        # Add points for clinical overrides
        override_score = len(result.get('overrides', [])) * 10
        
        # Add points if trending worse
        if len(self.history) >= 3:
            recent_cats = [h['category'] for h in self.history[-3:]]
            if recent_cats[-1] > recent_cats[0]:
                override_score += 20  # Trending worse
        
        return min(100, base_score + override_score)
    
    def _get_trend_direction(self) -> str:
        """Get trend direction based on history."""
        if len(self.history) < 3:
            return 'stable'
            
        recent_scores = [h['trend_score'] for h in self.history[-5:]]
        
        if len(recent_scores) < 2:
            return 'stable'
            
        slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        if slope > 5:
            return 'deteriorating'
        elif slope < -5:
            return 'improving'
        else:
            return 'stable'
    
    def _generate_explanation(
        self,
        clinical_result: Dict[str, Any],
        ml_result: Dict[str, Any],
        baseline: float,
        variability: float
    ) -> Dict[str, Any]:
        """Generate human-readable explanation of the classification."""
        
        factors = []
        
        # Baseline assessment
        if baseline < 110:
            factors.append({
                'factor': 'Baseline FHR',
                'value': f'{baseline:.0f} bpm',
                'assessment': 'Below normal range (<110)',
                'concern': 'high'
            })
        elif baseline > 160:
            factors.append({
                'factor': 'Baseline FHR', 
                'value': f'{baseline:.0f} bpm',
                'assessment': 'Above normal range (>160)',
                'concern': 'medium'
            })
        else:
            factors.append({
                'factor': 'Baseline FHR',
                'value': f'{baseline:.0f} bpm', 
                'assessment': 'Within normal range',
                'concern': 'low'
            })
        
        # Variability assessment
        if variability < 5:
            factors.append({
                'factor': 'Variability',
                'value': f'{variability:.1f} bpm',
                'assessment': 'Minimal/Absent (<5 bpm)',
                'concern': 'high'
            })
        elif variability < 6:
            factors.append({
                'factor': 'Variability',
                'value': f'{variability:.1f} bpm',
                'assessment': 'Reduced',
                'concern': 'medium'
            })
        else:
            factors.append({
                'factor': 'Variability',
                'value': f'{variability:.1f} bpm',
                'assessment': 'Moderate (normal)',
                'concern': 'low'
            })
        
        # ML confidence
        confidence_level = 'high' if ml_result['confidence'] > 0.8 else 'medium' if ml_result['confidence'] > 0.6 else 'low'
        factors.append({
            'factor': 'AI Confidence',
            'value': f'{ml_result["confidence"]:.0%}',
            'assessment': f'{confidence_level.capitalize()} confidence in classification',
            'concern': 'low' if confidence_level == 'high' else 'medium'
        })
        
        # Clinical overrides
        for override in clinical_result.get('overrides', []):
            factors.append({
                'factor': 'Clinical Rule',
                'value': override,
                'assessment': 'Safety rule triggered',
                'concern': 'high' if 'Category III' in clinical_result['category_name'] else 'medium'
            })
        
        # Primary reason
        if clinical_result['category'] == 0:
            primary_reason = 'Normal FHR pattern with adequate variability'
        elif clinical_result['category'] == 1:
            primary_reason = 'Indeterminate pattern requiring continued monitoring'
        else:
            primary_reason = 'Abnormal pattern requiring immediate evaluation'
        
        return {
            'category': clinical_result['category'],
            'category_name': clinical_result['category_name'],
            'primary_reason': primary_reason,
            'factors': factors,
            'confidence': ml_result['confidence'],
            'recommendation': self._get_recommendation(clinical_result['category'])
        }
    
    def _get_recommendation(self, category: int) -> str:
        """Get clinical recommendation based on category."""
        recommendations = {
            0: 'Continue routine monitoring. No intervention required.',
            1: 'Increase monitoring frequency. Consider intrauterine resuscitation measures.',
            2: 'Immediate bedside evaluation required. Prepare for possible intervention.'
        }
        return recommendations.get(category, 'Continue monitoring.')
    
    def _update_history(self, result: Dict[str, Any]):
        """Update analysis history."""
        self.history.append({
            'category': result['category'],
            'trend_score': result['trend_score'],
            'confidence': result['confidence']
        })
        
        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]


# Singleton instance
_analyzer_instance: Optional[TrendAnalyzer] = None


def get_trend_analyzer() -> TrendAnalyzer:
    """Get or create singleton TrendAnalyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TrendAnalyzer()
    return _analyzer_instance


def analyze_ctg(
    fhr: np.ndarray,
    uc: np.ndarray,
    baseline: float = 140.0,
    variability: float = 10.0
) -> Dict[str, Any]:
    """
    Convenience function for CTG analysis.
    
    Args:
        fhr: FHR signal array
        uc: UC signal array
        baseline: Current baseline
        variability: Current variability
        
    Returns:
        Complete analysis result
    """
    return get_trend_analyzer().analyze(fhr, uc, baseline, variability)
