"""
Trend Analyzer - Combines XGBoost Classification with Clinical Rule Engine
=========================================================================
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

try:
    from .xgboost_classifier import get_classifier, XGBoostCTGClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    
logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Analyzes CTG trends and provides clinical categorization.
    
    Combines:
    1. XGBoost ML classification for pattern recognition
    2. Clinical rule engine for NICHD compliance
    3. Trend scoring for deterioration detection
    """
    
    def __init__(self):
        self.classifier = get_classifier() if CLASSIFIER_AVAILABLE else None
        self.history: List[Dict[str, Any]] = []
        self.max_history = 60  # Keep 60 analysis results (e.g., 60 minutes)
        
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
            Analysis result with category, trend, and explanation
        """
        # Get ML classification
        ml_result = self._get_ml_classification(fhr, uc)
        
        # Apply clinical rules (may override ML)
        clinical_result = self._apply_clinical_rules(
            fhr, uc, baseline, variability, ml_result
        )
        
        # Calculate trend score
        trend_score = self._calculate_trend_score(clinical_result)
        
        # Generate explanation
        explanation = self._generate_explanation(
            clinical_result, ml_result, baseline, variability
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
            'ml_category': ml_result['category'],
            'clinical_overrides': clinical_result.get('overrides', [])
        }
        
        # Store in history
        self._update_history(result)
        
        return result
    
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
        Apply NICHD clinical rules that may override ML classification.
        
        Clinical rules take precedence for safety-critical patterns.
        """
        overrides = []
        category = ml_result['category']
        category_name = ml_result['category_name']
        
        # Rule 1: Prolonged bradycardia -> Category III
        bradycardia_threshold = 110
        bradycardia_duration = 180  # 3 minutes at 1 Hz, adjust for sample rate
        
        if len(fhr) >= bradycardia_duration:
            recent_fhr = fhr[-bradycardia_duration:]
            if np.mean(recent_fhr) < bradycardia_threshold:
                category = 2
                category_name = 'Category III (Pathological)'
                overrides.append('Prolonged bradycardia detected')
        
        # Rule 2: Absent variability with late decels -> Category III
        if variability < 5:
            # Check for late decels (simplified)
            if self._detect_late_decels(fhr, uc):
                category = max(category, 2)
                category_name = 'Category III (Pathological)'
                overrides.append('Absent variability with late decelerations')
        
        # Rule 3: Recurrent variable decels -> at least Category II
        if self._detect_recurrent_variables(fhr, uc):
            category = max(category, 1)
            if category == 1:
                category_name = 'Category II (Indeterminate)'
            overrides.append('Recurrent variable decelerations')
        
        # Rule 4: Good variability + accelerations = reassuring (stay Cat I)
        if variability >= 6 and self._detect_accelerations(fhr) and category == 0:
            overrides.append('Reassuring pattern: good variability with accelerations')
        
        return {
            'category': category,
            'category_name': category_name,
            'overrides': overrides
        }
    
    def _detect_late_decels(self, fhr: np.ndarray, uc: np.ndarray) -> bool:
        """Detect late decelerations (FHR nadir after UC peak)."""
        if len(fhr) < 120 or len(uc) < 120:
            return False
            
        # Simplified: check if FHR drops significantly after UC peaks
        fhr_mean = np.mean(fhr)
        drops = fhr < (fhr_mean - 20)
        
        return np.sum(drops) > len(fhr) * 0.1  # >10% of trace shows drops
    
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
                
        return rapid_drops >= 3  # At least 3 variable decels
    
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
