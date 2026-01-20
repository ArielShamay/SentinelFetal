# -*- coding: utf-8 -*-
"""
Alert Engine for SentinelFetal.

This module generates explanatory alerts in Hebrew for CTG monitoring.
All text follows the Israeli Position Paper terminology (Section 8).

References:
    SentinelFetal Gen3.5 Technical Specification, Section 8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union

from src.rules.decelerations import Deceleration, DecelerationType
from src.rules.variability import VariabilityResult, VariabilityCategory
from src.rules.baseline import BaselineResult
from src.rules.sinusoidal import SinusoidalResult
from src.rules.tachysystole import TachysystoleResult


@dataclass
class Alert:
    """
    מייצג התראה מלאה עם הסבר.
    
    Represents a complete alert with explanation in Hebrew.
    
    Attributes:
        category: Classification category (1, 2, or 3).
        confidence: Model confidence (0-1).
        headline: Short headline describing the alert.
        explanation: Detailed explanation in Hebrew.
        findings: List of medical findings supporting the decision.
        recommendations: List of recommendations for the medical team.
        timestamp: ISO format timestamp of alert generation.
    """
    category: int
    confidence: float
    headline: str
    explanation: str
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate category is 1, 2, or 3."""
        if self.category not in (1, 2, 3):
            raise ValueError(f"Category must be 1, 2, or 3, got {self.category}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


def _build_cat3_explanation(
    findings: List[str],
    sinusoidal: Union[dict, SinusoidalResult],
    variability: Union[dict, VariabilityResult]
) -> str:
    """
    Build detailed explanation for Category 3 (Pathological).
    
    Args:
        findings: List of medical findings.
        sinusoidal: Sinusoidal detection result.
        variability: Variability analysis result.
        
    Returns:
        Hebrew explanation string.
    """
    # Check sinusoidal
    if isinstance(sinusoidal, SinusoidalResult):
        is_sinusoidal = sinusoidal.detected
    else:
        is_sinusoidal = sinusoidal.get('detected', False)
    
    # Check variability
    if isinstance(variability, VariabilityResult):
        is_absent = variability.category == VariabilityCategory.ABSENT
    else:
        is_absent = variability.get('category', '') in ('Absent', 'ABSENT')
    
    if is_sinusoidal:
        return (
            "זוהתה תבנית סינוסואידלית - מצב זה מחייב הערכה דחופה. "
            "תבנית סינוסואידלית אמיתית קשורה לאנמיה עוברית חמורה או לאסון רפואי מתקרב. "
            "יש להתייעץ עם רופא בכיר באופן מיידי."
        )
    elif is_absent:
        return (
            "זוהה היעדר שונות בשילוב עם ממצאים מדאיגים נוספים. "
            "היעדר שונות מעיד על חוסר יכולת העובר להגיב לשינויים - מצב המעיד על מצוקה. "
            "נדרשת הערכה מיידית ושקילת התערבות."
        )
    else:
        return (
            "שילוב הממצאים מעיד על מצב פתולוגי הדורש התערבות מיידית. "
            "יש לבצע הערכה דחופה של מצב היולדת והעובר."
        )


def _build_cat2_explanation(findings: List[str]) -> str:
    """
    Build detailed explanation for Category 2 (Intermediate).
    
    Args:
        findings: List of medical findings.
        
    Returns:
        Hebrew explanation string.
    """
    decel_findings = [f for f in findings if 'האטות' in f or 'האטה' in f]
    var_findings = [f for f in findings if 'שונות' in f]
    
    if decel_findings and var_findings:
        return (
            "זוהו שינויים בדופק העובר הדורשים מעקב צמוד. "
            "השילוב של שינויים בשונות עם האטות מחייב ערנות מוגברת. "
            "יש להמשיך במעקב ולשקול התערבויות שמרניות."
        )
    elif decel_findings:
        return (
            "זוהו האטות בדופק העובר הדורשות מעקב. "
            "יש לעקוב אחר תדירות האירועים ומשך ההתאוששות."
        )
    elif var_findings:
        return (
            "השונות אינה בטווח התקין. "
            "יש להמשיך לעקוב ולהעריך שינויים לאורך זמן."
        )
    else:
        return (
            "ישנם ממצאים הדורשים מעקב צמוד. "
            "יש להמשיך לעקוב ולהעריך את ההתפתחות."
        )


def generate_alert(
    category: int,
    confidence: float,
    baseline: Union[float, BaselineResult],
    variability: Union[dict, VariabilityResult],
    decelerations: List[Deceleration],
    tachysystole: Union[dict, TachysystoleResult],
    sinusoidal: Union[dict, SinusoidalResult]
) -> Alert:
    """
    מייצר התראה מוסברת.
    
    Generates an explanatory alert with Hebrew text.
    
    Args:
        category: Classification category (1, 2, or 3).
        confidence: Model confidence (0-1).
        baseline: Baseline FHR value or BaselineResult.
        variability: Variability result (dict or VariabilityResult).
        decelerations: List of detected decelerations.
        tachysystole: Tachysystole detection result.
        sinusoidal: Sinusoidal pattern detection result.
        
    Returns:
        Alert object with Hebrew explanations.
        
    Example:
        >>> alert = generate_alert(
        ...     category=3,
        ...     confidence=0.92,
        ...     baseline=90,
        ...     variability={'value': 3, 'category': 'Absent'},
        ...     decelerations=[...],
        ...     tachysystole={'detected': False},
        ...     sinusoidal={'detected': True}
        ... )
        >>> print(alert.headline)
        'התראה אדומה - קטגוריה 3 (פתולוגי)'
        
    References:
        SentinelFetal Gen3.5 Technical Specification, Section 8.2
    """
    findings: List[str] = []
    recommendations: List[str] = []
    
    # === Extract values from dataclasses or dicts ===
    
    # Baseline
    if isinstance(baseline, BaselineResult):
        baseline_value = baseline.value
    else:
        baseline_value = float(baseline)
    
    # Variability
    if isinstance(variability, VariabilityResult):
        var_val = variability.value
        var_cat = variability.category.name if isinstance(variability.category, VariabilityCategory) else str(variability.category)
    else:
        var_val = variability.get('value', 0)
        var_cat = variability.get('category', 'Unknown')
        if isinstance(var_cat, VariabilityCategory):
            var_cat = var_cat.name
    
    # Tachysystole
    if isinstance(tachysystole, TachysystoleResult):
        tachy_detected = tachysystole.detected
        tachy_rate = tachysystole.contractions_per_10min
    else:
        tachy_detected = tachysystole.get('detected', False)
        tachy_rate = tachysystole.get('contractions_per_10min', 0)
    
    # Sinusoidal
    if isinstance(sinusoidal, SinusoidalResult):
        sinus_detected = sinusoidal.detected
    else:
        sinus_detected = sinusoidal.get('detected', False)
    
    # === איסוף ממצאים ===
    
    # Baseline findings
    if baseline_value < 110:
        findings.append(f'ברדיקרדיה - קצב בסיסי {baseline_value:.0f} bpm')
    elif baseline_value > 160:
        findings.append(f'טכיקרדיה - קצב בסיסי {baseline_value:.0f} bpm')
    else:
        findings.append(f'קצב בסיסי תקין: {baseline_value:.0f} bpm')
    
    # Variability findings
    var_cat_upper = var_cat.upper() if isinstance(var_cat, str) else str(var_cat)
    if var_cat_upper == 'ABSENT':
        findings.append(f'היעדר שונות ({var_val:.1f} bpm) - ממצא חמור!')
    elif var_cat_upper == 'MINIMAL':
        findings.append(f'שונות מזערית ({var_val:.1f} bpm) - דורש מעקב')
    elif var_cat_upper == 'MODERATE':
        findings.append(f'שונות תקינה ({var_val:.1f} bpm)')
    elif var_cat_upper == 'MARKED':
        findings.append(f'שונות מוגברת ({var_val:.1f} bpm)')
    else:
        findings.append(f'שונות: {var_val:.1f} bpm ({var_cat})')
    
    # Deceleration findings
    late_count = sum(1 for d in decelerations 
                    if d.decel_type == DecelerationType.LATE)
    variable_count = sum(1 for d in decelerations 
                        if d.decel_type == DecelerationType.VARIABLE)
    prolonged_count = sum(1 for d in decelerations 
                         if d.decel_type == DecelerationType.PROLONGED)
    early_count = sum(1 for d in decelerations 
                     if d.decel_type == DecelerationType.EARLY)
    
    if late_count > 0:
        findings.append(f'זוהו {late_count} האטות מאוחרות')
    if variable_count > 0:
        severe = sum(1 for d in decelerations 
                    if d.decel_type == DecelerationType.VARIABLE 
                    and d.has_severity_signs)
        if severe > 0:
            findings.append(f'זוהו {variable_count} האטות משתנות ({severe} עם סימני חומרה)')
        else:
            findings.append(f'זוהו {variable_count} האטות משתנות')
    if prolonged_count > 0:
        findings.append(f'זוהו {prolonged_count} האטות ממושכות')
    if early_count > 0:
        findings.append(f'זוהו {early_count} האטות מוקדמות (בד"כ שפירות)')
    
    # Tachysystole findings
    if tachy_detected:
        findings.append(f'Tachysystole - {tachy_rate:.1f} צירים/10 דקות')
    
    # Sinusoidal findings
    if sinus_detected:
        findings.append('זוהתה תבנית סינוסואידלית - ממצא חמור!')
    
    # === יצירת הסברים והמלצות לפי קטגוריה ===
    
    if category == 3:
        headline = 'התראה אדומה - קטגוריה 3 (פתולוגי)'
        explanation = _build_cat3_explanation(findings, sinusoidal, variability)
        recommendations = [
            'הערכה מיידית של הסיבות האפשריות',
            'שקילת החייאה תוך-רחמית',
            'היערכות ליילוד מיידי אם אין שיפור'
        ]
    elif category == 2:
        headline = 'התראה כתומה - קטגוריה 2 (ביניים)'
        explanation = _build_cat2_explanation(findings)
        recommendations = [
            'המשך ניטור צמוד',
            'שינוי תנוחת היולדת',
            'בדיקת לחץ דם והתייבשות'
        ]
    else:  # category == 1
        headline = 'סטטוס ירוק - קטגוריה 1 (תקין)'
        explanation = 'כל הפרמטרים בטווח התקין.'
        recommendations = ['המשך ניטור שגרתי']
    
    return Alert(
        category=category,
        confidence=confidence,
        headline=headline,
        explanation=explanation,
        findings=findings,
        recommendations=recommendations,
        timestamp=datetime.now().isoformat()
    )


def get_category_color(category: int) -> str:
    """
    Get the display color for a category.
    
    Args:
        category: Classification category (1, 2, or 3).
        
    Returns:
        Color string (green/orange/red).
    """
    colors = {1: 'green', 2: 'orange', 3: 'red'}
    return colors.get(category, 'gray')


def get_category_emoji(category: int) -> str:
    """
    Get the emoji indicator for a category.
    
    Args:
        category: Classification category (1, 2, or 3).
        
    Returns:
        Emoji string.
    """
    emojis = {1: '🟢', 2: '🟠', 3: '🔴'}
    return emojis.get(category, '⚪')
