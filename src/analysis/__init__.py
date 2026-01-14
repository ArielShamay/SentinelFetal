"""
Analysis module for SentinelFetal.

This module provides:
    - Alert generation with Hebrew XAI explanations
    - Medical Override safety net logic

Usage:
    >>> from src.analysis import generate_alert, apply_medical_override
    >>> alert = generate_alert(category=3, confidence=0.95, ...)
    >>> override = apply_medical_override(ml_prediction=0, ...)
"""

from .override import apply_medical_override, MedicalOverride, OverrideReason
from .alerts import generate_alert, Alert, get_category_emoji, get_category_color

__all__ = [
    # Medical Override
    'apply_medical_override',
    'MedicalOverride',
    'OverrideReason',
    # Alerts
    'generate_alert',
    'Alert',
    'get_category_emoji',
    'get_category_color',
]
