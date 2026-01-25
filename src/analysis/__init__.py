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

__all__ = [
    "apply_medical_override",
    "MedicalOverride",
    "OverrideReason",
    "generate_alert",
    "Alert",
    "get_category_emoji",
    "get_category_color",
]


def __getattr__(name):
    if name in {"apply_medical_override", "MedicalOverride", "OverrideReason"}:
        from .override import apply_medical_override, MedicalOverride, OverrideReason
        return {
            "apply_medical_override": apply_medical_override,
            "MedicalOverride": MedicalOverride,
            "OverrideReason": OverrideReason,
        }[name]
    if name in {"generate_alert", "Alert", "get_category_emoji", "get_category_color"}:
        from .alerts import generate_alert, Alert, get_category_emoji, get_category_color
        return {
            "generate_alert": generate_alert,
            "Alert": Alert,
            "get_category_emoji": get_category_emoji,
            "get_category_color": get_category_color,
        }[name]
    raise AttributeError(name)
