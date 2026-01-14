"""
SentinelFetal - Real-time Fetal Distress Detection System.

A hybrid AI system for CTG analysis combining:
- MOMENT foundation model for time series embedding
- XGBoost classifier for category prediction
- Rule-based analysis per Israeli Position Paper
- Medical Override safety net

Modules:
    config: Centralized configuration constants
    data: CTU-UHB dataset loading and preprocessing
    rules: Clinical rule implementations (baseline, variability, etc.)
    models: ML models (MOMENT encoder, XGBoost classifier)
    analysis: Alert generation and medical override
    training: Model training utilities
    ui: Streamlit dashboard and visualizations
    utils: Reusable utility functions

Quick Start:
    >>> from src.data import CTUDataLoader, CTGRecord
    >>> from src.rules import calculate_baseline, calculate_variability
    >>> from src.models import CTGClassifier
    >>> from src.analysis import generate_alert, apply_medical_override
    
    >>> loader = CTUDataLoader("data/path")
    >>> record = loader.load_record("1001")
    >>> baseline = calculate_baseline(record.fhr)
"""

__version__ = "3.5.0"
__author__ = "Ariel Shamay"

# Expose main configuration
from src.config import CTG, THRESHOLDS, COLORS, PATHS, MODEL

# Version info
VERSION_INFO = {
    'version': __version__,
    'phase': 6,
    'status': 'Production Ready',
    'model': 'XGBoost + MOMENT',
}

__all__ = [
    '__version__',
    '__author__',
    'VERSION_INFO',
    'CTG',
    'THRESHOLDS', 
    'COLORS',
    'PATHS',
    'MODEL',
]
