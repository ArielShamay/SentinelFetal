"""
UI Package for SentinelFetal Dashboard.

This package contains the Streamlit dashboard and visualization utilities.

Components:
    - app.py: Main Streamlit application
    - plots.py: CTG visualization with WebGL Plotly
    - styles.py: Professional CSS styling (hides Streamlit chrome)
    - simulation_app.py: Real-time simulation dashboard
"""

from pathlib import Path

UI_DIR = Path(__file__).parent

# Export styles module
from .styles import (
    inject_custom_css,
    COLORS,
    category_banner_html,
    finding_card_html,
    recommendation_html,
    section_header_html,
)

__all__ = [
    'UI_DIR',
    'inject_custom_css',
    'COLORS',
    'category_banner_html',
    'finding_card_html',
    'recommendation_html',
    'section_header_html',
]
