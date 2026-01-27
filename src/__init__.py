"""
SentinelFetal core package (V6 Pre-AI safe).

This module intentionally avoids importing optional/AI dependencies at import time.
Only lightweight configuration symbols are exported.
"""

__version__ = "3.6.0"
__author__ = "Ariel Shamay"

from src.config import CTG, THRESHOLDS, COLORS, PATHS, MODEL

__all__ = [
    "__version__",
    "__author__",
    "CTG",
    "THRESHOLDS",
    "COLORS",
    "PATHS",
    "MODEL",
]
