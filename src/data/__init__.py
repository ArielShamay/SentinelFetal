"""
Data loading and preprocessing modules for SentinelFetal.

Modules:
    loader: CTU-UHB dataset loading (CTGRecord, CTUDataLoader)
    preprocess: Signal preprocessing (CTGPreprocessor)

Usage:
    >>> from src.data import CTUDataLoader, CTGRecord, CTGPreprocessor
    >>> loader = CTUDataLoader("data/path")
    >>> record = loader.load_record("1001")
    >>> preprocessor = CTGPreprocessor()
    >>> fhr_clean = preprocessor.preprocess(record.fhr)
"""

from .loader import CTUDataLoader, CTGRecord
from .preprocess import CTGPreprocessor, PreprocessingConfig, PreprocessingResult

__all__ = [
    "CTUDataLoader",
    "CTGRecord",
    "CTGPreprocessor",
    "PreprocessingConfig",
    "PreprocessingResult",
]
