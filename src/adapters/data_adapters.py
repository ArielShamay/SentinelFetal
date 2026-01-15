"""
Adapters for data components.
These wrap existing data loading and preprocessing to conform to Protocol interfaces.
"""

from typing import List, Optional
import numpy as np

from src.interfaces.protocols import (
    IDataLoader,
    ICTGRecord,
    IPreprocessor,
    IPreprocessingResult,
)

from src.data.loader import CTUDataLoader, CTGRecord
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig


class DataLoaderAdapter(IDataLoader):
    """
    Adapter for data loading.
    
    Wraps CTUDataLoader to conform to IDataLoader interface.
    
    Args:
        data_path: Path to the CTU-UHB dataset directory.
        
    Example:
        >>> adapter = DataLoaderAdapter("data/ctu-uhb")
        >>> records = adapter.list_records()
        >>> record = adapter.load_record("1001")
        >>> print(f"Loaded {record.record_id}")
    """
    
    def __init__(self, data_path: Optional[str] = None):
        self._loader = CTUDataLoader(data_path) if data_path else CTUDataLoader()
    
    def list_records(self) -> List[str]:
        """List available record IDs."""
        return self._loader.list_records()
    
    def load_record(self, record_id: str) -> ICTGRecord:
        """Load a single record by ID."""
        return self._loader.load_record(record_id)
    
    def get_outcome_label(self, record_id: str) -> int:
        """Get outcome label (0/1/2) for a record."""
        return self._loader.get_outcome_label(record_id)


class PreprocessorAdapter(IPreprocessor):
    """
    Adapter for signal preprocessing.
    
    Wraps CTGPreprocessor to conform to IPreprocessor interface.
    
    Args:
        config: Preprocessing configuration. Uses defaults if None.
        
    Example:
        >>> adapter = PreprocessorAdapter()
        >>> result = adapter.process(fhr_signal)
        >>> clean_fhr = result.processed_signal
        >>> print(f"Filled {result.stats['filled_percent']:.1f}% of gaps")
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self._preprocessor = CTGPreprocessor(config or PreprocessingConfig())
    
    def process(self, fhr: np.ndarray, apply_smoothing: bool = False) -> IPreprocessingResult:
        """Process FHR signal using existing implementation."""
        return self._preprocessor.process(fhr, apply_smoothing=apply_smoothing)
