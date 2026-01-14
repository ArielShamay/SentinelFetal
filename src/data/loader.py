"""
CTU-UHB Dataset Loader.

Loads CTG recordings from the CTU-UHB Intrapartum Cardiotocography Database.
Dataset available at: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

This module provides:
    - CTGRecord: Dataclass for storing a single CTG recording
    - CTUDataLoader: Class for loading records from the CTU-UHB database

Example:
    >>> loader = CTUDataLoader("data/ctu-uhb")
    >>> record = loader.load_record("1001")
    >>> print(f"FHR duration: {record.duration_seconds} seconds")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional, Union

import numpy as np
import pandas as pd
import wfdb

# Configure module logger
logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Base exception for data loader errors."""
    pass


class RecordNotFoundError(DataLoaderError):
    """Raised when a requested record cannot be found."""
    pass


class InvalidRecordError(DataLoaderError):
    """Raised when a record has invalid or missing data."""
    pass


@dataclass
class CTGRecord:
    """
    Container for a single CTG recording.
    
    Attributes:
        record_id: Unique identifier for the record (e.g., '1001').
        fhr1: Primary fetal heart rate signal as numpy array.
        fhr2: Secondary FHR signal (optional, may be None).
        uc: Uterine contractions signal as numpy array.
        sampling_rate: Sampling frequency in Hz (typically 4 Hz).
        duration_seconds: Total duration of recording in seconds.
        metadata: Additional record information (signal names, units, etc.).
        
    Example:
        >>> record = loader.load_record("1001")
        >>> print(f"Duration: {record.duration_seconds / 60:.1f} minutes")
        >>> print(f"Mean FHR: {np.nanmean(record.fhr1):.1f} bpm")
    """
    
    record_id: str
    fhr1: np.ndarray
    fhr2: Optional[np.ndarray]
    uc: np.ndarray
    sampling_rate: float
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def timestamps(self) -> np.ndarray:
        """
        Generate timestamp array in seconds.
        
        Returns:
            numpy.ndarray: Array of timestamps from 0 to duration.
        """
        n_samples = len(self.fhr1)
        return np.arange(n_samples) / self.sampling_rate
    
    @property
    def n_samples(self) -> int:
        """Return the number of samples in the recording."""
        return len(self.fhr1)
    
    def __repr__(self) -> str:
        """Return string representation of the record."""
        return (
            f"CTGRecord(id={self.record_id}, "
            f"duration={self.duration_seconds/60:.1f}min, "
            f"samples={self.n_samples})"
        )


class CTUDataLoader:
    """
    Loader for CTU-UHB Intrapartum CTG Database.
    
    The CTU-UHB database contains 552 intrapartum recordings with:
        - FHR1: Primary fetal heart rate (4 Hz)
        - FHR2: Secondary FHR from second transducer (4 Hz)
        - UC: Uterine contractions (4 Hz)
    
    Attributes:
        SAMPLING_RATE: Default sampling rate (4 Hz).
        data_dir: Path to the dataset directory.
        
    Example:
        >>> loader = CTUDataLoader("data/ctu-uhb")
        >>> records = loader.list_records()
        >>> print(f"Found {len(records)} records")
        >>> record = loader.load_record(records[0])
        
    Raises:
        DataLoaderError: If the data directory is invalid.
    """
    
    SAMPLING_RATE: int = 4  # Hz
    
    def __init__(self, data_dir: Union[str, Path]) -> None:
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the CTU-UHB dataset directory.
            
        Raises:
            DataLoaderError: If the directory does not exist.
        """
        self.data_dir = Path(data_dir)
        self._validate_directory()
        
    def _validate_directory(self) -> None:
        """
        Validate that the data directory exists and contains .hea files.
        
        Raises:
            DataLoaderError: If directory doesn't exist or has no records.
        """
        if not self.data_dir.exists():
            raise DataLoaderError(
                f"Data directory not found: {self.data_dir}\n"
                "Please download the CTU-UHB dataset from:\n"
                "https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/"
            )
        
        # Check if there are any .hea files
        hea_files = list(self.data_dir.glob("*.hea"))
        if not hea_files:
            logger.warning(f"No .hea files found in {self.data_dir}")
    
    def list_records(self) -> list[str]:
        """
        List all available record IDs in the dataset.
        
        Returns:
            Sorted list of record IDs (e.g., ['1001', '1002', ...]).
            
        Example:
            >>> loader = CTUDataLoader("data/ctu-uhb")
            >>> records = loader.list_records()
            >>> print(f"Found {len(records)} records")
        """
        records: list[str] = []
        for hea_file in self.data_dir.glob("*.hea"):
            records.append(hea_file.stem)
        return sorted(records)
    
    def load_record(self, record_id: str) -> CTGRecord:
        """
        Load a single CTG record from the dataset.
        
        Args:
            record_id: The record identifier (e.g., '1001').
            
        Returns:
            CTGRecord containing all signals and metadata.
            
        Raises:
            RecordNotFoundError: If the record file doesn't exist.
            InvalidRecordError: If the record has no FHR signal.
            
        Example:
            >>> record = loader.load_record("1001")
            >>> print(f"FHR mean: {np.nanmean(record.fhr1):.1f} bpm")
        """
        record_path = self.data_dir / record_id
        
        # Check if record exists
        if not (self.data_dir / f"{record_id}.hea").exists():
            raise RecordNotFoundError(
                f"Record '{record_id}' not found in {self.data_dir}"
            )
        
        try:
            # Load using wfdb
            record = wfdb.rdrecord(str(record_path))
        except Exception as e:
            raise InvalidRecordError(
                f"Failed to load record '{record_id}': {e}"
            ) from e
        
        # Extract signals
        signals = record.p_signal
        sig_names = [name.upper() for name in record.sig_name]
        
        # Map signals to our structure
        fhr1 = self._extract_signal(signals, sig_names, 'FHR')
        fhr2 = self._extract_signal(signals, sig_names, 'FHR2')
        uc = self._extract_signal(signals, sig_names, 'UC')
        
        # If no specific FHR column, try alternatives
        if fhr1 is None:
            fhr1 = self._extract_signal(signals, sig_names, 'FHR1')
        
        if fhr1 is None:
            raise InvalidRecordError(
                f"No FHR signal found in record '{record_id}'. "
                f"Available signals: {record.sig_name}"
            )
        
        # Build metadata
        metadata: dict[str, Any] = {
            'n_signals': record.n_sig,
            'signal_names': record.sig_name,
            'units': record.units,
            'comments': getattr(record, 'comments', []),
        }
        
        duration = len(fhr1) / self.SAMPLING_RATE
        
        logger.debug(f"Loaded record {record_id}: {duration/60:.1f} minutes")
        
        return CTGRecord(
            record_id=record_id,
            fhr1=fhr1,
            fhr2=fhr2,
            uc=uc if uc is not None else np.zeros_like(fhr1),
            sampling_rate=float(self.SAMPLING_RATE),
            duration_seconds=duration,
            metadata=metadata
        )
    
    def _extract_signal(
        self, 
        signals: np.ndarray, 
        sig_names: list[str], 
        target: str
    ) -> Optional[np.ndarray]:
        """
        Extract a specific signal by name from the signals matrix.
        
        Args:
            signals: 2D array of signals [n_samples, n_channels].
            sig_names: List of signal names (uppercase).
            target: Target signal name to extract.
            
        Returns:
            Signal array if found, None otherwise.
        """
        target = target.upper()
        for i, name in enumerate(sig_names):
            if target in name.upper():
                return signals[:, i].copy()
        return None
    
    def load_record_as_dataframe(self, record_id: str) -> pd.DataFrame:
        """
        Load a record as a pandas DataFrame.
        
        Args:
            record_id: The record identifier.
            
        Returns:
            DataFrame with columns: timestamp, fhr1, uc, and optionally fhr2.
            
        Example:
            >>> df = loader.load_record_as_dataframe("1001")
            >>> print(df.head())
        """
        record = self.load_record(record_id)
        
        df = pd.DataFrame({
            'timestamp': record.timestamps,
            'fhr1': record.fhr1,
            'uc': record.uc
        })
        
        if record.fhr2 is not None:
            df['fhr2'] = record.fhr2
            
        return df
    
    def iter_records(
        self, 
        limit: Optional[int] = None
    ) -> Generator[CTGRecord, None, None]:
        """
        Iterate over all records in the dataset.
        
        Args:
            limit: Maximum number of records to load (None for all).
            
        Yields:
            CTGRecord for each recording.
            
        Example:
            >>> for record in loader.iter_records(limit=10):
            ...     print(f"Processing {record.record_id}")
        """
        records = self.list_records()
        if limit:
            records = records[:limit]
            
        for record_id in records:
            try:
                yield self.load_record(record_id)
            except DataLoaderError as e:
                logger.warning(f"Failed to load record {record_id}: {e}")
                continue

    def extract_ph(self, record_id: str) -> Optional[float]:
        """
        Extract pH value from a record's .hea file comments.
        
        The pH value is stored in the comments section of the .hea file
        in the format: '#pH           7.14'
        
        Args:
            record_id: The record identifier (e.g., '1001').
            
        Returns:
            pH value as float, or None if not found.
            
        Example:
            >>> loader = CTUDataLoader("data/ctu-uhb")
            >>> ph = loader.extract_ph("1001")
            >>> print(f"pH: {ph}")  # pH: 7.14
        """
        hea_path = self.data_dir / f"{record_id}.hea"
        
        if not hea_path.exists():
            logger.warning(f"Header file not found for record {record_id}")
            return None
        
        try:
            with open(hea_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#pH'):
                        # Parse line like '#pH           7.14'
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                ph_value = float(parts[1])
                                return ph_value
                            except ValueError:
                                logger.warning(
                                    f"Invalid pH value in record {record_id}: {parts[1]}"
                                )
                                return None
        except Exception as e:
            logger.warning(f"Error reading pH for record {record_id}: {e}")
            return None
        
        return None

    def get_outcome_label(self, record_id: str) -> int:
        """
        Get outcome label based on pH value per Israeli Position Paper.
        
        Label mapping (for 3-class classification):
            - pH < 7.15 → Label 2 (Category 3 - Pathological)
            - 7.15 ≤ pH < 7.20 → Label 1 (Category 2 - Intermediate)
            - pH ≥ 7.20 (or unknown) → Label 0 (Category 1 - Normal)
        
        Args:
            record_id: The record identifier.
            
        Returns:
            Integer label: 0 (Normal), 1 (Intermediate), or 2 (Pathological).
            
        Example:
            >>> label = loader.get_outcome_label("1001")
            >>> print(f"Label: {label}")  # Label: 2 (pH was 7.14)
        """
        ph = self.extract_ph(record_id)
        
        if ph is None:
            # Unknown pH → assume Normal (Category 1)
            return 0
        
        if ph < 7.15:
            # Pathological (Category 3)
            return 2
        elif ph < 7.20:
            # Intermediate (Category 2)
            return 1
        else:
            # Normal (Category 1)
            return 0

    def get_all_ph_values(self) -> dict[str, Optional[float]]:
        """
        Extract pH values for all records in the dataset.
        
        Returns:
            Dictionary mapping record_id to pH value (or None if unavailable).
            
        Example:
            >>> ph_values = loader.get_all_ph_values()
            >>> known = {k: v for k, v in ph_values.items() if v is not None}
            >>> print(f"Found pH for {len(known)} records")
        """
        ph_values: dict[str, Optional[float]] = {}
        
        for record_id in self.list_records():
            ph_values[record_id] = self.extract_ph(record_id)
        
        return ph_values
