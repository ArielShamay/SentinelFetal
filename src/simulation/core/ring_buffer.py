"""
Ring Buffer - Fixed-size circular buffer for CTG signals.

Purpose: Store last 10 minutes of data per patient without memory growth.
Memory: ~77KB for 8 patients (2400 samples × 8 × 4 bytes × 2 channels)

Uses collections.deque for O(1) append and automatic size limiting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np

from src.config import CTG


@dataclass
class RingBuffer:
    """
    Fixed-size circular buffer for CTG signals.
    
    Stores FHR, UC, and timestamp data in synchronized deques that
    automatically discard oldest samples when full.
    
    Attributes:
        max_samples: Maximum samples to store (default: 2400 = 10 min at 4Hz).
        sampling_rate: Sampling frequency in Hz (default: from config).
        
    Example:
        >>> buffer = RingBuffer()
        >>> buffer.append(140.0, 20.0, 0.0)
        >>> buffer.append(142.0, 22.0, 0.25)
        >>> data = buffer.get_window(duration_seconds=60)
        >>> print(data['fhr'].shape)  # (240,) for 60 seconds at 4Hz
    """
    
    max_samples: int = CTG.MOMENT_WINDOW_SAMPLES  # 2400 = 10 min at 4Hz
    sampling_rate: float = CTG.SAMPLING_RATE  # 4.0 Hz
    
    # Private deques - initialized in __post_init__
    _fhr: deque = field(default_factory=deque, repr=False)
    _uc: deque = field(default_factory=deque, repr=False)
    _timestamps: deque = field(default_factory=deque, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize deques with correct maxlen after dataclass init."""
        self._fhr = deque(maxlen=self.max_samples)
        self._uc = deque(maxlen=self.max_samples)
        self._timestamps = deque(maxlen=self.max_samples)
    
    def append(self, fhr: float, uc: float, timestamp: float) -> None:
        """
        Add a single sample to the buffer.
        
        Args:
            fhr: Fetal heart rate value in bpm.
            uc: Uterine contraction value (0-100 scale).
            timestamp: Time in seconds from simulation start.
        """
        self._fhr.append(fhr)
        self._uc.append(uc)
        self._timestamps.append(timestamp)
    
    def append_batch(
        self, 
        fhr: np.ndarray, 
        uc: np.ndarray, 
        timestamps: np.ndarray
    ) -> None:
        """
        Add multiple samples to the buffer.
        
        Args:
            fhr: Array of FHR values.
            uc: Array of UC values.
            timestamps: Array of timestamp values.
        """
        for f, u, t in zip(fhr, uc, timestamps):
            self.append(float(f), float(u), float(t))
    
    def get_window(self, duration_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Get data from the buffer.
        
        Args:
            duration_seconds: Seconds of data to return. If None, returns all data.
            
        Returns:
            Dictionary with keys:
                - 'fhr': numpy array of FHR values
                - 'uc': numpy array of UC values
                - 'timestamps': numpy array of timestamps
                - 'duration_seconds': actual duration of returned data
        """
        if duration_seconds is None:
            samples = len(self._fhr)
        else:
            samples = min(
                int(duration_seconds * self.sampling_rate),
                len(self._fhr)
            )
        
        # Get last N samples
        fhr_list = list(self._fhr)[-samples:] if samples > 0 else []
        uc_list = list(self._uc)[-samples:] if samples > 0 else []
        ts_list = list(self._timestamps)[-samples:] if samples > 0 else []
        
        return {
            'fhr': np.array(fhr_list, dtype=np.float64),
            'uc': np.array(uc_list, dtype=np.float64),
            'timestamps': np.array(ts_list, dtype=np.float64),
            'duration_seconds': len(fhr_list) / self.sampling_rate if fhr_list else 0.0
        }
    
    def get_last_n_minutes(self, minutes: float) -> Dict[str, Any]:
        """
        Convenience method for getting last N minutes of data.
        
        Args:
            minutes: Number of minutes of data to return.
            
        Returns:
            Same as get_window().
        """
        return self.get_window(minutes * 60)
    
    def get_latest(self) -> Dict[str, Optional[float]]:
        """
        Get the most recent sample.
        
        Returns:
            Dictionary with 'fhr', 'uc', 'timestamp' (None if buffer empty).
        """
        if len(self._fhr) == 0:
            return {'fhr': None, 'uc': None, 'timestamp': None}
        
        return {
            'fhr': self._fhr[-1],
            'uc': self._uc[-1],
            'timestamp': self._timestamps[-1]
        }
    
    def clear(self) -> None:
        """Clear all data from buffer."""
        self._fhr.clear()
        self._uc.clear()
        self._timestamps.clear()
    
    @property
    def size(self) -> int:
        """Current number of samples in buffer."""
        return len(self._fhr)
    
    @property
    def duration_seconds(self) -> float:
        """Current duration of data in buffer (seconds)."""
        return self.size / self.sampling_rate
    
    @property
    def duration_minutes(self) -> float:
        """Current duration of data in buffer (minutes)."""
        return self.duration_seconds / 60.0
    
    @property
    def is_full(self) -> bool:
        """Whether buffer has reached max capacity."""
        return self.size >= self.max_samples
    
    @property
    def is_empty(self) -> bool:
        """Whether buffer has no data."""
        return self.size == 0
    
    def __len__(self) -> int:
        """Return number of samples in buffer."""
        return self.size
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RingBuffer(size={self.size}/{self.max_samples}, "
            f"duration={self.duration_seconds:.1f}s)"
        )
