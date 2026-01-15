"""
Type aliases and common types used across the system.

These type aliases provide semantic meaning and documentation
for the various array types used throughout SentinelFetal.
"""

from typing import TypeVar, Generic, Callable, Any
import numpy as np

# Generic type variables
T = TypeVar('T')
ResultT = TypeVar('ResultT')

# Type aliases for clarity
FHRSignal = np.ndarray      # Shape: (n_samples,), dtype: float, Fetal Heart Rate signal
UCSignal = np.ndarray       # Shape: (n_samples,), dtype: float, Uterine Contraction signal
EmbeddingVector = np.ndarray  # Shape: (1024,), dtype: float32, MOMENT embedding
FeatureVector = np.ndarray  # Shape: (1035,), dtype: float32, Fused feature vector
CategoryLabel = int         # 0, 1, or 2 (Category 1, 2, or 3)

# Sampling rate constant
DEFAULT_SAMPLING_RATE = 4.0  # Hz
