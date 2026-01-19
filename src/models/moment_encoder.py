"""
MOMENT Feature Extractor Module.

PRODUCTION MODULE - NO MOCK/DEMO FALLBACKS

Implements the MOMENT foundation model integration for CTG embedding extraction.

MOMENT (Multi-task Originator for Multivariate Time-series) is a Transformer model
trained on millions of time series from diverse domains. It can identify:
    - Anomalies (sudden changes)
    - Trends (gradual increases/decreases)
    - Periodicity (repeating patterns)
    - Variance changes

Zero-Shot Mode:
    Gen3.5 uses MOMENT without fine-tuning. The model acts as a feature extractor,
    outputting 1024-dimensional embeddings that capture the signal's characteristics.

V2 Architecture (STRICT Production):
    The module supports optimized inference backends:
    1. ONNX/OpenVINO optimized inference (preferred, ~4-10x faster)
    2. PyTorch MOMENT (fallback if ONNX model not available)
    
    NO MOCK MODE - If no backend is available, the system raises an error.
    
    Use get_moment_encoder() factory function to automatically select
    the best available backend.

Technical Specifications:
    - Model: AutonLab/MOMENT-1-large
    - Parameters: 385 million
    - Embedding dimension: 1024
    - Patch size: 64 samples (16 seconds at 4Hz)
    - Maximum input: 512 patches = 8,192 samples (~34 minutes)
    - VRAM requirements: ~2GB (inference only)
    - Inference time: ~100-200ms per 10-minute window on CPU (ONNX)
                      ~2-5s per window (PyTorch)

References:
    - SentinelFetal Gen3.5 Technical Specification, Section 4
    - MOMENT Paper: https://arxiv.org/abs/2402.03885
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

# Check for optimized ONNX backend availability
ONNX_BACKEND_AVAILABLE = False
try:
    from .moment_onnx import (
        MomentONNXPredictor,
        ModelNotFoundError,
        InferenceError,
        extract_embeddings_sliding_window as onnx_extract_sliding
    )
    ONNX_BACKEND_AVAILABLE = True
    logger.info("ONNX backend available - will use optimized inference")
except ImportError:
    logger.debug("ONNX backend not available")

# Check for MOMENT availability
MOMENT_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    
    # Check device availability
    if torch.cuda.is_available():
        DEFAULT_DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEFAULT_DEVICE = "mps"
    else:
        DEFAULT_DEVICE = "cpu"
        
    logger.info(f"PyTorch available. Default device: {DEFAULT_DEVICE}")
    
except ImportError:
    DEFAULT_DEVICE = "cpu"
    logger.warning("PyTorch not available.")

try:
    from momentfm import MOMENTPipeline
    MOMENT_AVAILABLE = True
    logger.info("momentfm package available. Using real MOMENT model.")
except ImportError:
    logger.warning(
        "momentfm package not available. "
        "Install with: pip install momentfm"
    )


class MomentEncoderError(Exception):
    """Raised when MOMENT encoding fails."""
    pass


@dataclass
class EmbeddingResult:
    """
    Result of embedding extraction for a single window.
    
    Attributes:
        embedding: 1024-dimensional embedding vector.
        start_idx: Start index in the original signal.
        end_idx: End index in the original signal.
        start_time_sec: Start time in seconds.
        end_time_sec: End time in seconds.
        backend: Backend used for extraction (e.g., 'onnxruntime-int8', 'pytorch').
    """
    
    embedding: np.ndarray
    start_idx: int
    end_idx: int
    start_time_sec: float
    end_time_sec: float
    backend: str = "unknown"
    
    def __post_init__(self) -> None:
        """Validate embedding dimension."""
        if self.embedding.shape != (1024,):
            raise ValueError(
                f"Embedding must be 1024-dimensional, got {self.embedding.shape}"
            )
    
    def __repr__(self) -> str:
        return (
            f"EmbeddingResult(backend='{self.backend}', "
            f"time={self.start_time_sec:.1f}-{self.end_time_sec:.1f}s)"
        )


class MomentFeatureExtractor:
    """
    Feature extractor using MOMENT foundation model (PyTorch backend).
    
    PRODUCTION VERSION - NO MOCK MODE
    
    Extracts 1024-dimensional embeddings from CTG signals using the
    AutonLab/MOMENT-1-large model in Zero-Shot mode.
    
    If MOMENT is not available (momentfm package not installed), raises
    MomentEncoderError instead of falling back to mock data.
    
    Attributes:
        EMBEDDING_DIM: Output embedding dimension (1024).
        DEFAULT_WINDOW_SIZE: Default window size in samples (2400 = 10 min @ 4Hz).
        device: Device for computation (cpu/cuda/mps).
        
    Raises:
        MomentEncoderError: If MOMENT model cannot be loaded.
        
    Example:
        >>> extractor = MomentFeatureExtractor(device='cuda')
        >>> embedding = extractor.extract(fhr_window)
        >>> print(f"Embedding shape: {embedding.shape}")  # (1024,)
        
    References:
        SentinelFetal Gen3.5 Technical Specification, Section 4.4
    """
    
    EMBEDDING_DIM: int = 1024
    DEFAULT_WINDOW_SIZE: int = 2400  # 10 minutes @ 4Hz
    
    def __init__(
        self, 
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the MOMENT feature extractor.
        
        Args:
            device: Device for computation. Options: 'cpu', 'cuda', 'mps'.
                    If None, auto-detects best available device.
                     
        Raises:
            MomentEncoderError: If MOMENT model cannot be loaded.
        """
        if not MOMENT_AVAILABLE:
            raise MomentEncoderError(
                "MOMENT model not available. Install with: pip install momentfm"
            )
        if not TORCH_AVAILABLE:
            raise MomentEncoderError(
                "PyTorch not available. Install with: pip install torch"
            )
            
        self.device = device or DEFAULT_DEVICE
        self._model = None
        
        self._load_model()
            
    def _load_model(self) -> None:
        """
        Load the MOMENT model from HuggingFace.
        
        Uses task_name='embedding' for Zero-Shot feature extraction.
        
        Raises:
            MomentEncoderError: If model loading fails.
        """
        logger.info("Loading MOMENT-1-large model...")
        
        try:
            self._model = MOMENTPipeline.from_pretrained(
                'AutonLab/MOMENT-1-large',
                model_kwargs={
                    'task_name': 'embedding',  # Zero-Shot embedding mode
                    'n_channels': 1  # Single channel (FHR)
                }
            )
            self._model.to(self.device)
            self._model.eval()  # Inference mode
            
            logger.info(
                f"MOMENT model loaded successfully on {self.device}"
            )
            
        except Exception as e:
            raise MomentEncoderError(f"Failed to load MOMENT model: {e}")
    
    def extract(
        self, 
        fhr: np.ndarray,
        window_size: int = None
    ) -> np.ndarray:
        """
        Extract embedding from FHR window.
        
        Args:
            fhr: FHR signal array (1D numpy array in bpm).
            window_size: Expected window size. If signal is shorter, it's padded.
                        If longer, it's truncated. Default: 2400 (10 min @ 4Hz).
                        
        Returns:
            1024-dimensional embedding vector.
            
        Raises:
            MomentEncoderError: If extraction fails.
            
        Example:
            >>> embedding = extractor.extract(fhr_window)
            >>> assert embedding.shape == (1024,)
        """
        window_size = window_size or self.DEFAULT_WINDOW_SIZE
        
        # Validate input
        if fhr is None or len(fhr) == 0:
            raise MomentEncoderError("Input FHR signal is empty or None")
        
        # Prepare signal
        fhr_prepared = self._prepare_signal(fhr, window_size)
        
        return self._model_extract(fhr_prepared)
    
    def _prepare_signal(
        self, 
        fhr: np.ndarray, 
        window_size: int
    ) -> np.ndarray:
        """
        Prepare FHR signal for MOMENT input.
        
        Steps:
            1. Pad or truncate to window_size
            2. Replace NaN with zeros
            3. Normalize (zero mean, unit variance)
            
        Args:
            fhr: Raw FHR signal.
            window_size: Target length.
            
        Returns:
            Prepared signal array.
        """
        # Handle length
        if len(fhr) < window_size:
            # Pad with the mean (or 0 if all NaN)
            valid_values = fhr[~np.isnan(fhr)]
            pad_value = float(np.mean(valid_values)) if len(valid_values) > 0 else 0.0
            fhr = np.pad(
                fhr, 
                (0, window_size - len(fhr)), 
                mode='constant', 
                constant_values=pad_value
            )
        elif len(fhr) > window_size:
            fhr = fhr[:window_size]
        
        # Replace NaN with 0 (MOMENT doesn't handle NaN)
        fhr = np.nan_to_num(fhr, nan=0.0)
        
        # Normalize (zero mean, unit variance)
        mean = np.mean(fhr)
        std = np.std(fhr)
        if std > 1e-8:
            fhr = (fhr - mean) / std
        else:
            fhr = fhr - mean
            
        return fhr.astype(np.float32)
    
    def _model_extract(self, fhr: np.ndarray) -> np.ndarray:
        """
        Extract embedding using the real MOMENT model.
        
        Args:
            fhr: Prepared FHR signal.
            
        Returns:
            1024-dimensional embedding.
        """
        # Convert to tensor: [batch=1, channels=1, seq_len]
        x = torch.tensor(fhr, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        x = x.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self._model.embed(x_enc=x)
            embedding = output.embeddings  # [1, 1024]
        
        return embedding.cpu().numpy().flatten()
    
    def get_backend(self) -> str:
        """Get the backend name."""
        return "pytorch"


def extract_embeddings_sliding_window(
    fhr: np.ndarray,
    extractor: MomentFeatureExtractor,
    sampling_rate: float = 4.0,
    window_minutes: float = 10.0,
    step_minutes: float = 1.0
) -> List[EmbeddingResult]:
    """
    Extract embeddings using sliding window over entire recording.
    
    Processes the complete FHR recording using overlapping windows to capture
    temporal evolution of the signal.
    
    Args:
        fhr: Complete FHR signal array.
        extractor: MomentFeatureExtractor instance.
        sampling_rate: Sampling frequency in Hz (default: 4.0).
        window_minutes: Window size in minutes (default: 10.0).
        step_minutes: Step size in minutes (default: 1.0).
        
    Returns:
        List of EmbeddingResult objects, one per window.
        
    Example:
        >>> extractor = MomentFeatureExtractor()
        >>> embeddings = extract_embeddings_sliding_window(fhr, extractor)
        >>> for result in embeddings:
        ...     print(f"Window {result.start_time_sec/60:.1f}-{result.end_time_sec/60:.1f} min")
        
    References:
        SentinelFetal Gen3.5 Technical Specification, Section 4.5
    """
    window_samples = int(window_minutes * 60 * sampling_rate)  # e.g., 10 * 60 * 4 = 2400
    step_samples = int(step_minutes * 60 * sampling_rate)      # e.g., 1 * 60 * 4 = 240
    
    results: List[EmbeddingResult] = []
    n_samples = len(fhr)
    
    # Check if signal is long enough
    if n_samples < window_samples:
        logger.warning(
            f"Signal too short ({n_samples} samples) for window ({window_samples}). "
            "Processing entire signal as single window."
        )
        embedding = extractor.extract(fhr, window_samples)
        backend = extractor.get_backend() if hasattr(extractor, 'get_backend') else "unknown"
        results.append(EmbeddingResult(
            embedding=embedding,
            start_idx=0,
            end_idx=n_samples,
            start_time_sec=0.0,
            end_time_sec=n_samples / sampling_rate,
            backend=backend
        ))
        return results
    
    # Sliding window extraction
    window_count = 0
    backend = extractor.get_backend() if hasattr(extractor, 'get_backend') else "unknown"
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = fhr[start:end]
        
        try:
            embedding = extractor.extract(window)
            
            results.append(EmbeddingResult(
                embedding=embedding,
                start_idx=start,
                end_idx=end,
                start_time_sec=start / sampling_rate,
                end_time_sec=end / sampling_rate,
                backend=backend
            ))
            
            window_count += 1
            
            if window_count % 10 == 0:
                logger.debug(
                    f"Processed {window_count} windows "
                    f"(time: {start/sampling_rate/60:.1f} min)"
                )
                
        except Exception as e:
            logger.warning(f"Failed to extract embedding for window {start}-{end}: {e}")
            continue
    
    logger.info(
        f"Extracted {len(results)} embeddings from {n_samples/sampling_rate/60:.1f} min recording"
    )
    
    return results


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device availability information.
    """
    info = {
        'torch_available': TORCH_AVAILABLE,
        'moment_available': MOMENT_AVAILABLE,
        'onnx_backend_available': ONNX_BACKEND_AVAILABLE,
        'default_device': DEFAULT_DEVICE,
        'cuda_available': False,
        'mps_available': False,
        'cuda_device_count': 0,
    }
    
    if TORCH_AVAILABLE:
        info['cuda_available'] = torch.cuda.is_available()
        info['mps_available'] = (
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()
        )
        if info['cuda_available']:
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
    
    return info


# ============================================================================
# Factory Function for Best Available Encoder (STRICT - NO MOCK)
# ============================================================================

def get_moment_encoder(
    prefer_onnx: bool = True,
    device: Optional[str] = None
) -> Union['MomentFeatureExtractor', 'MomentONNXPredictor']:
    """
    Factory function to get the best available MOMENT encoder.
    
    STRICT PRODUCTION MODE - NO MOCK FALLBACK
    
    This function implements the V2 architecture with automatic
    backend selection:
    
    Priority (if prefer_onnx=True):
        1. MomentONNXPredictor (ONNX/OpenVINO - fastest)
        2. MomentFeatureExtractor (PyTorch - compatible)
    
    If no backend is available, raises ModelNotFoundError.
    
    Args:
        prefer_onnx: Whether to prefer ONNX backend. Default True.
        device: Device for PyTorch backend ('cpu', 'cuda', 'mps').
        
    Returns:
        Best available encoder instance.
        
    Raises:
        ModelNotFoundError: If no MOMENT backend is available.
        
    Example:
        >>> encoder = get_moment_encoder()
        >>> print(f"Using backend: {encoder.backend if hasattr(encoder, 'backend') else 'pytorch'}")
        >>> embedding = encoder.extract(fhr_window)
    """
    errors = []
    
    # Try ONNX backend first (fastest)
    if prefer_onnx and ONNX_BACKEND_AVAILABLE:
        try:
            encoder = MomentONNXPredictor()
            logger.info(f"Using optimized ONNX encoder (backend: {encoder.backend})")
            return encoder
        except ModelNotFoundError as e:
            errors.append(f"ONNX: {e}")
        except Exception as e:
            errors.append(f"ONNX: {e}")
    
    # Try PyTorch MOMENT
    if TORCH_AVAILABLE and MOMENT_AVAILABLE:
        try:
            encoder = MomentFeatureExtractor(device=device)
            logger.info("Using PyTorch MOMENT encoder")
            return encoder
        except MomentEncoderError as e:
            errors.append(f"PyTorch: {e}")
        except Exception as e:
            errors.append(f"PyTorch: {e}")
    
    # No backend available - raise error (NO MOCK FALLBACK)
    error_details = "\n".join(errors) if errors else "No backends available"
    
    raise ModelNotFoundError(
        f"No MOMENT encoder backend available.\n"
        f"Errors:\n{error_details}\n\n"
        f"To fix:\n"
        f"1. Export MOMENT to ONNX: python scripts/export_moment_onnx.py\n"
        f"2. Or install momentfm: pip install momentfm torch"
    )


def get_encoder_info() -> dict:
    """
    Get information about available encoder backends.
    
    Returns:
        Dictionary with encoder availability and recommended backend.
    """
    info = {
        'onnx_available': ONNX_BACKEND_AVAILABLE,
        'pytorch_available': TORCH_AVAILABLE and MOMENT_AVAILABLE,
        'recommended_backend': None,
        'ready': False
    }
    
    if ONNX_BACKEND_AVAILABLE:
        try:
            encoder = MomentONNXPredictor()
            info['recommended_backend'] = encoder.backend
            info['onnx_backend_info'] = encoder.get_backend_info()
            info['ready'] = True
        except ModelNotFoundError:
            pass
        except Exception:
            pass
    
    if info['recommended_backend'] is None and info['pytorch_available']:
        info['recommended_backend'] = 'pytorch'
        info['ready'] = True
    
    return info
