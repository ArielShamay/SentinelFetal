"""
MOMENT Feature Extractor with ONNX Runtime / OpenVINO optimization.

PRODUCTION MODULE - NO MOCK/DEMO FALLBACKS

This module provides an optimized MOMENT encoder for production deployment.
It requires either ONNX Runtime or OpenVINO with a valid model file.

Backend Priority:
    1. OpenVINO (best performance on Intel CPUs)
    2. ONNX Runtime INT8 (cross-platform, quantized)
    3. ONNX Runtime FP32 (cross-platform, full precision)
    4. PyTorch MOMENT (original implementation)

If no backend can be loaded, the system raises ModelNotFoundError.

Performance Targets:
    - Inference time: < 500ms on Intel i5
    - Memory usage: < 500MB
    - Accuracy: > 98% of original FP32

Usage:
    >>> from src.models.moment_onnx import MomentONNXPredictor
    >>> encoder = MomentONNXPredictor()  # Auto-selects best backend
    >>> embedding = encoder.extract(fhr_signal)  # 1024-dim vector

References:
    - SentinelFetal Gen3.5 Technical Specification, Section 4
    - ONNX Runtime: https://onnxruntime.ai/
    - OpenVINO: https://docs.openvino.ai/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class ModelNotFoundError(Exception):
    """Raised when no MOMENT model can be loaded."""
    pass


class InferenceError(Exception):
    """Raised when model inference fails."""
    pass


# ============================================================================
# Backend Availability Detection
# ============================================================================

OPENVINO_AVAILABLE = False
ONNX_AVAILABLE = False
PYTORCH_AVAILABLE = False
MOMENT_AVAILABLE = False

try:
    from openvino import Core, CompiledModel
    OPENVINO_AVAILABLE = True
    logger.debug("OpenVINO available")
except ImportError:
    pass

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.debug("ONNX Runtime available")
except ImportError:
    pass

try:
    import torch
    PYTORCH_AVAILABLE = True
    logger.debug("PyTorch available")
except ImportError:
    pass

try:
    from momentfm import MOMENTPipeline
    MOMENT_AVAILABLE = True
    logger.debug("momentfm available")
except ImportError:
    pass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MomentONNXConfig:
    """
    Configuration for the optimized MOMENT encoder.
    
    Attributes:
        model_dir: Directory containing model files.
        onnx_filename: ONNX model filename (FP32).
        onnx_int8_filename: Quantized ONNX model filename (INT8).
        openvino_dir: OpenVINO IR model subdirectory.
        embedding_dim: Output embedding dimension.
        default_window_samples: Default input window size (10 min @ 4Hz).
        sampling_rate: Signal sampling rate in Hz.
        device: Inference device ('CPU', 'GPU', 'AUTO').
        prefer_quantized: Whether to prefer INT8 over FP32.
        num_threads: Number of inference threads.
    """
    model_dir: str = "models"
    onnx_filename: str = "moment.onnx"
    onnx_int8_filename: str = "moment_int8.onnx"
    openvino_dir: str = "moment_openvino"
    embedding_dim: int = 1024
    default_window_samples: int = 2400  # 10 min * 60 sec * 4 Hz
    sampling_rate: float = 4.0
    device: str = "CPU"
    prefer_quantized: bool = True
    num_threads: int = 4

    @property
    def model_path_onnx(self) -> Path:
        """Path to FP32 ONNX model."""
        return Path(self.model_dir) / self.onnx_filename
    
    @property
    def model_path_onnx_int8(self) -> Path:
        """Path to INT8 quantized ONNX model."""
        return Path(self.model_dir) / self.onnx_int8_filename
    
    @property
    def model_path_openvino(self) -> Path:
        """Path to OpenVINO IR directory."""
        return Path(self.model_dir) / self.openvino_dir


# ============================================================================
# Embedding Result
# ============================================================================

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
        backend: Backend used for extraction.
    """
    embedding: np.ndarray
    start_idx: int = 0
    end_idx: int = 0
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    backend: str = "unknown"
    
    def __post_init__(self):
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


# ============================================================================
# Main Predictor Class (PRODUCTION - NO MOCK)
# ============================================================================

class MomentONNXPredictor:
    """
    Optimized MOMENT encoder using ONNX Runtime or OpenVINO.
    
    PRODUCTION VERSION - NO MOCK MODE
    
    This class requires a valid model to be loaded. If no model is available,
    it raises ModelNotFoundError rather than falling back to mock data.
    
    Backend Priority:
        1. OpenVINO (best for Intel CPUs)
        2. ONNX Runtime INT8 (quantized, fast)
        3. ONNX Runtime FP32 (full precision)
        4. PyTorch MOMENT (original, slowest)
    
    Attributes:
        EMBEDDING_DIM: Output dimension (1024).
        backend: Active backend name.
        config: Model configuration.
    
    Example:
        >>> encoder = MomentONNXPredictor()
        >>> embedding = encoder.extract(fhr_window)
        >>> print(f"Backend: {encoder.backend}")
        >>> print(f"Shape: {embedding.shape}")  # (1024,)
        
    Raises:
        ModelNotFoundError: If no model can be loaded.
    """
    
    EMBEDDING_DIM = 1024
    DEFAULT_WINDOW_SIZE = 2400
    
    def __init__(
        self,
        config: Optional[MomentONNXConfig] = None,
        force_backend: Optional[str] = None
    ):
        """
        Initialize the optimized MOMENT encoder.
        
        Args:
            config: Model configuration. Uses defaults if None.
            force_backend: Force a specific backend. Options:
                - 'openvino': Use OpenVINO only
                - 'onnx': Use ONNX Runtime only
                - 'pytorch': Use PyTorch MOMENT only
                - None: Auto-detect best available
                
        Raises:
            ModelNotFoundError: If no model can be loaded with available backends.
        """
        self.config = config or MomentONNXConfig()
        self.backend = None
        
        # Internal model references
        self._ov_model = None
        self._onnx_session = None
        self._pytorch_model = None
        self._pytorch_device = "cpu"
        
        # Load model with best backend (strict - no mock)
        self._load_model_strict(force_backend)
    
    def _load_model_strict(self, force_backend: Optional[str] = None) -> None:
        """
        Load model using the best available backend.
        
        STRICT MODE: Raises error if no model can be loaded.
        
        Priority: OpenVINO > ONNX-INT8 > ONNX-FP32 > PyTorch
        
        Raises:
            ModelNotFoundError: If no backend succeeds.
        """
        errors = []
        
        # Try OpenVINO first (best for Intel)
        if force_backend in (None, "openvino") and OPENVINO_AVAILABLE:
            try:
                if self._try_load_openvino():
                    return
            except Exception as e:
                errors.append(f"OpenVINO: {e}")
        
        # Try ONNX Runtime
        if force_backend in (None, "onnx") and ONNX_AVAILABLE:
            try:
                if self._try_load_onnx():
                    return
            except Exception as e:
                errors.append(f"ONNX Runtime: {e}")
        
        # Try PyTorch fallback
        if force_backend in (None, "pytorch") and PYTORCH_AVAILABLE and MOMENT_AVAILABLE:
            try:
                if self._try_load_pytorch():
                    return
            except Exception as e:
                errors.append(f"PyTorch: {e}")
        
        # No backend succeeded - raise error (NO MOCK FALLBACK)
        available_backends = []
        if OPENVINO_AVAILABLE:
            available_backends.append("OpenVINO")
        if ONNX_AVAILABLE:
            available_backends.append("ONNX Runtime")
        if PYTORCH_AVAILABLE and MOMENT_AVAILABLE:
            available_backends.append("PyTorch+momentfm")
        
        error_details = "\n".join(errors) if errors else "No backends available"
        
        raise ModelNotFoundError(
            f"MOMENT model could not be loaded. "
            f"Available backends: {available_backends or 'None'}. "
            f"Errors:\n{error_details}\n\n"
            f"To fix:\n"
            f"1. Export MOMENT to ONNX: python scripts/export_moment_onnx.py\n"
            f"2. Or install momentfm: pip install momentfm\n"
            f"3. Or install OpenVINO: pip install openvino"
        )
    
    def _try_load_openvino(self) -> bool:
        """Try to load OpenVINO model."""
        ov_path = self.config.model_path_openvino
        model_xml = ov_path / "moment.xml"
        
        if not model_xml.exists():
            logger.debug(f"OpenVINO model not found at {model_xml}")
            return False
        
        logger.info(f"Loading OpenVINO model from {ov_path}")
        
        core = Core()
        model = core.read_model(str(model_xml))
        
        # Compile with latency optimization
        self._ov_model = core.compile_model(
            model,
            device_name=self.config.device,
            config={
                'PERFORMANCE_HINT': 'LATENCY',
                'NUM_STREAMS': '1',
                'INFERENCE_PRECISION_HINT': 'f32'
            }
        )
        
        self.backend = "openvino"
        logger.info("✓ OpenVINO model loaded successfully")
        return True
    
    def _try_load_onnx(self) -> bool:
        """Try to load ONNX Runtime model (prefer INT8)."""
        # Try INT8 first if preferred
        if self.config.prefer_quantized:
            paths_to_try = [
                (self.config.model_path_onnx_int8, "onnxruntime-int8"),
                (self.config.model_path_onnx, "onnxruntime-fp32"),
            ]
        else:
            paths_to_try = [
                (self.config.model_path_onnx, "onnxruntime-fp32"),
                (self.config.model_path_onnx_int8, "onnxruntime-int8"),
            ]
        
        for model_path, backend_name in paths_to_try:
            if not model_path.exists():
                logger.debug(f"ONNX model not found at {model_path}")
                continue
            
            logger.info(f"Loading ONNX model from {model_path}")
            
            # Session options
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.num_threads
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            
            self._onnx_session = ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
            self.backend = backend_name
            logger.info(f"✓ ONNX model loaded successfully ({backend_name})")
            return True
        
        return False
    
    def _try_load_pytorch(self) -> bool:
        """Try to load PyTorch MOMENT model."""
        logger.info("Loading PyTorch MOMENT model...")
        
        pipeline = MOMENTPipeline.from_pretrained(
            'AutonLab/MOMENT-1-large',
            model_kwargs={
                'task_name': 'embedding',
                'n_channels': 1
            }
        )
        
        self._pytorch_model = pipeline.model
        self._pytorch_model.eval()
        
        # Move to appropriate device
        self._pytorch_device = "cpu"
        if torch.cuda.is_available():
            self._pytorch_device = "cuda"
        self._pytorch_model.to(self._pytorch_device)
        
        self.backend = "pytorch"
        logger.info(f"✓ PyTorch MOMENT loaded on {self._pytorch_device}")
        return True
    
    def extract(
        self,
        fhr: np.ndarray,
        normalize: bool = True,
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract 1024-dimensional embedding from FHR signal.
        
        Args:
            fhr: FHR signal array (1D, values in bpm).
            normalize: Whether to normalize input (zero mean, unit variance).
            window_size: Target window size. Uses default if None.
            
        Returns:
            1024-dimensional embedding vector as numpy array.
            
        Raises:
            InferenceError: If extraction fails.
            ValueError: If input is invalid.
        """
        if fhr is None or len(fhr) == 0:
            raise ValueError("Input FHR signal is empty or None")
        
        if self.backend is None:
            raise InferenceError("No model loaded - cannot extract embeddings")
        
        window_size = window_size or self.config.default_window_samples
        
        # Prepare input
        fhr_prepared = self._prepare_input(fhr, normalize, window_size)
        
        try:
            # Run inference with appropriate backend
            if self.backend == "openvino":
                return self._extract_openvino(fhr_prepared)
            elif self.backend.startswith("onnxruntime"):
                return self._extract_onnx(fhr_prepared)
            elif self.backend == "pytorch":
                return self._extract_pytorch(fhr_prepared)
            else:
                raise InferenceError(f"Unknown backend: {self.backend}")
        except Exception as e:
            raise InferenceError(f"Inference failed ({self.backend}): {e}")
    
    def _prepare_input(
        self,
        fhr: np.ndarray,
        normalize: bool,
        window_size: int
    ) -> np.ndarray:
        """
        Prepare FHR signal for model input.
        
        Steps:
            1. Pad or truncate to window_size
            2. Replace NaN with zeros
            3. Normalize (optional)
            4. Reshape to [1, 1, seq_len]
        """
        # Handle length
        if len(fhr) < window_size:
            # Pad with mean value
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
        
        # Replace NaN with 0
        fhr = np.nan_to_num(fhr, nan=0.0)
        
        # Normalize (zero mean, unit variance)
        if normalize:
            mean = np.mean(fhr)
            std = np.std(fhr)
            if std > 1e-8:
                fhr = (fhr - mean) / std
            else:
                fhr = fhr - mean
        
        # Reshape: [batch=1, channels=1, seq_len]
        return fhr.reshape(1, 1, -1).astype(np.float32)
    
    def _extract_openvino(self, fhr: np.ndarray) -> np.ndarray:
        """Extract embedding using OpenVINO."""
        result = self._ov_model([fhr])
        embedding = result[0].flatten()
        return embedding[:self.EMBEDDING_DIM]
    
    def _extract_onnx(self, fhr: np.ndarray) -> np.ndarray:
        """Extract embedding using ONNX Runtime."""
        output = self._onnx_session.run(None, {'fhr_input': fhr})
        embedding = output[0].flatten()
        return embedding[:self.EMBEDDING_DIM]
    
    def _extract_pytorch(self, fhr: np.ndarray) -> np.ndarray:
        """Extract embedding using PyTorch."""
        x = torch.tensor(fhr).to(self._pytorch_device)
        
        with torch.no_grad():
            output = self._pytorch_model(x)
            
            if hasattr(output, 'embeddings'):
                embedding = output.embeddings.cpu().numpy()
            else:
                embedding = output.cpu().numpy()
        
        return embedding.flatten()[:self.EMBEDDING_DIM]
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self.backend is not None
    
    def get_backend_info(self) -> dict:
        """Get information about the active backend."""
        info = {
            'backend': self.backend,
            'embedding_dim': self.EMBEDDING_DIM,
            'window_samples': self.config.default_window_samples,
            'window_minutes': self.config.default_window_samples / (60 * self.config.sampling_rate),
            'ready': self.backend is not None
        }
        
        if self.backend == "openvino":
            info['model_path'] = str(self.config.model_path_openvino)
        elif self.backend and self.backend.startswith("onnxruntime"):
            if "int8" in self.backend:
                info['model_path'] = str(self.config.model_path_onnx_int8)
                info['quantized'] = True
            else:
                info['model_path'] = str(self.config.model_path_onnx)
                info['quantized'] = False
        elif self.backend == "pytorch":
            info['model_path'] = "AutonLab/MOMENT-1-large (HuggingFace)"
        
        return info


# ============================================================================
# Convenience Functions (API Compatibility)
# ============================================================================

def extract_embeddings_sliding_window(
    fhr: np.ndarray,
    extractor: MomentONNXPredictor,
    sampling_rate: float = 4.0,
    window_minutes: float = 10.0,
    step_minutes: float = 1.0
) -> List[EmbeddingResult]:
    """
    Extract embeddings using sliding window approach.
    
    Args:
        fhr: Full FHR signal array.
        extractor: MomentONNXPredictor instance.
        sampling_rate: Signal sampling rate in Hz (default: 4.0).
        window_minutes: Window size in minutes (default: 10.0).
        step_minutes: Step size in minutes (default: 1.0).
        
    Returns:
        List of EmbeddingResult objects, one per window.
        
    Raises:
        InferenceError: If extraction fails.
    """
    window_samples = int(window_minutes * 60 * sampling_rate)
    step_samples = int(step_minutes * 60 * sampling_rate)
    
    results = []
    n_samples = len(fhr)
    
    # Handle short signals
    if n_samples < window_samples:
        embedding = extractor.extract(fhr, window_size=window_samples)
        results.append(EmbeddingResult(
            embedding=embedding,
            start_idx=0,
            end_idx=n_samples,
            start_time_sec=0.0,
            end_time_sec=n_samples / sampling_rate,
            backend=extractor.backend
        ))
        return results
    
    # Sliding window
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = fhr[start:end]
        
        embedding = extractor.extract(window)
        
        results.append(EmbeddingResult(
            embedding=embedding,
            start_idx=start,
            end_idx=end,
            start_time_sec=start / sampling_rate,
            end_time_sec=end / sampling_rate,
            backend=extractor.backend
        ))
    
    return results


# ============================================================================
# OpenVINO Conversion Utility
# ============================================================================

def convert_onnx_to_openvino(
    onnx_path: str,
    output_dir: str,
    compress_to_fp16: bool = False
) -> Path:
    """
    Convert ONNX model to OpenVINO IR format.
    
    Args:
        onnx_path: Path to ONNX model.
        output_dir: Directory to save OpenVINO IR files.
        compress_to_fp16: Whether to compress to FP16.
        
    Returns:
        Path to output directory.
        
    Raises:
        ImportError: If OpenVINO is not available.
    """
    if not OPENVINO_AVAILABLE:
        raise ImportError(
            "OpenVINO is required. Install with: pip install openvino"
        )
    
    from openvino import save_model
    
    logger.info(f"Converting {onnx_path} to OpenVINO IR...")
    
    core = Core()
    model = core.read_model(onnx_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as IR
    save_model(
        model,
        str(output_path / "moment.xml"),
        compress_to_fp16=compress_to_fp16
    )
    
    logger.info(f"✓ OpenVINO IR saved to {output_dir}")
    
    return output_path


# ============================================================================
# Alias for backward compatibility
# ============================================================================

MomentOpenVINO = MomentONNXPredictor
