"""
Export MOMENT model to ONNX format.

This script exports the PyTorch MOMENT-1-large model to ONNX format for
optimized inference using ONNX Runtime or OpenVINO.

Usage:
    python scripts/export_moment_onnx.py --output models/moment.onnx
    python scripts/export_moment_onnx.py --output models/moment.onnx --verify

Requirements:
    - torch
    - momentfm
    - onnx
    - onnxruntime (for verification)

Performance Notes:
    - Export takes ~30-60 seconds (downloads model if not cached)
    - Output ONNX file is ~1.5GB (FP32)
    - Use quantize_moment.py to reduce size to ~400MB (INT8)

References:
    - SentinelFetal Gen3.5 Technical Specification, Section 4
    - MOMENT Paper: https://arxiv.org/abs/2402.03885
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch not installed. Run: pip install torch")

# CRITICAL: Apply nanmean patch BEFORE importing momentfm
# This replaces torch.nanmean with an ONNX-compatible implementation
if TORCH_AVAILABLE:
    try:
        # Add parent directory to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.utils.onnx_compat import patch_nanmean_for_onnx
        patch_nanmean_for_onnx()
        logger.info("✓ Applied nanmean ONNX compatibility patch")
    except ImportError as e:
        logger.warning(f"Could not apply nanmean patch: {e}")

try:
    from momentfm import MOMENTPipeline
    MOMENT_AVAILABLE = True
except ImportError:
    MOMENT_AVAILABLE = False
    logger.error("momentfm not installed. Run: pip install momentfm")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.error("onnx not installed. Run: pip install onnx")


class MomentEmbedWrapper(torch.nn.Module):
    """
    Wrapper for MOMENT pipeline to enable ONNX export.
    
    MOMENT uses a non-standard `embed(x_enc=...)` method instead of `forward()`.
    This wrapper routes `forward()` to `embed()` and extracts the embedding tensor.
    """
    
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that extracts embeddings.
        
        Args:
            x: Input tensor [batch, channels=1, seq_len]
            
        Returns:
            Embeddings tensor [batch, 1024]
        """
        output = self.pipeline.embed(x_enc=x)
        return output.embeddings


class MomentONNXExporter:
    """
    Exports MOMENT PyTorch model to ONNX format.
    
    The exporter handles:
    - Loading the MOMENT-1-large model
    - Preparing for embedding extraction mode
    - Exporting with proper dynamic axes
    - Verifying export accuracy
    """
    
    # Model configuration
    MODEL_NAME = "AutonLab/MOMENT-1-large"
    EMBEDDING_DIM = 1024
    DEFAULT_WINDOW_SAMPLES = 2400  # 10 min * 60 sec * 4 Hz
    
    def __init__(self):
        """Initialize the exporter."""
        self._model = None
        self._pipeline = None
        
    def load_model(self) -> None:
        """
        Load the MOMENT model from HuggingFace.
        
        Uses task_name='embedding' for Zero-Shot feature extraction.
        """
        if not MOMENT_AVAILABLE:
            raise ImportError(
                "momentfm package is required. "
                "Install with: pip install momentfm"
            )
        
        logger.info(f"Loading {self.MODEL_NAME} model...")
        
        self._pipeline = MOMENTPipeline.from_pretrained(
            self.MODEL_NAME,
            model_kwargs={
                'task_name': 'embedding',  # Zero-Shot embedding mode
                'n_channels': 1  # Single channel (FHR)
            }
        )
        
        # Wrap the pipeline to enable ONNX export via forward()
        self._model = MomentEmbedWrapper(self._pipeline)
        self._model.eval()  # Inference mode
        
        # Count parameters
        n_params = sum(p.numel() for p in self._pipeline.parameters())
        logger.info(f"Model loaded: {n_params / 1e6:.1f}M parameters")
        
    def export_to_onnx(
        self,
        output_path: str,
        opset_version: int = 17,
        optimize: bool = True
    ) -> Path:
        """
        Export MOMENT model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model.
            opset_version: ONNX opset version (17 recommended for Transformers).
            optimize: Whether to apply constant folding optimization.
            
        Returns:
            Path to the saved ONNX model.
            
        Raises:
            RuntimeError: If export fails.
        """
        if self._model is None:
            self.load_model()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting to ONNX (opset {opset_version})...")
        
        # Create dummy input: [batch=1, channels=1, seq_len=2400]
        dummy_input = torch.randn(
            1, 1, self.DEFAULT_WINDOW_SAMPLES,
            dtype=torch.float32
        )
        
        # Dynamic axes for flexible batch size and sequence length
        dynamic_axes = {
            'fhr_input': {0: 'batch_size', 2: 'seq_len'},
            'embedding': {0: 'batch_size'}
        }
        
        # Export using TorchScript (legacy) exporter for better compatibility
        try:
            # Disable dynamo to use the legacy TorchScript-based exporter
            torch.onnx.export(
                self._model,
                dummy_input,
                str(output_path),
                opset_version=opset_version,
                input_names=['fhr_input'],
                output_names=['embedding'],
                dynamic_axes=dynamic_axes,
                do_constant_folding=optimize,
                export_params=True,
                verbose=False,
                dynamo=False  # Force legacy TorchScript exporter
            )
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise RuntimeError(f"Failed to export ONNX model: {e}")
        
        logger.info(f"✓ ONNX model saved to: {output_path}")
        
        # Verify the exported model
        self._verify_onnx_model(output_path)
        
        # Print model size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model size: {file_size_mb:.1f} MB")
        
        return output_path
    
    def _verify_onnx_model(self, onnx_path: Path) -> None:
        """
        Verify the ONNX model is valid.
        
        Args:
            onnx_path: Path to the ONNX model.
            
        Raises:
            ValueError: If model verification fails.
        """
        if not ONNX_AVAILABLE:
            logger.warning("onnx package not available, skipping verification")
            return
        
        logger.info("Verifying ONNX model...")
        
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model verification passed")
        except Exception as e:
            raise ValueError(f"ONNX model verification failed: {e}")
    
    def verify_output_accuracy(
        self,
        onnx_path: str,
        tolerance: float = 1e-4
    ) -> bool:
        """
        Verify ONNX output matches PyTorch output.
        
        Args:
            onnx_path: Path to the ONNX model.
            tolerance: Maximum allowed difference.
            
        Returns:
            True if outputs match within tolerance.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning(
                "onnxruntime not available, skipping accuracy verification"
            )
            return True
        
        if self._model is None:
            self.load_model()
        
        logger.info("Verifying output accuracy...")
        
        # Create test input
        test_input = np.random.randn(
            1, 1, self.DEFAULT_WINDOW_SAMPLES
        ).astype(np.float32)
        
        # PyTorch inference
        with torch.no_grad():
            pt_input = torch.tensor(test_input)
            pt_output = self._model(pt_input)
            
            # Handle different output formats
            if hasattr(pt_output, 'embeddings'):
                pt_embedding = pt_output.embeddings.numpy()
            else:
                pt_embedding = pt_output.numpy()
        
        # ONNX inference
        session = ort.InferenceSession(str(onnx_path))
        onnx_output = session.run(None, {'fhr_input': test_input})[0]
        
        # Compare
        max_diff = np.abs(pt_embedding - onnx_output).max()
        mean_diff = np.abs(pt_embedding - onnx_output).mean()
        
        logger.info(f"Max difference: {max_diff:.2e}")
        logger.info(f"Mean difference: {mean_diff:.2e}")
        
        if max_diff < tolerance:
            logger.info("✓ ONNX output matches PyTorch within tolerance")
            return True
        else:
            logger.warning(
                f"⚠ Output difference ({max_diff:.2e}) exceeds tolerance ({tolerance:.2e})"
            )
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export MOMENT model to ONNX format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/moment.onnx",
        help="Output path for ONNX model (default: models/moment.onnx)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output accuracy after export"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable constant folding optimization"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    
    if not MOMENT_AVAILABLE:
        logger.error("momentfm is required. Install with: pip install momentfm")
        sys.exit(1)
    
    if not ONNX_AVAILABLE:
        logger.error("onnx is required. Install with: pip install onnx")
        sys.exit(1)
    
    # Export
    exporter = MomentONNXExporter()
    
    try:
        output_path = exporter.export_to_onnx(
            output_path=args.output,
            opset_version=args.opset,
            optimize=not args.no_optimize
        )
        
        if args.verify:
            exporter.verify_output_accuracy(output_path)
        
        logger.info("=" * 50)
        logger.info("Export completed successfully!")
        logger.info(f"Next step: Quantize with scripts/quantize_moment.py")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
