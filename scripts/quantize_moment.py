"""
Quantize MOMENT ONNX model to INT8.

This script applies dynamic INT8 quantization to reduce model size
and improve inference speed on CPU.

Usage:
    python scripts/quantize_moment.py --input models/moment.onnx --output models/moment_int8.onnx
    python scripts/quantize_moment.py --input models/moment.onnx --output models/moment_int8.onnx --benchmark

Performance Improvements:
    - Model size: ~4x reduction (1.5GB → ~400MB)
    - Inference speed: ~2-3x faster on Intel CPUs
    - Accuracy: >98% of FP32 quality

Requirements:
    - onnxruntime
    - onnx

References:
    - SentinelFetal Gen3.5 Technical Specification, Section 4
    - ONNX Runtime Quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
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
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_dynamic,
        QuantType,
        quant_pre_process
    )
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False
    logger.error("onnxruntime not installed. Run: pip install onnxruntime")


def preprocess_model(input_path: str, output_path: str) -> str:
    """
    Preprocess ONNX model for quantization.
    
    This step handles shape inference and graph optimization
    to ensure the model is ready for quantization.
    
    Args:
        input_path: Path to input ONNX model.
        output_path: Path to save preprocessed model.
        
    Returns:
        Path to preprocessed model.
    """
    logger.info("Preprocessing model for quantization...")
    
    try:
        quant_pre_process(
            input_model_path=input_path,
            output_model_path=output_path,
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
            verbose=0
        )
        logger.info(f"✓ Preprocessed model saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"Preprocessing failed ({e}), using original model")
        return input_path


def quantize_model(
    input_path: str,
    output_path: str,
    weight_type: str = "int8",
    per_channel: bool = True
) -> Path:
    """
    Apply dynamic INT8 quantization to ONNX model.
    
    Dynamic quantization is ideal for Transformer models because:
    - Activations are quantized at runtime (no calibration data needed)
    - Weights are quantized statically
    - Good balance between speed and accuracy
    
    Args:
        input_path: Path to FP32 ONNX model.
        output_path: Path to save quantized model.
        weight_type: Quantization type ('int8' or 'uint8').
        per_channel: Whether to use per-channel quantization (better accuracy).
        
    Returns:
        Path to quantized model.
    """
    if not ONNX_RUNTIME_AVAILABLE:
        raise ImportError(
            "onnxruntime is required. Install with: pip install onnxruntime"
        )
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from: {input_path}")
    
    # Preprocess if needed
    preprocessed_path = output_path.parent / "moment_preprocessed.onnx"
    try:
        processed_input = preprocess_model(str(input_path), str(preprocessed_path))
    except Exception:
        processed_input = str(input_path)
    
    # Select weight type
    if weight_type.lower() == "int8":
        quant_type = QuantType.QInt8
    else:
        quant_type = QuantType.QUInt8
    
    logger.info(f"Applying {weight_type.upper()} dynamic quantization...")
    logger.info(f"Per-channel: {per_channel}")
    
    # Dynamic quantization
    quantize_dynamic(
        model_input=processed_input,
        model_output=str(output_path),
        weight_type=quant_type,
        per_channel=per_channel,
        reduce_range=False,  # Better for modern CPUs
        optimize_model=True,
        extra_options={
            'ActivationSymmetric': True,
            'WeightSymmetric': True
        }
    )
    
    # Clean up preprocessed file
    if Path(preprocessed_path).exists() and preprocessed_path != input_path:
        Path(preprocessed_path).unlink()
    
    # Compare sizes
    original_size = input_path.stat().st_size / (1024 * 1024)
    quantized_size = output_path.stat().st_size / (1024 * 1024)
    compression_ratio = original_size / quantized_size
    
    logger.info("=" * 50)
    logger.info(f"Original model size:  {original_size:.1f} MB")
    logger.info(f"Quantized model size: {quantized_size:.1f} MB")
    logger.info(f"Compression ratio:    {compression_ratio:.2f}x")
    logger.info("=" * 50)
    
    logger.info(f"✓ Quantized model saved to: {output_path}")
    
    return output_path


def benchmark_models(
    fp32_path: str,
    int8_path: str,
    n_warmup: int = 5,
    n_iterations: int = 20
) -> dict:
    """
    Benchmark FP32 vs INT8 inference speed.
    
    Args:
        fp32_path: Path to FP32 ONNX model.
        int8_path: Path to INT8 ONNX model.
        n_warmup: Number of warmup iterations.
        n_iterations: Number of benchmark iterations.
        
    Returns:
        Dictionary with benchmark results.
    """
    if not ONNX_RUNTIME_AVAILABLE:
        raise ImportError("onnxruntime required for benchmarking")
    
    logger.info("Running benchmark...")
    logger.info(f"Warmup iterations: {n_warmup}")
    logger.info(f"Benchmark iterations: {n_iterations}")
    
    # Prepare test input (10-minute window at 4Hz)
    window_samples = 2400
    test_input = np.random.randn(1, 1, window_samples).astype(np.float32)
    
    # Session options optimized for benchmarking
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    results = {}
    
    # Benchmark FP32
    logger.info("\nBenchmarking FP32 model...")
    session_fp32 = ort.InferenceSession(
        fp32_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    
    # Warmup
    for _ in range(n_warmup):
        session_fp32.run(None, {'fhr_input': test_input})
    
    # Benchmark
    times_fp32 = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        session_fp32.run(None, {'fhr_input': test_input})
        times_fp32.append((time.perf_counter() - start) * 1000)
    
    fp32_mean = np.mean(times_fp32)
    fp32_std = np.std(times_fp32)
    results['fp32'] = {'mean_ms': fp32_mean, 'std_ms': fp32_std}
    logger.info(f"FP32: {fp32_mean:.1f} ± {fp32_std:.1f} ms")
    
    # Benchmark INT8
    logger.info("Benchmarking INT8 model...")
    session_int8 = ort.InferenceSession(
        int8_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    
    # Warmup
    for _ in range(n_warmup):
        session_int8.run(None, {'fhr_input': test_input})
    
    # Benchmark
    times_int8 = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        session_int8.run(None, {'fhr_input': test_input})
        times_int8.append((time.perf_counter() - start) * 1000)
    
    int8_mean = np.mean(times_int8)
    int8_std = np.std(times_int8)
    results['int8'] = {'mean_ms': int8_mean, 'std_ms': int8_std}
    logger.info(f"INT8: {int8_mean:.1f} ± {int8_std:.1f} ms")
    
    # Calculate speedup
    speedup = fp32_mean / int8_mean
    results['speedup'] = speedup
    
    logger.info("=" * 50)
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info("=" * 50)
    
    return results


def verify_accuracy(
    fp32_path: str,
    int8_path: str,
    n_samples: int = 10
) -> dict:
    """
    Verify INT8 accuracy compared to FP32.
    
    Args:
        fp32_path: Path to FP32 ONNX model.
        int8_path: Path to INT8 ONNX model.
        n_samples: Number of test samples.
        
    Returns:
        Dictionary with accuracy metrics.
    """
    if not ONNX_RUNTIME_AVAILABLE:
        raise ImportError("onnxruntime required for verification")
    
    logger.info(f"\nVerifying accuracy with {n_samples} samples...")
    
    # Create sessions
    session_fp32 = ort.InferenceSession(
        fp32_path,
        providers=['CPUExecutionProvider']
    )
    session_int8 = ort.InferenceSession(
        int8_path,
        providers=['CPUExecutionProvider']
    )
    
    differences = []
    cosine_sims = []
    
    for i in range(n_samples):
        # Random input
        test_input = np.random.randn(1, 1, 2400).astype(np.float32)
        
        # Run both models
        fp32_output = session_fp32.run(None, {'fhr_input': test_input})[0].flatten()
        int8_output = session_int8.run(None, {'fhr_input': test_input})[0].flatten()
        
        # Calculate differences
        max_diff = np.abs(fp32_output - int8_output).max()
        differences.append(max_diff)
        
        # Cosine similarity
        cos_sim = np.dot(fp32_output, int8_output) / (
            np.linalg.norm(fp32_output) * np.linalg.norm(int8_output)
        )
        cosine_sims.append(cos_sim)
    
    results = {
        'max_difference': float(np.max(differences)),
        'mean_difference': float(np.mean(differences)),
        'min_cosine_similarity': float(np.min(cosine_sims)),
        'mean_cosine_similarity': float(np.mean(cosine_sims))
    }
    
    logger.info(f"Max difference: {results['max_difference']:.2e}")
    logger.info(f"Mean difference: {results['mean_difference']:.2e}")
    logger.info(f"Min cosine similarity: {results['min_cosine_similarity']:.6f}")
    logger.info(f"Mean cosine similarity: {results['mean_cosine_similarity']:.6f}")
    
    # Quality assessment
    if results['mean_cosine_similarity'] > 0.99:
        logger.info("✓ Excellent accuracy - embeddings highly similar")
    elif results['mean_cosine_similarity'] > 0.95:
        logger.info("✓ Good accuracy - acceptable for production")
    else:
        logger.warning("⚠ Lower accuracy - consider using FP32")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quantize MOMENT ONNX model to INT8"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="models/moment.onnx",
        help="Input FP32 ONNX model path (default: models/moment.onnx)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/moment_int8.onnx",
        help="Output INT8 ONNX model path (default: models/moment_int8.onnx)"
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        default="int8",
        choices=["int8", "uint8"],
        help="Quantization weight type (default: int8)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after quantization"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify accuracy after quantization"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not ONNX_RUNTIME_AVAILABLE:
        logger.error(
            "onnxruntime is required. "
            "Install with: pip install onnxruntime"
        )
        sys.exit(1)
    
    # Check input file exists
    if not Path(args.input).exists():
        logger.error(f"Input model not found: {args.input}")
        logger.error("Run export_moment_onnx.py first to create the FP32 model")
        sys.exit(1)
    
    try:
        # Quantize
        output_path = quantize_model(
            input_path=args.input,
            output_path=args.output,
            weight_type=args.weight_type
        )
        
        # Optional: benchmark
        if args.benchmark:
            benchmark_models(args.input, str(output_path))
        
        # Optional: verify accuracy
        if args.verify:
            verify_accuracy(args.input, str(output_path))
        
        logger.info("\n" + "=" * 50)
        logger.info("Quantization completed successfully!")
        logger.info(f"Quantized model: {output_path}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
