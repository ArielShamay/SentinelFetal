"""
ONNX Compatibility Utilities.

This module provides monkey-patches for PyTorch operations that are not
supported by ONNX export. Specifically, it replaces torch.nanmean with
an ONNX-compatible implementation.

Usage:
    from src.utils.onnx_compat import patch_nanmean_for_onnx, restore_nanmean
    
    # Before ONNX export
    patch_nanmean_for_onnx()
    
    # ... perform export ...
    
    # After export (optional, restores original)
    restore_nanmean()
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

# Store original function for restoration
_original_nanmean = None
_original_tensor_nanmean = None
_is_patched = False


def onnx_compatible_nanmean(
    input: torch.Tensor,
    dim: Optional[Union[int, tuple]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    ONNX-compatible implementation of torch.nanmean.
    
    Computes the mean of all non-NaN elements in the input tensor,
    using only operations supported by ONNX opset 17.
    
    Formula: nanmean(x) = nansum(x) / count(non-nan elements)
    
    Args:
        input: Input tensor
        dim: Dimension(s) along which to compute the mean
        keepdim: Whether to retain reduced dimensions
        dtype: Desired data type of the output
        out: Output tensor (not used, for API compatibility)
        
    Returns:
        Tensor with nanmean computed
    """
    if dtype is not None:
        input = input.to(dtype)
    
    # Create mask for non-NaN values
    mask = ~torch.isnan(input)
    
    # Replace NaN with 0 for sum computation
    input_no_nan = torch.where(mask, input, torch.zeros_like(input))
    
    if dim is None:
        # Global mean
        total = input_no_nan.sum()
        count = mask.sum().float()
    else:
        # Mean along specified dimension(s)
        total = input_no_nan.sum(dim=dim, keepdim=keepdim)
        count = mask.sum(dim=dim, keepdim=keepdim).float()
    
    # Avoid division by zero
    count = torch.clamp(count, min=1.0)
    
    return total / count


def patch_nanmean_for_onnx() -> bool:
    """
    Monkey-patch torch.nanmean with ONNX-compatible implementation.
    
    This patches both:
    - torch.nanmean (module-level function)
    - torch.Tensor.nanmean (method)
    
    Returns:
        True if patching was successful, False if already patched.
    """
    global _original_nanmean, _original_tensor_nanmean, _is_patched
    
    if _is_patched:
        logger.warning("torch.nanmean is already patched")
        return False
    
    # Store originals
    _original_nanmean = torch.nanmean
    _original_tensor_nanmean = torch.Tensor.nanmean
    
    # Patch module-level function
    torch.nanmean = onnx_compatible_nanmean
    
    # Patch Tensor method
    def tensor_nanmean(self, dim=None, keepdim=False, *, dtype=None):
        return onnx_compatible_nanmean(self, dim=dim, keepdim=keepdim, dtype=dtype)
    
    torch.Tensor.nanmean = tensor_nanmean
    
    _is_patched = True
    logger.info("✓ torch.nanmean patched for ONNX compatibility")
    return True


def restore_nanmean() -> bool:
    """
    Restore original torch.nanmean implementation.
    
    Returns:
        True if restoration was successful, False if not patched.
    """
    global _original_nanmean, _original_tensor_nanmean, _is_patched
    
    if not _is_patched:
        logger.warning("torch.nanmean is not patched, nothing to restore")
        return False
    
    # Restore originals
    torch.nanmean = _original_nanmean
    torch.Tensor.nanmean = _original_tensor_nanmean
    
    _is_patched = False
    logger.info("✓ torch.nanmean restored to original")
    return True


def is_patched() -> bool:
    """Check if nanmean is currently patched."""
    return _is_patched


def test_patch():
    """Test that the patch works correctly."""
    import numpy as np
    
    print("Testing ONNX-compatible nanmean...")
    
    # Test 1: Simple tensor with NaN
    x = torch.tensor([1.0, 2.0, float('nan'), 4.0])
    
    # Original result
    original = torch.nanmean(x).item()
    
    # Patch and test
    patch_nanmean_for_onnx()
    patched = torch.nanmean(x).item()
    
    # Restore
    restore_nanmean()
    restored = torch.nanmean(x).item()
    
    print(f"  Original nanmean: {original:.4f}")
    print(f"  Patched nanmean:  {patched:.4f}")
    print(f"  Restored nanmean: {restored:.4f}")
    print(f"  Match: {np.isclose(original, patched)}")
    
    # Test 2: 2D tensor with dim
    x2 = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
    
    patch_nanmean_for_onnx()
    result_dim = torch.nanmean(x2, dim=1)
    restore_nanmean()
    expected = torch.nanmean(x2, dim=1)
    
    print(f"\n  2D tensor nanmean(dim=1):")
    print(f"    Patched:  {result_dim.tolist()}")
    print(f"    Expected: {expected.tolist()}")
    print(f"    Match: {torch.allclose(result_dim, expected)}")
    
    # Test 3: Tensor method
    patch_nanmean_for_onnx()
    method_result = x.nanmean().item()
    restore_nanmean()
    
    print(f"\n  Tensor.nanmean() method: {method_result:.4f}")
    print(f"  Match: {np.isclose(original, method_result)}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_patch()
