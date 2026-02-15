#!/usr/bin/env python3
"""
Stage 2 Build Script - Windowing + Quality Gate

CLI script to run the Stage 2 pipeline on Stage 1 artifacts.

Usage:
    python scripts/pipeline/stage2_build.py
    python scripts/pipeline/stage2_build.py --input-dir processed_data_v1 --window-minutes 20

Author: SentinelFetal Team
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.stage2_windowing import (
    run_stage2_pipeline,
    WindowingConfig,
    WindowQualityThresholds
)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Windowing + Quality Gate Pipeline"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="processed_data_v1",
        help="Directory containing Stage 1 outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to input-dir)"
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=20,
        help="Window length in minutes (default: 20)"
    )
    parser.add_argument(
        "--stride-minutes",
        type=int,
        default=5,
        help="Stride between windows in minutes (default: 5)"
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=4,
        help="Sampling frequency in Hz (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Build paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    # Build config
    config = WindowingConfig(
        fs=args.fs,
        window_minutes=args.window_minutes,
        stride_minutes=args.stride_minutes,
        thresholds=WindowQualityThresholds()
    )
    
    # Validate input
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    required_files = ["X_norm_4hz.npy", "mask_fhr.npy", "mask_uc.npy", "manifest.csv"]
    for f in required_files:
        if not (input_dir / f).exists():
            print(f"Error: Missing Stage 1 artifact: {f}")
            sys.exit(1)
    
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Config: {config.window_minutes}min window, {config.stride_minutes}min stride @ {config.fs}Hz")
    
    # Run pipeline
    windows_df, qc_stats = run_stage2_pipeline(input_dir, output_dir, config)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Stage 2 Complete!")
    print("=" * 50)
    print(f"Total windows: {len(windows_df):,}")
    print(f"Quality distribution: {qc_stats.get('windows_by_quality', {})}")
    print(f"Valid for AI: {qc_stats.get('windows_valid_for_ai', 0):,}")
    print(f"Valid for Rules: {qc_stats.get('windows_valid_for_rules', 0):,}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
