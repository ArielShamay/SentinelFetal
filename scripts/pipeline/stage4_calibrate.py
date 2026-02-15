#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 Calibration Pipeline Script.

This script calibrates AI score thresholds (t_low, t_high) from validation data
and generates a calibration report with visualizations.

Usage:
    python scripts/pipeline/stage4_calibrate.py \\
        --scores data/processed/ai_scores.npy \\
        --labels data/processed/y.npy \\
        --output models/thresholds.yaml

References:
    SentinelFetal Stage 4 Documentation
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.calibration.calibrator import ThresholdCalibrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_distribution(
    scores: np.ndarray,
    t_low: float,
    t_high: float,
    output_path: Path
) -> None:
    """
    Plot score distribution with threshold markers.
    
    Args:
        scores: Negative sample scores.
        t_low: Lower threshold.
        t_high: Upper threshold.
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Threshold lines
    ax.axvline(t_low, color='orange', linestyle='--', linewidth=2, label=f't_low = {t_low:.4f}')
    ax.axvline(t_high, color='red', linestyle='--', linewidth=2, label=f't_high = {t_high:.4f}')
    
    ax.set_xlabel('AI Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('AI Score Distribution (Negative Samples)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved distribution plot to {output_path}")


def generate_report(
    calibrator: ThresholdCalibrator,
    output_dir: Path
) -> None:
    """
    Generate calibration report with statistics and plots.
    
    Args:
        calibrator: Calibrated ThresholdCalibrator instance.
        output_dir: Directory to save report files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get statistics
    stats = calibrator.get_calibration_stats()
    
    # Save JSON report
    report_path = output_dir / 'stage4_calibration_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved calibration report to {report_path}")
    
    # Generate distribution plot
    plot_path = output_dir / 'threshold_distribution.png'
    plot_distribution(
        calibrator.negative_scores,
        calibrator.t_low,
        calibrator.t_high,
        plot_path
    )
    
    # Print summary to console
    print("\n" + "="*60)
    print("STAGE 4 CALIBRATION SUMMARY")
    print("="*60)
    print(f"Number of negative samples: {stats['num_negatives']}")
    print(f"Negative score statistics:")
    print(f"  Mean:   {stats['neg_mean']:.4f}")
    print(f"  Std:    {stats['neg_std']:.4f}")
    print(f"  Min:    {stats['neg_min']:.4f}")
    print(f"  Max:    {stats['neg_max']:.4f}")
    print(f"  Median: {stats['neg_median']:.4f}")
    print(f"\nCalibrated thresholds:")
    print(f"  t_low  = {stats['t_low']:.4f} ({stats['t_low_percentile']}th percentile)")
    print(f"  t_high = {stats['t_high']:.4f} ({stats['t_high_percentile']}th percentile)")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Stage 4: Calibrate AI score thresholds'
    )
    parser.add_argument(
        '--scores',
        type=str,
        required=True,
        help='Path to AI scores file (.npy)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to labels file (.npy)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/thresholds.yaml',
        help='Output path for thresholds YAML file'
    )
    parser.add_argument(
        '--t-low-percentile',
        type=float,
        default=95.0,
        help='Percentile for t_low threshold (default: 95.0)'
    )
    parser.add_argument(
        '--t-high-percentile',
        type=float,
        default=99.0,
        help='Percentile for t_high threshold (default: 99.0)'
    )
    parser.add_argument(
        '--persistence-k',
        type=int,
        default=2,
        help='Persistence K parameter (default: 2)'
    )
    parser.add_argument(
        '--persistence-n',
        type=int,
        default=3,
        help='Persistence N parameter (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading scores from {args.scores}")
    scores = np.load(args.scores)
    
    logger.info(f"Loading labels from {args.labels}")
    labels = np.load(args.labels)
    
    logger.info(f"Loaded {len(scores)} samples")
    
    # Initialize calibrator
    calibrator = ThresholdCalibrator(
        t_low_percentile=args.t_low_percentile,
        t_high_percentile=args.t_high_percentile,
        persistence_k=args.persistence_k,
        persistence_n=args.persistence_n
    )
    
    # Calibrate
    logger.info("Running calibration...")
    thresholds = calibrator.calibrate(scores, labels)
    
    # Save thresholds
    output_path = Path(args.output)
    calibrator.save_thresholds(output_path)
    
    # Generate report
    report_dir = output_path.parent
    generate_report(calibrator, report_dir)
    
    logger.info("✓ Stage 4 calibration completed successfully")


if __name__ == '__main__':
    main()
