# -*- coding: utf-8 -*-
"""
Stage 4 Threshold Calibration Module.

This module calculates statistical thresholds (t_low, t_high) for the AI model
based on the distribution of scores from negative (healthy) samples.

References:
    SentinelFetal Stage 4 Documentation - Threshold Calibration
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import logging

logger = logging.getLogger(__name__)


class ThresholdCalibrator:
    """
    Calculates AI score thresholds from validation data.
    
    The calibrator works by analyzing the distribution of AI scores
    on negative (healthy) samples and selecting percentile-based thresholds
    that minimize false positives while maintaining high recall.
    """
    
    def __init__(
        self,
        t_low_percentile: float = 95.0,
        t_high_percentile: float = 99.0,
        persistence_k: int = 2,
        persistence_n: int = 3
    ):
        """
        Initialize the calibrator.
        
        Args:
            t_low_percentile: Percentile for lower threshold (default: 95%).
            t_high_percentile: Percentile for upper threshold (default: 99%).
            persistence_k: Minimum windows above threshold (K in K-of-N).
            persistence_n: Window buffer size (N in K-of-N).
        """
        self.t_low_percentile = t_low_percentile
        self.t_high_percentile = t_high_percentile
        self.persistence_k = persistence_k
        self.persistence_n = persistence_n
        
        # Results
        self.t_low: Optional[float] = None
        self.t_high: Optional[float] = None
        self.negative_scores: Optional[np.ndarray] = None
    
    def calibrate(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calibrate thresholds from scored validation data.
        
        Args:
            scores: AI model scores (0-1 range), shape (N,).
            labels: Ground truth labels (0=negative, 1=positive), shape (N,).
            
        Returns:
            Dictionary with calibrated thresholds:
                - t_low: Lower threshold
                - t_high: Upper threshold
                - K: Persistence K parameter
                - N: Persistence N parameter
        """
        if len(scores) != len(labels):
            raise ValueError("Scores and labels must have same length")
        
        # Extract negative samples only
        self.negative_scores = scores[labels == 0]
        
        if len(self.negative_scores) == 0:
            raise ValueError("No negative samples found in dataset")
        
        logger.info(f"Calibrating on {len(self.negative_scores)} negative samples")
        
        # Calculate percentile-based thresholds
        self.t_low = float(np.percentile(self.negative_scores, self.t_low_percentile))
        self.t_high = float(np.percentile(self.negative_scores, self.t_high_percentile))
        
        # Validate thresholds
        if self.t_low >= self.t_high:
            logger.warning(
                f"t_low ({self.t_low:.4f}) >= t_high ({self.t_high:.4f}), "
                f"adjusting t_low"
            )
            self.t_low = self.t_high - 0.05
        
        logger.info(f"Calibrated thresholds: t_low={self.t_low:.4f}, t_high={self.t_high:.4f}")
        
        return {
            't_low': self.t_low,
            't_high': self.t_high,
            'K': self.persistence_k,
            'N': self.persistence_n
        }
    
    def get_calibration_stats(self) -> Dict[str, float]:
        """
        Get statistics about the calibration.
        
        Returns:
            Dictionary with calibration statistics.
        """
        if self.negative_scores is None:
            raise RuntimeError("Must call calibrate() first")
        
        return {
            'num_negatives': len(self.negative_scores),
            'neg_mean': float(np.mean(self.negative_scores)),
            'neg_std': float(np.std(self.negative_scores)),
            'neg_min': float(np.min(self.negative_scores)),
            'neg_max': float(np.max(self.negative_scores)),
            'neg_median': float(np.median(self.negative_scores)),
            't_low': self.t_low,
            't_high': self.t_high,
            't_low_percentile': self.t_low_percentile,
            't_high_percentile': self.t_high_percentile
        }
    
    def save_thresholds(self, output_path: str | Path) -> None:
        """
        Save calibrated thresholds to YAML file.
        
        Args:
            output_path: Path to output YAML file.
        """
        if self.t_low is None or self.t_high is None:
            raise RuntimeError("Must call calibrate() first")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            'thresholds': {
                't_low': float(self.t_low),
                't_high': float(self.t_high)
            },
            'persistence': {
                'K': self.persistence_k,
                'N': self.persistence_n
            },
            'calibration_info': {
                'percentiles': {
                    't_low': self.t_low_percentile,
                    't_high': self.t_high_percentile
                },
                'num_negative_samples': len(self.negative_scores)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Saved thresholds to {output_path}")


def load_thresholds(yaml_path: str | Path) -> Dict[str, any]:
    """
    Load thresholds from YAML configuration file.
    
    Args:
        yaml_path: Path to thresholds YAML file.
        
    Returns:
        Dictionary with threshold configuration.
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config
