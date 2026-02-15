"""
AI Scorer Module for Stage 3 Inference

This module provides a clean inference interface for the Stage 3 AI Baseline.
It loads the trained MiniRocket + StandardScaler + LogisticRegression pipeline
and provides methods to score individual windows or batches.

IMPORTANT: This module does NOT perform any training (fit).
           It only uses transform and predict_proba for inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np


class AIScorer:
    """
    AI Scorer for SentinelFetal Stage 3.
    
    Loads a trained MiniRocket + StandardScaler + LogisticRegression pipeline
    and provides inference methods for scoring CTG windows.
    
    Attributes:
        model_version: Version string of the loaded model
        config: Configuration dict from training
    """
    
    def __init__(self, pipeline_path: Union[str, Path]):
        """
        Initialize AIScorer by loading a trained pipeline.
        
        Args:
            pipeline_path: Path to stage3_ai_pipeline.joblib
            
        Raises:
            FileNotFoundError: If pipeline file doesn't exist
            KeyError: If pipeline is missing required components
        """
        pipeline_path = Path(pipeline_path)
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        
        # Load pipeline
        self._pipeline = joblib.load(pipeline_path)
        
        # Extract components
        self._minirocket = self._pipeline["minirocket"]
        self._scaler = self._pipeline["scaler"]
        self._lr = self._pipeline["lr"]
        
        # Metadata
        self.model_version = self._pipeline.get("model_version", "unknown")
        self.config = self._pipeline.get("config", {})
        self.trained_at = self._pipeline.get("trained_at", "unknown")
        
        # Expected input shape
        self._n_channels = self.config.get("n_channels", 2)
        self._window_samples = self.config.get("window_samples", 4800)
    
    def score_window(self, x_window: np.ndarray) -> float:
        """
        Compute ai_score for a single window.
        
        Args:
            x_window: Window data, shape (n_channels, n_timestamps) 
                      or (n_timestamps, n_channels).
                      n_channels=2 (FHR, UC), n_timestamps=4800 (20min @ 4Hz)
        
        Returns:
            ai_score: Float in [0, 1] representing probability of adverse outcome
            
        Raises:
            ValueError: If input shape is invalid
        """
        x_window = np.asarray(x_window)
        
        # Handle different input shapes
        if x_window.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x_window.shape}")
        
        # Determine if we need to transpose
        if x_window.shape[0] == self._n_channels:
            # Already (n_channels, n_timestamps)
            pass
        elif x_window.shape[1] == self._n_channels:
            # Need to transpose from (n_timestamps, n_channels)
            x_window = x_window.T
        else:
            raise ValueError(
                f"Expected shape ({self._n_channels}, {self._window_samples}) "
                f"or ({self._window_samples}, {self._n_channels}), "
                f"got {x_window.shape}"
            )
        
        # Add batch dimension
        x_batch = x_window[np.newaxis, :, :]
        
        # Score
        scores = self.score_windows(x_batch)
        return float(scores[0])
    
    def score_windows(self, x_windows: np.ndarray) -> np.ndarray:
        """
        Compute ai_scores for a batch of windows.
        
        Args:
            x_windows: Batch of windows, shape (n_windows, n_channels, n_timestamps)
                       n_channels=2 (FHR, UC), n_timestamps=4800 (20min @ 4Hz)
        
        Returns:
            ai_scores: Array of floats in [0, 1], shape (n_windows,)
            
        Raises:
            ValueError: If input shape is invalid
        """
        x_windows = np.asarray(x_windows)
        
        if x_windows.ndim != 3:
            raise ValueError(f"Expected 3D array (n_windows, n_channels, n_timestamps), got shape {x_windows.shape}")
        
        n_windows, n_channels, n_timestamps = x_windows.shape
        
        if n_channels != self._n_channels:
            raise ValueError(f"Expected {self._n_channels} channels, got {n_channels}")
        
        # Apply pipeline (NO FIT!)
        features = self._minirocket.transform(x_windows)
        features_scaled = self._scaler.transform(features)
        ai_scores = self._lr.predict_proba(features_scaled)[:, 1]
        
        return ai_scores
    
    def __repr__(self) -> str:
        return f"AIScorer(model_version='{self.model_version}', trained_at='{self.trained_at}')"


def get_default_scorer() -> AIScorer:
    """
    Get AIScorer with default pipeline path.
    
    Returns:
        AIScorer instance loaded from processed_data_v1/stage3_ai_pipeline.joblib
    """
    # Find project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]  # src/inference/ai_scorer.py -> project root
    
    pipeline_path = project_root / "processed_data_v1" / "stage3_ai_pipeline.joblib"
    return AIScorer(pipeline_path)
