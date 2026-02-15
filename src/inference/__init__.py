"""
Inference module for SentinelFetal.

This module provides clean inference interfaces for trained models.
All modules here are designed for runtime/production use - they do NOT
perform training (fit), only transform and predict.
"""

from .ai_scorer import AIScorer, get_default_scorer

__all__ = ["AIScorer", "get_default_scorer"]
