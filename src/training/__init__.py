"""
Training module for SentinelFetal.

This module provides tools for preparing training datasets
and training the classifier model.
"""

from .prepare_data import prepare_dataset, DatasetConfig

__all__ = ['prepare_dataset', 'DatasetConfig']
