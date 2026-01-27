"""
Processing adapters for simulation integration.

V6 Pre-AI compatibility adapter lives in pipeline_adapter.py.
"""

from .pipeline_adapter import PipelineAdapter, PipelineAdapterConfig

__all__ = ["PipelineAdapter", "PipelineAdapterConfig"]
