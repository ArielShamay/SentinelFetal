"""
Processing adapters for simulation integration.

Connects simulation to existing SentinelFetal analysis pipeline.

Exports:
    - PipelineAdapter: Bridges simulation to Gen3.5 pipeline
    - PipelineAdapterConfig: Configuration for the adapter
"""

from .pipeline_adapter import PipelineAdapter, PipelineAdapterConfig

__all__ = [
    'PipelineAdapter',
    'PipelineAdapterConfig',
]
