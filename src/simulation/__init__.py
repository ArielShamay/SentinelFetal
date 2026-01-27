"""
SentinelFetal Real-Time Simulation Module.

Provides synthetic CTG data generation for testing and demonstration.

Sub-packages:
    - core: Ring buffer, orchestrator
    - generators: FHR, UC, and patient signal generators
    - events: Event types and injection logic
    - processing: Pipeline adapter for analysis
    - logging: Event logging utilities

Example:
    >>> from src.simulation import (
    ...     SimulationOrchestrator, OrchestratorConfig,
    ...     EventType, SinusoidalParams
    ... )
    >>> orchestrator = SimulationOrchestrator(
    ...     processing_callback=lambda pid, data: data
    ... )
    >>> orchestrator.start()
"""

__version__ = "1.0.0"

# Core components
from .core.orchestrator import SimulationOrchestrator, OrchestratorConfig
from .core.ring_buffer import RingBuffer

# Generators
from .generators.patient_generator import PatientGenerator, PatientConfig
from .generators.fhr_generator import FHRGenerator, FHRGeneratorConfig
from .generators.uc_generator import UCGenerator, UCGeneratorConfig

# Events
from .events.event_types import (
    EventType,
    EventSeverity,
    EventParameters,
    InjectedEvent,
    LateDecelerationParams,
    VariableDecelerationParams,
    BradycardiaParams,
    TachycardiaParams,
    VariabilityParams,
    SinusoidalParams,
    TachysystoleParams,
)

# Processing (V6 Pre-AI compatibility adapter)
from .processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig

# Logging
from .logging.event_logger import EventLogger

__all__ = [
    # Core
    'SimulationOrchestrator',
    'OrchestratorConfig',
    'RingBuffer',
    
    # Generators
    'PatientGenerator',
    'PatientConfig',
    'FHRGenerator',
    'FHRGeneratorConfig',
    'UCGenerator',
    'UCGeneratorConfig',
    
    # Events
    'EventType',
    'EventSeverity',
    'EventParameters',
    'InjectedEvent',
    'LateDecelerationParams',
    'VariableDecelerationParams',
    'BradycardiaParams',
    'TachycardiaParams',
    'VariabilityParams',
    'SinusoidalParams',
    'TachysystoleParams',

    # Processing
    'PipelineAdapter',
    'PipelineAdapterConfig',
    
    # Logging
    'EventLogger',
]
