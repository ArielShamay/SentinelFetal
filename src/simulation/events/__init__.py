"""
Event types and injection for CTG simulation.

Exports:
    - EventType: Enum of injectable clinical events
    - EventSeverity: Severity levels for events
    - EventParameters: Base event parameters
    - LateDecelerationParams, VariableDecelerationParams, etc.
    - InjectedEvent: Active event instance
"""

from .event_types import (
    EventType,
    EventSeverity,
    EventParameters,
    LateDecelerationParams,
    VariableDecelerationParams,
    BradycardiaParams,
    TachycardiaParams,
    VariabilityParams,
    SinusoidalParams,
    TachysystoleParams,
    InjectedEvent,
)

__all__ = [
    'EventType',
    'EventSeverity', 
    'EventParameters',
    'LateDecelerationParams',
    'VariableDecelerationParams',
    'BradycardiaParams',
    'TachycardiaParams',
    'VariabilityParams',
    'SinusoidalParams',
    'TachysystoleParams',
    'InjectedEvent',
]
