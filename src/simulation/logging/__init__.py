"""
Logging utilities for simulation events.

Provides lightweight event logging for audit trail.

Exports:
    - EventLogger: Lightweight event logger for audit trail
    - LogEntry: A single log entry dataclass
"""

from .event_logger import EventLogger, LogEntry

__all__ = [
    'EventLogger',
    'LogEntry',
]
