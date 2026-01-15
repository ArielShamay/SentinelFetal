"""
Event Logger - Lightweight logging without storing raw signals.

Provides audit trail logging for:
- Event injections (what was injected, when, to which patient)
- Category 2/3 alerts (when detected, which patient, confidence)

Memory-efficient design: only stores metadata, NOT raw signals.
Maximum ~100KB memory usage with max_entries=1000.

References:
    - SentinelFetal Real-Time Simulator SPEC Part 2, Section 8
"""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..events.event_types import InjectedEvent


@dataclass
class LogEntry:
    """
    A single log entry.
    
    Attributes:
        timestamp: Real-world timestamp when logged.
        simulation_time: Simulation time in seconds.
        event_type: Type of log entry ('INJECTION', 'ALERT', etc.).
        patient_id: Patient identifier.
        details: Additional details as dictionary.
    """
    timestamp: datetime
    simulation_time: float
    event_type: str
    patient_id: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'simulation_time': round(self.simulation_time, 2),
            'event_type': self.event_type,
            'patient_id': self.patient_id,
            **self.details
        }
    
    def __repr__(self) -> str:
        return (
            f"LogEntry({self.event_type}, patient={self.patient_id}, "
            f"sim_time={self.simulation_time:.1f}s)"
        )


class EventLogger:
    """
    Logs events with minimal memory (~100KB max).
    
    Only stores:
    - Event injections
    - Category 2/3 alerts
    - NOT raw signal data
    
    Uses a deque with maxlen for automatic old entry removal.
    
    Example:
        >>> logger = EventLogger(max_entries=1000)
        >>> logger.log_injection(event)
        >>> logger.log_alert('P1', {'category': 3, 'confidence': 0.95})
        >>> print(f"Logged {logger.size} entries")
    """
    
    def __init__(self, max_entries: int = 1000):
        """
        Initialize the event logger.
        
        Args:
            max_entries: Maximum number of log entries to keep.
                        Oldest entries are automatically removed.
        """
        self._entries: deque = deque(maxlen=max_entries)
        self._start_time = datetime.now()
        self._injection_count = 0
        self._alert_count = 0
    
    def log_injection(self, event: InjectedEvent) -> None:
        """
        Log an event injection.
        
        Args:
            event: The injected event to log.
        """
        # Extract severity if available
        severity = None
        if hasattr(event.params, 'severity'):
            severity = event.params.severity.name if event.params.severity else None
        
        entry = LogEntry(
            timestamp=datetime.now(),
            simulation_time=event.start_time,
            event_type='INJECTION',
            patient_id=event.patient_id,
            details={
                'injected_event': event.event_type.name,
                'duration_seconds': round(event.end_time - event.start_time, 1),
                'severity': severity
            }
        )
        self._entries.append(entry)
        self._injection_count += 1
    
    def log_alert(self, patient_id: str, results: Dict[str, Any]) -> None:
        """
        Log a Category 2/3 alert.
        
        Args:
            patient_id: Patient identifier.
            results: Processing results containing category, confidence, etc.
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            simulation_time=results.get('simulation_time', 0),
            event_type='ALERT',
            patient_id=patient_id,
            details={
                'category': results.get('category'),
                'confidence': round(results.get('confidence', 0), 3),
                'was_overridden': results.get('was_overridden', False),
                'ml_prediction': results.get('ml_prediction')
            }
        )
        self._entries.append(entry)
        self._alert_count += 1
    
    def log_custom(
        self,
        event_type: str,
        patient_id: str,
        simulation_time: float,
        details: Dict[str, Any]
    ) -> None:
        """
        Log a custom event.
        
        Args:
            event_type: Custom event type string.
            patient_id: Patient identifier.
            simulation_time: Current simulation time.
            details: Event details.
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            simulation_time=simulation_time,
            event_type=event_type,
            patient_id=patient_id,
            details=details
        )
        self._entries.append(entry)
    
    def get_entries(
        self,
        event_type: Optional[str] = None,
        patient_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[LogEntry]:
        """
        Get log entries, optionally filtered.
        
        Args:
            event_type: Filter by event type (e.g., 'INJECTION', 'ALERT').
            patient_id: Filter by patient ID.
            limit: Maximum number of entries to return.
            
        Returns:
            List of matching log entries.
        """
        entries = list(self._entries)
        
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        
        if patient_id:
            entries = [e for e in entries if e.patient_id == patient_id]
        
        if limit:
            entries = entries[-limit:]
        
        return entries
    
    def get_alerts(self, patient_id: Optional[str] = None) -> List[LogEntry]:
        """Get all alert entries, optionally for a specific patient."""
        return self.get_entries(event_type='ALERT', patient_id=patient_id)
    
    def get_injections(self, patient_id: Optional[str] = None) -> List[LogEntry]:
        """Get all injection entries, optionally for a specific patient."""
        return self.get_entries(event_type='INJECTION', patient_id=patient_id)
    
    def export_csv(self, filepath: str) -> None:
        """
        Export log to CSV file.
        
        Args:
            filepath: Path to the output CSV file.
        """
        if not self._entries:
            return
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Collect all possible keys from details
            all_keys = set()
            for entry in self._entries:
                all_keys.update(entry.details.keys())
            
            fieldnames = [
                'timestamp', 'simulation_time', 'event_type', 'patient_id'
            ] + sorted(all_keys)
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in self._entries:
                row = {
                    'timestamp': entry.timestamp.isoformat(),
                    'simulation_time': entry.simulation_time,
                    'event_type': entry.event_type,
                    'patient_id': entry.patient_id,
                }
                row.update(entry.details)
                writer.writerow(row)
    
    def export_json(self, filepath: str) -> None:
        """
        Export log to JSON file.
        
        Args:
            filepath: Path to the output JSON file.
        """
        data = {
            'export_time': datetime.now().isoformat(),
            'session_start': self._start_time.isoformat(),
            'total_injections': self._injection_count,
            'total_alerts': self._alert_count,
            'entries': [entry.to_dict() for entry in self._entries]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def clear(self) -> None:
        """Clear all log entries."""
        self._entries.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the log.
        
        Returns:
            Dictionary with summary statistics.
        """
        alerts = self.get_entries(event_type='ALERT')
        cat3_alerts = [a for a in alerts if a.details.get('category') == 3]
        cat2_alerts = [a for a in alerts if a.details.get('category') == 2]
        
        return {
            'session_start': self._start_time.isoformat(),
            'total_entries': len(self._entries),
            'total_injections': self._injection_count,
            'total_alerts': self._alert_count,
            'category_3_alerts': len(cat3_alerts),
            'category_2_alerts': len(cat2_alerts),
            'unique_patients': len(set(e.patient_id for e in self._entries))
        }
    
    @property
    def size(self) -> int:
        """Number of entries in log."""
        return len(self._entries)
    
    @property
    def max_size(self) -> int:
        """Maximum number of entries."""
        return self._entries.maxlen or 0
