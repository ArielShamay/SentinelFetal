# -*- coding: utf-8 -*-
"""
Thread-Safe State Bridge for SentinelFetal.

The DataBridge provides thread-safe communication between the high-speed
simulation backend (1kHz) and any frontend (React via FastAPI, or other clients).

This module provides:
- Thread-safe ring buffers for patient data
- Immutable snapshot objects for consistent state reads
- Singleton pattern for cross-module access

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  BACKEND (1kHz)          SHARED STATE           FRONTEND   │
    │  ┌────────────────┐     ┌──────────────┐      ┌─────────┐  │
    │  │ PipelineAdapter│──▶  │  DataBridge  │  ◀───│ FastAPI │  │
    │  │ (push_batch)   │     │  (singleton) │      │WebSocket│  │
    │  └────────────────┘     └──────────────┘      └─────────┘  │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Backend (in pipeline_adapter.py):
    from src.interfaces.state_bridge import get_data_bridge
    bridge = get_data_bridge()
    bridge.push_batch(patient_id, snapshot)

    # API (in websocket handler):
    from src.interfaces.state_bridge import get_data_bridge
    bridge = get_data_bridge()
    view = bridge.get_ward_view()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class HighlightRegion:
    """
    Represents a region to highlight on the CTG plot (Red Zone).

    Used for:
    - Deceleration events (late, variable, prolonged)
    - MHR contamination periods
    - Signal quality issues
    - Explainability highlights

    Attributes:
        start_idx: Start sample index
        end_idx: End sample index
        region_type: Type of region (e.g., 'late_decel', 'mhr_alert', 'low_quality')
        severity: Severity level ('info', 'warning', 'critical')
        label: Human-readable label for annotation
        color: RGBA color string (e.g., 'rgba(255,0,0,0.3)')
    """
    start_idx: int
    end_idx: int
    region_type: str
    severity: str = "warning"
    label: str = ""
    color: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "region_type": self.region_type,
            "severity": self.severity,
            "label": self.label,
            "color": self.color or self._default_color(),
        }

    def _default_color(self) -> str:
        """Get default color based on severity."""
        color_map = {
            'info': 'rgba(0, 123, 255, 0.15)',
            'warning': 'rgba(255, 193, 7, 0.25)',
            'critical': 'rgba(220, 53, 69, 0.35)',
        }
        return color_map.get(self.severity, 'rgba(128, 128, 128, 0.2)')


@dataclass
class PatientSnapshot:
    """
    Thread-safe snapshot of a single patient's state.

    This is what the frontend reads - a complete, immutable view
    of the patient's current status at a given timestamp.

    Attributes:
        patient_id: Unique patient identifier
        timestamp: Snapshot creation time (Unix timestamp)
        category: Current classification (1=Normal, 2=Intermediate, 3=Pathological)
        confidence: Classification confidence (0.0-1.0)
        fhr_recent: Recent FHR samples (last ~60 seconds)
        uc_recent: Recent UC samples
        baseline: Current baseline FHR value
        variability: Current variability value
        findings: Dictionary of clinical findings
        alerts: List of active alerts
        highlight_regions: List of HighlightRegion for red zones
        mhr_alert: MHR contamination alert (if any)
        trend_data: 60-minute trend analysis result
        explanation: Classification explanation
        active_events: List of currently active injected events
    """
    patient_id: str
    timestamp: float
    category: int = 1
    confidence: float = 0.0
    fhr_recent: List[float] = field(default_factory=list)
    uc_recent: List[float] = field(default_factory=list)
    baseline: float = 140.0
    variability: float = 10.0
    findings: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    highlight_regions: List[HighlightRegion] = field(default_factory=list)
    mhr_alert: Optional[Dict[str, Any]] = None
    trend_data: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    active_events: List[str] = field(default_factory=list)
    name: str = ""
    bed_number: int = 0


@dataclass
class WardSnapshot:
    """
    Complete snapshot of all patients for the ward view.

    This allows the frontend to get a consistent view of
    all patients at once, avoiding partial updates.
    """
    timestamp: float
    patients: Dict[str, PatientSnapshot] = field(default_factory=dict)
    simulation_time: float = 0.0
    is_running: bool = False
    is_paused: bool = False
    speed_multiplier: float = 1.0


# =============================================================================
# Thread-Safe Data Bridge (Singleton)
# =============================================================================

class DataBridge:
    """
    Thread-safe bridge between backend processing and frontend rendering.

    Uses a ring buffer (deque with maxlen) per patient to store recent
    snapshots. The frontend polls get_latest_view() at its own pace
    without blocking the backend.

    Thread Safety:
    - All public methods acquire self._lock before accessing shared state
    - Uses deque with maxlen for automatic memory management
    - Pure Python data structures (no framework dependencies)

    Performance:
    - O(1) push operations (deque append)
    - O(1) get operations (dict lookup + deque access)
    - Memory bounded by maxlen * num_patients
    """

    # Singleton instance
    _instance: Optional['DataBridge'] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> 'DataBridge':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        max_snapshots_per_patient: int = 100,
        max_fhr_samples: int = 2400,  # 10 minutes at 4Hz
    ):
        """
        Initialize the DataBridge.

        Args:
            max_snapshots_per_patient: Max snapshots to keep per patient
            max_fhr_samples: Max FHR samples to store in each snapshot
        """
        # Prevent re-initialization of singleton
        if getattr(self, '_initialized', False):
            return

        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._max_snapshots = max_snapshots_per_patient
        self._max_fhr_samples = max_fhr_samples

        # Per-patient ring buffers
        self._patient_buffers: Dict[str, Deque[PatientSnapshot]] = {}

        # Global state
        self._simulation_time: float = 0.0
        self._is_running: bool = False
        self._is_paused: bool = False
        self._speed_multiplier: float = 1.0

        # Alert queue (FIFO, limited size)
        self._alert_queue: Deque[Dict[str, Any]] = deque(maxlen=100)

        # Statistics
        self._push_count = 0
        self._last_push_time = 0.0

        self._initialized = True
        logger.info("DataBridge initialized (singleton)")

    # =========================================================================
    # Backend (Producer) Interface
    # =========================================================================

    def push_batch(
        self,
        patient_id: str,
        snapshot: PatientSnapshot,
    ) -> None:
        """
        Push a patient snapshot from the backend.

        This is called by the PipelineAdapter after processing.
        Thread-safe and non-blocking for the backend.

        Args:
            patient_id: Patient identifier
            snapshot: Complete patient state snapshot
        """
        with self._lock:
            # Initialize buffer for new patients
            if patient_id not in self._patient_buffers:
                self._patient_buffers[patient_id] = deque(
                    maxlen=self._max_snapshots
                )

            # Append snapshot (deque handles max size)
            self._patient_buffers[patient_id].append(snapshot)

            # Update stats
            self._push_count += 1
            self._last_push_time = time.time()

    def push_alert(self, alert: Dict[str, Any]) -> None:
        """
        Push a global alert to the alert queue.

        Args:
            alert: Alert dictionary with patient_id, severity, message, etc.
        """
        with self._lock:
            alert['timestamp'] = time.time()
            self._alert_queue.append(alert)

    def update_simulation_state(
        self,
        simulation_time: float,
        is_running: bool,
        is_paused: bool,
        speed_multiplier: float = 1.0,
    ) -> None:
        """
        Update global simulation state.

        Called by the orchestrator to keep UI in sync.
        """
        with self._lock:
            self._simulation_time = simulation_time
            self._is_running = is_running
            self._is_paused = is_paused
            self._speed_multiplier = speed_multiplier

    def remove_patient(self, patient_id: str) -> None:
        """Remove a patient from the bridge (e.g., on population change)."""
        with self._lock:
            self._patient_buffers.pop(patient_id, None)

    def clear_all(self) -> None:
        """Clear all patient data (e.g., on simulation reset)."""
        with self._lock:
            self._patient_buffers.clear()
            self._alert_queue.clear()
            self._simulation_time = 0.0
            self._push_count = 0

    # =========================================================================
    # Frontend (Consumer) Interface
    # =========================================================================

    def get_latest(self, patient_id: str) -> Optional[PatientSnapshot]:
        """
        Get the latest snapshot for a specific patient.

        This is the primary method for frontend components.
        Returns None if patient not found.

        Args:
            patient_id: Patient identifier

        Returns:
            Latest PatientSnapshot or None
        """
        with self._lock:
            buffer = self._patient_buffers.get(patient_id)
            if buffer and len(buffer) > 0:
                return buffer[-1]
            return None

    def get_latest_view(self, patient_id: str) -> Optional[PatientSnapshot]:
        """Alias for get_latest() for backwards compatibility."""
        return self.get_latest(patient_id)

    def get_ward_view(self) -> WardSnapshot:
        """
        Get a consistent snapshot of all patients for ward view.

        Returns a WardSnapshot with all patient data frozen at
        the same moment, ensuring consistent display.

        Returns:
            WardSnapshot with all patient data
        """
        with self._lock:
            patients = {}
            for patient_id, buffer in self._patient_buffers.items():
                if buffer and len(buffer) > 0:
                    patients[patient_id] = buffer[-1]

            return WardSnapshot(
                timestamp=time.time(),
                patients=patients,
                simulation_time=self._simulation_time,
                is_running=self._is_running,
                is_paused=self._is_paused,
                speed_multiplier=self._speed_multiplier,
            )

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent alerts from the alert queue.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts (newest first)
        """
        with self._lock:
            alerts = list(self._alert_queue)
            return list(reversed(alerts[-limit:]))

    def get_patient_history(
        self,
        patient_id: str,
        max_snapshots: int = 30,
    ) -> List[PatientSnapshot]:
        """
        Get historical snapshots for trend visualization.

        Args:
            patient_id: Patient identifier
            max_snapshots: Maximum snapshots to return

        Returns:
            List of snapshots (oldest first)
        """
        with self._lock:
            buffer = self._patient_buffers.get(patient_id)
            if not buffer:
                return []
            return list(buffer)[-max_snapshots:]

    def get_highlight_regions(
        self,
        patient_id: str,
    ) -> List[HighlightRegion]:
        """
        Get highlight regions (red zones) for a patient.

        Convenience method for plot rendering.

        Args:
            patient_id: Patient identifier

        Returns:
            List of HighlightRegion objects
        """
        snapshot = self.get_latest(patient_id)
        if snapshot:
            return snapshot.highlight_regions
        return []

    # =========================================================================
    # Statistics & Monitoring
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics for monitoring."""
        with self._lock:
            return {
                'push_count': self._push_count,
                'last_push_time': self._last_push_time,
                'num_patients': len(self._patient_buffers),
                'alert_queue_size': len(self._alert_queue),
                'buffer_sizes': {
                    pid: len(buf)
                    for pid, buf in self._patient_buffers.items()
                },
            }

    @property
    def is_active(self) -> bool:
        """Check if bridge has recent data."""
        return (time.time() - self._last_push_time) < 5.0


# =============================================================================
# Singleton Accessor
# =============================================================================

# Global instance
_global_bridge: Optional[DataBridge] = None
_global_lock = threading.Lock()


def get_data_bridge() -> DataBridge:
    """
    Get the DataBridge singleton instance.

    This function is safe to call from both:
    - Backend threads (PipelineAdapter, Orchestrator)
    - API handlers (FastAPI WebSocket endpoints)

    Uses a global singleton pattern for cross-module access.

    Returns:
        The DataBridge singleton instance
    """
    global _global_bridge

    if _global_bridge is None:
        with _global_lock:
            if _global_bridge is None:
                _global_bridge = DataBridge()

    return _global_bridge


def reset_data_bridge() -> None:
    """
    Reset the DataBridge singleton.

    Use with caution - only for testing or complete restart.
    """
    global _global_bridge

    with _global_lock:
        if _global_bridge is not None:
            _global_bridge.clear_all()
        _global_bridge = None


# =============================================================================
# Helper Functions
# =============================================================================

def create_snapshot_from_pipeline_result(
    patient_id: str,
    pipeline_result: Dict[str, Any],
    fhr_samples: List[float],
    uc_samples: List[float],
    patient_config: Any = None,
    active_events: List[str] = None,
) -> PatientSnapshot:
    """
    Create a PatientSnapshot from pipeline processing results.

    Helper function to convert PipelineAdapter output to the
    DataBridge format.

    Args:
        patient_id: Patient identifier
        pipeline_result: Result dictionary from PipelineAdapter.process_patient()
        fhr_samples: Recent FHR samples (last ~60 seconds)
        uc_samples: Recent UC samples
        patient_config: Optional patient configuration object
        active_events: List of active event type names

    Returns:
        PatientSnapshot ready for push_batch()
    """
    findings = pipeline_result.get('findings', {})
    baseline_info = findings.get('baseline', {})
    variability_info = findings.get('variability', {})
    decel_info = findings.get('decelerations', {})

    # Build highlight regions from findings
    highlight_regions = []

    # Add MHR alert region if present
    mhr_alert = pipeline_result.get('mhr_alert')
    if mhr_alert and mhr_alert.get('is_suspected'):
        highlight_regions.append(HighlightRegion(
            start_idx=max(0, len(fhr_samples) - 240),  # Last 60 seconds
            end_idx=len(fhr_samples),
            region_type='mhr_contamination',
            severity='warning',
            label='MHR?',
            color='rgba(255, 165, 0, 0.3)',  # Orange
        ))

    # Add decelerations as highlight regions
    if decel_info.get('late', 0) > 0:
        # Mark last portion as late decel zone
        highlight_regions.append(HighlightRegion(
            start_idx=max(0, len(fhr_samples) - 120),
            end_idx=len(fhr_samples),
            region_type='late_decel',
            severity='critical',
            label='Late',
            color='rgba(220, 53, 69, 0.35)',  # Red
        ))

    if decel_info.get('variable', 0) > 0:
        highlight_regions.append(HighlightRegion(
            start_idx=max(0, len(fhr_samples) - 120),
            end_idx=len(fhr_samples),
            region_type='variable_decel',
            severity='warning',
            label='Variable',
            color='rgba(255, 193, 7, 0.3)',  # Yellow
        ))

    # Extract name and bed from config
    name = ""
    bed_number = 0
    if patient_config:
        name = getattr(patient_config, 'name', '')
        bed_number = getattr(patient_config, 'bed_number', 0)

    return PatientSnapshot(
        patient_id=patient_id,
        timestamp=time.time(),
        category=pipeline_result.get('category', 1),
        confidence=pipeline_result.get('confidence', 0.0),
        fhr_recent=list(fhr_samples[-2400:]) if fhr_samples else [],
        uc_recent=list(uc_samples[-2400:]) if uc_samples else [],
        baseline=float(baseline_info.get('value', 140.0)),
        variability=float(variability_info.get('value', 10.0)),
        findings=findings,
        alerts=[],
        highlight_regions=highlight_regions,
        mhr_alert=mhr_alert,
        trend_data=pipeline_result.get('trend'),
        explanation=pipeline_result.get('explanation'),
        active_events=active_events or [],
        name=name,
        bed_number=bed_number,
    )
