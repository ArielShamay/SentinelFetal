# -*- coding: utf-8 -*-
"""
SentinelFetal V4 - Central Station Dashboard.

Real-time CTG monitoring with ECharts visualization.
Supports up to 20 simultaneous patients at 4Hz refresh.

Run from project root:
    cd SentinelFetal
    .venv\\Scripts\\activate  (Windows) or source .venv/bin/activate (Linux/Mac)
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add project root to Python path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

# Backend imports
from src.simulation import (
    SimulationOrchestrator,
    OrchestratorConfig,
    PipelineAdapter,
    PipelineAdapterConfig,
    EventType,
    LateDecelerationParams,
    VariableDecelerationParams,
    SinusoidalParams,
    TachysystoleParams,
    BradycardiaParams,
    TachycardiaParams,
)

# =============================================================================
# Constants & Colors (Medical Grade)
# =============================================================================

COLORS = {
    "background": "#FFFFFF",
    "grid": "#E0E0E0",
    "text": "#212121",
    "text_secondary": "#757575",
    "fhr_line": "#0D47A1",      # Deep Blue
    "uc_line": "#1B5E20",       # Dark Green
    "category_1": "#388E3C",    # Green - Normal
    "category_2": "#F57F17",    # Amber - Warning
    "category_3": "#D32F2F",    # Red - Critical
    "card_border": "#E0E0E0",
}

EVENT_OVERLAYS = {
    "LATE_DECELERATION": "rgba(211, 47, 47, 0.2)",
    "VARIABLE_DECELERATION": "rgba(245, 127, 23, 0.2)",
    "SINUSOIDAL_PATTERN": "rgba(183, 28, 28, 0.3)",
    "TACHYSYSTOLE": "rgba(255, 235, 59, 0.2)",
    "BRADYCARDIA": "rgba(211, 47, 47, 0.3)",
    "TACHYCARDIA": "rgba(255, 152, 0, 0.2)",
}

# Buffer configuration
MAX_SAMPLES = 2400    # 10 minutes at 4Hz
DISPLAY_SAMPLES = 600 # Downsampled for rendering


# =============================================================================
# Helper Functions
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely extract float from value that might be a dict with 'value' key."""
    if value is None:
        return default
    if isinstance(value, dict):
        return float(value.get("value", default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EventMarker:
    """Marker for event visualization on the graph."""
    event_name: str
    start_time: float  # Simulation time in seconds
    end_time: float    # Simulation time in seconds (or None if ongoing)
    is_active: bool = True


@dataclass
class PatientUIState:
    """UI state for a single patient - separate from backend state."""
    patient_id: str
    display_name: str
    bed_number: int = 1
    gestational_age: str = "39w"

    # Data buffers (deque for O(1) operations)
    fhr_buffer: deque = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))
    uc_buffer: deque = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))
    timestamp_buffer: deque = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))

    # Latest analysis (from backend)
    category: int = 1
    baseline_value: float = 140.0
    variability_value: float = 12.0
    confidence: float = 0.0
    active_events: List[str] = field(default_factory=list)
    event_markers: List[EventMarker] = field(default_factory=list)  # For graph marking
    current_sim_time: float = 0.0  # Current simulation time


# =============================================================================
# Backend Initialization
# =============================================================================

@st.cache_resource
def get_pipeline_adapter() -> PipelineAdapter:
    """Singleton pipeline adapter for Gen3.5 processing."""
    config = PipelineAdapterConfig(
        sampling_rate=4.0,
        min_data_seconds=60.0
    )
    return PipelineAdapter(config)


def create_orchestrator(num_patients: int) -> SimulationOrchestrator:
    """
    Create a new orchestrator instance.

    Note: We don't cache this because patient count can change.
    The orchestrator is stored in session_state instead.
    """
    adapter = get_pipeline_adapter()

    config = OrchestratorConfig(
        num_patients=num_patients,
        sampling_rate=4.0,
        tick_interval_seconds=1.0
    )

    orchestrator = SimulationOrchestrator(
        config=config,
        processing_callback=lambda pid, data: adapter.process_patient(pid, data)
    )

    return orchestrator


def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "current_view": "grid",         # "grid" or "detail"
        "selected_patient_id": None,
        "patient_states": {},           # Dict[str, PatientUIState]
        "god_mode_enabled": False,
        "is_running": False,
        "num_patients": 8,
        "orchestrator": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Data Processing
# =============================================================================

def minmax_downsample(data: List[float], target_points: int) -> List[float]:
    """Downsample preserving peaks and valleys using min-max algorithm."""
    if len(data) <= target_points:
        return list(data)

    arr = np.array(data)
    # Each chunk produces 2 values (min, max)
    chunk_size = max(1, len(arr) // (target_points // 2))

    result = []
    for i in range(0, len(arr) - chunk_size + 1, chunk_size):
        chunk = arr[i:i + chunk_size]
        result.extend([float(chunk.min()), float(chunk.max())])
        if len(result) >= target_points:
            break

    return result[:target_points]


def fetch_and_update_patient(orchestrator: SimulationOrchestrator, patient_id: str) -> None:
    """Fetch latest data from orchestrator and update UI state for one patient."""
    # Get patient status
    patient = orchestrator.get_patient(patient_id)
    if not patient:
        return

    status = patient.get_status()

    # Create UI state if not exists
    if patient_id not in st.session_state.patient_states:
        st.session_state.patient_states[patient_id] = PatientUIState(
            patient_id=patient_id,
            display_name=status.get("name", patient_id),
            bed_number=status.get("bed_number", 1)
        )

    ui_state = st.session_state.patient_states[patient_id]

    # Get full buffer data
    patient_data = orchestrator.get_patient_data(patient_id, duration_minutes=10)
    if patient_data:
        fhr = patient_data.get("fhr", np.array([]))
        uc = patient_data.get("uc", np.array([]))
        timestamps = patient_data.get("timestamps", np.array([]))

        # Replace buffer contents
        ui_state.fhr_buffer.clear()
        ui_state.uc_buffer.clear()
        ui_state.timestamp_buffer.clear()

        if len(fhr) > 0:
            ui_state.fhr_buffer.extend(fhr.tolist())
            ui_state.uc_buffer.extend(uc.tolist())
            ui_state.timestamp_buffer.extend(timestamps.tolist())

    # Update analysis results
    ui_state.category = status.get("category", 1)
    ui_state.active_events = status.get("active_events", [])
    ui_state.current_sim_time = status.get("simulation_time", 0.0)

    # Get event markers with timestamps from active events
    active_event_objects = patient.get_active_events()
    ui_state.event_markers = [
        EventMarker(
            event_name=e.event_type.name,
            start_time=e.start_time,
            end_time=e.end_time,
            is_active=not e.is_expired(ui_state.current_sim_time)
        )
        for e in active_event_objects
    ]

    # Get findings if available
    if hasattr(patient, 'latest_findings') and patient.latest_findings:
        findings = patient.latest_findings
        # Handle nested structure: findings["baseline"] is a dict with "value" key
        baseline_data = findings.get("baseline", {})
        if isinstance(baseline_data, dict):
            ui_state.baseline_value = baseline_data.get("value", 140.0)
        else:
            ui_state.baseline_value = float(baseline_data) if baseline_data else 140.0

        variability_data = findings.get("variability", {})
        if isinstance(variability_data, dict):
            ui_state.variability_value = variability_data.get("value", 12.0)
        else:
            ui_state.variability_value = float(variability_data) if variability_data else 12.0


def fetch_all_patients(orchestrator: SimulationOrchestrator) -> None:
    """Fetch data for all patients."""
    statuses = orchestrator.get_all_patients_status()
    for status in statuses:
        fetch_and_update_patient(orchestrator, status["patient_id"])


# =============================================================================
# ECharts Configuration
# =============================================================================

# Display window configuration
DISPLAY_WINDOW_SECONDS = 120  # Show 2 minutes of data (sliding window)
DISPLAY_WINDOW_SAMPLES = int(DISPLAY_WINDOW_SECONDS * 4)  # 480 samples at 4Hz


def build_ctg_echarts_options(
    fhr_data: List[float],
    uc_data: List[float],
    timestamps: List[float],
    event_markers: List[EventMarker] = None,
    current_sim_time: float = 0.0,
    compact: bool = True,
    show_zoom: bool = False
) -> Dict[str, Any]:
    """
    Build ECharts options for dual-track CTG visualization.

    Features:
    - Sliding window (shows last 2 minutes, scrollable back)
    - Clear time axis with MM:SS labels
    - Event markers as vertical red lines (start/end of events)
    - No compression - data scrolls off screen

    Critical settings for 4Hz performance:
    - animation: false
    - symbol: "none"
    - Canvas renderer (default in streamlit-echarts)
    """
    # Use sliding window - only show last DISPLAY_WINDOW_SAMPLES samples
    if len(fhr_data) > DISPLAY_WINDOW_SAMPLES:
        # Take only the last window for display (data scrolls off)
        fhr_display = list(fhr_data)[-DISPLAY_WINDOW_SAMPLES:]
        uc_display = list(uc_data)[-DISPLAY_WINDOW_SAMPLES:]
        ts_display = list(timestamps)[-DISPLAY_WINDOW_SAMPLES:] if timestamps else []
    else:
        fhr_display = list(fhr_data)
        uc_display = list(uc_data)
        ts_display = list(timestamps) if timestamps else []

    # Apply min-max downsampling for performance if still too many points
    if len(fhr_display) > DISPLAY_SAMPLES:
        fhr_display = minmax_downsample(fhr_display, DISPLAY_SAMPLES)
        uc_display = minmax_downsample(uc_display, DISPLAY_SAMPLES)
        # Resample timestamps to match
        if ts_display:
            step = max(1, len(ts_display) // len(fhr_display))
            ts_display = ts_display[::step][:len(fhr_display)]

    # Generate time labels as MM:SS (relative to window start)
    time_labels = []
    if ts_display and len(ts_display) > 0:
        window_start = ts_display[0]
        for ts in ts_display[:len(fhr_display)]:
            relative_sec = ts - window_start
            minutes = int(relative_sec // 60)
            seconds = int(relative_sec % 60)
            time_labels.append(f"{minutes}:{seconds:02d}")
    else:
        # Fallback: generate labels based on sample count
        for i in range(len(fhr_display)):
            relative_sec = i / 4.0  # 4Hz sampling
            minutes = int(relative_sec // 60)
            seconds = int(relative_sec % 60)
            time_labels.append(f"{minutes}:{seconds:02d}")

    # Build event markers as vertical lines instead of full overlay
    mark_lines_fhr = []
    mark_lines_uc = []

    if event_markers and ts_display and len(ts_display) > 0:
        window_start_time = ts_display[0]
        window_end_time = ts_display[-1] if ts_display else current_sim_time

        for marker in event_markers:
            # Convert event times to x-axis indices
            # Start line (always show if event started within or before window)
            if marker.start_time >= window_start_time:
                # Event started within the visible window
                start_relative = marker.start_time - window_start_time
                start_idx = int(start_relative * 4)  # 4Hz
                if 0 <= start_idx < len(fhr_display):
                    start_line = {
                        "xAxis": start_idx,
                        "lineStyle": {"color": "#D32F2F", "width": 2, "type": "solid"},
                        "label": {"show": True, "formatter": "START", "fontSize": 8, "color": "#D32F2F"}
                    }
                    mark_lines_fhr.append(start_line)
                    mark_lines_uc.append({"xAxis": start_idx, "lineStyle": {"color": "#D32F2F", "width": 2, "type": "solid"}})

            # End line
            if marker.is_active:
                # Event ongoing - end line at front (rightmost edge)
                end_idx = len(fhr_display) - 1
                end_line = {
                    "xAxis": end_idx,
                    "lineStyle": {"color": "#D32F2F", "width": 2, "type": "dashed"},
                    "label": {"show": True, "formatter": "ACTIVE", "fontSize": 8, "color": "#D32F2F"}
                }
                mark_lines_fhr.append(end_line)
                mark_lines_uc.append({"xAxis": end_idx, "lineStyle": {"color": "#D32F2F", "width": 2, "type": "dashed"}})
            elif marker.end_time >= window_start_time:
                # Event ended - show end line at actual end position
                end_relative = marker.end_time - window_start_time
                end_idx = int(end_relative * 4)  # 4Hz
                if 0 <= end_idx < len(fhr_display):
                    end_line = {
                        "xAxis": end_idx,
                        "lineStyle": {"color": "#D32F2F", "width": 2, "type": "solid"},
                        "label": {"show": True, "formatter": "END", "fontSize": 8, "color": "#D32F2F"}
                    }
                    mark_lines_fhr.append(end_line)
                    mark_lines_uc.append({"xAxis": end_idx, "lineStyle": {"color": "#D32F2F", "width": 2, "type": "solid"}})

    # Chart height ratios
    fhr_height = "38%" if compact else "42%"
    uc_height = "32%" if compact else "36%"
    uc_top = "58%" if compact else "55%"

    options = {
        "animation": False,  # CRITICAL for real-time performance
        "backgroundColor": COLORS["background"],
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
                "type": "cross",
                "link": [{"xAxisIndex": "all"}]
            },
            "formatter": None  # Let ECharts handle auto-formatting
        },
        "grid": [
            {"left": 50, "right": 15, "top": 15, "height": fhr_height},
            {"left": 50, "right": 15, "top": uc_top, "height": uc_height}
        ],
        "xAxis": [
            {
                "type": "category",
                "data": time_labels,
                "gridIndex": 0,
                "axisLabel": {"show": False},  # Hide labels on FHR chart
                "axisTick": {"show": False},
                "axisLine": {"show": True, "lineStyle": {"color": COLORS["grid"]}},
                "splitLine": {"show": True, "lineStyle": {"color": COLORS["grid"], "type": "dashed"}}
            },
            {
                "type": "category",
                "data": time_labels,
                "gridIndex": 1,
                "name": "Time (MM:SS)" if not compact else "",
                "nameLocation": "center",
                "nameGap": 25,
                "nameTextStyle": {"fontSize": 10, "color": COLORS["text_secondary"]},
                "axisLabel": {
                    "show": True,
                    "fontSize": 9,
                    "color": COLORS["text_secondary"],
                    "interval": max(0, len(time_labels) // 8 - 1)  # Show ~8 labels
                },
                "axisTick": {"show": True},
                "axisLine": {"show": True, "lineStyle": {"color": COLORS["grid"]}},
                "splitLine": {"show": True, "lineStyle": {"color": COLORS["grid"], "type": "dashed"}}
            }
        ],
        "yAxis": [
            {
                "type": "value",
                "name": "FHR",
                "nameLocation": "middle",
                "nameGap": 35,
                "nameTextStyle": {"fontSize": 10, "color": COLORS["text_secondary"]},
                "min": 50,
                "max": 210,
                "interval": 30,
                "gridIndex": 0,
                "axisLabel": {"fontSize": 9, "color": COLORS["text_secondary"]},
                "axisLine": {"show": True, "lineStyle": {"color": COLORS["grid"]}},
                "splitLine": {"show": True, "lineStyle": {"color": COLORS["grid"], "type": "dashed"}}
            },
            {
                "type": "value",
                "name": "UC",
                "nameLocation": "middle",
                "nameGap": 35,
                "nameTextStyle": {"fontSize": 10, "color": COLORS["text_secondary"]},
                "min": 0,
                "max": 100,
                "interval": 25,
                "gridIndex": 1,
                "axisLabel": {"fontSize": 9, "color": COLORS["text_secondary"]},
                "axisLine": {"show": True, "lineStyle": {"color": COLORS["grid"]}},
                "splitLine": {"show": True, "lineStyle": {"color": COLORS["grid"], "type": "dashed"}}
            }
        ],
        "series": [
            {
                "name": "FHR",
                "type": "line",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": fhr_display,
                "symbol": "none",
                "lineStyle": {"color": COLORS["fhr_line"], "width": 1.5},
                "markLine": {
                    "symbol": "none",
                    "data": mark_lines_fhr
                } if mark_lines_fhr else None
            },
            {
                "name": "UC",
                "type": "line",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "data": uc_display,
                "symbol": "none",
                "lineStyle": {"color": COLORS["uc_line"], "width": 1.5},
                "markLine": {
                    "symbol": "none",
                    "data": mark_lines_uc
                } if mark_lines_uc else None
            }
        ]
    }

    # Add dataZoom for scrolling back through history
    if show_zoom:
        options["dataZoom"] = [
            {
                "type": "slider",
                "xAxisIndex": [0, 1],
                "start": 0,
                "end": 100,
                "bottom": 5,
                "height": 25,
                "brushSelect": False
            },
            {
                "type": "inside",
                "xAxisIndex": [0, 1],
                "zoomOnMouseWheel": True,
                "moveOnMouseMove": True
            }
        ]
    else:
        # Even in compact mode, allow mouse wheel scroll
        options["dataZoom"] = [
            {
                "type": "inside",
                "xAxisIndex": [0, 1],
                "zoomOnMouseWheel": False,
                "moveOnMouseMove": True,
                "moveOnMouseWheel": True
            }
        ]

    return options


# =============================================================================
# God Mode Event Injection
# =============================================================================

GOD_MODE_EVENTS = {
    "Late Decel": (EventType.LATE_DECELERATION, lambda: LateDecelerationParams.moderate()),
    "Variable Decel": (EventType.VARIABLE_DECELERATION, lambda: VariableDecelerationParams.moderate()),
    "Sinusoidal": (EventType.SINUSOIDAL_PATTERN, lambda: SinusoidalParams.typical()),
    "Tachysystole": (EventType.TACHYSYSTOLE, lambda: TachysystoleParams.mild()),
    "Bradycardia": (EventType.BRADYCARDIA, lambda: BradycardiaParams.moderate()),
    "Tachycardia": (EventType.TACHYCARDIA, lambda: TachycardiaParams.moderate()),
}


def inject_event(patient_id: str, event_type: EventType, params) -> bool:
    """Inject an event via the orchestrator."""
    orchestrator = st.session_state.get("orchestrator")
    if orchestrator:
        duration = getattr(params, 'duration_seconds', 300.0)
        event = orchestrator.inject_event(
            patient_id=patient_id,
            event_type=event_type,
            params=params,
            duration_seconds=duration
        )
        return event is not None
    return False


def render_god_mode_controls(patient_id: str) -> None:
    """Render fault injection controls for a patient."""
    with st.expander("Dev Tools", expanded=False):
        col1, col2 = st.columns(2)

        items = list(GOD_MODE_EVENTS.items())
        for idx, (label, (event_type, params_factory)) in enumerate(items):
            col = col1 if idx % 2 == 0 else col2
            with col:
                if st.button(label, key=f"inject_{label}_{patient_id}", use_container_width=True):
                    params = params_factory()
                    if inject_event(patient_id, event_type, params):
                        st.success(f"Injected {label}")
                    else:
                        st.error("Injection failed")

        # Reset button
        if st.button("Reset Normal", key=f"reset_{patient_id}", use_container_width=True):
            orchestrator = st.session_state.get("orchestrator")
            if orchestrator:
                patient = orchestrator.get_patient(patient_id)
                if patient:
                    patient.clear_events()
                    st.success("Events cleared")


# =============================================================================
# UI Components
# =============================================================================

def get_category_color(category: int) -> str:
    """Get color for category badge."""
    return {
        1: COLORS["category_1"],
        2: COLORS["category_2"],
        3: COLORS["category_3"]
    }.get(category, COLORS["text"])


def calculate_grid_columns(patient_count: int) -> int:
    """Dynamic column layout per PRD requirements."""
    if patient_count <= 1:
        return 1
    elif patient_count <= 4:
        return 2
    elif patient_count <= 12:
        return 3
    else:
        return 4


def render_patient_card(patient_id: str, patient_state: PatientUIState) -> None:
    """Render a single patient monitoring card."""
    category_color = get_category_color(patient_state.category)

    # Card container with border
    border_color = category_color if patient_state.category > 1 else COLORS["card_border"]
    st.markdown(
        f'<div style="border-left: 4px solid {border_color}; padding-left: 8px; margin-bottom: 4px;">',
        unsafe_allow_html=True
    )

    # Header row
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button(
            f"{patient_state.display_name}",
            key=f"btn_{patient_id}",
            use_container_width=True
        ):
            st.session_state.current_view = "detail"
            st.session_state.selected_patient_id = patient_id
            st.rerun()
        st.caption(f"{patient_id} | Bed {patient_state.bed_number}")

    with col2:
        st.markdown(
            f'<div style="text-align: center; padding: 4px; background: {category_color}; '
            f'color: white; border-radius: 4px; font-weight: bold;">Cat {patient_state.category}</div>',
            unsafe_allow_html=True
        )

    # CTG Chart
    if len(patient_state.fhr_buffer) > 10:
        chart_options = build_ctg_echarts_options(
            fhr_data=list(patient_state.fhr_buffer),
            uc_data=list(patient_state.uc_buffer),
            timestamps=list(patient_state.timestamp_buffer),
            event_markers=patient_state.event_markers,
            current_sim_time=patient_state.current_sim_time,
            compact=True
        )

        st_echarts(
            options=chart_options,
            height="200px",
            key=f"chart_{patient_id}"
        )
    else:
        st.info("Collecting data...")

    # Metrics row
    col_bl, col_var = st.columns(2)
    with col_bl:
        baseline_val = safe_float(patient_state.baseline_value, 140.0)
        st.metric("Baseline", f"{baseline_val:.0f} bpm")
    with col_var:
        var_val = safe_float(patient_state.variability_value, 12.0)
        st.metric("Variability", f"{var_val:.1f} bpm")

    # Active events indicator
    if patient_state.active_events:
        events_str = ", ".join(patient_state.active_events)
        st.warning(f"Active: {events_str}")

    # God Mode controls
    if st.session_state.god_mode_enabled:
        render_god_mode_controls(patient_id)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_grid_view() -> None:
    """Render the patient grid view."""
    orchestrator = st.session_state.get("orchestrator")

    if not orchestrator or not st.session_state.is_running:
        st.info("Click 'Start Simulation' to begin monitoring")
        return

    # Fetch latest data
    fetch_all_patients(orchestrator)

    patient_states = st.session_state.patient_states
    patient_count = len(patient_states)

    if patient_count == 0:
        st.warning("No patients available")
        return

    # Calculate responsive columns
    num_columns = calculate_grid_columns(patient_count)
    columns = st.columns(num_columns)

    # Render patient cards in grid
    for idx, (patient_id, patient_state) in enumerate(patient_states.items()):
        col_idx = idx % num_columns
        with columns[col_idx]:
            render_patient_card(patient_id, patient_state)


def render_detail_view() -> None:
    """Expanded single-patient view with full history."""
    patient_id = st.session_state.selected_patient_id

    if st.button("Back to Dashboard", type="primary"):
        st.session_state.current_view = "grid"
        st.session_state.selected_patient_id = None
        st.rerun()

    # Refresh data for this patient
    orchestrator = st.session_state.get("orchestrator")
    if orchestrator and st.session_state.is_running:
        fetch_and_update_patient(orchestrator, patient_id)

    patient_state = st.session_state.patient_states.get(patient_id)
    if not patient_state:
        st.error(f"Patient {patient_id} not found")
        return

    # Header
    category_color = get_category_color(patient_state.category)
    st.markdown(
        f'## {patient_state.display_name} '
        f'<span style="background: {category_color}; color: white; padding: 4px 12px; '
        f'border-radius: 4px; font-size: 0.8em;">Category {patient_state.category}</span>',
        unsafe_allow_html=True
    )
    st.caption(f"Patient ID: {patient_id} | Bed: {patient_state.bed_number}")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Category", patient_state.category)
    with col2:
        baseline_val = safe_float(patient_state.baseline_value, 140.0)
        st.metric("Baseline", f"{baseline_val:.0f} bpm")
    with col3:
        var_val = safe_float(patient_state.variability_value, 12.0)
        st.metric("Variability", f"{var_val:.1f} bpm")
    with col4:
        if patient_state.fhr_buffer:
            current_fhr = safe_float(patient_state.fhr_buffer[-1], 140.0)
            st.metric("Current FHR", f"{current_fhr:.0f} bpm")

    st.markdown("---")

    # Full CTG chart with zoom
    if len(patient_state.fhr_buffer) > 10:
        st.subheader("CTG Trace (2-minute sliding window)")

        chart_options = build_ctg_echarts_options(
            fhr_data=list(patient_state.fhr_buffer),
            uc_data=list(patient_state.uc_buffer),
            timestamps=list(patient_state.timestamp_buffer),
            event_markers=patient_state.event_markers,
            current_sim_time=patient_state.current_sim_time,
            compact=False,
            show_zoom=True
        )

        st_echarts(
            options=chart_options,
            height="450px",
            key=f"detail_chart_{patient_id}"
        )
    else:
        st.info("Collecting data... Please wait.")

    # Active events section
    if patient_state.active_events:
        st.markdown("### Active Events")
        for event_name in patient_state.active_events:
            color = EVENT_OVERLAYS.get(event_name, "rgba(128, 128, 128, 0.5)")
            st.markdown(
                f'<div style="padding: 8px; background: {color}; border-radius: 4px; '
                f'margin-bottom: 4px;">{event_name.replace("_", " ").title()}</div>',
                unsafe_allow_html=True
            )

    # God Mode controls in detail view
    if st.session_state.god_mode_enabled:
        st.markdown("### Event Injection (God Mode)")
        render_god_mode_controls(patient_id)


# =============================================================================
# CSS Styling
# =============================================================================

def inject_custom_css() -> None:
    """Inject medical-grade CSS styling."""
    st.markdown(f"""
    <style>
    /* Main container */
    .main .block-container {{
        padding: 1rem 1.5rem;
        max-width: 100%;
    }}

    /* Hide Streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* White background */
    .stApp {{
        background-color: {COLORS["background"]};
    }}

    /* Metric styling */
    [data-testid="stMetricValue"] {{
        font-size: 1.1rem;
        color: {COLORS["text"]};
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 0.75rem;
        color: {COLORS["text_secondary"]};
    }}

    /* Button styling */
    .stButton > button {{
        border: 1px solid {COLORS["card_border"]};
        background-color: {COLORS["background"]};
        color: {COLORS["text"]};
    }}

    .stButton > button:hover {{
        background-color: #F5F5F5;
        border-color: {COLORS["text_secondary"]};
    }}

    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: #FAFAFA;
        border-radius: 4px;
    }}

    /* Remove default margins */
    .element-container {{
        margin-bottom: 0.5rem;
    }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Main Application
# =============================================================================

def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="SentinelFetal Central Station",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    inject_custom_css()
    init_session_state()

    # Header
    st.markdown("# SentinelFetal Central Station")
    st.markdown("**Real-time CTG Monitoring System**")
    st.markdown("---")

    # Control bar
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])

    with col1:
        num_patients = st.slider(
            "Number of Patients",
            min_value=1,
            max_value=20,
            value=st.session_state.num_patients,
            key="patient_slider"
        )
        st.session_state.num_patients = num_patients

    with col2:
        if st.button("Start", type="primary", use_container_width=True):
            # Create or recreate orchestrator with new patient count
            orchestrator = create_orchestrator(num_patients)
            orchestrator.start()
            st.session_state.orchestrator = orchestrator
            st.session_state.is_running = True
            st.session_state.patient_states = {}  # Clear old states
            st.rerun()

    with col3:
        if st.button("Stop", use_container_width=True):
            orchestrator = st.session_state.get("orchestrator")
            if orchestrator:
                orchestrator.stop()
            st.session_state.is_running = False

    with col4:
        st.session_state.god_mode_enabled = st.checkbox(
            "God Mode",
            value=st.session_state.god_mode_enabled
        )

    with col5:
        if st.session_state.is_running:
            orchestrator = st.session_state.get("orchestrator")
            if orchestrator:
                sim_time = orchestrator.get_simulation_time_formatted()
                st.markdown(f"**Status:** Running | **Sim Time:** {sim_time}")
            else:
                st.markdown("**Status:** Running")
        else:
            st.markdown("**Status:** Stopped")

    st.markdown("---")

    # Main content
    if st.session_state.current_view == "grid":
        render_grid_view()
    else:
        render_detail_view()

    # Auto-refresh when running
    if st.session_state.is_running:
        time.sleep(0.25)  # 4Hz refresh rate
        st.rerun()


if __name__ == "__main__":
    main()
