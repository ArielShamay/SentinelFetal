# SentinelFetal UI V4.0 — Technical Specifications (SPECS)

**Document Version:** 1.0  
**Date:** January 2026  
**Target:** AI Coding Agent Implementation

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BROWSER (Client)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                      Apache ECharts (Canvas)                         │ │
│  │   20 × Dual-Graph Patient Cards @ 4Hz refresh                       │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ JSON (options + data arrays)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT SERVER                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  st.fragment    │  │  Session State  │  │  streamlit-echarts      │  │
│  │  (4Hz polling)  │  │  (Data Store)   │  │  (Component Bridge)     │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────────────────┘  │
│           │                    │                                         │
│           └────────────────────┼─────────────────────────────────────────┤
│                                ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    SimulationOrchestrator                            │ │
│  │   - Patient lifecycle management                                     │ │
│  │   - Event injection (God Mode)                                       │ │
│  │   - Tick coordination                                                │ │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                      PipelineAdapter                                  │ │
│  │   Preprocess → FSQI → Rules → MiniRocket → Fusion → Override         │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 File Structure

```
src/ui/
├── app.py                    # Main Streamlit application (entry point)
├── components/
│   ├── __init__.py
│   ├── patient_card.py       # Single patient card component
│   ├── patient_grid.py       # Grid layout manager
│   ├── detail_view.py        # Drill-down single patient view
│   ├── sidebar.py            # Sidebar controls and patient admission
│   └── god_mode.py           # Fault injection controls
├── charts/
│   ├── __init__.py
│   ├── echarts_config.py     # ECharts option builders
│   ├── ctg_dual_track.py     # Dual FHR+UC chart configuration
│   └── mark_areas.py         # Event highlighting overlays
├── state/
│   ├── __init__.py
│   ├── session_manager.py    # Session state initialization
│   ├── patient_store.py      # Patient data management
│   └── event_bus.py          # Inter-component communication
├── styles/
│   ├── __init__.py
│   ├── colors.py             # Color constants
│   ├── css.py                # Custom CSS injection
│   └── layout.py             # Layout calculations
└── utils/
    ├── __init__.py
    ├── downsampling.py       # Data reduction for rendering
    └── time_format.py        # Timestamp formatting
```

---

## 2. Core Components

### 2.1 Main Application (`app.py`)

```python
"""
SentinelFetal Central Station - Main Application

Entry point for the Streamlit UI. Orchestrates layout and refresh cycles.
"""

import streamlit as st
from streamlit_echarts import st_echarts

from src.ui.state.session_manager import init_session_state
from src.ui.components.patient_grid import render_patient_grid
from src.ui.components.detail_view import render_detail_view
from src.ui.components.sidebar import render_sidebar
from src.ui.styles.css import inject_custom_css

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="SentinelFetal Central Station",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Inject custom CSS
    inject_custom_css()
    
    # Render sidebar (patient admission, settings)
    render_sidebar()
    
    # Main content area
    if st.session_state.current_view == "grid":
        render_patient_grid()
    else:
        render_detail_view(st.session_state.selected_patient_id)

if __name__ == "__main__":
    main()
```

### 2.2 Session State Manager (`state/session_manager.py`)

```python
"""
Session State Manager

Initializes and manages all session state variables.
Uses the Producer-Consumer pattern for data flow.
"""

import streamlit as st
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import time

@dataclass
class PatientState:
    """State container for a single patient."""
    patient_id: str
    display_name: str
    gestational_age: str
    room: str
    notes: str
    
    # Data buffers (O(1) operations via deque)
    fhr_buffer: deque = field(default_factory=lambda: deque(maxlen=2400))
    uc_buffer: deque = field(default_factory=lambda: deque(maxlen=2400))
    timestamp_buffer: deque = field(default_factory=lambda: deque(maxlen=2400))
    
    # Latest analysis results
    category: int = 1
    baseline_value: float = 140.0
    variability_value: float = 12.0
    confidence: float = 0.0
    
    # Active events (for highlighting)
    events: list = field(default_factory=list)
    
    # Simulation mode (God Mode)
    sim_mode: str = "normal"  # normal, late_decel, variable_decel, sinusoidal, etc.

def init_session_state() -> None:
    """Initialize all session state variables."""
    
    defaults = {
        # View state
        "current_view": "grid",  # "grid" or patient_id
        "selected_patient_id": None,
        
        # Patient data
        "patients": {},  # Dict[str, PatientState]
        "patient_count": 0,
        "max_patients": 20,
        
        # Simulation control
        "is_running": True,
        "god_mode_enabled": False,
        
        # Performance settings
        "refresh_rate_hz": 4,
        "history_minutes": 10,
        
        # Orchestrator reference
        "orchestrator": None,
        "pipeline_adapter": None,
        
        # Timestamps
        "last_update_time": time.time(),
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
```

### 2.3 Patient Grid (`components/patient_grid.py`)

```python
"""
Patient Grid Component

Renders the Central Station grid view with all active patients.
Uses st.fragment for independent refresh cycles.
"""

import streamlit as st
from streamlit_echarts import st_echarts

from src.ui.components.patient_card import render_patient_card
from src.ui.styles.layout import calculate_grid_columns

@st.fragment(run_every=0.25)  # 4Hz refresh
def render_patient_grid():
    """
    Render the patient grid with automatic 4Hz refresh.
    
    This function runs in its own fragment, decoupled from the main page.
    Updates to the grid do not trigger full page reruns.
    """
    patients = st.session_state.patients
    patient_count = len(patients)
    
    if patient_count == 0:
        st.info("No patients admitted. Use the sidebar to add a patient.")
        return
    
    # Calculate optimal column layout
    num_columns = calculate_grid_columns(patient_count)
    
    # Create column containers
    columns = st.columns(num_columns)
    
    # Render each patient card
    for idx, (patient_id, patient_state) in enumerate(patients.items()):
        col_idx = idx % num_columns
        
        with columns[col_idx]:
            render_patient_card(patient_id, patient_state)
    
    # Update data from orchestrator
    _fetch_latest_data()

def _fetch_latest_data():
    """Fetch latest data from the simulation orchestrator."""
    orchestrator = st.session_state.get("orchestrator")
    if orchestrator is None:
        return
    
    # Get latest tick data for all patients
    for patient_id, patient_state in st.session_state.patients.items():
        latest = orchestrator.get_patient_data(patient_id)
        if latest:
            # Append to buffers (O(1) with deque)
            patient_state.fhr_buffer.append(latest["fhr"])
            patient_state.uc_buffer.append(latest["uc"])
            patient_state.timestamp_buffer.append(latest["timestamp"])
            
            # Update analysis results
            patient_state.category = latest.get("category", 1)
            patient_state.baseline_value = latest.get("baseline", 140.0)
            patient_state.variability_value = latest.get("variability", 12.0)
            patient_state.events = latest.get("events", [])
```

### 2.4 Patient Card (`components/patient_card.py`)

```python
"""
Patient Card Component

Renders a single patient monitoring card with dual CTG graphs.
"""

import streamlit as st
from streamlit_echarts import st_echarts

from src.ui.charts.ctg_dual_track import build_ctg_options
from src.ui.styles.colors import COLORS, get_category_color
from src.ui.components.god_mode import render_god_mode_controls

def render_patient_card(patient_id: str, patient_state) -> None:
    """
    Render a single patient card.
    
    Args:
        patient_id: Unique patient identifier
        patient_state: PatientState object with data and metadata
    """
    category_color = get_category_color(patient_state.category)
    
    # Card container with border color based on category
    with st.container():
        # Header row
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Clickable patient ID to open detail view
            if st.button(
                f"**{patient_state.display_name or patient_id}**",
                key=f"btn_{patient_id}",
                use_container_width=True
            ):
                st.session_state.current_view = "detail"
                st.session_state.selected_patient_id = patient_id
                st.rerun()
        
        with col2:
            # Gestational age
            st.caption(patient_state.gestational_age)
        
        with col3:
            # Discharge button
            if st.button("✕", key=f"discharge_{patient_id}"):
                _confirm_discharge(patient_id)
        
        # Category indicator
        st.markdown(
            f'<span style="color: {category_color}; font-weight: bold;">'
            f'● Category {patient_state.category}</span>',
            unsafe_allow_html=True
        )
        
        # CTG Dual-Track Chart
        chart_options = build_ctg_options(
            fhr_data=list(patient_state.fhr_buffer),
            uc_data=list(patient_state.uc_buffer),
            timestamps=list(patient_state.timestamp_buffer),
            events=patient_state.events,
            compact=True  # Grid view uses compact charts
        )
        
        st_echarts(
            options=chart_options,
            height="200px",
            key=f"chart_{patient_id}"
        )
        
        # Metrics row
        col_bl, col_var = st.columns(2)
        with col_bl:
            st.metric("Baseline", f"{patient_state.baseline_value:.0f} bpm")
        with col_var:
            st.metric("Variability", f"{patient_state.variability_value:.1f} bpm")
        
        # Alert display (if any events)
        _render_alerts(patient_state.events)
        
        # God Mode controls (if enabled)
        if st.session_state.god_mode_enabled:
            render_god_mode_controls(patient_id)

def _render_alerts(events: list) -> None:
    """Render alert text for detected events."""
    for event in events:
        event_type = event.get("type", "Unknown")
        start_time = event.get("start_time", "")
        
        color = COLORS["category_3"] if event_type in ["SINUSOIDAL", "LATE"] else COLORS["category_2"]
        
        st.markdown(
            f'<p style="color: {color}; margin: 0; font-size: 12px;">'
            f'⚠ {event_type} detected at {start_time}</p>',
            unsafe_allow_html=True
        )

def _confirm_discharge(patient_id: str) -> None:
    """Show discharge confirmation dialog."""
    # In production, use st.dialog or confirmation modal
    del st.session_state.patients[patient_id]
    st.session_state.patient_count -= 1
    st.rerun()
```

---

## 3. ECharts Configuration

### 3.1 Dual-Track CTG Chart (`charts/ctg_dual_track.py`)

```python
"""
CTG Dual-Track Chart Configuration

Builds ECharts options for synchronized FHR + UC graphs.
"""

from typing import List, Dict, Any, Optional
from src.ui.styles.colors import COLORS
from src.ui.charts.mark_areas import build_event_mark_areas
from src.ui.utils.downsampling import minmax_downsample
from src.ui.utils.time_format import format_timestamps

def build_ctg_options(
    fhr_data: List[float],
    uc_data: List[float],
    timestamps: List[float],
    events: List[Dict] = None,
    compact: bool = False,
    show_zoom: bool = False
) -> Dict[str, Any]:
    """
    Build ECharts configuration for dual-track CTG visualization.
    
    Args:
        fhr_data: Fetal heart rate values (bpm)
        uc_data: Uterine contraction values (mmHg)
        timestamps: Unix timestamps
        events: List of detected events for highlighting
        compact: If True, uses minimal styling for grid view
        show_zoom: If True, enables dataZoom slider
    
    Returns:
        ECharts options dictionary
    """
    # Downsample if needed for performance (max 600 points)
    if len(fhr_data) > 600:
        fhr_data = minmax_downsample(fhr_data, 600)
        uc_data = minmax_downsample(uc_data, 600)
        timestamps = timestamps[::len(timestamps)//600][:600]
    
    # Format timestamps for display
    time_labels = format_timestamps(timestamps)
    
    # Build event highlight areas
    mark_areas = build_event_mark_areas(events, timestamps) if events else []
    
    options = {
        # CRITICAL: Disable all animations for real-time performance
        "animation": False,
        
        # Tooltip configuration (synchronized across both graphs)
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
                "type": "cross",
                "link": [{"xAxisIndex": "all"}]  # Sync crosshair
            }
        },
        
        # Grid layout for dual graphs
        "grid": [
            # FHR grid (top)
            {
                "left": 50,
                "right": 20,
                "top": 10,
                "height": "35%" if not compact else "40%"
            },
            # UC grid (bottom)
            {
                "left": 50,
                "right": 20,
                "top": "55%" if not compact else "58%",
                "height": "35%" if not compact else "38%"
            }
        ],
        
        # X-axes (linked for synchronization)
        "xAxis": [
            # FHR X-axis
            {
                "type": "category",
                "data": time_labels,
                "gridIndex": 0,
                "axisLine": {"lineStyle": {"color": COLORS["grid"]}},
                "axisLabel": {"show": not compact, "fontSize": 10},
                "splitLine": {"show": True, "lineStyle": {"color": COLORS["grid"]}}
            },
            # UC X-axis
            {
                "type": "category",
                "data": time_labels,
                "gridIndex": 1,
                "axisLine": {"lineStyle": {"color": COLORS["grid"]}},
                "axisLabel": {"fontSize": 10},
                "splitLine": {"show": True, "lineStyle": {"color": COLORS["grid"]}}
            }
        ],
        
        # Y-axes
        "yAxis": [
            # FHR Y-axis (50-210 bpm)
            {
                "type": "value",
                "name": "FHR" if not compact else "",
                "min": 50,
                "max": 210,
                "interval": 30,
                "gridIndex": 0,
                "axisLine": {"lineStyle": {"color": COLORS["grid"]}},
                "axisLabel": {"fontSize": 10, "color": COLORS["text"]},
                "splitLine": {"show": True, "lineStyle": {"color": COLORS["grid"]}}
            },
            # UC Y-axis (0-100 mmHg)
            {
                "type": "value",
                "name": "UC" if not compact else "",
                "min": 0,
                "max": 100,
                "interval": 25,
                "gridIndex": 1,
                "axisLine": {"lineStyle": {"color": COLORS["grid"]}},
                "axisLabel": {"fontSize": 10, "color": COLORS["text"]},
                "splitLine": {"show": True, "lineStyle": {"color": COLORS["grid"]}}
            }
        ],
        
        # Data series
        "series": [
            # FHR line
            {
                "name": "FHR",
                "type": "line",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": fhr_data,
                "symbol": "none",  # No data point symbols
                "lineStyle": {"color": COLORS["fhr_line"], "width": 1.5},
                "markArea": {"data": mark_areas} if mark_areas else None
            },
            # UC line
            {
                "name": "UC",
                "type": "line",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "data": uc_data,
                "symbol": "none",
                "lineStyle": {"color": COLORS["uc_line"], "width": 1.5}
            }
        ]
    }
    
    # Add dataZoom for detail view
    if show_zoom:
        options["dataZoom"] = [
            {
                "type": "slider",
                "xAxisIndex": [0, 1],
                "start": 80,
                "end": 100,
                "bottom": 5
            },
            {
                "type": "inside",
                "xAxisIndex": [0, 1]
            }
        ]
    
    return options
```

### 3.2 Event Mark Areas (`charts/mark_areas.py`)

```python
"""
Event Mark Areas

Builds ECharts markArea configurations for pathology highlighting.
"""

from typing import List, Dict, Any
from src.ui.styles.colors import COLORS

# Event type to color mapping
EVENT_COLORS = {
    "LATE": "rgba(211, 47, 47, 0.2)",        # Red - Late deceleration
    "VARIABLE": "rgba(245, 127, 23, 0.2)",   # Orange - Variable deceleration
    "SINUSOIDAL": "rgba(183, 28, 28, 0.3)",  # Deep red - Critical
    "TACHYSYSTOLE": "rgba(255, 235, 59, 0.2)", # Yellow
    "BRADYCARDIA": "rgba(211, 47, 47, 0.3)", # Red
}

def build_event_mark_areas(
    events: List[Dict],
    timestamps: List[float]
) -> List[List[Dict]]:
    """
    Build markArea data for event highlighting.
    
    Args:
        events: List of event dictionaries with type, start_idx, end_idx
        timestamps: Full timestamp array for index-to-time mapping
    
    Returns:
        List of markArea data pairs for ECharts
    """
    mark_areas = []
    
    for event in events:
        event_type = event.get("type", "UNKNOWN")
        start_idx = event.get("start_idx", 0)
        end_idx = event.get("end_idx", len(timestamps) - 1)
        
        # Get color for event type
        color = EVENT_COLORS.get(event_type, "rgba(128, 128, 128, 0.2)")
        
        # Build markArea pair
        mark_area = [
            {
                "xAxis": start_idx,
                "itemStyle": {"color": color}
            },
            {
                "xAxis": end_idx
            }
        ]
        
        mark_areas.append(mark_area)
    
    return mark_areas
```

---

## 4. Styles and Colors

### 4.1 Color Constants (`styles/colors.py`)

```python
"""
Color Constants

Medical-grade color palette for SentinelFetal UI.
"""

COLORS = {
    # Background and structure
    "background": "#FFFFFF",
    "grid": "#E0E0E0",
    "border": "#E0E0E0",
    "text": "#212121",
    "text_secondary": "#757575",
    
    # CTG trace colors
    "fhr_line": "#0D47A1",  # Deep Blue
    "uc_line": "#1B5E20",   # Dark Green
    
    # Category colors
    "category_1": "#388E3C",  # Green - Normal
    "category_2": "#F57F17",  # Amber - Warning
    "category_3": "#D32F2F",  # Red - Critical
    
    # Alert colors
    "alert_normal": "#388E3C",
    "alert_warning": "#F57F17",
    "alert_critical": "#D32F2F",
    
    # Event overlay colors (with alpha)
    "overlay_late": "rgba(211, 47, 47, 0.2)",
    "overlay_variable": "rgba(245, 127, 23, 0.2)",
    "overlay_sinusoidal": "rgba(183, 28, 28, 0.3)",
}

def get_category_color(category: int) -> str:
    """Get color for a given category number."""
    return {
        1: COLORS["category_1"],
        2: COLORS["category_2"],
        3: COLORS["category_3"],
    }.get(category, COLORS["text"])
```

### 4.2 Custom CSS (`styles/css.py`)

```python
"""
Custom CSS Injection

Injects medical-grade styling into Streamlit.
"""

import streamlit as st

def inject_custom_css():
    """Inject custom CSS for medical-grade appearance."""
    st.markdown("""
    <style>
    /* Global resets */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Remove Streamlit branding elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Patient card styling */
    .patient-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 8px;
        margin-bottom: 8px;
    }
    
    .patient-card.category-1 {
        border-left: 4px solid #388E3C;
    }
    
    .patient-card.category-2 {
        border-left: 4px solid #F57F17;
    }
    
    .patient-card.category-3 {
        border-left: 4px solid #D32F2F;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        color: #212121;
    }
    
    /* Button styling */
    .stButton > button {
        background: transparent;
        border: 1px solid #E0E0E0;
        color: #212121;
    }
    
    .stButton > button:hover {
        background: #F5F5F5;
        border-color: #BDBDBD;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #FAFAFA;
    }
    
    /* Grid gap adjustment */
    .row-widget.stHorizontalBlock {
        gap: 8px;
    }
    
    /* Chart container */
    .echarts-container {
        background: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)
```

### 4.3 Layout Calculations (`styles/layout.py`)

```python
"""
Layout Calculations

Dynamic grid layout based on patient count.
"""

def calculate_grid_columns(patient_count: int) -> int:
    """
    Calculate optimal number of columns for the patient grid.
    
    Args:
        patient_count: Number of active patients
    
    Returns:
        Number of columns (1-4)
    """
    if patient_count <= 1:
        return 1  # Full width
    elif patient_count <= 4:
        return 2  # 2 columns
    elif patient_count <= 12:
        return 3  # 3 columns
    else:
        return 4  # 4 columns (max density)

def calculate_card_height(patient_count: int) -> str:
    """
    Calculate card height based on patient count.
    
    Args:
        patient_count: Number of active patients
    
    Returns:
        CSS height value
    """
    if patient_count <= 4:
        return "350px"
    elif patient_count <= 9:
        return "280px"
    else:
        return "220px"
```

---

## 5. God Mode Implementation

### 5.1 Fault Injection Controls (`components/god_mode.py`)

```python
"""
God Mode - Fault Injection Controls

Development/demo feature for injecting pathologies.
"""

import streamlit as st

# Injection modes and their descriptions
INJECTION_MODES = {
    "normal": "Normal Baseline",
    "late_decel": "Late Deceleration",
    "variable_decel": "Variable Deceleration",
    "sinusoidal": "Sinusoidal Pattern",
    "tachysystole": "Tachysystole",
    "bradycardia": "Bradycardia",
    "tachycardia": "Tachycardia",
}

def render_god_mode_controls(patient_id: str) -> None:
    """
    Render fault injection controls for a patient.
    
    Args:
        patient_id: Patient to inject faults into
    """
    with st.expander("🔧 Dev Tools", expanded=False):
        st.caption("Fault Injection")
        
        # Current mode display
        current_mode = st.session_state.patients[patient_id].sim_mode
        st.text(f"Current: {INJECTION_MODES.get(current_mode, current_mode)}")
        
        # Injection buttons in columns
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Late Decel", key=f"inject_late_{patient_id}"):
                _inject_mode(patient_id, "late_decel")
            
            if st.button("Variable Decel", key=f"inject_var_{patient_id}"):
                _inject_mode(patient_id, "variable_decel")
            
            if st.button("Sinusoidal", key=f"inject_sin_{patient_id}"):
                _inject_mode(patient_id, "sinusoidal")
        
        with col2:
            if st.button("Tachysystole", key=f"inject_tachy_{patient_id}"):
                _inject_mode(patient_id, "tachysystole")
            
            if st.button("Bradycardia", key=f"inject_brady_{patient_id}"):
                _inject_mode(patient_id, "bradycardia")
            
            if st.button("Reset Normal", key=f"inject_normal_{patient_id}"):
                _inject_mode(patient_id, "normal")

def _inject_mode(patient_id: str, mode: str) -> None:
    """Apply injection mode to patient."""
    st.session_state.patients[patient_id].sim_mode = mode
    
    # Notify orchestrator if available
    orchestrator = st.session_state.get("orchestrator")
    if orchestrator:
        orchestrator.set_patient_mode(patient_id, mode)
```

---

## 6. Utility Functions

### 6.1 Downsampling (`utils/downsampling.py`)

```python
"""
Data Downsampling

Min-max downsampling for rendering performance.
"""

import numpy as np
from typing import List

def minmax_downsample(data: List[float], target_points: int) -> List[float]:
    """
    Downsample data while preserving peaks and valleys.
    
    Uses min-max method: for each chunk, keeps both min and max values.
    This preserves visual fidelity of the waveform.
    
    Args:
        data: Original data array
        target_points: Desired number of output points
    
    Returns:
        Downsampled data array
    """
    if len(data) <= target_points:
        return data
    
    arr = np.array(data)
    chunk_size = len(arr) // (target_points // 2)
    
    result = []
    for i in range(0, len(arr) - chunk_size + 1, chunk_size):
        chunk = arr[i:i + chunk_size]
        result.extend([float(chunk.min()), float(chunk.max())])
    
    return result[:target_points]
```

### 6.2 Time Formatting (`utils/time_format.py`)

```python
"""
Time Formatting Utilities

Formats timestamps for chart display.
"""

from typing import List
from datetime import datetime

def format_timestamps(timestamps: List[float]) -> List[str]:
    """
    Format Unix timestamps to HH:MM:SS strings.
    
    Args:
        timestamps: List of Unix timestamps
    
    Returns:
        List of formatted time strings
    """
    return [
        datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        for ts in timestamps
    ]

def format_relative_time(timestamp: float, reference: float) -> str:
    """
    Format timestamp as relative time from reference.
    
    Args:
        timestamp: Target timestamp
        reference: Reference timestamp (start time)
    
    Returns:
        Formatted string like "+5:30" (5 minutes 30 seconds)
    """
    delta = int(timestamp - reference)
    minutes = delta // 60
    seconds = delta % 60
    return f"+{minutes}:{seconds:02d}"
```

---

## 7. Backend Integration

### 7.1 Orchestrator Connection

```python
"""
Backend Integration

Connects UI to SimulationOrchestrator and PipelineAdapter.
"""

import streamlit as st
from src.simulation import SimulationOrchestrator
from src.simulation.processing import PipelineAdapter, PipelineAdapterConfig

@st.cache_resource
def get_orchestrator():
    """
    Get or create the simulation orchestrator (singleton).
    
    Uses @st.cache_resource to ensure single instance across reruns.
    """
    config = PipelineAdapterConfig(
        sampling_rate=4.0,
        min_data_seconds=60.0
    )
    
    pipeline = PipelineAdapter(config)
    orchestrator = SimulationOrchestrator(pipeline_adapter=pipeline)
    
    return orchestrator

def initialize_backend():
    """Initialize backend connections in session state."""
    if st.session_state.orchestrator is None:
        st.session_state.orchestrator = get_orchestrator()
        st.session_state.pipeline_adapter = st.session_state.orchestrator.pipeline
```

---

## 8. Performance Optimization Checklist

### 8.1 ECharts Optimizations

- [ ] `animation: false` — Disable all animations
- [ ] `symbol: "none"` — No data point markers
- [ ] Use `markArea` instead of separate highlight series
- [ ] Limit data points to 600 via downsampling
- [ ] Use canvas renderer (default in streamlit-echarts)

### 8.2 Streamlit Optimizations

- [ ] Use `@st.fragment(run_every=0.25)` for grid refresh
- [ ] Use `@st.cache_resource` for singleton services
- [ ] Avoid `st.rerun()` in update loops
- [ ] Use `deque(maxlen=N)` instead of list slicing
- [ ] Minimize session_state reads/writes per cycle

### 8.3 Data Flow Optimizations

- [ ] Buffer size: 2400 points (10 minutes at 4Hz)
- [ ] Downsample to 600 points for rendering
- [ ] Use NumPy arrays for bulk operations
- [ ] Batch event updates (don't update per-point)

---

## 9. Testing Requirements

### 9.1 Unit Tests

```python
# tests/ui/test_downsampling.py
def test_minmax_preserves_extrema():
    data = [1, 5, 2, 8, 3, 9, 4, 7]  # min=1, max=9
    result = minmax_downsample(data, 4)
    assert min(result) == 1
    assert max(result) == 9

# tests/ui/test_layout.py
def test_grid_columns_calculation():
    assert calculate_grid_columns(1) == 1
    assert calculate_grid_columns(4) == 2
    assert calculate_grid_columns(12) == 3
    assert calculate_grid_columns(20) == 4
```

### 9.2 Integration Tests

- [ ] Grid renders 20 patients without error
- [ ] Category changes update visual indicators
- [ ] Event injection triggers backend processing
- [ ] Detail view shows extended history

### 9.3 Performance Tests

- [ ] Measure frame time at 20 patients
- [ ] Memory profiling over 30-minute session
- [ ] Network payload size per refresh

---

## 10. Dependencies

### 10.1 Python Packages

```
streamlit>=1.30.0
streamlit-echarts>=0.4.0
numpy>=1.24.0
pandas>=2.0.0
```

### 10.2 NPM Packages (for ECharts)

ECharts is bundled in streamlit-echarts, no additional installation needed.

---

## 11. Deployment Notes

### 11.1 Environment Variables

```bash
# Production
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10

# Development
SENTINELFETAL_GOD_MODE=enabled
SENTINELFETAL_DEBUG=true
```

### 11.2 Run Command

```bash
streamlit run src/ui/app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
```

---

*End of Technical Specifications*
