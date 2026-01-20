# -*- coding: utf-8 -*-
"""
SentinelFetal Simulation Dashboard - Clinical Minimalism UI.

Two-view architecture:
- Ward View (Grid): All patient monitors side-by-side
- Detail View: Single patient focus with full analysis

Design principles:
- White background (#FFFFFF)
- Black text (#000000)
- No large alert banners - use colored indicators
- 1-20 dynamic patient population

Usage:
    streamlit run src/ui/simulation_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.simulation import (
    SimulationOrchestrator,
    OrchestratorConfig,
    PipelineAdapter,
    PipelineAdapterConfig,
    EventType,
    LateDecelerationParams,
    VariableDecelerationParams,
    BradycardiaParams,
    TachycardiaParams,
    VariabilityParams,
    SinusoidalParams,
    TachysystoleParams,
)
from src.config import COLORS
from src.ui.plots import create_ctg_plot


# =============================================================================
# Constants
# =============================================================================

DEFAULT_REFRESH_FPS = 3
REFRESH_FPS_OPTIONS = (2, 3, 4, 5)
DEFAULT_PATIENT_COUNT = 8
MAX_PATIENT_COUNT = 20

# Event options with Hebrew and English
EVENT_OPTIONS = {
    "האטות מאוחרות (Late Decels)": EventType.LATE_DECELERATION,
    "האטות משתנות (Variable Decels)": EventType.VARIABLE_DECELERATION,
    "ברדיקרדיה (Bradycardia)": EventType.BRADYCARDIA,
    "טכיקרדיה (Tachycardia)": EventType.TACHYCARDIA,
    "שונות נעדרת (Absent Variability)": EventType.ABSENT_VARIABILITY,
    "שונות מזערית (Minimal Variability)": EventType.MINIMAL_VARIABILITY,
    "סינוסואידלי (Sinusoidal)": EventType.SINUSOIDAL_PATTERN,
    "טכיסיסטולה (Tachysystole)": EventType.TACHYSYSTOLE,
}

SEVERITY_MAP = {
    "קל (Mild)": "mild",
    "בינוני (Moderate)": "moderate",
    "חמור (Severe)": "severe"
}

# Expected detection times for events (in seconds)
EXPECTED_DETECTION_TIMES = {
    EventType.LATE_DECELERATION: "10-15s",
    EventType.VARIABLE_DECELERATION: "10-15s",
    EventType.BRADYCARDIA: "~10s",
    EventType.TACHYCARDIA: "~10s",
    EventType.ABSENT_VARIABILITY: "30-60s",
    EventType.MINIMAL_VARIABILITY: "30-60s",
    EventType.SINUSOIDAL_PATTERN: "20-40s",
    EventType.TACHYSYSTOLE: "15-30s",
}

# Category colors and labels
CAT_COLORS = {1: "#28a745", 2: "#fd7e14", 3: "#dc3545"}
CAT_DOTS = {1: "🟢", 2: "🟠", 3: "🔴"}
CAT_LABELS = {1: "Normal", 2: "Intermediate", 3: "Pathological"}
CAT_LABELS_HE = {1: "תקין", 2: "ביניים", 3: "פתולוגי"}


# =============================================================================
# Cached Resources
# =============================================================================

@st.cache_resource
def get_pipeline_adapter() -> PipelineAdapter:
    """Get or create the pipeline adapter (persists across refreshes)."""
    config = PipelineAdapterConfig(
        use_real_moment=True,
        model_path="models/sentinel_classifier.json"
    )
    return PipelineAdapter(config)


@st.cache_resource
def get_orchestrator(_patient_count: int = DEFAULT_PATIENT_COUNT) -> SimulationOrchestrator:
    """Get or create the simulation orchestrator."""
    adapter = get_pipeline_adapter()

    def processing_callback(patient_id: str, data: Dict) -> Dict:
        return adapter.process_patient(patient_id, data, run_moment=True)

    config = OrchestratorConfig(
        num_patients=_patient_count,
        sampling_rate=4.0,
        tick_interval_seconds=1.0,
        moment_interval_seconds=30.0
    )

    return SimulationOrchestrator(config, processing_callback)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "grid"  # "grid" or patient_id
    if 'selected_patient' not in st.session_state:
        st.session_state.selected_patient = None
    if 'patient_count' not in st.session_state:
        st.session_state.patient_count = DEFAULT_PATIENT_COUNT
    if 'refresh_fps' not in st.session_state:
        st.session_state.refresh_fps = DEFAULT_REFRESH_FPS


# =============================================================================
# Event Parameter Helpers
# =============================================================================

def get_event_params(event_type: EventType, severity: str):
    """Get event parameters based on type and severity."""
    sev = SEVERITY_MAP.get(severity, "moderate")

    if event_type == EventType.LATE_DECELERATION:
        return getattr(LateDecelerationParams, sev)()
    elif event_type == EventType.VARIABLE_DECELERATION:
        return getattr(VariableDecelerationParams, sev)()
    elif event_type == EventType.BRADYCARDIA:
        return BradycardiaParams.mild() if sev == "mild" else BradycardiaParams.severe()
    elif event_type == EventType.TACHYCARDIA:
        return getattr(TachycardiaParams, sev if sev != "severe" else "moderate")()
    elif event_type == EventType.ABSENT_VARIABILITY:
        return VariabilityParams.absent()
    elif event_type == EventType.MINIMAL_VARIABILITY:
        return VariabilityParams.minimal()
    elif event_type == EventType.SINUSOIDAL_PATTERN:
        return SinusoidalParams()
    elif event_type == EventType.TACHYSYSTOLE:
        return TachysystoleParams.mild() if sev == "mild" else TachysystoleParams.severe()

    return VariabilityParams.minimal()


# =============================================================================
# CSS Injection - Clinical Minimalism
# =============================================================================

def inject_clinical_css():
    """Inject clean, clinical CSS - white background, black text."""
    st.markdown("""
    <style>
        /* White background everywhere */
        .stApp, .main, .block-container {
            background-color: #FFFFFF !important;
        }

        /* Black text */
        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: #000000 !important;
        }

        /* Clean header */
        .main-header {
            font-size: 1.75rem;
            font-weight: 700;
            color: #000000;
            text-align: left;
            padding: 0.5rem 0;
            border-bottom: 2px solid #000000;
            margin-bottom: 1rem;
        }

        /* Patient cards */
        .patient-card {
            background: #FFFFFF;
            border: 1px solid #E5E5E5;
            border-radius: 6px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: border-color 0.15s;
        }
        .patient-card:hover {
            border-color: #000000;
        }

        /* Category border colors */
        .cat-1 { border-left: 4px solid #28a745; }
        .cat-2 { border-left: 4px solid #fd7e14; }
        .cat-3 { border-left: 4px solid #dc3545; }

        /* Control bar */
        .control-bar {
            background: #F5F5F5;
            border: 1px solid #E5E5E5;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        /* Status indicator */
        .status-running { color: #28a745; font-weight: 600; }
        .status-paused { color: #fd7e14; font-weight: 600; }
        .status-stopped { color: #666666; font-weight: 600; }

        /* Hide Streamlit chrome */
        header[data-testid="stHeader"] { display: none !important; }
        footer { display: none !important; }
        #MainMenu { display: none !important; }

        /* Clean metrics */
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E5E5;
            border-radius: 4px;
            padding: 0.5rem;
        }

        /* Minimal alerts */
        .stAlert {
            background: transparent !important;
            border: none !important;
            padding: 0.25rem 0 !important;
        }

        /* Back button styling */
        .back-btn {
            background: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid #000000 !important;
        }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Control Bar Component
# =============================================================================

def render_control_bar(orchestrator: SimulationOrchestrator):
    """Render the simulation control bar at the top."""
    st.markdown("### Control Panel | לוח בקרה")

    # Row 1: Population, Controls, Status
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])

    with col1:
        # Population slider
        new_count = st.slider(
            "Population | מספר יולדות",
            min_value=1,
            max_value=MAX_PATIENT_COUNT,
            value=st.session_state.patient_count,
            key="pop_slider",
            help="Generate 1-20 synthetic patients"
        )
        if new_count != st.session_state.patient_count:
            st.session_state.patient_count = new_count
            orchestrator.set_patient_count(new_count)
            st.rerun()

    with col2:
        # Start/Resume button
        if not orchestrator._running:
            if st.button("▶ Start", key="start_btn", use_container_width=True):
                orchestrator.start()
                st.rerun()
        elif orchestrator._paused:
            if st.button("▶ Resume", key="resume_btn", use_container_width=True):
                orchestrator.resume()
                st.rerun()
        else:
            if st.button("⏸ Pause", key="pause_btn", use_container_width=True):
                orchestrator.pause()
                st.rerun()

    with col3:
        # Stop/Reset button
        if st.button("⏹ Reset", key="reset_btn", use_container_width=True):
            orchestrator.stop()
            orchestrator.reset_all()
            st.rerun()

    with col4:
        # Status display
        if orchestrator._running and not orchestrator._paused:
            st.markdown('<p class="status-running">● Running</p>', unsafe_allow_html=True)
        elif orchestrator._paused:
            st.markdown('<p class="status-paused">● Paused</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-stopped">○ Stopped</p>', unsafe_allow_html=True)
        st.caption(f"Time: {orchestrator.get_simulation_time_formatted()}")

    with col5:
        # Speed and FPS controls
        sub1, sub2 = st.columns(2)
        with sub1:
            speed = st.select_slider(
                "Speed",
                options=[0.5, 1.0, 2.0],
                value=orchestrator._speed_multiplier,
                key="speed_slider"
            )
            if speed != orchestrator._speed_multiplier:
                orchestrator.set_speed(speed)
        with sub2:
            fps = st.select_slider(
                "FPS",
                options=list(REFRESH_FPS_OPTIONS),
                value=st.session_state.refresh_fps,
                key="fps_slider"
            )
            st.session_state.refresh_fps = fps


# =============================================================================
# Event Injection Component
# =============================================================================

def render_event_injection(orchestrator: SimulationOrchestrator):
    """Render the event injection panel."""
    with st.expander("💉 Event Injection | הזרקת אירוע", expanded=False):
        col1, col2, col3, col4, col5, col6 = st.columns([1.5, 2, 1.5, 1, 1, 1.5])

        num_patients = orchestrator.config.num_patients
        patients = [f"P{i+1}" for i in range(num_patients)]

        with col1:
            target_patient = st.selectbox("Patient", patients, key="inject_patient")

        with col2:
            event_name = st.selectbox("Event Type", list(EVENT_OPTIONS.keys()), key="inject_event")
            event_type = EVENT_OPTIONS[event_name]

        with col3:
            severity = st.selectbox("Severity", list(SEVERITY_MAP.keys()), key="inject_severity")

        with col4:
            duration = st.number_input("Min", min_value=1, max_value=20, value=5, key="inject_duration")

        with col5:
            # Expected detection time display
            detection_time = EXPECTED_DETECTION_TIMES.get(event_type, "~15s")
            st.metric("Detection", detection_time)

        with col6:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Inject", key="inject_btn", type="primary", use_container_width=True):
                params = get_event_params(event_type, severity)
                orchestrator.inject_event(target_patient, event_type, params, duration * 60)
                st.toast(f"✅ Injected: {event_name} → {target_patient}")


# =============================================================================
# Ward/Grid View - Staggered Updates for Performance
# =============================================================================

def render_ward_view(orchestrator: SimulationOrchestrator):
    """
    Render the Ward view - all patient monitors in a grid.
    
    Uses staggered update pattern to prevent browser freeze with 20 patients.
    Each patient tile updates at a slightly different offset to distribute load.
    """
    st.markdown("### Ward View | תצוגת חדר לידה")

    statuses = orchestrator.get_all_patients_status()
    num_patients = len(statuses)

    # Dynamic columns (max 4 per row)
    n_cols = min(4, num_patients)
    n_rows = (num_patients + n_cols - 1) // n_cols

    for row_idx in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            patient_idx = row_idx * n_cols + col_idx
            if patient_idx < num_patients:
                with cols[col_idx]:
                    # Staggered update: each patient gets a different refresh offset
                    # Pattern: 0.5s base + (patient_idx % 5) * 0.1s offset
                    # This distributes 20 patients across 5 update groups
                    render_ward_card_staggered(
                        statuses[patient_idx], 
                        orchestrator,
                        patient_idx,
                        num_patients
                    )


def render_ward_card_staggered(
    status: Dict[str, Any], 
    orchestrator: SimulationOrchestrator,
    patient_idx: int,
    total_patients: int
):
    """
    Render a single patient card with staggered refresh for performance.
    
    Staggered pattern prevents all 20 patients from updating simultaneously,
    which would cause browser lag and high CPU usage.
    """
    patient_id = status['patient_id']
    cat = status['category']
    name = status['name']
    bed = status['bed_number']
    events = status.get('active_events', [])
    
    # Calculate staggered refresh time
    # Base rate: 0.5 seconds
    # Offset: distribute across 5 groups (0.1s apart)
    # Result: patients update at 0.5s, 0.6s, 0.7s, 0.8s, 0.9s intervals
    base_refresh = 0.5
    offset_group = patient_idx % 5
    stagger_offset = offset_group * 0.1
    refresh_interval = base_refresh + stagger_offset
    
    # Only auto-refresh if simulation is running
    is_running = orchestrator._running and not orchestrator._paused
    run_every = refresh_interval if is_running else None

    @st.fragment(run_every=run_every)
    def _ward_card_fragment():
        # Card container
        card_html = f"""
        <div class="patient-card cat-{cat}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600;">{patient_id} - {name}</span>
                <span>{CAT_DOTS[cat]}</span>
            </div>
            <div style="font-size: 0.85rem; color: #666;">
                Bed {bed} | {CAT_LABELS[cat]}
                {f' | ⚠️ {len(events)} events' if events else ''}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Mini sparkline - only last 50 points for performance
        patient = orchestrator.get_patient(patient_id)
        if patient:
            data = patient.get_buffer_data(duration_minutes=1)
            fhr = data.get('fhr', np.array([]))
            if len(fhr) > 10:
                # Limit to last 50 points for performance
                fig = create_mini_sparkline(fhr[-50:])
                st.plotly_chart(fig, use_container_width=True, key=f"spark_{patient_id}")

    _ward_card_fragment()
    
    # Click to detail view (outside fragment to avoid re-render issues)
    if st.button(f"View Details →", key=f"view_{patient_id}", use_container_width=True):
        st.session_state.current_view = patient_id
        st.session_state.selected_patient = patient_id
        st.rerun()


def render_ward_card(status: Dict[str, Any], orchestrator: SimulationOrchestrator):
    """Legacy non-staggered card render (kept for compatibility)."""
    patient_id = status['patient_id']
    cat = status['category']
    name = status['name']
    bed = status['bed_number']
    events = status.get('active_events', [])

    # Card container
    card_html = f"""
    <div class="patient-card cat-{cat}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600;">{patient_id} - {name}</span>
            <span>{CAT_DOTS[cat]}</span>
        </div>
        <div style="font-size: 0.85rem; color: #666;">
            Bed {bed} | {CAT_LABELS[cat]}
            {f' | ⚠️ {len(events)} events' if events else ''}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    # Mini sparkline
    patient = orchestrator.get_patient(patient_id)
    if patient:
        data = patient.get_buffer_data(duration_minutes=1)
        fhr = data.get('fhr', np.array([]))
        if len(fhr) > 10:
            fig = create_mini_sparkline(fhr[-60:])  # Last 60 samples
            st.plotly_chart(fig, use_container_width=True, key=f"spark_{patient_id}")

    # Click to detail view
    if st.button(f"View Details →", key=f"view_{patient_id}", use_container_width=True):
        st.session_state.current_view = patient_id
        st.session_state.selected_patient = patient_id
        st.rerun()


def create_mini_sparkline(fhr: np.ndarray) -> go.Figure:
    """Create a minimal sparkline chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=fhr,
        mode='lines',
        line=dict(color='#000000', width=1),
        hoverinfo='skip'
    ))
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[80, 200])
    )
    return fig


# =============================================================================
# Detail View
# =============================================================================

def render_detail_view(orchestrator: SimulationOrchestrator):
    """Render the Detail view - single patient focus."""
    patient_id = st.session_state.selected_patient
    patient = orchestrator.get_patient(patient_id)

    if not patient:
        st.warning(f"Patient {patient_id} not found")
        if st.button("← Back to Ward"):
            st.session_state.current_view = "grid"
            st.rerun()
        return

    # Back button
    if st.button("← Back to Ward", key="back_btn"):
        st.session_state.current_view = "grid"
        st.rerun()

    # Header with patient info and category
    cat = patient.latest_category
    config = patient.config

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;
                border-bottom: 3px solid {CAT_COLORS[cat]}; padding-bottom: 0.5rem; margin-bottom: 1rem;">
        <div>
            <h2 style="margin: 0;">{config.name}</h2>
            <span style="color: #666;">Bed {config.bed_number} | {patient_id}</span>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 1.5rem;">{CAT_DOTS[cat]}</span>
            <span style="color: {CAT_COLORS[cat]}; font-weight: 600;">Category {cat} - {CAT_LABELS[cat]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Layout: Patient Info | Monitor | Event Log
    col_info, col_monitor, col_events = st.columns([1, 2.5, 1])

    with col_info:
        render_patient_info_panel(patient)

    with col_monitor:
        render_monitor_panel(orchestrator, patient_id)

    with col_events:
        render_patient_events_panel(orchestrator, patient_id)


def render_patient_info_panel(patient):
    """Render patient information panel."""
    st.markdown("#### Patient Info")

    config = patient.config
    findings = getattr(patient, 'latest_findings', {}) or {}
    baseline = findings.get('baseline', {})
    variability = findings.get('variability', {})

    # Basic info
    st.markdown(f"**Name:** {config.name}")
    st.markdown(f"**Bed:** {config.bed_number}")
    st.markdown(f"**Patient ID:** {config.patient_id}")

    st.markdown("---")

    # Current metrics
    st.markdown("**Current Metrics:**")
    try:
        bl_val = float(baseline.get('value', config.baseline_fhr))
        st.metric("Baseline", f"{bl_val:.0f} bpm")
    except (TypeError, ValueError):
        st.metric("Baseline", f"{config.baseline_fhr:.0f} bpm")

    try:
        var_val = float(variability.get('value', config.baseline_variability))
        st.metric("Variability", f"{var_val:.1f} bpm")
    except (TypeError, ValueError):
        st.metric("Variability", f"{config.baseline_variability:.1f} bpm")

    # Active events
    active_events = patient.get_active_events()
    if active_events:
        st.markdown("---")
        st.markdown("**Active Events:**")
        for event in active_events:
            st.markdown(f"• {event.event_type.name}")


def render_monitor_panel(orchestrator: SimulationOrchestrator, patient_id: str):
    """Render the CTG monitor with fragment for auto-refresh."""
    st.markdown("#### CTG Monitor")

    refresh_fps = int(st.session_state.get("refresh_fps", DEFAULT_REFRESH_FPS))
    run_every = (1.0 / float(refresh_fps)) if (orchestrator._running and not orchestrator._paused) else None

    @st.fragment(run_every=run_every)
    def _monitor_fragment():
        patient = orchestrator.get_patient(patient_id)
        if not patient:
            st.warning("Patient data unavailable")
            return

        data = patient.get_buffer_data(duration_minutes=10)
        fhr = data.get('fhr', np.array([]))
        uc = data.get('uc', np.array([]))

        if len(fhr) > 0:
            fig = create_ctg_plot(
                fhr=fhr,
                uc=uc,
                sampling_rate=4.0,
                title="",
                mode="monitor",
                window_minutes=20.0,
                max_points=2000,
                use_webgl=True,
                uirevision=f"ctg::{patient_id}"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"ctg_{patient_id}")
        else:
            st.info("Waiting for data...")

        # Findings summary below monitor
        render_findings_summary(patient)

    _monitor_fragment()


def render_findings_summary(patient):
    """Render compact findings summary."""
    findings = getattr(patient, 'latest_findings', {}) or {}
    if not findings:
        return

    decels = findings.get('decelerations', {})
    tachy = findings.get('tachysystole', {})
    sinusoidal = findings.get('sinusoidal', {})

    alerts = []
    if int(decels.get('late', 0)) > 0:
        alerts.append(f"Late Decels: {decels.get('late')}")
    if int(decels.get('variable', 0)) > 0:
        alerts.append(f"Variable Decels: {decels.get('variable')}")
    if tachy.get('detected'):
        alerts.append("Tachysystole")
    if sinusoidal.get('detected'):
        alerts.append("Sinusoidal")

    if alerts:
        st.markdown(f"**Findings:** {' | '.join(alerts)}")


def render_patient_events_panel(orchestrator: SimulationOrchestrator, patient_id: str):
    """Render event log for specific patient."""
    st.markdown("#### Event Log")

    log = orchestrator.get_event_log()
    all_entries = log.get_entries(limit=50)

    # Filter for this patient
    patient_entries = [e for e in all_entries if e.patient_id == patient_id]

    if not patient_entries:
        st.caption("No events yet")
        return

    for entry in reversed(patient_entries[:10]):
        if entry.event_type == 'INJECTION':
            st.markdown(f"💉 `{entry.simulation_time:.0f}s` {entry.details.get('injected_event', '?')}")
        elif entry.event_type == 'ALERT':
            cat = entry.details.get('category', '?')
            st.markdown(f"{CAT_DOTS.get(cat, '⚪')} `{entry.simulation_time:.0f}s` Cat {cat}")


# =============================================================================
# Global Event Log (for Ward View)
# =============================================================================

def render_global_event_log(orchestrator: SimulationOrchestrator):
    """Render global event log for all patients."""
    st.markdown("### Event Log | יומן אירועים")

    log = orchestrator.get_event_log()
    entries = log.get_entries(limit=15)

    if not entries:
        st.caption("Log is empty")
        return

    for entry in reversed(entries):
        if entry.event_type == 'INJECTION':
            st.markdown(f"💉 `{entry.simulation_time:.0f}s` **{entry.patient_id}**: {entry.details.get('injected_event', '?')}")
        elif entry.event_type == 'ALERT':
            cat = entry.details.get('category', '?')
            st.markdown(f"{CAT_DOTS.get(cat, '⚪')} `{entry.simulation_time:.0f}s` **{entry.patient_id}**: Cat {cat}")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title="SentinelFetal Simulator",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    init_session_state()

    # Inject CSS
    inject_clinical_css()

    # Get orchestrator with current patient count
    orchestrator = get_orchestrator(st.session_state.patient_count)

    # Header
    st.markdown('<h1 class="main-header">SentinelFetal Simulator</h1>', unsafe_allow_html=True)

    # Control bar
    render_control_bar(orchestrator)

    # Event injection (collapsible)
    render_event_injection(orchestrator)

    st.markdown("---")

    # Main content: Ward View or Detail View
    if st.session_state.current_view == "grid":
        # Ward view with side panel for event log
        col_ward, col_log = st.columns([3, 1])
        with col_ward:
            render_ward_view(orchestrator)
        with col_log:
            render_global_event_log(orchestrator)
    else:
        # Detail view for selected patient
        render_detail_view(orchestrator)

    # Footer
    st.markdown("---")
    st.caption("SentinelFetal Simulator v2.0 | Clinical Minimalism UI | Powered by MOMENT AI")


if __name__ == "__main__":
    main()
