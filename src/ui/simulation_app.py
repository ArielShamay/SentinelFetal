"""
SentinelFetal Simulation Dashboard - Real-Time CTG Simulator UI.

This Streamlit application provides a visual interface for the real-time
CTG simulation system. It allows users to:
- Control simulation (start/stop/pause/reset)
- View 8 simulated patients in real-time
- Inject clinical events for training scenarios
- See live CTG plots and AI-generated alerts

Usage:
    streamlit run src/ui/simulation_app.py

References:
    SentinelFetal Real-Time Simulator SPEC Part 2, Section 9
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# Cached Resources (Persist across refreshes)
# =============================================================================

@st.cache_resource
def get_pipeline_adapter() -> PipelineAdapter:
    """
    Get or create the pipeline adapter.
    
    Uses st.cache_resource to persist across refreshes.
    CRITICAL: Uses real MOMENT model (use_mock=False by default).
    """
    config = PipelineAdapterConfig(
        use_real_moment=True,  # Use REAL MOMENT model
        model_path="models/xgb_demo.json"
    )
    return PipelineAdapter(config)


@st.cache_resource
def get_orchestrator() -> SimulationOrchestrator:
    """
    Get or create the simulation orchestrator.
    
    Uses st.cache_resource to persist across refreshes.
    """
    adapter = get_pipeline_adapter()
    
    def processing_callback(patient_id: str, data: Dict) -> Dict:
        """Callback for processing patient data through the AI pipeline."""
        return adapter.process_patient(patient_id, data, run_moment=True)
    
    config = OrchestratorConfig(
        num_patients=8,
        sampling_rate=4.0,
        tick_interval_seconds=1.0,
        moment_interval_seconds=30.0  # Process each patient's MOMENT every 30s total
    )
    
    return SimulationOrchestrator(config, processing_callback)


# =============================================================================
# Event Parameter Helpers
# =============================================================================

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

SEVERITY_MAP = {"קל (Mild)": "mild", "בינוני (Moderate)": "moderate", "חמור (Severe)": "severe"}


def get_event_params(event_type: EventType, severity: str):
    """
    Get event parameters based on type and severity.
    
    Args:
        event_type: The type of event to inject.
        severity: Severity level in Hebrew.
        
    Returns:
        EventParameters instance for the event.
    """
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
# UI Components
# =============================================================================

def render_header():
    """Render the application header."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #1E90FF, #FF8C00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .patient-card {
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
            cursor: pointer;
        }
        .cat-1 { background-color: rgba(40, 167, 69, 0.2); border-left: 4px solid #28a745; }
        .cat-2 { background-color: rgba(253, 126, 20, 0.2); border-left: 4px solid #fd7e14; }
        .cat-3 { background-color: rgba(220, 53, 69, 0.2); border-left: 4px solid #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏥 SentinelFetal Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">סימולציית זמן אמת למעקב עוברי | Real-Time CTG Simulation</p>', 
                unsafe_allow_html=True)


def render_control_panel(orchestrator: SimulationOrchestrator):
    """Render the simulation control panel."""
    st.markdown("### 🎛️ לוח בקרה | Control Panel")
    
    col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 2])
    
    with col1:
        if orchestrator._running and not orchestrator._paused:
            status_text = "▶️ פעיל | Running"
            status_color = "green"
        elif orchestrator._paused:
            status_text = "⏸️ מושהה | Paused"
            status_color = "orange"
        else:
            status_text = "⏹️ עצור | Stopped"
            status_color = "gray"
        
        st.markdown(f"**סטטוס:** <span style='color:{status_color}'>{status_text}</span>",
                    unsafe_allow_html=True)
        st.markdown(f"**זמן סימולציה:** {orchestrator.get_simulation_time_formatted()}")
    
    with col2:
        stats = orchestrator.get_statistics()
        st.markdown(f"**Ticks:** {stats['tick_count']}")
        st.markdown(f"**MOMENT Processes:** {stats['moment_process_count']}")
    
    with col3:
        if not orchestrator._running:
            if st.button("▶️ התחל", key="start_btn", use_container_width=True):
                orchestrator.start()
                st.rerun()
        else:
            if orchestrator._paused:
                if st.button("▶️ המשך", key="resume_btn", use_container_width=True):
                    orchestrator.resume()
                    st.rerun()
            else:
                if st.button("⏸️ השהה", key="pause_btn", use_container_width=True):
                    orchestrator.pause()
                    st.rerun()
    
    with col4:
        if st.button("🔄 אפס", key="reset_btn", use_container_width=True):
            orchestrator.stop()
            orchestrator.reset_all()
            st.rerun()
    
    with col5:
        speed = st.select_slider(
            "מהירות | Speed",
            options=[0.5, 1.0, 2.0],
            value=orchestrator._speed_multiplier,
            key="speed_slider"
        )
        if speed != orchestrator._speed_multiplier:
            orchestrator.set_speed(speed)


def render_event_injection(orchestrator: SimulationOrchestrator):
    """Render the event injection panel."""
    st.markdown("### 💉 הזרקת אירוע | Event Injection")
    
    col1, col2, col3, col4, col5 = st.columns([1.5, 2, 1.5, 1, 1])
    
    with col1:
        patients = [f"P{i+1}" for i in range(8)]
        target_patient = st.selectbox("יולדת | Patient", patients, key="inject_patient")
    
    with col2:
        event_name = st.selectbox("סוג אירוע | Event Type", 
                                   list(EVENT_OPTIONS.keys()), 
                                   key="inject_event")
    
    with col3:
        severity = st.selectbox("חומרה | Severity", 
                                 list(SEVERITY_MAP.keys()), 
                                 key="inject_severity")
    
    with col4:
        duration = st.number_input("דקות | Duration", 
                                    min_value=1, max_value=20, value=5,
                                    key="inject_duration")
    
    with col5:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer
        if st.button("💉 הזרק", key="inject_btn", type="primary", use_container_width=True):
            event_type = EVENT_OPTIONS[event_name]
            params = get_event_params(event_type, severity)
            orchestrator.inject_event(target_patient, event_type, params, duration * 60)
            st.success(f"✅ הוזרק: {event_name} → {target_patient}")
            time.sleep(0.5)
            st.rerun()


def render_patient_overview(orchestrator: SimulationOrchestrator):
    """Render the patient overview grid."""
    st.markdown("### 👥 סקירת יולדות | Patient Overview")
    
    statuses = orchestrator.get_all_patients_status()
    
    # Create 2 rows of 4 patients each
    for row in range(2):
        cols = st.columns(4)
        for col_idx, col in enumerate(cols):
            patient_idx = row * 4 + col_idx
            if patient_idx < len(statuses):
                status = statuses[patient_idx]
                with col:
                    render_patient_card(status, orchestrator)


def render_patient_card(status: Dict[str, Any], orchestrator: SimulationOrchestrator):
    """Render a single patient card in the overview grid."""
    cat = status['category']
    patient_id = status['patient_id']
    bed = status['bed_number']
    name = status['name']
    
    # Colors and emojis
    cat_colors = {1: "#28a745", 2: "#fd7e14", 3: "#dc3545"}
    cat_emojis = {1: "🟢", 2: "🟠", 3: "🔴"}
    cat_names = {1: "תקין", 2: "ביניים", 3: "פתולוגי"}
    
    # Active events
    events = status.get('active_events', [])
    event_badge = f" ⚠️ {len(events)}" if events else ""
    
    # Card content
    card_class = f"cat-{cat}"
    
    # Use a button for selection
    button_label = f"{cat_emojis[cat]} {patient_id} | מיטה {bed}\n{cat_names[cat]}{event_badge}"
    
    if st.button(
        button_label,
        key=f"patient_card_{patient_id}",
        use_container_width=True,
        help=f"לחץ לצפייה ביולדת {name}"
    ):
        st.session_state.selected_patient = patient_id
        st.rerun()


def render_patient_detail(orchestrator: SimulationOrchestrator):
    """Render detailed view for selected patient."""
    selected_patient = st.session_state.get('selected_patient', 'P1')
    patient = orchestrator.get_patient(selected_patient)
    
    if not patient:
        st.warning(f"יולדת {selected_patient} לא נמצאה")
        return
    
    # Get patient data
    data = patient.get_buffer_data(duration_minutes=10)
    status = patient.get_status()
    
    # Header with category
    cat = patient.latest_category
    cat_colors = {1: "#28a745", 2: "#fd7e14", 3: "#dc3545"}
    cat_names = {1: "תקין (Normal)", 2: "ביניים (Intermediate)", 3: "פתולוגי (Pathological)"}
    
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, {cat_colors[cat]}22, transparent);
                padding: 15px; border-radius: 10px; border-left: 5px solid {cat_colors[cat]}'>
        <h3 style='margin:0;'>👤 {patient.config.name} | מיטה {patient.config.bed_number}</h3>
        <h2 style='margin:5px 0; color:{cat_colors[cat]}'>
            קטגוריה {cat} - {cat_names[cat]}
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Active events
    active_events = patient.get_active_events()
    if active_events:
        event_names = [e.event_type.name for e in active_events]
        st.warning(f"⚠️ **אירועים פעילים:** {', '.join(event_names)}")
    
    # CTG Plot
    st.markdown("### 📈 ניטור CTG | CTG Monitor")
    
    fhr = data.get('fhr', np.array([]))
    uc = data.get('uc', np.array([]))
    
    if len(fhr) > 0:
        # Create CTG plot
        fig = create_simulation_ctg_plot(fhr, uc, patient.config.name)
        st.plotly_chart(fig, use_container_width=True, key=f"ctg_plot_{selected_patient}")
    else:
        st.info("⏳ ממתין לנתונים... | Waiting for data...")
    
    # Findings panel
    render_findings_panel(patient)
    
    # Alert panel
    render_alert_panel(patient)


def create_simulation_ctg_plot(
    fhr: np.ndarray,
    uc: np.ndarray,
    patient_name: str = ""
) -> go.Figure:
    """
    Create a CTG plot for simulation display.
    
    Args:
        fhr: FHR signal array.
        uc: UC signal array.
        patient_name: Patient name for title.
        
    Returns:
        Plotly Figure object.
    """
    n_samples = len(fhr)
    time_minutes = np.arange(n_samples) / 4.0 / 60.0  # 4Hz to minutes
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=('דופק עוברי | FHR', 'צירים | UC')
    )
    
    # FHR trace
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=fhr,
            mode='lines',
            name='FHR',
            line=dict(color=COLORS.FHR, width=1.5),
            hovertemplate='%{y:.0f} bpm<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Normal range bands
    fig.add_hrect(
        y0=110, y1=160,
        fillcolor='rgba(40, 167, 69, 0.1)',
        line_width=0,
        row=1, col=1
    )
    
    # Reference lines
    fig.add_hline(y=110, line_dash="dash", line_color="rgba(220, 53, 69, 0.5)", row=1, col=1)
    fig.add_hline(y=160, line_dash="dash", line_color="rgba(220, 53, 69, 0.5)", row=1, col=1)
    
    # UC trace
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=uc,
            mode='lines',
            name='UC',
            line=dict(color=COLORS.UC, width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 140, 0, 0.2)',
            hovertemplate='%{y:.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Layout
    fig.update_layout(
        height=450,
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    fig.update_yaxes(
        title_text="BPM",
        range=[50, 200],
        dtick=30,
        gridcolor='#E5E5E5',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="AU",
        range=[0, 100],
        gridcolor='#E5E5E5',
        row=2, col=1
    )
    
    fig.update_xaxes(
        title_text="זמן (דקות) | Time (min)",
        gridcolor='#E5E5E5',
        row=2, col=1
    )
    
    return fig


def render_findings_panel(patient):
    """Render the findings panel for a patient."""
    st.markdown("### 🔍 ממצאים | Findings")
    
    findings = patient.latest_findings
    
    if not findings:
        st.info("אין ממצאים זמינים עדיין | No findings available yet")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        baseline = findings.get('baseline', {})
        value = baseline.get('value', 'N/A')
        is_normal = baseline.get('is_normal', True)
        color = "green" if is_normal else "red"
        status = "תקין" if is_normal else ("ברדי" if baseline.get('is_bradycardia') else "טכי")
        
        st.metric(
            label="💓 קו בסיס | Baseline",
            value=f"{value:.0f} bpm" if isinstance(value, (int, float)) else str(value),
            delta=status,
            delta_color="normal" if is_normal else "inverse"
        )
    
    with col2:
        var = findings.get('variability', {})
        value = var.get('value', 'N/A')
        category = var.get('category', 'Unknown')
        is_normal = var.get('is_normal', False)
        
        st.metric(
            label="📊 שונות | Variability",
            value=f"{value:.1f} bpm" if isinstance(value, (int, float)) else str(value),
            delta=category,
            delta_color="normal" if is_normal else "inverse"
        )
    
    with col3:
        decels = findings.get('decelerations', {})
        total = decels.get('total', 0)
        late = decels.get('late', 0)
        variable = decels.get('variable', 0)
        
        st.metric(
            label="📉 האטות | Decelerations",
            value=f"{total}",
            delta=f"מאוחרות: {late}, משתנות: {variable}",
            delta_color="normal" if late == 0 else "inverse"
        )
    
    with col4:
        sinus = findings.get('sinusoidal', {})
        tachy = findings.get('tachysystole', {})
        
        alerts = []
        if sinus.get('detected', False):
            alerts.append("סינוסואידלי ⚠️")
        if tachy.get('detected', False):
            alerts.append("טכיסיסטולה ⚠️")
        
        st.metric(
            label="⚠️ התראות | Alerts",
            value=len(alerts),
            delta=", ".join(alerts) if alerts else "אין",
            delta_color="inverse" if alerts else "normal"
        )
    
    # Override info
    if findings.get('override_applied'):
        st.warning(f"🛡️ **Override הופעל:** {findings.get('override_reason', 'Unknown')}")


def render_alert_panel(patient):
    """Render the alert panel for a patient."""
    alert = patient.latest_alert
    
    if not alert:
        return
    
    st.markdown("### 🚨 התראה | Alert")
    
    cat = patient.latest_category
    cat_colors = {1: "green", 2: "orange", 3: "red"}
    
    # Alert box
    if cat == 3:
        st.error(f"**{alert.headline}**")
    elif cat == 2:
        st.warning(f"**{alert.headline}**")
    else:
        st.success(f"**{alert.headline}**")
    
    st.markdown(f"_{alert.explanation}_")
    
    # Findings list
    if alert.findings:
        st.markdown("**ממצאים:**")
        for finding in alert.findings:
            st.markdown(f"• {finding}")
    
    # Recommendations
    if alert.recommendations:
        st.markdown("**המלצות:**")
        for rec in alert.recommendations:
            st.markdown(f"• {rec}")


def render_event_log(orchestrator: SimulationOrchestrator):
    """Render the event log panel."""
    st.markdown("### 📋 יומן אירועים | Event Log")
    
    log = orchestrator.get_event_log()
    entries = log.get_entries(limit=10)
    
    if not entries:
        st.info("היומן ריק | Log is empty")
        return
    
    for entry in reversed(entries):
        if entry.event_type == 'INJECTION':
            st.markdown(
                f"💉 `{entry.simulation_time:.0f}s` | "
                f"**{entry.patient_id}**: {entry.details.get('injected_event', '?')}"
            )
        elif entry.event_type == 'ALERT':
            cat = entry.details.get('category', '?')
            emoji = {1: "🟢", 2: "🟠", 3: "🔴"}.get(cat, "⚪")
            st.markdown(
                f"{emoji} `{entry.simulation_time:.0f}s` | "
                f"**{entry.patient_id}**: Category {cat}"
            )


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
    if 'selected_patient' not in st.session_state:
        st.session_state.selected_patient = 'P1'
    
    # Get cached orchestrator
    orchestrator = get_orchestrator()
    
    # Render header
    render_header()
    
    st.markdown("---")
    
    # Control panel
    render_control_panel(orchestrator)
    
    st.markdown("---")
    
    # Event injection
    render_event_injection(orchestrator)
    
    st.markdown("---")
    
    # Main content area
    col_overview, col_detail = st.columns([1, 2])
    
    with col_overview:
        render_patient_overview(orchestrator)
        st.markdown("---")
        render_event_log(orchestrator)
    
    with col_detail:
        render_patient_detail(orchestrator)
    
    # Auto-refresh logic
    if orchestrator._running and not orchestrator._paused:
        time.sleep(1)  # Wait 1 second
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#888; font-size:0.8rem;'>"
        "SentinelFetal Simulator v1.0 | Real-Time CTG Training System | "
        "Powered by MOMENT AI"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
