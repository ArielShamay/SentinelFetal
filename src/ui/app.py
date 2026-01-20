# -*- coding: utf-8 -*-
"""
SentinelFetal - Multi-Patient Simulation Dashboard (Clinical Minimalism).

Key principles:
- No static CTU data. Everything is synthetic and generated on the fly.
- White background (#FFFFFF), black text (#000000).
- Semantic colors only for status/alerts: Green, Orange, Red.
- Lightweight grid view; richer XAI view only in the detail screen.

Run:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import random
import string
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.ui.plots import (
    create_ctg_plot,
    create_patient_detail_plot,
    create_patient_sparkline,
)


# =============================================================================
# Styling (clinical minimalism)
# =============================================================================

WHITE_BG = "#FFFFFF"
BLACK_TEXT = "#000000"
GREEN = "#1A8F2B"
ORANGE = "#CC7A00"
RED = "#B00020"


def inject_minimal_css() -> None:
    st.markdown(
        f"""
        <style>
        .main .block-container {{
            padding: 1rem 1.5rem 2rem 1.5rem;
            max-width: 1400px;
        }}
        body, .stApp {{
            background: {WHITE_BG};
            color: {BLACK_TEXT};
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {{
            color: {BLACK_TEXT};
            margin-bottom: 0.35rem;
        }}
        /* Hide Streamlit default decoration */
        header, footer {{ display: none !important; }}
        .st-emotion-cache-18ni7ap {{ padding: 0; }}
        .stButton>button {{
            background: {BLACK_TEXT};
            color: {WHITE_BG};
            border-radius: 4px;
            border: 1px solid {BLACK_TEXT};
            padding: 0.5rem 0.75rem;
        }}
        .stButton>button:hover {{
            border: 1px solid {BLACK_TEXT};
            background: {WHITE_BG};
            color: {BLACK_TEXT};
        }}
        .card {{
            border: 1px solid #e5e5e5;
            border-radius: 6px;
            padding: 0.75rem;
            margin-bottom: 1rem;
            background: {WHITE_BG};
        }}
        .card:hover {{ border-color: #bdbdbd; }}
        .status-pill {{
            display: inline-block;
            padding: 0.1rem 0.45rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .tooltip {{
            font-size: 0.85rem;
            opacity: 0.7;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class SyntheticEvent:
    event_type: str
    start_ts: float
    end_ts: float


@dataclass
class SyntheticPatient:
    patient_id: str
    name: str
    age: int
    gestational_age_weeks: int
    fhr: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    baseline: float = 140.0
    risk_level: str = "Normal"  # Normal | Warning | Critical
    events: List[SyntheticEvent] = field(default_factory=list)
    active_event: Optional[SyntheticEvent] = None

    def append_point(self, value: float, ts: float) -> None:
        self.fhr.append(value)
        self.timestamps.append(ts)
        # keep last 300 points (~75s at 4Hz) for UI responsiveness
        if len(self.fhr) > 300:
            self.fhr = self.fhr[-300:]
            self.timestamps = self.timestamps[-300:]

    @property
    def last_ts(self) -> float:
        return self.timestamps[-1] if self.timestamps else 0.0


# =============================================================================
# Session state helpers
# =============================================================================


PATIENT_NAMES = [
    "Sarah Cohen", "Rachel Levy", "Miri Golan", "Yael Barak", "Noa Shamir",
    "Dana Rosen", "Tali Avraham", "Liat Friedman", "Hila David", "Adi Moshe",
    "Ronit Ben", "Shira Gal", "Avital Cohen", "Maya Azulay", "Noga Hadar",
    "Eden Peretz", "Linor Saar", "Ortal Aviv", "Naama Oren", "Tamar Paz",
]


def _random_id() -> str:
    return "P" + "".join(random.choices(string.digits, k=3))


def _create_patient(idx: int) -> SyntheticPatient:
    name = PATIENT_NAMES[idx % len(PATIENT_NAMES)]
    age = random.randint(24, 39)
    ga = random.randint(36, 41)
    patient = SyntheticPatient(
        patient_id=_random_id(),
        name=name,
        age=age,
        gestational_age_weeks=ga,
    )
    # prime with 10 seconds of baseline noise
    now = time.time()
    for i in range(40):  # 4Hz * 10s
        value = 140 + np.random.normal(0, 2)
        patient.append_point(float(value), now - (40 - i) * 0.25)
    return patient


def ensure_session_state() -> None:
    if "patients" not in st.session_state:
        st.session_state.patients = []
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "current_view" not in st.session_state:
        st.session_state.current_view = "grid"  # or patient_id
    if "selected_patient" not in st.session_state:
        st.session_state.selected_patient = None
    if "last_rerun" not in st.session_state:
        st.session_state.last_rerun = 0.0


# =============================================================================
# Simulation core (lightweight)
# =============================================================================


def update_signals() -> None:
    """Generate new FHR samples for each active patient."""
    if not st.session_state.is_running:
        return
    now = time.time()
    for patient in st.session_state.patients:
        # Determine if an event is active
        if patient.active_event and now > patient.active_event.end_ts:
            patient.events.append(patient.active_event)
            patient.active_event = None

        drop = 0.0
        risk = "Normal"
        if patient.active_event:
            remaining = patient.active_event.end_ts - now
            if patient.active_event.event_type == "Late Deceleration":
                drop = 25 + np.random.uniform(0, 15)
                risk = "Warning"
            elif patient.active_event.event_type == "Variable Deceleration":
                drop = 35 + np.random.uniform(-5, 10)
                risk = "Critical"
            elif patient.active_event.event_type == "Tachysystole":
                drop = 15 + np.random.uniform(-5, 5)
                risk = "Warning"
            # taper toward end
            drop *= max(0.2, min(1.0, remaining / 10.0))

        noise = np.random.normal(0, 3)
        drift = np.random.normal(0, 0.1)
        patient.baseline = min(160, max(120, patient.baseline + drift))
        value = patient.baseline - drop + noise
        patient.risk_level = risk if patient.active_event else "Normal"
        patient.append_point(float(value), now)


def start_simulation(n: int) -> None:
    st.session_state.patients = [_create_patient(i) for i in range(n)]
    st.session_state.is_running = True
    st.session_state.current_view = "grid"
    st.session_state.last_rerun = time.time()


def stop_simulation() -> None:
    st.session_state.is_running = False


def inject_event(patient_id: str, event_type: str, duration_sec: int = 12) -> None:
    patient = next((p for p in st.session_state.patients if p.patient_id == patient_id), None)
    if not patient:
        return
    now = time.time()
    patient.active_event = SyntheticEvent(
        event_type=event_type,
        start_ts=now,
        end_ts=now + duration_sec,
    )


# =============================================================================
# UI helpers
# =============================================================================


def status_color(label: str) -> str:
    return {"Normal": GREEN, "Warning": ORANGE, "Critical": RED}.get(label, BLACK_TEXT)


def render_command_bar() -> None:
    st.markdown("## Multi-Patient Simulation")
    st.markdown("Clinical Minimalism · Real-time synthetic monitoring")

    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.2])
    with col1:
        num = st.slider("Number of Patients", min_value=1, max_value=20, value=6, step=1)
    with col2:
        if st.button("Start Simulation"):
            start_simulation(num)
            st.experimental_rerun()
    with col3:
        if st.button("Stop Simulation"):
            stop_simulation()
    with col4:
        st.markdown(f"**Status:** {'Running' if st.session_state.is_running else 'Stopped'}")

    st.markdown("---")

    # Event injection console
    st.markdown("### Event Injection Console")
    if st.session_state.patients:
        p_options = {f"{p.name} ({p.patient_id})": p.patient_id for p in st.session_state.patients}
        colp, cole, colb = st.columns([1.5, 1.2, 0.8])
        with colp:
            patient_key = st.selectbox("Select Patient", list(p_options.keys()))
        with cole:
            event_type = st.selectbox("Event Type", [
                "Late Deceleration",
                "Variable Deceleration",
                "Tachysystole",
            ])
        with colb:
            if st.button("Inject Event"):
                inject_event(p_options[patient_key], event_type)
                st.session_state["event_feedback"] = (
                    f"Event injected. System expected to detect within 10-15 seconds."
                )
    else:
        st.info("Start the simulation to enable event injection.")

    if st.session_state.get("event_feedback"):
        st.markdown(st.session_state["event_feedback"])

    st.markdown("---")


def render_grid_view() -> None:
    patients = st.session_state.patients
    if not patients:
        st.info("No patients yet. Start the simulation.")
        return

    # paced auto-refresh while running
    now = time.time()
    if st.session_state.is_running and now - st.session_state.last_rerun > 1.0:
        st.session_state.last_rerun = now
        st.experimental_rerun()

    n_cols = 3
    rows = [patients[i:i + n_cols] for i in range(0, len(patients), n_cols)]
    for row in rows:
        cols = st.columns(len(row))
        for col, patient in zip(cols, row):
            with col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**{patient.name}** · {patient.patient_id}")
                st.markdown(
                    f"<span class='tooltip'>Hover for details</span>", unsafe_allow_html=True
                )
                risk = patient.risk_level
                st.markdown(
                    f"<span class='status-pill' style='background:{status_color(risk)}20; color:{status_color(risk)};'>"
                    f"{risk}</span>",
                    unsafe_allow_html=True,
                )

                fig = create_patient_sparkline(patient)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                if st.button("Open Detail", key=f"btn-{patient.patient_id}"):
                    st.session_state.current_view = patient.patient_id
                    st.experimental_rerun()
                st.markdown("</div>", unsafe_allow_html=True)


def render_detail_view(patient_id: str) -> None:
    patient = next((p for p in st.session_state.patients if p.patient_id == patient_id), None)
    if not patient:
        st.session_state.current_view = "grid"
        st.experimental_rerun()
        return

    st.markdown(f"## Patient Detail · {patient.name} ({patient.patient_id})")
    col_meta, col_plot = st.columns([0.32, 0.68])

    with col_meta:
        st.markdown("**Profile**")
        st.markdown(f"Age: {patient.age}")
        st.markdown(f"Gestational Age: {patient.gestational_age_weeks} weeks")
        st.markdown(f"Current Status: ")
        st.markdown(
            f"<span class='status-pill' style='background:{status_color(patient.risk_level)}20; color:{status_color(patient.risk_level)};'>"
            f"{patient.risk_level}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("**Active/Recent Events**")
        active = patient.active_event
        if active:
            st.markdown(f"- {active.event_type} (active)")
        for ev in patient.events[-3:]:
            st.markdown(f"- {ev.event_type} (completed)")
        if not active and not patient.events:
            st.markdown("- None")
        if st.button("Back to Dashboard"):
            st.session_state.current_view = "grid"
            st.experimental_rerun()

    with col_plot:
        fig = create_patient_detail_plot(patient)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# =============================================================================
# Main entry
# =============================================================================


def main() -> None:
    ensure_session_state()
    inject_minimal_css()

    # Update signals on every rerun when running
    update_signals()

    render_command_bar()

    if st.session_state.current_view == "grid":
        render_grid_view()
    else:
        render_detail_view(st.session_state.current_view)


if __name__ == "__main__":
    main()


def render_no_model_warning():
    """Render warning when model is not available."""
    st.warning(
        """
        ⚠️ **מודל ML לא נמצא**
        
        לא נמצא קובץ מודל ב-`models/sentinel_classifier.json`.
        המערכת תשתמש בסיווג מבוסס חוקים בלבד.
        
        לאימון מודל הרץ:
        ```
        python src/training/train_demo.py
        ```
        """
    )


def render_no_moment_warning():
    """Render warning when MOMENT encoder is not available."""
    st.warning(
        """
        ⚠️ **מנוע MOMENT לא נמצא**
        
        לא ניתן לטעון את מודל MOMENT להפקת אמבדינגים.
        
        לתיקון:
        1. ייצא את המודל ל-ONNX: `python scripts/export_moment_onnx.py`
        2. או התקן את momentfm: `pip install momentfm torch`
        
        המערכת תשתמש בסיווג מבוסס חוקים בלבד.
        """
    )


def render_welcome_screen():
    """Render welcome screen when no patient is selected."""
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
            border-radius: 16px;
            margin: 2rem 0;
        ">
            <h1 style="color: white; font-size: 2.5rem; margin-bottom: 1rem;">
                🏥 SentinelFetal
            </h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.25rem; margin-bottom: 2rem;">
                מערכת AI לניטור דופק עוברי בזמן לידה
            </p>
            <p style="color: rgba(255,255,255,0.7); font-size: 1rem;">
                בחר רשומה מהתפריט בצד שמאל להתחלת ניתוח
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style="
                background: {COLORS['card']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
                height: 200px;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div>
                <h3 style="color: {COLORS['text_primary']}; margin-bottom: 0.5rem;">MOMENT AI</h3>
                <p style="color: {COLORS['text_secondary']}; font-size: 0.9rem;">
                    מודל Transformer מתקדם לניתוח סדרות זמן
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="
                background: {COLORS['card']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
                height: 200px;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📋</div>
                <h3 style="color: {COLORS['text_primary']}; margin-bottom: 0.5rem;">מנוע חוקים</h3>
                <p style="color: {COLORS['text_secondary']}; font-size: 0.9rem;">
                    זיהוי אוטומטי של האטות, שונות וקו בסיס
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="
                background: {COLORS['card']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
                height: 200px;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🚨</div>
                <h3 style="color: {COLORS['text_primary']}; margin-bottom: 0.5rem;">התראות בעברית</h3>
                <p style="color: {COLORS['text_secondary']}; font-size: 0.9rem;">
                    הסברים קליניים והמלצות לצוות רפואי
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_patient_detail_v2(results: Dict[str, Any], record_id: str):
    """
    Render patient details with professional medical UI layout.
    
    Layout:
        - Category banner (top, full width)
        - CTG Plot (main area, 2/3 width)
        - AI Analysis Panel (side, 1/3 width)
        - Clinical Explainability (bottom, full width)
    """
    alert: Alert = results['alert']
    category = results['final_category']
    emoji = get_category_emoji(category)
    
    # Category Banner
    st.markdown(
        category_banner_html(
            category=category,
            headline=f"{emoji} {alert.headline}",
            record_id=record_id,
            confidence=results['confidence']
        ),
        unsafe_allow_html=True
    )
    
    # Main content: CTG Plot (2/3) + AI Analysis Panel (1/3)
    col_plot, col_analysis = st.columns([2, 1])
    
    with col_plot:
        # CTG Plot Section
        st.markdown(section_header_html("גרף CTG", "📊"), unsafe_allow_html=True)
        fig = create_ctg_plot(
            fhr=results['fhr'],
            uc=results['uc'],
            decelerations=results['decelerations'],
            sampling_rate=SAMPLING_RATE,
            title=f"ניטור מיטה {record_id}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_analysis:
        # AI Analysis Panel
        st.markdown(section_header_html("ניתוח AI", "🤖"), unsafe_allow_html=True)
        
        # Metrics in a grid
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("קו בסיס", f"{results['baseline'].value:.0f} bpm")
            st.metric("שונות", f"{results['variability'].value:.1f} bpm")
        with metric_col2:
            st.metric("קטגוריה", f"Cat {category}")
            late_count = sum(1 for d in results['decelerations'] 
                           if d.decel_type == DecelerationType.LATE)
            st.metric("האטות מאוחרות", late_count)
        
        st.markdown("---")
        
        # Findings
        st.markdown(section_header_html("ממצאים", "🔍"), unsafe_allow_html=True)
        for finding in alert.findings:
            st.markdown(finding_card_html(finding), unsafe_allow_html=True)
    
    # Clinical Explainability Section (full width)
    st.markdown("---")
    col_recs, col_explain = st.columns(2)
    
    with col_recs:
        st.markdown(section_header_html("המלצות קליניות", "💡"), unsafe_allow_html=True)
        for rec in alert.recommendations:
            st.markdown(recommendation_html(rec, category), unsafe_allow_html=True)
    
    with col_explain:
        st.markdown(section_header_html("הסבר קליני", "📝"), unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="
                background: {COLORS['card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 1rem;
                color: {COLORS['text_primary']};
                line-height: 1.6;
            ">
                {alert.explanation}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Technical Details (expandable)
    with st.expander("🔧 פרטים טכניים מתקדמים"):
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            st.metric("קטגוריית שונות", results['variability'].category.name)
        
        with tech_col2:
            variable_count = sum(1 for d in results['decelerations'] 
                               if d.decel_type == DecelerationType.VARIABLE)
            st.metric("האטות משתנות", variable_count)
        
        with tech_col3:
            st.metric("סה״כ האטות", len(results['decelerations']))
        
        with tech_col4:
            st.metric("סינוסואידלי", "✓ נמצא" if results['sinusoidal'].detected else "✗ לא נמצא")
        
        # Override information if available
        if results.get('override_result'):
            override = results['override_result']
            if override.was_overridden:
                st.info(f"⚠️ Override: {override.reason}")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application."""
    # Page config (MUST be first Streamlit command)
    st.set_page_config(
        page_title="SentinelFetal - מערכת ניטור עוברי",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS (hide Streamlit chrome, apply medical theme)
    inject_custom_css()
    
    # Load data
    loader = load_data_loader()
    if loader is None:
        st.error("לא ניתן לטעון את מאגר הנתונים. בדוק את הנתיב.")
        st.stop()
    
    # Check for model
    classifier = load_classifier()
    if classifier is None:
        render_no_model_warning()
    
    # Check for MOMENT encoder
    moment_encoder = load_moment_encoder()
    if moment_encoder is None:
        render_no_moment_warning()
    
    # Sidebar - Patient selection
    selected_record = render_sidebar(loader)
    
    # Main content area
    if selected_record:
        # Process selected record with detailed progress indicators
        try:
            record = loader.load_record(selected_record)
            use_ml = st.session_state.get('use_ml', True) and classifier is not None and moment_encoder is not None
            results = run_full_pipeline(record, use_ml=use_ml, show_progress=True)
            
            # Render results with new professional layout
            render_patient_detail_v2(results, selected_record)
            
        except Exception as e:
            st.error(f"שגיאה בעיבוד הרשומה: {e}")
            logger.exception("Pipeline error")
    else:
        # Welcome screen
        render_welcome_screen()


if __name__ == "__main__":
    main()
