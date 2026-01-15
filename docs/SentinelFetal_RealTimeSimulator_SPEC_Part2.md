# SentinelFetal Real-Time Simulator
## Technical Specification Document (SPEC) - Part 2

**Version:** 1.0  
**Date:** January 2026  
**Continues from:** SPEC Part 1

---

## 5. Patient Generator (`src/simulation/generators/patient_generator.py`)

```python
"""
Patient Generator - Combined FHR and UC generation for a single patient.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import threading

from .fhr_generator import FHRGenerator, FHRGeneratorConfig
from .uc_generator import UCGenerator, UCGeneratorConfig
from ..core.ring_buffer import RingBuffer
from ..events.event_types import InjectedEvent, EventType, EventParameters


@dataclass
class PatientConfig:
    """Configuration for a simulated patient."""
    patient_id: str
    bed_number: int
    name: str = "יולדת סימולציה"
    baseline_fhr: float = 140.0
    baseline_variability: float = 10.0
    contractions_per_10min: float = 4.0
    buffer_duration_minutes: float = 10.0
    sampling_rate: float = 4.0


class PatientGenerator:
    """
    Generates and manages CTG data for a single patient.
    
    Coordinates FHR and UC generation, handles event injection,
    and maintains a ring buffer of recent data.
    """
    
    def __init__(self, config: PatientConfig):
        self.config = config
        self.patient_id = config.patient_id
        
        # Create generators
        fhr_config = FHRGeneratorConfig(
            baseline_fhr=config.baseline_fhr,
            baseline_variability=config.baseline_variability,
            sampling_rate=config.sampling_rate
        )
        uc_config = UCGeneratorConfig(
            contractions_per_10min=config.contractions_per_10min,
            sampling_rate=config.sampling_rate
        )
        
        self._fhr_generator = FHRGenerator(fhr_config)
        self._uc_generator = UCGenerator(uc_config)
        
        # Ring buffer
        buffer_samples = int(config.buffer_duration_minutes * 60 * config.sampling_rate)
        self._buffer = RingBuffer(max_samples=buffer_samples,
                                  sampling_rate=config.sampling_rate)
        
        # Event management
        self._active_events: List[InjectedEvent] = []
        self._event_lock = threading.Lock()
        
        # State
        self._simulation_time = 0.0
        
        # Latest results (updated by processing)
        self.latest_category: int = 1
        self.latest_alert: Optional[Any] = None
        self.latest_findings: Dict[str, Any] = {}
    
    def generate_tick(self, n_samples: int = 4) -> Dict[str, np.ndarray]:
        """Generate one tick of data (default: 1 second = 4 samples)."""
        dt = 1.0 / self.config.sampling_rate
        
        self._update_active_events()
        
        with self._event_lock:
            active_events = list(self._active_events)
        
        # Generate UC first (needed for decelerations)
        uc, contraction_peaks = self._uc_generator.generate_samples(
            n_samples, active_events
        )
        
        # Generate FHR
        fhr = self._fhr_generator.generate_samples(
            n_samples, active_events, contraction_peaks
        )
        
        # Timestamps
        timestamps = self._simulation_time + np.arange(n_samples) * dt
        
        # Add to buffer
        self._buffer.append_batch(fhr, uc, timestamps)
        
        self._simulation_time = timestamps[-1] + dt
        
        return {
            'fhr': fhr,
            'uc': uc,
            'timestamps': timestamps,
            'contraction_peaks': contraction_peaks
        }
    
    def inject_event(
        self,
        event_type: EventType,
        params: EventParameters,
        duration_seconds: Optional[float] = None
    ) -> InjectedEvent:
        """Inject a clinical event into this patient."""
        duration = duration_seconds or params.duration_seconds
        
        event = InjectedEvent(
            event_type=event_type,
            params=params,
            patient_id=self.patient_id,
            start_time=self._simulation_time,
            end_time=self._simulation_time + duration,
            is_active=True
        )
        
        with self._event_lock:
            self._active_events.append(event)
        
        return event
    
    def _update_active_events(self) -> None:
        """Remove expired events."""
        with self._event_lock:
            self._active_events = [
                e for e in self._active_events
                if e.end_time > self._simulation_time
            ]
    
    def get_buffer_data(self, duration_minutes: Optional[float] = None) -> Dict[str, Any]:
        """Get data from the ring buffer."""
        if duration_minutes:
            data = self._buffer.get_last_n_minutes(duration_minutes)
        else:
            data = self._buffer.get_window()
        
        return {
            **data,
            'patient_id': self.patient_id,
            'bed_number': self.config.bed_number,
            'name': self.config.name,
            'simulation_time': self._simulation_time
        }
    
    def get_active_events(self) -> List[InjectedEvent]:
        """Get currently active events."""
        with self._event_lock:
            return list(self._active_events)
    
    def get_status(self) -> Dict[str, Any]:
        """Get patient status summary."""
        return {
            'patient_id': self.patient_id,
            'bed_number': self.config.bed_number,
            'name': self.config.name,
            'category': self.latest_category,
            'active_events': [e.event_type.name for e in self.get_active_events()],
            'buffer_duration': self._buffer.duration_seconds,
            'simulation_time': self._simulation_time
        }
    
    def reset(self) -> None:
        """Reset patient to initial state."""
        self._fhr_generator.reset()
        self._uc_generator.reset()
        self._buffer.clear()
        with self._event_lock:
            self._active_events.clear()
        self._simulation_time = 0.0
        self.latest_category = 1
        self.latest_alert = None
        self.latest_findings = {}
```

---

## 6. Simulation Orchestrator (`src/simulation/core/orchestrator.py`)

```python
"""
Simulation Orchestrator - Coordinates multiple patient generators.
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import numpy as np

from ..generators.patient_generator import PatientGenerator, PatientConfig
from ..events.event_types import EventType, EventParameters, InjectedEvent
from ..logging.event_logger import EventLogger


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    num_patients: int = 8
    sampling_rate: float = 4.0
    tick_interval_seconds: float = 1.0
    moment_interval_seconds: float = 30.0
    
    patient_names: List[str] = field(default_factory=lambda: [
        "שרה כהן", "רחל לוי", "מירי גולן", "יעל ברק",
        "נועה שמיר", "דנה רוזן", "טלי אברהם", "ליאת פרידמן"
    ])


class SimulationOrchestrator:
    """
    Main orchestrator for real-time CTG simulation.
    
    Responsibilities:
    - Creates and manages PatientGenerators
    - Schedules data generation at regular intervals
    - Coordinates MOMENT processing (staggered)
    - Handles event injection
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        processing_callback: Optional[Callable] = None
    ):
        self.config = config or OrchestratorConfig()
        self._processing_callback = processing_callback
        
        self._patients: Dict[str, PatientGenerator] = {}
        self._create_patients()
        
        # MOMENT scheduling
        self._moment_schedule: List[str] = list(self._patients.keys())
        self._moment_index = 0
        self._last_moment_time = 0.0
        
        # Threading
        self._running = False
        self._paused = False
        self._simulation_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Timing
        self._simulation_time = 0.0
        self._speed_multiplier = 1.0
        
        # Logging
        self._logger = EventLogger()
    
    def _create_patients(self) -> None:
        """Create all patient generators."""
        for i in range(self.config.num_patients):
            patient_id = f"P{i+1}"
            name = self.config.patient_names[i % len(self.config.patient_names)]
            
            config = PatientConfig(
                patient_id=patient_id,
                bed_number=i + 1,
                name=name,
                baseline_fhr=np.random.uniform(130, 150),
                baseline_variability=np.random.uniform(8, 15),
                contractions_per_10min=np.random.uniform(3.5, 4.5)
            )
            
            self._patients[patient_id] = PatientGenerator(config)
    
    def start(self) -> None:
        """Start the simulation."""
        if self._running:
            return
        
        self._running = True
        self._paused = False
        
        self._simulation_thread = threading.Thread(
            target=self._simulation_loop,
            daemon=True
        )
        self._simulation_thread.start()
    
    def stop(self) -> None:
        """Stop the simulation."""
        self._running = False
        if self._simulation_thread:
            self._simulation_thread.join(timeout=2.0)
    
    def pause(self) -> None:
        """Pause the simulation."""
        self._paused = True
    
    def resume(self) -> None:
        """Resume the simulation."""
        self._paused = False
    
    def set_speed(self, multiplier: float) -> None:
        """Set simulation speed (0.5, 1.0, 2.0)."""
        self._speed_multiplier = max(0.5, min(2.0, multiplier))
    
    def _simulation_loop(self) -> None:
        """Main simulation loop in separate thread."""
        last_tick = time.time()
        
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            elapsed = current_time - last_tick
            target_interval = self.config.tick_interval_seconds / self._speed_multiplier
            
            if elapsed >= target_interval:
                self._tick()
                last_tick = current_time
            else:
                time.sleep(0.01)
    
    def _tick(self) -> None:
        """Execute one simulation tick."""
        with self._lock:
            samples_per_tick = int(self.config.sampling_rate * 
                                   self.config.tick_interval_seconds)
            
            # Generate data for all patients
            for patient in self._patients.values():
                patient.generate_tick(samples_per_tick)
            
            self._simulation_time += self.config.tick_interval_seconds
            
            # Check for MOMENT processing
            moment_interval = self.config.moment_interval_seconds / self.config.num_patients
            if self._simulation_time - self._last_moment_time >= moment_interval:
                self._process_next_patient_moment()
                self._last_moment_time = self._simulation_time
    
    def _process_next_patient_moment(self) -> None:
        """Process MOMENT for next patient in schedule."""
        patient_id = self._moment_schedule[self._moment_index]
        self._moment_index = (self._moment_index + 1) % len(self._moment_schedule)
        
        if self._processing_callback:
            patient = self._patients[patient_id]
            data = patient.get_buffer_data(duration_minutes=10)
            
            try:
                results = self._processing_callback(patient_id, data)
                
                if results:
                    patient.latest_category = results.get('category', 1)
                    patient.latest_alert = results.get('alert')
                    patient.latest_findings = results.get('findings', {})
                    
                    if results.get('category', 1) >= 2:
                        self._logger.log_alert(patient_id, results)
            except Exception as e:
                print(f"Error processing {patient_id}: {e}")
    
    def inject_event(
        self,
        patient_id: str,
        event_type: EventType,
        params: EventParameters,
        duration_seconds: Optional[float] = None
    ) -> Optional[InjectedEvent]:
        """Inject an event into a specific patient."""
        with self._lock:
            patient = self._patients.get(patient_id)
            if not patient:
                return None
            
            event = patient.inject_event(event_type, params, duration_seconds)
            self._logger.log_injection(event)
            return event
    
    def get_all_patients_status(self) -> List[Dict[str, Any]]:
        """Get status of all patients, sorted by category."""
        with self._lock:
            statuses = [p.get_status() for p in self._patients.values()]
        
        statuses.sort(key=lambda x: (-x['category'], x['bed_number']))
        return statuses
    
    def get_patient_data(
        self,
        patient_id: str,
        duration_minutes: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Get detailed data for a specific patient."""
        with self._lock:
            patient = self._patients.get(patient_id)
            if not patient:
                return None
            return patient.get_buffer_data(duration_minutes)
    
    def get_patient(self, patient_id: str) -> Optional[PatientGenerator]:
        """Get a patient generator by ID."""
        return self._patients.get(patient_id)
    
    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self._simulation_time
    
    def get_simulation_time_formatted(self) -> str:
        """Get simulation time as HH:MM:SS."""
        total = int(self._simulation_time)
        hours = total // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def reset_all(self) -> None:
        """Reset all patients to initial state."""
        with self._lock:
            for patient in self._patients.values():
                patient.reset()
            self._simulation_time = 0.0
            self._moment_index = 0
            self._last_moment_time = 0.0
            self._logger.clear()
    
    def export_log(self, filepath: str) -> None:
        """Export event log to CSV."""
        self._logger.export_csv(filepath)
```

---

## 7. Pipeline Adapter (`src/simulation/processing/pipeline_adapter.py`)

```python
"""
Pipeline Adapter - Adapts simulation to existing SentinelFetal pipeline.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

# IMPORTANT: Import from existing modules
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig
from src.rules.baseline import calculate_baseline
from src.rules.variability import calculate_variability
from src.rules.decelerations import detect_decelerations
from src.rules.tachysystole import detect_tachysystole
from src.rules.sinusoidal import detect_sinusoidal_pattern
from src.models.moment_encoder import MomentFeatureExtractor
from src.models.fusion import build_feature_vector
from src.models.classifier import XGBClassifierWrapper
from src.analysis.override import apply_medical_override
from src.analysis.alerts import generate_alert


@dataclass
class PipelineAdapterConfig:
    """Configuration for pipeline adapter."""
    use_real_moment: bool = True  # Use real MOMENT model
    model_path: str = "models/xgb_demo.json"
    sampling_rate: float = 4.0


class PipelineAdapter:
    """
    Bridges simulation with existing SentinelFetal pipeline.
    
    Uses:
    - Real MOMENT model (not mock) by default
    - Existing preprocessor, rule engine, classifier
    - Medical override and alert generation
    """
    
    def __init__(self, config: Optional[PipelineAdapterConfig] = None):
        self.config = config or PipelineAdapterConfig()
        
        # Initialize components
        self._preprocessor = CTGPreprocessor(PreprocessingConfig(
            sampling_rate=self.config.sampling_rate
        ))
        
        # MOMENT - use REAL model
        self._moment = MomentFeatureExtractor(
            use_mock=not self.config.use_real_moment
        )
        
        # Classifier
        self._classifier = XGBClassifierWrapper()
        try:
            self._classifier.load_model(self.config.model_path)
            self._classifier_loaded = True
        except Exception:
            print(f"Warning: Could not load classifier from {self.config.model_path}")
            self._classifier_loaded = False
        
        # Embedding cache
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def process_patient(
        self,
        patient_id: str,
        data: Dict[str, Any],
        run_moment: bool = True
    ) -> Dict[str, Any]:
        """
        Process patient data through full pipeline.
        
        Args:
            patient_id: Patient identifier
            data: Dict with 'fhr', 'uc', 'timestamps'
            run_moment: Whether to run MOMENT (expensive)
            
        Returns:
            Dict with category, alert, findings, confidence
        """
        fhr = data['fhr']
        uc = data['uc']
        
        # Check minimum data
        if len(fhr) < 240:  # Less than 1 minute
            return {
                'category': 1,
                'alert': None,
                'findings': {},
                'confidence': 0.0,
                'insufficient_data': True
            }
        
        # Step 1: Preprocessing
        preprocess_result = self._preprocessor.process(fhr.copy())
        fhr_clean = preprocess_result.processed_signal
        
        # Step 2: Rule Engine
        baseline_result = calculate_baseline(fhr_clean, self.config.sampling_rate)
        variability_result = calculate_variability(fhr_clean, self.config.sampling_rate)
        decelerations = detect_decelerations(
            fhr_clean, uc, baseline_result.value, self.config.sampling_rate
        )
        tachysystole_result = detect_tachysystole(uc, self.config.sampling_rate)
        sinusoidal_result = detect_sinusoidal_pattern(fhr_clean, self.config.sampling_rate)
        
        # Step 3: MOMENT Embedding
        if run_moment:
            embedding = self._moment.extract(fhr_clean)
            self._embedding_cache[patient_id] = embedding.embedding
        else:
            embedding_data = self._embedding_cache.get(patient_id)
            if embedding_data is None:
                embedding_data = np.random.randn(1024).astype(np.float32)
            
            class EmbeddingHolder:
                def __init__(self, emb):
                    self.embedding = emb
            embedding = EmbeddingHolder(embedding_data)
        
        # Step 4: Feature Vector
        feature_vector = build_feature_vector(
            embedding=embedding.embedding,
            baseline=baseline_result,
            variability=variability_result,
            decelerations=decelerations,
            tachysystole=tachysystole_result,
            sinusoidal=sinusoidal_result,
            start_idx=0,
            end_idx=len(fhr_clean),
            start_time_sec=0,
            end_time_sec=len(fhr_clean) / self.config.sampling_rate
        )
        
        # Step 5: Classification
        if self._classifier_loaded:
            X = feature_vector.vector.reshape(1, -1)
            ml_prediction = int(self._classifier.predict(X)[0])
            probas = self._classifier.predict_proba(X)[0]
            confidence = float(np.max(probas))
        else:
            ml_prediction = self._rule_based_fallback(
                variability_result, decelerations, baseline_result, sinusoidal_result
            )
            confidence = 0.7
        
        # Step 6: Medical Override
        override_result = apply_medical_override(
            ml_prediction=ml_prediction,
            baseline=baseline_result,
            variability=variability_result,
            decelerations=decelerations,
            tachysystole=tachysystole_result,
            sinusoidal=sinusoidal_result
        )
        
        final_category = override_result.final_category + 1
        
        # Step 7: Generate Alert
        alert = generate_alert(
            category=final_category,
            confidence=confidence,
            baseline=baseline_result,
            variability=variability_result,
            decelerations=decelerations,
            tachysystole=tachysystole_result,
            sinusoidal=sinusoidal_result
        )
        
        # Compile findings
        findings = {
            'baseline': {
                'value': baseline_result.value,
                'is_normal': baseline_result.is_normal,
                'is_bradycardia': baseline_result.is_bradycardia,
                'is_tachycardia': baseline_result.is_tachycardia
            },
            'variability': {
                'value': variability_result.value,
                'category': variability_result.category.name,
                'is_normal': variability_result.is_normal
            },
            'decelerations': {
                'total': len(decelerations),
                'late': sum(1 for d in decelerations if d.decel_type.name == 'LATE'),
                'variable': sum(1 for d in decelerations if d.decel_type.name == 'VARIABLE'),
                'with_severity': sum(1 for d in decelerations if d.has_severity_signs)
            },
            'tachysystole': {
                'detected': tachysystole_result.detected,
                'rate': tachysystole_result.contractions_per_10min
            },
            'sinusoidal': {
                'detected': sinusoidal_result.detected,
                'confidence': sinusoidal_result.confidence
            },
            'override_applied': override_result.should_override,
            'override_reason': override_result.reason.name if override_result.should_override else None
        }
        
        return {
            'category': final_category,
            'alert': alert,
            'findings': findings,
            'confidence': confidence,
            'ml_prediction': ml_prediction + 1,
            'was_overridden': override_result.should_override
        }
    
    def _rule_based_fallback(self, variability, decelerations, baseline, sinusoidal) -> int:
        """Fallback classification when classifier unavailable."""
        if sinusoidal.detected:
            return 2
        
        if variability.category.name == 'ABSENT':
            late_count = sum(1 for d in decelerations if d.decel_type.name == 'LATE')
            var_count = sum(1 for d in decelerations if d.decel_type.name == 'VARIABLE')
            
            if late_count >= 2 or var_count >= 2:
                return 2
            if baseline.is_bradycardia:
                return 2
        
        if (baseline.is_normal and 
            variability.category.name == 'MODERATE' and
            len(decelerations) == 0):
            return 0
        
        return 1
    
    def clear_cache(self, patient_id: Optional[str] = None) -> None:
        """Clear embedding cache."""
        if patient_id:
            self._embedding_cache.pop(patient_id, None)
        else:
            self._embedding_cache.clear()
```

---

## 8. Event Logger (`src/simulation/logging/event_logger.py`)

```python
"""
Event Logger - Lightweight logging without storing raw signals.
"""

import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

from ..events.event_types import InjectedEvent


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: datetime
    simulation_time: float
    event_type: str
    patient_id: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'simulation_time': self.simulation_time,
            'event_type': self.event_type,
            'patient_id': self.patient_id,
            **self.details
        }


class EventLogger:
    """
    Logs events with minimal memory (~100KB max).
    
    Only stores:
    - Event injections
    - Category 2/3 alerts
    - NOT raw signal data
    """
    
    def __init__(self, max_entries: int = 1000):
        self._entries: deque = deque(maxlen=max_entries)
        self._start_time = datetime.now()
    
    def log_injection(self, event: InjectedEvent) -> None:
        """Log an event injection."""
        entry = LogEntry(
            timestamp=datetime.now(),
            simulation_time=event.start_time,
            event_type='INJECTION',
            patient_id=event.patient_id,
            details={
                'injected_event': event.event_type.name,
                'duration_seconds': event.end_time - event.start_time,
                'severity': event.params.severity.name if hasattr(event.params, 'severity') else None
            }
        )
        self._entries.append(entry)
    
    def log_alert(self, patient_id: str, results: Dict[str, Any]) -> None:
        """Log a Category 2/3 alert."""
        entry = LogEntry(
            timestamp=datetime.now(),
            simulation_time=results.get('simulation_time', 0),
            event_type='ALERT',
            patient_id=patient_id,
            details={
                'category': results.get('category'),
                'confidence': results.get('confidence'),
                'was_overridden': results.get('was_overridden', False)
            }
        )
        self._entries.append(entry)
    
    def get_entries(self, event_type: Optional[str] = None) -> List[LogEntry]:
        """Get log entries, optionally filtered by type."""
        if event_type:
            return [e for e in self._entries if e.event_type == event_type]
        return list(self._entries)
    
    def export_csv(self, filepath: str) -> None:
        """Export log to CSV file."""
        if not self._entries:
            return
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'simulation_time', 'event_type', 
                         'patient_id', 'details']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in self._entries:
                row = entry.to_dict()
                row['details'] = str(row.get('details', {}))
                writer.writerow(row)
    
    def clear(self) -> None:
        """Clear all log entries."""
        self._entries.clear()
    
    @property
    def size(self) -> int:
        """Number of entries in log."""
        return len(self._entries)
```

---

## 9. Simulation Dashboard (`src/ui/simulation_app.py`)

```python
"""
Simulation Dashboard - Real-time CTG simulation UI.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional
import time

from src.simulation.core.orchestrator import SimulationOrchestrator, OrchestratorConfig
from src.simulation.processing.pipeline_adapter import PipelineAdapter
from src.simulation.events.event_types import (
    EventType, LateDecelerationParams, VariableDecelerationParams,
    BradycardiaParams, TachycardiaParams, VariabilityParams,
    SinusoidalParams, TachysystoleParams
)
from src.config import COLORS


# Initialize components
@st.cache_resource
def get_pipeline_adapter():
    """Get or create pipeline adapter."""
    return PipelineAdapter()


@st.cache_resource
def get_orchestrator(_pipeline_adapter):
    """Get or create orchestrator with processing callback."""
    config = OrchestratorConfig(num_patients=8)
    
    def processing_callback(patient_id, data):
        return _pipeline_adapter.process_patient(patient_id, data, run_moment=True)
    
    return SimulationOrchestrator(config, processing_callback)


def main():
    st.set_page_config(
        page_title="SentinelFetal - סימולציה",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize
    pipeline = get_pipeline_adapter()
    orchestrator = get_orchestrator(pipeline)
    
    # Session state
    if 'selected_patient' not in st.session_state:
        st.session_state.selected_patient = 'P1'
    
    # Title
    st.title("🏥 SentinelFetal - סימולציית זמן אמת")
    
    # Control Panel
    render_control_panel(orchestrator)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_patient_overview(orchestrator)
    
    with col2:
        render_patient_detail(orchestrator)
    
    # Auto-refresh
    if orchestrator._running and not orchestrator._paused:
        time.sleep(1)
        st.rerun()


def render_control_panel(orchestrator: SimulationOrchestrator):
    """Render simulation control panel."""
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
    
    with col1:
        status = "▶️ פעיל" if orchestrator._running and not orchestrator._paused else "⏸️ מושהה"
        st.metric("סטטוס", status)
    
    with col2:
        st.metric("זמן", orchestrator.get_simulation_time_formatted())
    
    with col3:
        if st.button("▶️ התחל" if not orchestrator._running else "⏸️ השהה"):
            if not orchestrator._running:
                orchestrator.start()
            elif orchestrator._paused:
                orchestrator.resume()
            else:
                orchestrator.pause()
    
    with col4:
        if st.button("🔄 אפס"):
            orchestrator.reset_all()
    
    with col5:
        speed = st.select_slider("מהירות", options=[0.5, 1.0, 2.0], value=1.0)
        orchestrator.set_speed(speed)
    
    # Event Injection
    st.markdown("### 💉 הזרקת אירוע")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        patients = [f"P{i+1}" for i in range(8)]
        target_patient = st.selectbox("יולדת יעד", patients)
    
    with col2:
        event_options = {
            "האטות מאוחרות": EventType.LATE_DECELERATION,
            "האטות משתנות": EventType.VARIABLE_DECELERATION,
            "ברדיקרדיה": EventType.BRADYCARDIA,
            "טכיקרדיה": EventType.TACHYCARDIA,
            "שונות נעדרת": EventType.ABSENT_VARIABILITY,
            "שונות מזערית": EventType.MINIMAL_VARIABILITY,
            "סינוסואידלי": EventType.SINUSOIDAL_PATTERN,
            "טכיסיסטוליה": EventType.TACHYSYSTOLE,
        }
        event_name = st.selectbox("סוג אירוע", list(event_options.keys()))
    
    with col3:
        severity = st.selectbox("חומרה", ["קל", "בינוני", "חמור"])
    
    with col4:
        duration = st.slider("משך (דקות)", 1, 20, 5)
    
    if st.button("💉 הזרק אירוע", type="primary"):
        event_type = event_options[event_name]
        params = get_event_params(event_type, severity)
        orchestrator.inject_event(target_patient, event_type, params, duration * 60)
        st.success(f"אירוע '{event_name}' הוזרק ליולדת {target_patient}")
    
    st.markdown("---")


def get_event_params(event_type: EventType, severity: str):
    """Get event parameters based on type and severity."""
    severity_map = {"קל": "mild", "בינוני": "moderate", "חמור": "severe"}
    sev = severity_map[severity]
    
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
    
    return None


def render_patient_overview(orchestrator: SimulationOrchestrator):
    """Render patient overview grid."""
    st.markdown("### 👥 סקירת יולדות")
    
    statuses = orchestrator.get_all_patients_status()
    
    for status in statuses:
        cat = status['category']
        color = {1: "🟢", 2: "🟠", 3: "🔴"}[cat]
        cat_text = {1: "תקין", 2: "ביניים", 3: "פתולוגי"}[cat]
        
        events = status.get('active_events', [])
        event_text = f" ({', '.join(events)})" if events else ""
        
        if st.button(
            f"{color} {status['patient_id']} - מיטה {status['bed_number']} | קטגוריה {cat} {cat_text}{event_text}",
            key=status['patient_id'],
            use_container_width=True
        ):
            st.session_state.selected_patient = status['patient_id']


def render_patient_detail(orchestrator: SimulationOrchestrator):
    """Render detailed view for selected patient."""
    patient_id = st.session_state.selected_patient
    patient = orchestrator.get_patient(patient_id)
    
    if not patient:
        st.warning("יולדת לא נמצאה")
        return
    
    data = patient.get_buffer_data(duration_minutes=10)
    
    # Header
    cat = patient.latest_category
    color = {1: "green", 2: "orange", 3: "red"}[cat]
    cat_text = {1: "תקין", 2: "ביניים", 3: "פתולוגי"}[cat]
    
    st.markdown(f"### {patient.config.name} - מיטה {patient.config.bed_number}")
    st.markdown(f"<h2 style='color:{color}'>קטגוריה {cat} - {cat_text}</h2>", 
                unsafe_allow_html=True)
    
    # CTG Plot
    fig = create_ctg_plot(data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Findings
    if patient.latest_findings:
        st.markdown("### ממצאים")
        findings = patient.latest_findings
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            baseline = findings.get('baseline', {})
            st.metric("קצב בסיסי", f"{baseline.get('value', 'N/A')} bpm")
        
        with col2:
            var = findings.get('variability', {})
            st.metric("שונות", f"{var.get('value', 'N/A')} bpm ({var.get('category', 'N/A')})")
        
        with col3:
            decels = findings.get('decelerations', {})
            st.metric("האטות", f"{decels.get('total', 0)} (מאוחרות: {decels.get('late', 0)})")
    
    # Alert
    if patient.latest_alert:
        alert = patient.latest_alert
        st.markdown("### 🚨 התראה")
        st.warning(alert.headline)
        st.write(alert.explanation)
        
        if alert.recommendations:
            st.markdown("**המלצות:**")
            for rec in alert.recommendations:
                st.write(f"• {rec}")


def create_ctg_plot(data: dict) -> go.Figure:
    """Create CTG plot with FHR and UC."""
    fhr = data.get('fhr', np.array([]))
    uc = data.get('uc', np.array([]))
    timestamps = data.get('timestamps', np.array([]))
    
    if len(timestamps) == 0:
        return go.Figure()
    
    # Convert to minutes
    times_min = (timestamps - timestamps[0]) / 60 if len(timestamps) > 0 else []
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("FHR - דופק עוברי", "UC - צירים")
    )
    
    # FHR
    fig.add_trace(
        go.Scatter(x=times_min, y=fhr, mode='lines', name='FHR',
                   line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # Reference lines
    fig.add_hline(y=110, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=160, line_dash="dash", line_color="red", row=1, col=1)
    
    # UC
    fig.add_trace(
        go.Scatter(x=times_min, y=uc, mode='lines', name='UC',
                   line=dict(color='green', width=1), fill='tozeroy'),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    fig.update_yaxes(title_text="BPM", range=[50, 200], row=1, col=1)
    fig.update_yaxes(title_text="AU", range=[0, 100], row=2, col=1)
    fig.update_xaxes(title_text="זמן (דקות)", row=2, col=1)
    
    return fig


if __name__ == "__main__":
    main()
```

---

## 10. Module Exports

### 10.1 `src/simulation/__init__.py`

```python
"""
SentinelFetal Real-Time Simulation Module

Provides synthetic CTG data generation for testing and demonstration.
"""

from .core.orchestrator import SimulationOrchestrator, OrchestratorConfig
from .core.ring_buffer import RingBuffer
from .generators.patient_generator import PatientGenerator, PatientConfig
from .generators.fhr_generator import FHRGenerator, FHRGeneratorConfig
from .generators.uc_generator import UCGenerator, UCGeneratorConfig
from .events.event_types import (
    EventType, EventSeverity, EventParameters, InjectedEvent,
    LateDecelerationParams, VariableDecelerationParams,
    BradycardiaParams, TachycardiaParams, VariabilityParams,
    SinusoidalParams, TachysystoleParams
)
from .processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig
from .logging.event_logger import EventLogger

__all__ = [
    # Core
    'SimulationOrchestrator', 'OrchestratorConfig',
    'RingBuffer',
    
    # Generators
    'PatientGenerator', 'PatientConfig',
    'FHRGenerator', 'FHRGeneratorConfig',
    'UCGenerator', 'UCGeneratorConfig',
    
    # Events
    'EventType', 'EventSeverity', 'EventParameters', 'InjectedEvent',
    'LateDecelerationParams', 'VariableDecelerationParams',
    'BradycardiaParams', 'TachycardiaParams', 'VariabilityParams',
    'SinusoidalParams', 'TachysystoleParams',
    
    # Processing
    'PipelineAdapter', 'PipelineAdapterConfig',
    
    # Logging
    'EventLogger',
]
```

---

## 11. Implementation Checklist

### Phase 1: Core Components (Week 1)
- [ ] Create `src/simulation/` directory structure
- [ ] Implement `ring_buffer.py`
- [ ] Implement `event_types.py`
- [ ] Implement `fhr_generator.py`
- [ ] Implement `uc_generator.py`
- [ ] Implement `patient_generator.py`
- [ ] Write unit tests for generators

### Phase 2: Orchestration (Week 2)
- [ ] Implement `orchestrator.py`
- [ ] Implement `pipeline_adapter.py`
- [ ] Implement `event_logger.py`
- [ ] Test MOMENT integration
- [ ] Test multi-patient processing
- [ ] Performance testing (8 patients, i5 CPU)

### Phase 3: UI & Integration (Week 3)
- [ ] Implement `simulation_app.py`
- [ ] Implement control panel components
- [ ] Implement patient overview
- [ ] Implement CTG plotting
- [ ] Integration testing
- [ ] Documentation

---

## 12. Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Memory (8 patients) | < 500MB | `psutil.Process().memory_info().rss` |
| CPU (i5) | < 70% | `psutil.cpu_percent()` |
| UI latency | < 500ms | Time from data generation to display |
| MOMENT latency | < 300ms | Time per patient embedding |
| Data generation | 4 samples/sec/patient | Consistent real-time |

---

## 13. Testing Strategy

### Unit Tests
```bash
# Run all simulation tests
pytest tests/test_simulation/ -v

# Run specific test file
pytest tests/test_simulation/test_generators.py -v
```

### Integration Tests
```bash
# Test full pipeline
pytest tests/test_simulation/test_integration.py -v
```

### Performance Tests
```bash
# Run performance benchmarks
python tests/test_simulation/benchmark.py
```

---

*End of SPEC Document | SentinelFetal Real-Time Simulator | Version 1.0 | January 2026*
