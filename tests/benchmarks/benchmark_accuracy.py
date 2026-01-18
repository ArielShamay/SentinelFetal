"""
Benchmark: Rule Engine accuracy on injected synthetic events.
Phase A scope (optimized):
- Generate synthetic signals with PatientGenerator and injected events.
- Use pipeline with mock MOMENT (fast) to measure rule detections.
- Patterns: sinusoidal, late decel (mild/moderate/severe), variable decel (mild/moderate/severe), prolonged decel, bradycardia, tachycardia, normal.
- Metrics: sensitivity per pattern, detection latency (sec) for decels, simple confusion (type mismatch).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import CTG
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.pipeline.container import PipelineContainer
from src.simulation.generators.patient_generator import PatientConfig, PatientGenerator
from src.simulation.events.event_types import (
    EventType,
    EventSeverity,
    LateDecelerationParams,
    VariableDecelerationParams,
    ProlongedDecelerationParams,
    BradycardiaParams,
    TachycardiaParams,
    SinusoidalParams,
)
from src.rules.decelerations import DecelerationType

# Output paths
RESULTS_DIR = Path("tests/benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = RESULTS_DIR / "accuracy_results.json"

# Simulation parameters
SAMPLING_RATE = CTG.SAMPLING_RATE  # typically 4 Hz
PRE_EVENT_SEC = 60.0
DEFAULT_TOTAL_SEC = 35 * 60.0  # 35 minutes to cover tachysystole window
SINUSOIDAL_TOTAL_SEC = 35 * 60.0  # keep parity and exceed sinusoidal 20m requirement
N_PER_PATTERN = 20


def _new_patient(total_seconds: float, baseline_fhr: float = 140.0, baseline_variability: float = 12.0) -> PatientGenerator:
    """Create a patient with buffer big enough for the whole scenario."""
    duration_minutes = total_seconds / 60.0
    cfg = PatientConfig(
        patient_id="P-bench",
        bed_number=1,
        baseline_fhr=baseline_fhr,
        baseline_variability=baseline_variability,
        contractions_per_10min=4.0,
        buffer_duration_minutes=max(duration_minutes + 1.0, 40.0),
    )
    return PatientGenerator(cfg)


def _generate_record(event_type: EventType | None, params, total_seconds: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate full FHR/UC arrays with an optional injected event.
    Returns: fhr, uc, event_start_time (seconds from start).
    """
    # Narrow variability for sinusoidal to make the wave clear; keep baseline stable for others.
    if event_type == EventType.SINUSOIDAL_PATTERN:
        patient = _new_patient(total_seconds, baseline_variability=3.0)
    else:
        patient = _new_patient(total_seconds)
    pre_samples = int(PRE_EVENT_SEC * SAMPLING_RATE)
    patient.generate_tick(pre_samples)  # warm-up baseline

    event_start = patient._simulation_time  # protected, acceptable for bench
    if event_type is not None:
        patient.inject_event(event_type, params)

    remaining_seconds = total_seconds - PRE_EVENT_SEC
    remaining_samples = int(remaining_seconds * SAMPLING_RATE)
    if remaining_samples > 0:
        patient.generate_tick(remaining_samples)

    window = patient._buffer.get_window(None)
    return window["fhr"], window["uc"], event_start


def _detect_from_result(result, expected_type: str) -> Tuple[bool, float | None, str | None]:
    """Check detection and latency from pipeline AnalysisResult."""
    detected = False
    latency = None
    confusion = None

    # Decelerations
    for d in result.decelerations:
        if d.decel_type.value.lower() == expected_type:
            detected = True
            latency = max(0.0, float(d.start_idx) / SAMPLING_RATE - PRE_EVENT_SEC)
            break
        elif expected_type in ("late", "variable", "prolonged"):
            confusion = d.decel_type.value

    # Baseline shifts
    if expected_type == "brady" and result.baseline.value < 110:
        detected = True
    if expected_type == "tachy" and result.baseline.value > 160:
        detected = True

    # Sinusoidal
    if expected_type == "sinusoidal" and getattr(result.sinusoidal, "detected", False):
        detected = True

    # Normal
    if expected_type == "normal" and result.category == 1:
        detected = True

    return detected, latency, confusion


def run_benchmark() -> Dict:
    container = PipelineContainer.create_default(use_mock_moment=True)
    pipeline = AnalysisPipeline(container)

    patterns = [
        # Phase B: longer windows and clearer separations between decel types
        ("sinusoidal", EventType.SINUSOIDAL_PATTERN, SinusoidalParams(duration_seconds=2100.0, amplitude_bpm=6.0, frequency_cycles_per_min=4.0)),
        ("late_mild", EventType.LATE_DECELERATION, LateDecelerationParams(depth_bpm=25.0, lag_seconds=25.0, recovery_seconds=25.0, recurrence_rate=0.4)),
        ("late_moderate", EventType.LATE_DECELERATION, LateDecelerationParams(depth_bpm=35.0, lag_seconds=35.0, recovery_seconds=35.0, recurrence_rate=0.65)),
        ("late_severe", EventType.LATE_DECELERATION, LateDecelerationParams(depth_bpm=55.0, lag_seconds=45.0, recovery_seconds=50.0, recurrence_rate=0.85)),
        ("variable_mild", EventType.VARIABLE_DECELERATION, VariableDecelerationParams(depth_bpm=25.0, duration_decel_seconds=28.0, recurrence_rate=0.35, has_shoulders=True)),
        ("variable_moderate", EventType.VARIABLE_DECELERATION, VariableDecelerationParams(depth_bpm=40.0, duration_decel_seconds=45.0, recurrence_rate=0.55, has_shoulders=True)),
        ("variable_severe", EventType.VARIABLE_DECELERATION, VariableDecelerationParams(depth_bpm=65.0, duration_decel_seconds=65.0, drops_below_70=True, slow_recovery=True, recurrence_rate=0.8)),
        ("prolonged", EventType.PROLONGED_DECELERATION, ProlongedDecelerationParams(duration_seconds=180.0, depth_bpm=45.0)),
        ("brady", EventType.BRADYCARDIA, BradycardiaParams(target_fhr=80.0, duration_seconds=1500.0, severity=EventSeverity.SEVERE)),
        ("tachy", EventType.TACHYCARDIA, TachycardiaParams(target_fhr=190.0, duration_seconds=1500.0, severity=EventSeverity.SEVERE)),
        ("normal", None, None),
    ]

    results: Dict[str, Dict] = {}

    for name, etype, params in patterns:
        total_seconds = SINUSOIDAL_TOTAL_SEC if name == "sinusoidal" else DEFAULT_TOTAL_SEC
        detections: List[bool] = []
        latencies: List[float] = []
        confusions: List[str] = []
        confidences: List[float] = []

        for _ in range(N_PER_PATTERN):
            fhr, uc, event_start = _generate_record(etype, params, total_seconds)
            result = pipeline.analyze(fhr, uc, sampling_rate=SAMPLING_RATE)
            detected, latency, confusion = _detect_from_result(result, _expected_key(name))
            detections.append(detected)
            if latency is not None:
                latencies.append(latency)
            if confusion:
                confusions.append(confusion)
            confidences.append(result.confidence)

        results[name] = {
            "sensitivity": float(np.mean(detections)),
            "n": len(detections),
            "latency_mean_sec": float(np.mean(latencies)) if latencies else None,
            "latency_p95_sec": float(np.percentile(latencies, 95)) if latencies else None,
            "confusions": confusions,
            "confidence_mean": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
        }

    payload = {
        "timestamp": time.time(),
        "sampling_rate": SAMPLING_RATE,
        "n_per_pattern": N_PER_PATTERN,
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[accuracy] wrote {RESULT_PATH}")
    return payload


def _expected_key(name: str) -> str:
    if name.startswith("late"):
        return "late"
    if name.startswith("variable"):
        return "variable"
    if name.startswith("prolonged"):
        return "prolonged"
    if name.startswith("brady"):
        return "brady"
    if name.startswith("tachy"):
        return "tachy"
    if name.startswith("sinusoidal"):
        return "sinusoidal"
    if name.startswith("normal"):
        return "normal"
    return name


if __name__ == "__main__":
    run_benchmark()
