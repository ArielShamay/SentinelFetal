"""
Benchmark: End-to-end clinical category validation (Phase C).

Cases:
- Sinusoidal -> Category 3 (Red)
- Late Severe -> Category >=2
- Bradycardia -> Category >=2
- Normal -> Category 1

Uses mock MOMENT for speed; focuses on rule + override path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import CTG
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.pipeline.container import PipelineContainer
from src.simulation.generators.patient_generator import PatientConfig, PatientGenerator
from src.simulation.events.event_types import (
    EventType,
    LateDecelerationParams,
    BradycardiaParams,
    SinusoidalParams,
    VariabilityParams,
)

RESULTS_DIR = Path("tests/benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = RESULTS_DIR / "clinical_results.json"

SAMPLING_RATE = CTG.SAMPLING_RATE
PRE_EVENT_SEC = 60.0
TOTAL_SEC = 35 * 60.0


def _new_patient(
    baseline_fhr: float = 140.0,
    baseline_variability: float = 8.0,
    noise_std: float = 0.2,
) -> PatientGenerator:
    cfg = PatientConfig(
        patient_id="P-clinical",
        bed_number=1,
        baseline_fhr=baseline_fhr,
        baseline_variability=baseline_variability,
        contractions_per_10min=4.0,
        buffer_duration_minutes=40.0,
    )
    patient = PatientGenerator(cfg)
    # Reduce noise to let absent variability manifest clearly in safety overrides.
    patient._fhr_generator.config.high_freq_noise_std = noise_std
    return patient


def _generate_record(events: List[Tuple[EventType, object]]) -> Tuple[np.ndarray, np.ndarray]:
    # Use lower variability for sinusoidal clarity
    if any(e[0] == EventType.SINUSOIDAL_PATTERN for e in events):
        patient = _new_patient(baseline_variability=3.0, noise_std=0.1)
    else:
        patient = _new_patient()
    pre_samples = int(PRE_EVENT_SEC * SAMPLING_RATE)
    patient.generate_tick(pre_samples)
    for etype, params in events:
        patient.inject_event(etype, params)
    remaining_samples = int((TOTAL_SEC - PRE_EVENT_SEC) * SAMPLING_RATE)
    patient.generate_tick(remaining_samples)
    window = patient._buffer.get_window(None)
    return window["fhr"], window["uc"]


def _expect_category(name: str, category: int) -> bool:
    if name == "sinusoidal":
        return category == 3
    if name == "late_severe":
        return category >= 2
    if name == "brady":
        return category >= 2
    if name == "normal":
        return category == 1
    return False


def run_benchmark() -> Dict:
    container = PipelineContainer.create_default(use_mock_moment=True)
    pipeline = AnalysisPipeline(container)

    cases = [
        ("sinusoidal", [(EventType.SINUSOIDAL_PATTERN, SinusoidalParams(duration_seconds=2100.0, amplitude_bpm=6.0, frequency_cycles_per_min=4.0))]),
        # Pair severe late decels with absent variability to trigger override
        ("late_severe", [
            (EventType.LATE_DECELERATION, LateDecelerationParams.severe()),
            (EventType.ABSENT_VARIABILITY, VariabilityParams.absent()),
        ]),
        # Bradycardia + absent variability to satisfy override rule
        ("brady", [
            (EventType.BRADYCARDIA, BradycardiaParams(target_fhr=80.0, duration_seconds=1500.0)),
            (EventType.ABSENT_VARIABILITY, VariabilityParams.absent()),
        ]),
        ("normal", []),
    ]

    results: List[Dict] = []
    for name, evts in cases:
        fhr, uc = _generate_record(evts)
        result = pipeline.analyze(fhr, uc, sampling_rate=SAMPLING_RATE)
        passed = _expect_category(name, result.category)
        results.append({
            "case": name,
            "category": int(result.category),
            "confidence": float(result.confidence),
            "passed": passed,
        })

    payload = {"results": results}
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[clinical] wrote {RESULT_PATH}")
    return payload


if __name__ == "__main__":
    run_benchmark()
