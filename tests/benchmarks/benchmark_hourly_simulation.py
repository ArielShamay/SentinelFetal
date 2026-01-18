"""
Offline one-hour simulation for 4 patients with scheduled event injections.
- Generates synthetic FHR/UC with PatientGenerator over 3600s.
- Injects pathologic patterns for three patients; one control stays normal.
- Runs AnalysisPipeline (mock MOMENT) on rolling 10-min windows every 2 minutes.
- Saves summary JSON to tests/benchmarks/results/hourly_simulation_results.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.config import CTG
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.pipeline.container import PipelineContainer
from src.simulation.generators.patient_generator import PatientConfig, PatientGenerator
from src.simulation.events.event_types import (
    EventType,
    SinusoidalParams,
    LateDecelerationParams,
    VariableDecelerationParams,
    BradycardiaParams,
)

RESULTS_DIR = Path("tests/benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = RESULTS_DIR / "hourly_simulation_results.json"

SAMPLING_RATE = CTG.SAMPLING_RATE
TOTAL_SECONDS = 3600
WINDOW_SECONDS = 10 * 60
STEP_SECONDS = 120


@dataclass
class PatientSchedule:
    name: str
    events: List[Tuple[float, EventType, object]]  # start_time_sec, type, params


def _build_patient(config_baseline: float = 140.0, variability: float = 10.0) -> PatientGenerator:
    cfg = PatientConfig(
        patient_id="P-sim",
        bed_number=1,
        baseline_fhr=config_baseline,
        baseline_variability=variability,
        contractions_per_10min=4.0,
        buffer_duration_minutes=70.0,
    )
    return PatientGenerator(cfg)


def _simulate_patient(schedule: PatientSchedule) -> dict:
    patient = _build_patient()
    current = 0.0
    for start, etype, params in sorted(schedule.events, key=lambda x: x[0]):
        # Generate until event start
        if start > current:
            delta = start - current
            patient.generate_tick(int(delta * SAMPLING_RATE))
            current = start
        patient.inject_event(etype, params)
    # Finish remaining time
    if current < TOTAL_SECONDS:
        patient.generate_tick(int((TOTAL_SECONDS - current) * SAMPLING_RATE))
    window = patient._buffer.get_window(None)
    return {"fhr": window["fhr"], "uc": window["uc"]}


def _analyze_timeline(fhr: np.ndarray, uc: np.ndarray, pipeline: AnalysisPipeline) -> dict:
    win = int(WINDOW_SECONDS * SAMPLING_RATE)
    step = int(STEP_SECONDS * SAMPLING_RATE)
    categories: List[int] = []
    alert_counts = {1: 0, 2: 0, 3: 0}
    timeline = []

    for start in range(0, len(fhr) - win + 1, step):
        end = start + win
        res = pipeline.analyze(fhr[start:end], uc[start:end], sampling_rate=SAMPLING_RATE)
        cat = int(res.category)
        categories.append(cat)
        alert_counts[cat] += 1
        timeline.append({
            "start_sec": start / SAMPLING_RATE,
            "category": cat,
            "confidence": float(res.confidence),
            "sinusoidal": getattr(res.sinusoidal, "detected", False),
        })

    cat_counts = {c: categories.count(c) for c in (1, 2, 3)}
    return {
        "cat_counts": cat_counts,
        "timeline": timeline,
    }


def run_simulation() -> dict:
    schedules = [
        PatientSchedule(
            name="P1_control",
            events=[],
        ),
        PatientSchedule(
            name="P2_sinusoidal",
            events=[(15 * 60, EventType.SINUSOIDAL_PATTERN, SinusoidalParams(duration_seconds=600.0, amplitude_bpm=8.0, frequency_cycles_per_min=4.0))],
        ),
        PatientSchedule(
            name="P3_late",
            events=[
                (20 * 60, EventType.LATE_DECELERATION, LateDecelerationParams.severe()),
                (40 * 60, EventType.LATE_DECELERATION, LateDecelerationParams.severe()),
            ],
        ),
        PatientSchedule(
            name="P4_variable_brady",
            events=[
                (30 * 60, EventType.VARIABLE_DECELERATION, VariableDecelerationParams.severe()),
                (50 * 60, EventType.BRADYCARDIA, BradycardiaParams(target_fhr=80.0, duration_seconds=900.0)),
            ],
        ),
    ]

    container = PipelineContainer.create_default(use_mock_moment=True)
    pipeline = AnalysisPipeline(container)

    results = {}
    for sched in schedules:
        signals = _simulate_patient(sched)
        analysis = _analyze_timeline(signals["fhr"], signals["uc"], pipeline)
        results[sched.name] = {
            "events": [(t, et.name, getattr(p, "duration_seconds", None)) for t, et, p in sched.events],
            "analysis": analysis,
        }

    payload = {
        "sampling_rate": SAMPLING_RATE,
        "total_seconds": TOTAL_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[hourly_simulation] wrote {RESULT_PATH}")
    return payload


if __name__ == "__main__":
    run_simulation()
