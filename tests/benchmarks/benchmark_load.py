"""
Benchmark: System load & capacity with SimulationOrchestrator.
Phased to 60s per stage at patient counts: 1, 2, 4, 8.
Measures CPU%, tick rate, and identifies when processing time exceeds 1s.
Uses lightweight processing callback (mock MOMENT) to simulate workload.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from src.pipeline.container import PipelineContainer
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.simulation.core.orchestrator import OrchestratorConfig, SimulationOrchestrator

RESULTS_DIR = Path("tests/benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = RESULTS_DIR / "load_results.json"

STAGES = [1, 2, 4, 8]
DURATION_SEC = 60


def _processing_callback(patient_id: str, data: Dict) -> Dict:
    """Lightweight processing to simulate work (mock MOMENT + simple ops)."""
    # Use a tiny numeric workload to create CPU activity
    _ = np.mean(data.get("fhr", [])) + np.std(data.get("uc", []))
    return {"ok": True}


def _measure_cpu() -> float | None:
    if psutil is None:
        return None
    return psutil.cpu_percent(interval=None)


def run_stage(num_patients: int) -> Dict:
    cfg = OrchestratorConfig(num_patients=num_patients, tick_interval_seconds=1.0, moment_interval_seconds=20.0)
    orch = SimulationOrchestrator(config=cfg, processing_callback=_processing_callback)
    orch.start()

    cpu_samples: List[float] = []
    tick_start = time.perf_counter()
    time.sleep(1.0)  # allow warm-up

    start = time.perf_counter()
    while time.perf_counter() - start < DURATION_SEC:
        if psutil is not None:
            cpu_samples.append(_measure_cpu())
        time.sleep(1.0)

    orch.stop()
    elapsed = time.perf_counter() - tick_start
    tick_rate = orch._tick_count / elapsed if elapsed > 0 else 0.0
    worst_tick_delay = elapsed / max(orch._tick_count, 1)

    return {
        "patients": num_patients,
        "cpu_mean": float(np.mean(cpu_samples)) if cpu_samples else None,
        "cpu_p95": float(np.percentile(cpu_samples, 95)) if cpu_samples else None,
        "ticks": orch._tick_count,
        "tick_rate_hz": tick_rate,
        "worst_tick_sec": worst_tick_delay,
        # Allow a small tolerance (20ms) for scheduler jitter when classifying lag.
        "lag": worst_tick_delay > 1.02,
    }


def run_benchmark() -> Dict:
    stages = []
    for n in STAGES:
        stages.append(run_stage(n))

    payload = {
        "timestamp": time.time(),
        "duration_sec_per_stage": DURATION_SEC,
        "stages": stages,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[load] wrote {RESULT_PATH}")
    return payload


if __name__ == "__main__":
    run_benchmark()
