"""
Benchmark: MOMENT model (real) CPU performance.
- Run 50 windows (10 minutes each) through MomentFeatureExtractor (real model).
- Measure load time memory delta (if psutil available).
- Measure mean/median/p95 inference time.
- Compute theoretical throughput (windows per second).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from src.config import CTG
from src.models.moment_encoder import MomentFeatureExtractor

RESULTS_DIR = Path("tests/benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = RESULTS_DIR / "moment_results.json"

N_WINDOWS = 50
WINDOW_SAMPLES = int(10 * 60 * CTG.SAMPLING_RATE)


def _measure_ram_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_benchmark() -> Dict:
    before_ram = _measure_ram_mb()
    load_start = time.perf_counter()
    extractor = MomentFeatureExtractor(device="cpu", use_mock=False)
    load_time = (time.perf_counter() - load_start) * 1000.0
    after_ram = _measure_ram_mb()
    ram_delta = after_ram - before_ram if before_ram is not None and after_ram is not None else None

    times_ms = []
    for _ in range(N_WINDOWS):
        signal = np.random.normal(0, 1, WINDOW_SAMPLES).astype(np.float32)
        t0 = time.perf_counter()
        _ = extractor.extract(signal)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    median_ms = float(np.median(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))

    throughput_windows_per_sec = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    payload = {
        "timestamp": time.time(),
        "load_time_ms": load_time,
        "ram_delta_mb": ram_delta,
        "n_windows": N_WINDOWS,
        "window_seconds": 10 * 60,
        "stats_ms": {
            "mean": mean_ms,
            "median": median_ms,
            "p95": p95_ms,
            "min": float(np.min(times_ms)),
            "max": float(np.max(times_ms)),
        },
        "throughput_windows_per_sec": throughput_windows_per_sec,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[moment] wrote {RESULT_PATH}")
    return payload


if __name__ == "__main__":
    run_benchmark()
