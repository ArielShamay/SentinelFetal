"""
Real-Time Endurance Benchmark

Simulates an 8-patient real-time session at 1 Hz to monitor stability.
- Runs synchronous ticks (no background thread) to preserve pacing.
- Logs CPU, RAM, and tick lag every 60s to CSV.
- Randomly injects a pathological event every 5 minutes to stress detection.
- Appends a summary to docs/ENDURANCE_TEST_REPORT.md after completion.

Usage:
    python tests/benchmarks/benchmark_endurance.py --duration 300
    python tests/benchmarks/benchmark_endurance.py --duration 3600
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "psutil is required for endurance benchmarking. Install with: pip install psutil"
    ) from exc

# Ensure local imports work when executed as a script
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import (  # noqa: E402
    SimulationOrchestrator,
    OrchestratorConfig,
    PipelineAdapter,
    PipelineAdapterConfig,
    EventType,
    LateDecelerationParams,
    SinusoidalParams,
)

LOG_EVERY_SECONDS = 60
INJECTION_INTERVAL_SECONDS = 300
DEFAULT_DURATION_SECONDS = 3600
RESULTS_PATH = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "endurance_log.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "ENDURANCE_TEST_REPORT.md"


def build_processing_callback(use_real_moment: bool = False):
    """Create a processing callback wired to the pipeline adapter."""
    adapter = PipelineAdapter(
        PipelineAdapterConfig(
            use_real_moment=use_real_moment,
            model_path="models/xgb_demo.json",
            min_data_seconds=60.0,
        )
    )

    def _callback(patient_id: str, data: Dict) -> Dict:
        return adapter.process_patient(patient_id, data, run_moment=True)

    return _callback


def ensure_result_paths() -> None:
    """Create result/report paths if missing."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_report(summary: Dict) -> None:
    """Append a short summary row to the endurance report."""
    header = [
        "timestamp",
        "duration_sec",
        "max_ram_mb",
        "start_ram_mb",
        "growth_mb_per_hr",
        "tick_count",
        "injections",
        "notes",
    ]

    if not REPORT_PATH.exists():
        with REPORT_PATH.open("w", encoding="utf-8") as f:
            f.write("# Endurance Test Report\n\n")
            f.write("| " + " | ".join(header) + " |\n")
            f.write("|" + "---|" * len(header) + "\n")

    # Append markdown table row
    line = "| " + " | ".join(
        [
            summary["timestamp"],
            f"{summary['duration_sec']:.0f}",
            f"{summary['max_ram_mb']:.1f}",
            f"{summary['start_ram_mb']:.1f}",
            f"{summary['growth_mb_per_hr']:.2f}",
            str(summary["tick_count"]),
            str(summary["injections"]),
            summary["notes"],
        ]
    ) + " |\n"
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def record_metrics_row(writer: csv.DictWriter, row: Dict) -> None:
    writer.writerow(row)


def run_endurance(duration_sec: int) -> Dict:
    ensure_result_paths()

    # Prime CPU percent measurement
    psutil.cpu_percent(interval=None)
    process = psutil.Process()

    processing_cb = build_processing_callback(use_real_moment=False)
    orchestrator = SimulationOrchestrator(
        OrchestratorConfig(
            num_patients=8,
            sampling_rate=4.0,
            tick_interval_seconds=1.0,
            moment_interval_seconds=30.0,
        ),
        processing_cb,
    )

    random.seed(42)
    injections = 0
    next_injection = INJECTION_INTERVAL_SECONDS

    metrics: List[Dict] = []
    start_time = time.time()
    last_log = 0.0
    tick_count = 0

    fieldnames = [
        "timestamp",
        "elapsed_sec",
        "ram_mb",
        "cpu_percent",
        "tick_ms",
        "injections",
    ]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                loop_start = time.time()

                # One synchronous tick
                orchestrator._tick()
                tick_count += 1

                tick_ms = (time.time() - loop_start) * 1000.0
                elapsed = time.time() - start_time

                # Pathology injection every interval
                if elapsed >= next_injection:
                    patient_id = f"P{random.randint(1, 8)}"
                    event_type = random.choice(
                        [EventType.SINUSOIDAL_PATTERN, EventType.LATE_DECELERATION]
                    )
                    if event_type == EventType.SINUSOIDAL_PATTERN:
                        params = SinusoidalParams.typical()
                    else:
                        params = LateDecelerationParams.severe()
                    orchestrator.inject_event(patient_id, event_type, params)
                    injections += 1
                    next_injection += INJECTION_INTERVAL_SECONDS

                # Periodic metrics logging
                if elapsed - last_log >= LOG_EVERY_SECONDS:
                    ram_mb = process.memory_info().rss / (1024 * 1024)
                    cpu_percent = psutil.cpu_percent(interval=None)
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_sec": round(elapsed, 1),
                        "ram_mb": round(ram_mb, 2),
                        "cpu_percent": round(cpu_percent, 2),
                        "tick_ms": round(tick_ms, 3),
                        "injections": injections,
                    }
                    metrics.append(row)
                    record_metrics_row(writer, row)
                    csvfile.flush()
                    last_log = elapsed

                if elapsed >= duration_sec:
                    break

                sleep_time = max(0.0, 1.0 - (time.time() - loop_start))
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            logging.info("Interrupted by user; finishing early.")

    total_elapsed = time.time() - start_time
    start_ram = metrics[0]["ram_mb"] if metrics else process.memory_info().rss / (1024 * 1024)
    max_ram = max(m["ram_mb"] for m in metrics) if metrics else start_ram
    growth_mb_per_hr = (max_ram - start_ram) / (total_elapsed / 3600) if total_elapsed > 0 else 0.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_sec": total_elapsed,
        "max_ram_mb": max_ram,
        "start_ram_mb": start_ram,
        "growth_mb_per_hr": growth_mb_per_hr,
        "tick_count": tick_count,
        "injections": injections,
        "notes": "sync tick loop",
    }

    append_report(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time endurance benchmark")
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help="Duration in seconds (default: 3600)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_endurance(args.duration)

    print("=" * 60)
    print("Endurance Test Complete")
    print(f"Duration: {summary['duration_sec']:.1f} sec")
    print(f"Ticks: {summary['tick_count']}")
    print(f"Injections: {summary['injections']}")
    print(f"Max RAM: {summary['max_ram_mb']:.2f} MB")
    print(f"Growth rate: {summary['growth_mb_per_hr']:.2f} MB/hr")
    print(f"Log: {RESULTS_PATH}")
    print(f"Report: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
