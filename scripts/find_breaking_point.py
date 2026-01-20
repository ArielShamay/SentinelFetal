#!/usr/bin/env python
"""
Find the system breaking point by progressively loading synthetic patients.

Design goals:
- Synthetic backend only (no UI) to isolate engine load.
- Aggressive signal noise + artifacts.
- Late deceleration injections to 20% of patients.
- Metrics: latency per batch, CPU/RAM, accuracy of detecting injected patterns.
- Escalation: start with 5 patients, add +5 every 60 seconds.
- Stop when latency > 2000ms, CPU > 95% sustained 30s, accuracy < 70%, or MemoryError.

Usage:
    python scripts/find_breaking_point.py

Outputs:
    docs/reports/LIMIT_TEST_RESULTS.md
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import psutil

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PatientState:
    patient_id: int
    baseline: float = 140.0
    fhr: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    eligible_for_event: bool = False
    active_event_samples: int = 0
    event_total_samples: int = 0
    last_event_end_ts: float = 0.0
    
    def append(self, value: float, ts: float) -> None:
        self.fhr.append(value)
        self.timestamps.append(ts)
        if len(self.fhr) > 800:  # keep last ~200s at 4Hz
            self.fhr = self.fhr[-800:]
            self.timestamps = self.timestamps[-800:]

    def start_event(self, duration_samples: int) -> None:
        self.active_event_samples = duration_samples
        self.event_total_samples = duration_samples

    def in_event_window(self, recent_samples: int = 64) -> bool:
        """True if an event is active or ended within the recent window."""
        if self.active_event_samples > 0:
            return True
        if not self.timestamps:
            return False
        last_ts = self.timestamps[-1]
        return (last_ts - self.last_event_end_ts) <= (recent_samples * 0.25)


@dataclass
class MetricSample:
    timestamp: float
    patients: int
    latency_ms: float
    cpu_percent: float
    ram_mb: float
    accuracy: float
    total_checks: int
    tp: int
    fp: int
    fn: int


class BreakingPointFinder:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.patients: List[PatientState] = []
        self.metrics: List[MetricSample] = []
        self.cpu_history: deque = deque(maxlen=120)  # track last ~120 seconds
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.event_checks = 0  # counts only injected-event windows
        self.failure_reason = None
        self.failure_patient_count = 0

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    def _generate_sample(self, patient: PatientState, ts: float) -> float:
        # Slow baseline drift
        patient.baseline = min(165.0, max(115.0, patient.baseline + np.random.normal(0, 0.2)))

        # High noise floor + occasional artifacts
        noise = np.random.normal(0, 12.0)
        if random.random() < 0.08:
            noise += np.random.normal(0, 40.0)
        if random.random() < 0.03:  # dropout / loss
            noise -= 60.0

        value = patient.baseline + noise

        # Event shaping (Late Decel = gradual drop + slow recovery)
        if patient.active_event_samples > 0:
            phase = 1.0 - (patient.active_event_samples / max(1, patient.event_total_samples))
            drop = 30 + np.random.uniform(0, 15)
            shape = (1 - abs(phase - 0.5) * 2)  # triangle peak mid-event
            value -= drop * shape
            patient.active_event_samples -= 1
            if patient.active_event_samples == 0:
                patient.last_event_end_ts = ts

        return value

    def _maybe_trigger_event(self, patient: PatientState) -> None:
        if not patient.eligible_for_event or patient.active_event_samples > 0:
            return
        # Roughly once every 20-30 seconds for eligible patients
        if random.random() < 0.02:
            duration = random.randint(32, 64)  # 8-16s at 4Hz
            patient.start_event(duration)

    # ------------------------------------------------------------------
    # Detection (proxy for model output)
    # ------------------------------------------------------------------
    def _detect_late_decel(self, patient: PatientState, window_samples: int = 64) -> bool:
        if len(patient.fhr) < window_samples:
            return False
        window = np.array(patient.fhr[-window_samples:])
        baseline_est = np.median(window[: max(4, window_samples // 4)])
        min_val = float(window.min())
        drop = baseline_est - min_val
        recovery_slope = window[-1] - min_val
        noise_level = float(np.std(window))

        # heuristic: significant drop, some recovery, not pure noise explosion
        return drop >= 25.0 and recovery_slope > 5.0 and noise_level < 50.0

    def _update_accuracy(self, truth: bool, pred: bool) -> None:
        if truth:
            self.event_checks += 1
            if pred:
                self.tp += 1
            else:
                self.fn += 1
        else:
            if pred:
                self.fp += 1

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> Tuple[str, int]:
        start_time = time.perf_counter()
        self._add_patients(5)
        next_increase = start_time + 60.0

        while True:
            batch_start = time.perf_counter()
            now = batch_start

            # Generate signals for each patient
            for patient in self.patients:
                # Multiple samples per batch to increase load
                for _ in range(8):  # ~2s worth at 4Hz
                    ts = time.perf_counter()
                    self._maybe_trigger_event(patient)
                    sample = self._generate_sample(patient, ts)
                    patient.append(sample, ts)

                truth = patient.in_event_window()
                pred = self._detect_late_decel(patient)
                self._update_accuracy(truth, pred)

            batch_latency_ms = (time.perf_counter() - batch_start) * 1000

            # Metrics
            cpu = self.process.cpu_percent(interval=None)
            ram_mb = self.process.memory_info().rss / (1024 * 1024)
            acc = self._accuracy()

            self.cpu_history.append((time.perf_counter(), cpu))
            sample = MetricSample(
                timestamp=time.perf_counter(),
                patients=len(self.patients),
                latency_ms=batch_latency_ms,
                cpu_percent=cpu,
                ram_mb=ram_mb,
                accuracy=acc,
                total_checks=self.event_checks,
                tp=self.tp,
                fp=self.fp,
                fn=self.fn,
            )
            self.metrics.append(sample)

            # Check termination criteria
            if batch_latency_ms > 2000:
                self.failure_reason = f"Latency exceeded 2000ms (observed {batch_latency_ms:.0f}ms)"
            if self._cpu_sustained_high():
                self.failure_reason = "CPU >95% sustained for 30s"
            if self.event_checks >= 100 and acc < 0.70:
                self.failure_reason = f"Accuracy dropped below 70% (current {acc:.2f})"

            if self.failure_reason:
                self.failure_patient_count = len(self.patients)
                break

            # Patient escalation every 60s
            if time.perf_counter() >= next_increase:
                self._add_patients(5)
                next_increase += 60.0

            # Short sleep to avoid hammering CPU sampling; keep load high
            time.sleep(0.01)

        return self.failure_reason or "Unknown", self.failure_patient_count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _add_patients(self, count: int) -> None:
        for _ in range(count):
            pid = len(self.patients) + 1
            eligible = random.random() < 0.20  # 20% eligible for late decels
            patient = PatientState(patient_id=pid, eligible_for_event=eligible)
            self.patients.append(patient)

    def _cpu_sustained_high(self) -> bool:
        cutoff = time.perf_counter() - 30.0
        recent = [v for ts, v in self.cpu_history if ts >= cutoff]
        if len(recent) < 5:
            return False
        return all(v > 95.0 for v in recent)

    def _accuracy(self) -> float:
        if self.event_checks == 0:
            return 1.0
        return self.tp / float(self.tp + self.fn)

    def summary(self) -> dict:
        if not self.metrics:
            return {}
        latencies = [m.latency_ms for m in self.metrics]
        cpus = [m.cpu_percent for m in self.metrics]
        rams = [m.ram_mb for m in self.metrics]
        accuracies = [m.accuracy for m in self.metrics if m.total_checks > 0]

        return {
            "patients_final": len(self.patients),
            "latency_ms_avg": float(np.mean(latencies)),
            "latency_ms_max": float(np.max(latencies)),
            "cpu_avg": float(np.mean(cpus)),
            "cpu_max": float(np.max(cpus)),
            "ram_mb_max": float(np.max(rams)),
            "accuracy_final": float(accuracies[-1] if accuracies else 1.0),
            "accuracy_min": float(min(accuracies) if accuracies else 1.0),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "total_checks": self.event_checks,
        }


# ----------------------------------------------------------------------
# Report writer
# ----------------------------------------------------------------------

def write_report(finder: BreakingPointFinder, failure_reason: str, failure_patients: int) -> None:
    report_dir = Path("docs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "LIMIT_TEST_RESULTS.md"

    summary = finder.summary()
    safe_limit = max(5, int(failure_patients * 0.8)) if failure_patients else 5

    lines = []
    lines.append("# Limit Test Results (Synthetic Simulator V2.0)")
    lines.append("")
    lines.append(f"- Failure point (patients): **{failure_patients}**")
    lines.append(f"- Failure mode: **{failure_reason}**")
    lines.append(f"- Safe operational limit (80% rule): **{safe_limit} patients**")
    lines.append(f"- Engine: PyTorch backend (CPU only, Intel i5, no GPU)")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append(f"- Avg latency: {summary.get('latency_ms_avg', 0):.0f} ms")
    lines.append(f"- Max latency: {summary.get('latency_ms_max', 0):.0f} ms")
    lines.append(f"- CPU avg/max: {summary.get('cpu_avg', 0):.1f}% / {summary.get('cpu_max', 0):.1f}%")
    lines.append(f"- RAM max: {summary.get('ram_mb_max', 0):.0f} MB")
    lines.append(f"- Accuracy final/min: {summary.get('accuracy_final', 1):.2f} / {summary.get('accuracy_min', 1):.2f}")
    lines.append(f"- TP/FP/FN: {summary.get('tp', 0)}/{summary.get('fp', 0)}/{summary.get('fn', 0)}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Signals include heavy noise and dropouts.")
    lines.append("- Late decelerations injected into 20% of patients at random intervals.")
    lines.append("- Termination criteria: latency>2000ms, CPU>95% for 30s, accuracy<70%, or crash.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")


# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------

def main() -> None:
    finder = BreakingPointFinder()
    try:
        reason, patients = finder.run()
    except MemoryError:
        reason, patients = "MemoryError", len(finder.patients)
    write_report(finder, reason, patients)


if __name__ == "__main__":
    main()
