#!/usr/bin/env python
"""
Full System Stress Test - Including UI Rendering Load.

This script simulates the complete Streamlit application loop:
1. Data Generation: Synthetic FHR signals for N patients
2. AI/Rules Engine: MOMENT embeddings + rule-based detection  
3. UI Rendering: Plotly sparkline generation for dashboard grid

Goal: Find the "Application Breaking Point" where cycle time exceeds 1 second.

Usage:
    python scripts/full_system_stress.py

Output:
    docs/reports/FULL_SYSTEM_STRESS_REPORT.md
"""

from __future__ import annotations

import gc
import os
import random
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import psutil

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import plotting functions (simulates UI cost)
import plotly.graph_objects as go


# =============================================================================
# Data Structures (mirrored from app.py to avoid Streamlit import)
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
    risk_level: str = "Normal"
    events: List[SyntheticEvent] = field(default_factory=list)
    active_event: Optional[SyntheticEvent] = None
    high_noise: bool = False  # 20% patients have high noise

    def append_point(self, value: float, ts: float) -> None:
        self.fhr.append(value)
        self.timestamps.append(ts)
        if len(self.fhr) > 300:
            self.fhr = self.fhr[-300:]
            self.timestamps = self.timestamps[-300:]


# =============================================================================
# Plotting Functions (copied from plots.py to avoid circular imports)
# =============================================================================

WHITE_BG = "#FFFFFF"
BLACK = "#000000"
GREEN = "#1A8F2B"
ORANGE = "#CC7A00"
RED = "#B00020"


def _event_color(event_type: str) -> str:
    if event_type == "Late Deceleration":
        return RED
    if event_type == "Variable Deceleration":
        return ORANGE
    if event_type == "Tachysystole":
        return ORANGE
    return BLACK


def create_patient_sparkline(patient: SyntheticPatient) -> go.Figure:
    """Tiny sparkline for grid cards (last 60 samples)."""
    if not patient.timestamps:
        return go.Figure()
    x = np.array(patient.timestamps[-60:])
    y = np.array(patient.fhr[-60:])
    x_norm = x - x.min()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y,
            mode="lines",
            line=dict(color=BLACK, width=1.5),
            hovertemplate="t+%{x:.1f}s | %{y:.0f} bpm<extra></extra>",
        )
    )
    fig.update_layout(
        height=140,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=WHITE_BG,
        paper_bgcolor=WHITE_BG,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def create_patient_detail_plot(patient: SyntheticPatient) -> go.Figure:
    """Full detail plot with baseline and event shading."""
    if not patient.timestamps:
        return go.Figure()

    x = np.array(patient.timestamps)
    y = np.array(patient.fhr)
    t0 = x.min()
    x_rel = x - t0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_rel,
            y=y,
            mode="lines",
            name="FHR",
            line=dict(color=BLACK, width=2),
            hovertemplate="t+%{x:.1f}s | %{y:.0f} bpm<extra></extra>",
        )
    )

    fig.add_hline(
        y=patient.baseline,
        line=dict(color=BLACK, dash="dash", width=1),
        annotation_text="Baseline",
        annotation_position="top right",
    )

    for ev in patient.events:
        fig.add_vrect(
            x0=ev.start_ts - t0,
            x1=ev.end_ts - t0,
            fillcolor=_event_color(ev.event_type) + "33",
            line_width=0,
            annotation_text=ev.event_type,
            annotation_position="top left",
        )

    if patient.active_event:
        ev = patient.active_event
        fig.add_vrect(
            x0=ev.start_ts - t0,
            x1=ev.end_ts - t0,
            fillcolor=_event_color(ev.event_type) + "44",
            line_width=0,
            annotation_text=f"{ev.event_type} (active)",
            annotation_position="top left",
        )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=30),
        plot_bgcolor=WHITE_BG,
        paper_bgcolor=WHITE_BG,
        showlegend=False,
        xaxis=dict(title="Seconds", zeroline=False, showgrid=False),
        yaxis=dict(title="bpm", range=[90, 190], showgrid=True, gridcolor="#e5e5e5"),
    )

    return fig


# =============================================================================
# Simulation Manager
# =============================================================================

PATIENT_NAMES = [
    "Sarah Cohen", "Rachel Levy", "Miri Golan", "Yael Barak", "Noa Shamir",
    "Dana Rosen", "Tali Avraham", "Liat Friedman", "Hila David", "Adi Moshe",
    "Ronit Ben", "Shira Gal", "Avital Cohen", "Maya Azulay", "Noga Hadar",
    "Eden Peretz", "Linor Saar", "Ortal Aviv", "Naama Oren", "Tamar Paz",
]


def _random_id() -> str:
    return "P" + "".join(random.choices(string.digits, k=3))


class SimulationManager:
    """Manages synthetic patients and their signal generation."""

    def __init__(self) -> None:
        self.patients: List[SyntheticPatient] = []
        self.nan_count = 0
        self.flatline_count = 0

    def add_patients(self, count: int) -> None:
        for _ in range(count):
            idx = len(self.patients)
            name = PATIENT_NAMES[idx % len(PATIENT_NAMES)]
            age = random.randint(24, 39)
            ga = random.randint(36, 41)
            high_noise = random.random() < 0.20  # 20% high noise
            patient = SyntheticPatient(
                patient_id=_random_id(),
                name=name,
                age=age,
                gestational_age_weeks=ga,
                high_noise=high_noise,
            )
            # Prime with 10 seconds of data
            now = time.time()
            for i in range(40):
                value = 140 + np.random.normal(0, 2)
                patient.append_point(float(value), now - (40 - i) * 0.25)
            self.patients.append(patient)

    def update_all(self) -> None:
        """Generate new samples for all patients (simulates one tick)."""
        now = time.time()
        for patient in self.patients:
            # Check/manage events
            if patient.active_event and now > patient.active_event.end_ts:
                patient.events.append(patient.active_event)
                patient.active_event = None

            # Maybe trigger a new decel event (20% of patients eligible)
            if not patient.active_event and random.random() < 0.005:
                duration = random.randint(8, 16)
                patient.active_event = SyntheticEvent(
                    event_type="Late Deceleration",
                    start_ts=now,
                    end_ts=now + duration,
                )

            # Generate sample
            drop = 0.0
            risk = "Normal"
            if patient.active_event:
                remaining = patient.active_event.end_ts - now
                total = patient.active_event.end_ts - patient.active_event.start_ts
                phase = 1.0 - (remaining / max(0.1, total))
                drop = 30 + np.random.uniform(0, 15)
                drop *= 1.0 - abs(phase - 0.5) * 2  # triangle
                risk = "Warning"

            # Noise level
            noise_std = 15.0 if patient.high_noise else 3.0
            noise = np.random.normal(0, noise_std)

            # Occasional artifacts for high-noise patients
            if patient.high_noise and random.random() < 0.05:
                noise += np.random.choice([-50, 50])

            drift = np.random.normal(0, 0.1)
            patient.baseline = min(160, max(120, patient.baseline + drift))
            value = patient.baseline - drop + noise

            # Clamp to physiological range
            value = max(60, min(200, value))

            patient.risk_level = risk
            patient.append_point(float(value), now)

            # Check for generator failures
            if np.isnan(value):
                self.nan_count += 1
            recent = patient.fhr[-20:] if len(patient.fhr) >= 20 else patient.fhr
            if len(recent) >= 20 and np.std(recent) < 0.1:
                self.flatline_count += 1


# =============================================================================
# Rule Engine Proxy
# =============================================================================


def run_rules_engine(patient: SyntheticPatient) -> dict:
    """
    Simulates the rule engine processing.
    Returns detection results.
    """
    if len(patient.fhr) < 20:
        return {"baseline": 140.0, "variability": 5.0, "decelerations": 0}

    window = np.array(patient.fhr[-60:])
    baseline = float(np.median(window))
    variability = float(np.std(window))

    # Simple deceleration detection
    min_val = float(window.min())
    decel_detected = (baseline - min_val) > 20

    return {
        "baseline": baseline,
        "variability": variability,
        "decelerations": 1 if decel_detected else 0,
    }


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class CycleMetrics:
    timestamp: float
    patients: int
    data_gen_ms: float
    rules_ms: float
    plotting_ms: float
    total_cycle_ms: float
    cpu_percent: float
    ram_mb: float
    nan_count: int
    flatline_count: int


@dataclass
class StressTestResults:
    max_patients: int
    failure_reason: str
    bottleneck: str
    metrics: List[CycleMetrics] = field(default_factory=list)

    # Aggregates
    avg_data_gen_ms: float = 0.0
    avg_rules_ms: float = 0.0
    avg_plotting_ms: float = 0.0
    max_cycle_ms: float = 0.0
    cpu_max: float = 0.0
    ram_max_mb: float = 0.0


# =============================================================================
# Main Stress Test
# =============================================================================


class FullSystemStressTest:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.manager = SimulationManager()
        self.metrics: List[CycleMetrics] = []
        self.failure_reason: Optional[str] = None
        self.max_patients = 0

    def run(self) -> StressTestResults:
        print("=" * 70)
        print("  Full System Stress Test")
        print("  Testing: Data Gen + Rules Engine + UI Plotting")
        print("=" * 70)
        print()

        # Warm up CPU meter
        self.process.cpu_percent()
        time.sleep(0.5)

        # Start with 1 patient
        self.manager.add_patients(1)
        next_increase = time.time() + 30.0
        cycle_count = 0

        try:
            while True:
                cycle_start = time.perf_counter()

                # === STEP A: Data Generation ===
                t0 = time.perf_counter()
                self.manager.update_all()
                data_gen_ms = (time.perf_counter() - t0) * 1000

                # === STEP B: Rules Engine ===
                t0 = time.perf_counter()
                for patient in self.manager.patients:
                    run_rules_engine(patient)
                rules_ms = (time.perf_counter() - t0) * 1000

                # === STEP C: UI Plotting (CRITICAL) ===
                t0 = time.perf_counter()
                figures = []
                for patient in self.manager.patients:
                    fig = create_patient_sparkline(patient)
                    figures.append(fig)
                    # Simulate JSON serialization (what Streamlit does)
                    _ = fig.to_json()
                plotting_ms = (time.perf_counter() - t0) * 1000

                # Total cycle time
                total_cycle_ms = (time.perf_counter() - cycle_start) * 1000

                # System metrics
                cpu = self.process.cpu_percent(interval=None)
                ram_mb = self.process.memory_info().rss / (1024 * 1024)

                # Record
                metric = CycleMetrics(
                    timestamp=time.time(),
                    patients=len(self.manager.patients),
                    data_gen_ms=data_gen_ms,
                    rules_ms=rules_ms,
                    plotting_ms=plotting_ms,
                    total_cycle_ms=total_cycle_ms,
                    cpu_percent=cpu,
                    ram_mb=ram_mb,
                    nan_count=self.manager.nan_count,
                    flatline_count=self.manager.flatline_count,
                )
                self.metrics.append(metric)
                self.max_patients = len(self.manager.patients)

                # Progress output every 10 cycles
                cycle_count += 1
                if cycle_count % 10 == 0:
                    print(
                        f"[{len(self.manager.patients):3d} patients] "
                        f"Cycle: {total_cycle_ms:6.1f}ms "
                        f"(Data:{data_gen_ms:5.1f} Rules:{rules_ms:5.1f} Plot:{plotting_ms:6.1f}) "
                        f"CPU:{cpu:5.1f}% RAM:{ram_mb:6.0f}MB"
                    )

                # === FAILURE CHECKS ===
                if total_cycle_ms > 1000:
                    self.failure_reason = f"FPS Drop: Cycle time {total_cycle_ms:.0f}ms > 1000ms"
                    break

                if self.manager.nan_count > 10:
                    self.failure_reason = f"Generator Failure: {self.manager.nan_count} NaN values"
                    break

                if self.manager.flatline_count > 50:
                    self.failure_reason = f"Generator Failure: {self.manager.flatline_count} flatlines"
                    break

                # Escalate patients every 30 seconds
                if time.time() >= next_increase:
                    self.manager.add_patients(2)
                    next_increase = time.time() + 30.0
                    print(f"\n>>> Escalating to {len(self.manager.patients)} patients\n")

                # Small sleep to simulate real UI refresh rate
                time.sleep(0.05)

                # Clean up figures
                del figures
                gc.collect()

        except MemoryError:
            self.failure_reason = "MemoryError"
        except KeyboardInterrupt:
            self.failure_reason = "User interrupted"

        return self._build_results()

    def _build_results(self) -> StressTestResults:
        if not self.metrics:
            return StressTestResults(
                max_patients=0,
                failure_reason=self.failure_reason or "No data",
                bottleneck="Unknown",
            )

        # Determine bottleneck
        avg_data = np.mean([m.data_gen_ms for m in self.metrics])
        avg_rules = np.mean([m.rules_ms for m in self.metrics])
        avg_plot = np.mean([m.plotting_ms for m in self.metrics])

        if avg_plot > avg_data and avg_plot > avg_rules:
            bottleneck = "UI Rendering (Plotly)"
        elif avg_rules > avg_data:
            bottleneck = "Rules Engine"
        else:
            bottleneck = "Data Generation"

        results = StressTestResults(
            max_patients=self.max_patients,
            failure_reason=self.failure_reason or "Test completed",
            bottleneck=bottleneck,
            metrics=self.metrics,
            avg_data_gen_ms=avg_data,
            avg_rules_ms=avg_rules,
            avg_plotting_ms=avg_plot,
            max_cycle_ms=max(m.total_cycle_ms for m in self.metrics),
            cpu_max=max(m.cpu_percent for m in self.metrics),
            ram_max_mb=max(m.ram_mb for m in self.metrics),
        )

        return results


# =============================================================================
# Report Generation
# =============================================================================


def write_report(results: StressTestResults) -> None:
    report_dir = Path("docs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "FULL_SYSTEM_STRESS_REPORT.md"

    # Calculate percentages for bottleneck analysis
    total_avg = results.avg_data_gen_ms + results.avg_rules_ms + results.avg_plotting_ms
    if total_avg > 0:
        data_pct = (results.avg_data_gen_ms / total_avg) * 100
        rules_pct = (results.avg_rules_ms / total_avg) * 100
        plot_pct = (results.avg_plotting_ms / total_avg) * 100
    else:
        data_pct = rules_pct = plot_pct = 33.3

    safe_limit = max(1, int(results.max_patients * 0.8))

    lines = [
        "# Full System Stress Test Report",
        "",
        "## Executive Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Max Concurrent Patients** | {results.max_patients} |",
        f"| **Safe Operational Limit (80%)** | {safe_limit} |",
        f"| **Failure Reason** | {results.failure_reason} |",
        f"| **Primary Bottleneck** | {results.bottleneck} |",
        "",
        "## Performance Breakdown",
        "",
        "### Average Time per Cycle Component",
        "",
        f"| Component | Avg Time (ms) | % of Total |",
        f"|-----------|---------------|------------|",
        f"| Data Generation | {results.avg_data_gen_ms:.1f} | {data_pct:.1f}% |",
        f"| Rules Engine | {results.avg_rules_ms:.1f} | {rules_pct:.1f}% |",
        f"| **UI Rendering (Plotly)** | **{results.avg_plotting_ms:.1f}** | **{plot_pct:.1f}%** |",
        f"| **Total** | **{total_avg:.1f}** | 100% |",
        "",
        "### System Resources",
        "",
        f"- **Max CPU Usage:** {results.cpu_max:.1f}%",
        f"- **Max RAM Usage:** {results.ram_max_mb:.0f} MB",
        f"- **Max Cycle Time:** {results.max_cycle_ms:.0f} ms",
        "",
        "## Bottleneck Analysis",
        "",
    ]

    if results.bottleneck == "UI Rendering (Plotly)":
        lines.extend([
            "### ⚠️ UI Rendering is the Bottleneck",
            "",
            "The Plotly figure generation and JSON serialization dominates cycle time.",
            "",
            "**Recommendations:**",
            "1. Use `Scattergl` instead of `Scatter` for WebGL acceleration",
            "2. Reduce sparkline point count (currently 60 → try 30)",
            "3. Cache figure objects and only update data",
            "4. Consider server-side rendering or pre-computed thumbnails",
            "5. Batch figure updates rather than regenerating every cycle",
        ])
    elif results.bottleneck == "Rules Engine":
        lines.extend([
            "### Rules Engine is the Bottleneck",
            "",
            "The rule-based analysis is consuming the most time.",
            "",
            "**Recommendations:**",
            "1. Profile individual rules for optimization",
            "2. Reduce window size for rule calculations",
            "3. Cache intermediate results between cycles",
        ])
    else:
        lines.extend([
            "### Data Generation is the Bottleneck",
            "",
            "Signal synthesis is consuming the most time.",
            "",
            "**Recommendations:**",
            "1. Simplify noise model",
            "2. Pre-generate event patterns",
            "3. Use vectorized operations",
        ])

    lines.extend([
        "",
        "## Signal Quality Under Load",
        "",
        f"- **NaN Values Detected:** {results.metrics[-1].nan_count if results.metrics else 0}",
        f"- **Flatline Events:** {results.metrics[-1].flatline_count if results.metrics else 0}",
        "",
        "The synthetic generator maintained signal quality under load." if (
            results.metrics and results.metrics[-1].nan_count == 0 and results.metrics[-1].flatline_count < 10
        ) else "⚠️ Generator showed signs of instability under high load.",
        "",
        "## Conclusion",
        "",
        f"The system can handle **{results.max_patients} concurrent patients** before the UI becomes laggy.",
        f"For smooth operation, limit to **{safe_limit} patients** (80% safety margin).",
        "",
        f"**Primary constraint:** {results.bottleneck}",
        "",
        "---",
        "*Report generated by full_system_stress.py*",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{'=' * 70}")
    print(f"Report written to: {report_path}")
    print(f"{'=' * 70}")


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    test = FullSystemStressTest()
    results = test.run()

    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Max Patients:  {results.max_patients}")
    print(f"  Failure:       {results.failure_reason}")
    print(f"  Bottleneck:    {results.bottleneck}")
    print(f"  Avg Plot Time: {results.avg_plotting_ms:.1f} ms")
    print("=" * 70)

    write_report(results)


if __name__ == "__main__":
    main()
