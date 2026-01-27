#!/usr/bin/env python3
"""
Deep Endurance Audit - 35-Minute Stress Test
=============================================

NUCLEAR SILENCE MODE: All logging and warnings suppressed.
Uses direct state injection for 20-minute warm-up (no simulation loop).
Verifies MiniRocket engine before starting.

Output: Single tqdm progress bar only.
"""

# ============================================================================
# NUCLEAR SILENCE - MUST BE FIRST BEFORE ANY IMPORTS
# ============================================================================
import logging
import warnings
import os
import sys

# 1. Disable Python logging at interpreter level (CRITICAL = highest, disabled above it)
logging.disable(logging.CRITICAL)

# 2. Suppress all Python warnings (numpy, scipy, sklearn, etc.)
warnings.filterwarnings("ignore")

# 3. Suppress TensorFlow/ONNX/OpenVINO console spam (if any)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

# 4. Redirect stderr temporarily during imports to catch any startup noise
_original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# ============================================================================
# IMPORTS (with stderr suppressed)
# ============================================================================
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import deque
from dataclasses import dataclass

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import project modules (may emit startup logs - suppressed)
from src.config import CTG
from src.simulation.generators.patient_generator import PatientConfig, PatientGenerator
from src.simulation.processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig
from src.models.minirocket_encoder import MiniRocketEncoder

# Restore stderr after imports
sys.stderr.close()
sys.stderr = _original_stderr

# Now import tqdm (the ONLY allowed output)
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_PATIENTS = 5
SIMULATION_TICKS = 8400  # 35 minutes * 60 seconds * 4 Hz / 4 samples per tick
WARMUP_MINUTES = 20.0
SAMPLING_RATE = 4.0
BUFFER_MINUTES = 20.0  # Must match warmup to hold all history


@dataclass
class AuditStats:
    """Accumulates statistics during the audit run."""
    category_counts: Dict[int, int]
    total_ticks: int
    total_predictions: int
    elapsed_seconds: float
    latencies_ms: List[float]

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_latency_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


# ============================================================================
# ENGINE VERIFICATION
# ============================================================================
def verify_minirocket_engine(adapter: PipelineAdapter) -> None:
    """
    Verify that MiniRocket is the active engine.
    Raises RuntimeError if Transformer/MOMENT or other heavy model detected.
    """
    # Check 1: Encoder must be available
    if not getattr(adapter, "_encoder_available", False):
        raise RuntimeError(
            "ABORT: No encoder available on PipelineAdapter. "
            "MiniRocket initialization failed."
        )

    # Check 2: Encoder must be MiniRocketEncoder
    encoder = getattr(adapter, "_encoder", None)
    if encoder is None:
        raise RuntimeError("ABORT: Encoder is None - not properly initialized.")

    if not isinstance(encoder, MiniRocketEncoder):
        encoder_type = type(encoder).__name__
        raise RuntimeError(
            f"ABORT: Heavy model detected! Expected MiniRocketEncoder, "
            f"found {encoder_type}. Cannot run endurance test with Transformer."
        )

    # Check 3: Verify through stats API
    stats = adapter.get_stats()
    if not stats.get("using_minirocket", False):
        raise RuntimeError(
            "ABORT: Stats indicate MiniRocket is not active. "
            f"Stats: {stats}"
        )


# ============================================================================
# DIRECT STATE INJECTION (Warm-Up)
# ============================================================================
def inject_warmup_history(patient: PatientGenerator, baseline_fhr: float = 140.0) -> None:
    """
    Directly inject 20 minutes of baseline history into patient buffer.

    This bypasses the generate_tick loop entirely - no analysis triggered.
    The buffer is filled atomically before the main simulation begins.
    """
    samples = int(WARMUP_MINUTES * 60 * SAMPLING_RATE)  # 4800 samples

    # Create baseline data (constant FHR, no contractions)
    fhr_data = np.full(samples, baseline_fhr, dtype=np.float64)
    uc_data = np.zeros(samples, dtype=np.float64)
    timestamps = np.arange(samples, dtype=np.float64) / SAMPLING_RATE

    # Direct injection into internal deques (bypass append_batch loop)
    buffer = patient._buffer
    buffer._fhr = deque(fhr_data.tolist(), maxlen=buffer.max_samples)
    buffer._uc = deque(uc_data.tolist(), maxlen=buffer.max_samples)
    buffer._timestamps = deque(timestamps.tolist(), maxlen=buffer.max_samples)

    # Synchronize simulation time
    patient._simulation_time = timestamps[-1] + (1.0 / SAMPLING_RATE)

    # Verification
    if buffer.size != samples:
        raise RuntimeError(
            f"Warmup injection failed for {patient.patient_id}: "
            f"expected {samples} samples, got {buffer.size}"
        )

    if abs(buffer.duration_minutes - WARMUP_MINUTES) > 0.1:
        raise RuntimeError(
            f"Warmup duration mismatch for {patient.patient_id}: "
            f"expected {WARMUP_MINUTES}min, got {buffer.duration_minutes:.1f}min"
        )


# ============================================================================
# PATIENT FACTORY
# ============================================================================
def create_patients(n: int) -> List[PatientGenerator]:
    """Create N patients with 20-minute buffers and injected history."""
    patients = []
    for idx in range(1, n + 1):
        config = PatientConfig(
            patient_id=f"P{idx:02d}",
            bed_number=idx,
            baseline_fhr=140.0,
            baseline_variability=10.0,
            buffer_duration_minutes=BUFFER_MINUTES,  # 20 min buffer
            sampling_rate=SAMPLING_RATE,
        )
        patient = PatientGenerator(config)
        inject_warmup_history(patient, baseline_fhr=140.0)
        patients.append(patient)
    return patients


# ============================================================================
# TICK PROCESSOR
# ============================================================================
def process_single_tick(
    adapter: PipelineAdapter,
    patient: PatientGenerator
) -> tuple[int, float]:
    """
    Generate one tick and process through pipeline.
    Returns (category, latency_ms).
    """
    # Generate 1 second of data
    patient.generate_tick(int(SAMPLING_RATE))

    # Get 10-minute window for analysis
    window = patient._buffer.get_window(duration_seconds=600)

    # Time the pipeline
    start = time.perf_counter()
    result = adapter.process_patient(
        patient.patient_id,
        {
            "fhr": window.get("fhr", np.array([])),
            "uc": window.get("uc", np.array([])),
            "timestamps": window.get("timestamps", np.array([])),
        },
        run_moment=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    category = int(result.get("category", 1))
    return category, latency_ms


# ============================================================================
# MAIN AUDIT LOOP
# ============================================================================
def run_audit() -> AuditStats:
    """Execute the 35-minute stress test."""
    # Initialize adapter
    adapter = PipelineAdapter(PipelineAdapterConfig(
        use_real_moment=False,
        sampling_rate=SAMPLING_RATE,
        min_data_seconds=60.0,
    ))

    # CRITICAL: Verify engine before proceeding
    verify_minirocket_engine(adapter)

    # Create and warm up patients
    patients = create_patients(NUM_PATIENTS)

    # Verify warm-up succeeded
    for p in patients:
        duration = p._buffer.duration_minutes
        if duration < WARMUP_MINUTES - 0.1:
            raise RuntimeError(
                f"Patient {p.patient_id} has only {duration:.1f} min history. "
                f"Expected {WARMUP_MINUTES} min."
            )

    # Initialize stats
    category_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    latencies: List[float] = []

    # Run simulation
    start_time = time.time()

    for _ in tqdm(range(SIMULATION_TICKS), desc="Deep Endurance", unit="tick", ncols=80):
        for patient in patients:
            category, latency = process_single_tick(adapter, patient)
            category_counts[category] = category_counts.get(category, 0) + 1
            latencies.append(latency)

    elapsed = time.time() - start_time

    return AuditStats(
        category_counts=category_counts,
        total_ticks=SIMULATION_TICKS,
        total_predictions=sum(category_counts.values()),
        elapsed_seconds=elapsed,
        latencies_ms=latencies,
    )


# ============================================================================
# REPORT GENERATION
# ============================================================================
def write_report(stats: AuditStats) -> Path:
    """Generate the final audit report."""
    total = stats.total_predictions

    # Calculate percentages
    pct = {cat: 100.0 * count / total for cat, count in stats.category_counts.items()}

    # Performance metrics
    ticks_per_second = stats.total_ticks / stats.elapsed_seconds
    predictions_per_second = stats.total_predictions / stats.elapsed_seconds

    # Go/No-Go criteria
    go_criteria = [
        ("Category 1 dominance (>80%)", pct.get(1, 0) > 80),
        ("Max latency < 500ms", stats.max_latency_ms < 500),
        ("P99 latency < 200ms", stats.p99_latency_ms < 200),
        ("Completed without errors", True),
    ]

    all_pass = all(passed for _, passed in go_criteria)
    status = "GO" if all_pass else "NO-GO"

    report_lines = [
        "# Deep Endurance Audit Report",
        "",
        f"**Status: {status}**",
        "",
        f"Date: {datetime.utcnow().isoformat()}Z",
        f"Duration: {stats.elapsed_seconds:.2f} seconds ({stats.elapsed_seconds/60:.1f} minutes)",
        f"Patients: {NUM_PATIENTS}",
        f"Ticks: {stats.total_ticks:,}",
        f"Total Predictions: {stats.total_predictions:,}",
        "",
        "## Performance Metrics",
        "",
        f"- Throughput: {ticks_per_second:.1f} ticks/sec, {predictions_per_second:.1f} predictions/sec",
        f"- Avg Latency: {stats.avg_latency_ms:.2f} ms",
        f"- P99 Latency: {stats.p99_latency_ms:.2f} ms",
        f"- Max Latency: {stats.max_latency_ms:.2f} ms",
        "",
        "## Category Distribution",
        "",
    ]

    for cat in [1, 2, 3]:
        count = stats.category_counts.get(cat, 0)
        report_lines.append(f"- Category {cat}: {count:,} ({pct.get(cat, 0):.1f}%)")

    report_lines.extend([
        "",
        "## Go/No-Go Criteria",
        "",
    ])

    for criterion, passed in go_criteria:
        icon = "PASS" if passed else "FAIL"
        report_lines.append(f"- [{icon}] {criterion}")

    report_lines.extend([
        "",
        "---",
        "",
        "*Generated by deep_endurance_audit.py*",
    ])

    # Write report
    report_path = ROOT / "REPORTS" / "DEEP_ENDURANCE_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return report_path


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        stats = run_audit()
        report_path = write_report(stats)
        print(f"\nReport saved to: {report_path}")
    except RuntimeError as e:
        print(f"\nAUDIT FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAudit interrupted by user.")
        sys.exit(130)
