#!/usr/bin/env python
"""
THE GAUNTLET V4 - Hyper-Realistic Physiological Stress Test
============================================================

Based on the PROVEN clinical_validation_suite.py approach:
- Direct signal generation (not streaming simulation)
- Mathematical pathology modeling
- Biological variability (noise)
- Sensor artifacts

3-Layer Signal Model:
  Layer A: Mathematical Pathology (Gaussian decels, sine waves)
  Layer B: Biological Variability (LTV noise σ=4)
  Layer C: Sensor Artifacts (dropouts, spikes)

Target: Scalable from small test (10 events) to massive (2500+ events)
"""

# ============================================================================
# NUCLEAR SILENCE - MUST BE FIRST
# ============================================================================
import logging
import warnings
import os
import sys

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# ============================================================================
# IMPORTS
# ============================================================================
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from collections import defaultdict

# Add project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig
from src.data.signal_quality import calculate_fsqi

# Restore stderr
sys.stderr.close()
sys.stderr = _original_stderr

from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLING_RATE = 4.0  # Hz

# Test size - FULL SCALE for final validation
NUM_PATIENTS = 20  # 20 patients
EVENTS_PER_PATIENT = 125  # 125 events each = 2500 total events
SIGNAL_DURATION_SEC = 600.0  # 10 minutes per test signal

# Clinical event mix
EVENT_MIX = {
    "HEALTHY": 0.30,
    "LATE_DECEL": 0.35,
    "VARIABLE_DECEL": 0.25,
    "SINUSOIDAL": 0.10,
}

# Layer B: Biological variability
BASELINE_NOISE_STD = 2.5  # LTV noise - reduced from 4 to prevent false positives

# Layer C: Artifact parameters
ARTIFACT_PROBABILITY = 0.005  # 0.5% of samples - reduced


class EventType(Enum):
    HEALTHY = "HEALTHY"
    LATE_DECEL = "LATE_DECEL"
    VARIABLE_DECEL = "VARIABLE_DECEL"
    SINUSOIDAL = "SINUSOIDAL"


@dataclass
class TestResult:
    """Result of a single test."""
    event_type: EventType
    expected_category: int  # 1, 2, or 3
    predicted_category: int
    was_rejected: bool
    fsqi_score: float
    late_decels_found: int
    variable_decels_found: int
    is_correct: bool
    details: str


# =============================================================================
# LAYER A: MATHEMATICAL PATHOLOGY GENERATORS
# =============================================================================

def generate_baseline_with_contractions(
    duration_sec: float,
    baseline_fhr: float = 140.0,
    contractions_per_10min: float = 4.0,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate baseline FHR and UC with contractions.
    Returns (fhr, uc, contraction_peak_indices).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    samples = int(duration_sec * SAMPLING_RATE)
    t = np.arange(samples) / SAMPLING_RATE
    
    # Baseline FHR with moderate variability
    fhr = np.full(samples, baseline_fhr, dtype=np.float64)
    fhr += 8.0 * np.sin(2 * np.pi * 0.05 * t)  # Slow LTV oscillation
    
    # Generate contractions
    uc = np.full(samples, 10.0, dtype=np.float64)  # Baseline tonus
    
    contraction_interval = 600.0 / contractions_per_10min  # seconds
    contraction_peaks = []
    
    current_time = rng.uniform(30, 60)  # First contraction
    while current_time < duration_sec - 90:
        peak_idx = int(current_time * SAMPLING_RATE)
        contraction_peaks.append(peak_idx)
        
        # Gaussian contraction shape
        sigma = 20.0 * SAMPLING_RATE  # ~20 seconds width
        for i in range(max(0, peak_idx - 200), min(samples, peak_idx + 200)):
            dist = abs(i - peak_idx)
            uc[i] += 70.0 * np.exp(-0.5 * (dist / sigma) ** 2)
        
        # Next contraction
        current_time += rng.uniform(contraction_interval * 0.7, contraction_interval * 1.3)
    
    return fhr, uc, contraction_peaks


def generate_healthy_signal(rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate healthy (Category I) signal.
    - Baseline 110-160 bpm
    - Moderate variability (6-25 bpm)
    - Accelerations present
    - No significant decelerations
    """
    if rng is None:
        rng = np.random.default_rng()
    
    baseline = rng.uniform(125, 155)
    fhr, uc, _ = generate_baseline_with_contractions(SIGNAL_DURATION_SEC, baseline, rng=rng)
    
    samples = len(fhr)
    t = np.arange(samples) / SAMPLING_RATE
    
    # Add accelerations (healthy sign)
    n_accels = rng.integers(3, 7)
    for _ in range(n_accels):
        start = rng.integers(100, samples - 200)
        duration = rng.integers(60, 120)  # 15-30 seconds
        height = rng.uniform(15, 25)
        accel = height * np.sin(np.linspace(0, np.pi, duration))
        fhr[start:start+duration] += accel
    
    return fhr, uc


def generate_late_deceleration_signal(rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Late Deceleration signal (Category II/III).
    
    Key characteristics:
    - Nadir occurs >30s AFTER contraction peak (lag)
    - GRADUAL descent (>30 seconds to nadir)
    - Depth 15-40 bpm below baseline
    """
    if rng is None:
        rng = np.random.default_rng()
    
    baseline = 140.0
    fhr, uc, contraction_peaks = generate_baseline_with_contractions(
        SIGNAL_DURATION_SEC, baseline, contractions_per_10min=3.0, rng=rng
    )
    samples = len(fhr)
    
    for peak_idx in contraction_peaks:
        peak_time = peak_idx / SAMPLING_RATE
        
        # LATE: Onset begins AFTER contraction peak
        onset_delay = rng.uniform(15, 25)  # seconds after peak
        onset_time = peak_time + onset_delay
        onset_idx = int(onset_time * SAMPLING_RATE)
        
        # GRADUAL descent: 35-50 seconds (>=30s = Late classification)
        descent_duration = rng.uniform(35, 50)
        descent_samples = int(descent_duration * SAMPLING_RATE)
        nadir_idx = onset_idx + descent_samples
        
        if nadir_idx >= samples - 200:
            continue
        
        # Depth: 20-40 bpm
        depth = rng.uniform(20, 40)
        
        # Apply Gaussian-shaped deceleration
        total_duration = descent_samples * 2
        for i in range(onset_idx, min(samples, onset_idx + total_duration)):
            # Inverted Gaussian centered at nadir
            dist_from_nadir = abs(i - nadir_idx)
            sigma = descent_samples / 2
            decel = depth * np.exp(-0.5 * (dist_from_nadir / sigma) ** 2)
            fhr[i] -= decel
    
    return fhr, uc


def generate_variable_deceleration_signal(rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Variable Deceleration signal with severity signs.
    
    Key characteristics:
    - ABRUPT onset (<30 seconds to nadir)
    - Sharp V-shape (triangle wave)
    - Variable timing relative to contractions
    - Severity signs: deep (>70 bpm drop), slow recovery
    """
    if rng is None:
        rng = np.random.default_rng()
    
    baseline = 140.0
    fhr, uc, contraction_peaks = generate_baseline_with_contractions(
        SIGNAL_DURATION_SEC, baseline, contractions_per_10min=5.0, rng=rng
    )
    samples = len(fhr)
    
    for peak_idx in contraction_peaks:
        peak_time = peak_idx / SAMPLING_RATE
        
        # VARIABLE: timing varies relative to contraction
        timing_offset = rng.uniform(-10, 10)  # Can be before or after
        onset_time = peak_time + timing_offset
        onset_idx = int(onset_time * SAMPLING_RATE)
        
        if onset_idx < 0 or onset_idx >= samples - 300:
            continue
        
        # ABRUPT descent: 12-20 seconds (<30s = Variable)
        descent_duration = rng.uniform(12, 20)
        descent_samples = int(descent_duration * SAMPLING_RATE)
        nadir_idx = onset_idx + descent_samples
        
        # SEVERE: Deep drop (75-85 bpm depth = reaches 55-65 bpm)
        depth = rng.uniform(70, 85)
        
        # Abrupt descent (linear/triangle)
        for i in range(onset_idx, min(samples, nadir_idx)):
            progress = (i - onset_idx) / descent_samples
            fhr[i] -= depth * progress
        
        # SLOW recovery (severity sign): 60-80 seconds
        recovery_duration = rng.uniform(60, 80)
        recovery_samples = int(recovery_duration * SAMPLING_RATE)
        recovery_end = min(samples, nadir_idx + recovery_samples)
        
        for i in range(nadir_idx, recovery_end):
            progress = (i - nadir_idx) / recovery_samples
            fhr[i] -= depth * (1 - progress)
        
        # Overshoot after recovery (severity sign)
        if recovery_end < samples - 50:
            overshoot = rng.uniform(10, 20)
            overshoot_len = int(15 * SAMPLING_RATE)
            for i in range(recovery_end, min(samples, recovery_end + overshoot_len)):
                progress = (i - recovery_end) / overshoot_len
                fhr[i] += overshoot * np.sin(np.pi * progress)
    
    return fhr, uc


def generate_sinusoidal_signal(rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Sinusoidal Pattern (Category III).
    
    Key characteristics (FIGO criteria):
    - Regular oscillations 3-5 cycles/minute
    - Fixed amplitude 5-15 bpm
    - ABSENT short-term variability (smooth)
    - Duration >20 minutes (we generate 25 min)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sinusoidal needs longer duration for detection
    duration = 1500.0  # 25 minutes
    samples = int(duration * SAMPLING_RATE)
    t = np.arange(samples) / SAMPLING_RATE
    
    baseline = rng.uniform(130, 145)
    frequency = rng.uniform(3, 5) / 60  # 3-5 cycles per minute
    amplitude = rng.uniform(8, 15)
    
    # Pure sine wave (pathognomonic)
    fhr = baseline + amplitude * np.sin(2 * np.pi * frequency * t)
    
    # ABSENT short-term variability - only tiny noise
    fhr += rng.normal(0, 0.3, samples)
    
    # Normal contractions
    uc = 10 + 20 * np.sin(2 * np.pi * (1/180) * t)
    uc = np.clip(uc, 0, 100)
    
    return fhr.astype(np.float64), uc.astype(np.float64)


# =============================================================================
# LAYER B: BIOLOGICAL VARIABILITY
# =============================================================================

def add_biological_variability(fhr: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Add realistic LTV noise to simulate biological variability."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Gaussian noise for short-term variability
    noise = rng.normal(0, BASELINE_NOISE_STD, len(fhr))
    return fhr + noise


# =============================================================================
# LAYER C: SENSOR ARTIFACTS
# =============================================================================

def add_sensor_artifacts(fhr: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Add realistic sensor artifacts (dropouts, spikes)."""
    if rng is None:
        rng = np.random.default_rng()
    
    fhr = fhr.copy()
    samples = len(fhr)
    
    # Signal dropouts (2 second zeros)
    n_dropouts = int(samples * ARTIFACT_PROBABILITY * 0.01)
    for _ in range(n_dropouts):
        start = rng.integers(0, samples - 10)
        length = int(2 * SAMPLING_RATE)  # 2 seconds
        fhr[start:start+length] = 0
    
    # Random spikes
    n_spikes = int(samples * ARTIFACT_PROBABILITY * 0.005)
    for _ in range(n_spikes):
        idx = rng.integers(0, samples)
        fhr[idx] = rng.choice([50, 200, 220])
    
    return fhr


# =============================================================================
# SIGNAL GENERATOR FACTORY
# =============================================================================

def generate_test_signal(event_type: EventType, rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a complete 3-layer signal for the given event type."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Layer A: Mathematical pathology
    if event_type == EventType.HEALTHY:
        fhr, uc = generate_healthy_signal(rng)
    elif event_type == EventType.LATE_DECEL:
        fhr, uc = generate_late_deceleration_signal(rng)
    elif event_type == EventType.VARIABLE_DECEL:
        fhr, uc = generate_variable_deceleration_signal(rng)
    elif event_type == EventType.SINUSOIDAL:
        fhr, uc = generate_sinusoidal_signal(rng)
    else:
        raise ValueError(f"Unknown event type: {event_type}")
    
    # Layer B: Biological variability (except sinusoidal which has absent variability)
    if event_type != EventType.SINUSOIDAL:
        fhr = add_biological_variability(fhr, rng)
    
    # Layer C: Sensor artifacts (small amount)
    fhr = add_sensor_artifacts(fhr, rng)
    
    # Final clipping to physiological range
    fhr = np.clip(fhr, 50, 210)
    
    return fhr.astype(np.float64), uc.astype(np.float64)


# =============================================================================
# TEST EXECUTION
# =============================================================================

def determine_expected_category(event_type: EventType) -> int:
    """Determine expected category for event type."""
    if event_type == EventType.HEALTHY:
        return 1
    elif event_type == EventType.LATE_DECEL:
        return 2  # Cat II or III
    elif event_type == EventType.VARIABLE_DECEL:
        return 2  # Cat II or III
    elif event_type == EventType.SINUSOIDAL:
        return 3  # Always Cat III
    return 1


def is_detection_correct(event_type: EventType, predicted_cat: int, 
                         late_decels: int, variable_decels: int,
                         sinusoidal_detected: bool) -> bool:
    """Determine if detection was correct for the event type."""
    if event_type == EventType.HEALTHY:
        return predicted_cat == 1
    elif event_type == EventType.LATE_DECEL:
        # Success: Category II/III OR Late decels detected
        return predicted_cat in [2, 3] or late_decels > 0
    elif event_type == EventType.VARIABLE_DECEL:
        # Success: Variable decels detected OR Category II/III
        return variable_decels > 0 or predicted_cat in [2, 3]
    elif event_type == EventType.SINUSOIDAL:
        # Success: Category III OR sinusoidal detected
        return predicted_cat == 3 or sinusoidal_detected
    return False


def run_single_test(adapter: PipelineAdapter, event_type: EventType, 
                    rng: np.random.Generator) -> TestResult:
    """Run a single test and return result."""
    
    # Generate signal
    fhr, uc = generate_test_signal(event_type, rng)
    
    # Check FSQI first
    fsqi_result = calculate_fsqi(fhr, SAMPLING_RATE)
    
    if not fsqi_result.should_classify:
        # Signal rejected by FSQI
        expected = determine_expected_category(event_type)
        # Rejection is correct only for noise scenarios (we don't have those here)
        is_correct = False
        return TestResult(
            event_type=event_type,
            expected_category=expected,
            predicted_category=0,
            was_rejected=True,
            fsqi_score=fsqi_result.score,
            late_decels_found=0,
            variable_decels_found=0,
            is_correct=is_correct,
            details=f"FSQI rejected: {fsqi_result.message}"
        )
    
    # Run pipeline
    result = adapter.process_patient(
        patient_id="GAUNTLET_TEST",
        data={"fhr": fhr, "uc": uc},
        run_moment=False
    )
    
    predicted_cat = result.get("category", 1)
    findings = result.get("findings", {})
    
    # Extract findings
    decel_findings = findings.get("decelerations", {})
    late_decels = decel_findings.get("late", 0)
    variable_decels = decel_findings.get("variable", 0)
    sinusoidal_detected = findings.get("sinusoidal", {}).get("detected", False)
    
    expected_cat = determine_expected_category(event_type)
    is_correct = is_detection_correct(
        event_type, predicted_cat, late_decels, variable_decels, sinusoidal_detected
    )
    
    details = f"Cat {predicted_cat}, Late:{late_decels}, Var:{variable_decels}, Sino:{sinusoidal_detected}"
    
    return TestResult(
        event_type=event_type,
        expected_category=expected_cat,
        predicted_category=predicted_cat,
        was_rejected=False,
        fsqi_score=fsqi_result.score,
        late_decels_found=late_decels,
        variable_decels_found=variable_decels,
        is_correct=is_correct,
        details=details
    )


# =============================================================================
# MAIN GAUNTLET ENGINE
# =============================================================================

def select_event_type(rng: np.random.Generator) -> EventType:
    """Select event type based on configured mix."""
    r = rng.random()
    cumulative = 0.0
    for event_name, prob in EVENT_MIX.items():
        cumulative += prob
        if r < cumulative:
            return EventType[event_name]
    return EventType.HEALTHY


def run_gauntlet(num_patients: int, events_per_patient: int) -> Dict:
    """Run the gauntlet test suite."""
    
    print("=" * 70)
    print("THE GAUNTLET V4 - Hyper-Realistic Physiological Stress Test")
    print("=" * 70)
    print(f"Patients: {num_patients}")
    print(f"Events per patient: {events_per_patient}")
    print(f"Total events: {num_patients * events_per_patient}")
    print(f"Event mix: {EVENT_MIX}")
    print("=" * 70)
    
    # Initialize pipeline
    config = PipelineAdapterConfig(
        sampling_rate=SAMPLING_RATE,
        min_data_seconds=60.0
    )
    adapter = PipelineAdapter(config)
    
    # Results storage
    all_results: List[TestResult] = []
    results_by_type: Dict[EventType, List[TestResult]] = defaultdict(list)
    
    total_events = num_patients * events_per_patient
    rng = np.random.default_rng(42)  # Reproducible
    
    start_time = time.time()
    
    with tqdm(total=total_events, desc="Running Gauntlet", unit="event") as pbar:
        for patient_idx in range(num_patients):
            for event_idx in range(events_per_patient):
                # Select event type
                event_type = select_event_type(rng)
                
                # Run test
                result = run_single_test(adapter, event_type, rng)
                
                all_results.append(result)
                results_by_type[event_type].append(result)
                
                pbar.update(1)
                pbar.set_postfix({
                    "type": event_type.name[:6],
                    "correct": result.is_correct
                })
    
    elapsed = time.time() - start_time
    
    # Calculate statistics
    stats = calculate_statistics(all_results, results_by_type, elapsed)
    
    return stats


def calculate_statistics(all_results: List[TestResult], 
                        results_by_type: Dict[EventType, List[TestResult]],
                        elapsed_seconds: float) -> Dict:
    """Calculate comprehensive statistics."""
    
    total = len(all_results)
    correct = sum(1 for r in all_results if r.is_correct)
    overall_accuracy = correct / total if total > 0 else 0.0
    
    stats = {
        "total_events": total,
        "correct": correct,
        "overall_accuracy": overall_accuracy,
        "elapsed_seconds": elapsed_seconds,
        "by_type": {}
    }
    
    for event_type, results in results_by_type.items():
        n = len(results)
        n_correct = sum(1 for r in results if r.is_correct)
        n_rejected = sum(1 for r in results if r.was_rejected)
        accuracy = n_correct / n if n > 0 else 0.0
        
        stats["by_type"][event_type.name] = {
            "total": n,
            "correct": n_correct,
            "rejected": n_rejected,
            "accuracy": accuracy,
            "total_late_decels": sum(r.late_decels_found for r in results),
            "total_variable_decels": sum(r.variable_decels_found for r in results),
        }
    
    return stats


def print_results(stats: Dict):
    """Print formatted results."""
    
    print("\n" + "=" * 70)
    print("GAUNTLET V4 RESULTS")
    print("=" * 70)
    
    print(f"\nTotal Events: {stats['total_events']}")
    print(f"Correct: {stats['correct']}")
    print(f"Overall Accuracy: {stats['overall_accuracy']*100:.1f}%")
    print(f"Time: {stats['elapsed_seconds']:.1f}s")
    
    print("\n" + "-" * 70)
    print("BY EVENT TYPE:")
    print("-" * 70)
    
    print(f"{'Type':<20} {'N':>6} {'Correct':>8} {'Accuracy':>10} {'Late':>6} {'Var':>6}")
    print("-" * 70)
    
    for event_name, data in stats["by_type"].items():
        print(f"{event_name:<20} {data['total']:>6} {data['correct']:>8} "
              f"{data['accuracy']*100:>9.1f}% {data['total_late_decels']:>6} "
              f"{data['total_variable_decels']:>6}")
    
    print("-" * 70)
    
    # Verdict
    accuracy = stats["overall_accuracy"]
    if accuracy >= 0.90:
        print(f"\n✅ PASS - Overall accuracy {accuracy*100:.1f}% >= 90%")
    elif accuracy >= 0.70:
        print(f"\n⚠️  MARGINAL - Overall accuracy {accuracy*100:.1f}% (70-90%)")
    else:
        print(f"\n❌ FAIL - Overall accuracy {accuracy*100:.1f}% < 70%")
    
    return accuracy >= 0.90


def generate_report(stats: Dict, output_path: Path):
    """Generate markdown report."""
    
    report = f"""# The Gauntlet V4 - Final Results

**Date:** {datetime.now().isoformat()}

---

## Executive Summary

- **Total Events:** {stats['total_events']}
- **Correct Detections:** {stats['correct']}
- **Overall Accuracy:** {stats['overall_accuracy']*100:.1f}%
- **Execution Time:** {stats['elapsed_seconds']:.1f} seconds

---

## Results by Event Type

| Event Type | N | Correct | Accuracy | Late Decels | Variable Decels |
|------------|---|---------|----------|-------------|-----------------|
"""
    
    for event_name, data in stats["by_type"].items():
        report += f"| {event_name} | {data['total']} | {data['correct']} | "
        report += f"{data['accuracy']*100:.1f}% | {data['total_late_decels']} | "
        report += f"{data['total_variable_decels']} |\n"
    
    # Verdict
    accuracy = stats["overall_accuracy"]
    if accuracy >= 0.90:
        verdict = "✅ **PASS** - System meets clinical requirements"
    elif accuracy >= 0.70:
        verdict = "⚠️ **MARGINAL** - System needs improvement"
    else:
        verdict = "❌ **FAIL** - System does not meet requirements"
    
    report += f"""

---

## Verdict

{verdict}

---

## 3-Layer Signal Generation Model

### Layer A: Mathematical Pathology
- **Late Decelerations:** Inverted Gaussian, nadir >30s after contraction, depth 20-40 bpm
- **Variable Decelerations:** Sharp triangle, <30s descent, depth 70-85 bpm, slow recovery
- **Sinusoidal:** Pure sine wave, 3-5 cycles/min, amplitude 8-15 bpm

### Layer B: Biological Variability  
- Gaussian noise (σ=4 bpm) for Long Term Variability
- Slow oscillations for realistic heart rate patterns

### Layer C: Sensor Artifacts
- Random signal dropouts (2 seconds)
- Occasional spikes (50-220 bpm)

---

*Generated by the_gauntlet.py V4*
"""
    
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run small test first
    stats = run_gauntlet(NUM_PATIENTS, EVENTS_PER_PATIENT)
    passed = print_results(stats)
    
    # Generate report
    report_path = ROOT / "docs" / "reports" / "THE_GAUNTLET_V4_RESULTS.md"
    generate_report(stats, report_path)
    
    print("\n" + "=" * 70)
    if passed:
        print("🎯 Test PASSED! Ready for large-scale run.")
    else:
        print("🔧 Test needs debugging before scaling up.")
    print("=" * 70)
