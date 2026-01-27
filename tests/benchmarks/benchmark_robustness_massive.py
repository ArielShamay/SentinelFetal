"""
Massive Robustness Campaign Benchmark

Generates 500+ randomized scenarios covering:
- 7 patterns: Sinusoidal, Late, Variable, Prolonged Decels, Brady, Tachy, Normal
- 3 severities: Mild, Moderate, Severe
- 5 noise levels: 0, 2, 5, 10, 15 bpm
- 3 dropout rates: 0%, 5%, 10%
- Randomized durations: 30-60 min (≥25 min for sinusoidal)

Runs batch-mode (no time.sleep). Outputs REPORTS/MASSIVE_ROBUSTNESS_REPORT.md.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CTG
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.pipeline.container import PipelineContainer
from src.simulation.generators.patient_generator import PatientConfig, PatientGenerator
from src.simulation.events.event_types import (
    EventType,
    EventSeverity,
    SinusoidalParams,
    LateDecelerationParams,
    VariableDecelerationParams,
    ProlongedDecelerationParams,
    BradycardiaParams,
    TachycardiaParams,
)
from src.signal_invariants import assert_signal_length, assert_pair_aligned
from src.utils.runtime_config import load_runtime_config

RUNTIME_CFG = load_runtime_config()
if float(CTG.SAMPLING_RATE) != float(RUNTIME_CFG.fs_hz):
    raise RuntimeError(
        f"Runtime fs_hz mismatch: runtime={RUNTIME_CFG.fs_hz} ctg={CTG.SAMPLING_RATE}"
    )
SAMPLING_RATE = RUNTIME_CFG.fs_hz
REPORT_PATH = PROJECT_ROOT / "REPORTS" / "MASSIVE_ROBUSTNESS_REPORT.md"

# Scenario config
NUM_SCENARIOS = 500
NOISE_LEVELS = [0.0, 2.0, 5.0, 10.0, 15.0]  # σ in bpm
DROPOUT_RATES = [0.0, 0.05, 0.10]
SEVERITIES = ["mild", "moderate", "severe"]


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
@dataclass
class PatternDef:
    name: str
    event_type: Optional[EventType]
    expected_cat: int  # expected minimum category to be "detected"
    min_duration_sec: int
    max_duration_sec: int


PATTERN_DEFS: List[PatternDef] = [
    PatternDef("Sinusoidal", EventType.SINUSOIDAL_PATTERN, 3, 25 * 60, 60 * 60),
    PatternDef("Late Decel", EventType.LATE_DECELERATION, 2, 30 * 60, 60 * 60),
    PatternDef("Variable Decel", EventType.VARIABLE_DECELERATION, 2, 30 * 60, 60 * 60),
    PatternDef("Prolonged Decel", EventType.PROLONGED_DECELERATION, 2, 30 * 60, 60 * 60),
    PatternDef("Bradycardia", EventType.BRADYCARDIA, 2, 30 * 60, 60 * 60),
    PatternDef("Tachycardia", EventType.TACHYCARDIA, 2, 30 * 60, 60 * 60),
    PatternDef("Normal", None, 1, 30 * 60, 60 * 60),  # Normal should stay Cat 1
]


def _get_event_params(event_type: EventType, severity: str, duration_sec: float):
    """Build event params for given type, severity, duration."""
    if event_type == EventType.SINUSOIDAL_PATTERN:
        return SinusoidalParams(
            duration_seconds=duration_sec,
            amplitude_bpm=10.0,
            frequency_cycles_per_min=4.0,
        )
    elif event_type == EventType.LATE_DECELERATION:
        base = getattr(LateDecelerationParams, severity)()
        base.duration_seconds = duration_sec
        return base
    elif event_type == EventType.VARIABLE_DECELERATION:
        base = getattr(VariableDecelerationParams, severity)()
        base.duration_seconds = duration_sec
        return base
    elif event_type == EventType.PROLONGED_DECELERATION:
        base = getattr(ProlongedDecelerationParams, severity if severity != "mild" else "moderate")()
        base.duration_seconds = duration_sec
        return base
    elif event_type == EventType.BRADYCARDIA:
        base = getattr(BradycardiaParams, severity)()
        base.duration_seconds = duration_sec
        return base
    elif event_type == EventType.TACHYCARDIA:
        base = getattr(TachycardiaParams, severity)()
        base.duration_seconds = duration_sec
        return base
    return None


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    id: int
    pattern: str
    severity: str
    duration_sec: float
    noise_sigma: float
    dropout_rate: float
    expected_cat: int


def generate_scenarios(n: int, rng: np.random.Generator) -> List[Scenario]:
    """Generate n randomized scenarios."""
    scenarios = []
    for i in range(n):
        pdef = rng.choice(PATTERN_DEFS)
        duration = rng.integers(pdef.min_duration_sec, pdef.max_duration_sec + 1)
        if duration < int(RUNTIME_CFG.min_case_minutes * 60):
            raise RuntimeError(
                f"STRICT_DURATION: duration {duration}s < min_case {int(RUNTIME_CFG.min_case_minutes * 60)}s"
            )
        severity = rng.choice(SEVERITIES)
        noise = rng.choice(NOISE_LEVELS)
        dropout = rng.choice(DROPOUT_RATES)
        scenarios.append(Scenario(
            id=i + 1,
            pattern=pdef.name,
            severity=severity,
            duration_sec=float(duration),
            noise_sigma=noise,
            dropout_rate=dropout,
            expected_cat=pdef.expected_cat,
        ))
    return scenarios


# ---------------------------------------------------------------------------
# Signal generation and corruption
# ---------------------------------------------------------------------------

def _generate_signal(pattern: str, severity: str, duration_sec: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate clean FHR/UC signal for pattern."""
    cfg = PatientConfig(
        patient_id="massive",
        bed_number=1,
        baseline_fhr=140.0,
        baseline_variability=10.0,
        buffer_duration_minutes=(duration_sec / 60) + 5,
    )
    patient = PatientGenerator(cfg)
    # warmup
    patient.generate_tick(int(60 * SAMPLING_RATE))

    # inject event if not Normal
    pdef = next((p for p in PATTERN_DEFS if p.name == pattern), None)
    if pdef and pdef.event_type is not None:
        params = _get_event_params(pdef.event_type, severity, duration_sec)
        if params:
            patient.inject_event(pdef.event_type, params)

    # generate duration
    patient.generate_tick(int(duration_sec * SAMPLING_RATE))
    window = patient._buffer.get_window(duration_seconds=duration_sec)
    return window["fhr"].copy(), window["uc"].copy()


def add_noise(fhr: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0.0:
        return fhr
    return fhr + rng.normal(0.0, sigma, fhr.shape)


def add_dropouts(fhr: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    if rate <= 0.0:
        return fhr
    mask = rng.random(fhr.shape) < rate
    out = fhr.copy()
    out[mask] = np.nan
    return out


def fill_nans(fhr: np.ndarray) -> np.ndarray:
    """Linear interpolation to fill NaNs."""
    nans = np.isnan(fhr)
    if not nans.any():
        return fhr
    indices = np.arange(len(fhr))
    fhr[nans] = np.interp(indices[nans], indices[~nans], fhr[~nans])
    return fhr


# ---------------------------------------------------------------------------
# Run scenarios
# ---------------------------------------------------------------------------
@dataclass
class ScenarioResult:
    scenario: Scenario
    predicted_cat: int
    detected: bool  # True if predicted >= expected (for pathological) or predicted == expected (for Normal)


def run_scenario(s: Scenario, pipeline: AnalysisPipeline, rng: np.random.Generator) -> ScenarioResult:
    fhr, uc = _generate_signal(s.pattern, s.severity, s.duration_sec, rng)
    fhr = add_noise(fhr, s.noise_sigma, rng)
    fhr = add_dropouts(fhr, s.dropout_rate, rng)
    fhr = fill_nans(fhr)

    # Phase 13: Use longer window for sinusoidal patterns (requires 20+ min)
    # Use 25-min window for Sinusoidal, runtime window for others (>= 20 min).
    if s.pattern == "Sinusoidal":
        window_minutes = max(RUNTIME_CFG.window_minutes, 25)
    else:
        window_minutes = RUNTIME_CFG.window_minutes
    
    window_samples = int(window_minutes * 60 * SAMPLING_RATE)
    if len(fhr) > window_samples:
        fhr_win = fhr[-window_samples:]
        uc_win = uc[-window_samples:]
    else:
        fhr_win, uc_win = fhr, uc

    assert_signal_length(fhr_win, SAMPLING_RATE, RUNTIME_CFG.min_window_minutes, "BENCH:WINDOW:FHR:MIN")
    assert_signal_length(uc_win, SAMPLING_RATE, RUNTIME_CFG.min_window_minutes, "BENCH:WINDOW:UC:MIN")
    assert_pair_aligned(fhr_win, uc_win, SAMPLING_RATE)

    result = pipeline.analyze(fhr_win, uc_win, sampling_rate=SAMPLING_RATE)
    predicted = result.category

    # Detection logic
    if s.pattern == "Normal":
        # Normal: detected = stays Cat 1 (no false positive)
        detected = predicted == 1
    else:
        # Pathological: detected = predicted >= expected
        detected = predicted >= s.expected_cat

    return ScenarioResult(scenario=s, predicted_cat=predicted, detected=detected)


def run_all(scenarios: List[Scenario]) -> List[ScenarioResult]:
    container = PipelineContainer.create_default(use_mock_moment=True)
    pipeline = AnalysisPipeline(container)
    rng = np.random.default_rng(seed=42)
    results = []
    for i, s in enumerate(scenarios):
        if (i + 1) % 50 == 0:
            print(f"[massive] Processed {i + 1}/{len(scenarios)} scenarios...")
        res = run_scenario(s, pipeline, rng)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _rate(results: List[ScenarioResult], **filters) -> float:
    subset = [r for r in results if all(getattr(r.scenario, k) == v for k, v in filters.items())]
    if not subset:
        return float("nan")
    return sum(1 for r in subset if r.detected) / len(subset)


def _count(results: List[ScenarioResult], **filters) -> int:
    return len([r for r in results if all(getattr(r.scenario, k) == v for k, v in filters.items())])


def generate_report(results: List[ScenarioResult]) -> str:
    lines = [
        "---",
        "title: Phase 12 – Massive Robustness Campaign",
        f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "description: 500+ randomized scenarios testing detection under noise, dropouts, and varied durations.",
        "---",
        "",
        "# Massive Robustness Test Report",
        "",
        f"**Total Scenarios**: {len(results)}",
        f"**Patterns**: {', '.join(p.name for p in PATTERN_DEFS)}",
        f"**Noise Levels (σ bpm)**: {NOISE_LEVELS}",
        f"**Dropout Rates**: {DROPOUT_RATES}",
        f"**Duration Range**: 30-60 min (≥25 min for Sinusoidal)",
        "",
        "---",
        "",
        "## Overall Summary",
        "",
    ]

    # Overall detection rate
    total_detected = sum(1 for r in results if r.detected)
    lines.append(f"- **Overall Detection Rate**: {total_detected}/{len(results)} ({total_detected/len(results):.1%})")

    # False positive rate (Normal flagged as Cat 2/3)
    normal_results = [r for r in results if r.scenario.pattern == "Normal"]
    if normal_results:
        fp = sum(1 for r in normal_results if r.predicted_cat > 1)
        lines.append(f"- **False Positive Rate (Normal → Cat 2/3)**: {fp}/{len(normal_results)} ({fp/len(normal_results):.1%})")
    lines.append("")

    # Heatmap: Pattern vs Noise Level
    lines += [
        "## Detection Rate: Pattern × Noise Level",
        "",
        "| Pattern | σ=0 | σ=2 | σ=5 | σ=10 | σ=15 |",
        "|---------|-----|-----|-----|------|------|",
    ]
    for pdef in PATTERN_DEFS:
        row = [pdef.name]
        for sigma in NOISE_LEVELS:
            rate = _rate(results, pattern=pdef.name, noise_sigma=sigma)
            row.append(f"{rate:.0%}" if not np.isnan(rate) else "–")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Heatmap: Pattern vs Dropout Rate
    lines += [
        "## Detection Rate: Pattern × Dropout Rate",
        "",
        "| Pattern | 0% | 5% | 10% |",
        "|---------|-----|-----|------|",
    ]
    for pdef in PATTERN_DEFS:
        row = [pdef.name]
        for dr in DROPOUT_RATES:
            rate = _rate(results, pattern=pdef.name, dropout_rate=dr)
            row.append(f"{rate:.0%}" if not np.isnan(rate) else "–")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Heatmap: Pattern vs Severity
    lines += [
        "## Detection Rate: Pattern × Severity",
        "",
        "| Pattern | Mild | Moderate | Severe |",
        "|---------|------|----------|--------|",
    ]
    for pdef in PATTERN_DEFS:
        row = [pdef.name]
        for sev in SEVERITIES:
            rate = _rate(results, pattern=pdef.name, severity=sev)
            row.append(f"{rate:.0%}" if not np.isnan(rate) else "–")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Sinusoidal specific check
    lines += [
        "## Sinusoidal Detection (Clean Signals)",
        "",
    ]
    sino_clean = [r for r in results if r.scenario.pattern == "Sinusoidal" and r.scenario.noise_sigma == 0.0 and r.scenario.dropout_rate == 0.0]
    if sino_clean:
        detected = sum(1 for r in sino_clean if r.detected)
        lines.append(f"- Clean sinusoidal scenarios: {len(sino_clean)}")
        lines.append(f"- Detected (Cat 3): {detected}/{len(sino_clean)} ({detected/len(sino_clean):.0%})")
    else:
        lines.append("- No clean sinusoidal scenarios in sample.")
    lines.append("")

    # Category distribution
    lines += [
        "## Predicted Category Distribution",
        "",
        "| Category | Count | Percentage |",
        "|----------|-------|------------|",
    ]
    cat_counts = defaultdict(int)
    for r in results:
        cat_counts[r.predicted_cat] += 1
    for cat in [1, 2, 3]:
        cnt = cat_counts.get(cat, 0)
        lines.append(f"| Cat {cat} | {cnt} | {cnt/len(results):.1%} |")
    lines.append("")

    # Sample failures
    failures = [r for r in results if not r.detected][:20]
    if failures:
        lines += [
            "## Sample Failures (first 20)",
            "",
            "| ID | Pattern | Severity | Noise | Dropout | Expected | Predicted |",
            "|----|---------|----------|-------|---------|----------|-----------|",
        ]
        for r in failures:
            s = r.scenario
            lines.append(f"| {s.id} | {s.pattern} | {s.severity} | {s.noise_sigma} | {s.dropout_rate:.0%} | Cat{s.expected_cat} | Cat{r.predicted_cat} |")
        lines.append("")

    return "\n".join(lines)


def main():
    print(f"[massive] Generating {NUM_SCENARIOS} scenarios...")
    rng = np.random.default_rng(seed=123)
    scenarios = generate_scenarios(NUM_SCENARIOS, rng)

    print(f"[massive] Running scenarios (batch mode, no sleep)...")
    results = run_all(scenarios)

    print("[massive] Generating report...")
    report = generate_report(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    total_detected = sum(1 for r in results if r.detected)
    print(f"[massive] Done: {total_detected}/{len(results)} scenarios detected ({total_detected/len(results):.1%})")
    print(f"[massive] Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
