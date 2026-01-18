"""
Robustness "Torture Test" Benchmark

Verifies system detection capability under degraded signal conditions:
- Gaussian noise (σ = 2, 5, 10 bpm)
- Random dropouts (NaN segments)
- Artifact spikes (±30 bpm)

Runs batch-mode: no real-time pacing; processes ~50 scenarios instantly.
Outputs: docs/ROBUSTNESS_TEST_REPORT.md with detection-rate matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple
import itertools

import numpy as np

# Ensure project root importable
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CTG
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.pipeline.container import PipelineContainer
from src.simulation.generators.patient_generator import PatientConfig, PatientGenerator
from src.simulation.events.event_types import (
    EventType,
    SinusoidalParams,
    LateDecelerationParams,
    VariableDecelerationParams,
)

SAMPLING_RATE = CTG.SAMPLING_RATE
WINDOW_SECONDS = 10 * 60  # 10-min analysis window
REPORT_PATH = PROJECT_ROOT / "docs" / "ROBUSTNESS_TEST_REPORT.md"


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------
@dataclass
class PatternSpec:
    name: str
    event_type: EventType
    params: object
    expected_cat: int  # 2 or 3


PATTERNS: List[PatternSpec] = [
    PatternSpec(
        name="Sinusoidal",
        event_type=EventType.SINUSOIDAL_PATTERN,
        params=SinusoidalParams(
            duration_seconds=WINDOW_SECONDS,
            amplitude_bpm=10.0,
            frequency_cycles_per_min=4.0,
        ),
        expected_cat=3,
    ),
    PatternSpec(
        name="Late Decel (severe)",
        event_type=EventType.LATE_DECELERATION,
        params=LateDecelerationParams.severe(),
        expected_cat=2,
    ),
    PatternSpec(
        name="Variable Decel (severe)",
        event_type=EventType.VARIABLE_DECELERATION,
        params=VariableDecelerationParams.severe(),
        expected_cat=2,
    ),
]

NOISE_LEVELS = [0.0, 2.0, 5.0, 10.0]  # σ in bpm
DROPOUT_RATES = [0.0, 0.02, 0.05]  # fraction of samples set to NaN
ARTIFACT_RATES = [0.0, 0.01, 0.02]  # fraction of samples with spikes


def _generate_clean_pattern(spec: PatternSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a clean 10-min signal with the specified pathology."""
    cfg = PatientConfig(
        patient_id="robust",
        bed_number=1,
        baseline_fhr=140.0,
        baseline_variability=10.0,
        buffer_duration_minutes=12.0,
    )
    patient = PatientGenerator(cfg)
    # warm-up 1 min
    patient.generate_tick(int(60 * SAMPLING_RATE))
    # inject event and generate duration
    patient.inject_event(spec.event_type, spec.params)
    patient.generate_tick(int(WINDOW_SECONDS * SAMPLING_RATE))
    window = patient._buffer.get_window(duration_seconds=WINDOW_SECONDS)
    return window["fhr"].copy(), window["uc"].copy()


# ---------------------------------------------------------------------------
# Corruption layers
# ---------------------------------------------------------------------------

def add_noise(fhr: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to FHR signal."""
    if sigma <= 0.0:
        return fhr
    return fhr + rng.normal(0.0, sigma, fhr.shape)


def add_dropouts(fhr: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly set segments to NaN (sensor loss simulation)."""
    if rate <= 0.0:
        return fhr
    mask = rng.random(fhr.shape) < rate
    out = fhr.copy()
    out[mask] = np.nan
    return out


def add_artifacts(fhr: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Add random spikes (+/- 30 bpm)."""
    if rate <= 0.0:
        return fhr
    mask = rng.random(fhr.shape) < rate
    spikes = rng.choice([-30.0, 30.0], size=fhr.shape)
    out = fhr.copy()
    out[mask] += spikes[mask]
    return out


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    pattern: str
    noise_sigma: float
    dropout_rate: float
    artifact_rate: float
    expected_cat: int
    predicted_cat: int
    detected: bool


def run_scenario(
    spec: PatternSpec,
    noise_sigma: float,
    dropout_rate: float,
    artifact_rate: float,
    pipeline: AnalysisPipeline,
    rng: np.random.Generator,
) -> ScenarioResult:
    fhr_clean, uc = _generate_clean_pattern(spec)
    fhr = add_noise(fhr_clean, noise_sigma, rng)
    fhr = add_dropouts(fhr, dropout_rate, rng)
    fhr = add_artifacts(fhr, artifact_rate, rng)
    # Fill NaNs with linear interp for pipeline (preprocessing also does this)
    nans = np.isnan(fhr)
    if nans.any():
        indices = np.arange(len(fhr))
        fhr[nans] = np.interp(indices[nans], indices[~nans], fhr[~nans])
    result = pipeline.analyze(fhr, uc, sampling_rate=SAMPLING_RATE)
    detected = result.category >= spec.expected_cat
    return ScenarioResult(
        pattern=spec.name,
        noise_sigma=noise_sigma,
        dropout_rate=dropout_rate,
        artifact_rate=artifact_rate,
        expected_cat=spec.expected_cat,
        predicted_cat=result.category,
        detected=detected,
    )


def run_all_scenarios() -> List[ScenarioResult]:
    container = PipelineContainer.create_default(use_mock_moment=True)
    pipeline = AnalysisPipeline(container)
    rng = np.random.default_rng(seed=12345)

    results: List[ScenarioResult] = []
    combos = list(itertools.product(PATTERNS, NOISE_LEVELS, DROPOUT_RATES, ARTIFACT_RATES))
    print(f"[robustness] Running {len(combos)} scenarios...")
    for spec, noise, dropout, artifact in combos:
        res = run_scenario(spec, noise, dropout, artifact, pipeline, rng)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _detection_rate(results: List[ScenarioResult], **filters) -> float:
    subset = [
        r for r in results
        if all(getattr(r, k) == v for k, v in filters.items())
    ]
    if not subset:
        return float("nan")
    return sum(1 for r in subset if r.detected) / len(subset)


def generate_report(results: List[ScenarioResult]) -> str:
    lines = [
        "---",
        "title: Phase 11 – Robustness Torture Test",
        f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "---",
        "",
        "# Robustness Test Report",
        "",
        "This benchmark injects **Gaussian noise**, **signal dropouts** (NaN), and **artifact spikes** into clean pathological patterns, then measures whether the pipeline still detects them.",
        "",
        "## Configuration",
        "",
        f"- **Window**: {WINDOW_SECONDS // 60} min @ {SAMPLING_RATE} Hz",
        f"- **Patterns**: {', '.join(p.name for p in PATTERNS)}",
        f"- **Noise σ (bpm)**: {NOISE_LEVELS}",
        f"- **Dropout rates**: {DROPOUT_RATES}",
        f"- **Artifact rates**: {ARTIFACT_RATES}",
        f"- **Total scenarios**: {len(results)}",
        "",
        "## Detection Rate by Noise Level (all patterns combined)",
        "",
        "| Noise σ (bpm) | Detection Rate |",
        "|---------------|----------------|",
    ]
    for sigma in NOISE_LEVELS:
        rate = _detection_rate(results, noise_sigma=sigma)
        lines.append(f"| {sigma} | {rate:.0%} |")
    lines += [
        "",
        "## Detection Rate by Dropout Rate",
        "",
        "| Dropout Rate | Detection Rate |",
        "|--------------|----------------|",
    ]
    for dr in DROPOUT_RATES:
        rate = _detection_rate(results, dropout_rate=dr)
        lines.append(f"| {dr:.0%} | {rate:.0%} |")
    lines += [
        "",
        "## Detection Rate by Artifact Rate",
        "",
        "| Artifact Rate | Detection Rate |",
        "|---------------|----------------|",
    ]
    for ar in ARTIFACT_RATES:
        rate = _detection_rate(results, artifact_rate=ar)
        lines.append(f"| {ar:.0%} | {rate:.0%} |")
    lines += [
        "",
        "## Detection Rate per Pattern",
        "",
        "| Pattern | Clean | σ=2 | σ=5 | σ=10 |",
        "|---------|-------|-----|-----|------|",
    ]
    for spec in PATTERNS:
        row = [spec.name]
        for sigma in NOISE_LEVELS:
            rate = _detection_rate(results, pattern=spec.name, noise_sigma=sigma)
            row.append(f"{rate:.0%}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Detailed Scenario Results",
        "",
        "| Pattern | Noise σ | Dropout | Artifact | Expected | Predicted | Detected |",
        "|---------|---------|---------|----------|----------|-----------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r.pattern} | {r.noise_sigma} | {r.dropout_rate:.0%} | {r.artifact_rate:.0%} | "
            f"Cat{r.expected_cat} | Cat{r.predicted_cat} | {'✅' if r.detected else '❌'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    results = run_all_scenarios()
    report = generate_report(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    detected = sum(1 for r in results if r.detected)
    print(f"[robustness] {detected}/{len(results)} scenarios detected")
    print(f"[robustness] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
