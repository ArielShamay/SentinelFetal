"""Generate synthetic Gauntlet datasets for logic stress testing (no model training).

Outputs three CSV files under data/synthetic_gauntlet:
- synthetic_logic_check.csv
- synthetic_noise_check.csv
- synthetic_edge_cases.csv

Each row contains:
  case_id, case_type, true_label (1=pathology, 0=normal), noise_type, severity_level,
  sampling_rate, fhr (JSON list), uc (JSON list), description.

Notes:
- Uses the real-time simulator patient generator and injectable events to create coupled FHR/UC.
- Focus is on logic validation, not texture-perfect realism.
- DO NOT use these for training the main AI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.generators.patient_generator import PatientGenerator, PatientConfig
from src.simulation.events.event_types import (
    EventType,
    LateDecelerationParams,
    VariableDecelerationParams,
    BradycardiaParams,
    TachycardiaParams,
)
from src.config import CTG
from src.signal_invariants import assert_signal_length, assert_pair_aligned

OUTPUT_DIR = Path("data/synthetic_gauntlet")
DURATION_SEC = 1800  # 30 minutes baseline duration
LONG_DURATION_SEC = 3600  # optional 60-minute stability traces
MIN_DURATION_SEC = 1200  # stop if shorter than 20 minutes
SAMPLE_RATE = CTG.SAMPLING_RATE
SAMPLES_PER_TICK = int(SAMPLE_RATE)  # generate 1-second ticks


def run_patient(case_id: str, inject_events: List[Tuple[EventType, object]], baseline_fhr: float = 140.0,
                variability: float = 10.0, contractions_per_10min: float = 4.0, duration_sec: int = DURATION_SEC) -> Dict:
    # Enforce 30-minute (or specified) duration at generation time.
    duration_sec = int(duration_sec)
    cfg = PatientConfig(
        patient_id=case_id,
        bed_number=1,
        baseline_fhr=baseline_fhr,
        baseline_variability=variability,
        contractions_per_10min=contractions_per_10min,
    )
    patient = PatientGenerator(cfg)

    for etype, params in inject_events:
        patient.inject_event(etype, params)

    ticks = duration_sec
    fhr_all: List[float] = []
    uc_all: List[float] = []
    for _ in range(ticks):
        data = patient.generate_tick(SAMPLES_PER_TICK)
        fhr_all.extend(data["fhr"].tolist())
        uc_all.extend(data["uc"].tolist())

    expected = duration_sec * SAMPLE_RATE
    assert_signal_length(fhr_all, SAMPLE_RATE, 30, "GEN:FHR")
    assert_signal_length(uc_all, SAMPLE_RATE, 30, "GEN:UC")
    assert_pair_aligned(fhr_all, uc_all, SAMPLE_RATE)
    if len(fhr_all) != expected or len(uc_all) != expected:
        raise ValueError(
            f"CRITICAL: Generated length mismatch for {case_id}. fhr={len(fhr_all)} uc={len(uc_all)} expected={expected}"
        )

    return {
        "fhr": fhr_all,
        "uc": uc_all,
        "sampling_rate": SAMPLE_RATE,
        "duration_minutes": duration_sec / 60.0,
    }


def build_logic_cases() -> List[Dict]:
    cases = []
    case_specs = [
        ("logic_late_clean", 1, "Clean late decelerations", [(EventType.LATE_DECELERATION, LateDecelerationParams.moderate())], DURATION_SEC),
        ("logic_variable_with_overshoot", 1, "Variable decel with overshoot", [(EventType.VARIABLE_DECELERATION, VariableDecelerationParams.severe())], DURATION_SEC),
        ("logic_variable_no_overshoot", 1, "Variable decel without overshoot", [(EventType.VARIABLE_DECELERATION, VariableDecelerationParams.moderate())], DURATION_SEC),
        ("logic_gradual_hypoxia", 1, "Gradual hypoxia (bradycardia onset)", [(EventType.BRADYCARDIA, BradycardiaParams.moderate())], LONG_DURATION_SEC),
    ]

    for cid, label, desc, events, dur in case_specs:
        trace = run_patient(cid, events, duration_sec=dur)
        cases.append({
            "case_id": cid,
            "case_type": "logic",
            "true_label": label,
            "noise_type": "none",
            "severity_level": "pathological",
            "sampling_rate": trace["sampling_rate"],
            "fhr": json.dumps(trace["fhr"]),
            "uc": json.dumps(trace["uc"]),
            "duration_minutes": trace["duration_minutes"],
            "description": desc,
        })
    return cases


def build_noise_cases() -> List[Dict]:
    cases = []
    base = run_patient("noise_base", [], duration_sec=DURATION_SEC)
    fhr = np.array(base["fhr"])
    uc = np.array(base["uc"])

    def add_case(cid: str, fhr_sig: np.ndarray, uc_sig: np.ndarray, noise_type: str, desc: str, duration_minutes: float):
        duration_sec = int(round(duration_minutes * 60))
        expected = duration_sec * SAMPLE_RATE
        assert_signal_length(fhr_sig, SAMPLE_RATE, 30, "GEN:FHR")
        assert_signal_length(uc_sig, SAMPLE_RATE, 30, "GEN:UC")
        assert_pair_aligned(fhr_sig, uc_sig, SAMPLE_RATE)
        if len(fhr_sig) != expected or len(uc_sig) != expected:
            raise ValueError(
                f"CRITICAL: Length mismatch for {cid}. fhr={len(fhr_sig)}, uc={len(uc_sig)}, expected={expected}"
            )
        cases.append({
            "case_id": cid,
            "case_type": "noise",
            "true_label": 0,
            "noise_type": noise_type,
            "severity_level": "artifact",
            "sampling_rate": SAMPLE_RATE,
            "fhr": json.dumps(fhr_sig.tolist()),
            "uc": json.dumps(uc_sig.tolist()),
            "duration_minutes": duration_minutes,
            "description": desc,
        })

    # Motion spikes
    fhr_spike = fhr.copy()
    spike_idx = np.random.choice(len(fhr_spike), size=50, replace=False)
    fhr_spike[spike_idx] += np.random.uniform(40, 80, size=50)
    add_case("noise_motion_spikes", fhr_spike, uc, "motion_spikes", "Random motion spikes on FHR", base["duration_minutes"])

    # Signal gaps / dropouts
    fhr_gap = fhr.copy()
    gap_start = len(fhr_gap) // 3
    fhr_gap[gap_start:gap_start + 200] = np.nan
    add_case("noise_dropouts", fhr_gap, uc, "dropout", "Short signal dropout", base["duration_minutes"])

    # Baseline drift
    drift = np.linspace(-15, 15, len(fhr))
    fhr_drift = fhr + drift
    add_case("noise_baseline_drift", fhr_drift, uc, "baseline_drift", "Slow baseline drift", base["duration_minutes"])

    # Quantization artifacts
    fhr_quant = np.round(fhr / 2) * 2
    add_case("noise_quantization", fhr_quant, uc, "quantization", "Quantized FHR values", base["duration_minutes"])

    # UC-only noise (no FHR response)
    uc_noisy = uc + np.random.normal(0, 15, size=len(uc))
    add_case("noise_uc_only", fhr, uc_noisy, "uc_only", "Noisy UC without FHR response", base["duration_minutes"])

    # Optional 60-minute stability noise case (sparse spikes, not per-sample noise)
    long_base = run_patient("noise_base_long", [], duration_sec=LONG_DURATION_SEC)
    fhr_long = np.array(long_base["fhr"])
    uc_long = np.array(long_base["uc"])
    fhr_long_spikes = fhr_long.copy()
    spike_idx_long = np.random.choice(len(fhr_long_spikes), size=100, replace=False)
    fhr_long_spikes[spike_idx_long] += np.random.uniform(30, 60, size=100)
    add_case("noise_motion_spikes_long", fhr_long_spikes, uc_long, "motion_spikes", "Long run with motion spikes", long_base["duration_minutes"])

    return cases


def build_edge_cases() -> List[Dict]:
    cases = []
    # Variable decel + overshoot (rare but critical)
    trace1 = run_patient("edge_variable_overshoot", [(EventType.VARIABLE_DECELERATION, VariableDecelerationParams.severe())], duration_sec=DURATION_SEC)
    cases.append({
        "case_id": "edge_variable_overshoot",
        "case_type": "edge",
        "true_label": 1,
        "noise_type": "none",
        "severity_level": "severe",
        "sampling_rate": trace1["sampling_rate"],
        "fhr": json.dumps(trace1["fhr"]),
        "uc": json.dumps(trace1["uc"]),
        "description": "Variable decel with overshoot",
        "duration_minutes": trace1["duration_minutes"],
    })

    # Borderline baselines (110-115 bpm)
    trace2 = run_patient("edge_borderline_baseline", [], baseline_fhr=112.0, variability=8.0, duration_sec=DURATION_SEC)
    cases.append({
        "case_id": "edge_borderline_baseline",
        "case_type": "edge",
        "true_label": 0,
        "noise_type": "none",
        "severity_level": "borderline",
        "sampling_rate": trace2["sampling_rate"],
        "fhr": json.dumps(trace2["fhr"]),
        "uc": json.dumps(trace2["uc"]),
        "description": "Borderline low baseline without pathology",
        "duration_minutes": trace2["duration_minutes"],
    })

    # High variability but no pathology
    trace3 = run_patient("edge_high_variability", [], variability=12.0, duration_sec=DURATION_SEC)
    fhr_hi_var = np.array(trace3["fhr"], dtype=float)
    stable_len = int(5 * 60 * SAMPLE_RATE)
    stable_base = float(np.nanmedian(fhr_hi_var)) if fhr_hi_var.size else 140.0
    stable_start = int(15 * 60 * SAMPLE_RATE)
    stable_end = min(stable_start + stable_len, len(fhr_hi_var))
    fhr_hi_var[stable_start:stable_end] = stable_base + np.random.normal(0, 2.0, size=stable_end - stable_start)
    trace3["fhr"] = fhr_hi_var.tolist()
    cases.append({
        "case_id": "edge_high_variability",
        "case_type": "edge",
        "true_label": 0,
        "noise_type": "none",
        "severity_level": "benign_high_variability",
        "sampling_rate": trace3["sampling_rate"],
        "fhr": json.dumps(trace3["fhr"]),
        "uc": json.dumps(trace3["uc"]),
        "description": "Marked variability without decels",
        "duration_minutes": trace3["duration_minutes"],
    })

    # Gradual hypoxia with mild late decels + drift
    trace4 = run_patient("edge_gradual_hypoxia", [(EventType.LATE_DECELERATION, LateDecelerationParams.mild())], baseline_fhr=125.0, duration_sec=LONG_DURATION_SEC)
    drift = np.linspace(-10, 5, len(trace4["fhr"]))
    fhr_drifted = (np.array(trace4["fhr"]) + drift).tolist()
    cases.append({
        "case_id": "edge_gradual_hypoxia",
        "case_type": "edge",
        "true_label": 1,
        "noise_type": "baseline_drift",
        "severity_level": "moderate",
        "sampling_rate": trace4["sampling_rate"],
        "fhr": json.dumps(fhr_drifted),
        "uc": json.dumps(trace4["uc"]),
        "description": "Slowly dropping baseline with mild late decels",
        "duration_minutes": trace4["duration_minutes"],
    })

    return cases


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def verify_lengths(label: str, rows: List[Dict]) -> None:
    """Fail fast if any generated trace is shorter than the requested 30m duration."""
    for row in rows:
        fhr = json.loads(row["fhr"])
        uc = json.loads(row["uc"])
        assert_signal_length(fhr, SAMPLE_RATE, 30, f"GEN_VERIFY:{label}:FHR")
        assert_signal_length(uc, SAMPLE_RATE, 30, f"GEN_VERIFY:{label}:UC")
        assert_pair_aligned(fhr, uc, SAMPLE_RATE)


def verify_post_save(csv_path: Path) -> None:
    """Reload CSV and re-assert lengths to catch truncation."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    sample_indices = df.sample(n=min(5, len(df)), random_state=42).index
    for idx in sample_indices:
        row = df.loc[idx]
        fhr = json.loads(row["fhr"])
        uc = json.loads(row["uc"])
        sr = int(row.get("sampling_rate", SAMPLE_RATE))
        assert_signal_length(fhr, sr, 30, "CSV_RELOAD:FHR")
        assert_signal_length(uc, sr, 30, "CSV_RELOAD:UC")
        assert_pair_aligned(fhr, uc, sr)


def summarize_cases(all_rows: List[Dict]) -> None:
    durations = []
    samples = []
    for row in all_rows:
        fhr = json.loads(row["fhr"])
        sr = int(row.get("sampling_rate", SAMPLE_RATE))
        samples.append(len(fhr))
        durations.append(len(fhr) / sr / 60.0)

    def stats(vals: List[float]) -> Tuple[float, float, float]:
        return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))

    min_d, mean_d, max_d = stats(durations)
    min_s, mean_s, max_s = stats(samples)
    print(f"Duration minutes (min/mean/max): {min_d:.2f} / {mean_d:.2f} / {max_d:.2f}")
    print(f"Sample count (min/mean/max): {min_s:.0f} / {mean_s:.0f} / {max_s:.0f}")


def main():
    logic_cases = build_logic_cases()
    noise_cases = build_noise_cases()
    edge_cases = build_edge_cases()

    verify_lengths("logic", logic_cases)
    verify_lengths("noise", noise_cases)
    verify_lengths("edge", edge_cases)

    write_csv(OUTPUT_DIR / "synthetic_logic_check.csv", logic_cases)
    write_csv(OUTPUT_DIR / "synthetic_noise_check.csv", noise_cases)
    write_csv(OUTPUT_DIR / "synthetic_edge_cases.csv", edge_cases)

    verify_post_save(OUTPUT_DIR / "synthetic_logic_check.csv")
    verify_post_save(OUTPUT_DIR / "synthetic_noise_check.csv")
    verify_post_save(OUTPUT_DIR / "synthetic_edge_cases.csv")

    all_rows = logic_cases + noise_cases + edge_cases
    summarize_cases(all_rows)

    print(f"Saved {len(logic_cases)} logic cases, {len(noise_cases)} noise cases, {len(edge_cases)} edge cases to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
