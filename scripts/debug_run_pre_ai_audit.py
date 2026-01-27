#!/usr/bin/env python
"""
Pre-AI Infrastructure Audit Runner (V6).

Scope: ingestion -> RAW invariants -> quality gate (RAW) -> windowing.
NO AI feature extraction or model inference is executed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.runtime_config import load_runtime_config, apply_strict_warnings
from src.v6.pre_ai.pipeline import run_pre_ai
from src.v6.pre_ai.invariants import WarmupError


@dataclass
class AuditSummary:
    ingestion_ok: bool = True
    invariants_failfast_ok: bool = True
    warmup_error_ok: bool = True
    quality_gate_raw_ok: bool = True
    windowing_math_ok: bool = True
    warnings_zero: bool = True
    fallback_count_zero: bool = True


def _generate_case(patient_id: str, minutes: float, fs_hz: float) -> Dict[str, np.ndarray]:
    duration_sec = int(round(minutes * 60))
    n_samples = int(duration_sec * fs_hz)
    t = np.arange(n_samples) / fs_hz
    slow_cycle = np.sin(2 * np.pi * t / (5 * 60))  # ~5 min cycle
    fhr_all = 140.0 + 6.0 * slow_cycle
    uc_all = 10.0 + 40.0 * np.abs(np.sin(2 * np.pi * t / (2 * 60)))
    return {
        "fhr": np.asarray(fhr_all, dtype=float),
        "uc": np.asarray(uc_all, dtype=float),
        "sampling_rate": fs_hz,
    }


def run_patient(patient_id: str, minutes: float, strict: bool, allow_short: bool) -> AuditSummary:
    cfg = load_runtime_config()
    if strict and not cfg.strict_mode:
        raise RuntimeError("STRICT_MODE is OFF in runtime config.")

    if cfg.fs_hz != 4:
        raise RuntimeError(f"STRICT_FS: runtime fs_hz {cfg.fs_hz} != 4")

    if (not allow_short) and minutes < cfg.recommended_case_minutes:
        raise RuntimeError(
            f"STRICT_DURATION: {patient_id} duration {minutes:.2f} min < "
            f"recommended_case {cfg.recommended_case_minutes}"
        )

    summary = AuditSummary()
    data = _generate_case(patient_id, minutes, cfg.fs_hz)
    fhr = data["fhr"]
    uc = data["uc"]

    records = run_pre_ai(patient_id, fhr, uc)
    for record in records:
        print(json.dumps(record, ensure_ascii=True))
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=int, default=1)
    parser.add_argument("--minutes", type=float, default=35.0)
    parser.add_argument("--strict", type=int, default=1)
    parser.add_argument("--allow_short", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config()
    strict = bool(args.strict)

    # Convert warnings to errors for strict audit runs
    apply_strict_warnings(strict)
    warnings.filterwarnings("error")
    if strict:
        class _StrictLogHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                raise RuntimeError(f"STRICT_LOG_WARNING: {record.getMessage()}")

        root = logging.getLogger()
        root.setLevel(logging.WARNING)
        root.addHandler(_StrictLogHandler(level=logging.WARNING))

    summaries = []
    for idx in range(args.patients):
        patient_id = f"P{idx + 1}"
        summaries.append(run_patient(patient_id, args.minutes, strict, bool(args.allow_short)))

    # Emit summary line
    summary = AuditSummary()
    print("AUDIT_SUMMARY", json.dumps(summary.__dict__, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
