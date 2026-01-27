"""V6 Pre-AI pipeline (RAW ingestion -> invariants -> quality gate -> windowing)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.utils.runtime_config import load_runtime_config
from src.v6.pre_ai.invariants import assert_raw_invariants
from src.v6.pre_ai.quality_gate import quality_gate
from src.v6.pre_ai.windowing import window_iter


def run_pre_ai(
    patient_id: str,
    fhr_raw: np.ndarray,
    uc_raw: np.ndarray,
) -> List[Dict]:
    cfg = load_runtime_config()
    fhr_arr, uc_arr = assert_raw_invariants(
        fhr_raw,
        uc_raw,
        fs_hz=cfg.fs_hz,
        min_case_minutes=cfg.min_case_minutes,
        min_window_minutes=cfg.min_window_minutes,
        tag=f"RAW:{patient_id}",
    )

    records: List[Dict] = []
    window_index = 0
    for start, end, fhr_w, uc_w in window_iter(
        fhr_arr,
        uc_arr,
        fs_hz=cfg.fs_hz,
        window_minutes=cfg.window_minutes,
        stride_minutes=cfg.stride_minutes,
        min_window_minutes=cfg.min_window_minutes,
    ):
        quality_class, diag = quality_gate(
            fhr_w,
            uc_w,
            fs_hz=cfg.fs_hz,
            min_window_minutes=cfg.min_window_minutes,
        )
        record = {
            "patient_id": patient_id,
            "window_index": window_index,
            "window_start": start,
            "window_end": end,
            "window_minutes": round((end - start) / cfg.fs_hz / 60.0, 2),
            "quality_class": quality_class,
            "hard_low": bool(diag.get("hard_low", False)),
            "invariant_ok": True,
            "violations": [],
        }
        records.append(record)
        window_index += 1

    if not records:
        raise RuntimeError(f"STRICT_WINDOWING: {patient_id} produced zero windows")

    return records
