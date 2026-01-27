"""V6 Pre-AI compatibility adapter (replaces legacy PipelineAdapter)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

import numpy as np

from src.utils.runtime_config import load_runtime_config
from src.v6.pre_ai.pipeline import run_pre_ai

logger = logging.getLogger(__name__)


@dataclass
class PipelineAdapterConfig:
    """Compatibility config (legacy fields accepted but unused in V6 Pre-AI)."""
    use_real_moment: bool = False
    model_path: str = ""
    sampling_rate: float = 4.0
    min_data_seconds: float = 1200.0
    enable_mhr_guard: bool = False
    enable_trend_analysis: bool = False
    enable_explanations: bool = False


class PipelineAdapter:
    """Compatibility adapter that redirects to V6 Pre-AI pipeline only."""

    def __init__(self, config: Optional[PipelineAdapterConfig] = None) -> None:
        self.config = config or PipelineAdapterConfig()
        self._cfg = load_runtime_config()

    def process_patient(
        self,
        patient_id: str,
        data: Dict[str, Any],
        run_moment: bool = False,
    ) -> Dict[str, Any]:
        fhr = np.asarray(data.get("fhr", []), dtype=float).reshape(-1)
        uc = np.asarray(data.get("uc", []), dtype=float).reshape(-1)

        if fhr.size == 0 or uc.size == 0:
            raise ValueError("PRE_AI_ADAPTER | missing FHR/UC data")

        records = run_pre_ai(patient_id, fhr, uc)
        baseline = _safe_mean(fhr, default=140.0)
        variability = _safe_std(fhr, default=10.0)

        summary = {
            "patient_id": patient_id,
            "category": 1,
            "baseline_fhr": baseline,
            "variability": variability,
            "fsqi": 1.0,
            "quality_class": records[-1].get("quality_class") if records else None,
            "window_count": len(records),
            "windows": records,
            "ai_enabled": False,
            "ai_prob": None,
            "legacy_disabled": True,
        }
        return summary


def _safe_mean(arr: np.ndarray, default: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.mean(finite))


def _safe_std(arr: np.ndarray, default: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.std(finite))
