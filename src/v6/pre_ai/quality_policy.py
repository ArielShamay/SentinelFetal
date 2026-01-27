"""Quality policy loader for V6 Pre-AI ingest + window quality."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

import yaml

DEFAULT_POLICY: Dict[str, Any] = {
    "record": {
        "max_fhr_nan_frac_for_keep": 0.80,
        "max_fhr_out_of_range_frac_for_keep": 0.60,
        "max_fhr_zero_frac_for_keep": 0.95,
    },
    "window": {
        "hard_low_nan_frac": 0.60,
        "hard_low_max_nan_run": 200,
        "hard_low_out_of_range_frac": 0.20,
        "hard_low_unique_ratio": 0.05,
        "hard_low_std_min": 1.0,
        "hard_low_std_max": 50.0,
        "hard_low_jump_count_gt25": 10,
        "hard_low_max_abs_jump": 25.0,
        "med_nan_frac": 0.35,
        "med_zero_frac": 0.50,
        "med_flatline_ratio": 0.90,
    },
    "interpolation": {
        "max_gap_samples_to_fill": 10,
        "fill_method": "linear",
        "do_not_fill_if_gap_too_large": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=None)
def load_quality_policy(path: Path | None = None) -> Dict[str, Any]:
    config_path = path or Path("config/v6_quality_policy.yaml")
    if not config_path.exists():
        return dict(DEFAULT_POLICY)

    raw = yaml.safe_load(config_path.read_text()) or {}
    policy = _deep_merge(DEFAULT_POLICY, raw)
    return policy
