"""Smart Tiered Hybrid Logic for SentinelFetal.

Implements quantile-calibrated, quality-aware three-tier logic:
- Tier 1: if quality=HIGH and ai_score >= T_high -> alert
- Tier 2: if quality in {HIGH, MED} and T_low <= ai_score < T_high and rule_score >= suspicious and quality >= Q_min -> alert
- Tier 3: if rule_score >= pathological -> alert
- If quality=LOW: disable AI; only tier 3 can alert.

Defaults are placeholders; thresholds should be calibrated from real negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

from src.config import CTG
from src.signal_invariants import assert_signal_length


@dataclass
class SmartLogicConfig:
    t_high: float = 0.65
    t_low: float = 0.35
    q_min: float = 0.7
    suspicious_rule: float = 0.5
    pathological_rule: float = 0.8


@dataclass
class SmartLogicResult:
    alert: bool
    tier: str
    reason: str


def evaluate(
    ai_score: float,
    rule_score: float,
    cfg: SmartLogicConfig = SmartLogicConfig(),
    quality_class: str | None = None,
    signal_quality: float = 1.0,
    hard_low: bool = False,
    boredom_block: bool = False,
) -> SmartLogicResult:
    # Hard safety: if quality flagged LOW, AI is disabled; only pathological rule can alert.
    if hard_low or (quality_class == "LOW") or (signal_quality < 0.6):
        if rule_score >= cfg.pathological_rule:
            return SmartLogicResult(True, "tier3", f"Low quality but rule pathological {rule_score:.3f} >= {cfg.pathological_rule}")
        return SmartLogicResult(False, "dismiss", "Low quality, AI disabled")

    # Tier 3: severe rule always alerts (safety net)
    if rule_score >= cfg.pathological_rule:
        return SmartLogicResult(True, "tier3", f"Rule pathological {rule_score:.3f} >= {cfg.pathological_rule}")

    if ai_score >= cfg.t_high and signal_quality >= cfg.q_min:
        return SmartLogicResult(True, "tier1", f"AI {ai_score:.3f} >= {cfg.t_high} with quality {signal_quality:.3f}")
    if cfg.t_low <= ai_score < cfg.t_high:
        if boredom_block:
            # Boredom suppresses grey-zone hallucinations only (Tier 2). Tier 1 remains active.
            return SmartLogicResult(False, "tier2", "Boredom gate: grey-zone suppressed")
        if rule_score >= cfg.suspicious_rule and signal_quality >= cfg.q_min:
            return SmartLogicResult(True, "tier2", f"AI grey {ai_score:.3f}, rule {rule_score:.3f} >= {cfg.suspicious_rule} and quality {signal_quality:.3f} >= {cfg.q_min}")
        return SmartLogicResult(False, "tier2", f"AI grey {ai_score:.3f} but rule/quality low")
    return SmartLogicResult(False, "dismiss", "Below thresholds")


def to_dict(cfg: SmartLogicConfig) -> Dict:
    return {
        "t_high": cfg.t_high,
        "t_low": cfg.t_low,
        "q_min": cfg.q_min,
        "suspicious_rule": cfg.suspicious_rule,
        "pathological_rule": cfg.pathological_rule,
    }


def load_threshold_config(path: Path) -> Tuple[SmartLogicConfig, int, int, int]:
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    logic_cfg = SmartLogicConfig(
        t_high=float(cfg.get("t_high", SmartLogicConfig().t_high)),
        t_low=float(cfg.get("t_low", SmartLogicConfig().t_low)),
        q_min=float(cfg.get("q_min", SmartLogicConfig().q_min)),
        suspicious_rule=float(cfg.get("suspicious_rule", SmartLogicConfig().suspicious_rule)),
        pathological_rule=float(cfg.get("pathological_rule", SmartLogicConfig().pathological_rule)),
    )
    k = int(cfg.get("persistence_k", 2))
    n = int(cfg.get("persistence_n", 3))
    window_minutes = int(cfg.get("persistence_window_minutes", 15))
    return logic_cfg, k, n, window_minutes


def persistence_alert(window_flags: list[bool], k: int = 2, n: int = 3) -> bool:
    if len(window_flags) < n:
        return False
    for i in range(0, len(window_flags) - n + 1):
        if sum(1 for flag in window_flags[i:i + n] if flag) >= k:
            return True
    return False


def persistence_with_cooldown(window_flags: list[bool], k: int = 2, n: int = 3, cooldown_windows: int = 0) -> Tuple[bool, int]:
    """Apply K-of-N persistence with optional cooldown after first alert.

    Returns (alerted, suppressed_count).
    """
    if len(window_flags) < n:
        return False, 0

    alerted = False
    suppressed = 0
    i = 0
    while i <= len(window_flags) - n:
        window = window_flags[i:i + n]
        if sum(1 for flag in window if flag) >= k:
            if not alerted:
                alerted = True
                if cooldown_windows > 0:
                    i += cooldown_windows
                    continue
            else:
                suppressed += 1
        i += 1

    return alerted, suppressed


def compute_signal_quality(
    raw_fhr_window: np.ndarray,
    raw_uc_window: np.ndarray | None = None,
    fs: float = CTG.SAMPLING_RATE,
    return_metrics: bool = False,
) -> Tuple[float, float, float, str]:
    """Compute quality on RAW signal only. If any hard rule fires -> quality LOW.

    Hard rules:
    - >25 bpm jump count > 5
    - std > 30
    - std < 1.0
    - valid_frac < 0.6
    - unique_ratio < 0.05 (quantization)
    - max_nan_run >= 50 (dropouts)

    When return_metrics=True, a diagnostics dict is returned instead of the tuple.
    """

    assert_signal_length(raw_fhr_window, fs, 20, "QUALITY:RAW_FHR")
    if raw_uc_window is not None:
        assert_signal_length(raw_uc_window, fs, 20, "QUALITY:RAW_UC")

    window = np.asarray(raw_fhr_window, dtype=float)
    if window.size == 0:
        diag = {
            "valid_frac": 0.0,
            "nan_frac": 1.0,
            "zeros_frac": 1.0,
            "max_nan_run": 0,
            "std": 0.0,
            "mad": 0.0,
            "max_abs_jump": 0.0,
            "jump_count_gt25": 0,
            "unique_ratio": 0.0,
            "out_of_range_frac": 1.0,
            "flatline_ratio": 1.0,
            "quality_class": "LOW",
            "hard_low": True,
        }
        return diag if return_metrics else ("LOW", True, diag)

    finite = np.isfinite(window)
    valid_frac = float(np.mean(finite)) if window.size else 0.0
    nan_frac = float(1.0 - valid_frac)
    clean = window[finite]
    zeros_frac = float(np.mean((window == 0.0) & finite)) if window.size else 1.0

    # Long NaN gaps are a strong dropout indicator.
    isnan = ~finite
    max_nan_run = 0
    current_run = 0
    for flag in isnan:
        if flag:
            current_run += 1
            if current_run > max_nan_run:
                max_nan_run = current_run
        else:
            current_run = 0

    std = float(np.std(clean)) if clean.size else 0.0
    mad = float(np.median(np.abs(clean - np.median(clean)))) if clean.size else 0.0
    diffs = np.diff(clean) if clean.size > 1 else np.array([])
    abs_diffs = np.abs(diffs)
    jump_count_gt25 = int(np.sum(abs_diffs > 25.0)) if abs_diffs.size else 0
    max_abs_jump = float(np.max(abs_diffs)) if abs_diffs.size else 0.0
    flatline_ratio = float(np.mean(abs_diffs < 0.5)) if abs_diffs.size else 1.0
    out_of_range_frac = float(np.mean((clean < 60.0) | (clean > 220.0))) if clean.size else 1.0
    unique_ratio = float(len(np.unique(clean)) / len(clean)) if clean.size else 0.0

    hard_low = (
        nan_frac > 0.40
        or max_nan_run >= 50
        or unique_ratio < 0.05
        or std < 1.0
        or std > 50.0
        or jump_count_gt25 > 5
        or max_abs_jump > 25.0
        or out_of_range_frac > 0.05
    )

    if hard_low:
        quality_class = "LOW"
    elif nan_frac > 0.20 or zeros_frac > 0.30 or flatline_ratio > 0.8:
        quality_class = "MED"
    else:
        quality_class = "HIGH"

    diag = {
        "valid_frac": valid_frac,
        "nan_frac": nan_frac,
        "zeros_frac": zeros_frac,
        "max_nan_run": max_nan_run,
        "std": std,
        "mad": mad,
        "max_abs_jump": max_abs_jump,
        "jump_count_gt25": jump_count_gt25,
        "unique_ratio": unique_ratio,
        "out_of_range_frac": out_of_range_frac,
        "flatline_ratio": flatline_ratio,
        "quality_class": quality_class,
        "hard_low": hard_low,
    }

    if return_metrics:
        return diag

    return quality_class, hard_low, diag
