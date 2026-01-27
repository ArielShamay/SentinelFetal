"""Run full SentinelFetal pipeline on synthetic Gauntlet datasets (no training).

Inputs: data/synthetic_gauntlet/*.csv produced by generate_gauntlet.py
Outputs: data/synthetic_gauntlet/synthetic_fp_analysis.csv with columns:
  case_id, case_type, noise_type, ai_score, rule_score, signal_quality,
  final_decision, is_false_positive, true_label

Also prints FP breakdowns by noise type, signal quality bin, and cases where
Rule is low but AI high.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.container import PipelineContainer
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.config import CTG, THRESHOLDS
from src.adapters.model_adapters import MiniRocketAdapter, ClassifierAdapter
from src.adapters import (
    PreprocessorAdapter,
    BaselineAdapter,
    VariabilityAdapter,
    DecelerationAdapter,
    TachysystoleAdapter,
    SinusoidalAdapter,
    FusionAdapter,
    OverrideAdapter,
    AlertAdapter,
)
from src.decision.smart_hybrid_logic import SmartLogicConfig, evaluate, compute_signal_quality
from src.signal_invariants import assert_signal_length, assert_pair_aligned
from src.analysis.fallback_audit import (
    reset_fallback_audit,
    set_case_context,
    set_window_context,
    get_fallback_records,
    write_fallback_audit,
    raise_if_any_fallback,
)
from src.utils.runtime_config import load_runtime_config, apply_strict_warnings

GAUNTLET_DIR = Path("data/synthetic_gauntlet")
OUTPUT_PATH = GAUNTLET_DIR / "synthetic_fp_analysis.csv"
HARD_NEG_PATH = GAUNTLET_DIR / "synthetic_hard_negatives.csv"
CONFIG_PATH = Path("config/ensemble_v5_optuna.yaml")
LOGIC_CONFIG_PATH = Path("config/logic_v5_2.yaml")
RUNTIME_CFG = load_runtime_config()
if float(CTG.SAMPLING_RATE) != float(RUNTIME_CFG.fs_hz):
    raise RuntimeError(
        f"Runtime fs_hz mismatch: runtime={RUNTIME_CFG.fs_hz} ctg={CTG.SAMPLING_RATE}"
    )
# DO NOT change analysis windows to silence warnings. Fix generator/pipeline instead.
WINDOW_MINUTES = RUNTIME_CFG.window_minutes
STRIDE_MINUTES = RUNTIME_CFG.stride_minutes
MIN_WINDOW_MINUTES = RUNTIME_CFG.min_window_minutes
RECOMMENDED_CASE_MINUTES = RUNTIME_CFG.recommended_case_minutes
MIN_CASE_MINUTES = RUNTIME_CFG.min_case_minutes
STRICT_MODE = RUNTIME_CFG.strict_mode
WINDOW_SAMPLES = int(WINDOW_MINUTES * 60 * RUNTIME_CFG.fs_hz)
STRIDE_SAMPLES = int(STRIDE_MINUTES * 60 * RUNTIME_CFG.fs_hz)
SAFE_WARNING_ALLOWLIST = [
    # Add allowlisted warnings here if they are proven safe.
]


def load_threshold() -> float:
    if CONFIG_PATH.exists():
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        return float(cfg.get("threshold", 0.5))
    return 0.5


def load_logic_cfg() -> SmartLogicConfig:
    if LOGIC_CONFIG_PATH.exists():
        cfg = yaml.safe_load(LOGIC_CONFIG_PATH.read_text())
        return SmartLogicConfig(
            t_high=float(cfg.get("t_high", 0.65)),
            t_low=float(cfg.get("t_low", 0.35)),
            q_min=float(cfg.get("q_min", 0.7)),
            suspicious_rule=float(cfg.get("suspicious_rule", 0.5)),
            pathological_rule=float(cfg.get("pathological_rule", 0.8)),
        )
    return SmartLogicConfig()


def load_gauntlet() -> pd.DataFrame:
    frames = []
    for name in ["synthetic_logic_check.csv", "synthetic_noise_check.csv", "synthetic_edge_cases.csv"]:
        path = GAUNTLET_DIR / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("Gauntlet CSVs not found. Run generate_gauntlet.py first.")
    return pd.concat(frames, ignore_index=True)


def compute_rule_score(decelerations, tachysystole, sinusoidal, variability) -> float:
    score = 0.0
    if decelerations:
        severities = [getattr(d, "severity", None) for d in decelerations]
        severe_count = sum(1 for s in severities if s and s.name == "SEVERE")
        score += 0.5 if severe_count else 0.3
    if tachysystole and getattr(tachysystole, "is_present", False):
        score += 0.2
    if sinusoidal and getattr(sinusoidal, "is_present", False):
        score += 0.2
    if variability and getattr(variability, "value", None) is not None and variability.value < 3:
        score += 0.1
    return float(min(score, 1.0))


def assess_quality(fhr: np.ndarray) -> float:
    finite = np.isfinite(fhr)
    if len(fhr) == 0:
        return 0.0
    return float(np.mean(finite))


def window_iter(fhr: np.ndarray, uc: np.ndarray, context_samples: int):
    if len(fhr) < WINDOW_SAMPLES:
        return
    start_offset = max(0, context_samples - WINDOW_SAMPLES)
    for start in range(start_offset, len(fhr) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        end = start + WINDOW_SAMPLES
        yield start, end, fhr[start:end], uc[start:end]


def run_case(pipeline: AnalysisPipeline, row: Dict, ai_thr: float, logic_cfg: SmartLogicConfig):
    fhr = np.array(json.loads(row["fhr"]), dtype=float)
    uc = np.array(json.loads(row["uc"]), dtype=float)
    sampling_rate = float(row.get("sampling_rate", CTG.SAMPLING_RATE))
    case_id = row.get("case_id")
    set_case_context(case_id)
    try:
        if sampling_rate != RUNTIME_CFG.fs_hz:
            raise RuntimeError(
                f"STRICT_FS: {case_id} sampling_rate={sampling_rate} != runtime_fs={RUNTIME_CFG.fs_hz}"
            )
        assert_signal_length(fhr, sampling_rate, MIN_CASE_MINUTES, "RUNNER_RAW:FHR:MIN")
        assert_signal_length(uc, sampling_rate, MIN_CASE_MINUTES, "RUNNER_RAW:UC:MIN")
        duration_minutes = len(fhr) / sampling_rate / 60.0
        if duration_minutes < RECOMMENDED_CASE_MINUTES:
            raise RuntimeError(
                f"STRICT_DURATION: {case_id} duration {duration_minutes:.2f} min "
                f"< recommended_case {RECOMMENDED_CASE_MINUTES}"
            )
        assert_pair_aligned(fhr, uc, sampling_rate)
        assert_signal_length(fhr, sampling_rate, MIN_CASE_MINUTES, "PRE_SLICE:FHR:MIN")
        assert_signal_length(uc, sampling_rate, MIN_CASE_MINUTES, "PRE_SLICE:UC:MIN")

        c = pipeline.container
        quality_counts = {"LOW": 0, "MED": 0, "HIGH": 0}

        flagged_windows = 0
        max_ai = 0.0
        max_rule = 0.0
        quality_scores: List[float] = []
        ai_flagged_windows = 0

        context_samples = int(CTG.TACHYSYSTOLE_WINDOW_MINUTES * 60 * sampling_rate)
        for window_idx, (start, end, fhr_w, uc_w) in enumerate(window_iter(fhr, uc, context_samples)):
            set_window_context(window_idx)
            assert_signal_length(fhr_w, sampling_rate, MIN_WINDOW_MINUTES, "WINDOW:FHR:MIN")
            assert_signal_length(uc_w, sampling_rate, MIN_WINDOW_MINUTES, "WINDOW:UC:MIN")

            quality_diag = compute_signal_quality(
                fhr_w,
                raw_uc_window=uc_w,
                fs=sampling_rate,
                return_metrics=True,
            )
            quality_class = quality_diag["quality_class"]
            quality_counts[quality_class] += 1
            q_value = 0.95 if quality_class == "HIGH" else (0.75 if quality_class == "MED" else 0.3)
            quality_scores.append(q_value)

            uc_ctx = uc[end - context_samples:end] if context_samples > 0 else uc[:end]
            assert_signal_length(uc_ctx, sampling_rate, MIN_WINDOW_MINUTES, "UC_CONTEXT:MIN_WINDOW")
            if len(uc_ctx) < context_samples:
                raise ValueError(
                    f"CRITICAL: Tachysystole context too short for {case_id}: "
                    f"len={len(uc_ctx)} expected={context_samples}"
                )

            preprocess_result = c.preprocessor.process(fhr_w.copy())
            fhr_clean = preprocess_result.processed_signal

            baseline = c.baseline_calculator.calculate(fhr_clean, sampling_rate)
            variability = c.variability_calculator.calculate(fhr_clean, sampling_rate)
            decelerations = c.deceleration_detector.detect(fhr_clean, uc_w, baseline.value, sampling_rate)
            tachysystole = c.tachysystole_detector.detect(uc_ctx, sampling_rate)
            sinusoidal = c.sinusoidal_detector.detect(fhr_clean, sampling_rate)

            embedding_result = c.feature_extractor.extract(fhr_clean)
            feature_vector = c.feature_fusion.fuse(
                embedding=embedding_result.embedding,
                baseline=baseline,
                variability=variability,
                decelerations=decelerations,
                tachysystole=tachysystole,
                sinusoidal=sinusoidal,
            )
            X = feature_vector.vector.reshape(1, -1)
            proba = c.classifier.predict_proba(X)[0]
            ai_score = float(np.max(proba))
            if quality_class == "LOW":
                ai_score = 0.0
            max_ai = max(max_ai, ai_score)

            rule_score = compute_rule_score(decelerations, tachysystole, sinusoidal, variability)
            baseline_normal = (
                baseline.value is not None
                and THRESHOLDS.BASELINE_NORMAL_MIN <= baseline.value <= THRESHOLDS.BASELINE_NORMAL_MAX
            )
            variability_normal = (
                variability.value is not None
                and THRESHOLDS.VARIABILITY_MODERATE_MIN <= variability.value <= THRESHOLDS.VARIABILITY_MODERATE_MAX
            )
            boredom_block = (
                rule_score == 0.0
                and quality_class == "HIGH"
                and baseline_normal
                and variability_normal
            )
            max_rule = max(max_rule, rule_score)

            res = evaluate(
                ai_score,
                rule_score,
                logic_cfg,
                quality_class=quality_class,
                signal_quality=q_value,
                hard_low=bool(quality_diag.get("hard_low", False)),
                boredom_block=boredom_block,
            )
            if res.alert:
                flagged_windows += 1
                if ai_score >= ai_thr:
                    ai_flagged_windows += 1
        if not quality_scores:
            raise RuntimeError(f"STRICT_WINDOWING: {case_id} produced zero windows")

        signal_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
        final_decision = int(flagged_windows >= 2)
        transient_fp = int(final_decision == 1 and flagged_windows < 2)

        return {
            "case_id": row["case_id"],
            "case_type": row["case_type"],
            "noise_type": row.get("noise_type", "none"),
            "true_label": int(row.get("true_label", 0)),
            "ai_score": max_ai,
            "rule_score": max_rule,
            "signal_quality": signal_quality,
            "flagged_windows": flagged_windows,
            "ai_flagged_windows": ai_flagged_windows,
            "final_decision": final_decision,
            "transient_fp": transient_fp,
            "quality_low_windows": quality_counts["LOW"],
            "quality_med_windows": quality_counts["MED"],
            "quality_high_windows": quality_counts["HIGH"],
            "quality_class": "LOW" if quality_counts["LOW"] > 0 else ("MED" if quality_counts["MED"] > 0 else "HIGH"),
        }
    finally:
        set_window_context(None)
        set_case_context(None)


def analyze_fp(df: pd.DataFrame):
    if "quality_class" not in df.columns:
        df["quality_class"] = "UNKNOWN"
    fp = df[(df["true_label"] == 0) & (df["final_decision"] == 1)]
    print("FP rate by noise_type:")
    total_by_noise = df.groupby("noise_type").size()
    fp_rate = fp.groupby("noise_type").size() / total_by_noise
    print(fp_rate.fillna(0))

    print("FP rate by quality_class:")
    total_by_quality = df.groupby("quality_class").size()
    fp_by_quality = fp.groupby("quality_class").size().reindex(total_by_quality.index, fill_value=0)
    print(fp_by_quality)

    if fp.empty:
        print("No false positives detected on Gauntlet.")
        return

    transient = fp[fp["flagged_windows"] < 2]
    print("FP where AI spikes <2 windows (transient):", transient[["case_id", "case_type", "noise_type", "flagged_windows", "ai_score", "rule_score"]])

    high_ai_low_rule = df[(df["ai_score"] >= 0.7) & (df["rule_score"] < 0.3)]
    print("Cases where AI high but Rule low:", high_ai_low_rule[["case_id", "case_type", "noise_type", "ai_score", "rule_score"]])


def main():
    apply_strict_warnings(STRICT_MODE, SAFE_WARNING_ALLOWLIST)
    reset_fallback_audit()
    ai_thr = load_threshold()
    logic_cfg = load_logic_cfg()
    gauntlet = load_gauntlet()

    # Pre-flight enforcement: ensure all signals are at least 30 minutes and lengths match.
    fhr_durations = []
    uc_durations = []
    for _, row in gauntlet.iterrows():
        fhr = json.loads(row["fhr"])
        uc = json.loads(row["uc"])
        fs = float(row.get("sampling_rate", CTG.SAMPLING_RATE))
        if fs != RUNTIME_CFG.fs_hz:
            raise RuntimeError(
                f"STRICT_FS: {row.get('case_id')} sampling_rate={fs} != runtime_fs={RUNTIME_CFG.fs_hz}"
            )
        assert_signal_length(fhr, fs, MIN_CASE_MINUTES, "RUNNER_PREFLIGHT:FHR:MIN")
        assert_signal_length(uc, fs, MIN_CASE_MINUTES, "RUNNER_PREFLIGHT:UC:MIN")
        duration_minutes = len(fhr) / fs / 60.0
        if duration_minutes < RECOMMENDED_CASE_MINUTES:
            raise RuntimeError(
                f"STRICT_DURATION: {row.get('case_id')} duration {duration_minutes:.2f} min "
                f"< recommended_case {RECOMMENDED_CASE_MINUTES}"
            )
        assert_pair_aligned(fhr, uc, fs)
        fhr_durations.append(len(fhr) / fs / 60.0)
        uc_durations.append(len(uc) / fs / 60.0)

    def _stats(vals):
        return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))

    min_fhr_d, mean_fhr_d, max_fhr_d = _stats(fhr_durations)
    min_uc_d, mean_uc_d, max_uc_d = _stats(uc_durations)
    print(f"Gauntlet cases loaded: {len(gauntlet)}")
    print(f"FHR duration minutes (min/mean/max): {min_fhr_d:.2f} / {mean_fhr_d:.2f} / {max_fhr_d:.2f}")
    print(f"UC  duration minutes (min/mean/max): {min_uc_d:.2f} / {mean_uc_d:.2f} / {max_uc_d:.2f}")
    print("Skipped cases due to short signal: 0")
    print(f"STRICT_MODE: {STRICT_MODE}")
    print(
        f"Window config: window={WINDOW_MINUTES} min, stride={STRIDE_MINUTES} min, "
        f"min_window={MIN_WINDOW_MINUTES} min"
    )

    container = PipelineContainer(
        preprocessor=PreprocessorAdapter(),
        baseline_calculator=BaselineAdapter(),
        variability_calculator=VariabilityAdapter(),
        deceleration_detector=DecelerationAdapter(),
        tachysystole_detector=TachysystoleAdapter(),
        sinusoidal_detector=SinusoidalAdapter(),
        feature_extractor=MiniRocketAdapter(model_path="models/minirocket_encoder.joblib", auto_fit=False),
        feature_fusion=FusionAdapter(),
        classifier=ClassifierAdapter(model_path="models/sentinel_classifier.json"),
        medical_override=OverrideAdapter(),
        alert_generator=AlertAdapter(),
    )
    pipeline = AnalysisPipeline(container)

    rows: List[Dict] = []
    for _, row in gauntlet.iterrows():
        rows.append(run_case(pipeline, row, ai_thr, logic_cfg))

    fallback_records = get_fallback_records()
    print(f"fallback_count: {len(fallback_records)}")
    if fallback_records:
        audit_path = GAUNTLET_DIR / "gauntlet_fallback_audit.csv"
        write_fallback_audit(audit_path)
        fb_df = pd.DataFrame(fallback_records)
        print("Fallback summary by module:")
        print(fb_df.groupby(["module", "reason"]).size())
        print("Fallback details:")
        print(fb_df)
        raise_if_any_fallback(STRICT_MODE)
        raise RuntimeError(f"CRITICAL: Fallbacks detected. See {audit_path}")

    df = pd.DataFrame(rows)
    df["is_false_positive"] = ((df["true_label"] == 0) & (df["final_decision"] == 1)).astype(int)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    fp = df[df["is_false_positive"] == 1]
    fp.to_csv(HARD_NEG_PATH, index=False)

    analyze_fp(df)
    total_low = int(df["quality_low_windows"].sum())
    total_med = int(df["quality_med_windows"].sum())
    total_high = int(df["quality_high_windows"].sum())
    print(f"Window quality totals: LOW={total_low} MED={total_med} HIGH={total_high}")
    print(f"Saved analysis to {OUTPUT_PATH}")
    print(f"Saved hard negatives to {HARD_NEG_PATH}")


if __name__ == "__main__":
    main()
