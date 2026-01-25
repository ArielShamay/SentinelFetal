import numpy as np

from src.ml.features.clinical import compute_clinical_features, compute_rule_score


def test_low_quality_defaults():
    fhr = np.array([np.nan, np.nan])
    feats = compute_clinical_features(fhr, None, fs=4.0, start_time_min=0.0)
    assert feats["baseline_fhr"] == 0.0
    assert feats["signal_quality"] == 0.0


def test_basic_metrics_and_uc_rate():
    fhr = np.array([140, 142, 138, 141, 139], dtype=float)
    uc = np.concatenate([np.zeros(20), np.ones(40) * 20, np.zeros(20)])
    feats = compute_clinical_features(fhr, uc, fs=4.0, start_time_min=5.0)
    assert feats["baseline_fhr"] == 140.0
    assert feats["stv_proxy"] > 0
    assert feats["uc_contractions"] >= 1
    assert feats["uc_rate"] > 0
    assert feats["time_since_start_min"] == 5.0


def test_rule_score_levels():
    assert compute_rule_score(100, 1.0, 1.0) == 2
    assert compute_rule_score(118, 4.0, 6.0) == 1
    assert compute_rule_score(140, 6.0, 10.0) == 0
