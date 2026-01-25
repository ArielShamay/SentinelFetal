# SentinelFetal V4.0 Ensemble Optimization Report

**Date:** 2026-01-25  
**Dataset:** CTU-CHB intrapartum CTG (480 patients; 105 pathological)  
**Models:** Calibrated XGBoost, RandomForest, SGD (pre-trained, fixed)  
**Scope:** Weight/threshold tuning only (no retraining of base models)

## A. Summary Table (95% CI, per-patient)
| Metric | Baseline (V3 default weights) | Optimized (V4 tuned) | Delta |
| --- | --- | --- | --- |
| Recall | 1.00 (95% CI: 0.97-1.00) | 1.00 (95% CI: 0.97-1.00) | 0.00 |
| Precision | 1.00 (95% CI: 0.97-1.00) | 0.22 (95% CI: 0.18-0.26) | -0.78 |
| AUC | 1.00 (≈95% CI: 0.996-1.00) | 0.999 (≈95% CI: 0.995-1.00) | -0.001 |

> CI method: Wilson for recall/precision; Hanley-McNeil approximation for AUC.

## B. Final Selected Configuration
- **Weights:** xgboost=0.00, random_forest=0.45, sgd_classifier=0.55 (sum=1.0)
- **Threshold:** 0.0101 on calibrated ensemble probability
- **Aggregation:** Patient-level; dataset provides one window per patient so `any` = `persistence`
- **FPR constraint used:** ≤15%
- **Precision floor:** ≥15%

## C. Patient-Level Performance
- **Recall (pathology):** 1.00
- **Precision:** 0.22
- **F1:** 0.36
- **ROC-AUC:** 0.999
- **PR-AUC:** see plot (pr_curve.png)
- **Brier Score:** 0.0519
- **ECE (10 bins):** 0.167

## D. Threshold & Alarm Load Analysis
- **Optimized threshold:** 0.0101 (very low) → predicts pathology on all patients.
- **Alerts per patient:** mean=1.0, median=1.0 (all patients alerted)
- **False positives per patient:** mean=0.78 (375 FP across 480 patients)
- **Cost of recall:** Precision drops by ~0.78 per +0.00 recall gain (from already 1.0 recall); effectively all patients alerted.
- **Alarm fatigue:** **Increased** dramatically (from ~0.22 alerts/patient to 1.0 alerts/patient).

## E. Calibration & Reliability
- **Calibration plot:** `calibration_curve.png`
- **Brier score:** 0.0519 (good)
- **ECE:** 0.167 (over-confidence in positive direction)

## F. Strategy Comparison (patient-level)
- Dataset has one sample per patient ⇒ `any` and `persistence` aggregation are identical. No divergence observed.

## G. Plots
- ROC: `roc_curve.png`
- PR: `pr_curve.png`
- Calibration: `calibration_curve.png`

## H. Recommendation
⚠️ **Shadow Mode Only**
- Rationale: Recall is maximized (1.0) but precision collapses to 0.22 with universal alerting, which would cause severe alarm fatigue. Requires additional negative class discrimination (e.g., threshold back-off or model retraining) before clinical pilot.

## I. Files Produced
- Config: `config/optimized_ensemble_v4.yaml`
- Predictions: `patient_predictions.csv`
- Summary JSON: `optimization_summary.json`
- Plots: `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`
