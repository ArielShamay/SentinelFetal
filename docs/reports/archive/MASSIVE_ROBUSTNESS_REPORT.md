---
title: Phase 13 – Signal Processing Upgrade & Detector Tuning
date: 2025-01-19
description: 500+ randomized scenarios testing detection under noise, dropouts, and varied durations.
---

# Massive Robustness Test Report

**Phase 13 Improvements:**
- Savitzky-Golay filter (preserves peak shapes vs median filter)
- Fixed sinusoidal detection window (25 min for sinusoidal patterns)
- Improved deceleration classification (timing-based Late vs descent-rate-based Variable)

**Total Scenarios**: 500
**Patterns**: Sinusoidal, Late Decel, Variable Decel, Prolonged Decel, Bradycardia, Tachycardia, Normal
**Noise Levels (σ bpm)**: [0.0, 2.0, 5.0, 10.0, 15.0]
**Dropout Rates**: [0.0, 0.05, 0.1]
**Duration Range**: 30-60 min (≥25 min for Sinusoidal)

---

## Overall Summary

- **Overall Detection Rate**: 320/500 (64.0%)
- **False Positive Rate (Normal → Cat 2/3)**: 0/68 (0.0%)

## Detection Rate: Pattern × Noise Level

| Pattern | σ=0 | σ=2 | σ=5 | σ=10 | σ=15 |
|---------|-----|-----|-----|------|------|
| Sinusoidal | 100% | 0% | 0% | 0% | 0% |
| Late Decel | 23% | 5% | 0% | 0% | 0% |
| Variable Decel | 14% | 15% | 0% | 17% | 6% |
| Prolonged Decel | 100% | 100% | 100% | 100% | 100% |
| Bradycardia | 100% | 100% | 100% | 100% | 100% |
| Tachycardia | 100% | 100% | 100% | 100% | 100% |
| Normal | 100% | 100% | 100% | 100% | 100% |

## Detection Rate: Pattern × Dropout Rate

| Pattern | 0% | 5% | 10% |
|---------|-----|-----|------|
| Sinusoidal | 14% | 24% | 36% |
| Late Decel | 7% | 5% | 4% |
| Variable Decel | 18% | 0% | 10% |
| Prolonged Decel | 100% | 100% | 100% |
| Bradycardia | 100% | 100% | 100% |
| Tachycardia | 100% | 100% | 100% |
| Normal | 100% | 100% | 100% |

## Detection Rate: Pattern × Severity

| Pattern | Mild | Moderate | Severe |
|---------|------|----------|--------|
| Sinusoidal | 26% | 7% | 35% |
| Late Decel | 0% | 4% | 13% |
| Variable Decel | 0% | 5% | 31% |
| Prolonged Decel | 100% | 100% | 100% |
| Bradycardia | 100% | 100% | 100% |
| Tachycardia | 100% | 100% | 100% |
| Normal | 100% | 100% | 100% |

## Sinusoidal Detection (Clean Signals)

- Clean sinusoidal scenarios: 3
- Detected (Cat 3): 3/3 (100%)

## Predicted Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Cat 1 | 248 | 49.6% |
| Cat 2 | 236 | 47.2% |
| Cat 3 | 16 | 3.2% |

## Sample Failures (first 20)

| ID | Pattern | Severity | Noise | Dropout | Expected | Predicted |
|----|---------|----------|-------|---------|----------|-----------|
| 2 | Late Decel | mild | 2.0 | 0% | Cat2 | Cat1 |
| 3 | Variable Decel | moderate | 15.0 | 5% | Cat2 | Cat1 |
| 4 | Late Decel | severe | 15.0 | 10% | Cat2 | Cat1 |
| 5 | Sinusoidal | mild | 2.0 | 0% | Cat3 | Cat1 |
| 7 | Late Decel | moderate | 15.0 | 10% | Cat2 | Cat1 |
| 8 | Late Decel | severe | 2.0 | 5% | Cat2 | Cat1 |
| 11 | Variable Decel | mild | 5.0 | 0% | Cat2 | Cat1 |
| 14 | Late Decel | severe | 2.0 | 10% | Cat2 | Cat1 |
| 18 | Late Decel | severe | 15.0 | 0% | Cat2 | Cat1 |
| 20 | Sinusoidal | moderate | 10.0 | 5% | Cat3 | Cat1 |
| 22 | Sinusoidal | moderate | 5.0 | 5% | Cat3 | Cat1 |
| 23 | Late Decel | moderate | 15.0 | 0% | Cat2 | Cat1 |
| 25 | Variable Decel | moderate | 10.0 | 5% | Cat2 | Cat1 |
| 30 | Variable Decel | moderate | 15.0 | 0% | Cat2 | Cat1 |
| 34 | Sinusoidal | moderate | 15.0 | 0% | Cat3 | Cat1 |
| 35 | Variable Decel | mild | 15.0 | 0% | Cat2 | Cat1 |
| 36 | Variable Decel | severe | 15.0 | 5% | Cat2 | Cat1 |
| 37 | Variable Decel | mild | 10.0 | 10% | Cat2 | Cat1 |
| 44 | Late Decel | severe | 2.0 | 0% | Cat2 | Cat1 |
| 46 | Late Decel | mild | 10.0 | 5% | Cat2 | Cat1 |
