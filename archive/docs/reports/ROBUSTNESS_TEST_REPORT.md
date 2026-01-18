---
title: Phase 11 – Robustness Torture Test
date: 2026-01-18
---

# Robustness Test Report

This benchmark injects **Gaussian noise**, **signal dropouts** (NaN), and **artifact spikes** into clean pathological patterns, then measures whether the pipeline still detects them.

## Configuration

- **Window**: 10 min @ 4.0 Hz
- **Patterns**: Sinusoidal, Late Decel (severe), Variable Decel (severe)
- **Noise σ (bpm)**: [0.0, 2.0, 5.0, 10.0]
- **Dropout rates**: [0.0, 0.02, 0.05]
- **Artifact rates**: [0.0, 0.01, 0.02]
- **Total scenarios**: 108

## Detection Rate by Noise Level (all patterns combined)

| Noise σ (bpm) | Detection Rate |
|---------------|----------------|
| 0.0 | 11% |
| 2.0 | 11% |
| 5.0 | 4% |
| 10.0 | 7% |

## Detection Rate by Dropout Rate

| Dropout Rate | Detection Rate |
|--------------|----------------|
| 0% | 6% |
| 2% | 11% |
| 5% | 8% |

## Detection Rate by Artifact Rate

| Artifact Rate | Detection Rate |
|---------------|----------------|
| 0% | 14% |
| 1% | 6% |
| 2% | 6% |

## Detection Rate per Pattern

| Pattern | Clean | σ=2 | σ=5 | σ=10 |
|---------|-------|-----|-----|------|
| Sinusoidal | 0% | 0% | 0% | 0% |
| Late Decel (severe) | 22% | 11% | 0% | 0% |
| Variable Decel (severe) | 11% | 22% | 11% | 22% |

## Detailed Scenario Results

| Pattern | Noise σ | Dropout | Artifact | Expected | Predicted | Detected |
|---------|---------|---------|----------|----------|-----------|----------|
| Sinusoidal | 0.0 | 0% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 0.0 | 0% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 0.0 | 0% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 0.0 | 2% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 0.0 | 2% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 0.0 | 2% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 0.0 | 5% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 0.0 | 5% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 0.0 | 5% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 2.0 | 0% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 2.0 | 0% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 2.0 | 0% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 2.0 | 2% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 2.0 | 2% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 2.0 | 2% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 2.0 | 5% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 2.0 | 5% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 2.0 | 5% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 5.0 | 0% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 5.0 | 0% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 5.0 | 0% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 5.0 | 2% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 5.0 | 2% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 5.0 | 2% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 5.0 | 5% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 5.0 | 5% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 5.0 | 5% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 10.0 | 0% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 10.0 | 0% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 10.0 | 0% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 10.0 | 2% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 10.0 | 2% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 10.0 | 2% | 2% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 10.0 | 5% | 0% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 10.0 | 5% | 1% | Cat3 | Cat1 | ❌ |
| Sinusoidal | 10.0 | 5% | 2% | Cat3 | Cat1 | ❌ |
| Late Decel (severe) | 0.0 | 0% | 0% | Cat2 | Cat2 | ✅ |
| Late Decel (severe) | 0.0 | 0% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 0.0 | 0% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 0.0 | 2% | 0% | Cat2 | Cat2 | ✅ |
| Late Decel (severe) | 0.0 | 2% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 0.0 | 2% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 0.0 | 5% | 0% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 0.0 | 5% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 0.0 | 5% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 2.0 | 0% | 0% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 2.0 | 0% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 2.0 | 0% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 2.0 | 2% | 0% | Cat2 | Cat2 | ✅ |
| Late Decel (severe) | 2.0 | 2% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 2.0 | 2% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 2.0 | 5% | 0% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 2.0 | 5% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 2.0 | 5% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 5.0 | 0% | 0% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 5.0 | 0% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 5.0 | 0% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 5.0 | 2% | 0% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 5.0 | 2% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 5.0 | 2% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 5.0 | 5% | 0% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 5.0 | 5% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 5.0 | 5% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 10.0 | 0% | 0% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 10.0 | 0% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 10.0 | 0% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 10.0 | 2% | 0% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 10.0 | 2% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 10.0 | 2% | 2% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 10.0 | 5% | 0% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 10.0 | 5% | 1% | Cat2 | Cat1 | ❌ |
| Late Decel (severe) | 10.0 | 5% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 0.0 | 0% | 0% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 0.0 | 0% | 1% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 0.0 | 0% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 0.0 | 2% | 0% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 0.0 | 2% | 1% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 0.0 | 2% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 0.0 | 5% | 0% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 0.0 | 5% | 1% | Cat2 | Cat2 | ✅ |
| Variable Decel (severe) | 0.0 | 5% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 2.0 | 0% | 0% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 2.0 | 0% | 1% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 2.0 | 0% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 2.0 | 2% | 0% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 2.0 | 2% | 1% | Cat2 | Cat2 | ✅ |
| Variable Decel (severe) | 2.0 | 2% | 2% | Cat2 | Cat2 | ✅ |
| Variable Decel (severe) | 2.0 | 5% | 0% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 2.0 | 5% | 1% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 2.0 | 5% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 5.0 | 0% | 0% | Cat2 | Cat2 | ✅ |
| Variable Decel (severe) | 5.0 | 0% | 1% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 5.0 | 0% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 5.0 | 2% | 0% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 5.0 | 2% | 1% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 5.0 | 2% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 5.0 | 5% | 0% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 5.0 | 5% | 1% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 5.0 | 5% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 10.0 | 0% | 0% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 10.0 | 0% | 1% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 10.0 | 0% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 10.0 | 2% | 0% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 10.0 | 2% | 1% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 10.0 | 2% | 2% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 10.0 | 5% | 0% | Cat2 | Cat2 | ✅ |
| Variable Decel (severe) | 10.0 | 5% | 1% | Cat2 | Cat1 | ❌ |
| Variable Decel (severe) | 10.0 | 5% | 2% | Cat2 | Cat2 | ✅ |
