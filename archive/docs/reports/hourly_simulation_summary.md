---
title: Phase 9 – Hourly Batch Simulation (4 Patients)
date: 2026-01-18
description: Fast-forward offline simulation with scheduled event injections across 4 patients.
---

# Phase 9: Hourly Simulation Summary

## Run configuration
- Script: [tests/benchmarks/benchmark_hourly_simulation.py](tests/benchmarks/benchmark_hourly_simulation.py)
- Output: [tests/benchmarks/results/hourly_simulation_results.json](tests/benchmarks/results/hourly_simulation_results.json)
- Duration: 3600 s; sampling: 4 Hz; analysis window: 600 s; hop: 120 s (26 windows/patient)
- Injected events: sinusoidal (900s, 10 min), late decelerations (1200s & 2400s, 5 min each), variable decel (1800s, 5 min), bradycardia (3000s, 15 min)

## Results by patient
- P1_control: Cat1 in all 26/26 windows; no events injected.
- P2_sinusoidal: Cat1 in all 26/26 windows; sinusoidal not detected (detector requires ≥20 min window vs provided 10 min segment).
- P3_late: Cat1 19/26, Cat2 7/26. Cat2 windows align with recurrent late deceleration overrides around 1080–1560s and 1920–2760s.
- P4_variable_brady: Cat1 23/26, Cat2 3/26. Cat2 triggered following bradycardia injection (2760–3000s windows); variable decel at 1800s did not flip category.

## Notable observations
- Sinusoidal detector never triggered because current analyzer demands a ≥20 min segment while only 10 min was injected; extend the injected sinusoidal length or reduce the detector requirement to validate detection.
- Multiple warnings during the run: UC signal length (10 min) shorter than the 30 min analysis window; sinusoidal check windows too short. Consider aligning signal/analysis window lengths to reduce noise in logs.
- Overrides worked as expected: recurrent late decelerations and bradycardia elevated categories to Cat2 in affected windows; no Cat3 states observed.

## Follow-ups
1) Rerun with longer sinusoidal injection or shorter sinusoidal detection window to confirm sensitivity.
2) Align UC window length with analyzer window (or reduce analyzer window) to avoid repeated short-signal warnings.
3) If needed, export per-window plot snapshots for Cat2 intervals around late decels (P3) and brady (P4).
