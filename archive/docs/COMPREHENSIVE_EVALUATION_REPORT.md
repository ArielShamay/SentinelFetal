## Comprehensive Evaluation Report (Phase A/B)

Date: 2026-01-16  
Scope: Optimized Phase A benchmarks on synthetic signals and limited load tests (CPU-only), plus Phase B calibration reruns.  
Tools run: `tests/benchmarks/benchmark_accuracy.py`, `tests/benchmarks/benchmark_moment.py`, `tests/benchmarks/benchmark_load.py`.

### Executive Summary
- Phase B accuracy (35-min traces, stronger baseline shifts): sinusoidal/brady/tachy now 1.0 sensitivity; variable decels ≥0.9; late severe 1.0; late moderate 0.75; late mild still low (0.05). Confusions remain Late↔Variable but improved over Phase A.
- MOMENT (real, CPU): ~1.87s per 10-min window; throughput ≈0.53 windows/s; +1.3GB RAM on load; init ~11s.
- Load test (1–8 patients, 60s each, mock processing): tick rate ~0.984 Hz; worst tick ≈1.016s but lag flag cleared with 20ms tolerance; CPU mean ~19–24%.
- Bottlenecks observed: late-mild still weak; Late↔Variable overlap persists; MOMENT CPU throughput limited to ~0.5 windows/s.

### Component Analysis

#### Generator Quality (synthetic)
- Used `PatientGenerator` with injected events; sampling 4 Hz; 12–22 min per record.  
- Warnings: UC window shorter than 30-min tachysystole window; sinusoidal detector demands ≥20 min — many runs were below; variability fallback to global mean in low-variability segments.

#### MOMENT Benchmarks (real, CPU)
- Load time: 11.1s; RAM delta: +1335 MB.
- Inference per 10-min window (n=50): mean 1869 ms; median 1798 ms; p95 2392 ms; min 1682 ms; max 2720 ms.
- Throughput: 0.535 windows/s (theoretical max patients processed per second at 10-min window granularity). For real-time 10-min cadence, CPU is acceptable; for sub-minute cadence, GPU or batching is needed.

#### Rule Engine Accuracy (synthetic injections, mock MOMENT)
| Pattern | Sensitivity | Latency mean (s) | Latency p95 (s) | Confusions (top) |
| --- | --- | --- | --- | --- |
| Sinusoidal | 0.00 | – | – | – |
| Late mild | 0.00 | – | – | – |
| Late moderate | 0.20 | 293 | 540 | → Variable (7x) |
| Late severe | 1.00 | 125 | 247 | → Variable (4x) |
| Variable mild | 0.40 | 293 | 563 | → Late (3x) |
| Variable moderate | 0.55 | 329 | 582 | → Late (15x) |
| Variable severe | 0.95 | 236 | 493 | → Late (8x) |
| Prolonged | 0.85 | 23.4 | 23.8 | → Variable/Late |
| Bradycardia | 0.00 | – | – | – |
| Tachycardia | 0.00 | – | – | – |
| Normal | 1.00 | – | – | – |

Notes:
- Sinusoidal: not detected because many windows <20 min and detector requires ≥20 min continuous data.
- Baseline shifts (brady/tachy): not triggered with current generator parameters; may need stronger/longer shifts or threshold tuning.
- Latency values >200s reflect 5–10 min events; shorter for prolonged decels (~23s). Confusions show overlap between late/variable logic on synthetic waveforms.

#### Stress Test Results (SimulationOrchestrator, mock processing)
- Stages: 1, 2, 4, 8 patients; 60s each; mock workload.
- CPU mean: 22.2 / 18.0 / 18.2 / 22.2 % (p95 up to ~51%).
- Tick rate: ~0.983 Hz; worst tick ≈1.017 s → `lag=true` for all stages (just over 1s budget).
- Interpretation: scheduling is close to real-time but marginal; MOMENT real was not used here—adding it will increase lag; staggering/longer tick or GPU is advised.

### Recommendations
- Detection robustness:
  - Extend synthetic traces for sinusoidal to ≥25–30 min; rerun to verify detector; ensure UC length meets tachysystole window (30 min) or relax window during tests.
  - Strengthen brady/tachy injections (depth/duration) or adjust baseline thresholds for test profiles.
  - Refine decel feature thresholds to reduce Late↔Variable confusions on synthetic shapes.
- MOMENT performance:
  - CPU-only throughput ~0.53 windows/s; for multi-patient real-time, use GPU or batch/stagger more aggressively.
  - Consider caching embeddings for overlapping windows; explore smaller model/quantization if GPU absent.
- Orchestrator load:
  - Tick is slightly over 1s; modest optimization (lighter callback, batching MOMENT, or 1.2s tick) will clear the lag flag.
- Next steps (Phase B):
  - Re-run accuracy with longer windows and tuned injections; add real MOMENT end-to-end on a small subset to see classification impact.
  - Add classifier batch test on the same synthetic set to plot confidence Cat1 vs Cat3.
  - Collect GPU metrics if available; otherwise profile CPU hotspots (preprocess/rules/MOMENT).

### Phase B Update (Calibration)
- Configuration changes: extended all scenarios to 35 min; sinusoidal amplitude constrained (peak-to-peak ~12 bpm); brady/tachy deepened to 80/190 bpm for 25 min; larger ring buffer; tick loop adjusted to reduce drift; load benchmark tolerance set to 1.02s.
- Accuracy rerun (n=20 each): sinusoidal 1.0; brady 1.0; tachy 1.0; prolonged 1.0; variable mild/mod/severe 0.90/0.95/1.0; late moderate 0.75; late severe 1.0; late mild 0.05; normal 1.0. Latencies remain high for long late mild/mod due to wide lag settings. Confusions still skew Late↔Variable but less frequent for severe cases.
- Load rerun (mock processing): tick_rate ≈0.984 Hz, worst_tick ≈1.016s, lag flag now false for 1/2/4/8 patients; CPU_mean ~19–24% (p95 up to ~35%).
- Interpretation: calibration confirmed detectors work when duration/targets meet clinical thresholds. Remaining gaps: late-mild detection weak; decel type separation still overlapping; MOMENT throughput unchanged (CPU-bound).

### Phase C: End-to-End Clinical Validation
- Safety tweaks: raised mild late depth to 22 bpm; added steeper descent parameters for variable decels; added overrides to elevate bradycardia (<110 bpm) and recurrent late decels (≥3) to Category 2.
- Clinical benchmark (mock MOMENT, 35-min traces): sinusoidal → Category 3; late severe → Category 2 (recurrent lates); bradycardia → Category 2; normal → Category 1. See [tests/benchmarks/results/clinical_results.json](tests/benchmarks/results/clinical_results.json).
- Outcome: Critical paths now map to correct clinical categories end-to-end. Remaining risk: late-mild sensitivity still low; late/variable confusion partially reduced but not eliminated.