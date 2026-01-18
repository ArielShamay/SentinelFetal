# SentinelFetal Master Documentation

## 1. System Overview (Hybrid AI)
- **Inputs:** CTG (FHR + UC) at 4 Hz.
- **Rule Engine:** Baseline, variability, decelerations (late/variable/prolonged), tachysystole, sinusoidal.
- **MOMENT Path:** 10-minute window embeddings (1024-dim), real or mock.
- **Fusion & Classifier:** Rule features (11-dim) + MOMENT (1024) → 1035-dim → XGBoost classifier.
- **Safety Net:** Medical overrides enforce critical findings (sinusoidal, bradycardia, recurrent late decels) regardless of ML output.
- **UI & Simulation:** Streamlit dashboards with an 8-patient orchestrator, synthetic generators, and event injection.

## 2. Directory Tree (abridged)
```
SentinelFetal/
├─ README.md
├─ SENTINEL_FETAL_MASTER_DOC.md
├─ DEMO_SCRIPT.md
├─ DEMO_CHEAT_SHEET.md
├─ requirements.txt
├─ archive/
│  └─ docs/        # Archived PRDs, specs, evaluation reports
├─ data/           # CTU-UHB and processed features
├─ models/         # xgb_demo.json + config
├─ scripts/        # run_simulation.py, verify_system.py, visualize_preprocessing.py
├─ src/
│  ├─ config.py    # CTG, thresholds, colors, model paths, Hebrew strings
│  ├─ pipeline/    # container.py (DI), analysis_pipeline.py
│  ├─ analysis/    # alerts.py (Hebrew XAI), override.py (safety net)
│  ├─ rules/       # baseline.py, variability.py, decelerations.py, tachysystole.py, sinusoidal.py
│  ├─ simulation/  # core/orchestrator.py, generators/*, events/event_types.py, processing/pipeline_adapter.py
│  ├─ models/      # moment_encoder.py (real/mock), fusion.py, classifier.py
│  ├─ ui/          # app.py, simulation_app.py, plots.py
│  └─ utils/       # signal_utils.py
└─ tests/          # unit/integration + benchmarks (accuracy, moment, load, clinical)
```

## 3. Module Deep-Dive (API Reference)

### Data & Preprocessing
- **data/loader.py**: `CTUDataLoader` lists/loads CTU-UHB records; exposes `CTGRecord` with FHR/UC, timestamps, duration, pH-based labels.
- **data/preprocess.py**: `CTGPreprocessor.process(fhr, apply_smoothing=False)`
  - Steps: out-of-range → NaN (50–240 bpm), spike removal (>30 bpm), gap fill ≤10s (linear), leave >10s, optional smoothing.
  - Config: `PreprocessingConfig` (sampling_rate 4 Hz, max_gap_seconds 10, spike_threshold 30, smoothing_window 5).

### Clinical Rules (rules/)
- **baseline.py**: `calculate_baseline(fhr, sampling_rate)` finds lowest-variability 2-minute segment (step 10s, variability<25 bpm), outputs `BaselineResult` (value, brady/tachy flags, confidence, stable_segment_found).
- **variability.py**: `calculate_variability(fhr, sampling_rate)` 1-minute windows (50% overlap), classifies Absent/Minimal/Moderate/Marked via thresholds (Absent ≤2 bpm, Minimal ≤5 bpm, Moderate ≤25 bpm).
- **decelerations.py**: Detects late, variable, prolonged decels; classifies `DecelerationType`; computes onset/nadir/recovery timing; flags severity (depth, drops_below_70, slow_recovery, overshoot, biphasic).
- **tachysystole.py**: Detects >5 contractions per 10 minutes; outputs `TachysystoleResult` with contraction rate and flag.
- **sinusoidal.py**: FFT-based detection over last ≥20 minutes; freq 3–5 cycles/min, amplitude 5–15 bpm, dominance ratio ≥0.3 → `SinusoidalResult(detected=True)` (forces Category 3).

### AI Models
- **models/moment_encoder.py**: `MomentFeatureExtractor` (real) loads MOMENT encoder; `MockMomentFeatureExtractor` returns zeros for speed. Consumes 10-minute window (2400 samples) → 1024-dim embedding.
- **models/fusion.py**: `FeatureFusion.fuse` merges embedding + rule features into 1035-dim vector.
- **models/classifier.py**: `XGBClassifierWrapper` loads `models/xgb_demo.json`; exposes `predict`, `predict_proba` on fused vectors.

### Analysis & Alerts
- **analysis/analysis_pipeline.py**: Orchestrates preprocess → rules → MOMENT → fusion → XGB → overrides → alert generation. Entry: `AnalysisPipeline.analyze(fhr, uc, sampling_rate=4.0)` returns `AnalysisResult` (category 1/2/3, confidence, findings).
- **analysis/alerts.py**: Generates Hebrew XAI: headlines, rationale per finding (late/variable decels, tachysystole, sinusoidal, baseline shifts).
- **analysis/override.py**: Safety rules (see Clinical Logic). Elevates Category: sinusoidal → 3; bradycardia <110 bpm → 2; recurrent late decels (≥3) → 2; absent variability + ominous decels → 3; absent variability safety floor when ML says Cat 1 → Cat 2.

### Simulation (Real-Time)
- **simulation/core/orchestrator.py**: Manages N patients (default 8), tick interval 1s, staggered MOMENT scheduling. Generates data, routes through `processing_callback`, tracks tick counts.
- **simulation/generators/**: `PatientGenerator` with FHR/UC generators, ring buffer, event injection; FHR generator supports decel patterns, baseline shifts, sinusoidal; UC generator controls contraction rate.
- **simulation/events/event_types.py**: Parameter factories for late/variable/prolonged decels, brady/tachy, variability changes, sinusoidal, tachysystole.
- **simulation/processing/pipeline_adapter.py**: Bridges orchestrator buffers to `AnalysisPipeline` (real or mock MOMENT).
- **ui/simulation_app.py**: Streamlit multi-patient dashboard with injectors and trend plots.

## 4. Clinical Logic Specification

### Thresholds (config.py → THRESHOLDS)
- Baseline normal: 110–160 bpm; bradycardia: <110; tachycardia: >160.
- Variability: Absent ≤2 bpm; Minimal 3–5 bpm; Moderate 6–25 bpm; Marked >25 bpm.
- Tachysystole: >5 contractions per 10 minutes.
- Deceleration depth examples (event factories): mild late depth 22 bpm, moderate 35 bpm, severe 55 bpm; variable mild 25 bpm (fast descent), severe 60+ bpm with drops <70 bpm.
- Sinusoidal detection: 20-minute window min, amplitude 5–15 bpm, freq 3–5 cycles/min.

### Safety Override Rules (analysis/override.py)
- Sinusoidal detected → Category 3.
- Absent variability + (recurrent late OR recurrent variable OR bradycardia) → Category 3.
- Bradycardia (<110 bpm for ≥10 min) → Category 2.
- Recurrent late decelerations (≥3) → Category 2.
- ML predicts Category 1 but variability is Absent → Category 2 (safety floor).

### Hebrew Alert Logic (analysis/alerts.py)
- Headlines localized to Hebrew per finding (late/variable decels, tachysystole, sinusoidal, baseline shifts, variability concerns).
- Alert body includes category, confidence, and contributing findings; severe patterns (sinusoidal) explicitly labeled כפתולוגי (Pathological).

## 5. Performance & Validation (Phase C)
- Clinical benchmark (tests/benchmarks/benchmark_clinical.py, mock MOMENT):
  - Sinusoidal → Category 3 (pass)
  - Late Severe (+ absent variability) → Category 2 via override (pass)
  - Bradycardia (+ absent variability) → Category 2 via override (pass)
  - Normal → Category 1 (pass)
- Rule accuracy (Phase B, synthetic injections, mock MOMENT): sinusoidal 1.0; brady 1.0; tachy 1.0; prolonged 1.0; variable mild/mod/severe 0.90/0.95/1.0; late moderate 0.75; late severe 1.0; late mild low (0.05).
- Load test (mock processing): tick ≈0.984 Hz, worst tick ≈1.016s, lag flag cleared with 20ms tolerance (1/2/4/8 patients), CPU_mean ~19–24%.
- MOMENT CPU benchmark (real encoder): ~1.87s per 10-min window; throughput ~0.53 windows/s; +1.3 GB RAM; init ~11s.

## 6. Setup & Usage
```bash
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=.
```
- **Run simulation dashboard:** `py scripts/run_simulation.py`
- **Run main Streamlit app:** `streamlit run src/ui/app.py`
- **Verify system:** `py scripts/verify_system.py`
- **Benchmarks:**
  - Accuracy: `py tests/benchmarks/benchmark_accuracy.py`
  - Clinical categories: `py tests/benchmarks/benchmark_clinical.py`
  - Load: `py tests/benchmarks/benchmark_load.py`

## 7. Training & Data
- Data: CTU-UHB recordings under `data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/`.
- Processed features: `data/processed/X.npy`, `y.npy` (3240 × 1035).
- Training script: `src/training/train_demo.py` (uses XGBoost; config in `models/xgb_demo.config.json`).

## 8. Simulation Scenarios (quick reference)
- Sinusoidal: Inject `SinusoidalParams(severe, duration ~5–35 min)` → Cat 3 via override.
- Late severe: Inject `LateDecelerationParams.severe()`; recurrent events + absent variability will elevate to Cat 2.
- Bradycardia: Inject `BradycardiaParams(target_fhr≈80, duration≥15 min)` → Cat 2 via override.
- Normal: No injections; expect Cat 1, moderate variability.