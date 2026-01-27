# PROJECT_CONTEXT_SNAPSHOT

Generated: 2026-01-27 14:51:07
OS: Microsoft Windows NT 10.0.26200.0
CWD: C:/Users/ariel/OneDrive/שולחן העבודה/SentinelFetal/SentinelFetal

Read-only mode: no files modified except this snapshot.

## 1) Environment
- python -V: Python 3.11.9
- pip --version: pip 24.0 from C:\Users\ariel\AppData\Local\Programs\Python\Python311\Lib\site-packages\pip (python 3.11)
- pip freeze: (empty output)

## 1.1) Requirements / Build Files
### requirements.txt
- Path: requirements.txt
- Size: 1371 bytes
- Mtime: 2026-01-25 22:01:49
- SHA256: A6581EE2C332180C67367EDE3FD519C17FDC9C48B7BFBF402F2DC19DDF3BA8AA
- Head (first 60 lines):

```text
# SentinelFetal Dependencies
# Updated: Phase 2 refactor - MiniRocket replaces MOMENT

# Data handling
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# PhysioNet data loading
wfdb>=4.1.0

# Visualization
matplotlib>=3.7.0
plotly>=5.14.0

# Testing
pytest>=7.3.0
pytest-cov>=4.1.0

# ML - Core (lightweight)
scikit-learn>=1.2.0
xgboost>=2.0.0  # XGBoost for classification
imbalanced-learn>=0.11.0  # SMOTE for handling class imbalance

# Time Series Classification - MiniRocket (replaces MOMENT)
# MiniRocket: 75x faster than ROCKET, 84 fixed kernels vs 341M params
sktime>=0.24.0
numba>=0.58.0  # Required by MiniRocket for JIT compilation

# Deep Learning (optional - InceptionTime alternative)
# tsai>=0.3.7  # Uncomment if deep learning approach needed

# Signal Processing
pywavelets>=1.4.0  # Coiflet 4 wavelet denoising

# ONNX Optimization
onnx>=1.14.0
onnxruntime>=1.15.0

# OpenVINO for Intel CPU optimization (optional but recommended)
# Provides ~3-5x speedup on Intel processors
# Uncomment if using Intel CPU:
# openvino>=2023.1.0

# Utilities
tqdm>=4.65.0
python-dateutil>=2.8.0
psutil>=5.9.0
joblib>=1.3.0  # Model serialization

# FastAPI Backend (V3 Architecture)
fastapi>=0.109.0
uvicorn[standard]>=0.25.0
websockets>=12.0
msgpack>=1.0.0
pydantic>=2.0.0
python-multipart>=0.0.6  # Form data support
```

### pyproject.toml
- Path: pyproject.toml
- Size: 1943 bytes
- Mtime: 2026-01-24 22:05:36
- SHA256: 2FA8BA96D745C49A5AB9E2B306DCDA2179DA818128778CCA476074DF34C39608
- Head (first 60 lines):

```text
[project]
name = "sentinelfetal"
version = "3.0.0"
description = "Real-time CTG monitoring with AI classification"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}

dependencies = [
    # Existing dependencies
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "wfdb>=4.1.0",
    "matplotlib>=3.7.0",
    "plotly>=5.14.0",
    "scikit-learn>=1.2.0",
    "sktime>=0.24.0",
    "numba>=0.58.0",
    "pywavelets>=1.4.0",
    "onnx>=1.14.0",
    "onnxruntime>=1.15.0",
    "tqdm>=4.65.0",
    "python-dateutil>=2.8.0",
    "psutil>=5.9.0",
    "joblib>=1.3.0",
    # NEW: FastAPI dependencies
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "msgpack>=1.0.7",
    "python-multipart>=0.0.6",
    "websockets>=12.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.26.0",
    "black>=24.1.0",
    "isort>=5.13.0",
    "ruff>=0.1.14",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]
```

## 2) Git Status
- git rev-parse --show-toplevel: C:/Users/ariel/OneDrive/שולחן העבודה/SentinelFetal/SentinelFetal
- git branch --show-current: main
- git rev-parse HEAD: 8d561d493b2f978c45b3277c0caec48ae8a96b58
- git status --porcelain: 8 modified files (working tree is NOT clean)
  - README.md
  - docs/explain/SYSTEM_REFERENCE.md
  - scripts/build_v6_training_pack.py
  - scripts/deep_endurance_audit.py
  - scripts/load_test_suite.py
  - src/ml/analysis/hybrid_validation_v5.py
  - src/ml/analysis/optimize_tiered_logic.py
  - src/ml/training/optimize_v5_optuna.py

## 3) Directory Map (summary)
File counts (recursive):
- config: files=8, dirs=0
- src: files=272, dirs=57
- scripts: files=16, dirs=0
- docs: files=7, dirs=3
- REPORTS: files=8076, dirs=36
- data: files=3977, dirs=32

Key directories:
- config/: ensemble_v4.yaml, ensemble_v5.yaml, ensemble_v5_optuna.yaml, logic_v5_2.yaml, optimized_ensemble_v4.yaml, runtime.yaml, smart_logic_v5_thresholds.yaml, v6_quality_policy.yaml
- src/: adapters, analysis, data, decision, explainability, interfaces, legacy, ml, models, pipeline, rules, safety, simulation, synthetic, training, ui, utils, v6
- src/v6/: pre_ai (core V6 Pre-AI pipeline)
- src/v6/pre_ai/: ingest/, invariants.py, pipeline.py, quality_gate.py, quality_policy.py, windowing.py
- scripts/: build_v6_training_pack.py, verify_v6_pipeline_e2e.py, smoke_import_pre_ai.py, debug_run_pre_ai_audit.py, deep_endurance_audit.py, clinical_validation_suite.py, the_gauntlet.py, the_gauntlet_v3.py, etc.
- docs/: explain/, plan/, reports/
- REPORTS/: audit_artifacts/, doc_cleanup/, ensemble_v5/, optimization_v5_latest/ (+ files)
- data/: dataset roots and extracted assets (not enumerated here)

## 4) Mandatory Configs
### config/runtime.yaml
- Path: config/runtime.yaml
- Mtime: 2026-01-26 13:02:55
- Size: 137 bytes
- SHA256: 4BDEFBA10DEC25C1B7A00B318658EF52AA927B755BADE8D35822929B13EEF72C
- Head (all lines):

```text
fs_hz: 4
window_minutes: 20
stride_minutes: 5
min_window_minutes: 20
recommended_case_minutes: 30
min_case_minutes: 20
strict_mode: true
```

Key points:
- 4 Hz sampling; 20-minute window; 5-minute stride.
- min_window_minutes=20 and min_case_minutes=20 (warmup/strict minimums).
- strict_mode=true (warnings can be treated as errors by scripts).

### config/v6_quality_policy.yaml
- Path: config/v6_quality_policy.yaml
- Mtime: 2026-01-27 02:00:37
- Size: 522 bytes
- SHA256: 136D0E676D797C9FB40025B11007F4CFEC3F0A00393AEB401B6A247F05340EA4
- Head (all lines):

```text
record:
  max_fhr_nan_frac_for_keep: 0.80
  max_fhr_out_of_range_frac_for_keep: 0.60
  max_fhr_zero_frac_for_keep: 0.95
window:
  hard_low_nan_frac: 0.60
  hard_low_max_nan_run: 200
  hard_low_out_of_range_frac: 0.40
  hard_low_unique_ratio: 0.01
  hard_low_std_min: 0.5
  hard_low_std_max: 80.0
  med_nan_frac: 0.40
  med_zero_frac: 0.60
  med_flatline_ratio: 0.92
  med_jump_count_gt25: 10
  med_max_abs_jump: 30.0
interpolation:
  max_gap_samples_to_fill: 10
  fill_method: linear
  do_not_fill_if_gap_too_large: true
```

Key points:
- Record-level keep thresholds for nan/out-of-range/zeros.
- Window-level thresholds for HARD_LOW and MED quality classes.
- Gap-fill policy: linear fill, max 10 samples, no fill if gap too large.

## 5) Training Pack / Data Artifacts (local only)
### training_pack_v6_2.zip
- Path: training_pack_v6_2.zip
- Size: 96,635,284 bytes
- Mtime: 2026-01-27 03:21:53
- SHA256: 47670211D5DAEA1B6E4067DB1B9E86C333B04F5849B423D7FBA64C5BD7498F4D
- Zip top-level entries: manifest.json, records/
- manifest.json present in zip: yes

Manifest (from zip) summary:
- Top-level keys: build_timestamp, ctgdl_suggestions, datasets, git_commit, patients, quality_policy, runtime_config, stats
- counts: total_records=3341, included=2768, skipped=573
- counts_by_dataset: CTU-CHB, CTGDL, FHRMA
- counts_by_target_task: Anatomy, Outcome, Quality
- runtime_config: fs_hz=4.0, window_minutes=20.0, stride_minutes=5.0, min_window_minutes=20.0

records/*.npz (inside zip):
- total records in zip: 2081
- sample names (first 10):
  - records/1001.npz
  - records/1002.npz
  - records/1003.npz
  - records/1004.npz
  - records/1005.npz
  - records/1006.npz
  - records/1007.npz
  - records/1008.npz
  - records/1009.npz
  - records/1010.npz

Local manifest.json files (outside zip):
- Found 7 manifest.json under REPORTS/audit_artifacts
- Latest: REPORTS/audit_artifacts/20260127_0318/training_pack_v6/manifest.json
  - Size: 4,007,513 bytes
  - SHA256: F3DB3D63C0D051997733B19B7CE6C65C76BB0DF8D6DE23139D2AF526A2EC4E3D

Local records/*.npz (outside zip):
- Total .npz under REPORTS: 7,993
- Sample (first 10):
  - REPORTS/audit_artifacts/20260127_0118/training_pack_v6/records/1002.npz
  - REPORTS/audit_artifacts/20260127_0118/training_pack_v6/records/1004.npz
  - REPORTS/audit_artifacts/20260127_0118/training_pack_v6/records/1008.npz
  - REPORTS/audit_artifacts/20260127_0118/training_pack_v6/records/1010.npz
  - REPORTS/audit_artifacts/20260127_0118/training_pack_v6/records/1011.npz
  - REPORTS/audit_artifacts/20260127_0118/training_pack_v6/records/1014.npz
  - REPORTS/audit_artifacts/20260127_0118/training_pack_v6/records/1017.npz
  - REPORTS/audit_artifacts/20260127_0118/training_pack_v6/records/1019.npz
  - REPORTS/audit_artifacts/20260127_0118/training_pack_v6/records/1020.npz
  - REPORTS/audit_artifacts/20260127_0118/training_pack_v6/records/1021.npz
- Sample npz keys (from REPORTS/audit_artifacts/20260127_0118/.../1002.npz):
  - fhr_raw.npy, uc_raw.npy, fs_hz.npy, patient_id.npy, source.npy, labels_json.npy, meta_json.npy, record_quality_json.npy

## 6) REPORTS (latest items)
10 most recent files (by mtime):
- REPORTS/doc_cleanup/DELETE_SWEEP_2026.md (9358 bytes, 2026-01-27 14:41:41)
- REPORTS/audit_artifacts/20260127_0322/phase5_verify_v6_2.log (673 bytes, 2026-01-27 03:22:01)
- REPORTS/audit_artifacts/20260127_0318/build_v6_training_pack.log (28236 bytes, 2026-01-27 03:21:53)
- REPORTS/audit_artifacts/20260127_0318/training_pack_v6/manifest.json (4007513 bytes, 2026-01-27 03:20:50)
- REPORTS/audit_artifacts/20260127_0318/training_pack_v6/records/DopMHRVal0142.npz (52594 bytes, 2026-01-27 03:20:47)
- REPORTS/audit_artifacts/20260127_0318/training_pack_v6/records/DopMHRVal0140.npz (24386 bytes, 2026-01-27 03:20:47)
- REPORTS/audit_artifacts/20260127_0318/training_pack_v6/records/DopMHRVal0139.npz (19826 bytes, 2026-01-27 03:20:47)
- REPORTS/audit_artifacts/20260127_0318/training_pack_v6/records/DopMHRVal0136.npz (18443 bytes, 2026-01-27 03:20:47)
- REPORTS/audit_artifacts/20260127_0318/training_pack_v6/records/DopMHRVal0134.npz (19299 bytes, 2026-01-27 03:20:47)
- REPORTS/audit_artifacts/20260127_0318/training_pack_v6/records/DopMHRVal0133.npz (18014 bytes, 2026-01-27 03:20:47)

Latest report summary: REPORTS/doc_cleanup/DELETE_SWEEP_2026.md
Headings:
- Inventory (all markdown)
- Reference checks (before removal)
- Deletions (executed)
- Final remaining markdown files
- Delete sweep summary
- Verification commands
- Kept exceptions

Key points (high-level):
- Inventory enumerates markdown files and categorizes them (CANONICAL / IMMUTABLE / KEEP_NONLEGACY / LEGACY_DELETE).
- Reference checks table records before/after match counts for each legacy doc.
- Multiple legacy docs marked DELETED after refs removed (see table for each entry).
- Deletions list includes removed legacy report files and history entries.
- Final remaining markdown files list is restricted to canonical + kept exceptions.
- Delete sweep summary reports deleted_files=31 and remaining_files=10.
- Verification commands recorded: import check OK; smoke_import_pre_ai failed due to missing numpy.
- Kept exceptions explicitly noted (UI docs out of scope, and infra audit referenced by immutable docs).

## 7) Backend -> UI Contract (no UI edits)
Primary contract reference:
- docs/explain/UI_REFERENCE.md

Key contract surfaces (summary):
- REST base: /api (or VITE_API_URL), WebSocket: ws://localhost:8001/ws/stream
- REST endpoints include /api/health, /api/patients, /api/patients/{id}, /api/patients/{id}/history, /api/patients/{id}/event
- Simulation control endpoints: /api/simulation/start|stop|pause|resume|reset, /api/simulation/command, /api/simulation/config
- WebSocket control endpoints: /ws/stats, /ws/health; stream endpoints: /ws/stream and /ws/stream/{patient_id}

Key payload fields (from UI_REFERENCE, api/models/schemas.py):
- PatientSnapshot: patient_id, bed_number, category, category_name, metrics, fhr_history, uc_history, timestamps, alerts, fsqi_score, has_active_event, last_update
- EventInjection: event_type, severity, duration_seconds/duration_minutes, params
- WebSocket patient_update: type, timestamp, patient_id, category, baseline, variability, fhr_latest, uc_latest, fsqi, confidence, findings, mhr_alert, trend_score, trend_slope, explanation, highlight_regions

Example (generic) WebSocket patient_update payload (illustrative):
```json
{
  "type": "patient_update",
  "timestamp": 1737940000.0,
  "patient_id": "P1",
  "category": 1,
  "baseline": 140.0,
  "variability": 10.0,
  "fhr_latest": [140.0, 141.0, 140.5, 139.8],
  "uc_latest": [5.0, 6.0, 5.5, 5.2],
  "fsqi": 1.0,
  "confidence": 0.0,
  "findings": {},
  "mhr_alert": {"is_mhr": false, "confidence": 0.0, "recommended_action": "NONE", "detection_methods": []},
  "trend_score": 0.0,
  "trend_slope": 0.0,
  "explanation": null,
  "highlight_regions": []
}
```

## 8) Checklist
- runtime.yaml: ✅ found
- v6_quality_policy.yaml: ✅ found
- requirements.txt: ✅ found
- pyproject.toml: ✅ found
- pip freeze: ✅ (empty output on this machine)
- git commit: ✅ found
- REPORTS folder: ✅ found
- training_pack_*.zip: ✅ found (training_pack_v6_2.zip)
- manifest.json: ✅ found (in zip + REPORTS/audit_artifacts)
- contract schema: ✅ found (docs/explain/UI_REFERENCE.md)

Recommended Next Attachments (if needed):
- If pip freeze should reflect a venv, run pip freeze inside the intended environment and attach that output.
- If sharing training data context, provide the specific manifest.json you want analyzed (path listed above).
