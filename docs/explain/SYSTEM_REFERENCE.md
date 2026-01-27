# System Reference (Truth-Aligned, V6 Pre-AI)

Date: 2026-01-27
Scope: V6 Pre-AI pipeline, ingest, windowing, and packaging as implemented in code.

## Overview
This reference describes the **Pre-AI** stack only: standardized ingest → RAW invariants → windowing → quality gate → packaging.
AI training and model details are **out of scope** (handled externally in Colab).

## Canonical documents (Immutable)
- `docs/plan/PRD.md`
- `docs/plan/SPECS.md`
- `docs/plan/SentinelFetal V6_ תוכנית עבודה מפורטת.md`
- `docs/plan/middlePlan.md` (handoff summary; do not edit)

## Key directories
- `src/v6/pre_ai/` — Pre-AI pipeline (invariants, quality gate, windowing)
- `src/v6/pre_ai/ingest/` — standardized ingest loaders (CTU/CTGDL/FHRMA)
- `config/runtime.yaml` — runtime invariants (fs/window/stride/min/strict)
- `config/v6_quality_policy.yaml` — quality policy & interpolation policy
- `scripts/` — build pack, verification, smoke and audit scripts

## Runtime invariants (single source)
From `config/runtime.yaml` via `src/utils/runtime_config.py`:
- `fs_hz = 4`
- `window_minutes = 20`
- `stride_minutes = 5`
- `min_window_minutes = 20`
- `min_case_minutes = 20`
- `recommended_case_minutes = 30`
- `strict_mode = true`

**Fail-fast behavior:**
- `WarmupError` is raised when signals/windows are shorter than `min_window_minutes`.
- `apply_strict_warnings(strict_mode)` (used by scripts) converts warnings to errors.

## Pre-AI pipeline (execution order)
Implemented in `src/v6/pre_ai/pipeline.py`:
1. **RAW invariants** (`assert_raw_invariants`) — numeric, alignment, min duration checks.
2. **Windowing** (`window_iter`) — 20-minute windows at 4Hz, stride 5 minutes.
3. **Quality gate** (`quality_gate`) — per-window metrics (nan/zeros/jumps/out-of-range, etc.).

If no windows are produced, `STRICT_WINDOWING` error is raised.

## Standardized ingest (datasets)
Loaders create `StandardizedRecord` (`src/v6/pre_ai/ingest/schema.py`):
- `patient_id`, `fhr_raw`, `uc_raw`, `fs_hz`, `source`, `labels`, `meta`, `record_quality`, optional `fhr_filled/uc_filled`.

Datasets used by `scripts/build_v6_training_pack.py`:
- **CTU-CHB (WFDB)** → target task `Outcome`.
- **CTGDL (CSV / extracted tar.gz)** → target task `Anatomy` (or `Outcome` if labels indicate pH/outcome).
- **FHRMA / FSdataset (CSV/MAT/binary)** → target task `Quality`.

## Packaging (training pack)
`scripts/build_v6_training_pack.py` builds `training_pack_v6_2.zip` with:
- `manifest.json` at the zip root.
- `records/*.npz` with fields:
  - `fhr_raw`, `uc_raw`
  - `fhr_filled`, `uc_filled`
  - `fs_hz`, `patient_id`, `source`
  - `labels_json`, `meta_json`, `record_quality_json`

Manifest includes:
- `runtime_config`, `quality_policy`, `datasets`, `patients[]`, `stats`, `ctgdl_suggestions`.

Additional reports are written to `REPORTS/` (dataset summary, CTGDL extraction, FHRMA forensics/decoding).

## Commands (guardrails & audits)
All commands run from repo root:

```powershell
# Import guardrails (strict warnings)
python -W error -c "import src"
python -W error scripts/smoke_import_pre_ai.py

# Synthetic gauntlet generation (>=30m cases)
python -W error src\synthetic\generate_gauntlet.py

# Pre-AI audit (single patient)
python -W error scripts\debug_run_pre_ai_audit.py --patients 1 --minutes 35 --strict 1

# Build training pack
python -W error scripts\build_v6_training_pack.py --data-root data --out training_pack_v6_2.zip

# Verify E2E pack
python -W error scripts\verify_v6_pipeline_e2e.py --pack training_pack_v6_2.zip
```

Logs are written under `REPORTS/audit_artifacts/` by the build/verify scripts.

## Out of scope
- AI model training, hyperparameter tuning, and Colab workflows.
- Live FastAPI/React UI behavior beyond the API/WS contract in `UI_REFERENCE.md`.
