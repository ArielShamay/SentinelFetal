# SentinelFetal V6 — SPECS (Pre‑AI Infrastructure)

Source of truth: `docs/plan/SentinelFetal V6_ תוכנית עבודה מפורטת.md`.
Scope: Pre‑AI pipeline only (ingestion → RAW invariants → Quality Gate → windowing).

## 1) Single‑Source Runtime Config
- **Config file**: `config/runtime.yaml`
- **Loader**: `src/utils/runtime_config.py`
 - **Quality policy**: `config/v6_quality_policy.yaml` (record/window/interpolation thresholds)
- Required fields (must align to PRD invariants):
  - `fs_hz: 4`
  - `window_minutes: 20`
  - `stride_minutes: 5`
  - `min_window_minutes: 20`
  - `recommended_case_minutes: 30`
  - `strict_mode: true`

## 1.1) Canonical V6 Pre‑AI Modules
- `src/v6/pre_ai/invariants.py` — RAW invariants + WARMUP_ERROR
- `src/v6/pre_ai/quality_gate.py` — RAW quality gate (no AI deps)
- `src/v6/pre_ai/windowing.py` — 20m/5m windowing at 4 Hz
- `src/v6/pre_ai/pipeline.py` — orchestration (ingestion → invariants → quality → windowing)
- `src/v6/pre_ai/ingest/schema.py` — standardized record + record_quality map
- `src/v6/pre_ai/ingest/ctu_loader.py` — CTU‑CHB WFDB ingest (resample to 4 Hz)
- `src/v6/pre_ai/ingest/ctgd_loader.py` — CTGDL CSV ingest (column map + resample)
- `src/v6/pre_ai/ingest/fhrma_loader.py` — FHRMA/FSdataset ingest (quality/noise labels)
- `src/v6/pre_ai/ingest/gap_fill.py` — small-gap interpolation (RAW preserved)
- `src/v6/pre_ai/quality_policy.py` — policy loader (record/window/interpolation thresholds)

## 2) Data Interfaces (RAW)
Required inputs for ingestion stage:
- `patient_id`: string (non‑empty)
- `fhr_raw`: 1‑D numeric array (list/np.ndarray) at 4 Hz
- `uc_raw`: 1‑D numeric array (list/np.ndarray) at 4 Hz
- `timestamps` (optional for audit): 1‑D array aligned to raw samples
Standardized ingest record (for datasets):
- `record_quality`: dict (fractions for NaN/zero/out‑of‑range + duration)

Minimum properties:
- `len(fhr_raw) == len(uc_raw)`
- numeric dtype only (no strings)
- non‑empty

## 3) RAW Invariants Gate
Must run **before any preprocessing**. Required checks:
- Length check against `min_window_minutes` and `min_case_minutes` (for gauntlet/synthetic audit).
- Numeric‑only: non‑numeric values cause immediate failure.
- Alignment: FHR and UC must have identical lengths and sampling rate.

**Failure behavior**:
- Any <20m usable window MUST raise explicit **WARMUP_ERROR** (or an equivalent explicit exception) and halt the pipeline.
- Any misalignment or non‑numeric raw data MUST raise an explicit exception and halt the pipeline.
- No silent fallbacks in strict mode.

## 4) Quality Gate (RAW)
Purpose: classify signal quality using **RAW** signal only.

Interface:
- Input: `fhr_raw_window`, optional `uc_raw_window`, `fs_hz`
- Output (minimum):
  - `quality_class`: HIGH / MED / LOW
  - `hard_low`: bool
  - `metrics`: dict of diagnostics (e.g., valid_frac, std, max_nan_run)

Rules:
- Must operate **before** any AI feature extraction.
- Must be invoked **after RAW invariants**.

## 5) Windowing (20m / 5m at 4 Hz)
- `window_minutes = 20` → `window_samples = 20 * 60 * 4 = 4800`
- `stride_minutes = 5` → `stride_samples = 5 * 60 * 4 = 1200`
- Window generation rule: sliding windows of exactly 4800 samples, every 1200 samples.
- Any attempt to process a window shorter than 4800 samples is a **WARMUP_ERROR**.

## 6) Logging / Telemetry (Pre‑AI)
Minimal counters for audit reporting:
- `warnings_count`
- `fallback_count` (must be zero in strict mode)
- `invariant_violations_count`
- `warmup_errors_count`

Logs should include:
- `patient_id`
- window start/end indices
- `quality_class`
- invariant status (pass/fail)

## 7) Canonical Audit Commands
These commands must be reproducible (see runbook for details):
- Import smoke (strict):
  - `.\.venv311\Scripts\python -W error -c "import src"`
  - `.\.venv311\Scripts\python -W error scripts\smoke_import_pre_ai.py`
- Gauntlet generation (>=30m continuous):
  - `.\.venv311\Scripts\python -W error src\synthetic\generate_gauntlet.py`
- Pre‑AI audit runner (no AI features):
  - `.\.venv311\Scripts\python -W error scripts\debug_run_pre_ai_audit.py --patients 1 --minutes 35 --strict 1`
  - `.\.venv311\Scripts\python -W error scripts\debug_run_pre_ai_audit.py --patients 5 --minutes 35 --strict 1`
- Training pack build (standardized ingest):
  - `.\.venv311\Scripts\python -W error scripts\build_v6_training_pack.py --data-root data --out training_pack_v6.zip`
- E2E pack verification:
  - `.\.venv311\Scripts\python -W error scripts\verify_v6_pipeline_e2e.py --pack training_pack_v6.zip`

## Conflicts
Any conflicts between this spec and repo reality must be documented in `REPORTS/INFRA_AUDIT_2026.md`.
