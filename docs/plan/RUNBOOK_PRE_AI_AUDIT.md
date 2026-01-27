# RUNBOOK — Pre‑AI Infrastructure Audit (V6)

Scope: Pre‑AI only (ingestion → RAW invariants → Quality Gate → windowing). No AI inference or model loading.

## Prerequisites
- Windows + PowerShell
- `.venv311` exists and is usable
- Repo root: `C:\Users\ariel\OneDrive\שולחן העבודה\SentinelFetal\SentinelFetal`
- Required folders:
  - `REPORTS/`
  - `REPORTS/audit_artifacts/`

## Standard Environment Setup
```powershell
# From repo root
.\.venv311\Scripts\python -V
.\.venv311\Scripts\pip -V
```

## Audit Stages (copy/paste)

### Stage 0 — Prepare log folder
```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$logdir = "REPORTS/audit_artifacts/$ts"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
```

### Stage 0b — Import smoke (strict)
```powershell
.\.venv311\Scripts\python -W error -c "import src" 2>&1 | Tee-Object "$logdir/import_src.log"
.\.venv311\Scripts\python -W error scripts/smoke_import_pre_ai.py 2>&1 | Tee-Object "$logdir/import_pre_ai.log"
```

### Stage 1 — Synthetic/Gauntlet generation (>=30m)
```powershell
.\.venv311\Scripts\python -W error src\synthetic\generate_gauntlet.py 2>&1 | Tee-Object "$logdir/run1_generate_gauntlet.log"
```

PASS criteria:
- duration stats min/mean/max >= 30.00 minutes
- no warnings (warnings treated as errors)
- no fallbacks

### Stage 2 — Pre‑AI pipeline (single patient)
```powershell
.\.venv311\Scripts\python -W error scripts\debug_run_pre_ai_audit.py --patients 1 --minutes 35 --strict 1 2>&1 | Tee-Object "$logdir/run2_pre_ai_single.log"
```

PASS criteria (all must be YES):
- `ingestion_ok`
- `invariants_failfast_ok`
- `warmup_error_ok`
- `quality_gate_raw_ok`
- `windowing_math_ok`
- `warnings_zero`
- `fallback_count_zero`

### Stage 3 — Pre‑AI pipeline concurrency smoke test
```powershell
.\.venv311\Scripts\python -W error scripts\debug_run_pre_ai_audit.py --patients 5 --minutes 35 --strict 1 2>&1 | Tee-Object "$logdir/run3_pre_ai_concurrent.log"
```

PASS criteria:
- no patient stream mixing
- stable per‑patient outputs
- `warnings_zero` and `fallback_count_zero`

### Stage 2b — Warmup Error Probe (explicit)
```powershell
.\.venv311\Scripts\python -W error scripts\debug_run_pre_ai_audit.py --patients 1 --minutes 10 --strict 1 --allow_short 1 2>&1 | Tee-Object "$logdir/run2b_warmup_error.log"
```
Expected: command exits non‑zero with `WARMUP_ERROR: ... < min_window 20.0`.

### Stage 4 — Standardized ingest + training pack (v6.1)
```powershell
.\.venv311\Scripts\python -W error scripts\build_v6_training_pack.py --data-root data --out training_pack_v6.zip 2>&1 | Tee-Object "$logdir/run4_build_training_pack.log"
```

PASS criteria:
- `training_pack_v6.zip` created
- `manifest.json` included inside the zip
- `REPORTS/DATASET_SUMMARY_V6_FINAL.md` created

### Stage 5 — E2E pack verification
```powershell
.\.venv311\Scripts\python -W error scripts\verify_v6_pipeline_e2e.py --pack training_pack_v6.zip 2>&1 | Tee-Object "$logdir/run5_verify_e2e.log"
```

PASS criteria:
- `GREEN LIGHT` message in output

## What logs/artifacts to collect
Store under `REPORTS/audit_artifacts/YYYYMMDD_HHMM/`:
- `run1_generate_gauntlet.log`
- `run2_pre_ai_single.log`
- `run3_pre_ai_concurrent.log`

## PASS/FAIL Summary Fields (for report)
Record these in `REPORTS/INFRA_AUDIT_2026.md`:
- ingestion_ok: yes/no
- invariants_failfast_ok: yes/no
- warmup_error_ok: yes/no
- quality_gate_raw_ok: yes/no
- windowing_math_ok: yes/no
- warnings_zero: yes/no
- fallback_count_zero: yes/no
