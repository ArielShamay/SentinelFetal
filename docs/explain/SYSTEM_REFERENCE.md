# System Reference (V6 Pre-AI, Truth-Aligned)

Date: 2026-01-27
Scope: V6 Pre-AI ingest, invariants, windowing, quality gate, and training-pack packaging (code-backed only).

## Canonical docs (Immutable)
- docs/plan/PRD.md
- docs/plan/SPECS.md
- docs/plan/SentinelFetal V6_ תוכנית עבודה מפורטת.md
- docs/plan/middlePlan.md

## V6 code map (steel wall)
- V6 Pre-AI pipeline entrypoint: src/v6/pre_ai/pipeline.py (run_pre_ai)
- V6 ingest loaders: src/v6/pre_ai/ingest/ (ctu_loader.py, ctgd_loader.py, fhrma_loader.py)
- Runtime config and strict-mode: config/runtime.yaml + src/utils/runtime_config.py (RuntimeConfig, load_runtime_config, apply_strict_warnings)
- Quality policy: config/v6_quality_policy.yaml + src/v6/pre_ai/quality_policy.py (load_quality_policy)
- Pack build/verify scripts: scripts/build_v6_training_pack.py, scripts/verify_v6_pipeline_e2e.py, scripts/smoke_import_pre_ai.py

## Runtime invariants (source of truth)
Configured in config/runtime.yaml and loaded/validated by src/utils/runtime_config.py (load_runtime_config, _validate_runtime_config):
- fs_hz: 4
- window_minutes: 20
- stride_minutes: 5
- min_window_minutes: 20
- min_case_minutes: 20
- recommended_case_minutes: 30
- strict_mode: true

Strict-mode behavior is implemented by src/utils/runtime_config.py (apply_strict_warnings) and used by scripts/build_v6_training_pack.py and scripts/verify_v6_pipeline_e2e.py.

## Pre-AI pipeline (execution order)
Implemented in src/v6/pre_ai/pipeline.py (run_pre_ai):
1) RAW invariants (src/v6/pre_ai/invariants.py: assert_raw_invariants)
2) Windowing (src/v6/pre_ai/windowing.py: window_iter)
3) Quality gate per window (src/v6/pre_ai/quality_gate.py: quality_gate)

Warmup and minimum-length enforcement raise WarmupError in src/v6/pre_ai/invariants.py and src/v6/pre_ai/windowing.py.

## Quality gate (RAW, no AI)
Quality classification is computed in src/v6/pre_ai/quality_gate.py (quality_gate) and emits HIGH/MED/LOW based on metrics including nan_frac, zeros_frac, max_nan_run, jump_count_gt25, max_abs_jump, flatline_ratio, out_of_range_frac, unique_ratio.

## Standardized ingest (datasets)
The standardized record schema is defined in src/v6/pre_ai/ingest/schema.py (StandardizedRecord, compute_record_quality).

Loaders (source-specific):
- CTU-CHB WFDB ingest and resample: src/v6/pre_ai/ingest/ctu_loader.py (load_ctu_record)
- CTGDL CSV ingest: src/v6/pre_ai/ingest/ctgd_loader.py (load_ctgdl_record)
- FHRMA/FSdataset ingest (CSV/binary heuristics): src/v6/pre_ai/ingest/fhrma_loader.py (load_fhrma_record)

Target task mapping for packs is defined in scripts/build_v6_training_pack.py (_target_task):
- CTU-CHB → Outcome
- CTGDL → Anatomy (or Outcome if outcome labels exist)
- FHRMA → Quality

## Training pack format
The training pack is built in scripts/build_v6_training_pack.py (main, _save_npz, manifest construction, zipfile write):
- Output zip default: training_pack_v6_2.zip (arg --out)
- Zip root contains manifest.json (written at manifest_path)
- records/*.npz entries are written into the zip

NPZ fields written by _save_npz (scripts/build_v6_training_pack.py):
- fhr_raw, uc_raw
- fhr_filled, uc_filled
- fs_hz, patient_id, source
- labels_json, meta_json, record_quality_json

Manifest structure (scripts/build_v6_training_pack.py: manifest dict) includes:
- runtime_config (fs_hz, window_minutes, stride_minutes, min_window_minutes)
- quality_policy
- datasets
- patients
- ctgdl_suggestions
- stats (counts, counts_by_dataset, counts_by_target_task, skipped_by_reason, duration_minutes, quality_class_counts)

End-to-end verification reads manifest.json and records/*.npz and re-runs run_pre_ai in scripts/verify_v6_pipeline_e2e.py.

## Commands (guardrails & audits)
All commands are defined by these entrypoints and are run from repo root:

```powershell
python -W error -c "import src"                         # import guard (module load)
python -W error scripts/smoke_import_pre_ai.py           # Pre-AI smoke import
python -W error src\synthetic\generate_gauntlet.py      # synthetic gauntlet generation
python -W error scripts\debug_run_pre_ai_audit.py --patients 1 --minutes 35 --strict 1
python -W error scripts\build_v6_training_pack.py --data-root data --out training_pack_v6_2.zip
python -W error scripts\verify_v6_pipeline_e2e.py --pack training_pack_v6_2.zip
```

Log directories for build/verify default to REPORTS/audit_artifacts/<timestamp> (scripts/build_v6_training_pack.py: _ensure_log_dir, scripts/verify_v6_pipeline_e2e.py: log_dir default).

## Out of scope (this document)
- AI model training, tuning, and hybrid inference paths (refer to docs/plan/PRD.md and docs/plan/SPECS.md for future scope).
- UI and frontend integration (refer to UI docs if needed; not covered here).
