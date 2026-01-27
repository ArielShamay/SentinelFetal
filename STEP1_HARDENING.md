# Step 1 Hardening Report

## Changes
- `src/decision/smart_hybrid_logic.py`: compute_signal_quality now uses runtime-configured min window minutes for RAW length assertions.
- `src/analysis/fallback_audit.py`: added window index context to fallback records for strict-mode traceability.
- `src/synthetic/run_gauntlet_stress.py`: sets window context per window and prints `fallback_count` for evidence.
- `src/synthetic/generate_gauntlet.py`: post-save verification now checks every row (no sampling) to catch truncation.
- `src/utils/fallback_audit.py`: re-exported analysis fallback audit to keep a single source of truth.
- `src/utils/signal_invariants.py`: re-exported signal invariants to avoid duplicate implementations.

## Why
- Enforce single-source runtime config for RAW quality gating and window-length invariants.
- Fail fast with complete context (case_id + window_idx) when any rule fallback occurs in strict mode.
- Guarantee no short or truncated gauntlet cases can slip through save/reload.
- Reduce drift by eliminating duplicate invariant/audit implementations.

## Verification
- Command: `.\\.venv311\\Scripts\\python src\\synthetic\\generate_gauntlet.py`
  - `Duration minutes (min/mean/max): 30.00 / 36.43 / 60.00`
  - `Sample count (min/mean/max): 7200 / 8743 / 14400`
  - `Saved 4 logic cases, 6 noise cases, 4 edge cases to data\\synthetic_gauntlet`
  - Notes: `PyTorch not available.` `momentfm package not available. Install with: pip install momentfm`
- Command: `.\\.venv311\\Scripts\\python src\\synthetic\\run_gauntlet_stress.py`
  - `Gauntlet cases loaded: 14`
  - `FHR duration minutes (min/mean/max): 30.00 / 36.43 / 60.00`
  - `UC  duration minutes (min/mean/max): 30.00 / 36.43 / 60.00`
  - `Skipped cases due to short signal: 0`
  - `STRICT_MODE: True`
  - `Window config: window=20.0 min, stride=5.0 min, min_window=20.0 min`
  - `fallback_count: 0`
  - `No false positives detected on Gauntlet.`
  - `Saved analysis to data\\synthetic_gauntlet\\synthetic_fp_analysis.csv`
  - `Saved hard negatives to data\\synthetic_gauntlet\\synthetic_hard_negatives.csv`
  - Notes: `PyTorch not available.` `momentfm package not available. Install with: pip install momentfm`

## PASS/FAIL Criteria
- Runtime config is the single source of truth (window=20m, stride=5m, min_window=20m, recommended_case>=30m, strict_mode=true): PASS
- Any RAW signal < 20m fails fast before processing: PASS
- All gauntlet cases are >= 30m; none < 20m: PASS
- Signal quality computed on RAW windows only: PASS
- Strict mode raises on any rule-module fallback; fallback_count == 0 in gauntlet: PASS
- Gauntlet window slices enforced >= 20m: PASS

## Final Status
PASS
