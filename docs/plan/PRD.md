# SentinelFetal V6 — PRD (Pre‑AI Infrastructure)

Source of truth: `docs/plan/SentinelFetal V6_ תוכנית עבודה מפורטת.md`.
Scope: Pre‑AI plumbing only (ingestion → RAW invariants → Quality Gate → windowing).

## Product Vision (brief)
Build a reliable, fail‑fast pre‑AI pipeline that guarantees clean, aligned, and sufficiently long CTG signals before any AI work. This establishes the foundation for V6 by enforcing strict quality and windowing invariants locally.

## Core Invariants (single source)
These are mandatory for all pre‑AI audits and synthetic/gauntlet runs:
- Sampling rate: **4 Hz**.
- Windowing: **window=20m**, **stride=5m**, **min_window=20m**.
- Case duration (gauntlet/synthetic audit): **>= 30 minutes** continuous.
- **Strict mode ON** with **warnings treated as errors** during audits.
- **Fail‑fast** on:
  - any <20m usable window (explicit WARMUP_ERROR or equivalent exception),
  - FHR/UC misalignment (length or start mismatch),
  - non‑numeric or invalid raw samples.
- **No silent fallbacks**: any fallback in strict mode is a defect and must surface with explicit module/path/reason.

## Pre‑AI Pipeline Definition
**Ingestion → RAW Invariants Gate → Quality Gate (Iron Dome on RAW) → Windowing (20m / 5m)**

Notes:
- RAW invariants must run **before any preprocessing**.
- Quality Gate operates on **RAW** (pre‑interpolation/smoothing).

## Out of Scope (until trained models exist)
- AI inference (MiniRocket/features/ensemble/Hybrid tiers).
- Model manifests and loading trained artifacts (pkl/joblib).
- Explainability from AI models (SHAP or model‑based explanations).
- Calibration/threshold optimization.

## Conflicts
Any conflict between this PRD and the current repo implementation must be logged in `REPORTS/INFRA_AUDIT_2026.md`.
