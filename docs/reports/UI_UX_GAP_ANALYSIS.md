# SentinelFetal UI/UX Gap Analysis Report

**Forensic Audit - Backend/Frontend Disconnect Investigation**

**Date:** January 23, 2026
**Auditor:** AI UI/UX Architect & Code Auditor
**Audit Type:** Deep Dive - Code Level Analysis
**Version:** 2.0 (Comprehensive Rewrite)

---

## Executive Summary

This audit reveals a **critical disconnect** between SentinelFetal's backend capabilities (which passed validation tests with 98% accuracy) and its frontend implementation. The UI layer has multiple structural defects that prevent users from seeing the results of backend processing, creating a "ghost" experience where the system appears non-functional despite working correctly under the hood.

**Key Finding:** The backend V2.0 modules (MHR Guard, Trend Analyzer, Explainability) are fully implemented but have **zero UI integration**. The frontend is essentially running a V1.0 display layer on top of a V2.0 backend.

---

## Issue #1: Ghost Alerts - P2 Logic Failure

### Symptoms
User Report: "Injected events into Patient P2. The backend supposedly detects them, but the UI Card remained Green (Category I)."

### Root Cause (Code Level)

**File:** `src/ui/simulation_app.py`
**Lines:** 421-479 (`render_ward_card_staggered` function)

The bug is a **closure stale capture** anti-pattern:

```python
def render_ward_card_staggered(
    status: Dict[str, Any],  # ← CAPTURED AT CALL TIME
    orchestrator: SimulationOrchestrator,
    patient_idx: int,
    total_patients: int
):
    patient_id = status['patient_id']
    cat = status['category']           # ← STALE VALUE (line 434)
    name = status['name']              # ← STALE VALUE (line 435)

    @st.fragment(run_every=run_every)
    def _ward_card_fragment():
        # This HTML uses OUTER SCOPE variables (captured at page load!)
        card_html = f"""
        <div class="patient-card cat-{cat}">   # ← cat IS STALE! (line 456)
            <span>{CAT_DOTS[cat]}</span>        # ← STALE! (line 459)
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Patient data IS refetched... but not used for category!
        patient = orchestrator.get_patient(patient_id)  # line 470
        if patient:
            data = patient.get_buffer_data(duration_minutes=1)
            # ... only FHR data is used for sparkline, NOT the category
```

**The Problem:** When `@st.fragment(run_every=0.5)` triggers re-execution, the inner function `_ward_card_fragment` runs with **stale** `status`, `cat`, `name` variables captured from the outer scope at page load time. The fragment DOES refetch `patient` data, but it only uses it for the sparkline chart - the **category badge still displays the stale value**.

### The Disconnect

- **Why Backend Test Passed:** The test script calls `orchestrator.get_patient(pid).get_status()['category']` directly, which returns the correct value.
- **Why UI Shows Wrong Value:** The UI captures status at page load, then only partially refreshes data inside the fragment. The category display is NOT inside the data refresh logic.

### Severity: **CRITICAL**

---

## Issue #2: Visualization Failure - No "Red Zones"

### Symptoms
User Report: "No visual indication on the graph where the problem occurred."

### Root Cause (Code Level)

**File:** `src/ui/simulation_app.py`
**Lines:** 658-669 (`render_monitor_panel` function)

```python
fig = create_ctg_plot(
    fhr=fhr,
    uc=uc,
    sampling_rate=4.0,
    title="",
    mode="monitor",
    window_minutes=20.0,
    max_points=2000,
    use_webgl=True,
    uirevision=f"ctg::{patient_id}"
    # MISSING: decelerations=decelerations  ← NOT PASSED!
)
```

**The Problem:** `create_ctg_plot()` in `src/ui/plots.py` (lines 219-380) **fully supports** deceleration visualization via the `decelerations` parameter. When provided, it draws red/orange `vrect` overlays on the chart. However, the UI **never passes this parameter**.

**File:** `src/ui/plots.py`
**Lines:** 332-350 (Deceleration shading logic - UNUSED)

```python
# This code EXISTS but is never executed because decelerations=None
if decelerations:
    for decel in decelerations:
        bounds = _extract_deceleration_bounds_minutes(decel, sampling_rate)
        if not bounds:
            continue
        x0, x1 = bounds
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=f"rgba(255, 0, 0, {alpha})",  # RED OVERLAY
            ...
        )
```

**Additional Gap - V2.0 Visual Mapper NOT Wired:**

**File:** `src/explainability/visual_mapper.py` (EXISTS but unused)

The V2.0 Explainability module includes a `VisualMapper` class that generates `HighlightRegion` objects for graph overlays. This is **fully implemented** in the backend but has **zero integration** with the UI layer.

### The Disconnect

- **Why Backend Test Passed:** Tests verified that `detect_decelerations()` correctly identifies decelerations - which it does.
- **Why UI Shows No Highlights:** The UI layer never queries for deceleration data or passes it to the plotting function. The plot function is called with `decelerations=None` (default).

### Severity: **HIGH**

---

## Issue #3: Navigation & Timeline Locked

### Symptoms
User Report: "Cannot scroll back/forward. Cannot zoom."

### Root Cause (Code Level)

**File:** `src/ui/app.py`
**Lines:** 291-293 (Constants)

```python
DISPLAY_WINDOW_SECONDS = 120  # Show 2 minutes of data (sliding window)
DISPLAY_WINDOW_SAMPLES = int(DISPLAY_WINDOW_SECONDS * 4)  # 480 samples at 4Hz
```

**Lines:** 319-327 (Data Slicing - THE PROBLEM)

```python
# Use sliding window - only show last DISPLAY_WINDOW_SAMPLES samples
if len(fhr_data) > DISPLAY_WINDOW_SAMPLES:
    # Take only the last window for display (data scrolls off)
    fhr_display = list(fhr_data)[-DISPLAY_WINDOW_SAMPLES:]  # ← DATA DESTROYED!
    uc_display = list(uc_data)[-DISPLAY_WINDOW_SAMPLES:]
    ts_display = list(timestamps)[-DISPLAY_WINDOW_SAMPLES:] if timestamps else []
```

**The Problem:** Historical data is **discarded** before reaching the chart. The dataZoom component (lines 513-541) IS configured correctly, but it operates on **already-truncated data**. You can zoom into the last 2 minutes, but you cannot scroll back because there's nothing to scroll to.

**Lines:** 513-530 (DataZoom - configured but useless)

```python
if show_zoom:
    options["dataZoom"] = [
        {
            "type": "slider",  # SLIDER EXISTS
            "start": 0,
            "end": 100,        # BUT OPERATES ON TRUNCATED DATA
            ...
        }
    ]
```

### The Disconnect

- **Why Backend Works:** The `RingBuffer` maintains 10 minutes of data (2400 samples at 4Hz). This data EXISTS in memory.
- **Why UI Can't Access It:** The UI explicitly slices away historical data at line 321 before passing to the chart builder.

### Severity: **MEDIUM**

---

## Issue #4: "Perfect Sine Wave" - Unrealistic Data

### Symptoms
User Report: "FHR turned into a perfect mathematical sine wave (impossible biologically)."

### Root Cause (Code Level)

**File:** `src/simulation/generators/fhr_generator.py`
**Lines:** 472-497 (`_apply_sinusoidal` method)

```python
def _apply_sinusoidal(
    self,
    fhr: np.ndarray,
    times: np.ndarray,
    event: InjectedEvent
) -> np.ndarray:
    """Apply sinusoidal pattern. SEVERE finding - always Category 3."""
    params: SinusoidalParams = event.params
    event_mask = (times >= event.start_time) & (times <= event.end_time)

    if not np.any(event_mask):
        return fhr

    t_rel = times[event_mask] - event.start_time
    freq = params.frequency_cycles_per_min / 60.0

    # PROBLEM: Pure sinusoidal REPLACES normal FHR
    sinusoidal = np.sin(2 * np.pi * freq * t_rel) * params.amplitude_bpm
    fhr[event_mask] = self.config.baseline_fhr + sinusoidal  # ← COMPLETE OVERRIDE!

    return fhr
```

**The Problem:** The sinusoidal pattern **completely replaces** the FHR signal with a pure mathematical sine wave. There is:
- NO noise injection (unlike `_generate_variability()` which adds `high_freq_noise_std`)
- NO residual variability
- NO biological irregularity

Compare to how `_apply_tachycardia()` (lines 440-470) or `_apply_bradycardia()` (lines 408-438) work - they ADD to the existing signal using `+=`. But `_apply_sinusoidal()` does `fhr[mask] = ...` (assignment) instead of `fhr[mask] += ...` (addition).

### The Disconnect

- **Why Backend Test Passed:** The detection test checks for sinusoidal PATTERN presence (FFT frequency analysis), which a pure sine wave will perfectly match. Detection works.
- **Why It Looks Fake:** The signal generator creates a clinically impossible waveform that no clinician would recognize as realistic.

### Severity: **MEDIUM** (cosmetic/realism issue, not functional)

---

## Issue #5: Broken UI Layout (</div> Error)

### Symptoms
User Report: "Main screen is cluttered, shows a literal `</div>` string in a banner."

### Root Cause (Code Level)

**File:** `src/ui/app.py`
**Lines:** 632-635 and 695-696

```python
# Line 632-635: Opening div
st.markdown(
    f'<div style="border-left: 4px solid {border_color}; padding-left: 8px; margin-bottom: 4px;">',
    unsafe_allow_html=True
)

# ... many lines of Streamlit components in between ...

# Line 695-696: Closing div (BROKEN!)
st.markdown('</div>', unsafe_allow_html=True)  # ← THIS IS THE BUG!
```

**The Problem:** Streamlit's `st.markdown()` calls are **independent DOM operations**. Each call creates a separate HTML fragment. The closing `</div>` at line 695 is NOT associated with the opening `<div>` at line 632 - Streamlit doesn't maintain cross-call DOM state.

This is a **fundamental misunderstanding of Streamlit's rendering model**. The closing tag renders as a literal string because there's no matching opening tag in the same `st.markdown()` call.

### The Disconnect

- **Why This Wasn't Caught:** The developer likely tested with fast reloads where the rendering glitch might be masked. Full page loads expose the issue.

### Severity: **LOW** (cosmetic, but unprofessional)

---

## Issue #6: Performance Stutter

### Symptoms
User Report: "The monitor movement is jerky/laggy."

### Root Cause (Code Level)

**File:** `src/ui/app.py`
**Lines:** 951-953 (`main()` function)

```python
# Auto-refresh when running
if st.session_state.is_running:
    time.sleep(0.25)  # 4Hz refresh rate
    st.rerun()        # ← FULL PAGE RERUN!
```

**The Problem:** `st.rerun()` triggers a **complete page re-render**, including:
- Re-fetching ALL patient data
- Re-building ALL chart options
- Re-rendering ALL DOM elements
- Re-running ALL Python logic

This is the **anti-pattern** for real-time Streamlit apps.

**Contrast with `simulation_app.py`** (lines 452-479):

```python
@st.fragment(run_every=run_every)  # ← CORRECT: Partial updates
def _ward_card_fragment():
    # Only THIS fragment re-runs, not the whole page
    ...
```

The `simulation_app.py` uses `@st.fragment` correctly for partial updates, but `app.py` uses the old full-rerun approach.

### The Disconnect

- **Why Backend is Fast:** Backend processing runs in ~31ms average (per endurance tests).
- **Why UI is Slow:** Every 250ms, the entire page is torn down and rebuilt from scratch.

### Severity: **HIGH**

---

## Hidden Gaps Found

### Gap #1: V2.0 Features Have Zero UI Integration

**File:** `src/simulation/processing/pipeline_adapter.py`
**Lines:** 159-195 (V2.0 Module Initialization)

The backend correctly initializes:
- `self._mhr_detector` (MHR Guard Module) - Line 164
- `self._trend_analyzer` (Trend Analyzer Module) - Line 174
- `self._explanation_engine` (Explainability Module) - Line 183

**File:** `src/simulation/processing/pipeline_adapter.py`
**Lines:** 532-544 (V2.0 Return Values)

```python
return {
    'category': final_category,
    'alert': alert,
    'findings': findings,
    # V2.0 additions - THESE ARE RETURNED!
    'mhr_alert': mhr_result.to_dict() if mhr_result and mhr_result.is_suspected else None,
    'trend': trend_result.to_dict() if trend_result else None,
    'explanation': explanation_result.to_dict() if explanation_result else None,
}
```

**The Problem:** Both UI files (`app.py` and `simulation_app.py`) **completely ignore** these V2.0 fields:
- No MHR alert banner (described in PRD lines 80-106)
- No Trend panel (described in PRD lines 159-188)
- No Explanation panel (described in PRD lines 230-266)

### Gap #2: Hardcoded Default Values Override Detection

**File:** `src/ui/app.py`
**Lines:** 246-248

```python
# Update analysis results
ui_state.category = status.get("category", 1)  # ← Default to Cat I (Normal)!
```

If `status` is malformed or `category` key is missing for any reason, the UI defaults to showing "Normal" - masking potential problems.

**File:** `src/ui/app.py`
**Lines:** 100-129 (`PatientUIState` dataclass)

```python
@dataclass
class PatientUIState:
    category: int = 1                      # ← Hardcoded Cat I
    baseline_value: float = 140.0          # ← Hardcoded normal baseline
    variability_value: float = 12.0        # ← Hardcoded normal variability
```

These defaults are used whenever findings are missing or incomplete, which creates false reassurance.

### Gap #3: Documentation Promises Undelivered Features

**File:** `docs/reports/TECHNICAL_WHITEPAPER.md`
**Section 8.5:** "Visual Mapper: Graph Highlighting" (lines 1029-1066)

Documents a fully functional `VisualMapper` class that "converts explanation regions to UI overlay coordinates."

**Reality:** The class EXISTS at `src/explainability/visual_mapper.py` but is **never called** by any UI code.

**File:** `docs/plan/SentinelFetal_V2_PRD_SPECS.md`
**Section "User Experience (UI)":** (lines 80-106, 159-188, 230-266)

Documents specific UI mockups for:
- MHR Alert Banner with "SUSPENDED" status
- 60-Minute Trend Analysis panel with deterioration score bar
- Classification Explanation panel with contributor list

**Reality:** None of these UI components exist. The PRD notes "UI components pending Phase 9" but this is buried in a small note at the top (lines 11-17), while the rest of the document describes them as if they were requirements to be implemented.

---

## Documentation Reality Check Summary

| Document | Feature | Backend Status | Frontend Status |
|----------|---------|----------------|-----------------|
| TECHNICAL_WHITEPAPER.md | MHR Guard Module | **Implemented** | **NOT Implemented** |
| TECHNICAL_WHITEPAPER.md | Trend Analyzer Module | **Implemented** | **NOT Implemented** |
| TECHNICAL_WHITEPAPER.md | Explainability Module | **Implemented** | **NOT Implemented** |
| TECHNICAL_WHITEPAPER.md | Visual Mapper (Highlights) | **Implemented** | **NOT Wired** |
| PRD_SPECS.md | MHR Alert Banner | **Backend Ready** | **Missing** |
| PRD_SPECS.md | Trend Panel UI | **Backend Ready** | **Missing** |
| PRD_SPECS.md | Explanation Panel UI | **Backend Ready** | **Missing** |
| PRD_SPECS.md | Graph Highlights | **Backend Ready** | **Not Wired** |

---

## Severity Summary

| Issue | Severity | Impact |
|-------|----------|--------|
| #1: Ghost Alerts (Stale State) | **CRITICAL** | Clinicians see wrong category - patient safety risk |
| #2: No Red Zones (Missing Param) | **HIGH** | Critical findings not visualized |
| #3: Timeline Locked (Data Truncation) | **MEDIUM** | Cannot review history |
| #4: Perfect Sine Wave (No Noise) | **MEDIUM** | Unrealistic appearance undermines trust |
| #5: Literal `</div>` | **LOW** | Cosmetic/unprofessional |
| #6: Performance Stutter | **HIGH** | Poor user experience, potential alarm fatigue |
| Gap #1: V2.0 UI Missing | **CRITICAL** | 3 major safety features invisible to users |
| Gap #2: Hardcoded Defaults | **HIGH** | False "Normal" display possible |
| Gap #3: Doc/Code Mismatch | **MEDIUM** | Misleading documentation |

---

## Recommendations (DO NOT IMPLEMENT - Diagnosis Only)

1. **Issue #1:** Refetch `status` INSIDE the fragment, not outside. Move `cat = status['category']` inside `_ward_card_fragment()`.
2. **Issue #2:** Pass `decelerations` parameter to `create_ctg_plot()`; wire `VisualMapper` output to chart overlays.
3. **Issue #3:** Remove data truncation at line 321; let dataZoom handle windowing natively.
4. **Issue #4:** Change line 495 from `fhr[event_mask] = ...` to `fhr[event_mask] = ... + noise` where noise = `self._rng.normal(0, 2, ...)`.
5. **Issue #5:** Use single `st.markdown()` call with complete HTML, or use `st.container()` context manager.
6. **Issue #6:** Convert `app.py` to use `@st.fragment` instead of `st.rerun()`.
7. **V2.0 UI:** Implement Phase 9 UI components for MHR Alert, Trend Panel, and Explanation Panel as specified in PRD.

---

**End of Forensic Audit Report**

*Report generated: January 23, 2026*
*Auditor: AI UI/UX Architect & Code Auditor*
