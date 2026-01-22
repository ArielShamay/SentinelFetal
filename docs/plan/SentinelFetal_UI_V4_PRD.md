# SentinelFetal UI V4.0 — Product Requirements Document (PRD)

**Document Version:** 1.0  
**Date:** January 2026  
**Status:** Ready for Development

---

## 1. Executive Summary

### 1.1 Product Vision

SentinelFetal UI V4.0 is a high-performance "Central Station" dashboard for real-time fetal monitoring. The interface must render up to 20 simultaneous patient monitors at 4Hz refresh rate while maintaining smooth, lag-free operation that matches the backend's P99 latency of 58ms.

### 1.2 Problem Statement

The current UI implementation is a non-functional placeholder that:
- Uses fake synthetic data with no backend integration
- Has O(N) memory operations that degrade with history size
- Blocks the UI with `time.sleep()` calls
- Cannot scale beyond 5-10 patients without severe lag
- Lacks clinical diagnostic features (event highlighting, dual-graph synchronization)

### 1.3 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Concurrent Patients | 20 | UI renders all without freezing |
| Refresh Rate | 4Hz (250ms) | Measured via browser DevTools |
| Input Latency | <100ms | Time from click to UI response |
| Memory Usage | <100MB | Chrome Task Manager |
| Visual Fidelity | Medical-grade | Clinical review approval |

---

## 2. User Personas

### 2.1 Primary: Labor & Delivery Nurse

- **Context:** Monitors 4-8 patients during a typical shift
- **Goal:** Quickly identify which patients need attention
- **Pain Points:** Alarm fatigue, cluttered interfaces, slow systems
- **Success:** Can scan 20 monitors in <5 seconds and spot anomalies

### 2.2 Secondary: Attending Obstetrician

- **Context:** Called to review specific patient tracings
- **Goal:** Drill down into detailed waveform analysis
- **Pain Points:** Needs precise timing for deceleration classification
- **Success:** Can measure deceleration timing against contractions

### 2.3 Tertiary: System Administrator / Demo Operator

- **Context:** Demonstrates system capabilities to stakeholders
- **Goal:** Inject specific pathologies and show detection
- **Pain Points:** Needs reproducible scenarios
- **Success:** Can trigger any pathology type on any patient instantly

---

## 3. Core Features

### 3.1 Central Station Grid View (P0 — Must Have)

**Description:** A responsive grid displaying all active patient monitors on a single screen.

**Requirements:**
- Display 0-20 patient monitors simultaneously
- Dynamic grid layout:
  - 1 patient → Full width (100%)
  - 2-4 patients → 2 columns
  - 5-12 patients → 3 columns
  - 13-20 patients → 4 columns
- Each patient card shows:
  - Patient identifier (clickable to expand)
  - Dual-track CTG graph (FHR + UC)
  - Current category status indicator (I/II/III)
  - Latest vital metrics (baseline, variability)
- Auto-refresh at 4Hz without page reload
- Smooth scrolling if grid exceeds viewport

### 3.2 Dual-Track CTG Visualization (P0 — Must Have)

**Description:** Medical-grade cardiotocography display per patient.

**Requirements:**
- **FHR Track (Top):**
  - Y-axis: 50-210 bpm, gridlines every 30 bpm
  - Line color: Deep Blue (#0D47A1)
  - Display 10-minute rolling window (2400 points at 4Hz)
  
- **UC Track (Bottom):**
  - Y-axis: 0-100 mmHg, gridlines every 25 mmHg
  - Line color: Dark Green (#1B5E20)
  - Perfectly synchronized X-axis with FHR

- **X-axis (Shared):**
  - Time-based (HH:MM:SS format)
  - Gridlines every 1 minute
  - Synchronized crosshair on hover

- **Medical Grid:**
  - Light grey gridlines (#E0E0E0) on white background
  - Replicates standard paper CTG aspect ratios

### 3.3 Event Highlighting — "Doctor's Eye" (P0 — Must Have)

**Description:** Visual diagnostic overlays that highlight detected pathologies.

**Requirements:**
- When backend detects an event, highlight the exact time span on the graph
- Use `markArea` shading behind the affected region
- Color coding:
  - Late Deceleration → Red overlay (rgba(211, 47, 47, 0.2))
  - Variable Deceleration → Orange overlay (rgba(245, 127, 23, 0.2))
  - Sinusoidal Pattern → Deep Red overlay (rgba(183, 28, 28, 0.3))
  - Tachysystole → Yellow overlay (rgba(255, 235, 59, 0.2))
- Alert text appears below graph: "Late Deceleration Detected — 14:02:10"
- Clicking the alert auto-scrolls/zooms to the event location

### 3.4 Category Status Display (P0 — Must Have)

**Description:** Clear visual indication of current fetal status.

**Requirements:**
- Category badge on each patient card header
- Color semantics:
  - Category I (Normal) → Green (#388E3C)
  - Category II (Intermediate) → Orange (#F57F17)
  - Category III (Pathological) → Red (#D32F2F)
- Text color matches category (not background fill for subtlety)
- Badge pulses gently when category changes (transition animation)
- No auditory alarms (visual-only to reduce alarm fatigue)

### 3.5 Patient Detail View — Drill-Down (P0 — Must Have)

**Description:** Expanded single-patient view for detailed analysis.

**Requirements:**
- Trigger: Click patient name/ID in grid view
- Full-width display of selected patient
- Extended history view (up to 60 minutes)
- Interactive features enabled:
  - Zoom slider (`dataZoom`) for time range selection
  - Pan to scroll through history
  - Precise tooltip showing exact values on hover
- Patient metadata panel:
  - Name, gestational age, admission time
  - Medical notes field
- "Back to Grid" button to return to Central Station

### 3.6 Patient Management (P1 — Should Have)

**Description:** Add and remove patients from monitoring.

**Requirements:**
- **Admit Patient Form:**
  - Patient ID (required, auto-generated if blank)
  - Display Name (optional)
  - Gestational Age (weeks + days)
  - Notes field (free text)
  - Form in sidebar (non-blocking to main grid)
  
- **Discharge Patient:**
  - "X" button on patient card header (with confirmation)
  - Removes from grid, frees resources
  
- **Maximum Capacity:** 20 patients with clear feedback if exceeded

### 3.7 "God Mode" — Fault Injection (P1 — Should Have)

**Description:** Development/demo feature to inject pathologies.

**Requirements:**
- Collapsible "Dev Tools" panel on each patient card (hidden by default)
- Injection buttons for each pathology type:
  - "Inject Late Decel"
  - "Inject Variable Decel"
  - "Inject Sinusoidal"
  - "Inject Tachysystole"
  - "Inject Bradycardia"
  - "Reset to Normal"
- Instantaneous effect on signal generation
- Backend processes injected signal and triggers appropriate alerts
- Toggle to enable/disable God Mode globally (settings)

---

## 4. Design Specifications

### 4.1 Design Philosophy: "Medical Grade"

The interface must prioritize:
1. **Clarity over aesthetics** — Information density, not decoration
2. **Contrast for readability** — Works under bright hospital lighting
3. **Standard clinical semantics** — Colors and layouts match industry expectations
4. **Minimal cognitive load** — Doctors should scan, not search

### 4.2 Color Palette

| Element | Color | Hex | Rationale |
|---------|-------|-----|-----------|
| Background | White | #FFFFFF | Standard medical records |
| Grid Lines | Light Grey | #E0E0E0 | Visible but unobtrusive |
| Text | Off-Black | #212121 | Reduced eye strain |
| FHR Line | Deep Blue | #0D47A1 | Industry standard |
| UC Line | Dark Green | #1B5E20 | Distinct from FHR |
| Category I | Green | #388E3C | Normal/healthy |
| Category II | Amber | #F57F17 | Warning/attention |
| Category III | Red | #D32F2F | Critical/urgent |
| Card Border | Light Grey | #E0E0E0 | Subtle separation |

### 4.3 Typography

- **Font Family:** System default (Arial fallback)
- **Patient ID:** 16px, bold
- **Metrics:** 14px, regular
- **Axis Labels:** 10px, grey
- **Alert Text:** 14px, bold, colored by severity

### 4.4 Layout Specifications

```
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER: SentinelFetal Central Station          [Settings] [+Add]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────  │
│  │ Patient 1    │  │ Patient 2    │  │ Patient 3    │  │ Patien   │
│  │ ─────────    │  │ ─────────    │  │ ─────────    │  │ ─────    │
│  │  FHR Graph   │  │  FHR Graph   │  │  FHR Graph   │  │  FHR     │
│  │  ~~~~~~~~~~~│  │  ~~~~~~~~~~~│  │  ~~~~~~~~~~~│  │  ~~~~~    │
│  │  UC Graph    │  │  UC Graph    │  │  UC Graph    │  │  UC      │
│  │  ___________│  │  ___________│  │  ___________│  │  ____     │
│  │ Cat I ● 142  │  │ Cat II ● 138│  │ Cat I ● 145 │  │ Cat I    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────  │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ...                           │
│  │ Patient 5    │  │ Patient 6    │                                 │
│  │ ...          │  │ ...          │                                 │
│  └──────────────┘  └──────────────┘                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.5 Patient Card Anatomy

```
┌─────────────────────────────────────────┐
│ Patient ID: P-001          [x] [expand] │  ← Header (clickable)
│ 39w 2d • Room 4B                        │  ← Subheader
├─────────────────────────────────────────┤
│                                         │
│     FHR (bpm)                           │
│  180 ┼────────────────────────────      │
│  150 ┼─────────────/\──────────/\─      │  ← FHR Graph
│  120 ┼────────────────────────────      │
│                                         │
│     UC (mmHg)                           │
│   75 ┼────────/\────────────/\────      │  ← UC Graph  
│   50 ┼─────────────────────────────     │
│                                         │
├─────────────────────────────────────────┤
│ ● Category I    BL: 142   Var: 12      │  ← Status Footer
│ ⚠ Late Decel @ 14:02 (click to view)   │  ← Alert (if any)
└─────────────────────────────────────────┘
```

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Requirement | Specification |
|-------------|---------------|
| Refresh Rate | 4Hz (250ms cycle) |
| Frame Budget | <16ms render time (60fps capable) |
| Memory per Patient | <5MB |
| Total Memory (20 patients) | <100MB |
| Network Payload | <50KB per refresh cycle |
| Initial Load | <3 seconds |

### 5.2 Scalability

- Support minimum 20 concurrent patients
- Graceful degradation at 25+ patients (reduce refresh rate)
- Single-user optimization (multi-user is future scope)

### 5.3 Reliability

- No data loss on brief network interruption
- Automatic reconnection after disconnect
- Clear "Connection Lost" indicator
- Last-known-good state preserved

### 5.4 Browser Support

- Chrome 100+ (primary target)
- Firefox 100+ (secondary)
- Safari 15+ (tertiary)
- Edge (Chromium-based)
- No Internet Explorer support

---

## 6. Technical Constraints

### 6.1 Framework: Streamlit

- Must use Streamlit as the UI framework (existing codebase)
- Leverage `st.fragment` for partial page updates
- Use `st.session_state` for state management

### 6.2 Charting: Apache ECharts

- Must use `streamlit-echarts` component
- Canvas-based rendering (not SVG)
- Disable all animations for real-time performance

### 6.3 Backend Integration

- Connect to existing `PipelineAdapter` class
- Use existing `RingBuffer` for O(1) data management
- Respect existing category/alert data structures

### 6.4 Data Protocol

- FHR: float32 array, 4Hz sampling
- UC: float32 array, 4Hz sampling
- Timestamps: Unix epoch (float)
- Categories: 1, 2, or 3 (integer)
- Events: List of {type, start_time, end_time, severity}

---

## 7. Out of Scope (V4.0)

The following are explicitly excluded from this release:

- Multi-fetal (twins) monitoring
- Audio alarms
- HL7 FHIR integration
- Hospital EHR connectivity
- User authentication/authorization
- Multi-user/multi-session support
- Mobile-responsive layout
- Offline mode
- Historical playback from recordings
- Print/export functionality

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Browser memory leak | High | Strict data windowing, explicit cleanup |
| ECharts performance limit | Medium | Downsampling, disable animations |
| Streamlit rerun overhead | Medium | Use st.fragment extensively |
| Network latency spikes | Low | Buffer recent data client-side |
| God Mode in production | High | Feature flag, disabled by default |

---

## 9. Acceptance Criteria

### 9.1 Grid View
- [ ] Displays 20 patients simultaneously without freezing
- [ ] Dynamic column layout matches patient count
- [ ] 4Hz refresh maintained across all monitors

### 9.2 CTG Visualization
- [ ] FHR and UC graphs perfectly X-axis aligned
- [ ] Medical-grade grid visible and correctly scaled
- [ ] Hover shows synchronized crosshair on both tracks

### 9.3 Event Highlighting
- [ ] Late deceleration shows red overlay at correct time span
- [ ] Alert text appears below graph
- [ ] Clicking alert navigates to event

### 9.4 Patient Management
- [ ] Can add patient with form
- [ ] Can remove patient with confirmation
- [ ] Maximum 20 patient limit enforced

### 9.5 Performance
- [ ] Memory under 100MB with 20 patients
- [ ] No visible lag or stutter during normal operation
- [ ] Page remains responsive during updates

---

## 10. Appendix

### 10.1 Glossary

| Term | Definition |
|------|------------|
| CTG | Cardiotocography — fetal heart rate + uterine contractions |
| FHR | Fetal Heart Rate |
| UC | Uterine Contractions |
| Category I | Normal fetal status |
| Category II | Indeterminate — requires attention |
| Category III | Abnormal — immediate intervention |
| Late Decel | Deceleration starting after contraction peak |
| Variable Decel | Abrupt deceleration (variable shape) |
| Sinusoidal | Smooth sine-wave pattern — critical finding |
| FSQI | Fetal Signal Quality Index |

### 10.2 References

- FIGO Consensus Guidelines on Intrapartum Fetal Monitoring (2015)
- Israeli Position Paper on CTG Interpretation (2020)
- SentinelFetal Technical Whitepaper (2026)
- SentinelFetal UI Master Plan V4.0

---

*End of PRD*
