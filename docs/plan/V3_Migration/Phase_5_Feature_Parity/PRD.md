# Phase 5: Feature Parity & Polish - PRD

**Phase:** 5 of 6
**Duration:** 3-4 days
**Priority:** High
**Risk Level:** Low
**Dependencies:** Phase 4 Complete

---

## 1. Overview

### 1.1 Purpose
Phase 5 ensures the V3 React application achieves full feature parity with the Streamlit V2.0 version. This includes implementing God Mode (event injection), Hebrew/English localization, trend panels, explanation panels, and all remaining UI features.

### 1.2 Goals
1. Implement God Mode control panel for event injection
2. Add Hebrew/English internationalization (i18n)
3. Create Trend Analysis panel with deterioration score
4. Create Explanation panel for AI classifications
5. Implement simulation controls (start/pause/reset)
6. Add patient count configuration
7. Polish UI/UX to match or exceed V2.0 quality

### 1.3 Non-Goals
- Performance optimization beyond Phase 4
- Production deployment (Phase 6)
- New features not in V2.0

---

## 2. User Stories

### US-5.1: God Mode - Event Injection
**As a** demonstrator
**I want** to inject clinical events on demand
**So that** I can showcase the system's detection capabilities

**Acceptance Criteria:**
- [ ] Dropdown to select event type (8 types)
- [ ] Severity selector (Mild/Moderate/Severe)
- [ ] Duration slider (30-600 seconds)
- [ ] Patient selector for target
- [ ] Visual feedback on injection
- [ ] Expected detection time displayed

### US-5.2: Hebrew Language Support
**As a** Hebrew-speaking clinician
**I want** the interface in Hebrew
**So that** I can understand alerts and recommendations

**Acceptance Criteria:**
- [ ] Language toggle (EN/HE) in header
- [ ] All static text translatable
- [ ] Alert headlines in Hebrew
- [ ] Recommendations in Hebrew
- [ ] RTL layout when Hebrew selected
- [ ] Clinical terms correctly translated

### US-5.3: Trend Analysis Panel
**As a** clinician
**I want** to see trend analysis over 60 minutes
**So that** I can identify deterioration patterns

**Acceptance Criteria:**
- [ ] Deterioration score (0-100) with progress bar
- [ ] Color-coded severity (green/orange/red)
- [ ] Variability trend direction (↑↓→)
- [ ] Deceleration counts (30 min, 15 min)
- [ ] Trend alerts with severity

### US-5.4: Explanation Panel
**As a** clinician
**I want** to understand why the AI classified as it did
**So that** I can validate the recommendation

**Acceptance Criteria:**
- [ ] Primary reason for classification
- [ ] Contributing factors list (max 4)
- [ ] Confidence percentage
- [ ] Clear, concise language

### US-5.5: Simulation Controls
**As a** user
**I want** to control the simulation
**So that** I can start, pause, and reset the demonstration

**Acceptance Criteria:**
- [ ] Start button (when stopped)
- [ ] Pause button (when running)
- [ ] Resume button (when paused)
- [ ] Reset button (always available)
- [ ] Patient count adjustment (1-20)
- [ ] Status indicator in header

### US-5.6: Alert Notifications
**As a** clinician
**I want** visual alerts for category changes
**So that** I am notified of deterioration

**Acceptance Criteria:**
- [ ] Toast notification on category upgrade
- [ ] Sound alert option (configurable)
- [ ] Category history in patient view
- [ ] No alert spam (debounced)

---

## 3. Functional Requirements

### FR-5.1: God Mode Events

| Event Type | Display Name | Detection Time | Severity Options |
|------------|--------------|----------------|------------------|
| LATE_DECEL | Late Deceleration | ~15s | Mild, Moderate, Severe |
| VARIABLE_DECEL | Variable Deceleration | ~10s | Mild, Moderate, Severe |
| EARLY_DECEL | Early Deceleration | ~12s | Mild, Moderate |
| PROLONGED_DECEL | Prolonged Deceleration | ~8s | Moderate, Severe |
| TACHYCARDIA | Tachycardia | ~20s | Mild, Moderate, Severe |
| BRADYCARDIA | Bradycardia | ~5s | Moderate, Severe |
| REDUCED_VARIABILITY | Reduced Variability | ~30s | Mild, Moderate, Severe |
| SINUSOIDAL | Sinusoidal Pattern | ~25s | Severe |

### FR-5.2: Localization Strings

All user-facing strings shall be externalized:

```typescript
// Example structure
{
  "header.title": "SentinelFetal",
  "nav.ward": "Ward View",
  "nav.detail": "Patient Detail",
  "patient.baseline": "Baseline",
  "patient.variability": "Variability",
  "alert.category1": "Green Alert - Category 1 (Normal)",
  "alert.category2": "Orange Alert - Category 2 (Intermediate)",
  "alert.category3": "Red Alert - Category 3 (Pathological)",
  "recommendation.normal": "Routine monitoring",
  "recommendation.intermediate": "Increased surveillance recommended",
  "recommendation.pathological": "Immediate clinical evaluation required",
  // ... more strings
}
```

### FR-5.3: Component Requirements

| Component | Features |
|-----------|----------|
| GodModePanel | Event selection, severity, duration, target, inject button |
| LanguageToggle | EN/HE switch, persists to localStorage |
| TrendPanel | Deterioration bar, variability trend, alerts |
| ExplanationPanel | Primary reason, factors, confidence |
| SimulationControls | Start/Pause/Resume/Reset buttons |
| PatientCountSlider | 1-20 range, live update |
| CategoryHistory | Timeline of category changes |

### FR-5.4: RTL Support

When Hebrew is selected:
- Document direction: `rtl`
- Text alignment: right
- Flex/Grid order: reversed where needed
- Icons: not mirrored (clinical symbols)

---

## 4. Non-Functional Requirements

### NFR-5.1: Performance
- Language switch < 100ms
- No additional network requests for translations
- Bundle size increase < 50KB

### NFR-5.2: Accessibility
- All form controls labeled
- ARIA announcements for alerts
- Keyboard navigation complete

### NFR-5.3: Localization Quality
- Native Hebrew speaker review
- Consistent medical terminology
- No truncated strings in UI

---

## 5. UI Design Specifications

### 5.1 God Mode Panel Layout

```
┌──────────────────────────────────────────────────────────┐
│ ⚡ God Mode - Event Injection                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Event Type:  [Late Deceleration        ▼]              │
│                                                          │
│  Severity:    ○ Mild   ● Moderate   ○ Severe            │
│                                                          │
│  Duration:    [====●=============] 2:00 min             │
│                                                          │
│  Target:      [Patient-1 ▼]                             │
│                                                          │
│  Detection:   ~15 seconds                                │
│                                                          │
│             [       💉 Inject Event        ]             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Trend Panel Layout

```
┌─────────────────────────────────────────┐
│ 📈 Trend Analysis (60 min)              │
├─────────────────────────────────────────┤
│                                         │
│  Deterioration Score                    │
│  [███████████░░░░░░░░░░░░░░] 42        │
│                                         │
│  Variability: 📉 Decreasing             │
│                                         │
│  Decels (30 min): 3                     │
│  Late Decels (15 min): 1 ⚠️             │
│                                         │
│  ⚠️ Trend alert: Increasing decels     │
│                                         │
└─────────────────────────────────────────┘
```

### 5.3 Explanation Panel Layout

```
┌─────────────────────────────────────────┐
│ 🔍 Classification Explanation           │
├─────────────────────────────────────────┤
│                                         │
│  Why Category 2?                        │
│  Late decelerations detected            │
│                                         │
│  Contributing Factors:                  │
│  • Reduced variability (4.2 bpm)        │
│  • 2 late decelerations in 15 min       │
│  • Baseline at upper normal (158 bpm)   │
│                                         │
│  Confidence: 87%                        │
│                                         │
└─────────────────────────────────────────┘
```

---

## 6. Dependencies

### 6.1 External Dependencies
```json
{
  "react-i18next": "^14.0.0",
  "i18next": "^23.7.0",
  "react-hot-toast": "^2.4.0"
}
```

### 6.2 Internal Dependencies
- Phase 4 complete (charts working)
- API endpoints for event injection
- WebSocket for real-time updates

---

## 7. Acceptance Criteria Summary

| ID | Criteria | Verification Method |
|----|----------|---------------------|
| AC-5.1 | God Mode injects events | Visual + API log |
| AC-5.2 | Hebrew translation complete | UI review |
| AC-5.3 | RTL layout correct | Visual inspection |
| AC-5.4 | Trend panel displays correctly | With running simulation |
| AC-5.5 | Explanation panel populated | Category 2/3 patient |
| AC-5.6 | Simulation controls functional | Manual test |
| AC-5.7 | Patient count updates | Add/remove patients |
| AC-5.8 | Toast notifications appear | Category change test |

---

## 8. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Hebrew translation quality | Medium | Medium | Native speaker review |
| RTL layout bugs | Medium | Low | Dedicated RTL testing |
| God Mode UX confusion | Low | Low | Clear labels, confirmation |
| Toast notification spam | Medium | Low | Debouncing logic |

---

## 9. Timeline

| Day | Tasks |
|-----|-------|
| Day 1 | God Mode panel, API integration |
| Day 2 | i18n setup, Hebrew translations |
| Day 3 | Trend panel, Explanation panel |
| Day 4 | Simulation controls, polish, testing |

---

## 10. Deliverables Checklist

- [x] `frontend/src/components/godmode/GodModePanel.tsx` ✅ Full event injection UI
- [x] `frontend/src/components/panels/TrendPanel.tsx` ✅ Trend analysis with deterioration score
- [x] `frontend/src/components/panels/ExplanationPanel.tsx` ✅ AI explanation panel
- [x] `frontend/src/components/status/SimulationControls.tsx` ✅ Updated with i18n
- [ ] `frontend/src/components/controls/PatientCountSlider.tsx` ⏳ Pending
- [x] `frontend/src/components/common/LanguageToggle.tsx` ✅ EN/HE switch
- [x] `frontend/src/i18n/en.json` ✅ Full English translations
- [x] `frontend/src/i18n/he.json` ✅ Full Hebrew translations
- [x] `frontend/src/i18n/index.ts` ✅ i18n configuration with RTL
- [x] `frontend/src/App.tsx` ✅ i18n + Toast provider
- [x] `frontend/src/components/layout/Header.tsx` ✅ Language toggle added
- [x] RTL stylesheet support ✅ Dynamic dir attribute
- [x] Toast notification system ✅ react-hot-toast integrated

---

## 12. Implementation Status

### Completed (2025-01-XX)
- **i18n System**: i18next + react-i18next with EN/HE
- **RTL Support**: Dynamic document direction based on language
- **God Mode Panel**: Event injection with all 8 event types
- **Trend Panel**: Deterioration score, variability trend, decel counts
- **Explanation Panel**: Primary reason, contributing factors, confidence
- **Toast Notifications**: react-hot-toast with dark theme
- **Language Toggle**: EN/HE switch in header

### Build Status
- **Bundle Size**: 474KB (gzipped: 151KB)
- **Added Dependencies**: i18next, react-i18next, react-hot-toast
- **TypeScript**: Clean compilation ✅

### Pending
- [ ] PatientCountSlider component
- [ ] Integration of GodModePanel into WardView sidebar
- [ ] Integration of TrendPanel/ExplanationPanel into DetailView

---

## 11. Hebrew String Examples

| Key | English | Hebrew |
|-----|---------|--------|
| alert.category1 | Green Alert - Normal | התראה ירוקה - קטגוריה 1 (תקין) |
| alert.category2 | Orange Alert - Intermediate | התראה כתומה - קטגוריה 2 (בינוני) |
| alert.category3 | Red Alert - Pathological | התראה אדומה - קטגוריה 3 (פתולוגי) |
| rec.normal | Routine monitoring | מעקב שגרתי |
| rec.intermediate | Increased surveillance | הגברת מעקב מומלצת |
| rec.pathological | Immediate evaluation | נדרשת הערכה קלינית מיידית |
| trend.deterioration | Deterioration Score | ציון התדרדרות |
| trend.variability | Variability | וריאביליות |
| godmode.inject | Inject Event | הזרקת אירוע |

---

*End of Phase 5 PRD*
