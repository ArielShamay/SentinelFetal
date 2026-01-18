# SentinelFetal Real-Time Simulator
## Product Requirements Document (PRD)

**Version:** 1.0  
**Date:** January 2026  
**Project:** SentinelFetal Gen3.5 - Real-Time Simulation Module

---

## 1. Executive Summary

### 1.1 Vision
Create a real-time CTG simulation system that generates synthetic fetal monitoring data for multiple patients simultaneously, enabling comprehensive testing and demonstration of the SentinelFetal system under realistic conditions.

### 1.2 Problem Statement
- The existing SentinelFetal system processes static recordings from the CTU-UHB database
- There is no way to test the system's real-time performance with multiple concurrent patients
- Demonstrations require pre-recorded data rather than live, dynamic scenarios
- No ability to intentionally trigger specific clinical events for testing/training purposes

### 1.3 Proposed Solution
A synthetic data generation system that:
- Simulates 8 concurrent patients with realistic CTG patterns
- Runs smoothly on standard hardware (Intel i5 CPU)
- Provides manual control to inject specific clinical events
- Integrates seamlessly with the existing SentinelFetal pipeline
- Uses the real MOMENT model (not mock) for accurate classification

---

## 2. Goals and Success Metrics

### 2.1 Business Goals

| Goal | Success Metric | Target Value |
|------|----------------|--------------|
| Realistic simulation | Clinical pattern accuracy | Validated by domain expert |
| Smooth performance | Frame rate / update frequency | 1 update/second minimum |
| Multiple patients | Concurrent patient count | 8 patients on i5 CPU |
| Full control | Event injection response time | < 2 seconds |
| Integration | Uses existing pipeline | 100% compatibility |

### 2.2 Technical Goals

| Goal | Metric | Target |
|------|--------|--------|
| Memory efficiency | RAM usage | < 500MB total |
| CPU efficiency | CPU usage | < 70% on i5 |
| MOMENT integration | Real model usage | 100% (no mock) |
| Data retention | Storage per hour | < 10MB (events only) |
| Latency | End-to-end processing | < 3 seconds |

---

## 3. User Stories

### 3.1 Primary User: Developer/Tester
- **As a** developer testing SentinelFetal
- **I want to** simulate multiple patients with various conditions
- **So that** I can verify the system handles real-time load correctly

### 3.2 Secondary User: Medical Trainer
- **As a** medical trainer demonstrating the system
- **I want to** inject specific clinical events on demand
- **So that** I can show how the system responds to different scenarios

### 3.3 Tertiary User: Product Demo
- **As a** product manager giving demos
- **I want to** control the simulation flow
- **So that** I can showcase specific features at the right time

---

## 4. Functional Requirements

### 4.1 Simulation Core

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-001 | System shall generate synthetic FHR data at 4Hz per patient | Must |
| FR-002 | System shall generate synthetic UC (contractions) data at 4Hz | Must |
| FR-003 | System shall support 8 concurrent patient simulations | Must |
| FR-004 | System shall maintain 10-minute rolling buffer per patient | Must |
| FR-005 | Generated data shall follow physiological constraints (FHR 50-240 bpm) | Must |
| FR-006 | Baseline FHR shall be configurable per patient (110-160 bpm default) | Should |
| FR-007 | Variability shall be realistic (5-25 bpm for normal) | Must |
| FR-008 | Contractions shall occur at realistic intervals (3-5 per 10 min) | Must |

### 4.2 Event Injection

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-010 | User shall be able to inject Late Decelerations | Must |
| FR-011 | User shall be able to inject Variable Decelerations | Must |
| FR-012 | User shall be able to inject Prolonged Decelerations | Must |
| FR-013 | User shall be able to inject Bradycardia episodes | Must |
| FR-014 | User shall be able to inject Tachycardia episodes | Must |
| FR-015 | User shall be able to inject Absent Variability | Must |
| FR-016 | User shall be able to inject Minimal Variability | Should |
| FR-017 | User shall be able to inject Sinusoidal Pattern | Must |
| FR-018 | User shall be able to inject Tachysystole | Must |
| FR-019 | User shall be able to select target patient for injection | Must |
| FR-020 | User shall be able to set duration for injected events | Should |
| FR-021 | User shall be able to set severity level for events | Should |

### 4.3 Control Panel

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-030 | System shall provide pause/resume functionality | Must |
| FR-031 | System shall display simulation speed control (0.5x, 1x, 2x) | Should |
| FR-032 | System shall allow adding/removing patients dynamically | Should |
| FR-033 | System shall show current status of each patient | Must |
| FR-034 | System shall provide "Reset All" functionality | Should |
| FR-035 | System shall display elapsed simulation time | Must |

### 4.4 Integration with Existing System

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-040 | Simulated data shall feed into existing CTGPreprocessor | Must |
| FR-041 | System shall use existing Rule Engine for analysis | Must |
| FR-042 | System shall use real MOMENT model for embeddings | Must |
| FR-043 | System shall use existing XGBClassifier for classification | Must |
| FR-044 | System shall use existing Alert Engine for notifications | Must |
| FR-045 | System shall display results in existing Dashboard format | Must |

### 4.5 Logging and Export

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-050 | System shall log all injected events with timestamps | Must |
| FR-051 | System shall log all Category 2/3 alerts generated | Must |
| FR-052 | System shall NOT store raw signal data continuously | Must |
| FR-053 | System shall allow export of event log to CSV | Should |
| FR-054 | Log file size shall not exceed 10MB per hour | Must |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-001 | Memory usage for 8 patients | < 500MB |
| NFR-002 | CPU usage on Intel i5 | < 70% |
| NFR-003 | UI update latency | < 500ms |
| NFR-004 | MOMENT processing (staggered) | 1 patient every 4 seconds |
| NFR-005 | Rule Engine processing | All 8 patients every 1 second |

### 5.2 Reliability

| ID | Requirement |
|----|-------------|
| NFR-010 | System shall handle MOMENT failures gracefully (fallback to rules) |
| NFR-011 | System shall recover from UI disconnection |
| NFR-012 | Data generation shall continue even if dashboard freezes |

### 5.3 Usability

| ID | Requirement |
|----|-------------|
| NFR-020 | Event injection shall take maximum 2 clicks |
| NFR-021 | Patient status shall be visible at a glance (color coding) |
| NFR-022 | All text shall be in Hebrew where clinically relevant |

---

## 6. Clinical Event Specifications

### 6.1 Event Types and Parameters

#### Late Deceleration
| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| Depth | 30 | 15-60 | bpm |
| Duration | 60 | 30-120 | seconds |
| Lag from contraction peak | 20 | 15-45 | seconds |
| Recovery time | 30 | 15-60 | seconds |
| Recurrence | 60% | 30-100% | of contractions |

#### Variable Deceleration
| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| Depth | 40 | 15-80 | bpm |
| Duration | 45 | 15-120 | seconds |
| Descent rate | Abrupt | - | - |
| Has severity signs | false | true/false | - |
| Severity: drops below 70 | false | true/false | - |
| Severity: slow recovery | false | true/false | - |

#### Bradycardia
| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| Target FHR | 100 | 60-109 | bpm |
| Duration | 180 | 60-600 | seconds |
| Onset | Gradual | Gradual/Sudden | - |

#### Tachycardia
| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| Target FHR | 170 | 161-200 | bpm |
| Duration | 300 | 60-1200 | seconds |
| Onset | Gradual | - | - |

#### Absent Variability
| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| Variability | 2 | 0-2 | bpm |
| Duration | 300 | 120-1200 | seconds |

#### Minimal Variability
| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| Variability | 4 | 3-5 | bpm |
| Duration | 600 | 300-5400 | seconds |

#### Sinusoidal Pattern
| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| Frequency | 4 | 3-5 | cycles/min |
| Amplitude | 10 | 5-15 | bpm |
| Duration | 1200 | 1200+ | seconds |

#### Tachysystole
| Parameter | Default | Range | Unit |
|-----------|---------|-------|------|
| Contractions per 10 min | 6 | 6-10 | count |
| Duration | 600 | 300-1800 | seconds |

### 6.2 Expected Category Classification

| Event Type | Expected Category | Conditions |
|------------|-------------------|------------|
| Late Decels (isolated) | 2 | Single occurrence |
| Late Decels (recurrent) | 2-3 | >50% of contractions |
| Late Decels + Absent Var | 3 | Any recurrence |
| Variable Decels (mild) | 2 | No severity signs |
| Variable Decels (severe) | 2-3 | With severity signs |
| Variable Decels + Absent Var | 3 | Any recurrence |
| Bradycardia alone | 2 | With normal variability |
| Bradycardia + Absent Var | 3 | Combined |
| Tachycardia alone | 2 | With normal variability |
| Absent Variability alone | 2 | Without decelerations |
| Minimal Variability | 1-2 | Duration dependent |
| Sinusoidal Pattern | 3 | Always |
| Tachysystole | 2 | Check for decelerations |

---

## 7. User Interface Requirements

### 7.1 Control Panel Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMULATION CONTROL                           │
├─────────────────────────────────────────────────────────────────┤
│ Status: ▶ Running    Time: 00:15:32    Patients: 8/8 Active    │
├─────────────────────────────────────────────────────────────────┤
│ Speed: [0.5x] [1x●] [2x]    [⏸ Pause] [🔄 Reset All]           │
├─────────────────────────────────────────────────────────────────┤
│                    INJECT EVENT                                 │
├─────────────────────────────────────────────────────────────────┤
│ Patient: [▼ Patient 3 - Bed 7 ]                                │
│                                                                 │
│ Event Type:                                                     │
│ [Late Decels] [Variable Decels] [Prolonged] [Bradycardia]      │
│ [Tachycardia] [Absent Var] [Minimal Var] [Sinusoidal]          │
│ [Tachysystole]                                                  │
│                                                                 │
│ Duration: [5 min ▼]  Severity: [Moderate ▼]                    │
│                                                                 │
│ [                    💉 INJECT EVENT                    ]       │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Patient Status Display

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATIENT OVERVIEW                             │
├────────┬────────┬────────┬────────┬────────┬────────┬─────────┤
│ P1 🟢  │ P2 🟢  │ P3 🔴  │ P4 🟢  │ P5 🟠  │ P6 🟢  │ P7 🟢   │
│ Cat 1  │ Cat 1  │ Cat 3  │ Cat 1  │ Cat 2  │ Cat 1  │ Cat 1   │
│ Normal │ Normal │ Late D │ Normal │ MinVar │ Normal │ Normal  │
│ Bed 1  │ Bed 2  │ Bed 3  │ Bed 4  │ Bed 5  │ Bed 6  │ Bed 7   │
└────────┴────────┴────────┴────────┴────────┴────────┴─────────┘
```

### 7.3 Detailed Patient View

```
┌─────────────────────────────────────────────────────────────────┐
│ Patient 3 - מיטה 7 - שרה כהן (סימולציה)                         │
├─────────────────────────────────────────────────────────────────┤
│                    🔴 קטגוריה 3 - פתולוגי                        │
├─────────────────────────────────────────────────────────────────┤
│ [Real-time CTG Graph - Last 10 minutes]                         │
│ FHR: ═══════════════════╲___/═══════════════════               │
│ UC:  ___/\___/\___/\___/\___/\___                              │
├─────────────────────────────────────────────────────────────────┤
│ ממצאים:                                                         │
│ • קצב בסיסי: 140 bpm (תקין)                                     │
│ • שונות: 3 bpm (מזערית) ⚠️                                      │
│ • זוהו 4 האטות מאוחרות ב-10 דקות אחרונות                        │
├─────────────────────────────────────────────────────────────────┤
│ המלצות:                                                         │
│ • הערכה מיידית של הסיבות האפשריות                               │
│ • שקילת החייאה תוך-רחמית                                        │
│ • היערכות ליילוד מיידי אם אין שיפור                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Out of Scope (Version 1.0)

The following features are NOT included in this version:
- Recording and playback of simulation sessions
- Network/multi-machine distributed simulation
- Integration with real medical devices
- Historical data import for replay
- Automated scenario sequences
- Machine learning model training on synthetic data
- Mobile application support
- Multi-language support (Hebrew only)

---

## 9. Dependencies

### 9.1 Existing System Components Required

| Component | Location | Purpose |
|-----------|----------|---------|
| CTGPreprocessor | src/data/preprocess.py | Signal preprocessing |
| Rule Engine | src/rules/*.py | Clinical feature extraction |
| MomentFeatureExtractor | src/models/moment_encoder.py | MOMENT embeddings |
| XGBClassifierWrapper | src/models/classifier.py | Classification |
| Alert Engine | src/analysis/alerts.py | Alert generation |
| Medical Override | src/analysis/override.py | Safety rules |
| Config | src/config.py | Thresholds and constants |

### 9.2 New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| threading | stdlib | Concurrent patient simulation |
| queue | stdlib | Thread-safe data passing |
| collections.deque | stdlib | Ring buffer implementation |
| time | stdlib | Timing control |

---

## 10. Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| MOMENT too slow for real-time | High | Medium | Staggered processing (1 patient/4 sec) |
| Memory overflow with 8 patients | High | Low | Ring buffer limits (10 min per patient) |
| UI freezes during processing | Medium | Medium | Separate threads for generation/processing |
| Unrealistic synthetic data | Medium | Medium | Validate with clinical expert |
| Integration breaks existing code | High | Low | Use existing interfaces only |

---

## 11. Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Core Simulator | SyntheticPatientGenerator, RingBuffer, EventInjector |
| 2 | Integration | ProcessingOrchestrator, MOMENT scheduling |
| 3 | UI + Testing | Control Panel, Live Dashboard, Integration tests |

---

## 12. Acceptance Criteria

### 12.1 Must Pass
- [ ] 8 patients run simultaneously without freezing
- [ ] MOMENT processes each patient at least once per 30 seconds
- [ ] All event types can be injected successfully
- [ ] Category 3 events generate appropriate alerts
- [ ] Memory usage stays below 500MB for 1 hour run
- [ ] CPU usage stays below 70% on i5 processor

### 12.2 Should Pass
- [ ] Event injection takes effect within 2 seconds
- [ ] UI updates at minimum 1 FPS
- [ ] Event log exports correctly to CSV
- [ ] Pause/Resume works without data loss

---

*End of Document | SentinelFetal Real-Time Simulator PRD | Version 1.0 | January 2026*
