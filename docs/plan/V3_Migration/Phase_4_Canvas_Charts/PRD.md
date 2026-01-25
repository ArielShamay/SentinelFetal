# Phase 4: Canvas Charts (Lightweight-Charts) - PRD

**Phase:** 4 of 6
**Duration:** 4-5 days
**Priority:** Critical (Core differentiator from Streamlit)
**Risk Level:** Medium
**Dependencies:** Phase 3 Complete

---

## 1. Overview

### 1.1 Purpose
Phase 4 implements the high-performance CTG charting system using TradingView's Lightweight-Charts library. This is the core technical differentiator that enables 60 FPS rendering versus Streamlit's ~5 FPS, providing a professional trading-style monitoring experience.

### 1.2 Goals
1. Integrate Lightweight-Charts for FHR and UC visualization
2. Achieve 60 FPS rendering with 20+ patients
3. Implement real-time streaming data updates
4. Add red zone highlighting for detected events
5. Preserve zoom/pan state across updates
6. Create mini-sparklines for ward view cards

### 1.3 Non-Goals
- God Mode controls (Phase 5)
- Hebrew localization (Phase 5)
- Production deployment (Phase 6)

---

## 2. User Stories

### US-4.1: Real-Time CTG Display
**As a** clinician
**I want** to see FHR and UC traces updating in real-time
**So that** I can monitor fetal wellbeing continuously

**Acceptance Criteria:**
- [ ] FHR trace displays as continuous line (blue)
- [ ] UC trace displays below FHR (orange)
- [ ] Updates at 4Hz without visible stuttering
- [ ] Maintains 60 FPS during normal operation
- [ ] 20-minute visible window by default

### US-4.2: Red Zone Visualization
**As a** clinician
**I want** to see highlighted regions for detected events
**So that** I can quickly identify concerning patterns

**Acceptance Criteria:**
- [ ] Red zones rendered as semi-transparent overlay
- [ ] Zone severity indicated by color intensity
- [ ] Labels displayed for zone type (e.g., "Late Decel")
- [ ] Zones update in real-time as events are detected
- [ ] Zones preserve position during scroll/zoom

### US-4.3: Chart Interaction
**As a** clinician
**I want** to zoom and pan the CTG trace
**So that** I can examine specific time periods in detail

**Acceptance Criteria:**
- [ ] Mouse wheel zooms in/out
- [ ] Click-drag pans the visible window
- [ ] Pinch-to-zoom on touch devices
- [ ] "Live" button returns to real-time view
- [ ] Zoom/pan state preserved during data updates

### US-4.4: Ward View Sparklines
**As a** clinician
**I want** mini FHR traces in patient cards
**So that** I can see trends at a glance

**Acceptance Criteria:**
- [ ] Sparkline shows last 60 seconds of FHR
- [ ] Color reflects current category (green/orange/red)
- [ ] Sparkline updates every 500ms
- [ ] Minimal CPU usage (~1% per sparkline)

### US-4.5: Performance Under Load
**As a** system
**I want** to render 20 patients simultaneously
**So that** the ward view remains responsive

**Acceptance Criteria:**
- [ ] Ward view with 20 sparklines < 30% CPU
- [ ] Detail view with full chart < 10% CPU
- [ ] Memory usage < 100MB for charts
- [ ] No memory leaks over 24-hour operation

---

## 3. Functional Requirements

### FR-4.1: Chart Components

| Component | Purpose | Used In |
|-----------|---------|---------|
| `CTGChart` | Full-size FHR+UC chart | Detail View |
| `FHRSparkline` | Mini FHR trace | Patient Cards |
| `RedZoneOverlay` | Highlighted regions | CTGChart |
| `ChartControls` | Zoom/pan/reset buttons | Detail View |

### FR-4.2: Data Flow

```
WebSocket Update
       │
       ▼
┌─────────────────┐
│ Patient Store   │
│ (fhr_latest)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────────┐
│Spark- │ │CTG Chart  │
│line   │ │(buffered) │
└───────┘ └───────────┘
```

### FR-4.3: Chart Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| FHR Y-Range | 60-200 bpm | Fixed scale for consistency |
| UC Y-Range | 0-100 units | Configurable |
| Time Window | 20 minutes | Default visible range |
| Sample Rate | 4 Hz | Matches backend |
| Buffer Size | 4800 samples | 20 minutes of data |

### FR-4.4: Red Zone Styling

| Severity | Color | Opacity | Label |
|----------|-------|---------|-------|
| Info | Blue | 0.1 | Optional |
| Warning | Orange | 0.2 | Event type |
| Critical | Red | 0.3 | Event type + alert |

### FR-4.5: Responsive Behavior

| Viewport | Chart Height | Sparkline Size |
|----------|--------------|----------------|
| Desktop (>1200px) | 400px | 48x120px |
| Tablet (768-1200px) | 300px | 40x100px |
| Mobile (<768px) | 200px | 32x80px |

---

## 4. Non-Functional Requirements

### NFR-4.1: Performance
- Frame rate: 60 FPS sustained
- Latency: < 16.67ms per frame
- CPU usage: < 30% for full ward view
- Memory: < 100MB for chart data

### NFR-4.2: Visual Quality
- Anti-aliased line rendering
- Smooth transitions on zoom/pan
- No flicker during updates
- Consistent colors across browsers

### NFR-4.3: Accessibility
- Chart data available as table (screen readers)
- Keyboard navigation for zoom/pan
- High contrast mode support

---

## 5. Technical Approach

### 5.1 Why Lightweight-Charts

| Library | Rendering | Bundle Size | Real-time | Decision |
|---------|-----------|-------------|-----------|----------|
| Plotly | SVG | 3.5MB | Polling | ❌ Too slow |
| ECharts | Canvas/SVG | 1MB | Yes | ⚠️ Complex API |
| Chart.js | Canvas | 200KB | Yes | ⚠️ Not streaming |
| **Lightweight-Charts** | **Canvas** | **45KB** | **Native** | ✅ **Best fit** |
| uPlot | Canvas | 35KB | Yes | ⚠️ Less features |

### 5.2 Rendering Strategy

```
Backend (4Hz) → WebSocket → Store → Chart.update()
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                    requestAnimationFrame    Direct Canvas
                    (for smooth animation)   (for static)
```

### 5.3 Memory Management

- Ring buffer for FHR/UC data (fixed size)
- Lazy rendering for off-screen charts
- Garbage collection on view change
- Shared WebGL context where possible

---

## 6. Dependencies

### 6.1 External Dependencies
```json
{
  "lightweight-charts": "^4.1.0"
}
```

### 6.2 Internal Dependencies
- Phase 3 complete (React scaffold)
- WebSocket streaming working
- Patient store with fhr_latest

---

## 7. Acceptance Criteria Summary

| ID | Criteria | Verification Method |
|----|----------|---------------------|
| AC-4.1 | CTG chart renders FHR+UC | Visual inspection |
| AC-4.2 | Chart updates at 4Hz | Frame counter |
| AC-4.3 | 60 FPS maintained | Chrome DevTools |
| AC-4.4 | Red zones display correctly | Event injection test |
| AC-4.5 | Zoom/pan works | Manual interaction |
| AC-4.6 | Sparklines render in cards | Ward view test |
| AC-4.7 | 20 patients < 30% CPU | Performance profiler |
| AC-4.8 | No memory leaks | 1-hour soak test |

---

## 8. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Lightweight-Charts API changes | Low | Medium | Pin version, read changelog |
| Canvas performance on mobile | Medium | Medium | Reduce sparkline frequency |
| WebGL context limits | Low | High | Fallback to 2D canvas |
| Memory growth | Medium | High | Implement ring buffers |

---

## 9. Timeline

| Day | Tasks |
|-----|-------|
| Day 1 | Lightweight-Charts setup, basic FHR line |
| Day 2 | UC trace, dual-axis layout, styling |
| Day 3 | Real-time updates, streaming integration |
| Day 4 | Red zones, interaction (zoom/pan) |
| Day 5 | Sparklines, performance optimization |

---

## 10. Deliverables Checklist

- [x] `frontend/src/components/charts/CTGChart.tsx` ✅ Created with FHR+UC dual pane
- [x] `frontend/src/components/charts/FHRSparkline.tsx` ✅ Created with SVG sparkline
- [ ] `frontend/src/components/charts/RedZoneOverlay.tsx` ⏳ Deferred (basic support in CTGChart)
- [x] `frontend/src/components/charts/ChartControls.tsx` ✅ Created with zoom/pan/time range
- [x] `frontend/src/hooks/useChartData.ts` ✅ Created with RingBuffer integration
- [x] `frontend/src/utils/chartConfig.ts` ✅ Created with LC config, clinical ranges
- [x] `frontend/src/hooks/useLightweightChart.ts` ✅ Created for chart lifecycle
- [x] `frontend/src/utils/ringBuffer.ts` ✅ Created for efficient data buffering
- [x] `frontend/src/utils/chartHelpers.ts` ✅ Created with helper functions
- [x] `frontend/src/types/chart.ts` ✅ Created with all chart types
- [ ] Performance benchmarks documented
- [x] Integration with Detail View ✅ CTGChartPanel replaces placeholder
- [ ] Integration with Patient Cards (sparkline wiring pending)

---

## 12. Implementation Status

### Completed (2025-01-XX)
- **lightweight-charts 4.1.0** installed
- **Chart Infrastructure**: Types, config, utilities, ring buffer
- **Chart Hooks**: useChartData for buffer management, useLightweightChart for instance lifecycle
- **Components**: CTGChart (main chart), FHRSparkline (SVG mini-chart), ChartControls (zoom/pan/time)
- **Integration**: DetailView now uses CTGChartPanel with real CTGChart
- **Build**: 401KB bundle, verified TypeScript compilation

### Pending
- Real WebSocket data wiring to charts
- Performance benchmarking
- Red zone overlay refinement
- Sparkline integration in PatientCard

---

## 11. Visual Reference

### 11.1 CTG Chart Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Patient-1                              [Zoom] [Pan] [Live] │
├─────────────────────────────────────────────────────────────┤
│ 200 ┤                                                       │
│     │                    ╱╲    ╱╲                          │
│ 150 ┤  ──────────────╱    ╲──╱  ╲──────                   │
│     │ FHR (bpm)                        ▓▓▓▓ Red Zone       │
│ 100 ┤                                                       │
│     │                                                       │
│  60 ┤───────────────────────────────────────────────────── │
├─────────────────────────────────────────────────────────────┤
│ 100 ┤      ╱╲          ╱╲          ╱╲                      │
│     │     ╱  ╲        ╱  ╲        ╱  ╲                     │
│  50 ┤────╱────╲──────╱────╲──────╱────╲────────────────── │
│     │ UC                                                    │
│   0 ┤                                                       │
└─────────────────────────────────────────────────────────────┘
      14:00    14:05    14:10    14:15    14:20 (time)
```

### 11.2 Sparkline in Patient Card

```
┌─────────────────────────┐
│ Patient-1    [Normal]   │
│ Baseline: 142 bpm       │
│ Variability: 12.3 bpm   │
│ ╭──╮  ╭──╮  ╭──╮       │
│ │  ╰──╯  ╰──╯  │ ← Mini │
│ ╰──────────────╯   FHR  │
│        [View →]         │
└─────────────────────────┘
```

---

*End of Phase 4 PRD*
