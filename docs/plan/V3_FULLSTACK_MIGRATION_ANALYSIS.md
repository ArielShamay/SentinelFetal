# SentinelFetal V3.0 - Full-Stack Migration Analysis

**Document Type:** Technical Architecture Blueprint
**Author:** AI Chief Software Architect
**Date:** January 24, 2026
**Status:** Analysis Complete - Awaiting Decision

---

## Executive Summary

This document analyzes the feasibility of migrating SentinelFetal from its current Streamlit-based architecture to a professional real-time stack similar to TradingView. The analysis covers technical bottlenecks, proposed solutions, effort estimation, and risk assessment.

**Bottom Line:** Migration is **technically feasible** and would deliver significant UX improvements. However, it requires **3-4 weeks of development** and introduces new operational complexity. The decision depends on whether "buttery smooth 60FPS" is a clinical requirement or a nice-to-have.

---

## Phase 1: Current State Analysis ("As-Is")

### 1.1 Current Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **UI Framework** | Streamlit | >=1.28.0 | Page rendering, state management |
| **Charting** | Plotly | >=5.14.0 | SVG-based CTG graphs |
| **Charting (Alt)** | streamlit-echarts | >=0.4.0 | Canvas-based charts (unused) |
| **State Bridge** | Custom DataBridge | - | Thread-safe Python singleton |
| **Backend Thread** | Python threading | - | SimulationOrchestrator |
| **ML Inference** | scikit-learn, sktime | - | MiniRocket + XGBoost |
| **Data Processing** | NumPy, SciPy | - | Signal processing |

### 1.2 Current Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     CURRENT ARCHITECTURE (V2.0 Pulse)                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   PYTHON BACKEND                    SHARED STATE                STREAMLIT  │
│   ┌─────────────────┐              ┌──────────────┐         ┌───────────┐ │
│   │ Orchestrator    │──(4Hz)──────▶│ DataBridge   │◀────────│ Fragment  │ │
│   │ (Python Thread) │              │ (Python Dict)│ (poll)  │ run_every │ │
│   └────────┬────────┘              └──────────────┘         └─────┬─────┘ │
│            │                                                      │       │
│   ┌────────▼────────┐                                    ┌────────▼─────┐ │
│   │ PipelineAdapter │                                    │ Plotly JSON  │ │
│   │ (MOMENT+Rules)  │                                    │ Serialization│ │
│   └─────────────────┘                                    └──────────────┘ │
│                                                                            │
│   ════════════════════════ PYTHON GIL BOUNDARY ════════════════════════   │
│                                                                            │
│                              ┌──────────────────┐                          │
│                              │   HTTP Bridge    │                          │
│                              │ (Streamlit Core) │                          │
│                              └────────┬─────────┘                          │
│                                       │ WebSocket (Streamlit internal)     │
│                              ┌────────▼─────────┐                          │
│                              │   Browser DOM    │                          │
│                              │   (React-based)  │                          │
│                              └──────────────────┘                          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Why Streamlit Cannot Achieve 60FPS

#### The Fundamental Bottlenecks

| Bottleneck | Technical Explanation | Impact |
|------------|----------------------|--------|
| **Python GIL** | Global Interpreter Lock blocks concurrent execution. Even with threading, only one Python thread executes at a time. | Backend processing blocks UI updates. |
| **Plotly JSON Serialization** | Each Plotly figure is serialized to JSON (~100-500KB per frame), sent over HTTP, and re-parsed in JavaScript. | 50-200ms per chart update. |
| **SVG Rendering** | Plotly uses SVG by default. SVG DOM manipulation is O(n) complexity - 2400 data points = 2400 DOM nodes to update. | Cannot exceed ~15 FPS even with optimization. |
| **Full Page Re-renders** | Streamlit's `st.rerun()` recreates the entire React component tree. Even `@st.fragment` only reduces scope, not the mechanism. | UI "flickers" on complex pages. |
| **No Direct WebGL Access** | Plotly's WebGL mode (`Scattergl`) exists but still goes through the Python→JSON→JS bridge. | GPU acceleration is indirect, not native. |
| **Polling Architecture** | `run_every=0.1` polls DataBridge every 100ms. This is fundamentally different from push-based WebSocket streaming. | Inherent 50-100ms latency floor. |

#### Latency Breakdown (Current)

```
Backend Processing (MOMENT):     ~50ms
Python→JSON Serialization:       ~30ms
HTTP Transfer:                   ~10ms
JSON Parsing in Browser:         ~20ms
SVG DOM Update (2400 points):    ~80ms
─────────────────────────────────────────
TOTAL FRAME TIME:               ~190ms → ~5 FPS effective
```

**Target for 60FPS:** Each frame must complete in **16.67ms**.

### 1.4 Current Data Flow

```
SimulationOrchestrator._tick()
    │
    ├──▶ PatientGenerator.generate_tick(4 samples)
    │
    └──▶ PipelineAdapter.process_patient()
              │
              ├──▶ MHRGuard.analyze()
              ├──▶ TrendAnalyzer.analyze()
              ├──▶ ExplanationEngine.explain()
              │
              └──▶ DataBridge.push_batch(snapshot)
                        │
                        └──▶ [POLL] UI Fragment reads every 100-333ms
                                    │
                                    └──▶ Plotly Figure JSON → HTTP → Browser
```

**Key Observation:** The backend already generates data at 4Hz (250ms intervals). The bottleneck is entirely in the **UI rendering pipeline**.

---

## Phase 2: Target Architecture ("To-Be" - TradingView Style)

### 2.1 How TradingView Achieves 60FPS

TradingView and similar professional trading platforms use a fundamentally different architecture:

| Aspect | TradingView Approach | Why It's Fast |
|--------|---------------------|---------------|
| **Rendering** | HTML5 Canvas / WebGL | Direct GPU access, O(1) draw calls regardless of data points |
| **Data Transfer** | WebSocket push | Server pushes data, no polling overhead |
| **Chart Library** | Lightweight-Charts (own library) | Zero-dependency, optimized for time series |
| **State Management** | Client-side (Redux/Zustand) | No server round-trip for UI state |
| **Data Format** | Binary (Protobuf/MessagePack) | 10x smaller than JSON, faster parsing |
| **Incremental Updates** | Delta/Diff streaming | Only send changed data, not full dataset |

### 2.2 Proposed V3.0 Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     V3.0 ARCHITECTURE (Professional Stack)                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   PYTHON BACKEND                   FASTAPI                   BROWSER       │
│   ┌─────────────────┐         ┌──────────────┐         ┌────────────────┐ │
│   │ Orchestrator    │──push──▶│  WebSocket   │══WS════▶│  React App     │ │
│   │ (Python Thread) │         │  /ws/stream  │         │  (Vite/TS)     │ │
│   └────────┬────────┘         └──────────────┘         └───────┬────────┘ │
│            │                                                    │         │
│   ┌────────▼────────┐         ┌──────────────┐         ┌───────▼────────┐ │
│   │ PipelineAdapter │         │  REST API    │         │ Lightweight-   │ │
│   │ (unchanged)     │         │  /api/...    │         │ Charts (WebGL) │ │
│   └─────────────────┘         └──────────────┘         └────────────────┘ │
│                                                                            │
│   [Pure Python - No UI concerns]    [API Layer]      [Pure JS - No Python]│
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Technology Recommendations

#### Frontend Framework: **React + TypeScript**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **React** | Massive ecosystem, best charting libraries, TypeScript support | Learning curve if unfamiliar | **Recommended** |
| Vue 3 | Simpler syntax, good for small teams | Fewer medical/charting libraries | Good alternative |
| Svelte | Fastest runtime, smallest bundle | Immature charting ecosystem | Not recommended for this use case |

**Recommendation:** React with TypeScript. The ecosystem advantage for real-time charting is decisive.

#### Charting Library: **Lightweight-Charts**

| Library | Render Mode | Performance | Medical Suitability |
|---------|-------------|-------------|---------------------|
| **Lightweight-Charts** (TradingView) | Canvas/WebGL | 60FPS, 100K+ points | Excellent - designed for time series |
| Recharts | SVG | ~15 FPS | Poor - too slow |
| Victory | SVG | ~15 FPS | Poor - too slow |
| Chart.js | Canvas | 30-60 FPS | Good - but less customizable |
| D3 + Canvas | Canvas | 60 FPS | Excellent - but requires custom code |
| uPlot | Canvas | 60 FPS, tiny bundle | Very Good - worth considering |

**Recommendation:** **Lightweight-Charts** for main CTG display (it's literally what TradingView uses). Alternative: **uPlot** if bundle size is critical.

#### Communication: **WebSocket (FastAPI)**

| Protocol | Latency | Bi-directional | Server Push | Verdict |
|----------|---------|----------------|-------------|---------|
| **WebSocket** | ~5ms | Yes | Yes | **Recommended** |
| SSE (Server-Sent Events) | ~10ms | No (one-way) | Yes | Good for read-only |
| HTTP Polling | 50-100ms | N/A | No | Current approach - too slow |
| WebRTC | ~1ms | Yes | Yes | Overkill for this use case |

**Recommendation:** **WebSocket** via FastAPI. Bi-directional is needed for event injection ("God Mode").

#### Backend Framework: **FastAPI**

```python
# Example WebSocket endpoint structure
@app.websocket("/ws/stream/{patient_id}")
async def patient_stream(websocket: WebSocket, patient_id: str):
    await websocket.accept()
    async for snapshot in data_bridge.subscribe(patient_id):
        await websocket.send_bytes(msgpack.packb(snapshot))
```

**Why FastAPI:**
- Native async/await (no GIL blocking for I/O)
- Built-in WebSocket support
- Auto-generated OpenAPI docs
- Same Python codebase - minimal learning curve

---

## Phase 3: Gap Analysis (What Changes?)

### 3.1 Files to DISCARD (Streamlit-Specific)

| File | Lines | Reason |
|------|-------|--------|
| `src/ui/simulation_app.py` | ~980 | Entire Streamlit app - replaced by React |
| `src/ui/app.py` | ~960 | Legacy Streamlit app |
| `src/ui/styles.py` | ~200 | Streamlit CSS injection |
| `src/ui/plots.py` | ~450 | Plotly figure builders - replaced by Lightweight-Charts |
| `src/ui/state_bridge.py` | ~580 | DataBridge singleton - replaced by WebSocket |
| `src/ui/__init__.py` | - | Package marker |

**Total Lines to Discard:** ~3,170 lines

### 3.2 Files to ADAPT (Wrap in API)

| File | Current Role | V3 Role | Changes Needed |
|------|--------------|---------|----------------|
| `src/simulation/core/orchestrator.py` | Direct DataBridge push | WebSocket broadcast | Add async broadcast method |
| `src/simulation/processing/pipeline_adapter.py` | Returns dict | Returns Pydantic model | Add FastAPI-compatible models |
| `src/config.py` | Python dataclasses | Shared config | Export as JSON for frontend |

**Estimated Changes:** ~200-300 lines modified

### 3.3 Files to PRESERVE (Core Logic - Zero Changes)

| Module | Files | Lines | Notes |
|--------|-------|-------|-------|
| **Rules Engine** | `src/rules/*.py` | ~800 | Pure Python, no UI coupling |
| **ML Models** | `src/models/*.py` | ~600 | MiniRocket, Fusion - untouched |
| **Analysis** | `src/analysis/*.py` | ~800 | TrendAnalyzer, Override - untouched |
| **Safety** | `src/safety/*.py` | ~500 | MHRGuard, SpectralAnalyzer - untouched |
| **Explainability** | `src/explainability/*.py` | ~600 | RuleExplainer, VisualMapper - untouched |
| **Data Processing** | `src/data/*.py` | ~400 | Preprocessing - untouched |
| **Generators** | `src/simulation/generators/*.py` | ~800 | FHR/UC generators - untouched |

**Total Lines Preserved:** ~4,500 lines (100% of core clinical logic)

### 3.4 New Files to CREATE

| File/Directory | Purpose | Estimated Lines |
|----------------|---------|-----------------|
| `api/main.py` | FastAPI app entry point | ~100 |
| `api/routers/patients.py` | REST endpoints for patient data | ~150 |
| `api/routers/websocket.py` | WebSocket stream handler | ~200 |
| `api/models/schemas.py` | Pydantic models (request/response) | ~200 |
| `api/services/broadcaster.py` | Async WebSocket broadcaster | ~150 |
| `frontend/` | Entire React application | ~3,000 |
| `frontend/src/components/CTGChart.tsx` | Lightweight-Charts wrapper | ~300 |
| `frontend/src/components/PatientCard.tsx` | Patient status card | ~150 |
| `frontend/src/hooks/useWebSocket.ts` | WebSocket connection hook | ~100 |
| `frontend/src/stores/patientStore.ts` | Zustand state management | ~150 |

**Total New Lines:** ~4,500 lines

---

## Phase 4: Execution Roadmap

### Step 1: Backend API Layer (FastAPI)

**Duration:** 3-4 days
**Risk:** Low

```
Tasks:
├── Create api/ directory structure
├── Define Pydantic models for PatientSnapshot, HighlightRegion
├── Implement REST endpoints:
│   ├── GET /api/patients - List all patients
│   ├── GET /api/patients/{id} - Get patient details
│   ├── POST /api/patients/{id}/inject - Inject event
│   └── GET /api/simulation/status - Get simulation state
├── Implement WebSocket endpoint:
│   └── WS /ws/stream - Real-time patient data stream
└── Add async wrapper around SimulationOrchestrator
```

**Deliverable:** FastAPI server that exposes all backend functionality via REST + WebSocket.

### Step 2: WebSocket Stream Implementation

**Duration:** 2-3 days
**Risk:** Medium (async complexity)

```
Tasks:
├── Create AsyncBroadcaster class
│   ├── subscribe(patient_id) -> AsyncGenerator
│   ├── broadcast(patient_id, data) -> None
│   └── Handle client disconnection gracefully
├── Modify Orchestrator to call broadcaster.broadcast()
├── Implement binary serialization (MessagePack)
└── Add connection health monitoring (ping/pong)
```

**Deliverable:** Backend pushes data to all connected clients at 4Hz with <10ms latency.

### Step 3: Frontend Scaffold (React + Vite)

**Duration:** 3-4 days
**Risk:** Low

```
Tasks:
├── Initialize project: npm create vite@latest frontend -- --template react-ts
├── Install dependencies:
│   ├── lightweight-charts (charting)
│   ├── zustand (state management)
│   ├── @tanstack/react-query (REST data fetching)
│   └── tailwindcss (styling)
├── Create base layout:
│   ├── WardView (grid of patients)
│   ├── DetailView (single patient focus)
│   └── ControlBar (simulation controls)
└── Implement WebSocket connection hook
```

**Deliverable:** React app skeleton with routing and state management.

### Step 4: Canvas Chart Integration

**Duration:** 4-5 days
**Risk:** Medium (custom chart requirements)

```
Tasks:
├── Create CTGChart component
│   ├── Dual-pane layout (FHR + UC)
│   ├── Baseline reference lines
│   ├── Red zone highlighting (markArea)
│   └── Real-time data streaming
├── Implement chart update logic:
│   ├── Efficient data append (not full redraw)
│   ├── Auto-scroll with user override
│   └── Zoom/pan preservation
├── Add category-based color coding
└── Performance testing (verify 60FPS)
```

**Deliverable:** Professional CTG monitor achieving 60FPS with 20 concurrent patients.

### Step 5: Feature Parity & Polish

**Duration:** 3-4 days
**Risk:** Low

```
Tasks:
├── Implement God Mode (event injection UI)
├── Add Hebrew/English i18n
├── Implement Trend Panel, Explanation Panel
├── Add alert sounds/notifications
├── Responsive design (tablet/mobile)
└── Error handling and offline mode
```

**Deliverable:** Full feature parity with current Streamlit UI.

---

## Effort & Cost Summary

### Development Effort

| Phase | Duration | Effort (Person-Days) |
|-------|----------|---------------------|
| Backend API | 3-4 days | 4 |
| WebSocket Stream | 2-3 days | 3 |
| Frontend Scaffold | 3-4 days | 4 |
| Canvas Charts | 4-5 days | 5 |
| Feature Parity | 3-4 days | 4 |
| Testing & QA | 2-3 days | 3 |
| **TOTAL** | **17-23 days** | **~23 person-days** |

### Skill Requirements

| Skill | Required Level | Notes |
|-------|----------------|-------|
| Python/FastAPI | Intermediate | Existing team likely has this |
| React/TypeScript | Intermediate | May need learning time |
| WebSocket Protocol | Basic | Well-documented |
| Canvas/WebGL Charting | Basic | Library handles complexity |

### Operational Changes

| Aspect | Current | V3 |
|--------|---------|-----|
| Deployment | Single `streamlit run` | Backend + Frontend (2 processes) |
| Hosting | Streamlit Cloud / single server | Separate API + Static hosting |
| Dependencies | `pip install` only | `pip` + `npm install` |
| Build Process | None | Frontend bundling (Vite) |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| React learning curve | Medium | Medium | Use component libraries (shadcn/ui) |
| WebSocket connection stability | Low | High | Implement reconnection logic |
| Chart performance issues | Low | High | Benchmark early, use uPlot fallback |
| Feature regression | Medium | Medium | Comprehensive test coverage |
| Increased operational complexity | High | Low | Docker Compose for deployment |

---

## Decision Framework

### When to Migrate (V3 Worth It)

- [ ] Clinical users complain about UI responsiveness
- [ ] You need to support >20 concurrent patients
- [ ] Mobile/tablet support is required
- [ ] You plan to add more real-time visualizations
- [ ] The project will be maintained long-term

### When to Stay with Streamlit (V2 Sufficient)

- [ ] Current performance is "good enough" for clinical use
- [ ] Development time is limited (hackathon constraints)
- [ ] Team lacks JavaScript/React experience
- [ ] Rapid prototyping is more important than polish
- [ ] Single-user deployment is acceptable

---

## Recommendation

**For a hackathon:** Stay with Streamlit V2.0. The current Pulse Architecture is functional, and 3-4 weeks of migration work is not justified for a demo.

**For production deployment:** Migrate to V3.0. The professional UX, scalability, and maintainability benefits outweigh the development cost.

**Hybrid approach:** Keep Streamlit for internal/demo use, build React frontend incrementally when resources allow.

---

## Appendix A: Alternative Quick Wins (Without Full Migration)

If full migration is too expensive, consider these improvements to the current Streamlit stack:

| Improvement | Effort | Impact |
|-------------|--------|--------|
| Switch Plotly → ECharts (already in deps) | 2 days | 2x FPS improvement |
| Reduce data points (downsample to 1Hz display) | 1 day | 3x FPS improvement |
| Use `st.empty()` containers for targeted updates | 1 day | Reduces flicker |
| Enable Plotly WebGL mode (`Scattergl`) | 0.5 days | 1.5x FPS improvement |
| Batch patient cards into single HTML render | 1 day | Reduces DOM operations |

**Combined effect:** Could achieve ~20-30 FPS without migration.

---

## Appendix B: File Inventory Summary

```
PRESERVE (Core Logic):     4,500 lines  │████████████████████│ 100%
DISCARD (Streamlit UI):    3,170 lines  │██████████████      │  70%
ADAPT (API Wrapper):         300 lines  │█                   │   7%
CREATE (New Frontend):     4,500 lines  │████████████████████│ 100%
───────────────────────────────────────────────────────────────
NET CHANGE:               +1,330 lines
```

---

*Document generated by AI Chief Software Architect*
*January 24, 2026*
