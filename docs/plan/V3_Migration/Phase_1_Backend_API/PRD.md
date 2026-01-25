# Phase 1: Backend API Layer (FastAPI) - PRD

**Phase:** 1 of 6
**Duration:** 3-4 days
**Priority:** Critical (Foundation for Phases 2-6)
**Risk Level:** Low
**Dependencies:** Phase 0 Complete

---

## 1. Overview

### 1.1 Purpose
Phase 1 creates the FastAPI backend that exposes the existing SentinelFetal pipeline via REST and WebSocket endpoints. This phase wraps the existing Python logic without modifying it, providing a clean API layer for the React frontend.

### 1.2 Goals
1. Create FastAPI application with proper structure
2. Define Pydantic models mirroring existing Python dataclasses
3. Implement REST endpoints for patient data and simulation control
4. Set up basic WebSocket endpoint structure (full implementation in Phase 2)
5. Bridge FastAPI to existing SimulationOrchestrator

### 1.3 Non-Goals
- Full WebSocket streaming logic (Phase 2)
- Frontend consumption of APIs (Phase 3+)
- Performance optimization (Phase 6)
- Production deployment (Phase 6)

---

## 2. User Stories

### US-1.1: List All Patients
**As a** frontend developer
**I want** a REST endpoint to get all patients
**So that** I can render the ward view grid

**Acceptance Criteria:**
- [ ] `GET /api/patients` returns list of all patients
- [ ] Response includes: patient_id, status, category, baseline, variability
- [ ] Response time < 50ms for 20 patients
- [ ] Empty array returned when no patients exist

### US-1.2: Get Patient Details
**As a** frontend developer
**I want** a REST endpoint to get detailed patient data
**So that** I can render the detail view

**Acceptance Criteria:**
- [ ] `GET /api/patients/{patient_id}` returns full patient data
- [ ] Response includes: FHR buffer (last N samples), UC buffer, findings, events
- [ ] Returns 404 if patient doesn't exist
- [ ] Response includes trend data and explanation

### US-1.3: Start/Stop Simulation
**As a** frontend developer
**I want** REST endpoints to control the simulation
**So that** I can implement start/pause/reset controls

**Acceptance Criteria:**
- [ ] `POST /api/simulation/start` starts the simulation
- [ ] `POST /api/simulation/pause` pauses the simulation
- [ ] `POST /api/simulation/resume` resumes from pause
- [ ] `POST /api/simulation/reset` resets all patients
- [ ] `GET /api/simulation/status` returns current state

### US-1.4: Inject Event (God Mode)
**As a** frontend developer
**I want** an endpoint to inject clinical events
**So that** I can implement the God Mode control panel

**Acceptance Criteria:**
- [ ] `POST /api/patients/{patient_id}/inject` triggers an event
- [ ] Request body specifies: event_type, severity, duration
- [ ] Returns 400 if patient is already in an event
- [ ] Returns 202 Accepted on success

### US-1.5: Configure Patient Count
**As a** frontend developer
**I want** to change the number of simulated patients
**So that** I can scale the demonstration

**Acceptance Criteria:**
- [ ] `PUT /api/simulation/config` updates patient count
- [ ] New patients are initialized with random profiles
- [ ] Existing patients are preserved when count increases
- [ ] Excess patients are removed when count decreases

### US-1.6: Get System Health
**As a** DevOps engineer
**I want** a health check endpoint
**So that** I can monitor the API's status

**Acceptance Criteria:**
- [ ] `GET /health` returns system status
- [ ] Response includes: API version, orchestrator state, uptime
- [ ] Response time < 10ms

---

## 3. Functional Requirements

### FR-1.1: API Structure
The API shall follow RESTful conventions:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| GET | `/api/patients` | List all patients |
| GET | `/api/patients/{id}` | Get patient details |
| POST | `/api/patients/{id}/inject` | Inject event |
| GET | `/api/simulation/status` | Get simulation state |
| POST | `/api/simulation/start` | Start simulation |
| POST | `/api/simulation/pause` | Pause simulation |
| POST | `/api/simulation/resume` | Resume simulation |
| POST | `/api/simulation/reset` | Reset simulation |
| PUT | `/api/simulation/config` | Update configuration |

### FR-1.2: Pydantic Models
All API responses shall use Pydantic models for validation and documentation:

```python
# Core models to implement
PatientSummary       # For list endpoint
PatientDetail        # For detail endpoint
PatientSnapshot      # Real-time state (used in WebSocket)
SimulationStatus     # Orchestrator state
EventInjectionRequest  # God Mode input
HighlightRegion      # Red zone data
TrendData            # Trend analysis result
ExplanationData      # Classification explanation
```

### FR-1.3: Orchestrator Bridge
The API shall interface with the existing SimulationOrchestrator:

```python
# Required integration points
orchestrator.start()
orchestrator.pause()
orchestrator.resume()
orchestrator.stop()
orchestrator.reset_all()
orchestrator.get_patient(patient_id)
orchestrator.get_all_patients()
orchestrator.inject_event(patient_id, event_type, params, duration)
orchestrator.set_patient_count(count)
```

### FR-1.4: Error Handling
The API shall return appropriate HTTP status codes:

| Status | Condition |
|--------|-----------|
| 200 | Successful GET |
| 201 | Resource created |
| 202 | Accepted (async action started) |
| 400 | Bad request (validation error) |
| 404 | Resource not found |
| 409 | Conflict (e.g., already running) |
| 500 | Internal server error |

### FR-1.5: CORS Configuration
The API shall allow cross-origin requests from the frontend:
- Allow origins: `http://localhost:3000`, `http://localhost:5173`
- Allow methods: GET, POST, PUT, DELETE, OPTIONS
- Allow headers: Content-Type, Authorization

---

## 4. Non-Functional Requirements

### NFR-1.1: Performance
- Response time for list endpoints: < 50ms
- Response time for detail endpoints: < 100ms
- Startup time: < 5 seconds

### NFR-1.2: Documentation
- OpenAPI (Swagger) documentation auto-generated
- Available at `/docs` (Swagger UI)
- Available at `/redoc` (ReDoc)

### NFR-1.3: Logging
- Request/response logging for debugging
- Error logging with stack traces
- Configurable log level via environment variable

### NFR-1.4: Type Safety
- All endpoints use typed Pydantic models
- mypy strict mode compliance
- No `Any` types in public API

---

## 5. Data Models

### 5.1 PatientSummary (List View)

| Field | Type | Description |
|-------|------|-------------|
| patient_id | str | Unique identifier (e.g., "Patient-1") |
| category | int | Classification category (1, 2, 3) |
| category_name | str | Human-readable category |
| baseline | float | Current FHR baseline (bpm) |
| variability | float | Current variability (bpm) |
| has_active_event | bool | Whether an event is in progress |
| mhr_alert | bool | MHR contamination suspected |
| last_update | datetime | Timestamp of last update |

### 5.2 PatientDetail (Detail View)

| Field | Type | Description |
|-------|------|-------------|
| patient_id | str | Unique identifier |
| category | int | Classification category |
| alert | dict | Current alert information |
| baseline | float | FHR baseline |
| variability | float | Variability |
| fhr_buffer | list[float] | Last N FHR samples |
| uc_buffer | list[float] | Last N UC samples |
| findings | dict | Detailed clinical findings |
| active_events | list[dict] | Currently active events |
| highlight_regions | list[HighlightRegion] | Red zones for visualization |
| trend_data | TrendData | 60-min trend analysis |
| explanation | ExplanationData | Classification explanation |
| mhr_alert | bool | MHR contamination flag |

### 5.3 SimulationStatus

| Field | Type | Description |
|-------|------|-------------|
| running | bool | Whether simulation is active |
| paused | bool | Whether simulation is paused |
| patient_count | int | Number of active patients |
| tick_count | int | Total ticks processed |
| uptime_seconds | float | Time since start |

### 5.4 EventInjectionRequest

| Field | Type | Description |
|-------|------|-------------|
| event_type | str | Event type (e.g., "LATE_DECEL") |
| severity | str | Severity level ("mild", "moderate", "severe") |
| duration_seconds | int | Duration in seconds |
| params | dict | Additional parameters (optional) |

---

## 6. API Response Examples

### 6.1 GET /api/patients

```json
{
  "patients": [
    {
      "patient_id": "Patient-1",
      "category": 1,
      "category_name": "Normal",
      "baseline": 142.5,
      "variability": 12.3,
      "has_active_event": false,
      "mhr_alert": false,
      "last_update": "2026-01-24T14:30:00Z"
    },
    {
      "patient_id": "Patient-2",
      "category": 2,
      "category_name": "Intermediate",
      "baseline": 155.2,
      "variability": 4.8,
      "has_active_event": true,
      "mhr_alert": false,
      "last_update": "2026-01-24T14:30:00Z"
    }
  ],
  "count": 2
}
```

### 6.2 GET /api/patients/Patient-1

```json
{
  "patient_id": "Patient-1",
  "category": 1,
  "alert": {
    "headline": "התראה ירוקה - קטגוריה 1 (תקין)",
    "recommendation": "מעקב שגרתי"
  },
  "baseline": 142.5,
  "variability": 12.3,
  "fhr_buffer": [141, 143, 142, 144, ...],
  "uc_buffer": [10, 12, 15, 18, ...],
  "findings": {
    "baseline": {"value": 142.5, "status": "normal"},
    "variability": {"value": 12.3, "category": "moderate"},
    "decelerations": []
  },
  "active_events": [],
  "highlight_regions": [],
  "trend_data": {
    "deterioration_score": 15,
    "variability_slope": 0.1,
    "decel_count_30min": 0,
    "late_decel_count_15min": 0,
    "alerts": []
  },
  "explanation": {
    "primary_reason": "Normal baseline and variability",
    "contributing_factors": ["Moderate variability (12.3 bpm)", "No decelerations"],
    "confidence": 0.92
  },
  "mhr_alert": false
}
```

### 6.3 POST /api/patients/Patient-1/inject

Request:
```json
{
  "event_type": "LATE_DECEL",
  "severity": "moderate",
  "duration_seconds": 120
}
```

Response (202 Accepted):
```json
{
  "status": "accepted",
  "message": "Event injection started",
  "patient_id": "Patient-1",
  "event_type": "LATE_DECEL",
  "expected_duration_seconds": 120
}
```

---

## 7. Dependencies

### 7.1 External Dependencies
- FastAPI >= 0.109.0
- Pydantic >= 2.5.0
- Uvicorn >= 0.27.0

### 7.2 Internal Dependencies
- Phase 0 complete (project structure)
- Existing SimulationOrchestrator
- Existing PatientGenerator
- Existing PipelineAdapter

---

## 8. Acceptance Criteria Summary

| ID | Criteria | Verification Method |
|----|----------|---------------------|
| AC-1.1 | `/health` returns 200 with status | `curl` test |
| AC-1.2 | `/api/patients` returns patient list | `curl` test |
| AC-1.3 | `/api/patients/{id}` returns detail or 404 | `curl` test |
| AC-1.4 | `/api/simulation/start` starts orchestrator | State change verification |
| AC-1.5 | `/api/patients/{id}/inject` triggers event | FHR pattern change |
| AC-1.6 | Swagger docs available at `/docs` | Browser verification |
| AC-1.7 | All responses use Pydantic models | mypy verification |
| AC-1.8 | CORS headers present | Browser console check |

---

## 9. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Orchestrator threading issues | Medium | High | Use thread-safe adapter pattern |
| Pydantic v2 migration issues | Low | Medium | Follow v2 migration guide |
| CORS misconfiguration | Low | Low | Test with actual browser |

---

## 10. Timeline

| Day | Tasks |
|-----|-------|
| Day 1 | FastAPI structure, Pydantic models |
| Day 2 | REST endpoints (patients, simulation) |
| Day 3 | Event injection, orchestrator bridge |
| Day 4 | Testing, documentation, CORS config |

---

## 11. Deliverables Checklist

- [x] `api/main.py` with FastAPI app
- [x] `api/routers/patients.py` with patient endpoints
- [x] `api/routers/simulation.py` with simulation endpoints
- [x] `api/models/schemas.py` with all Pydantic models
- [x] `api/services/orchestrator_adapter.py` for thread-safe access
- [x] `api/tests/test_api.py` with endpoint tests
- [x] OpenAPI docs accessible at `/api/docs`

---

## 12. Implementation Status

### ✅ Phase 1 Complete - January 2025

**Completed:**
1. **OrchestratorAdapter** (`api/services/orchestrator_adapter.py`)
   - Thread-safe singleton bridging FastAPI to SimulationOrchestrator
   - Full integration with existing pipeline and DataBridge
   - Methods: start/stop/pause/resume/reset, patient access, event injection
   
2. **Dependencies** (`api/dependencies.py`)
   - FastAPI dependency injection with type hints
   - `OrchestratorDep` and `SettingsDep` type aliases
   
3. **Pydantic Schemas** (`api/models/schemas.py`)
   - Full schema set with enums: FIGOCategory, EventTypeEnum, SeverityEnum
   - PatientMetrics, PatientSnapshot, PatientSummary, AlertInfo
   - SimulationStatus, SimulationConfig, SimulationCommand, SimulationResponse
   - WSMessage, WSPatientUpdate, WSBatchUpdate
   - EventInjection, EventInjectionResponse
   - HealthCheck, APIError
   
4. **Simulation Router** (`api/routers/simulation.py`)
   - GET /status - simulation state with tick count
   - POST /start, /stop, /pause, /resume, /reset
   - POST /command - unified action endpoint
   - PATCH /config - patient count, speed multiplier
   
5. **Patients Router** (`api/routers/patients.py`)
   - GET / - full patient list with configurable history
   - GET /summary - lightweight overview
   - GET /{patient_id} - detailed snapshot
   - GET /{patient_id}/history - extended history data
   - POST /{patient_id}/event - God Mode event injection
   
6. **Main App** (`api/main.py`)
   - Lifespan handler initializes/shuts down orchestrator
   - CORS, logging, health check
   - Docs at /api/docs, /api/redoc
   
7. **Tests** (`api/tests/test_api.py`)
   - Health check, root endpoint
   - Simulation start/stop/pause/resume/config
   - Patient list, summary, not found
   - Schema validation tests

**Deviations from Plan:**
- `/api/patients/{id}/inject` → `/api/patients/{id}/event` for clearer naming
- Added `/api/patients/summary` lightweight endpoint
- Added `/api/patients/{id}/history` for extended data
- Added `/api/simulation/command` unified endpoint
- Docs moved to `/api/docs` from `/docs`

---

*End of Phase 1 PRD*
