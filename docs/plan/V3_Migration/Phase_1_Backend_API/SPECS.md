# Phase 1: Backend API Layer (FastAPI) - Technical Specifications

**Phase:** 1 of 6
**Document Type:** Technical Specifications
**Target Audience:** Backend Developers

---

## 1. File Structure

```
api/
├── __init__.py
├── main.py                      # FastAPI application entry point
├── config.py                    # API configuration (from env vars)
├── dependencies.py              # Dependency injection
│
├── routers/
│   ├── __init__.py
│   ├── patients.py              # /api/patients/* routes
│   ├── simulation.py            # /api/simulation/* routes
│   └── websocket.py             # /ws/* routes (placeholder for Phase 2)
│
├── models/
│   ├── __init__.py
│   ├── schemas.py               # Pydantic request/response models
│   └── enums.py                 # Shared enumerations
│
├── services/
│   ├── __init__.py
│   ├── orchestrator_adapter.py  # Thread-safe orchestrator access
│   └── patient_mapper.py        # Convert internal models to API schemas
│
└── tests/
    ├── __init__.py
    ├── conftest.py              # pytest fixtures
    ├── test_patients.py
    └── test_simulation.py
```

---

## 2. Core Implementation

### 2.1 api/main.py

```python
"""
SentinelFetal API - FastAPI Application Entry Point

This module initializes the FastAPI application and configures
middleware, CORS, and route registration.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.config import settings
from api.routers import patients, simulation, websocket
from api.services.orchestrator_adapter import OrchestratorAdapter

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes the orchestrator on startup and cleans up on shutdown.
    """
    logger.info("Starting SentinelFetal API...")

    # Initialize orchestrator adapter (singleton)
    adapter = OrchestratorAdapter.get_instance()
    adapter.initialize(patient_count=settings.sim_patients_default)

    logger.info(f"Orchestrator initialized with {settings.sim_patients_default} patients")

    yield  # Application runs here

    # Cleanup
    logger.info("Shutting down SentinelFetal API...")
    adapter.shutdown()


app = FastAPI(
    title="SentinelFetal API",
    version="3.0.0",
    description="Real-time CTG monitoring with AI classification",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # Vite dev server
        "http://localhost:5173",     # Vite alternative port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(patients.router, prefix="/api/patients", tags=["Patients"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns API status and orchestrator state for monitoring.
    """
    adapter = OrchestratorAdapter.get_instance()
    return {
        "status": "ok",
        "version": "3.0.0",
        "orchestrator": {
            "running": adapter.is_running(),
            "patient_count": adapter.get_patient_count(),
        },
    }
```

### 2.2 api/config.py

```python
"""
API Configuration

Loads configuration from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """API configuration settings."""

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True
    api_reload: bool = True

    # WebSocket
    ws_heartbeat_interval: int = 30
    ws_max_connections: int = 100

    # Simulation
    sim_patients_default: int = 4
    sim_tick_rate_hz: float = 4.0

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
```

### 2.3 api/dependencies.py

```python
"""
FastAPI Dependency Injection

Provides reusable dependencies for route handlers.
"""

from fastapi import Depends, HTTPException, status
from api.services.orchestrator_adapter import OrchestratorAdapter


def get_orchestrator() -> OrchestratorAdapter:
    """
    Get the orchestrator adapter instance.

    This is a singleton that provides thread-safe access to the
    simulation orchestrator.
    """
    return OrchestratorAdapter.get_instance()


def get_patient_or_404(
    patient_id: str,
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Get a patient by ID or raise 404.

    Use as a dependency in routes that require a valid patient.
    """
    patient = orchestrator.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found",
        )
    return patient
```

---

## 3. Pydantic Models

### 3.1 api/models/schemas.py

```python
"""
Pydantic Models for API Request/Response Schemas

These models define the structure of all API inputs and outputs,
providing automatic validation and documentation.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Enums (could move to enums.py)
# =============================================================================

from enum import Enum


class Category(int, Enum):
    """CTG classification category."""
    NORMAL = 1
    INTERMEDIATE = 2
    PATHOLOGICAL = 3


class EventType(str, Enum):
    """Injectable event types."""
    LATE_DECEL = "LATE_DECEL"
    VARIABLE_DECEL = "VARIABLE_DECEL"
    EARLY_DECEL = "EARLY_DECEL"
    PROLONGED_DECEL = "PROLONGED_DECEL"
    TACHYCARDIA = "TACHYCARDIA"
    BRADYCARDIA = "BRADYCARDIA"
    REDUCED_VARIABILITY = "REDUCED_VARIABILITY"
    SINUSOIDAL = "SINUSOIDAL"
    TACHYSYSTOLE = "TACHYSYSTOLE"


class Severity(str, Enum):
    """Event severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


# =============================================================================
# Nested Models
# =============================================================================

class HighlightRegion(BaseModel):
    """
    Red zone region for chart visualization.

    Represents a time range that should be highlighted on the CTG chart.
    """
    start_idx: int = Field(..., description="Start sample index")
    end_idx: int = Field(..., description="End sample index")
    severity: str = Field(default="warning", description="Severity level")
    label: str = Field(default="", description="Display label")
    color: Optional[str] = Field(default=None, description="Override color (CSS)")


class TrendData(BaseModel):
    """60-minute trend analysis data."""
    deterioration_score: float = Field(..., ge=0, le=100, description="0-100 score")
    variability_slope: float = Field(..., description="Variability trend slope")
    decel_count_30min: int = Field(default=0, ge=0)
    late_decel_count_15min: int = Field(default=0, ge=0)
    alerts: list[dict[str, Any]] = Field(default_factory=list)


class ExplanationData(BaseModel):
    """Classification explanation for XAI."""
    primary_reason: str = Field(default="")
    contributing_factors: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)


class AlertInfo(BaseModel):
    """Alert information for UI display."""
    headline: str = Field(..., description="Hebrew headline")
    recommendation: str = Field(..., description="Hebrew recommendation")
    category: int = Field(..., ge=1, le=3)


class FindingsInfo(BaseModel):
    """Detailed clinical findings."""
    baseline: dict[str, Any] = Field(default_factory=dict)
    variability: dict[str, Any] = Field(default_factory=dict)
    decelerations: list[dict[str, Any]] = Field(default_factory=list)


class ActiveEvent(BaseModel):
    """Currently active simulation event."""
    event_type: str
    start_time: datetime
    duration_seconds: int
    remaining_seconds: Optional[int] = None


# =============================================================================
# Response Models
# =============================================================================

class PatientSummary(BaseModel):
    """
    Patient summary for list view.

    Minimal data for rendering patient cards in the ward grid.
    """
    patient_id: str = Field(..., example="Patient-1")
    category: int = Field(..., ge=1, le=3, example=1)
    category_name: str = Field(..., example="Normal")
    baseline: float = Field(..., example=142.5)
    variability: float = Field(..., example=12.3)
    has_active_event: bool = Field(default=False)
    mhr_alert: bool = Field(default=False)
    last_update: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "Patient-1",
                "category": 1,
                "category_name": "Normal",
                "baseline": 142.5,
                "variability": 12.3,
                "has_active_event": False,
                "mhr_alert": False,
                "last_update": "2026-01-24T14:30:00Z"
            }
        }


class PatientListResponse(BaseModel):
    """Response for GET /api/patients."""
    patients: list[PatientSummary]
    count: int


class PatientDetail(BaseModel):
    """
    Detailed patient data for single-patient view.

    Includes full FHR/UC buffers and all analysis results.
    """
    patient_id: str
    category: int = Field(..., ge=1, le=3)
    alert: AlertInfo
    baseline: float
    variability: float
    fhr_buffer: list[float] = Field(..., max_length=4800)  # 20 min @ 4Hz
    uc_buffer: list[float] = Field(..., max_length=4800)
    findings: FindingsInfo
    active_events: list[ActiveEvent] = Field(default_factory=list)
    highlight_regions: list[HighlightRegion] = Field(default_factory=list)
    trend_data: Optional[TrendData] = None
    explanation: Optional[ExplanationData] = None
    mhr_alert: bool = Field(default=False)


class SimulationStatus(BaseModel):
    """Current simulation state."""
    running: bool
    paused: bool
    patient_count: int
    tick_count: int
    uptime_seconds: float


# =============================================================================
# Request Models
# =============================================================================

class EventInjectionRequest(BaseModel):
    """Request body for event injection (God Mode)."""
    event_type: EventType
    severity: Severity = Severity.MODERATE
    duration_seconds: int = Field(default=120, ge=30, le=600)
    params: Optional[dict[str, Any]] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "LATE_DECEL",
                "severity": "moderate",
                "duration_seconds": 120
            }
        }


class EventInjectionResponse(BaseModel):
    """Response for successful event injection."""
    status: str = "accepted"
    message: str
    patient_id: str
    event_type: str
    expected_duration_seconds: int


class SimulationConfigUpdate(BaseModel):
    """Request body for simulation configuration update."""
    patient_count: Optional[int] = Field(default=None, ge=1, le=50)
    tick_rate_hz: Optional[float] = Field(default=None, ge=1.0, le=10.0)


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None
```

---

## 4. Router Implementations

### 4.1 api/routers/patients.py

```python
"""
Patient Routes

Endpoints for retrieving and manipulating patient data.
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_orchestrator, get_patient_or_404
from api.models.schemas import (
    PatientSummary,
    PatientListResponse,
    PatientDetail,
    EventInjectionRequest,
    EventInjectionResponse,
    HighlightRegion,
    TrendData,
    ExplanationData,
    AlertInfo,
    FindingsInfo,
    ActiveEvent,
)
from api.services.orchestrator_adapter import OrchestratorAdapter
from api.services.patient_mapper import PatientMapper

router = APIRouter()


@router.get("", response_model=PatientListResponse)
async def list_patients(
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Get all patients.

    Returns a summary of each patient for the ward view grid.
    """
    patients = orchestrator.get_all_patients()
    summaries = [PatientMapper.to_summary(p) for p in patients]
    return PatientListResponse(patients=summaries, count=len(summaries))


@router.get("/{patient_id}", response_model=PatientDetail)
async def get_patient(
    patient_id: str,
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Get detailed patient data.

    Returns full FHR/UC buffers and all analysis results for
    the detail view.
    """
    patient = orchestrator.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found",
        )

    snapshot = orchestrator.get_patient_snapshot(patient_id)
    return PatientMapper.to_detail(patient, snapshot)


@router.post(
    "/{patient_id}/inject",
    response_model=EventInjectionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def inject_event(
    patient_id: str,
    request: EventInjectionRequest,
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Inject a clinical event (God Mode).

    Triggers a simulated event on the specified patient.
    The event will affect the FHR/UC signals for the specified duration.
    """
    patient = orchestrator.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found",
        )

    # Check if patient already has an active event
    if orchestrator.has_active_event(patient_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patient '{patient_id}' already has an active event",
        )

    # Get event parameters based on severity
    params = PatientMapper.get_event_params(request.event_type, request.severity)
    if request.params:
        params.update(request.params)

    # Inject the event
    orchestrator.inject_event(
        patient_id=patient_id,
        event_type=request.event_type.value,
        params=params,
        duration=request.duration_seconds,
    )

    return EventInjectionResponse(
        status="accepted",
        message=f"Event '{request.event_type.value}' injection started",
        patient_id=patient_id,
        event_type=request.event_type.value,
        expected_duration_seconds=request.duration_seconds,
    )


@router.get("/{patient_id}/fhr", response_model=list[float])
async def get_patient_fhr(
    patient_id: str,
    limit: int = 2400,  # 10 minutes @ 4Hz
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Get raw FHR buffer.

    Useful for lightweight polling or debugging.
    """
    patient = orchestrator.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found",
        )

    fhr = patient.fhr_buffer[-limit:] if len(patient.fhr_buffer) > limit else patient.fhr_buffer
    return list(fhr)
```

### 4.2 api/routers/simulation.py

```python
"""
Simulation Control Routes

Endpoints for controlling the simulation lifecycle.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_orchestrator
from api.models.schemas import SimulationStatus, SimulationConfigUpdate
from api.services.orchestrator_adapter import OrchestratorAdapter

router = APIRouter()


@router.get("/status", response_model=SimulationStatus)
async def get_simulation_status(
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Get current simulation status.

    Returns whether the simulation is running, paused, and statistics.
    """
    return SimulationStatus(
        running=orchestrator.is_running(),
        paused=orchestrator.is_paused(),
        patient_count=orchestrator.get_patient_count(),
        tick_count=orchestrator.get_tick_count(),
        uptime_seconds=orchestrator.get_uptime_seconds(),
    )


@router.post("/start", response_model=SimulationStatus)
async def start_simulation(
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Start the simulation.

    Begins generating FHR/UC signals for all patients.
    """
    if orchestrator.is_running() and not orchestrator.is_paused():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Simulation is already running",
        )

    orchestrator.start()
    return await get_simulation_status(orchestrator)


@router.post("/pause", response_model=SimulationStatus)
async def pause_simulation(
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Pause the simulation.

    Stops signal generation while preserving state.
    """
    if not orchestrator.is_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Simulation is not running",
        )

    if orchestrator.is_paused():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Simulation is already paused",
        )

    orchestrator.pause()
    return await get_simulation_status(orchestrator)


@router.post("/resume", response_model=SimulationStatus)
async def resume_simulation(
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Resume a paused simulation.
    """
    if not orchestrator.is_paused():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Simulation is not paused",
        )

    orchestrator.resume()
    return await get_simulation_status(orchestrator)


@router.post("/reset", response_model=SimulationStatus)
async def reset_simulation(
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Reset the simulation.

    Stops the simulation and reinitializes all patients.
    """
    orchestrator.reset()
    return await get_simulation_status(orchestrator)


@router.put("/config", response_model=SimulationStatus)
async def update_simulation_config(
    config: SimulationConfigUpdate,
    orchestrator: OrchestratorAdapter = Depends(get_orchestrator),
):
    """
    Update simulation configuration.

    Allows changing the number of patients dynamically.
    """
    if config.patient_count is not None:
        orchestrator.set_patient_count(config.patient_count)

    return await get_simulation_status(orchestrator)
```

### 4.3 api/routers/websocket.py (Placeholder)

```python
"""
WebSocket Routes (Placeholder)

Full implementation in Phase 2.
"""

from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket stream endpoint.

    Placeholder - full implementation in Phase 2.
    """
    await websocket.accept()
    await websocket.send_json({
        "message": "WebSocket connected",
        "note": "Full streaming implementation in Phase 2"
    })
    await websocket.close()
```

---

## 5. Orchestrator Adapter

### 5.1 api/services/orchestrator_adapter.py

```python
"""
Orchestrator Adapter

Provides thread-safe access to the SimulationOrchestrator.
This adapter pattern isolates the FastAPI layer from the
internal simulation implementation.
"""

import threading
import time
from typing import Any, Optional
import logging

# Import existing orchestrator
from src.simulation.core.orchestrator import SimulationOrchestrator
from src.ui.state_bridge import DataBridge, PatientSnapshot

logger = logging.getLogger(__name__)


class OrchestratorAdapter:
    """
    Thread-safe singleton adapter for SimulationOrchestrator.

    This class provides:
    - Singleton pattern for global access
    - Thread-safe method calls
    - Clean API for FastAPI routes
    """

    _instance: Optional["OrchestratorAdapter"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "OrchestratorAdapter":
        """Get the singleton instance."""
        return cls()

    def initialize(self, patient_count: int = 4) -> None:
        """
        Initialize the orchestrator.

        This should be called once during application startup.
        """
        if self._initialized:
            logger.warning("OrchestratorAdapter already initialized")
            return

        self._orchestrator = SimulationOrchestrator()
        self._orchestrator.set_patient_count(patient_count)
        self._data_bridge = DataBridge.get_instance()
        self._start_time: Optional[float] = None
        self._tick_count = 0
        self._initialized = True

        logger.info(f"OrchestratorAdapter initialized with {patient_count} patients")

    def shutdown(self) -> None:
        """Clean shutdown of the orchestrator."""
        if hasattr(self, '_orchestrator'):
            self._orchestrator.stop()
            logger.info("OrchestratorAdapter shut down")

    # =========================================================================
    # Simulation Control
    # =========================================================================

    def start(self) -> None:
        """Start the simulation."""
        self._orchestrator.start()
        self._start_time = time.time()

    def pause(self) -> None:
        """Pause the simulation."""
        self._orchestrator.pause()

    def resume(self) -> None:
        """Resume the simulation."""
        self._orchestrator.resume()

    def reset(self) -> None:
        """Reset the simulation."""
        self._orchestrator.stop()
        self._orchestrator.reset_all()
        self._start_time = None
        self._tick_count = 0

    # =========================================================================
    # State Queries
    # =========================================================================

    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._orchestrator._running

    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self._orchestrator._paused

    def get_patient_count(self) -> int:
        """Get current patient count."""
        return len(self._orchestrator._patients)

    def get_tick_count(self) -> int:
        """Get total tick count."""
        return self._orchestrator._tick_count if hasattr(self._orchestrator, '_tick_count') else 0

    def get_uptime_seconds(self) -> float:
        """Get simulation uptime."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    # =========================================================================
    # Patient Access
    # =========================================================================

    def get_patient(self, patient_id: str) -> Optional[Any]:
        """Get a patient by ID."""
        return self._orchestrator.get_patient(patient_id)

    def get_all_patients(self) -> list[Any]:
        """Get all patients."""
        return list(self._orchestrator._patients.values())

    def get_patient_snapshot(self, patient_id: str) -> Optional[PatientSnapshot]:
        """Get the latest snapshot for a patient from DataBridge."""
        return self._data_bridge.get_latest(patient_id)

    def has_active_event(self, patient_id: str) -> bool:
        """Check if patient has an active event."""
        patient = self.get_patient(patient_id)
        if patient is None:
            return False
        active_events = patient.get_active_events()
        return len(active_events) > 0

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_patient_count(self, count: int) -> None:
        """Update patient count."""
        self._orchestrator.set_patient_count(count)

    # =========================================================================
    # Event Injection
    # =========================================================================

    def inject_event(
        self,
        patient_id: str,
        event_type: str,
        params: dict[str, Any],
        duration: int,
    ) -> None:
        """Inject an event on a patient."""
        self._orchestrator.inject_event(patient_id, event_type, params, duration)
```

### 5.2 api/services/patient_mapper.py

```python
"""
Patient Mapper

Converts internal patient models to API response schemas.
"""

from datetime import datetime
from typing import Any, Optional

from api.models.schemas import (
    PatientSummary,
    PatientDetail,
    HighlightRegion,
    TrendData,
    ExplanationData,
    AlertInfo,
    FindingsInfo,
    ActiveEvent,
    EventType,
    Severity,
)
from src.ui.state_bridge import PatientSnapshot
from src.config import HEBREW


class PatientMapper:
    """Maps internal patient data to API schemas."""

    CATEGORY_NAMES = {
        1: "Normal",
        2: "Intermediate",
        3: "Pathological",
    }

    @classmethod
    def to_summary(cls, patient: Any) -> PatientSummary:
        """Convert patient to summary for list view."""
        findings = getattr(patient, 'latest_findings', {}) or {}
        baseline_info = findings.get('baseline', {})
        variability_info = findings.get('variability', {})

        category = getattr(patient, 'category', 1)
        baseline = baseline_info.get('value', patient.config.baseline_fhr)
        variability = variability_info.get('value', patient.config.baseline_variability)

        return PatientSummary(
            patient_id=patient.patient_id,
            category=category,
            category_name=cls.CATEGORY_NAMES.get(category, "Unknown"),
            baseline=round(baseline, 1),
            variability=round(variability, 1),
            has_active_event=len(patient.get_active_events()) > 0,
            mhr_alert=getattr(patient, 'mhr_alert', False),
            last_update=datetime.utcnow(),
        )

    @classmethod
    def to_detail(
        cls,
        patient: Any,
        snapshot: Optional[PatientSnapshot] = None,
    ) -> PatientDetail:
        """Convert patient to detailed response."""
        findings = getattr(patient, 'latest_findings', {}) or {}
        category = snapshot.category if snapshot else getattr(patient, 'category', 1)

        # Build alert info
        if category == 1:
            headline = HEBREW.HEADLINE_CATEGORY_1
            recommendation = HEBREW.REC_NORMAL
        elif category == 2:
            headline = HEBREW.HEADLINE_CATEGORY_2
            recommendation = HEBREW.REC_INTERMEDIATE
        else:
            headline = HEBREW.HEADLINE_CATEGORY_3
            recommendation = HEBREW.REC_PATHOLOGICAL

        alert = AlertInfo(
            headline=headline,
            recommendation=recommendation,
            category=category,
        )

        # Extract buffers
        fhr_buffer = list(patient.fhr_buffer) if hasattr(patient, 'fhr_buffer') else []
        uc_buffer = list(patient.uc_buffer) if hasattr(patient, 'uc_buffer') else []

        # Build findings info
        findings_info = FindingsInfo(
            baseline=findings.get('baseline', {}),
            variability=findings.get('variability', {}),
            decelerations=findings.get('decelerations', []),
        )

        # Active events
        active_events = [
            ActiveEvent(
                event_type=e.event_type.name,
                start_time=datetime.fromtimestamp(e.start_time),
                duration_seconds=e.duration,
            )
            for e in patient.get_active_events()
        ]

        # Highlight regions from snapshot
        highlight_regions = []
        if snapshot and snapshot.highlight_regions:
            highlight_regions = [
                HighlightRegion(**r) if isinstance(r, dict) else r
                for r in snapshot.highlight_regions
            ]

        # Trend data
        trend_data = None
        if snapshot and snapshot.trend_data:
            trend_data = TrendData(**snapshot.trend_data)

        # Explanation
        explanation = None
        if snapshot and snapshot.explanation:
            explanation = ExplanationData(**snapshot.explanation)

        return PatientDetail(
            patient_id=patient.patient_id,
            category=category,
            alert=alert,
            baseline=snapshot.baseline if snapshot else patient.config.baseline_fhr,
            variability=snapshot.variability if snapshot else patient.config.baseline_variability,
            fhr_buffer=fhr_buffer[-4800:],  # Last 20 min
            uc_buffer=uc_buffer[-4800:],
            findings=findings_info,
            active_events=active_events,
            highlight_regions=highlight_regions,
            trend_data=trend_data,
            explanation=explanation,
            mhr_alert=snapshot.mhr_alert if snapshot else False,
        )

    @classmethod
    def get_event_params(cls, event_type: EventType, severity: Severity) -> dict[str, Any]:
        """Get event parameters based on type and severity."""
        severity_multipliers = {
            Severity.MILD: 0.7,
            Severity.MODERATE: 1.0,
            Severity.SEVERE: 1.3,
        }
        multiplier = severity_multipliers.get(severity, 1.0)

        base_params = {
            EventType.LATE_DECEL: {"depth": 25 * multiplier, "count": 3},
            EventType.VARIABLE_DECEL: {"depth": 40 * multiplier, "count": 2},
            EventType.EARLY_DECEL: {"depth": 15 * multiplier, "count": 2},
            EventType.PROLONGED_DECEL: {"depth": 30 * multiplier, "duration": 120},
            EventType.TACHYCARDIA: {"target_bpm": 170 + 10 * multiplier},
            EventType.BRADYCARDIA: {"target_bpm": 100 - 10 * multiplier},
            EventType.REDUCED_VARIABILITY: {"target_variability": 3 / multiplier},
            EventType.SINUSOIDAL: {"amplitude": 10, "frequency": 0.05},
            EventType.TACHYSYSTOLE: {"contractions_per_10min": int(6 * multiplier)},
        }

        return base_params.get(event_type, {})
```

---

## 6. Tests

### 6.1 api/tests/conftest.py

```python
"""
Test Fixtures

Shared fixtures for API tests.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def started_simulation(client):
    """Start simulation and return client."""
    client.post("/api/simulation/start")
    yield client
    client.post("/api/simulation/reset")
```

### 6.2 api/tests/test_patients.py

```python
"""
Patient Endpoint Tests
"""

import pytest


def test_list_patients_empty(client):
    """Test listing patients returns array."""
    response = client.get("/api/patients")
    assert response.status_code == 200
    data = response.json()
    assert "patients" in data
    assert "count" in data


def test_get_patient_not_found(client):
    """Test 404 for missing patient."""
    response = client.get("/api/patients/nonexistent")
    assert response.status_code == 404


def test_inject_event_not_found(client):
    """Test 404 for injecting on missing patient."""
    response = client.post(
        "/api/patients/nonexistent/inject",
        json={"event_type": "LATE_DECEL", "duration_seconds": 60}
    )
    assert response.status_code == 404


def test_inject_event_success(started_simulation):
    """Test successful event injection."""
    # First get a valid patient ID
    response = started_simulation.get("/api/patients")
    patients = response.json()["patients"]
    if not patients:
        pytest.skip("No patients available")

    patient_id = patients[0]["patient_id"]

    response = started_simulation.post(
        f"/api/patients/{patient_id}/inject",
        json={"event_type": "LATE_DECEL", "severity": "moderate", "duration_seconds": 60}
    )
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
```

### 6.3 api/tests/test_simulation.py

```python
"""
Simulation Endpoint Tests
"""


def test_get_status(client):
    """Test getting simulation status."""
    response = client.get("/api/simulation/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "paused" in data
    assert "patient_count" in data


def test_start_simulation(client):
    """Test starting simulation."""
    response = client.post("/api/simulation/start")
    assert response.status_code == 200
    assert response.json()["running"] is True

    # Cleanup
    client.post("/api/simulation/reset")


def test_pause_without_start(client):
    """Test pause without start returns 409."""
    client.post("/api/simulation/reset")  # Ensure stopped
    response = client.post("/api/simulation/pause")
    assert response.status_code == 409


def test_update_config(client):
    """Test updating patient count."""
    response = client.put(
        "/api/simulation/config",
        json={"patient_count": 8}
    )
    assert response.status_code == 200
    assert response.json()["patient_count"] == 8
```

---

## 7. Running the API

### 7.1 Development Mode

```bash
# From project root
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 7.2 Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# List patients
curl http://localhost:8000/api/patients

# Start simulation
curl -X POST http://localhost:8000/api/simulation/start

# Get status
curl http://localhost:8000/api/simulation/status
```

### 7.3 Swagger UI

Open `http://localhost:8000/api/docs` in browser for interactive API documentation.

---

## 8. Implementation Status

### ✅ Phase 1 Complete - January 2025

**Files Implemented:**

| File | Status | Notes |
|------|--------|-------|
| `api/main.py` | ✅ Complete | Lifespan handler, CORS, logging, health check |
| `api/config.py` | ✅ Complete | pydantic-settings with SENTINEL_ prefix |
| `api/dependencies.py` | ✅ Complete | OrchestratorDep, SettingsDep type aliases |
| `api/routers/patients.py` | ✅ Complete | List, summary, detail, history, event injection |
| `api/routers/simulation.py` | ✅ Complete | start/stop/pause/resume/reset/command/config |
| `api/routers/websocket.py` | ⏳ Stub | Placeholder for Phase 2 |
| `api/models/schemas.py` | ✅ Complete | Full Pydantic v2 schemas with enums |
| `api/services/orchestrator_adapter.py` | ✅ Complete | Thread-safe singleton, full orchestrator bridge |
| `api/services/broadcaster.py` | ⏳ Stub | WebSocket broadcasting (Phase 2) |
| `api/tests/test_api.py` | ✅ Complete | Comprehensive endpoint tests |

**API Endpoints Implemented:**

| Method | Endpoint | Status |
|--------|----------|--------|
| GET | `/` | ✅ Root info |
| GET | `/api/health` | ✅ Health check |
| GET | `/api/patients` | ✅ List all patients |
| GET | `/api/patients/summary` | ✅ Lightweight summary |
| GET | `/api/patients/{id}` | ✅ Patient detail |
| GET | `/api/patients/{id}/history` | ✅ Extended history |
| POST | `/api/patients/{id}/event` | ✅ Event injection |
| GET | `/api/simulation/status` | ✅ Simulation state |
| POST | `/api/simulation/start` | ✅ Start simulation |
| POST | `/api/simulation/stop` | ✅ Stop simulation |
| POST | `/api/simulation/pause` | ✅ Pause simulation |
| POST | `/api/simulation/resume` | ✅ Resume simulation |
| POST | `/api/simulation/reset` | ✅ Reset simulation |
| POST | `/api/simulation/command` | ✅ Unified command |
| PATCH | `/api/simulation/config` | ✅ Update config |

**Verified Working:**
```bash
python -c "from api.main import app; print('Import OK')"
# Output: Import OK
```

**Deviations from Specs:**
1. `enums.py` merged into `schemas.py` for simplicity
2. `patient_mapper.py` merged into `patients.py` as helper functions
3. Docs URL changed from `/docs` to `/api/docs` for consistency
4. Added `/api/patients/summary` lightweight endpoint (not in original spec)
5. Added `/api/simulation/command` unified endpoint (not in original spec)

---

*End of Phase 1 Technical Specifications*
