"""
Pydantic Schemas
================
Data models for API request/response validation.
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Enums
# =============================================================================

class FIGOCategory(int, Enum):
    """FIGO category classification."""
    NORMAL = 1
    SUSPICIOUS = 2
    PATHOLOGICAL = 3


class EventTypeEnum(str, Enum):
    """Clinical event types."""
    LATE_DECEL = "LATE_DECEL"
    VARIABLE_DECEL = "VARIABLE_DECEL"
    PROLONGED_DECEL = "PROLONGED_DECEL"
    BRADYCARDIA = "BRADYCARDIA"
    TACHYCARDIA = "TACHYCARDIA"
    MINIMAL_VARIABILITY = "MINIMAL_VARIABILITY"
    HYPERSTIM = "HYPERSTIM"
    SINUSOIDAL = "SINUSOIDAL"
    RECOVERY = "RECOVERY"


class SeverityEnum(str, Enum):
    """Event severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


# =============================================================================
# Patient Schemas
# =============================================================================

class PatientMetrics(BaseModel):
    """Current patient vital metrics."""
    model_config = ConfigDict(extra="allow")
    
    baseline_fhr: float = Field(..., description="Baseline FHR in bpm", ge=60, le=200)
    current_fhr: float = Field(..., description="Current FHR reading", ge=0, le=250)
    variability: float = Field(..., description="FHR variability in bpm", ge=0, le=50)
    current_uc: float = Field(..., description="Current UC reading", ge=0, le=100)
    acceleration_count: int = Field(default=0, ge=0, description="Accelerations in window")
    deceleration_count: int = Field(default=0, ge=0, description="Decelerations in window")


class AlertInfo(BaseModel):
    """Alert information."""
    type: str = Field(..., description="Alert type")
    message: str = Field(..., description="Human-readable message")
    severity: str = Field(default="info", description="info, warning, critical")
    timestamp: float = Field(..., description="Unix timestamp")


class PatientSnapshot(BaseModel):
    """Complete patient state snapshot."""
    model_config = ConfigDict(extra="allow")
    
    patient_id: str = Field(..., description="Unique patient identifier (e.g., 'P1')")
    bed_number: int = Field(..., ge=1, le=20, description="Bed number")
    category: int = Field(..., ge=1, le=3, description="FIGO category 1-3")
    category_name: str = Field(default="Normal", description="Category display name")
    metrics: PatientMetrics
    fhr_history: List[float] = Field(default_factory=list, description="Recent FHR values")
    uc_history: List[float] = Field(default_factory=list, description="Recent UC values")
    timestamps: List[float] = Field(default_factory=list, description="Timestamps for history")
    alerts: List[AlertInfo] = Field(default_factory=list)
    trend_data: Optional[Dict[str, Any]] = Field(None, description="Trend analysis data")
    explanation: Optional[Dict[str, Any]] = Field(None, description="AI explanation")
    highlight_regions: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Highlighted regions for chart annotations",
    )
    fsqi_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Signal quality")
    has_active_event: bool = Field(default=False, description="Has injected event active")
    last_update: float = Field(..., description="Unix timestamp of last update")


class PatientSummary(BaseModel):
    """Lightweight patient summary for list views."""
    patient_id: str
    bed_number: int
    category: int
    category_name: str
    current_fhr: float
    baseline_fhr: float
    has_alerts: bool = False
    last_update: float


class PatientList(BaseModel):
    """List of patients response."""
    patients: List[PatientSnapshot]
    count: int
    timestamp: float = Field(..., description="Response timestamp")


class PatientListSummary(BaseModel):
    """Lightweight list of patients for overview."""
    patients: List[PatientSummary]
    count: int
    timestamp: float


# =============================================================================
# Simulation Schemas
# =============================================================================

class SimulationStatus(BaseModel):
    """Current simulation state."""
    running: bool
    paused: bool
    patient_count: int
    tick_count: int = Field(default=0, description="Total simulation ticks")
    elapsed_seconds: float = Field(default=0.0, ge=0)
    uptime_seconds: float = Field(default=0.0, ge=0)


class SimulationConfig(BaseModel):
    """Simulation configuration update."""
    patient_count: Optional[int] = Field(None, ge=1, le=20)
    speed_multiplier: Optional[float] = Field(None, ge=0.1, le=10.0)


class SimulationCommand(BaseModel):
    """Command to control simulation."""
    action: Literal["start", "stop", "pause", "resume", "reset"]


class SimulationResponse(BaseModel):
    """Response for simulation commands."""
    success: bool
    message: str
    status: SimulationStatus


# =============================================================================
# WebSocket Schemas
# =============================================================================

class WSMessage(BaseModel):
    """WebSocket message wrapper."""
    type: Literal["data", "heartbeat", "error", "status"] = Field(..., description="Message type")
    timestamp: float
    payload: Optional[Dict[str, Any]] = None


class WSPatientUpdate(BaseModel):
    """WebSocket patient data update."""
    patient_id: str
    fhr: float
    uc: float
    category: int
    baseline: float
    variability: float
    alerts: List[AlertInfo] = Field(default_factory=list)


class WSBatchUpdate(BaseModel):
    """Batch update for all patients."""
    patients: List[WSPatientUpdate]
    simulation_time: float
    tick_count: int


# =============================================================================
# Event Injection (God Mode)
# =============================================================================

class EventInjection(BaseModel):
    """Request to inject a clinical event."""
    event_type: EventTypeEnum = Field(..., description="Type of clinical event")
    severity: SeverityEnum = Field(default=SeverityEnum.MODERATE, description="Event severity")
    duration_seconds: int = Field(default=120, ge=30, le=600, description="Duration in seconds")
    params: Optional[Dict[str, Any]] = Field(None, description="Additional event parameters")


class EventInjectionResponse(BaseModel):
    """Response for event injection."""
    success: bool
    message: str
    patient_id: str
    event_type: str


# =============================================================================
# API Health & Info
# =============================================================================

class HealthCheck(BaseModel):
    """API health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: float


class APIError(BaseModel):
    """Standard API error response."""
    detail: str
    code: Optional[str] = None
    timestamp: float
