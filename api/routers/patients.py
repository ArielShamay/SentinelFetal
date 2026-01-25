"""
Patients Router
===============
REST endpoints for patient data access and event injection.
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Query

from api.dependencies import OrchestratorDep
from api.models.schemas import (
    PatientSnapshot,
    PatientSummary,
    PatientMetrics,
    PatientList,
    PatientListSummary,
    AlertInfo,
    EventInjection,
    EventInjectionResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def _category_to_name(category: int) -> str:
    """Convert category number to display name."""
    names = {1: "Normal", 2: "Suspicious", 3: "Pathological"}
    return names.get(category, "Unknown")


def _patient_to_snapshot(patient, patient_data: dict, orchestrator) -> PatientSnapshot:
    """Convert internal patient data to API snapshot."""
    now = time.time()
    
    # Extract metrics
    def _to_list(values):
        if hasattr(values, "tolist"):
            return values.tolist()
        if isinstance(values, (list, tuple)):
            return list(values)
        return []

    fhr_history = _to_list(patient_data.get("fhr", []))
    uc_history = _to_list(patient_data.get("uc", []))
    timestamps = _to_list(patient_data.get("timestamps", []))

    # Handle numpy arrays - check length instead of truthiness
    current_fhr = fhr_history[-1] if len(fhr_history) > 0 else 140.0
    current_uc = uc_history[-1] if len(uc_history) > 0 else 0.0
    
    # Get patient state
    category = patient_data.get("category", 1)
    baseline = patient_data.get("baseline", 140.0)
    variability = patient_data.get("variability", 10.0)
    
    # Build metrics
    metrics = PatientMetrics(
        baseline_fhr=baseline,
        current_fhr=current_fhr,
        variability=variability,
        current_uc=current_uc,
        acceleration_count=patient_data.get("accelerations", 0),
        deceleration_count=patient_data.get("decelerations", 0),
    )
    
    # Convert alerts
    alerts = []
    for alert_data in patient_data.get("alerts", []):
        if isinstance(alert_data, dict):
            alerts.append(AlertInfo(
                type=alert_data.get("type", "info"),
                message=alert_data.get("message", ""),
                severity=alert_data.get("severity", "info"),
                timestamp=alert_data.get("timestamp", now),
            ))
    
    return PatientSnapshot(
        patient_id=patient_data.get("patient_id", patient.patient_id if patient else "P0"),
        bed_number=patient_data.get("bed_number", 1),
        category=category,
        category_name=_category_to_name(category),
        metrics=metrics,
        fhr_history=fhr_history,
        uc_history=uc_history,
        timestamps=timestamps,
        alerts=alerts,
        trend_data=patient_data.get("trend_data"),
        explanation=patient_data.get("explanation"),
        highlight_regions=patient_data.get("highlight_regions"),
        fsqi_score=patient_data.get("fsqi", 1.0),
        has_active_event=orchestrator.has_active_event(patient_data.get("patient_id", "")),
        last_update=now,
    )


def _merge_bridge_snapshot(patient_data: dict, bridge_snapshot: Optional[object]) -> None:
    """Merge DataBridge snapshot fields into patient_data in-place."""
    if bridge_snapshot is None:
        return

    try:
        patient_data["category"] = getattr(bridge_snapshot, "category", patient_data.get("category", 1))
        patient_data["trend_data"] = getattr(bridge_snapshot, "trend_data", patient_data.get("trend_data"))
        patient_data["explanation"] = getattr(bridge_snapshot, "explanation", patient_data.get("explanation"))

        alerts = getattr(bridge_snapshot, "alerts", None)
        if alerts is not None:
            patient_data["alerts"] = alerts

        regions = getattr(bridge_snapshot, "highlight_regions", None)
        if regions is not None:
            patient_data["highlight_regions"] = [
                r.to_dict() if hasattr(r, "to_dict") else r
                for r in regions
            ]
    except Exception as exc:
        logger.debug(f"Failed to merge DataBridge snapshot: {exc}")


def _patient_to_summary(patient, patient_data: dict) -> PatientSummary:
    """Convert internal patient data to lightweight summary."""
    fhr_data = patient_data.get("fhr", [])
    current_fhr = fhr_data[-1] if fhr_data else 140.0
    category = patient_data.get("category", 1)
    
    return PatientSummary(
        patient_id=patient_data.get("patient_id", "P0"),
        bed_number=patient_data.get("bed_number", 1),
        category=category,
        category_name=_category_to_name(category),
        current_fhr=current_fhr,
        baseline_fhr=patient_data.get("baseline", 140.0),
        has_alerts=len(patient_data.get("alerts", [])) > 0,
        last_update=time.time(),
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=PatientList)
async def list_patients(
    orchestrator: OrchestratorDep,
    duration_minutes: Optional[float] = Query(
        default=5.0, 
        ge=1.0, 
        le=60.0, 
        description="Minutes of history to include"
    ),
):
    """
    Get all patients with full data.
    
    Returns complete snapshots for all active patients.
    """
    patients_status = orchestrator.get_all_patients_status()
    snapshots = []
    
    for patient_info in patients_status:
        patient_id = patient_info.get("patient_id")
        patient = orchestrator.get_patient(patient_id)
        patient_data = orchestrator.get_patient_data(patient_id, duration_minutes)
        bridge_snapshot = orchestrator.get_patient_snapshot(patient_id)
        
        if patient_data:
            # Merge status info with data
            patient_data.update(patient_info)
            _merge_bridge_snapshot(patient_data, bridge_snapshot)
            snapshots.append(_patient_to_snapshot(patient, patient_data, orchestrator))
    
    return PatientList(
        patients=snapshots,
        count=len(snapshots),
        timestamp=time.time(),
    )


@router.get("/summary", response_model=PatientListSummary)
async def list_patients_summary(orchestrator: OrchestratorDep):
    """
    Get lightweight summary of all patients.
    
    Faster endpoint for overview displays.
    """
    patients_status = orchestrator.get_all_patients_status()
    summaries = []
    
    for patient_info in patients_status:
        patient_id = patient_info.get("patient_id")
        patient = orchestrator.get_patient(patient_id)
        summaries.append(_patient_to_summary(patient, patient_info))
    
    return PatientListSummary(
        patients=summaries,
        count=len(summaries),
        timestamp=time.time(),
    )


@router.get("/{patient_id}", response_model=PatientSnapshot)
async def get_patient(
    patient_id: str,
    orchestrator: OrchestratorDep,
    duration_minutes: Optional[float] = Query(
        default=5.0, 
        ge=1.0, 
        le=60.0, 
        description="Minutes of history to include"
    ),
):
    """
    Get detailed data for a specific patient.
    
    Returns full snapshot with configurable history length.
    """
    patient = orchestrator.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found",
        )
    
    patient_data = orchestrator.get_patient_data(patient_id, duration_minutes)
    if not patient_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data for patient {patient_id}",
        )

    bridge_snapshot = orchestrator.get_patient_snapshot(patient_id)
    _merge_bridge_snapshot(patient_data, bridge_snapshot)
    
    return _patient_to_snapshot(patient, patient_data, orchestrator)


@router.post("/{patient_id}/event", response_model=EventInjectionResponse)
async def inject_event(
    patient_id: str,
    event: EventInjection,
    orchestrator: OrchestratorDep,
):
    """
    Inject a clinical event into a patient's simulation.
    
    God Mode endpoint for testing clinical scenarios.
    """
    patient = orchestrator.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found",
        )
    
    if not orchestrator.is_running():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Simulation must be running to inject events",
        )
    
    # Inject the event
    params = event.params or {}
    params["severity"] = event.severity.value

    duration_seconds = event.duration_seconds
    if event.duration_minutes is not None:
        duration_seconds = int(event.duration_minutes * 60)
    
    success = orchestrator.inject_event(
        patient_id=patient_id,
        event_type=event.event_type.value,
        params=params,
        duration=duration_seconds,
    )
    
    if success:
        logger.info(f"Injected {event.event_type.value} on {patient_id}")
        return EventInjectionResponse(
            success=True,
            message=f"Event {event.event_type.value} injected successfully",
            patient_id=patient_id,
            event_type=event.event_type.value,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to inject event {event.event_type.value}",
        )


@router.get("/{patient_id}/history")
async def get_patient_history(
    patient_id: str,
    orchestrator: OrchestratorDep,
    duration_minutes: float = Query(default=30.0, ge=1.0, le=120.0),
):
    """
    Get extended history data for a patient.
    
    Returns raw FHR/UC data for charting.
    """
    patient = orchestrator.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found",
        )
    
    patient_data = orchestrator.get_patient_data(patient_id, duration_minutes)
    if not patient_data:
        return {"patient_id": patient_id, "fhr": [], "uc": [], "timestamps": []}
    
    return {
        "patient_id": patient_id,
        "fhr": patient_data.get("fhr", []),
        "uc": patient_data.get("uc", []),
        "timestamps": patient_data.get("timestamps", []),
        "duration_minutes": duration_minutes,
    }
