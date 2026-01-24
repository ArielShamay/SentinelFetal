"""
Simulation Router
=================
REST endpoints for simulation control.
"""

import time
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, status

from api.dependencies import OrchestratorDep
from api.models.schemas import (
    SimulationStatus,
    SimulationConfig,
    SimulationCommand,
    SimulationResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_simulation_status(orchestrator: OrchestratorDep) -> SimulationStatus:
    """Helper to build simulation status response."""
    return SimulationStatus(
        running=orchestrator.is_running(),
        paused=orchestrator.is_paused(),
        patient_count=orchestrator.get_patient_count(),
        tick_count=orchestrator.get_tick_count(),
        elapsed_seconds=orchestrator.get_uptime_seconds(),
        uptime_seconds=orchestrator.get_uptime_seconds(),
    )


@router.get("/status", response_model=SimulationStatus)
async def get_status(orchestrator: OrchestratorDep):
    """
    Get current simulation status.
    
    Returns running state, patient count, and timing info.
    """
    return _get_simulation_status(orchestrator)


@router.post("/start", response_model=SimulationResponse)
async def start_simulation(orchestrator: OrchestratorDep):
    """
    Start the simulation.
    
    Initializes patients and begins the simulation loop.
    """
    try:
        if orchestrator.is_running() and not orchestrator.is_paused():
            return SimulationResponse(
                success=False,
                message="Simulation already running",
                status=_get_simulation_status(orchestrator),
            )
        
        orchestrator.start()
        logger.info("Simulation started via API")
        
        return SimulationResponse(
            success=True,
            message="Simulation started successfully",
            status=_get_simulation_status(orchestrator),
        )
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start simulation: {str(e)}",
        )


@router.post("/pause", response_model=SimulationResponse)
async def pause_simulation(orchestrator: OrchestratorDep):
    """
    Pause the simulation.
    
    Suspends data generation but preserves state.
    """
    if not orchestrator.is_running():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Simulation not running",
        )
    
    orchestrator.pause()
    logger.info("Simulation paused via API")
    
    return SimulationResponse(
        success=True,
        message="Simulation paused",
        status=_get_simulation_status(orchestrator),
    )


@router.post("/resume", response_model=SimulationResponse)
async def resume_simulation(orchestrator: OrchestratorDep):
    """
    Resume a paused simulation.
    """
    if not orchestrator.is_paused():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Simulation not paused",
        )
    
    orchestrator.resume()
    logger.info("Simulation resumed via API")
    
    return SimulationResponse(
        success=True,
        message="Simulation resumed",
        status=_get_simulation_status(orchestrator),
    )


@router.post("/stop", response_model=SimulationResponse)
async def stop_simulation(orchestrator: OrchestratorDep):
    """
    Stop the simulation.
    
    Halts all data generation and resets state.
    """
    orchestrator.stop()
    logger.info("Simulation stopped via API")
    
    return SimulationResponse(
        success=True,
        message="Simulation stopped",
        status=_get_simulation_status(orchestrator),
    )


@router.post("/reset", response_model=SimulationResponse)
async def reset_simulation(orchestrator: OrchestratorDep):
    """
    Reset the simulation.
    
    Stops and reinitializes with fresh patient data.
    """
    orchestrator.reset()
    logger.info("Simulation reset via API")
    
    return SimulationResponse(
        success=True,
        message="Simulation reset successfully",
        status=_get_simulation_status(orchestrator),
    )


@router.post("/command", response_model=SimulationResponse)
async def simulation_command(
    command: SimulationCommand,
    orchestrator: OrchestratorDep,
):
    """
    Execute a simulation command.
    
    Unified endpoint for all simulation control actions.
    """
    action_map = {
        "start": start_simulation,
        "stop": stop_simulation,
        "pause": pause_simulation,
        "resume": resume_simulation,
        "reset": reset_simulation,
    }
    
    handler = action_map.get(command.action)
    if handler:
        return await handler(orchestrator)
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unknown action: {command.action}",
    )


@router.patch("/config", response_model=SimulationResponse)
async def update_config(
    config: SimulationConfig,
    orchestrator: OrchestratorDep,
):
    """
    Update simulation configuration.
    
    Can change patient count and speed multiplier.
    Changes may require restart to take effect.
    """
    changes = []
    
    if config.patient_count is not None:
        orchestrator.set_patient_count(config.patient_count)
        changes.append(f"patient_count={config.patient_count}")
    
    if config.speed_multiplier is not None:
        orchestrator.set_speed(config.speed_multiplier)
        changes.append(f"speed={config.speed_multiplier}x")
    
    message = f"Configuration updated: {', '.join(changes)}" if changes else "No changes"
    logger.info(message)
    
    return SimulationResponse(
        success=True,
        message=message,
        status=_get_simulation_status(orchestrator),
    )
