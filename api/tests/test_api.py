"""
API Tests
=========
Tests for the FastAPI backend (Phase 1).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app
from api.models.schemas import SeverityEnum, EventTypeEnum


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


# =============================================================================
# Health & Root Endpoints
# =============================================================================

def test_health_check(client):
    """Test health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["version"] == "3.0.0"
    assert "timestamp" in data


def test_root_endpoint(client):
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data


# =============================================================================
# Simulation Endpoints
# =============================================================================

def test_simulation_status(client):
    """Test simulation status endpoint."""
    response = client.get("/api/simulation/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "paused" in data
    assert "patient_count" in data
    assert "tick_count" in data


def test_simulation_start(client):
    """Test simulation start endpoint."""
    response = client.post("/api/simulation/start")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "status" in data
    
    # Cleanup - stop simulation
    client.post("/api/simulation/stop")


def test_simulation_stop(client):
    """Test simulation stop endpoint."""
    # Start first
    client.post("/api/simulation/start")
    
    response = client.post("/api/simulation/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_simulation_pause_resume(client):
    """Test pause/resume workflow."""
    # Start simulation
    client.post("/api/simulation/start")
    
    # Pause
    response = client.post("/api/simulation/pause")
    assert response.status_code == 200
    assert response.json()["success"] is True
    
    # Resume
    response = client.post("/api/simulation/resume")
    assert response.status_code == 200
    assert response.json()["success"] is True
    
    # Cleanup
    client.post("/api/simulation/stop")


def test_simulation_config_update(client):
    """Test configuration update."""
    response = client.patch(
        "/api/simulation/config",
        json={"patient_count": 6}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "patient_count=6" in data["message"]


def test_simulation_command_endpoint(client):
    """Test unified command endpoint."""
    # Start via command
    response = client.post(
        "/api/simulation/command",
        json={"action": "start"}
    )
    assert response.status_code == 200
    
    # Stop via command
    response = client.post(
        "/api/simulation/command",
        json={"action": "stop"}
    )
    assert response.status_code == 200


# =============================================================================
# Patient Endpoints
# =============================================================================

def test_list_patients(client):
    """Test list patients endpoint."""
    response = client.get("/api/patients")
    assert response.status_code == 200
    data = response.json()
    assert "patients" in data
    assert "count" in data
    assert "timestamp" in data


def test_list_patients_summary(client):
    """Test lightweight summary endpoint."""
    response = client.get("/api/patients/summary")
    assert response.status_code == 200
    data = response.json()
    assert "patients" in data
    assert "count" in data


def test_get_patient_not_found(client):
    """Test getting non-existent patient."""
    response = client.get("/api/patients/P999")
    assert response.status_code == 404


def test_inject_event_simulation_not_running(client):
    """Test event injection when simulation not running."""
    # Make sure simulation is stopped
    client.post("/api/simulation/stop")
    
    response = client.post(
        "/api/patients/P1/event",
        json={
            "event_type": "LATE_DECEL",
            "severity": "moderate",
            "duration_seconds": 120
        }
    )
    # Should fail because simulation not running
    assert response.status_code in [400, 404]


# =============================================================================
# Schema Validation
# =============================================================================

def test_simulation_config_validation(client):
    """Test config validation."""
    # Invalid patient count
    response = client.patch(
        "/api/simulation/config",
        json={"patient_count": 100}  # Max is 20
    )
    assert response.status_code == 422  # Validation error


def test_event_injection_validation(client):
    """Test event injection validation."""
    response = client.post(
        "/api/patients/P1/event",
        json={
            "event_type": "INVALID_EVENT",
            "severity": "moderate",
            "duration_seconds": 120
        }
    )
    assert response.status_code == 422  # Validation error
