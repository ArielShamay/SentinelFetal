"""
Orchestrator Adapter
====================
Thread-safe bridge between FastAPI async world and existing synchronous Orchestrator.
Provides singleton access to the simulation core.
"""

from typing import Optional, Dict, Any, List
import threading
import time
import logging
from pathlib import Path
import sys

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.core.orchestrator import SimulationOrchestrator, OrchestratorConfig
from src.simulation.processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig
from src.simulation.events.event_types import EventType, EventParameters

# Import DataBridge for snapshots
try:
    from src.interfaces.state_bridge import get_data_bridge, PatientSnapshot
    DATA_BRIDGE_AVAILABLE = True
except ImportError:
    DATA_BRIDGE_AVAILABLE = False
    PatientSnapshot = None

logger = logging.getLogger(__name__)


class OrchestratorAdapter:
    """
    Singleton adapter that bridges FastAPI to the existing simulation orchestrator.
    Provides thread-safe access to simulation state.
    """
    
    _instance: Optional["OrchestratorAdapter"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "OrchestratorAdapter":
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
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._orchestrator: Optional[SimulationOrchestrator] = None
        self._pipeline: Optional[PipelineAdapter] = None
        self._start_time: Optional[float] = None
        self._patient_count = 4
        logger.info("OrchestratorAdapter singleton created")
    
    def initialize(self, patient_count: int = 4) -> None:
        """Initialize the orchestrator with given patient count."""
        if self._orchestrator is not None:
            logger.warning("OrchestratorAdapter already initialized")
            return
        
        self._patient_count = patient_count
        
        # Create pipeline adapter for MOMENT processing
        self._pipeline = PipelineAdapter(PipelineAdapterConfig(
            use_real_moment=True,
            sampling_rate=4.0,
            min_data_seconds=60.0,
        ))
        
        # Create orchestrator config
        config = OrchestratorConfig(
            num_patients=patient_count,
            sampling_rate=4.0,
            tick_interval_seconds=1.0,
            moment_interval_seconds=30.0,
        )
        
        # Create orchestrator with pipeline callback
        self._orchestrator = SimulationOrchestrator(
            config=config,
            processing_callback=self._process_patient_data,
        )
        
        logger.info(f"OrchestratorAdapter initialized with {patient_count} patients")
    
    def _process_patient_data(self, patient_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Callback for MOMENT processing.
        Called by orchestrator for each patient in staggered fashion.
        Also pushes updates to WebSocket for real-time streaming.
        """
        if self._pipeline is None:
            return {}
        
        try:
            result = self._pipeline.process_patient(
                patient_id=patient_id,
                data=data,
                run_moment=True,
            )
            
            # Push to WebSocket (non-blocking)
            self._push_websocket_update(patient_id, data, result)
            
            return result
        except Exception as e:
            logger.error(f"Pipeline processing error for {patient_id}: {e}")
            return {}
    
    def _push_websocket_update(
        self, 
        patient_id: str, 
        data: Dict[str, Any], 
        result: Dict[str, Any]
    ) -> None:
        """Push patient update to WebSocket clients."""
        try:
            from api.services.orchestrator_bridge import push_to_websocket
            
            # Prepare update data
            fhr_data = data.get('fhr', [])
            uc_data = data.get('uc', [])
            
            # Convert numpy arrays if needed
            if hasattr(fhr_data, 'tolist'):
                fhr_data = fhr_data.tolist()
            if hasattr(uc_data, 'tolist'):
                uc_data = uc_data.tolist()
            
            update = {
                "type": "patient_update",
                "timestamp": time.time(),
                "patient_id": patient_id,
                "category": result.get('category', 1),
                "baseline": result.get('baseline_fhr', 140),
                "variability": result.get('variability', 10),
                "fhr_latest": fhr_data[-16:] if fhr_data else [],  # Last 4 seconds at 4Hz
                "uc_latest": uc_data[-16:] if uc_data else [],
                "fsqi": result.get('fsqi', 1.0),
                "confidence": result.get('confidence', 0.0),
                "findings": result.get('findings', {}),
            }
            
            push_to_websocket(update)
            
        except Exception as e:
            # Don't let WebSocket errors affect simulation
            logger.debug(f"WebSocket push failed (non-critical): {e}")
    
    def shutdown(self) -> None:
        """Clean shutdown of the orchestrator."""
        if self._orchestrator is not None:
            self._orchestrator.stop()
            logger.info("OrchestratorAdapter shut down")
    
    # =========================================================================
    # Simulation Control
    # =========================================================================
    
    def start(self) -> None:
        """Start the simulation."""
        if self._orchestrator is None:
            self.initialize(self._patient_count)
        self._orchestrator.start()
        self._start_time = time.time()
        logger.info("Simulation started via adapter")
    
    def pause(self) -> None:
        """Pause the simulation."""
        if self._orchestrator:
            self._orchestrator.pause()
    
    def resume(self) -> None:
        """Resume a paused simulation."""
        if self._orchestrator:
            self._orchestrator.resume()
    
    def stop(self) -> None:
        """Stop the simulation."""
        if self._orchestrator:
            self._orchestrator.stop()
            self._start_time = None
    
    def reset(self) -> None:
        """Reset the simulation."""
        if self._orchestrator:
            self._orchestrator.stop()
            # Recreate orchestrator
            old_count = self._patient_count
            self._orchestrator = None
            self.initialize(old_count)
    
    # =========================================================================
    # State Queries
    # =========================================================================
    
    def is_running(self) -> bool:
        """Check if simulation is running."""
        if self._orchestrator is None:
            return False
        return self._orchestrator._running and not self._orchestrator._paused
    
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        if self._orchestrator is None:
            return False
        return self._orchestrator._paused
    
    def get_patient_count(self) -> int:
        """Get current patient count."""
        if self._orchestrator is None:
            return self._patient_count
        return len(self._orchestrator._patients)
    
    def get_tick_count(self) -> int:
        """Get total tick count."""
        if self._orchestrator is None:
            return 0
        return self._orchestrator._tick_count
    
    def get_uptime_seconds(self) -> float:
        """Get simulation uptime."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def get_status(self) -> Dict[str, Any]:
        """Get full simulation status."""
        return {
            "running": self.is_running(),
            "paused": self.is_paused(),
            "patient_count": self.get_patient_count(),
            "tick_count": self.get_tick_count(),
            "uptime_seconds": self.get_uptime_seconds(),
        }
    
    # =========================================================================
    # Patient Access
    # =========================================================================
    
    def get_patient(self, patient_id: str) -> Optional[Any]:
        """Get a patient generator by ID."""
        if self._orchestrator is None:
            return None
        return self._orchestrator._patients.get(patient_id)
    
    def get_all_patients(self) -> List[Any]:
        """Get all patient generators."""
        if self._orchestrator is None:
            return []
        return list(self._orchestrator._patients.values())
    
    def get_all_patients_status(self) -> List[Dict[str, Any]]:
        """Get status of all patients (sorted by category)."""
        if self._orchestrator is None:
            return []
        return self._orchestrator.get_all_patients_status()
    
    def get_patient_data(self, patient_id: str, duration_minutes: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get detailed data for a specific patient."""
        if self._orchestrator is None:
            return None
        return self._orchestrator.get_patient_data(patient_id, duration_minutes)
    
    def get_patient_snapshot(self, patient_id: str) -> Optional[Any]:
        """Get the latest snapshot for a patient from DataBridge."""
        if not DATA_BRIDGE_AVAILABLE:
            return None
        try:
            bridge = get_data_bridge()
            return bridge.get_latest(patient_id)
        except Exception:
            return None
    
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
        """Update patient count (requires restart)."""
        count = max(1, min(20, count))
        self._patient_count = count
        if self._orchestrator:
            self._orchestrator.set_patient_count(count)
    
    def set_speed(self, multiplier: float) -> None:
        """Set simulation speed multiplier."""
        if self._orchestrator:
            self._orchestrator.set_speed(multiplier)
    
    # =========================================================================
    # Event Injection
    # =========================================================================
    
    def inject_event(
        self,
        patient_id: str,
        event_type: str,
        params: Optional[Dict[str, Any]] = None,
        duration: int = 120,
    ) -> bool:
        """
        Inject a clinical event on a patient.
        
        Args:
            patient_id: Target patient ID (e.g., 'P1')
            event_type: Event type name (e.g., 'LATE_DECEL')
            params: Optional event parameters
            duration: Duration in seconds
            
        Returns:
            True if injection succeeded
        """
        if self._orchestrator is None:
            return False
        
        try:
            # Convert string to EventType enum
            event_enum = EventType[event_type]
            
            # Create parameters
            event_params = EventParameters(
                severity=params.get('severity', 'moderate') if params else 'moderate',
                depth=params.get('depth', 30) if params else 30,
                duration_seconds=duration,
            )
            
            # Inject event
            result = self._orchestrator.inject_event(
                patient_id=patient_id,
                event_type=event_enum,
                params=event_params,
                duration_seconds=duration,
            )
            
            return result is not None
            
        except (KeyError, Exception) as e:
            logger.error(f"Event injection failed: {e}")
            return False


# Global singleton accessor
def get_orchestrator_adapter() -> OrchestratorAdapter:
    """Get the singleton orchestrator adapter."""
    return OrchestratorAdapter.get_instance()

