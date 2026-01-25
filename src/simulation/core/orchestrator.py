"""
Simulation Orchestrator - Coordinates multiple patient generators.

This is the main controller for the real-time CTG simulation, responsible for:
- Creating and managing multiple PatientGenerators
- Scheduling data generation at regular intervals
- Coordinating MOMENT processing with staggered scheduling
- Handling event injection across patients
- Thread-safe operation with background simulation loop
- Pushing processed data to DataBridge for real-time UI updates

Key Design Decisions:
- Rule Engine runs every tick (fast, ~10ms)
- MOMENT processing is STAGGERED - one patient every few seconds to prevent CPU freeze
- Uses threading for background simulation while UI remains responsive
- DataBridge integration for Pulse Architecture (thread-safe UI updates)

References:
    - SentinelFetal Real-Time Simulator SPEC Part 2, Section 6
    - SentinelFetal UI Pulse Architecture Blueprint
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

import numpy as np

from ..generators.patient_generator import PatientGenerator, PatientConfig
from ..events.event_types import EventType, EventParameters, InjectedEvent
from ..logging.event_logger import EventLogger

# Import DataBridge for state management integration
try:
    from src.interfaces.state_bridge import (
        get_data_bridge,
        create_snapshot_from_pipeline_result,
        PatientSnapshot,
    )
    DATA_BRIDGE_AVAILABLE = True
except ImportError:
    DATA_BRIDGE_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """
    Configuration for the simulation orchestrator.
    
    Attributes:
        num_patients: Number of simulated patients (default: 8).
        sampling_rate: Samples per second (default: 4.0 Hz).
        tick_interval_seconds: Time between generation ticks (default: 1.0s).
        moment_interval_seconds: Total time to cycle through all patients for MOMENT.
        patient_names: Hebrew names for patients.
    """
    num_patients: int = 8
    sampling_rate: float = 4.0
    tick_interval_seconds: float = 1.0
    moment_interval_seconds: float = 30.0  # Full cycle through all patients
    
    patient_names: List[str] = field(default_factory=lambda: [
        "שרה כהן", "רחל לוי", "מירי גולן", "יעל ברק",
        "נועה שמיר", "דנה רוזן", "טלי אברהם", "ליאת פרידמן",
        "הילה דוד", "עדי משה", "רונית בן", "שירה גל",
        "מיכל אלון", "תמר רז", "אורית שלום", "גלית עוז",
        "רותי ים", "ענת כרמל", "לירון נהר", "אביגיל הר"
    ])
    
    @property
    def moment_per_patient_interval(self) -> float:
        """Time between MOMENT calls for each patient."""
        return self.moment_interval_seconds / max(self.num_patients, 1)


class SimulationOrchestrator:
    """
    Main orchestrator for real-time CTG simulation.
    
    Responsibilities:
    - Creates and manages PatientGenerators
    - Schedules data generation at regular intervals
    - Coordinates MOMENT processing (staggered to prevent CPU freeze)
    - Handles event injection
    - Provides thread-safe access to patient data
    
    Staggered Processing Strategy:
        With 8 patients and 30s MOMENT interval:
        - Patient P1 processed at t=0s
        - Patient P2 processed at t=3.75s
        - Patient P3 processed at t=7.5s
        - ... and so on
        This spreads MOMENT calls (100-300ms each) to prevent CPU spikes.
    
    Example:
        >>> orchestrator = SimulationOrchestrator()
        >>> orchestrator.start()
        >>> orchestrator.inject_event('P1', EventType.SINUSOIDAL_PATTERN, params)
        >>> time.sleep(10)
        >>> statuses = orchestrator.get_all_patients_status()
        >>> orchestrator.stop()
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        processing_callback: Optional[Callable[[str, Dict], Dict]] = None,
        tick_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Orchestrator configuration. Uses defaults if None.
            processing_callback: Function called for each patient when MOMENT
                               processing is triggered. Signature:
                               callback(patient_id, data) -> results
            tick_callback: Function called after each simulation tick with all
                          patient data. Signature: callback(tick_data) -> None
        """
        self.config = config or OrchestratorConfig()
        self._processing_callback = processing_callback
        self._tick_callback = tick_callback
        
        # DEBUG: Check if tick_callback was provided
        if tick_callback:
            logger.info(f"✅ TICK CALLBACK PROVIDED: {type(tick_callback).__name__}")
        else:
            logger.warning("❌ NO TICK CALLBACK PROVIDED!")
        
        # Patient generators
        self._patients: Dict[str, PatientGenerator] = {}
        self._create_patients()
        
        # MOMENT scheduling (round-robin)
        self._moment_schedule: List[str] = list(self._patients.keys())
        self._moment_index = 0
        self._last_moment_time = 0.0
        
        # Threading
        self._running = False
        self._paused = False
        self._simulation_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        
        # Timing
        self._simulation_time = 0.0
        self._speed_multiplier = 1.0
        self._start_real_time: Optional[float] = None
        
        # Event logging
        self._logger = EventLogger(max_entries=1000)
        
        # Statistics
        self._tick_count = 0
        self._moment_process_count = 0
        
        logger.info(
            f"Orchestrator initialized with {self.config.num_patients} patients, "
            f"MOMENT interval per patient: "
            f"{self.config.moment_per_patient_interval:.1f}s"
        )
    
    def _create_patients(self) -> None:
        """Create all patient generators with randomized parameters."""
        for i in range(self.config.num_patients):
            patient_id = f"P{i+1}"
            name = self.config.patient_names[i % len(self.config.patient_names)]
            
            # Randomize baseline parameters for variety
            config = PatientConfig(
                patient_id=patient_id,
                bed_number=i + 1,
                name=name,
                baseline_fhr=np.random.uniform(130, 150),
                baseline_variability=np.random.uniform(8, 15),
                contractions_per_10min=np.random.uniform(3.5, 4.5)
            )
            
            self._patients[patient_id] = PatientGenerator(config)
            logger.debug(
                f"Created patient {patient_id}: baseline={config.baseline_fhr:.0f}bpm"
            )
    
    # ========================================================================
    # Simulation Control
    # ========================================================================
    
    def start(self) -> None:
        """Start the simulation in a background thread."""
        if self._running:
            logger.warning("Simulation already running")
            return
        
        self._running = True
        self._paused = False
        self._start_real_time = time.time()
        
        self._simulation_thread = threading.Thread(
            target=self._simulation_loop,
            name="SimulationLoop",
            daemon=True
        )
        self._simulation_thread.start()
        
        logger.info("Simulation started")
    
    def stop(self) -> None:
        """Stop the simulation."""
        self._running = False
        if self._simulation_thread:
            self._simulation_thread.join(timeout=2.0)
            self._simulation_thread = None
        
        logger.info(
            f"Simulation stopped. Total ticks: {self._tick_count}, "
            f"MOMENT processes: {self._moment_process_count}"
        )
    
    def pause(self) -> None:
        """Pause the simulation (data generation stops)."""
        self._paused = True
        logger.info("Simulation paused")
    
    def resume(self) -> None:
        """Resume a paused simulation."""
        self._paused = False
        logger.info("Simulation resumed")
    
    def set_speed(self, multiplier: float) -> None:
        """
        Set simulation speed multiplier.
        
        Args:
            multiplier: Speed multiplier (0.5 = half speed, 2.0 = double speed).
                       Clamped to range [0.5, 2.0].
        """
        self._speed_multiplier = max(0.5, min(2.0, multiplier))
        logger.info(f"Simulation speed set to {self._speed_multiplier}x")

    def set_patient_count(self, count: int) -> None:
        """
        Dynamically change the number of patients.

        Stops simulation, recreates patients, and resets state.
        Auto-restarts if simulation was running.

        Args:
            count: Number of patients (1-20, clamped).
        """
        count = max(1, min(20, count))
        if count == self.config.num_patients:
            return

        was_running = self._running

        # Stop simulation
        if self._running:
            self.stop()

        with self._lock:
            # Update config
            self.config.num_patients = count

            # Clear and recreate patients
            self._patients.clear()
            self._create_patients()

            # Reset MOMENT schedule
            self._moment_schedule = list(self._patients.keys())
            self._moment_index = 0
            self._last_moment_time = 0.0

            # Reset statistics
            self._tick_count = 0
            self._moment_process_count = 0
            self._simulation_time = 0.0

            # Clear event log
            self._logger = EventLogger(max_entries=1000)

        logger.info(f"Patient count changed to {count}")
        
        # Auto-restart if was running
        if was_running:
            self.start()
            logger.info(f"Auto-restarted simulation with {count} patients")
    
    def resize_ward(self, new_count: int) -> None:
        """
        Resize the ward by adding or removing patients WITHOUT stopping simulation.
        
        If simulation is running, maintains state and just adjusts patient list.
        This is the PREFERRED method for live patient count changes.
        
        Args:
            new_count: Target number of patients (1-20, clamped).
        """
        new_count = max(1, min(20, new_count))
        current_count = len(self._patients)
        
        if new_count == current_count:
            return
        
        with self._lock:
            if new_count > current_count:
                # ADD patients
                for i in range(current_count, new_count):
                    patient_id = f"P{i+1}"
                    name = self.config.patient_names[i % len(self.config.patient_names)]
                    
                    config = PatientConfig(
                        patient_id=patient_id,
                        bed_number=i + 1,
                        name=name,
                        baseline_fhr=np.random.uniform(130, 150),
                        baseline_variability=np.random.uniform(8, 15),
                        contractions_per_10min=np.random.uniform(3.5, 4.5)
                    )
                    
                    self._patients[patient_id] = PatientGenerator(config)
                    self._moment_schedule.append(patient_id)
                    logger.info(f"Added patient {patient_id} (live resize)")
                    
            elif new_count < current_count:
                # REMOVE patients (last ones first)
                to_remove = [f"P{i+1}" for i in range(new_count, current_count)]
                for patient_id in to_remove:
                    if patient_id in self._patients:
                        del self._patients[patient_id]
                        if patient_id in self._moment_schedule:
                            self._moment_schedule.remove(patient_id)
                        logger.info(f"Removed patient {patient_id} (live resize)")
            
            # Update config
            self.config.num_patients = new_count
            
            # Adjust MOMENT index if needed
            if self._moment_index >= len(self._moment_schedule):
                self._moment_index = 0
        
        logger.info(f"Ward resized from {current_count} to {new_count} patients")
        
        # Trigger immediate broadcast if DATA_BRIDGE is available (NOT CRITICAL - just for instant UI update)
        # The main broadcast happens in the simulation loop
        try:
            if DATA_BRIDGE_AVAILABLE and new_count > current_count:
                # Only broadcast new patients
                bridge = get_data_bridge()
                for i in range(current_count, new_count):
                    patient_id = f"P{i+1}"
                    if patient_id in self._patients:
                        patient = self._patients[patient_id]
                        # Generate initial tick
                        fhr_samples, uc_samples = patient.generate_tick(
                            int(self.config.sampling_rate * self.config.tick_interval_seconds)
                        )
                        
                        # Create basic snapshot for immediate display
                        snapshot = create_snapshot_from_pipeline_result(
                            patient_id=patient_id,
                            pipeline_result={'category': 1, 'confidence': 1.0, 'findings': {}},
                            fhr_samples=fhr_samples,
                            uc_samples=uc_samples,
                            patient_config=patient.config,
                            active_events=[],
                        )
                        bridge.update_patient(snapshot)
        except Exception as e:
            logger.debug(f"Non-critical: failed to broadcast after resize: {e}")

    @property
    def is_running(self) -> bool:
        """Check if simulation is currently running."""
        return self._running and not self._paused
    
    # ========================================================================
    # Simulation Loop
    # ========================================================================
    
    def _simulation_loop(self) -> None:
        """
        Main simulation loop running in a separate thread.
        
        Generates data ticks at regular intervals and schedules
        MOMENT processing in a staggered fashion.
        """
        last_tick = time.time()
        
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            elapsed = current_time - last_tick
            target_interval = (
                self.config.tick_interval_seconds / self._speed_multiplier
            )
            
            # Catch up if we fell behind to avoid cumulative drift.
            if elapsed >= target_interval:
                while elapsed >= target_interval and self._running and not self._paused:
                    self._tick()
                    last_tick += target_interval
                    elapsed = time.time() - last_tick
            else:
                # Sleep slightly less than the remaining interval to stay ahead of drift.
                sleep_time = max(0.005, target_interval - elapsed - 0.002)
                time.sleep(sleep_time)
    
    def _tick(self) -> None:
        """
        Execute one simulation tick.
        
        Each tick:
        1. Generates data for all patients (fast, ~1ms per patient)
        2. Calls tick_callback with current patient states
        3. Checks if it's time for MOMENT processing (staggered)
        """
        with self._lock:
            samples_per_tick = int(
                self.config.sampling_rate * self.config.tick_interval_seconds
            )
            
            # Generate data for all patients
            for patient_id, patient in self._patients.items():
                patient.generate_tick(samples_per_tick)
            
            self._simulation_time += self.config.tick_interval_seconds
            self._tick_count += 1
            
            # DEBUG: Check if callback is set
            if self._tick_count <= 5:
                logger.info(f"📊 Tick #{self._tick_count}: callback={'SET ✅' if self._tick_callback else 'NONE ❌'}")
            
            # Call tick callback if provided (for real-time updates)
            if self._tick_callback:
                try:
                    # DEBUG: Log first few calls
                    if self._tick_count <= 5:
                        logger.info(f"🎯 Orchestrator calling tick_callback: tick={self._tick_count}")
                    
                    tick_data = {
                        'simulation_time': self._simulation_time,
                        'tick_count': self._tick_count,
                        'patients': {}
                    }
                    
                    # Add each patient's recent data
                    for patient_id, patient in self._patients.items():
                        # Get last few samples for real-time display
                        recent_buffer = patient.get_buffer_data(duration_minutes=0.25)  # 15 seconds
                        tick_data['patients'][patient_id] = {
                            'fhr': recent_buffer.get('fhr', [])[-16:],  # Last 4 seconds
                            'uc': recent_buffer.get('uc', [])[-16:],
                            'baseline': patient.config.baseline_fhr,
                            'variability': patient.config.baseline_variability,
                            'category': getattr(patient, 'latest_category', 1),
                        }
                    
                    self._tick_callback(tick_data)
                except Exception as e:
                    # Don't let callback errors stop simulation
                    logger.error(f"❌ Tick callback error: {e}", exc_info=True)  # Always log with stack
            
            # Check for MOMENT processing (staggered)
            moment_interval = self.config.moment_per_patient_interval
            if self._simulation_time - self._last_moment_time >= moment_interval:
                self._process_next_patient_moment()
                self._last_moment_time = self._simulation_time
    
    def _process_next_patient_moment(self) -> None:
        """
        Process MOMENT for the next patient in the round-robin schedule.
        
        This staggered approach ensures:
        - Each patient gets MOMENT processing every ~30 seconds
        - Only ONE patient is processed at a time
        - CPU usage is spread out, not spiked
        - DataBridge is updated for real-time UI (Pulse Architecture)
        """
        if not self._processing_callback:
            return
        
        # Get next patient in schedule
        patient_id = self._moment_schedule[self._moment_index]
        self._moment_index = (self._moment_index + 1) % len(self._moment_schedule)
        
        patient = self._patients.get(patient_id)
        if not patient:
            return
        
        # Get patient data
        data = patient.get_buffer_data(duration_minutes=10)
        data['simulation_time'] = self._simulation_time
        
        try:
            # Call the processing callback (runs MOMENT + analysis)
            results = self._processing_callback(patient_id, data)
            self._moment_process_count += 1
            
            if results:
                # Update patient state
                patient.latest_category = results.get('category', 1)
                patient.latest_alert = results.get('alert')
                patient.latest_findings = results.get('findings', {})
                
                # ============================================================
                # Pulse Architecture: Push to DataBridge (Thread-Safe)
                # ============================================================
                if DATA_BRIDGE_AVAILABLE:
                    try:
                        bridge = get_data_bridge()
                        
                        # Get FHR/UC samples for snapshot
                        fhr_samples = data.get('fhr', [])
                        uc_samples = data.get('uc', [])
                        if hasattr(fhr_samples, 'tolist'):
                            fhr_samples = fhr_samples.tolist()
                        if hasattr(uc_samples, 'tolist'):
                            uc_samples = uc_samples.tolist()
                        
                        # Get active events
                        active_events = [
                            e.event_type.name 
                            for e in patient.get_active_events()
                        ]
                        
                        # Create and push snapshot
                        snapshot = create_snapshot_from_pipeline_result(
                            patient_id=patient_id,
                            pipeline_result=results,
                            fhr_samples=fhr_samples,
                            uc_samples=uc_samples,
                            patient_config=patient.config,
                            active_events=active_events,
                        )
                        bridge.push_batch(patient_id, snapshot)
                        
                        # Update simulation state in bridge
                        bridge.update_simulation_state(
                            simulation_time=self._simulation_time,
                            is_running=self._running,
                            is_paused=self._paused,
                            speed_multiplier=self._speed_multiplier,
                        )
                        
                        # Push alert if Category 2 or 3
                        if results.get('category', 1) >= 2:
                            bridge.push_alert({
                                'patient_id': patient_id,
                                'category': results.get('category'),
                                'confidence': results.get('confidence', 0.0),
                                'simulation_time': self._simulation_time,
                                'findings': results.get('findings', {}),
                            })
                            
                    except Exception as e:
                        logger.debug(f"DataBridge push failed (non-critical): {e}")
                
                # Log Category 2/3 alerts
                if results.get('category', 1) >= 2:
                    results['simulation_time'] = self._simulation_time
                    self._logger.log_alert(patient_id, results)
                    
                    logger.info(
                        f"Alert for {patient_id}: Category {results['category']}"
                    )
                    
        except Exception as e:
            logger.error(f"Error processing {patient_id}: {e}", exc_info=True)
    
    # ========================================================================
    # Event Injection
    # ========================================================================
    
    def inject_event(
        self,
        patient_id: str,
        event_type: EventType,
        params: EventParameters,
        duration_seconds: Optional[float] = None
    ) -> Optional[InjectedEvent]:
        """
        Inject a clinical event into a specific patient.
        
        Args:
            patient_id: Target patient ID (e.g., 'P1').
            event_type: Type of event to inject.
            params: Event parameters (severity, characteristics).
            duration_seconds: Event duration. If None, uses params default.
        
        Returns:
            The InjectedEvent if successful, None if patient not found.
        """
        with self._lock:
            patient = self._patients.get(patient_id)
            if not patient:
                logger.warning(f"Patient {patient_id} not found for injection")
                return None
            
            event = patient.inject_event(event_type, params, duration_seconds)
            self._logger.log_injection(event)
            
            logger.info(
                f"Injected {event_type.name} into {patient_id} for "
                f"{duration_seconds or params.duration_seconds}s"
            )
            
            return event
    
    def inject_event_all(
        self,
        event_type: EventType,
        params: EventParameters,
        duration_seconds: Optional[float] = None
    ) -> List[InjectedEvent]:
        """
        Inject the same event into all patients.
        
        Useful for mass testing scenarios.
        
        Returns:
            List of injected events.
        """
        events = []
        for patient_id in self._patients.keys():
            event = self.inject_event(patient_id, event_type, params, duration_seconds)
            if event:
                events.append(event)
        return events
    
    # ========================================================================
    # Status & Data Access
    # ========================================================================
    
    def get_all_patients_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all patients, sorted by category (highest first).
        
        Returns:
            List of patient status dictionaries.
        """
        with self._lock:
            statuses = [p.get_status() for p in self._patients.values()]
        
        # Sort by category (descending), then bed number (ascending)
        statuses.sort(key=lambda x: (-x['category'], x['bed_number']))
        return statuses
    
    def get_patient_data(
        self,
        patient_id: str,
        duration_minutes: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed data for a specific patient.
        
        Args:
            patient_id: Patient identifier.
            duration_minutes: Minutes of data to retrieve. None = full buffer.
        
        Returns:
            Patient data dictionary, or None if patient not found.
        """
        with self._lock:
            patient = self._patients.get(patient_id)
            if not patient:
                return None
            return patient.get_buffer_data(duration_minutes)
    
    def get_patient(self, patient_id: str) -> Optional[PatientGenerator]:
        """Get a patient generator by ID."""
        return self._patients.get(patient_id)
    
    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self._simulation_time
    
    def get_simulation_time_formatted(self) -> str:
        """Get simulation time as HH:MM:SS."""
        total = int(self._simulation_time)
        hours = total // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_real_elapsed_time(self) -> float:
        """Get actual elapsed time since simulation start."""
        if self._start_real_time:
            return time.time() - self._start_real_time
        return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return {
            'simulation_time': self._simulation_time,
            'real_elapsed_time': self.get_real_elapsed_time(),
            'tick_count': self._tick_count,
            'moment_process_count': self._moment_process_count,
            'speed_multiplier': self._speed_multiplier,
            'is_running': self._running,
            'is_paused': self._paused,
            'num_patients': len(self._patients),
            'log_entries': self._logger.size
        }
    
    # ========================================================================
    # Reset & Logging
    # ========================================================================
    
    def reset_all(self) -> None:
        """Reset all patients to initial state."""
        with self._lock:
            for patient in self._patients.values():
                patient.reset()
            self._simulation_time = 0.0
            self._moment_index = 0
            self._last_moment_time = 0.0
            self._tick_count = 0
            self._moment_process_count = 0
            self._logger.clear()
            
            # Clear DataBridge for Pulse Architecture
            if DATA_BRIDGE_AVAILABLE:
                try:
                    bridge = get_data_bridge()
                    bridge.clear_all()
                except Exception as e:
                    logger.debug(f"DataBridge clear failed (non-critical): {e}")
        
        logger.info("All patients reset")
    
    def export_log(self, filepath: str) -> None:
        """Export event log to CSV file."""
        self._logger.export_csv(filepath)
        logger.info(f"Log exported to {filepath}")
    
    def get_event_log(self) -> EventLogger:
        """Get the event logger."""
        return self._logger
