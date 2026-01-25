#!/usr/bin/env python3
"""
Debug Scenario Runner - Forensic Audit Tool

This script bypasses the frontend to trace the complete data flow:
    Injection → RingBuffer → PipelineAdapter → Alert

Usage:
    python tests/debug_scenario_runner.py

Output:
    tests/debug_report.txt - Detailed forensic report
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import simulation components
from src.simulation.core.orchestrator import SimulationOrchestrator, OrchestratorConfig
from src.simulation.processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig
from src.simulation.generators.patient_generator import PatientGenerator, PatientConfig
from src.simulation.events.event_types import (
    EventType,
    EventSeverity,
    LateDecelerationParams,
    VariableDecelerationParams,
    ProlongedDecelerationParams,
    BradycardiaParams,
    TachycardiaParams,
    VariabilityParams,
    SinusoidalParams,
    TachysystoleParams,
)
import numpy as np


# ============================================================================
# All 9 God Mode Event Definitions
# ============================================================================
GOD_MODE_EVENTS = [
    {
        "name": "LATE_DECELERATION",
        "event_type": EventType.LATE_DECELERATION,
        "params_factory": LateDecelerationParams.severe,
        "expected_detection": "late decel count > 0",
        "min_ticks": 300,  # Need contractions + lag time
    },
    {
        "name": "VARIABLE_DECELERATION",
        "event_type": EventType.VARIABLE_DECELERATION,
        "params_factory": VariableDecelerationParams.severe,
        "expected_detection": "variable decel count > 0",
        "min_ticks": 300,
    },
    {
        "name": "PROLONGED_DECELERATION",
        "event_type": EventType.PROLONGED_DECELERATION,
        "params_factory": ProlongedDecelerationParams.severe,
        "expected_detection": "FHR drop sustained > 2 min",
        "min_ticks": 200,
    },
    {
        "name": "BRADYCARDIA",
        "event_type": EventType.BRADYCARDIA,
        "params_factory": BradycardiaParams.severe,
        "expected_detection": "baseline < 110 bpm",
        "min_ticks": 300,
    },
    {
        "name": "TACHYCARDIA",
        "event_type": EventType.TACHYCARDIA,
        "params_factory": TachycardiaParams.severe,
        "expected_detection": "baseline > 160 bpm",
        "min_ticks": 300,
    },
    {
        "name": "ABSENT_VARIABILITY",
        "event_type": EventType.ABSENT_VARIABILITY,
        "params_factory": VariabilityParams.absent,
        "expected_detection": "variability category = ABSENT",
        "min_ticks": 300,
    },
    {
        "name": "MINIMAL_VARIABILITY",
        "event_type": EventType.MINIMAL_VARIABILITY,
        "params_factory": VariabilityParams.minimal,
        "expected_detection": "variability < 5 bpm",
        "min_ticks": 300,
    },
    {
        "name": "SINUSOIDAL_PATTERN",
        "event_type": EventType.SINUSOIDAL_PATTERN,
        "params_factory": SinusoidalParams.typical,
        "expected_detection": "sinusoidal detected = True",
        "min_ticks": 400,  # Needs longer for FFT detection
    },
    {
        "name": "TACHYSYSTOLE",
        "event_type": EventType.TACHYSYSTOLE,
        "params_factory": TachysystoleParams.severe,
        "expected_detection": "contractions > 5 per 10 min",
        "min_ticks": 300,
    },
]


class ForensicReport:
    """Collects and formats forensic findings."""
    
    def __init__(self):
        self.entries: List[str] = []
        self.event_results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
    
    def add(self, message: str):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.entries.append(f"[{timestamp}] {message}")
        logger.info(message)
    
    def add_section(self, title: str):
        """Add a section header."""
        self.entries.append("")
        self.entries.append("=" * 70)
        self.entries.append(f"  {title}")
        self.entries.append("=" * 70)
        logger.info(f"\n{'='*50}\n{title}\n{'='*50}")
    
    def add_event_result(self, event_name: str, result: Dict[str, Any]):
        """Record event test result."""
        self.event_results.append({
            "event": event_name,
            **result
        })
    
    def save(self, filepath: Path):
        """Save report to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("SENTINELFETAL FORENSIC AUDIT REPORT\n")
            f.write(f"Generated: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            # Summary Table
            f.write("DETECTION SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Event Type':<25} | {'Injected':<8} | {'In Buffer':<10} | {'Detected':<10} | {'Category':<8}\n")
            f.write("-" * 70 + "\n")
            
            for result in self.event_results:
                event = result.get('event', 'Unknown')[:24]
                injected = '✓' if result.get('injected') else '✗'
                in_buffer = '✓' if result.get('in_buffer') else '✗'
                detected = '✓' if result.get('detected') else '✗'
                category = result.get('category', '-')
                f.write(f"{event:<25} | {injected:<8} | {in_buffer:<10} | {detected:<10} | {category:<8}\n")
            
            f.write("-" * 70 + "\n\n")
            
            # Detailed Log
            f.write("DETAILED LOG\n")
            f.write("-" * 70 + "\n")
            for entry in self.entries:
                f.write(entry + "\n")
        
        logger.info(f"Report saved to: {filepath}")


def trace_event(
    event_config: Dict[str, Any],
    report: ForensicReport
) -> Dict[str, Any]:
    """
    Trace a single event through the complete data flow.
    
    Steps:
        A. Inject event programmatically
        B. Generate ticks to populate buffer
        C. Check buffer for FHR changes
        D. Run pipeline analysis
        E. Check detection result
    """
    event_name = event_config["name"]
    event_type = event_config["event_type"]
    params_factory = event_config["params_factory"]
    min_ticks = event_config["min_ticks"]
    
    report.add_section(f"Testing: {event_name}")
    
    result = {
        "injected": False,
        "in_buffer": False,
        "detected": False,
        "category": None,
        "raw_fhr_min": None,
        "raw_fhr_max": None,
        "pipeline_output": None,
        "rejection_reason": None,
    }
    
    try:
        # Create fresh patient and pipeline for isolation
        patient_config = PatientConfig(
            patient_id="TEST_P1",
            bed_number=1,
            name="Test Patient",
            baseline_fhr=140.0,
            baseline_variability=10.0,
        )
        patient = PatientGenerator(patient_config)
        pipeline = PipelineAdapter(PipelineAdapterConfig(
            sampling_rate=4.0,
            min_data_seconds=60.0,
        ))
        
        # Step A: Inject Event
        report.add(f"Step A: Injecting {event_name}...")
        params = params_factory()
        event = patient.inject_event(
            event_type=event_type,
            params=params,
            duration_seconds=300.0  # 5 minutes
        )
        result["injected"] = True
        report.add(f"  ✓ Event injected: {event.event_type.name} | duration={event.end_time - event.start_time:.0f}s")
        
        # Step B: Generate Ticks
        report.add(f"Step B: Generating {min_ticks} ticks...")
        samples_per_tick = 4  # 1 second at 4Hz
        
        for tick in range(min_ticks):
            patient.generate_tick(samples_per_tick)
        
        buffer_size = patient.buffer.size
        report.add(f"  ✓ Buffer filled: {buffer_size} samples ({buffer_size/4:.0f}s)")
        
        # Step C: Check Buffer for FHR Changes
        report.add("Step C: Analyzing raw buffer data...")
        buffer_data = patient.get_buffer_data()
        fhr = buffer_data.get('fhr', np.array([]))
        uc = buffer_data.get('uc', np.array([]))
        
        if len(fhr) > 0:
            result["raw_fhr_min"] = float(np.min(fhr))
            result["raw_fhr_max"] = float(np.max(fhr))
            fhr_mean = float(np.mean(fhr))
            fhr_std = float(np.std(fhr))
            
            report.add(f"  FHR Stats: min={result['raw_fhr_min']:.1f}, max={result['raw_fhr_max']:.1f}, mean={fhr_mean:.1f}, std={fhr_std:.1f}")
            
            # Check if event affected the signal
            baseline = patient_config.baseline_fhr
            
            if event_type in [EventType.BRADYCARDIA]:
                if result["raw_fhr_min"] < baseline - 20:
                    result["in_buffer"] = True
                    report.add(f"  ✓ Bradycardia visible: min FHR = {result['raw_fhr_min']:.1f} (baseline={baseline})")
                else:
                    report.add(f"  ✗ Bradycardia NOT visible in buffer")
            
            elif event_type in [EventType.TACHYCARDIA]:
                if result["raw_fhr_max"] > baseline + 20:
                    result["in_buffer"] = True
                    report.add(f"  ✓ Tachycardia visible: max FHR = {result['raw_fhr_max']:.1f} (baseline={baseline})")
                else:
                    report.add(f"  ✗ Tachycardia NOT visible in buffer")
            
            elif event_type in [EventType.LATE_DECELERATION, EventType.VARIABLE_DECELERATION, 
                               EventType.PROLONGED_DECELERATION]:
                if result["raw_fhr_min"] < baseline - 15:
                    result["in_buffer"] = True
                    report.add(f"  ✓ Deceleration visible: min FHR = {result['raw_fhr_min']:.1f}")
                else:
                    report.add(f"  ✗ Deceleration NOT visible in buffer")
            
            elif event_type in [EventType.ABSENT_VARIABILITY, EventType.MINIMAL_VARIABILITY]:
                if fhr_std < 5:
                    result["in_buffer"] = True
                    report.add(f"  ✓ Reduced variability visible: std = {fhr_std:.2f}")
                else:
                    report.add(f"  ✗ Variability reduction NOT visible (std={fhr_std:.2f})")
            
            elif event_type == EventType.SINUSOIDAL_PATTERN:
                # Check for sinusoidal-like oscillation
                result["in_buffer"] = True  # Assume visible if std is reasonable
                report.add(f"  ? Sinusoidal pattern: std = {fhr_std:.2f} (requires FFT for confirmation)")
            
            elif event_type == EventType.TACHYSYSTOLE:
                # Check UC for increased contractions
                if len(uc) > 0:
                    uc_peaks = np.sum(uc > 50)
                    result["in_buffer"] = uc_peaks > 5
                    report.add(f"  ? Tachysystole: {uc_peaks} high UC samples detected")
            
            else:
                result["in_buffer"] = True  # Default assume visible
        
        # Step D: Run Pipeline Analysis
        report.add("Step D: Running PipelineAdapter analysis...")
        
        pipeline_input = {
            'fhr': fhr,
            'uc': uc,
            'timestamps': buffer_data.get('timestamps', np.arange(len(fhr)) / 4.0),
        }
        
        pipeline_result = pipeline.process_patient(
            patient_id="TEST_P1",
            data=pipeline_input,
            run_moment=True,
        )
        
        result["pipeline_output"] = pipeline_result
        result["category"] = pipeline_result.get('category')
        
        report.add(f"  Pipeline Result:")
        report.add(f"    - Category: {result['category']}")
        report.add(f"    - Confidence: {pipeline_result.get('confidence', 0):.2f}")
        report.add(f"    - Insufficient Data: {pipeline_result.get('insufficient_data', False)}")
        report.add(f"    - Was Overridden: {pipeline_result.get('was_overridden', False)}")
        
        # Step E: Check Detection
        report.add("Step E: Checking detection result...")
        
        findings = pipeline_result.get('findings', {})
        
        if event_type in [EventType.LATE_DECELERATION]:
            decels = findings.get('decelerations', {})
            late_count = decels.get('late', 0)
            if late_count > 0:
                result["detected"] = True
                report.add(f"  ✓ DETECTED: {late_count} late decelerations")
            else:
                report.add(f"  ✗ NOT DETECTED: late decel count = 0")
                result["rejection_reason"] = "Late decel detection algorithm did not fire"
        
        elif event_type in [EventType.VARIABLE_DECELERATION]:
            decels = findings.get('decelerations', {})
            var_count = decels.get('variable', 0)
            if var_count > 0:
                result["detected"] = True
                report.add(f"  ✓ DETECTED: {var_count} variable decelerations")
            else:
                report.add(f"  ✗ NOT DETECTED: variable decel count = 0")
                result["rejection_reason"] = "Variable decel detection algorithm did not fire"
        
        elif event_type in [EventType.PROLONGED_DECELERATION]:
            decels = findings.get('decelerations', {})
            total = decels.get('total', 0)
            if total > 0 or result["category"] >= 2:
                result["detected"] = True
                report.add(f"  ✓ DETECTED: prolonged decel triggered cat {result['category']}")
            else:
                report.add(f"  ✗ NOT DETECTED: prolonged decel not found")
        
        elif event_type == EventType.BRADYCARDIA:
            baseline = findings.get('baseline', {})
            is_brady = baseline.get('is_bradycardia', False)
            value = baseline.get('value', 0)
            if is_brady:
                result["detected"] = True
                report.add(f"  ✓ DETECTED: bradycardia (baseline={value:.1f})")
            else:
                report.add(f"  ✗ NOT DETECTED: bradycardia flag = False (baseline={value:.1f})")
                result["rejection_reason"] = f"Baseline {value:.1f} >= 110 threshold"
        
        elif event_type == EventType.TACHYCARDIA:
            baseline = findings.get('baseline', {})
            is_tachy = baseline.get('is_tachycardia', False)
            value = baseline.get('value', 0)
            if is_tachy:
                result["detected"] = True
                report.add(f"  ✓ DETECTED: tachycardia (baseline={value:.1f})")
            else:
                report.add(f"  ✗ NOT DETECTED: tachycardia flag = False (baseline={value:.1f})")
                result["rejection_reason"] = f"Baseline {value:.1f} <= 160 threshold"
        
        elif event_type in [EventType.ABSENT_VARIABILITY, EventType.MINIMAL_VARIABILITY]:
            variability = findings.get('variability', {})
            var_cat = variability.get('category', '')
            var_val = variability.get('value', 0)
            if var_cat in ['ABSENT', 'MINIMAL'] or var_val < 5:
                result["detected"] = True
                report.add(f"  ✓ DETECTED: variability = {var_cat} ({var_val:.1f} bpm)")
            else:
                report.add(f"  ✗ NOT DETECTED: variability = {var_cat} ({var_val:.1f} bpm)")
                result["rejection_reason"] = f"Variability {var_val:.1f} > threshold"
        
        elif event_type == EventType.SINUSOIDAL_PATTERN:
            sino = findings.get('sinusoidal', {})
            detected = sino.get('detected', False)
            conf = sino.get('confidence', 0)
            if detected or result["category"] >= 3:
                result["detected"] = True
                report.add(f"  ✓ DETECTED: sinusoidal (conf={conf:.2f})")
            else:
                report.add(f"  ✗ NOT DETECTED: sinusoidal (conf={conf:.2f})")
                result["rejection_reason"] = "FFT sinusoidal detection requires 20+ min of specific pattern"
        
        elif event_type == EventType.TACHYSYSTOLE:
            tachy = findings.get('tachysystole', {})
            detected = tachy.get('detected', False)
            rate = tachy.get('rate', 0)
            if detected:
                result["detected"] = True
                report.add(f"  ✓ DETECTED: tachysystole ({rate:.1f} contractions/10min)")
            else:
                report.add(f"  ✗ NOT DETECTED: tachysystole (rate={rate:.1f})")
                result["rejection_reason"] = "Contraction rate below 5/10min threshold"
        
        # Overall category check
        if result["category"] and result["category"] >= 2:
            if not result["detected"]:
                result["detected"] = True
                report.add(f"  ✓ Category {result['category']} triggered (override or other finding)")
        
    except Exception as e:
        report.add(f"  ✗ ERROR: {e}")
        result["rejection_reason"] = str(e)
        import traceback
        report.add(traceback.format_exc())
    
    return result


def main():
    """Run the complete forensic audit."""
    report = ForensicReport()
    
    report.add_section("FORENSIC AUDIT STARTED")
    report.add("Testing all 9 God Mode event types...")
    report.add(f"Events to test: {len(GOD_MODE_EVENTS)}")
    
    for event_config in GOD_MODE_EVENTS:
        result = trace_event(event_config, report)
        report.add_event_result(event_config["name"], result)
    
    # Summary
    report.add_section("AUDIT COMPLETE")
    
    detected_count = sum(1 for r in report.event_results if r.get('detected'))
    in_buffer_count = sum(1 for r in report.event_results if r.get('in_buffer'))
    
    report.add(f"Total Events Tested: {len(GOD_MODE_EVENTS)}")
    report.add(f"Events Visible in Buffer: {in_buffer_count}/{len(GOD_MODE_EVENTS)}")
    report.add(f"Events Detected by AI: {detected_count}/{len(GOD_MODE_EVENTS)}")
    
    if detected_count < len(GOD_MODE_EVENTS):
        report.add("")
        report.add("FAILURES:")
        for r in report.event_results:
            if not r.get('detected'):
                report.add(f"  - {r['event']}: {r.get('rejection_reason', 'Unknown')}")
    
    # Save report
    output_path = ROOT / "tests" / "debug_report.txt"
    report.save(output_path)
    
    print(f"\n{'='*60}")
    print(f"FORENSIC REPORT SAVED: {output_path}")
    print(f"{'='*60}")
    print(f"Detection Success Rate: {detected_count}/{len(GOD_MODE_EVENTS)} ({100*detected_count/len(GOD_MODE_EVENTS):.0f}%)")


if __name__ == "__main__":
    main()
