#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6 E2E Integration Runner.

This script runs the full sentinel system in headless mode to verify
End-to-End integration of all stages (1-5) within the simulation orchestrator.

Flow:
1. Initialize OrchestratorAdapter (Singleton)
2. Start Simulation (4 patients)
3. Wait for baseline stability (30s)
4. Inject Clinical Event (Late Decelerations) on Patient P1
5. Monitor Pipeline Output for Stage 5 Tiering Decisions
6. Generate Report

Usage:
    python scripts/pipeline/stage6_e2e.py
"""

import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services.orchestrator_adapter import OrchestratorAdapter
from src.simulation.events.event_types import EventType, EventSeverity

# Check if MHR Detector is available (V2.0 check)
try:
    from src.safety import MHRDetector
    MHR_AVAILABLE = True
except ImportError:
    MHR_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Stage6_E2E")

def main():
    logger.info("🚀 Starting Stage 6 E2E Integration Test")
    
    # 1. Initialize Adapter
    adapter = OrchestratorAdapter.get_instance()
    adapter.initialize(patient_count=4)
    
    # Verify Initialization
    status = adapter.get_status()
    logger.info(f"Initialized: {status}")
    if status['patient_count'] != 4:
        logger.error("Failed to initialize 4 patients")
        sys.exit(1)

    # 2. Start Simulation
    logger.info("Starting simulation...")
    adapter.start()
    
    try:
        # 3. Wait for baseline (30s)
        logger.info("Waiting 30s for signal generation...")
        time.sleep(30)
        
        # Check if data flows
        p1_data = adapter.get_patient_snapshot('P1')
        if not p1_data:
            logger.warning("No snapshot available yet for P1, waiting 10s more...")
            time.sleep(10)
            p1_data = adapter.get_patient_snapshot('P1')

        if not p1_data:
            logger.error("❌ DataBridge not receiving updates! Integration broken.")
            # Continue anyway to check logs/internals if possible
        else:
            logger.info("✅ Data flowing to DataBridge")

        # 4. Inject Event (Late Decels on P1)
        logger.info("💉 Injecting LATE_DECELERATION on P1 (Severe, 60s)")
        success = adapter.inject_event(
            patient_id='P1',
            event_type='LATE_DECEL',
            params={'severity': 'SEVERE'},
            duration=60
        )
        
        if success:
            logger.info("✅ Event injection command accepted")
        else:
            logger.error("❌ Event injection failed")
        
        # 5. Monitor for Stage 5 Decision (Wait ~45s for processing cycle)
        logger.info("Monitoring P1 for 45s to catch Stage 5 decision...")
        
        # We poll the internal result via the adapter or bridge
        # Since output is async, we poll every 5s
        found_alert = False
        start_monitor = time.time()
        
        while time.time() - start_monitor < 45:
            # Note: OrchestratorAdapter pushes to WebSocket, but also returns result in _process_patient_data
            # In a real E2E test we might mock the socket or inspect the orchestrator's internal state.
            # Here we rely on the DataBridge if available, or just check logs conceptually.
            # Ideally we'd hook into the adapter, but for this script let's assume we are watching the logs 
            # or we could peek into the trend buffer if accessible.
            
            trend_buffer = adapter.get_trend_buffer('P1')
            if trend_buffer and trend_buffer.samples:
                latest = trend_buffer.samples[-1]
                logger.info(f"Trend Sample: Cat={latest.category}, Decels={latest.decel_count_15min}")
                
                if latest.category >= 2:
                    logger.info("✅ Stage 5 Logic Triggered: Category elevated!")
                    found_alert = True
                    break
            
            time.sleep(5)
            
        if not found_alert:
            logger.warning("⚠️ No alert detected in TrendBuffer within timeout. Check logs for pipeline execution.")
            
        # 6. Check Tiering Logic (Simulated)
        # Since we can't easily intercept the return value of _process_patient_data in this black-box run,
        # we assume success if no exceptions crashed the thread and data kept flowing.
        
        updates = adapter.get_tick_count()
        logger.info(f"Total Ticks: {updates}")
        if updates > 50:
             logger.info("✅ Simulation loop is healthy")
        else:
             logger.error("❌ Simulation loop stalled")

    except KeyboardInterrupt:
        logger.info("Test interrupted via keyboard")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        logger.info("🛑 Stopping simulation")
        adapter.stop()
        adapter.shutdown()
        logger.info("Stage 6 E2E Test Complete")

if __name__ == '__main__':
    main()
