#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification Script for Stage 6: E2E Integration.

This script validates the integrity of the Stage 6 implementation by checking
file existence, importability, and basic function contracts.

Checks:
1. Stage 6 E2E Runner script exists
2. PipelineAdapter correctly integrates Stage 5
3. WebSocket endpoints are configured
4. Orchestrator integration points are present

Usage:
    python scripts/validation/verify_stage6.py
"""

import sys
import unittest
import importlib
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class TestStage6Integration(unittest.TestCase):
    
    def test_scripts_exist(self):
        """Verify that required scripts exist."""
        scripts = [
            ROOT / "scripts/pipeline/stage6_e2e.py",
        ]
        for script in scripts:
            self.assertTrue(script.exists(), f"Missing script: {script}")

    def test_pipeline_adapter_imports(self):
        """Verify PipelineAdapter imports Stage 5 components."""
        try:
            from api.services.orchestrator_adapter import OrchestratorAdapter
            from src.simulation.processing.pipeline_adapter import PipelineAdapter
            
            # Check if Stage5Pipeline is imported/used (inspecting source or attributes)
            # Since we can't easily inspect source at runtime in packaged envs, we try to instantiate
            adapter = PipelineAdapter()
            self.assertTrue(hasattr(adapter, '_stage5_pipeline'), "PipelineAdapter missing _stage5_pipeline attribute")
            
        except ImportError as e:
            self.fail(f"Failed to import adapters: {e}")
        except Exception as e:
            self.fail(f"Failed to instantiate PipelineAdapter: {e}")

    def test_websocket_endpoints(self):
        """Verify WebSocket router is importable."""
        try:
            from api.routers.websocket import router
            self.assertIsNotNone(router)
        except ImportError as e:
            self.fail(f"Failed to import WebSocket router: {e}")

    def test_orchestrator_integration(self):
        """Verify OrchestratorAdapter singleton."""
        from api.services.orchestrator_adapter import OrchestratorAdapter
        adapter = OrchestratorAdapter.get_instance()
        self.assertIsNotNone(adapter)
        
        # Check if it has methods we expect for Stage 6
        self.assertTrue(hasattr(adapter, 'inject_event'), "Missing inject_event method")
        self.assertTrue(hasattr(adapter, 'get_trend_buffer'), "Missing get_trend_buffer integration")

if __name__ == '__main__':
    print("🔍 Running Stage 6 Integration Verification...")
    unittest.main(verbosity=2)
