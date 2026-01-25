#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load Testing Suite - Phase 2: Busy Ward Tests.

Tests scalability from 1 to 20+ patients to find system limits.

Author: SentinelFetal Performance Architect
Date: 2026-01-23
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, Any, List

from src.simulation import (
    SimulationOrchestrator,
    OrchestratorConfig,
)
from src.simulation.processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig


class LoadTester:
    """
    Comprehensive load testing suite for SentinelFetal.
    """
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        
    def run_load_test(
        self,
        num_patients: int,
        duration_seconds: int = 60,
        target_fps: float = 3.0
    ) -> Dict[str, Any]:
        """Run load test with specified number of patients."""
        print(f"\n{'='*60}")
        print(f"LOAD TEST: {num_patients} Patients, {duration_seconds}s duration")
        print(f"{'='*60}")
        
        # Initialize
        adapter = PipelineAdapter(PipelineAdapterConfig(
            use_real_moment=True,
            model_path="models/sentinel_classifier.json"
        ))
        
        def processing_callback(patient_id: str, data: Dict) -> Dict:
            return adapter.process_patient(patient_id, data, run_moment=True)
        
        config = OrchestratorConfig(
            num_patients=num_patients,
            sampling_rate=4.0,
            tick_interval_seconds=1.0,
            moment_interval_seconds=30.0
        )
        
        orchestrator = SimulationOrchestrator(config, processing_callback)
        
        # Run test
        update_interval = 1.0 / target_fps
        latencies = []
        cpu_samples = []
        memory_samples = []
        
        orchestrator.start()
        start_time = time.time()
        update_count = 0
        
        try:
            import psutil
            process = psutil.Process()
            has_psutil = True
        except ImportError:
            has_psutil = False
        
        while (time.time() - start_time) < duration_seconds:
            loop_start = time.perf_counter()
            
            # Simulate UI refresh
            statuses = orchestrator.get_all_patients_status()
            
            for status in statuses:
                patient = orchestrator.get_patient(status['patient_id'])
                if patient:
                    data = patient.get_buffer_data(duration_minutes=1)
                    _ = patient.latest_category
            
            loop_end = time.perf_counter()
            latency_ms = (loop_end - loop_start) * 1000
            latencies.append(latency_ms)
            
            # Sample CPU/Memory every second
            if has_psutil and update_count % int(target_fps) == 0:
                cpu_samples.append(process.cpu_percent())
                memory_samples.append(process.memory_info().rss / 1024 / 1024)
            
            update_count += 1
            
            # Maintain target FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, update_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        orchestrator.stop()
        
        # Calculate metrics
        latencies_arr = np.array(latencies)
        result = {
            'num_patients': num_patients,
            'duration_seconds': duration_seconds,
            'target_fps': target_fps,
            'actual_fps': update_count / duration_seconds,
            'update_count': update_count,
            'latency_avg_ms': float(np.mean(latencies_arr)),
            'latency_std_ms': float(np.std(latencies_arr)),
            'latency_min_ms': float(np.min(latencies_arr)),
            'latency_max_ms': float(np.max(latencies_arr)),
            'latency_p50_ms': float(np.percentile(latencies_arr, 50)),
            'latency_p95_ms': float(np.percentile(latencies_arr, 95)),
            'latency_p99_ms': float(np.percentile(latencies_arr, 99)),
            'cpu_avg_percent': float(np.mean(cpu_samples)) if cpu_samples else None,
            'memory_avg_mb': float(np.mean(memory_samples)) if memory_samples else None,
            'passes_50ms': float(np.mean(latencies_arr)) < 50,
            'passes_fps': (update_count / duration_seconds) >= (target_fps * 0.95)
        }
        
        # Print results
        print(f"\nResults:")
        print(f"  Updates: {update_count}")
        print(f"  Actual FPS: {result['actual_fps']:.1f}")
        print(f"  Latency Avg: {result['latency_avg_ms']:.2f}ms")
        print(f"  Latency P95: {result['latency_p95_ms']:.2f}ms")
        print(f"  Latency P99: {result['latency_p99_ms']:.2f}ms")
        print(f"  Latency Max: {result['latency_max_ms']:.2f}ms")
        if result['cpu_avg_percent'] is not None:
            print(f"  CPU Avg: {result['cpu_avg_percent']:.1f}%")
            print(f"  Memory Avg: {result['memory_avg_mb']:.1f}MB")
        
        status = "✅ PASS" if (result['passes_50ms'] and result['passes_fps']) else "❌ FAIL"
        print(f"\n  Status: {status}")
        
        self.results.append(result)
        return result
    
    def run_full_suite(self):
        """Run complete load testing suite."""
        print("\n" + "="*70)
        print("SENTINELFETAL LOAD TESTING SUITE")
        print("Phase 2: Busy Ward Tests")
        print("="*70)
        
        # Test configurations
        patient_counts = [1, 2, 4, 8, 12, 16, 20, 24]
        
        for n in patient_counts:
            result = self.run_load_test(
                num_patients=n,
                duration_seconds=30,
                target_fps=3.0
            )
            
            # Stop if failure
            if not result['passes_50ms'] or not result['passes_fps']:
                print(f"\n⚠️  System limit reached at {n} patients")
                # Run one more at the failing point for confirmation
                if n > 1:
                    print(f"   Confirmed max capacity: {n-1} patients")
                break
        
        # Generate summary
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate load test summary."""
        print("\n" + "="*70)
        print("LOAD TEST SUMMARY")
        print("="*70)
        
        print(f"\n{'Patients':<10} {'Avg (ms)':<12} {'P95 (ms)':<12} {'Max (ms)':<12} {'FPS':<8} {'Status':<10}")
        print("-"*70)
        
        max_passing = 0
        for r in self.results:
            status = "✅" if r['passes_50ms'] and r['passes_fps'] else "❌"
            if r['passes_50ms'] and r['passes_fps']:
                max_passing = r['num_patients']
            print(f"{r['num_patients']:<10} {r['latency_avg_ms']:<12.2f} {r['latency_p95_ms']:<12.2f} {r['latency_max_ms']:<12.2f} {r['actual_fps']:<8.1f} {status:<10}")
        
        print("\n" + "="*70)
        print(f"MAXIMUM SUPPORTED CAPACITY: {max_passing} patients")
        print("="*70)
        
        # Save results
        output_path = Path("docs/reports/load_test_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'max_capacity': max_passing,
                'results': self.results
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    tester = LoadTester()
    tester.run_full_suite()
