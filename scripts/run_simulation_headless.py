#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Headless Simulation Runner for Performance Profiling.

Simulates the UI loop without Streamlit overhead to isolate
backend processing bottlenecks.

Usage:
    python -m cProfile -o profile_output.prof scripts/run_simulation_headless.py
    
    # Then analyze:
    python scripts/analyze_profile.py

Author: SentinelFetal Performance Architect
Date: 2026-01-23
"""

import sys
import time
import cProfile
import pstats
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, Any

from src.simulation import (
    SimulationOrchestrator,
    OrchestratorConfig,
)
from src.simulation.processing.pipeline_adapter import PipelineAdapter, PipelineAdapterConfig


class HeadlessSimulator:
    """
    Headless simulator that mimics UI processing loop without Streamlit.
    
    Isolates backend performance from UI rendering overhead.
    """
    
    def __init__(self, num_patients: int = 1):
        """Initialize headless simulator."""
        print(f"[INIT] Initializing headless simulator with {num_patients} patients...")
        
        # Create pipeline adapter
        self.adapter = PipelineAdapter(PipelineAdapterConfig(
            use_real_moment=True,
            model_path="models/sentinel_classifier.json"
        ))
        
        def processing_callback(patient_id: str, data: Dict) -> Dict:
            return self.adapter.process_patient(patient_id, data, run_moment=True)
        
        # Create orchestrator
        config = OrchestratorConfig(
            num_patients=num_patients,
            sampling_rate=4.0,
            tick_interval_seconds=1.0,
            moment_interval_seconds=30.0
        )
        
        self.orchestrator = SimulationOrchestrator(config, processing_callback)
        print(f"[INIT] Initialization complete.")
    
    def run_simulation(self, duration_seconds: int = 60, update_interval: float = 0.333):
        """
        Run headless simulation for specified duration.
        
        Mimics UI update loop at ~3 FPS (default).
        
        Args:
            duration_seconds: How long to run simulation
            update_interval: Time between updates (0.333 = 3 FPS)
        """
        print(f"\n[RUN] Starting {duration_seconds}s simulation, update interval={update_interval}s")
        
        self.orchestrator.start()
        
        start_time = time.time()
        update_count = 0
        total_process_time = 0.0
        max_process_time = 0.0
        min_process_time = float('inf')
        
        latencies = []
        
        while (time.time() - start_time) < duration_seconds:
            loop_start = time.perf_counter()
            
            # Get all patient statuses (simulates UI refresh)
            statuses = self.orchestrator.get_all_patients_status()
            
            # Process each patient (simulates what ward view does)
            for status in statuses:
                patient_id = status['patient_id']
                patient = self.orchestrator.get_patient(patient_id)
                
                if patient:
                    # Get buffer data (simulates sparkline rendering)
                    data = patient.get_buffer_data(duration_minutes=1)
                    fhr = data.get('fhr', np.array([]))
                    
                    # This is what the UI does - access latest results
                    _ = patient.latest_category
                    _ = patient.latest_alert
            
            loop_end = time.perf_counter()
            process_time = (loop_end - loop_start) * 1000  # ms
            
            total_process_time += process_time
            max_process_time = max(max_process_time, process_time)
            min_process_time = min(min_process_time, process_time)
            latencies.append(process_time)
            
            update_count += 1
            
            # Simulate UI refresh rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, update_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.orchestrator.stop()
        
        # Calculate statistics
        avg_latency = total_process_time / update_count if update_count > 0 else 0
        
        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted)//2] if latencies_sorted else 0
        p95 = latencies_sorted[int(len(latencies_sorted)*0.95)] if latencies_sorted else 0
        p99 = latencies_sorted[int(len(latencies_sorted)*0.99)] if latencies_sorted else 0
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        print(f"Duration: {duration_seconds}s")
        print(f"Patients: {self.orchestrator.config.num_patients}")
        print(f"Updates: {update_count}")
        print(f"Target FPS: {1/update_interval:.1f}")
        print(f"Actual FPS: {update_count/duration_seconds:.1f}")
        print("-"*60)
        print(f"Avg Latency: {avg_latency:.2f}ms")
        print(f"Min Latency: {min_process_time:.2f}ms")
        print(f"Max Latency: {max_process_time:.2f}ms")
        print(f"P50 Latency: {p50:.2f}ms")
        print(f"P95 Latency: {p95:.2f}ms")
        print(f"P99 Latency: {p99:.2f}ms")
        print("-"*60)
        
        # Assess
        if avg_latency < 50:
            print("✅ PASS: Latency < 50ms requirement")
        else:
            print("❌ FAIL: Latency > 50ms requirement")
        
        return {
            'updates': update_count,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_process_time,
            'p95_latency_ms': p95,
            'p99_latency_ms': p99,
            'actual_fps': update_count / duration_seconds
        }
    
    def profile_pipeline(self, iterations: int = 100):
        """
        Profile the pipeline adapter directly.
        
        Isolates pipeline processing time from orchestrator overhead.
        """
        print(f"\n[PROFILE] Profiling pipeline adapter ({iterations} iterations)...")
        
        # Generate synthetic data
        np.random.seed(42)
        fhr = 140 + np.random.randn(480) * 5  # 2 minutes at 4Hz
        uc = np.abs(np.sin(np.linspace(0, 4*np.pi, 480))) * 50 + 10
        
        data = {
            'fhr': fhr,
            'uc': uc,
            'timestamps': np.linspace(0, 120, 480)
        }
        
        latencies = []
        
        for i in range(iterations):
            start = time.perf_counter()
            result = self.adapter.process_patient(f"P_PROFILE", data, run_moment=True, compute_shap=False)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        avg = np.mean(latencies)
        std = np.std(latencies)
        p95 = np.percentile(latencies, 95)
        
        print(f"\nPipeline Processing Time ({iterations} iterations):")
        print(f"  Average: {avg:.2f}ms ± {std:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        
        return {
            'avg_ms': avg,
            'std_ms': std,
            'p95_ms': p95,
            'samples': latencies
        }


def run_with_cprofile():
    """Run simulation with cProfile for detailed function-level analysis."""
    profiler = cProfile.Profile()
    
    sim = HeadlessSimulator(num_patients=1)
    
    profiler.enable()
    sim.run_simulation(duration_seconds=60, update_interval=0.333)
    profiler.disable()
    
    # Save profile
    profiler.dump_stats('profile_output.prof')
    print("\n[PROFILE] Saved to profile_output.prof")
    
    # Print top 30 functions by cumulative time
    print("\n" + "="*80)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("="*80)
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(30)


def main():
    """Main entry point."""
    print("="*60)
    print("SENTINELFETAL HEADLESS PERFORMANCE PROFILER")
    print("="*60)
    
    # Test 1: Single patient baseline
    print("\n[TEST 1] Single Patient Baseline")
    sim1 = HeadlessSimulator(num_patients=1)
    result1 = sim1.run_simulation(duration_seconds=30, update_interval=0.333)
    
    # Test 2: Pipeline profiling
    print("\n[TEST 2] Pipeline Adapter Profiling")
    sim1.profile_pipeline(iterations=50)
    
    # Test 3: Multi-patient scaling
    print("\n[TEST 3] Multi-Patient Scaling Test")
    for n_patients in [4, 8, 12, 16, 20]:
        print(f"\n--- {n_patients} Patients ---")
        sim = HeadlessSimulator(num_patients=n_patients)
        result = sim.run_simulation(duration_seconds=15, update_interval=0.333)
        
        if result['avg_latency_ms'] > 100:
            print(f"⚠️  WARNING: Latency too high at {n_patients} patients!")
            print(f"   Max capacity likely around {n_patients-4} patients")
            break


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Headless simulation profiler')
    parser.add_argument('--profile', action='store_true', help='Run with cProfile')
    parser.add_argument('--patients', type=int, default=1, help='Number of patients')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    
    args = parser.parse_args()
    
    if args.profile:
        run_with_cprofile()
    else:
        main()
