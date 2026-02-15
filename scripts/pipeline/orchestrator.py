#!/usr/bin/env python3
"""
SentinelFetal Pipeline Orchestrator

Professional pipeline manager that orchestrates the execution of Stages 1-4.
Provides dependency management, validation, error handling, and reporting.

Architecture:
    Stage 1: Data Preparation → X_norm_4hz.npy, manifest.csv
    Stage 2: Windowing → windows_index.csv
    Stage 3 (Training): AI Model Training → stage3_ai_pipeline.joblib
    Stage 3 (Eval): Generate Scores → ai_scores.parquet
    Stage 4: Calibration → thresholds.yaml
    
    Stages 5-6 are runtime components that use the outputs from Stages 1-4.

Usage:
    # Run full pipeline
    python scripts/pipeline/orchestrator.py --run-all
    
    # Run specific stages
    python scripts/pipeline/orchestrator.py --from-stage 3 --to-stage 4
    
    # Validate only
    python scripts/pipeline/orchestrator.py --validate-only
    
    # Resume from failed stage
    python scripts/pipeline/orchestrator.py --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "processed_data_v1"
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "pipeline"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PipelineOrchestrator")


class StageStatus(Enum):
    """Stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of a stage execution."""
    stage_num: int
    stage_name: str
    status: StageStatus
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    artifacts: List[str] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Stage 1
    stage1_script: Path = SCRIPTS_DIR / "build_dataset_v1.py"
    
    # Stage 2
    stage2_script: Path = SCRIPTS_DIR / "stage2_build.py"
    
    # Stage 3 Training
    stage3_train_script: Path = SCRIPTS_DIR / "train_stage3_minirocket_lr.py"
    
    # Stage 3 Evaluation
    stage3_eval_script: Path = PROJECT_ROOT / "scripts" / "validation" / "eval_stage3_minirocket_lr.py"
    
    # Stage 4
    stage4_script: Path = SCRIPTS_DIR / "stage4_calibrate.py"
    
    # Expected artifacts
    expected_artifacts: Dict[str, List[Path]] = None
    
    def __post_init__(self):
        if self.expected_artifacts is None:
            self.expected_artifacts = {
                "stage1": [
                    DATA_DIR / "X_norm_4hz.npy",
                    DATA_DIR / "manifest.csv",
                ],
                "stage2": [
                    DATA_DIR / "windows_index.csv",
                ],
                "stage3_train": [
                    DATA_DIR / "stage3_ai_pipeline.joblib",
                    DATA_DIR / "stage3_eval_report.json",
                    DATA_DIR / "STAGE3_TRUTH.md",
                ],
                "stage3_eval": [
                    DATA_DIR / "ai_scores.parquet",
                ],
                "stage4": [
                    PROJECT_ROOT / "models" / "thresholds.yaml",
                    PROJECT_ROOT / "models" / "stage4_calibration_report.json",
                ],
            }


class PipelineOrchestrator:
    """
    Professional pipeline orchestrator for SentinelFetal.
    
    Manages execution, validation, and reporting for Stages 1-4.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize orchestrator.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self.results: Dict[str, StageResult] = {}
        self.state_file = DATA_DIR / "pipeline_state.json"
        
    def validate_stage_artifacts(self, stage_key: str) -> tuple[bool, List[str]]:
        """
        Validate that all expected artifacts exist for a stage.
        
        Args:
            stage_key: Stage key (e.g., "stage1", "stage3_train")
            
        Returns:
            (is_valid, missing_files) tuple
        """
        expected = self.config.expected_artifacts.get(stage_key, [])
        missing = []
        
        for artifact_path in expected:
            if not artifact_path.exists():
                missing.append(str(artifact_path))
        
        is_valid = len(missing) == 0
        return is_valid, missing
    
    def validate_stage_output_quality(self, stage_key: str) -> tuple[bool, List[str]]:
        """
        Validate quality/integrity of stage outputs.
        
        Args:
            stage_key: Stage key
            
        Returns:
            (is_valid, error_messages) tuple
        """
        errors = []
        
        try:
            if stage_key == "stage1":
                # Validate X_norm_4hz.npy and manifest.csv
                X_norm = np.load(DATA_DIR / "X_norm_4hz.npy")
                manifest = pd.read_csv(DATA_DIR / "manifest.csv")
                
                if X_norm.ndim != 3:
                    errors.append(f"X_norm_4hz.npy has wrong dimensions: {X_norm.shape}")
                
                if X_norm.shape[2] != 2:
                    errors.append(f"X_norm_4hz.npy should have 2 channels, got {X_norm.shape[2]}")
                
                if len(manifest) == 0:
                    errors.append("manifest.csv is empty")
                
                required_cols = ["record_id", "ph", "samples"]
                missing_cols = [c for c in required_cols if c not in manifest.columns]
                if missing_cols:
                    errors.append(f"manifest.csv missing columns: {missing_cols}")
                    
            elif stage_key == "stage2":
                # Validate windows_index.csv
                windows = pd.read_csv(DATA_DIR / "windows_index.csv")
                
                if len(windows) == 0:
                    errors.append("windows_index.csv is empty")
                
                required_cols = ["record_id", "window_idx", "start", "end", "valid_for_ai"]
                missing_cols = [c for c in required_cols if c not in windows.columns]
                if missing_cols:
                    errors.append(f"windows_index.csv missing columns: {missing_cols}")
                    
            elif stage_key == "stage3_train":
                # Validate stage3_ai_pipeline.joblib
                import joblib
                pipeline = joblib.load(DATA_DIR / "stage3_ai_pipeline.joblib")
                
                required_keys = ["minirocket", "scaler", "lr", "model_version"]
                missing_keys = [k for k in required_keys if k not in pipeline]
                if missing_keys:
                    errors.append(f"Pipeline missing keys: {missing_keys}")
                    
            elif stage_key == "stage3_eval":
                # Validate ai_scores.parquet
                scores = pd.read_parquet(DATA_DIR / "ai_scores.parquet")
                
                if len(scores) == 0:
                    errors.append("ai_scores.parquet is empty")
                
                if "ai_score" not in scores.columns:
                    errors.append("ai_scores.parquet missing 'ai_score' column")
                
                # Check score range
                if "ai_score" in scores.columns:
                    min_score = scores["ai_score"].min()
                    max_score = scores["ai_score"].max()
                    if min_score < 0 or max_score > 1:
                        errors.append(f"ai_score out of range [0,1]: [{min_score:.4f}, {max_score:.4f}]")
                        
            elif stage_key == "stage4":
                # Validate thresholds.yaml
                try:
                    import yaml
                except ImportError as e:
                    errors.append(f"PyYAML is not installed: {e}")
                    return False, errors
                thresholds_path = PROJECT_ROOT / "models" / "thresholds.yaml"
                with open(thresholds_path) as f:
                    thresholds = yaml.safe_load(f)
                required_keys = ["t_low", "t_high"]
                missing_keys = [k for k in required_keys if k not in thresholds]
                if missing_keys:
                    errors.append(f"thresholds.yaml missing keys: {missing_keys}")
                    
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def run_stage(
        self,
        stage_key: str,
        stage_num: int,
        stage_name: str,
        script_path: Path,
        args: Optional[List[str]] = None,
        skip_if_exists: bool = True,
    ) -> StageResult:
        """
        Run a single pipeline stage.
        
        Args:
            stage_key: Stage key for artifact validation
            stage_num: Stage number
            stage_name: Human-readable stage name
            script_path: Path to stage script
            args: Additional command-line arguments
            skip_if_exists: Skip if artifacts already exist
            
        Returns:
            StageResult with execution details
        """
        logger.info("=" * 70)
        logger.info(f"STAGE {stage_num}: {stage_name}")
        logger.info("=" * 70)
        
        # Check if artifacts exist
        artifacts_exist, missing = self.validate_stage_artifacts(stage_key)
        if skip_if_exists and artifacts_exist:
            logger.info(f"✓ Artifacts already exist. Skipping.")
            return StageResult(
                stage_num=stage_num,
                stage_name=stage_name,
                status=StageStatus.SKIPPED,
                artifacts=[str(p) for p in self.config.expected_artifacts[stage_key]],
            )
        
        # Execute stage
        start_time = datetime.now()
        cmd = ["python3", str(script_path)]
        if args:
            cmd.extend(args)
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log output
            if result.stdout:
                logger.info("Output:")
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"  {line}")
            
            # Validate artifacts
            artifacts_exist, missing = self.validate_stage_artifacts(stage_key)
            if not artifacts_exist:
                raise RuntimeError(f"Stage completed but artifacts missing: {missing}")
            
            # Validate output quality
            quality_ok, errors = self.validate_stage_output_quality(stage_key)
            if not quality_ok:
                raise RuntimeError(f"Output validation failed: {errors}")
            
            logger.info(f"✓ Stage {stage_num} completed successfully in {duration:.1f}s")
            
            return StageResult(
                stage_num=stage_num,
                stage_name=stage_name,
                status=StageStatus.SUCCESS,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                artifacts=[str(p) for p in self.config.expected_artifacts[stage_key]],
            )
            
        except subprocess.CalledProcessError as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            error_msg = f"Exit code: {e.returncode}"
            if e.stderr:
                error_msg += f"\nStderr:\n{e.stderr}"
            
            logger.error(f"✗ Stage {stage_num} failed after {duration:.1f}s")
            logger.error(error_msg)
            
            return StageResult(
                stage_num=stage_num,
                stage_name=stage_name,
                status=StageStatus.FAILED,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                error_message=error_msg,
            )
        
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"✗ Stage {stage_num} failed after {duration:.1f}s")
            logger.error(error_msg)
            
            return StageResult(
                stage_num=stage_num,
                stage_name=stage_name,
                status=StageStatus.FAILED,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                error_message=error_msg,
            )
    
    def run_full_pipeline(self, from_stage: int = 1, to_stage: int = 4, skip_existing: bool = True) -> Dict[str, StageResult]:
        """
        Run the full pipeline from stage `from_stage` to `to_stage`.
        
        Args:
            from_stage: Starting stage (1-4)
            to_stage: Ending stage (1-4)
            skip_existing: Skip stages with existing artifacts
            
        Returns:
            Dictionary of stage results
        """
        logger.info("=" * 70)
        logger.info("SENTINELFETAL PIPELINE ORCHESTRATOR")
        logger.info("=" * 70)
        logger.info(f"Starting pipeline: Stages {from_stage} → {to_stage}")
        logger.info(f"Skip existing artifacts: {skip_existing}")
        logger.info("")
        
        pipeline_start = datetime.now()
        
        # Define stages
        stages = []
        
        if from_stage <= 1 <= to_stage:
            stages.append(("stage1", 1, "Data Preparation", self.config.stage1_script, None))
        
        if from_stage <= 2 <= to_stage:
            stages.append(("stage2", 2, "Windowing", self.config.stage2_script, None))
        
        if from_stage <= 3 <= to_stage:
            stages.append(("stage3_train", 3, "AI Training (MiniRocket + LR)", self.config.stage3_train_script, None))
            stages.append(("stage3_eval", 3, "AI Evaluation (Generate Scores)", self.config.stage3_eval_script, None))
        
        if from_stage <= 4 <= to_stage:
            # Stage 4 needs ai_scores and labels
            scores_path = DATA_DIR / "ai_scores.parquet"
            # Generate numpy files for stage 4 from parquet
            stage4_args = [
                "--scores", str(DATA_DIR / "ai_scores_stage4.npy"),
                "--labels", str(DATA_DIR / "labels_stage4.npy"),
                "--output", str(PROJECT_ROOT / "models" / "thresholds.yaml"),
            ]
            stages.append(("stage4", 4, "Calibration (Thresholds)", self.config.stage4_script, stage4_args))
        
        # Execute stages
        for stage_key, stage_num, stage_name, script_path, args in stages:
            result = self.run_stage(
                stage_key=stage_key,
                stage_num=stage_num,
                stage_name=stage_name,
                script_path=script_path,
                args=args,
                skip_if_exists=skip_existing,
            )
            
            self.results[stage_key] = result
            
            # Stop on failure
            if result.status == StageStatus.FAILED:
                logger.error(f"Pipeline stopped due to stage {stage_num} failure")
                break
        
        pipeline_end = datetime.now()
        total_duration = (pipeline_end - pipeline_start).total_seconds()
        
        # Generate report
        self.generate_report(total_duration)
        
        return self.results
    
    def generate_report(self, total_duration: float) -> None:
        """Generate pipeline execution report."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE EXECUTION REPORT")
        logger.info("=" * 70)
        
        for stage_key, result in self.results.items():
            status_emoji = {
                StageStatus.SUCCESS: "✅",
                StageStatus.FAILED: "❌",
                StageStatus.SKIPPED: "⏭️",
                StageStatus.PENDING: "⏸️",
                StageStatus.RUNNING: "⏳",
            }
            
            emoji = status_emoji.get(result.status, "❓")
            duration_str = f"{result.duration_seconds:.1f}s" if result.duration_seconds else "N/A"
            
            logger.info(f"{emoji} Stage {result.stage_num} ({result.stage_name}): {result.status.value} ({duration_str})")
            
            if result.status == StageStatus.FAILED and result.error_message:
                logger.info(f"    Error: {result.error_message}")
        
        logger.info("")
        logger.info(f"Total Duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
        
        # Success/failure summary
        n_success = sum(1 for r in self.results.values() if r.status == StageStatus.SUCCESS)
        n_failed = sum(1 for r in self.results.values() if r.status == StageStatus.FAILED)
        n_skipped = sum(1 for r in self.results.values() if r.status == StageStatus.SKIPPED)
        
        logger.info(f"Results: {n_success} success, {n_failed} failed, {n_skipped} skipped")
        logger.info("=" * 70)
        
        # Save state
        self.save_state()
    
    def save_state(self) -> None:
        """Save pipeline state to JSON."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "results": {k: asdict(v) for k, v in self.results.items()},
        }
        
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Pipeline state saved to {self.state_file}")
    
    def load_state(self) -> bool:
        """
        Load pipeline state from JSON.
        
        Returns:
            True if state loaded successfully
        """
        if not self.state_file.exists():
            return False
        
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            self.results = {}
            for k, v in state["results"].items():
                v["status"] = StageStatus(v["status"])
                self.results[k] = StageResult(**v)
            
            logger.info(f"Loaded pipeline state from {self.state_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return False
    
    def validate_all(self) -> None:
        """Validate all stage artifacts without running."""
        logger.info("=" * 70)
        logger.info("VALIDATING PIPELINE ARTIFACTS")
        logger.info("=" * 70)
        
        for stage_key in ["stage1", "stage2", "stage3_train", "stage3_eval", "stage4"]:
            exists, missing = self.validate_stage_artifacts(stage_key)
            quality_ok, errors = self.validate_stage_output_quality(stage_key) if exists else (False, [])
            
            if exists and quality_ok:
                logger.info(f"✅ {stage_key}: All artifacts exist and valid")
            elif exists and not quality_ok:
                logger.warning(f"⚠️  {stage_key}: Artifacts exist but validation failed:")
                for err in errors:
                    logger.warning(f"    - {err}")
            else:
                logger.error(f"❌ {stage_key}: Missing artifacts:")
                for m in missing:
                    logger.error(f"    - {m}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SentinelFetal Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run full pipeline (Stages 1-4)",
    )
    
    parser.add_argument(
        "--from-stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Start from this stage (default: 1)",
    )
    
    parser.add_argument(
        "--to-stage",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="End at this stage (default: 4)",
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate artifacts without running",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last failed stage",
    )
    
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-run stages even if artifacts exist",
    )
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    if args.validate_only:
        orchestrator.validate_all()
        return 0
    
    if args.resume:
        if orchestrator.load_state():
            # Find first failed stage
            failed_stages = [k for k, v in orchestrator.results.items() if v.status == StageStatus.FAILED]
            if failed_stages:
                # Extract stage number from key
                stage_nums = [orchestrator.results[k].stage_num for k in failed_stages]
                from_stage = min(stage_nums)
                logger.info(f"Resuming from failed Stage {from_stage}")
                args.from_stage = from_stage
            else:
                logger.info("No failed stages found in state. Running from Stage 1.")
        else:
            logger.warning("No previous state found. Running from Stage 1.")
    
    if args.run_all:
        args.from_stage = 1
        args.to_stage = 4
    
    skip_existing = not args.no_skip
    
    results = orchestrator.run_full_pipeline(
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        skip_existing=skip_existing,
    )
    
    # Exit code
    failed = any(r.status == StageStatus.FAILED for r in results.values())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
