#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5 Build Script.

Runs the Stage 5 Smart Hybrid pipeline on processed data.

Usage:
    python scripts/pipeline/stage5_build.py \\
        --windows data/processed/windows.parquet \\
        --ai-scores data/processed/ai_scores.parquet \\
        --output-dir data/processed/stage5
        
Or with demo data:
    python scripts/pipeline/stage5_build.py --demo
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import load_dynamic_thresholds, DynamicThresholds
from src.pipeline.stage5_hybrid import Stage5Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo():
    """Run demo with synthetic data."""
    logger.info("Running Stage 5 demo with synthetic data...")
    
    thresholds = DynamicThresholds(
        t_low=0.45,
        t_high=0.72,
        K=2,
        N=3,
        r_min=0.3
    )
    
    pipeline = Stage5Pipeline(thresholds=thresholds)
    
    # Simulate 10 windows
    import random
    random.seed(42)
    
    for i in range(10):
        quality = random.choice(["GOOD", "GOOD", "GOOD", "POOR"])
        ai_score = random.uniform(0.1, 0.9)
        rule_score = random.uniform(0.0, 0.6)
        rule_hits = []
        is_severe = False
        
        if rule_score > 0.3:
            rule_hits.append("LATE_DECEL")
        if rule_score > 0.5:
            rule_hits.append("ABSENT_VARIABILITY")
        if ai_score > 0.8 and random.random() > 0.8:
            is_severe = True
            rule_hits.append("SINUSOIDAL")
        
        decision = pipeline.process_window(
            record_id="demo_001",
            window_index=i,
            window_start=i * 60.0,
            window_end=(i + 1) * 60.0,
            quality_class=quality,
            ai_score=ai_score,
            rule_score=rule_score,
            rule_hits=rule_hits,
            is_severe=is_severe
        )
        
        status = "🔴 ALERT" if decision.final_alert else (
            "🟡 PENDING" if decision.tier_decision else "✅ OK"
        )
        logger.info(
            f"Window {i:2d}: {status} | "
            f"AI={decision.ai_score:.2f} | "
            f"Rules={decision.rule_score:.2f} | "
            f"Tier={decision.tier}"
        )
    
    # Summary
    stats = pipeline.get_summary_stats()
    print("\n" + "="*50)
    print("STAGE 5 DEMO RESULTS")
    print("="*50)
    print(f"Total Windows:      {stats['total_windows']}")
    print(f"Final Alerts:       {stats['final_alerts']} ({stats['alert_rate']}%)")
    print(f"Boredom Suppressed: {stats['boredom_suppressed']}")
    print(f"Tier Distribution:  {stats['tier_distribution']}")
    print("="*50)
    
    return pipeline


def run_pipeline(
    windows_path: Path,
    ai_scores_path: Path,
    output_dir: Path,
    thresholds_path: Optional[Path] = None
):
    """
    Run full Stage 5 pipeline on data files.
    
    Args:
        windows_path: Path to windows Parquet file.
        ai_scores_path: Path to AI scores Parquet file.
        output_dir: Directory to save outputs.
        thresholds_path: Optional path to thresholds YAML.
    """
    import pandas as pd
    
    logger.info(f"Loading windows from: {windows_path}")
    logger.info(f"Loading AI scores from: {ai_scores_path}")
    
    # Load data
    windows_df = pd.read_parquet(windows_path)
    scores_df = pd.read_parquet(ai_scores_path)
    
    # Merge on common key
    # Assuming both have record_id and window_index
    df = windows_df.merge(
        scores_df,
        on=['record_id', 'window_index'],
        how='inner'
    )
    
    logger.info(f"Merged {len(df)} windows")
    
    # Load thresholds
    if thresholds_path:
        thresholds = load_dynamic_thresholds(str(thresholds_path))
    else:
        thresholds = load_dynamic_thresholds()
    
    # Create pipeline
    pipeline = Stage5Pipeline(thresholds=thresholds)
    
    # Process each window
    for _, row in df.iterrows():
        # Note: This assumes rule_score and rule_hits are already computed
        # In a full implementation, we'd calculate them here using override.calculate_rule_score
        pipeline.process_window(
            record_id=row['record_id'],
            window_index=row['window_index'],
            window_start=row.get('window_start', 0),
            window_end=row.get('window_end', 0),
            quality_class=row.get('quality_class', 'GOOD'),
            ai_score=row.get('ai_score', 0.5),
            rule_score=row.get('rule_score', 0.0),
            rule_hits=row.get('rule_hits', []),
            is_severe=row.get('is_severe', False)
        )
    
    # Save results
    paths = pipeline.save_results(output_dir)
    
    # Print summary
    stats = pipeline.get_summary_stats()
    logger.info(f"Pipeline complete. Stats: {stats}")
    
    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 5 Smart Hybrid Pipeline"
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run demo with synthetic data'
    )
    parser.add_argument(
        '--windows', type=Path,
        help='Path to windows Parquet file'
    )
    parser.add_argument(
        '--ai-scores', type=Path,
        help='Path to AI scores Parquet file'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('data/processed/stage5'),
        help='Output directory'
    )
    parser.add_argument(
        '--thresholds', type=Path,
        help='Path to thresholds YAML file'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.windows and args.ai_scores:
        run_pipeline(
            windows_path=args.windows,
            ai_scores_path=args.ai_scores,
            output_dir=args.output_dir,
            thresholds_path=args.thresholds
        )
    else:
        parser.print_help()
        print("\nError: Provide --demo or both --windows and --ai-scores")
        sys.exit(1)


if __name__ == '__main__':
    from typing import Optional
    main()
