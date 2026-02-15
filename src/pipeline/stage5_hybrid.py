# -*- coding: utf-8 -*-
"""
Stage 5 Smart Hybrid Pipeline.

This module integrates all Stage 5 components to process windows and
produce final decision outputs with explainability.

Pipeline Flow:
    1. Load windows from Stage 2
    2. Load AI scores from Stage 3
    3. Load thresholds from Stage 4
    4. For each window:
       a. Calculate rule score (from override.py)
       b. Check Boredom Gate (from boredom_gate.py)
       c. Apply Tiering (from tiering.py)
       d. Update Persistence (from persistence.py)
       e. Record decision + reasons
    5. Save outputs to Parquet

References:
    SentinelFetal Stage 5 Documentation - Smart Hybrid
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import logging
import json

import numpy as np
import pandas as pd

from src.config import load_dynamic_thresholds, DynamicThresholds
from src.analysis.tiering import Tier, TieringDecision, decide_tier
from src.analysis.boredom_gate import should_suppress_alert, BoredGateResult
from src.analysis.persistence import PersistenceManager

logger = logging.getLogger(__name__)


@dataclass
class WindowDecision:
    """
    Complete decision record for a single window.
    
    This captures all inputs, intermediate values, and outputs
    for full traceability and debugging.
    """
    # Identification
    record_id: str
    window_index: int
    window_start: float
    window_end: float
    
    # Quality
    quality_class: str
    
    # Scores
    ai_score: float
    rule_score: float
    
    # Thresholds used
    t_low: float
    t_high: float
    r_min: float
    
    # Rule details
    rule_hits: List[str] = field(default_factory=list)
    is_severe: bool = False
    
    # Boredom Gate
    boredom_suppressed: bool = False
    
    # Tiering
    tier: str = "no_alert"
    tier_decision: bool = False
    
    # Persistence
    persistence_state: str = "pending"
    final_alert: bool = False
    
    # Explainability
    reason_codes: List[str] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        d = asdict(self)
        # Convert lists to JSON strings for Parquet
        d['rule_hits'] = json.dumps(d['rule_hits'])
        d['reason_codes'] = json.dumps(d['reason_codes'])
        return d


class Stage5Pipeline:
    """
    Stage 5 Smart Hybrid Decision Pipeline.
    
    Processes windows with AI scores and rule analysis,
    applying tiering, boredom gate, and persistence to
    produce final alert decisions.
    """
    
    def __init__(
        self,
        thresholds: Optional[DynamicThresholds] = None,
        thresholds_path: str = 'models/thresholds.yaml'
    ):
        """
        Initialize the pipeline.
        
        Args:
            thresholds: Pre-loaded thresholds (optional).
            thresholds_path: Path to thresholds YAML if not provided.
        """
        self.thresholds = thresholds or load_dynamic_thresholds(thresholds_path)
        self.persistence = PersistenceManager(
            K=self.thresholds.K,
            N=self.thresholds.N
        )
        self.decisions: List[WindowDecision] = []
        
        logger.info(
            f"Stage5Pipeline initialized with thresholds: "
            f"t_low={self.thresholds.t_low}, t_high={self.thresholds.t_high}, "
            f"r_min={self.thresholds.r_min}, K={self.thresholds.K}, N={self.thresholds.N}"
        )
    
    def process_window(
        self,
        record_id: str,
        window_index: int,
        window_start: float,
        window_end: float,
        quality_class: str,
        ai_score: float,
        rule_score: float,
        rule_hits: List[str],
        is_severe: bool,
        reason_codes: Optional[List[str]] = None
    ) -> WindowDecision:
        """
        Process a single window through the Stage 5 pipeline.
        
        Args:
            record_id: Recording identifier.
            window_index: Index of the window.
            window_start: Start time in seconds.
            window_end: End time in seconds.
            quality_class: Quality classification ("GOOD", "POOR", "DISCARD").
            ai_score: AI model score (0-1).
            rule_score: Rule-based score (0-1).
            rule_hits: List of triggered rule names.
            is_severe: Whether a critical rule was triggered.
            reason_codes: Optional initial reason codes.
            
        Returns:
            WindowDecision with complete decision record.
        """
        codes = list(reason_codes) if reason_codes else []
        
        # Step 1: Boredom Gate
        boredom_result = should_suppress_alert(
            quality_class=quality_class,
            rule_hits=rule_hits,
            ai_score=ai_score,
            t_low=self.thresholds.t_low
        )
        
        if boredom_result.should_suppress:
            codes.append("BOREDOM_GATE_SUPPRESSED")
            # Skip tiering - no alert
            decision = WindowDecision(
                record_id=record_id,
                window_index=window_index,
                window_start=window_start,
                window_end=window_end,
                quality_class=quality_class,
                ai_score=ai_score,
                rule_score=rule_score,
                t_low=self.thresholds.t_low,
                t_high=self.thresholds.t_high,
                r_min=self.thresholds.r_min,
                rule_hits=rule_hits,
                is_severe=is_severe,
                boredom_suppressed=True,
                tier="no_alert",
                tier_decision=False,
                persistence_state="suppressed",
                final_alert=False,
                reason_codes=codes,
                summary=boredom_result.reason
            )
            self.decisions.append(decision)
            return decision
        
        # Step 2: Tiering
        tiering_result = decide_tier(
            ai_score=ai_score,
            rule_score=rule_score,
            is_severe=is_severe,
            t_low=self.thresholds.t_low,
            t_high=self.thresholds.t_high,
            r_min=self.thresholds.r_min,
            reason_codes=codes
        )
        
        codes = tiering_result.reason_codes
        
        # Step 3: Persistence (K-of-N)
        persistence_input = tiering_result.should_alert
        persistence_state = self.persistence.update(record_id, persistence_input)
        final_alert = persistence_state
        
        if persistence_input and not final_alert:
            codes.append("PERSISTENCE_PENDING")
            pers_state_str = "pending"
        elif final_alert:
            codes.append("PERSISTENCE_CONFIRMED")
            pers_state_str = "confirmed"
        else:
            pers_state_str = "inactive"
        
        decision = WindowDecision(
            record_id=record_id,
            window_index=window_index,
            window_start=window_start,
            window_end=window_end,
            quality_class=quality_class,
            ai_score=ai_score,
            rule_score=rule_score,
            t_low=self.thresholds.t_low,
            t_high=self.thresholds.t_high,
            r_min=self.thresholds.r_min,
            rule_hits=rule_hits,
            is_severe=is_severe,
            boredom_suppressed=False,
            tier=tiering_result.tier.value,
            tier_decision=tiering_result.should_alert,
            persistence_state=pers_state_str,
            final_alert=final_alert,
            reason_codes=codes,
            summary=tiering_result.summary
        )
        
        self.decisions.append(decision)
        return decision
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert all decisions to a DataFrame."""
        if not self.decisions:
            return pd.DataFrame()
        
        records = [d.to_dict() for d in self.decisions]
        return pd.DataFrame(records)
    
    def save_results(
        self,
        output_dir: Path,
        prefix: str = "stage5"
    ) -> Dict[str, Path]:
        """
        Save all results to Parquet and JSON files.
        
        Args:
            output_dir: Directory to save outputs.
            prefix: Filename prefix.
            
        Returns:
            Dict mapping output type to path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Decision outputs
        df = self.get_results_dataframe()
        if not df.empty:
            parquet_path = output_dir / f"{prefix}_decisions.parquet"
            df.to_parquet(parquet_path, index=False)
            paths['decisions'] = parquet_path
            logger.info(f"Saved {len(df)} decisions to {parquet_path}")
        
        # Summary statistics
        stats = self.get_summary_stats()
        stats_path = output_dir / f"{prefix}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        paths['stats'] = stats_path
        
        return paths
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of pipeline run."""
        if not self.decisions:
            return {"total_windows": 0}
        
        total = len(self.decisions)
        alerts = sum(1 for d in self.decisions if d.final_alert)
        suppressed = sum(1 for d in self.decisions if d.boredom_suppressed)
        
        tier_counts = {}
        for d in self.decisions:
            tier_counts[d.tier] = tier_counts.get(d.tier, 0) + 1
        
        return {
            "total_windows": total,
            "final_alerts": alerts,
            "alert_rate": round(alerts / total * 100, 2) if total > 0 else 0,
            "boredom_suppressed": suppressed,
            "tier_distribution": tier_counts,
            "thresholds": {
                "t_low": self.thresholds.t_low,
                "t_high": self.thresholds.t_high,
                "r_min": self.thresholds.r_min,
                "K": self.thresholds.K,
                "N": self.thresholds.N
            }
        }
    
    def reset(self, record_id: Optional[str] = None):
        """
        Reset pipeline state.
        
        Args:
            record_id: If provided, reset only that record's persistence.
                      If None, reset everything.
        """
        if record_id:
            self.persistence.reset(record_id)
        else:
            self.decisions = []
            self.persistence = PersistenceManager(
                K=self.thresholds.K,
                N=self.thresholds.N
            )


def test_pipeline():
    """Simple test of the Stage 5 pipeline."""
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline with default thresholds
    thresholds = DynamicThresholds(
        t_low=0.45,
        t_high=0.72,
        K=2,
        N=3,
        r_min=0.3
    )
    pipeline = Stage5Pipeline(thresholds=thresholds)
    
    # Test windows
    test_cases = [
        # Window 1: Good quality, low AI, no rules → Suppressed
        {"quality_class": "GOOD", "ai_score": 0.2, "rule_score": 0.0, 
         "rule_hits": [], "is_severe": False},
        # Window 2: Poor quality, medium AI, some rules → Tier 2
        {"quality_class": "POOR", "ai_score": 0.55, "rule_score": 0.4, 
         "rule_hits": ["LATE_DECEL"], "is_severe": False},
        # Window 3: High AI → Tier 1
        {"quality_class": "GOOD", "ai_score": 0.85, "rule_score": 0.2, 
         "rule_hits": [], "is_severe": False},
        # Window 4: Severe rule → Tier 3
        {"quality_class": "GOOD", "ai_score": 0.3, "rule_score": 0.8, 
         "rule_hits": ["SINUSOIDAL"], "is_severe": True},
    ]
    
    for i, tc in enumerate(test_cases):
        decision = pipeline.process_window(
            record_id="test_001",
            window_index=i,
            window_start=i * 60.0,
            window_end=(i + 1) * 60.0,
            **tc
        )
        logger.info(
            f"Window {i}: tier={decision.tier}, alert={decision.final_alert}, "
            f"suppressed={decision.boredom_suppressed}"
        )
    
    # Get stats
    stats = pipeline.get_summary_stats()
    logger.info(f"Stats: {stats}")
    
    assert stats['total_windows'] == 4
    assert stats['boredom_suppressed'] == 1
    logger.info("✓ Stage 5 Pipeline tests passed")


if __name__ == '__main__':
    test_pipeline()
