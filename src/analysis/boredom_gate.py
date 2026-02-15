# -*- coding: utf-8 -*-
"""
Boredom Gate for Stage 5 Smart Hybrid.

This module implements alert suppression logic to prevent unnecessary
alerts when the signal quality is good and no significant findings exist.

The Boredom Gate helps reduce alert fatigue by suppressing alerts in
"boring" (stable, high-quality) windows where the AI score is low and
no rules were triggered.

References:
    SentinelFetal Stage 5 Documentation - Section 5A.4 Boredom Gate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoredGateResult:
    """
    Result of boredom gate evaluation.
    
    Attributes:
        should_suppress: Whether the alert should be suppressed.
        reason: Human-readable explanation.
    """
    should_suppress: bool
    reason: str


def should_suppress_alert(
    quality_class: str,
    rule_hits: List[str],
    ai_score: float,
    t_low: float
) -> BoredGateResult:
    """
    Determine if an alert should be suppressed (Boredom Gate).
    
    The gate suppresses alerts when:
    1. Signal quality is GOOD
    2. No rules were triggered (empty rule_hits)
    3. AI score is below the low threshold
    
    This prevents unnecessary alerts in stable, high-quality recordings
    where nothing concerning is happening.
    
    Args:
        quality_class: Window quality classification ("GOOD", "POOR", "DISCARD").
        rule_hits: List of triggered rule names.
        ai_score: AI model score (0-1 range).
        t_low: Lower AI threshold.
        
    Returns:
        BoredGateResult indicating if alert should be suppressed.
    """
    # Normalize quality class
    quality_upper = quality_class.upper() if quality_class else ""
    
    # Check suppression conditions
    is_good_quality = quality_upper == "GOOD"
    no_rule_hits = len(rule_hits) == 0
    low_ai_score = ai_score < t_low
    
    if is_good_quality and no_rule_hits and low_ai_score:
        logger.debug(
            f"BOREDOM GATE: Suppressing alert (quality={quality_class}, "
            f"rules={len(rule_hits)}, ai={ai_score:.3f} < {t_low})"
        )
        return BoredGateResult(
            should_suppress=True,
            reason="חלון יציב - איכות טובה, ללא ממצאים, AI נמוך"
        )
    
    # No suppression - explain why
    reasons = []
    if not is_good_quality:
        reasons.append(f"איכות לא טובה ({quality_class})")
    if not no_rule_hits:
        reasons.append(f"זוהו {len(rule_hits)} ממצאים")
    if not low_ai_score:
        reasons.append(f"AI Score גבוה ({ai_score:.2%})")
    
    return BoredGateResult(
        should_suppress=False,
        reason=" | ".join(reasons) if reasons else "עובר לבדיקת Tiering"
    )


def test_boredom_gate():
    """Simple test function for boredom gate logic."""
    # Should suppress: good quality, no hits, low AI
    result = should_suppress_alert("GOOD", [], 0.2, t_low=0.45)
    assert result.should_suppress == True
    
    # Should NOT suppress: poor quality
    result = should_suppress_alert("POOR", [], 0.2, t_low=0.45)
    assert result.should_suppress == False
    
    # Should NOT suppress: has rule hits
    result = should_suppress_alert("GOOD", ["LATE_DECEL"], 0.2, t_low=0.45)
    assert result.should_suppress == False
    
    # Should NOT suppress: high AI score
    result = should_suppress_alert("GOOD", [], 0.6, t_low=0.45)
    assert result.should_suppress == False
    
    logger.info("✓ Boredom Gate tests passed")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_boredom_gate()
