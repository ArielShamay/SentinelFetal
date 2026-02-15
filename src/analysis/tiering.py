# -*- coding: utf-8 -*-
"""
Tiering Decision Logic for Stage 5 Smart Hybrid.

This module implements the three-tier decision system that combines
AI scores with rule-based scores to produce final alert decisions.

Tiering Logic:
    Tier-1: ai_score >= t_high → Alert (High AI Confidence)
    Tier-2: t_low <= ai_score < t_high AND rule_score >= r_min → Alert (AI + Rules)
    Tier-3: is_severe == True → Alert (Critical Rule - Override)
    Default: No Alert

References:
    SentinelFetal Stage 5 Documentation - Smart Hybrid
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class Tier(Enum):
    """Decision tier classifications."""
    
    NO_ALERT = "no_alert"      # Default - no concerns
    TIER_1 = "tier_1"          # High AI confidence alone
    TIER_2 = "tier_2"          # AI suspicion + Rule confirmation
    TIER_3 = "tier_3"          # Critical rule override


@dataclass
class TieringDecision:
    """
    Result of tiering decision.
    
    Attributes:
        tier: The tier classification.
        should_alert: Whether an alert should be raised.
        ai_score: The input AI score.
        rule_score: The input rule score.
        is_severe: Whether a severe rule was triggered.
        reason_codes: List of reason codes explaining the decision.
        summary: Human-readable summary in Hebrew.
    """
    tier: Tier
    should_alert: bool
    ai_score: float
    rule_score: float
    is_severe: bool
    reason_codes: List[str] = field(default_factory=list)
    summary: str = ""
    
    @property
    def tier_name(self) -> str:
        """Get Hebrew tier name."""
        names = {
            Tier.NO_ALERT: "ללא התראה",
            Tier.TIER_1: "Tier 1 - ביטחון AI גבוה",
            Tier.TIER_2: "Tier 2 - AI + חוקים",
            Tier.TIER_3: "Tier 3 - חוק קריטי"
        }
        return names.get(self.tier, "לא ידוע")


def decide_tier(
    ai_score: float,
    rule_score: float,
    is_severe: bool,
    t_low: float,
    t_high: float,
    r_min: float = 0.3,
    reason_codes: Optional[List[str]] = None
) -> TieringDecision:
    """
    Make tiering decision based on AI and rule scores.
    
    Decision Logic (in priority order):
    1. Tier-3: is_severe == True → Alert (override everything)
    2. Tier-1: ai_score >= t_high → Alert (high confidence)
    3. Tier-2: t_low <= ai_score < t_high AND rule_score >= r_min → Alert
    4. Default: No Alert
    
    Args:
        ai_score: AI model score (0-1 range).
        rule_score: Rule-based score (0-1 range).
        is_severe: Whether a critical rule was triggered.
        t_low: Lower AI threshold.
        t_high: Upper AI threshold.
        r_min: Minimum rule score for Tier-2 (default: 0.3).
        reason_codes: Optional list of reason codes from rules.
        
    Returns:
        TieringDecision with tier, alert status, and explanation.
    """
    codes = list(reason_codes) if reason_codes else []
    
    # --- Tier 3: Critical Rule Override ---
    if is_severe:
        logger.info(f"TIER 3: Severe rule triggered (is_severe=True)")
        codes.append("TIER_3_SEVERE_RULE")
        
        return TieringDecision(
            tier=Tier.TIER_3,
            should_alert=True,
            ai_score=ai_score,
            rule_score=rule_score,
            is_severe=is_severe,
            reason_codes=codes,
            summary="זוהה ממצא קליני קריטי - התראה ללא תלות ב-AI"
        )
    
    # --- Tier 1: High AI Confidence ---
    if ai_score >= t_high:
        logger.info(f"TIER 1: High AI score ({ai_score:.3f} >= {t_high})")
        codes.append("TIER_1_HIGH_AI_CONFIDENCE")
        
        return TieringDecision(
            tier=Tier.TIER_1,
            should_alert=True,
            ai_score=ai_score,
            rule_score=rule_score,
            is_severe=is_severe,
            reason_codes=codes,
            summary=f"AI ברמת ביטחון גבוהה ({ai_score:.2%})"
        )
    
    # --- Tier 2: AI Suspicion + Rule Confirmation ---
    if t_low <= ai_score < t_high and rule_score >= r_min:
        logger.info(
            f"TIER 2: AI suspicion ({ai_score:.3f}) + Rule confirmation ({rule_score:.3f})"
        )
        codes.append("TIER_2_AI_AND_RULES")
        
        return TieringDecision(
            tier=Tier.TIER_2,
            should_alert=True,
            ai_score=ai_score,
            rule_score=rule_score,
            is_severe=is_severe,
            reason_codes=codes,
            summary=f"שילוב AI ({ai_score:.2%}) וחוקים ({rule_score:.2%}) מעל סף"
        )
    
    # --- Default: No Alert ---
    logger.debug(f"NO ALERT: ai={ai_score:.3f}, rule={rule_score:.3f}")
    codes.append("NO_ALERT_DEFAULT")
    
    return TieringDecision(
        tier=Tier.NO_ALERT,
        should_alert=False,
        ai_score=ai_score,
        rule_score=rule_score,
        is_severe=is_severe,
        reason_codes=codes,
        summary="לא זוהו ממצאים חריגים"
    )


def test_tiering():
    """Simple test function for tiering logic."""
    # Test Tier 3 (severe rule)
    result = decide_tier(0.2, 0.5, is_severe=True, t_low=0.45, t_high=0.72)
    assert result.tier == Tier.TIER_3
    assert result.should_alert == True
    
    # Test Tier 1 (high AI)
    result = decide_tier(0.85, 0.1, is_severe=False, t_low=0.45, t_high=0.72)
    assert result.tier == Tier.TIER_1
    assert result.should_alert == True
    
    # Test Tier 2 (AI + rules)
    result = decide_tier(0.55, 0.4, is_severe=False, t_low=0.45, t_high=0.72)
    assert result.tier == Tier.TIER_2
    assert result.should_alert == True
    
    # Test No Alert
    result = decide_tier(0.3, 0.1, is_severe=False, t_low=0.45, t_high=0.72)
    assert result.tier == Tier.NO_ALERT
    assert result.should_alert == False
    
    logger.info("✓ Tiering tests passed")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_tiering()
