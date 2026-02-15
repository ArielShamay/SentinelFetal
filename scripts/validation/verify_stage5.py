#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5 Standalone Verification Script.

Tests all Stage 5 components without heavy dependencies.

Usage:
    python3 scripts/validation/verify_stage5.py
"""

import sys
from pathlib import Path
import logging
import importlib.util

# Setup
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_module(name: str, path: Path):
    """Load a module directly from file."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_tiering():
    """Test tiering module."""
    logger.info("Testing Tiering Module...")
    
    tiering = load_module("tiering", project_root / "src/analysis/tiering.py")
    
    # Test Tier 3 (severe)
    result = tiering.decide_tier(0.2, 0.5, is_severe=True, t_low=0.45, t_high=0.72)
    assert result.tier == tiering.Tier.TIER_3
    assert result.should_alert == True
    logger.info("  ✓ Tier 3 (severe rule) works")
    
    # Test Tier 1 (high AI)
    result = tiering.decide_tier(0.85, 0.1, is_severe=False, t_low=0.45, t_high=0.72)
    assert result.tier == tiering.Tier.TIER_1
    assert result.should_alert == True
    logger.info("  ✓ Tier 1 (high AI) works")
    
    # Test Tier 2 (AI + rules)
    result = tiering.decide_tier(0.55, 0.4, is_severe=False, t_low=0.45, t_high=0.72)
    assert result.tier == tiering.Tier.TIER_2
    assert result.should_alert == True
    logger.info("  ✓ Tier 2 (AI + rules) works")
    
    # Test No Alert
    result = tiering.decide_tier(0.3, 0.1, is_severe=False, t_low=0.45, t_high=0.72)
    assert result.tier == tiering.Tier.NO_ALERT
    assert result.should_alert == False
    logger.info("  ✓ No Alert (default) works")
    
    return True


def test_boredom_gate():
    """Test boredom gate module."""
    logger.info("Testing Boredom Gate Module...")
    
    boredom = load_module("boredom_gate", project_root / "src/analysis/boredom_gate.py")
    
    # Should suppress: good quality, no hits, low AI
    result = boredom.should_suppress_alert("GOOD", [], 0.2, t_low=0.45)
    assert result.should_suppress == True
    logger.info("  ✓ Suppresses stable windows")
    
    # Should NOT suppress: poor quality
    result = boredom.should_suppress_alert("POOR", [], 0.2, t_low=0.45)
    assert result.should_suppress == False
    logger.info("  ✓ Does not suppress poor quality")
    
    # Should NOT suppress: has rule hits
    result = boredom.should_suppress_alert("GOOD", ["LATE_DECEL"], 0.2, t_low=0.45)
    assert result.should_suppress == False
    logger.info("  ✓ Does not suppress with rule hits")
    
    # Should NOT suppress: high AI score
    result = boredom.should_suppress_alert("GOOD", [], 0.6, t_low=0.45)
    assert result.should_suppress == False
    logger.info("  ✓ Does not suppress high AI")
    
    return True


def test_persistence():
    """Test persistence module."""
    logger.info("Testing Persistence Module (K=2, N=3)...")
    
    persistence = load_module("persistence", project_root / "src/analysis/persistence.py")
    
    manager = persistence.PersistenceManager(K=2, N=3)
    
    # Gradual buildup
    assert manager.update("test", False) == False  # 0/3
    assert manager.update("test", True) == False   # 1/3
    assert manager.update("test", True) == True    # 2/3 ✓
    logger.info("  ✓ K-of-N triggers correctly")
    
    # Persistence after trigger
    assert manager.update("test", False) == True   # 2/3 still ✓
    assert manager.update("test", False) == False  # 1/3 drops
    logger.info("  ✓ Persistence drops correctly")
    
    return True


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("STAGE 5 VERIFICATION")
    print("="*60 + "\n")
    
    tests = [
        ("Tiering", test_tiering),
        ("Boredom Gate", test_boredom_gate),
        ("Persistence", test_persistence),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            logger.error(f"✗ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            logger.error(f"✗ {name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if failed > 0:
        print(f"⚠  {failed} tests failed")
        print("="*60 + "\n")
        sys.exit(1)
    else:
        print("✓ All Stage 5 verification tests passed!")
        print("="*60 + "\n")
        sys.exit(0)


if __name__ == '__main__':
    main()
