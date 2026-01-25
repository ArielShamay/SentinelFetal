"""
Simple Trace Execution Test - Validates XGBoost Integration
===========================================================
Tests the real-data trained XGBoost classifier.
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

def test_classifier():
    """Test XGBoost classifier loading and basic functionality."""
    print("Testing XGBoost Classifier...")
    
    try:
        from src.analysis.xgboost_classifier import get_classifier
        
        classifier = get_classifier()
        
        if not classifier.is_loaded:
            print("[WARNING] Model not trained. Run: python scripts/train_xgboost_realdata.py")
            return False
        
        # Generate test data (10 min at 4 Hz = 2400 samples)
        fhr_normal = np.random.normal(140, 10, 2400)
        uc_normal = np.random.normal(20, 5, 2400)
        
        result = classifier.classify(fhr_normal, uc_normal)
        
        print("[OK] Classification successful")
        print(f"   Category: {result['category']} - {result['category_name']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Probabilities: Cat1={result['probabilities'].get(1, 0):.2f}, Cat2={result['probabilities'].get(2, 0):.2f}, Cat3={result['probabilities'].get(3, 0):.2f}")
        print(f"   Model type: {'Real Data' if classifier.is_realdata_model else 'Synthetic'}")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_classifier()
    sys.exit(0 if success else 1)
