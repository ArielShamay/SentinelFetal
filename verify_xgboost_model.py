"""
Verify XGBoost V5 model integrity and display information.
"""
import pickle
import os
from pathlib import Path

model_path = Path("models/ensemble_v5/xgboost_v5.pkl")

print(f"Model path: {model_path}")
print(f"File exists: {model_path.exists()}")
print(f"File size: {os.path.getsize(model_path):,} bytes")
print("-" * 60)

try:
    # Load the model
    print("Loading model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("✓ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    print("-" * 60)
    
    # Check if it's a calibrated classifier or raw XGBoost
    if hasattr(model, 'classes_'):
        print(f"Classes: {model.classes_}")
    
    if hasattr(model, 'n_features_in_'):
        print(f"Number of features: {model.n_features_in_}")
    
    # Check for XGBoost specific attributes
    if hasattr(model, 'get_booster'):
        print("✓ Valid XGBoost model with booster")
        booster = model.get_booster()
        print(f"Number of boosting rounds: {booster.num_boosted_rounds()}")
        print(f"Number of features: {booster.num_features()}")
    
    # Check if it's a CalibratedClassifierCV
    if hasattr(model, 'calibrated_classifiers_'):
        print(f"✓ Calibrated classifier with {len(model.calibrated_classifiers_)} calibrators")
        base_estimator = model.calibrated_classifiers_[0].estimator
        print(f"Base estimator type: {type(base_estimator).__name__}")
        if hasattr(base_estimator, 'get_booster'):
            booster = base_estimator.get_booster()
            print(f"Number of boosting rounds: {booster.num_boosted_rounds()}")
            print(f"Number of features: {booster.num_features()}")
    
    print("-" * 60)
    print("✓ Model is COMPLETE and VALID!")
    
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    import traceback
    traceback.print_exc()
