# Hybrid Validation V5

## Config (Optuna best)
{
  "aggregation_mode": "persistence",
  "feature_spec": "v5_super_features",
  "objective": "F2",
  "study_best_value": 0.6221889055472264,
  "threshold": 0.3844,
  "timestamp": "2026-01-25T19:40:11.331427+00:00",
  "weights": {
    "random_forest": 0.1041,
    "sgd_classifier": 0.1463,
    "xgboost": 0.7495
  }
}

## Metrics
| System | Threshold | Weights (xgb/rf/sgd) | Recall | Precision | F2 | % Alerted |
| --- | --- | --- | --- | --- | --- | --- |
| AI V5 (Optuna) | 0.384 | 0.750/0.104/0.146 | 0.790 | 0.336 | 0.622 | 0.515 |
| Hybrid (AI OR Rule) | 0.384 | 0.750/0.104/0.146 | 0.876 | 0.333 | 0.661 | 0.575 |
