"""
Centralized configuration for SentinelFetal.

This module contains all hardcoded constants used throughout the application.
Centralizing configuration enables easy modification and ensures consistency.

Usage:
    from src.config import CTG, THRESHOLDS, COLORS
    
    window_samples = CTG.WINDOW_SAMPLES
    pathological_ph = THRESHOLDS.PH_PATHOLOGICAL
"""

from dataclasses import dataclass
from typing import Dict, Final


# =============================================================================
# CTG Signal Configuration
# =============================================================================

@dataclass(frozen=True)
class CTGConfig:
    """CTG signal processing constants."""
    
    # Sampling
    SAMPLING_RATE: float = 4.0  # Hz (samples per second)
    
    # MOMENT Encoder Windows
    MOMENT_WINDOW_MINUTES: float = 10.0  # Window size for MOMENT embeddings
    MOMENT_STEP_MINUTES: float = 1.0     # Step size for sliding window
    MOMENT_WINDOW_SAMPLES: int = 2400    # 10 min * 60 sec * 4 Hz
    MOMENT_STEP_SAMPLES: int = 240       # 1 min * 60 sec * 4 Hz
    
    # Feature Dimensions
    MOMENT_EMBEDDING_DIM: int = 1024     # MOMENT model output dimension
    RULE_FEATURES_DIM: int = 11          # Number of rule-based features
    TOTAL_FEATURES_DIM: int = 1035       # 1024 + 11
    
    # Baseline Calculation
    BASELINE_WINDOW_MINUTES: float = 2.0
    BASELINE_STEP_SECONDS: float = 10.0
    BASELINE_WINDOW_SAMPLES: int = 480   # 2 min * 60 sec * 4 Hz
    
    # Variability Calculation
    VARIABILITY_WINDOW_SECONDS: float = 60.0
    VARIABILITY_OVERLAP_RATIO: float = 0.5
    VARIABILITY_MIN_VALID_RATIO: float = 0.5
    
    # Deceleration Detection
    DECEL_DROP_THRESHOLD_BPM: float = 15.0
    DECEL_MIN_DURATION_SECONDS: float = 15.0
    DECEL_MAX_DURATION_SECONDS: float = 180.0
    
    # Sinusoidal Pattern Detection
    SINUSOIDAL_MIN_DURATION_MINUTES: float = 20.0
    SINUSOIDAL_FREQUENCY_MIN_CPM: float = 2.0  # Cycles per minute
    SINUSOIDAL_FREQUENCY_MAX_CPM: float = 6.0
    SINUSOIDAL_AMPLITUDE_MIN_BPM: float = 5.0
    SINUSOIDAL_AMPLITUDE_MAX_BPM: float = 15.0
    
    # Tachysystole Detection
    TACHYSYSTOLE_THRESHOLD: int = 5  # Contractions per 10 minutes
    TACHYSYSTOLE_WINDOW_MINUTES: float = 30.0
    TACHYSYSTOLE_MIN_DISTANCE_SECONDS: float = 45.0


CTG: Final[CTGConfig] = CTGConfig()


# =============================================================================
# Clinical Thresholds (Based on Israeli Position Paper)
# =============================================================================

@dataclass(frozen=True)
class ClinicalThresholds:
    """Clinical threshold values for classification."""
    
    # pH Thresholds for outcome labeling
    PH_PATHOLOGICAL: float = 7.15   # pH < 7.15 → Category 3
    PH_INTERMEDIATE: float = 7.20   # 7.15 ≤ pH < 7.20 → Category 2
    # pH ≥ 7.20 → Category 1 (Normal)
    
    # FHR Baseline Ranges (bpm)
    BASELINE_NORMAL_MIN: int = 110
    BASELINE_NORMAL_MAX: int = 160
    BASELINE_TACHYCARDIA: int = 160
    BASELINE_BRADYCARDIA: int = 110
    BASELINE_SEVERE_BRADYCARDIA: int = 100
    
    # Variability Categories (bpm) - Per Israeli Position Paper
    VARIABILITY_ABSENT_MAX: float = 2.0       # 0-2 bpm = Absent
    VARIABILITY_MINIMAL_MAX: float = 5.0      # 3-5 bpm = Minimal
    VARIABILITY_MODERATE_MIN: float = 6.0
    VARIABILITY_MODERATE_MAX: float = 25.0    # 6-25 bpm = Moderate (Normal)
    VARIABILITY_MARKED_MIN: float = 25.0      # > 25 bpm = Marked


THRESHOLDS: Final[ClinicalThresholds] = ClinicalThresholds()


# =============================================================================
# UI Colors
# =============================================================================

@dataclass(frozen=True)
class UIColors:
    """Color scheme for UI components."""
    
    # CTG Plot Colors
    FHR: str = '#1E90FF'         # Dodger Blue
    UC: str = '#FF8C00'          # Dark Orange
    DECEL_HIGHLIGHT: str = 'rgba(255, 0, 0, 0.15)'  # Light red for deceleration regions
    
    # Background & Grid
    GRID: str = '#E5E5E5'
    BACKGROUND: str = '#FAFAFA'
    
    # Category Colors
    CATEGORY_1_NORMAL: str = '#28a745'       # Green
    CATEGORY_2_INTERMEDIATE: str = '#fd7e14' # Orange
    CATEGORY_3_PATHOLOGICAL: str = '#dc3545' # Red
    
    @property
    def category_colors(self) -> Dict[int, str]:
        """Get color mapping for categories."""
        return {
            1: self.CATEGORY_1_NORMAL,
            2: self.CATEGORY_2_INTERMEDIATE,
            3: self.CATEGORY_3_PATHOLOGICAL
        }


COLORS: Final[UIColors] = UIColors()


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """XGBoost and ML model configuration."""
    
    # XGBoost Parameters
    N_ESTIMATORS: int = 100
    MAX_DEPTH: int = 6
    LEARNING_RATE: float = 0.1
    MIN_CHILD_WEIGHT: int = 1
    SUBSAMPLE: float = 0.8
    COLSAMPLE_BYTREE: float = 0.8
    
    # Cross-Validation
    CV_FOLDS: int = 3
    RANDOM_STATE: int = 42
    
    # Classification
    NUM_CLASSES: int = 3
    CLASS_NAMES: tuple = ('Normal (Cat 1)', 'Intermediate (Cat 2)', 'Pathological (Cat 3)')
    
    # Model Paths
    DEFAULT_MODEL_PATH: str = 'models/xgb_demo.json'


MODEL: Final[ModelConfig] = ModelConfig()


# =============================================================================
# Data Paths
# =============================================================================

@dataclass(frozen=True)
class DataPaths:
    """Default data paths."""
    
    CTU_UHB_DATA_DIR: str = 'data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0'
    PROCESSED_DATA_DIR: str = 'data/processed'
    X_PATH: str = 'data/processed/X.npy'
    Y_PATH: str = 'data/processed/y.npy'
    MODELS_DIR: str = 'models'
    DEFAULT_MODEL_PATH: str = 'models/xgb_demo.json'


PATHS: Final[DataPaths] = DataPaths()


# =============================================================================
# Hebrew XAI Strings (from Section 8 of Specification)
# =============================================================================

@dataclass(frozen=True)
class HebrewStrings:
    """Hebrew explanatory strings for alerts."""
    
    # Category Headlines
    HEADLINE_CATEGORY_1: str = "התראה ירוקה - קטגוריה 1 (תקין)"
    HEADLINE_CATEGORY_2: str = "התראה צהובה - קטגוריה 2 (לא מוגדר)"
    HEADLINE_CATEGORY_3: str = "התראה אדומה - קטגוריה 3 (פתולוגי)"
    
    # Recommendations
    REC_NORMAL: str = "מעקב שגרתי"
    REC_INTERMEDIATE: str = "יש להעריך את המצב מחדש, לתקן גורמים הפיכים אם קיימים ולשקול טיפולים נוספים"
    REC_PATHOLOGICAL: str = "הערכה מיידית של הסיבות האפשריות"
    
    # Finding Templates
    FINDING_BASELINE_NORMAL: str = "קו בסיס תקין ({value} פעימות/דקה)"
    FINDING_BASELINE_TACHY: str = "טכיקרדיה ({value} פעימות/דקה)"
    FINDING_BASELINE_BRADY: str = "ברדיקרדיה ({value} פעימות/דקה)"
    
    FINDING_VARIABILITY_NORMAL: str = "שונות תקינה ({value:.1f} פעימות/דקה)"
    FINDING_VARIABILITY_ABSENT: str = "שונות חסרה ({value:.1f} פעימות/דקה) - ממצא חמור"
    FINDING_VARIABILITY_MINIMAL: str = "שונות מופחתת ({value:.1f} פעימות/דקה)"
    FINDING_VARIABILITY_MARKED: str = "שונות מוגברת ({value:.1f} פעימות/דקה)"
    
    FINDING_DECELS_NONE: str = "לא זוהו האטות משמעותיות"
    FINDING_DECELS_EARLY: str = "האטות מוקדמות ({count})"
    FINDING_DECELS_VARIABLE: str = "האטות משתנות ({count})"
    FINDING_DECELS_LATE: str = "האטות מאוחרות ({count}) - ממצא חמור"
    FINDING_DECELS_PROLONGED: str = "האטות ממושכות ({count})"
    
    FINDING_TACHYSYSTOLE: str = "טכיסיסטולה ({rate:.1f} התכווצויות ב-10 דקות)"
    FINDING_NO_TACHYSYSTOLE: str = "תדירות התכווצויות תקינה"
    
    FINDING_SINUSOIDAL: str = "דפוס סינוסואידלי - ממצא חמור ביותר"
    FINDING_NO_SINUSOIDAL: str = "לא זוהה דפוס סינוסואידלי"


HEBREW: Final[HebrewStrings] = HebrewStrings()


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    'CTG',
    'THRESHOLDS', 
    'COLORS',
    'MODEL',
    'PATHS',
    'HEBREW',
    'CTGConfig',
    'ClinicalThresholds',
    'UIColors',
    'ModelConfig',
    'DataPaths',
    'HebrewStrings',
]
