# SentinelFetal – Updated Project Documentation

**Version:** Gen3.5 (Phase 5 Complete)
**Last Updated:** January 2026
**Author:** Ariel Shamay
**Total Lines of Code:** ~8,500 lines (6,130 source + 2,387 tests)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Change Log](#2-change-log)
3. [System Architecture](#3-system-architecture)
4. [Directory Structure](#4-directory-structure)
5. [Module Documentation](#5-module-documentation)
   - [5.1 Data Module](#51-data-module)
   - [5.2 Rules Module](#52-rules-module)
   - [5.3 Models Module](#53-models-module)
   - [5.4 Analysis Module (NEW)](#54-analysis-module-new)
   - [5.5 Training Module (NEW)](#55-training-module-new)
   - [5.6 UI Module (NEW)](#56-ui-module-new)
   - [5.7 Visualization Module](#57-visualization-module)
6. [Data Flow & Pipeline](#6-data-flow--pipeline)
7. [Clinical Rules Reference](#7-clinical-rules-reference)
8. [Testing Documentation](#8-testing-documentation)
9. [Configuration & Dependencies](#9-configuration--dependencies)
10. [Development Status](#10-development-status)

---

## 1. Project Overview

### 1.1 Purpose

**SentinelFetal Gen3.5** is a hybrid AI system designed for **real-time fetal distress detection** during labor. The system analyzes Cardiotocography (CTG) signals - specifically Fetal Heart Rate (FHR) and Uterine Contractions (UC) - to classify fetal status into three medical categories.

### 1.2 Core Capabilities

| Capability | Description |
|------------|-------------|
| **CTG Analysis** | Processes 4Hz CTG recordings from the CTU-UHB database |
| **Signal Preprocessing** | Gap filling (10-second rule), spike detection, normalization |
| **Rule-Based Analysis** | Clinical rules from Israeli Position Paper on CTG Interpretation |
| **AI Feature Extraction** | MOMENT foundation model for 1024-dim embeddings |
| **XGBoost Classification** | Category 1/2/3 classification with Stratified K-Fold CV |
| **Medical Override** | Safety net logic that can override ML predictions |
| **Alert Generation** | Hebrew explanatory alerts with XAI strings |
| **Dashboard UI** | Streamlit-based real-time monitoring interface |

### 1.3 Classification Categories

| Category | Status | Color | Action Required |
|----------|--------|-------|-----------------|
| **Category 1** | Normal (תקין) | Green 🟢 | Routine monitoring |
| **Category 2** | Intermediate (ביניים) | Orange 🟠 | Close monitoring, possible intervention |
| **Category 3** | Pathological (פתולוגי) | Red 🔴 | Immediate intervention required |

### 1.4 Technology Stack

- **Language:** Python 3.9+
- **AI Model:** MOMENT (AutonLab/MOMENT-1-large) - 385M parameters
- **Classifier:** XGBoost with Stratified K-Fold Cross-Validation
- **Dashboard:** Streamlit with Plotly visualizations
- **Dataset:** CTU-UHB Intrapartum Cardiotocography Database (552 recordings)
- **Clinical Basis:** Israeli Position Paper on CTG Interpretation

---

## 2. Change Log

### Summary of Changes Since Previous Documentation

The project has significantly expanded from Phase 3 (MOMENT Integration) to Phase 5 (Alert Engine & Dashboard). All 98 tests are passing.

### New Files Added

| File | Lines | Description |
|------|-------|-------------|
| `src/analysis/__init__.py` | 9 | Analysis module package init |
| `src/analysis/alerts.py` | 331 | Alert Engine with Hebrew XAI strings |
| `src/analysis/override.py` | 310 | Medical Override safety logic |
| `src/models/classifier.py` | 457 | XGBoost classifier wrapper |
| `src/training/__init__.py` | 10 | Training module package init |
| `src/training/prepare_data.py` | 344 | Dataset preparation pipeline |
| `src/training/train_demo.py` | 187 | Demo model training script |
| `src/ui/__init__.py` | 9 | UI module package init |
| `src/ui/app.py` | 450 | Streamlit dashboard application |
| `src/ui/plots.py` | 315 | Plotly CTG visualization utilities |
| `tests/test_phase4.py` | 515 | Phase 4 tests (classifier & safety net) |
| `tests/test_ui_components.py` | 389 | Phase 5 tests (UI & alerts) |

### Modified Files

| File | Change Description |
|------|-------------------|
| `src/data/loader.py` | Added `extract_ph()`, `get_outcome_label()`, `get_all_ph_values()` methods for pH extraction and label generation |
| `src/rules/decelerations.py` | Added `nadir_value` field to Deceleration dataclass |

### New Features by Phase

#### Phase 4: Classifier & Medical Override
- **pH Extraction:** Extract pH values from CTU-UHB header files for ground truth labels
- **Label Generation:** Map pH to categories (pH < 7.15 → Pathological, 7.15-7.20 → Intermediate, ≥7.20 → Normal)
- **XGBoost Classifier:** Multi-class classifier with Stratified K-Fold CV and class weighting
- **Medical Override:** Safety net that can force classifications based on critical findings

#### Phase 5: Alert Engine & Dashboard
- **Alert Engine:** Generate explanatory alerts in Hebrew with findings and recommendations
- **Plotly Visualization:** Interactive CTG plots with deceleration highlighting
- **Streamlit Dashboard:** Real-time monitoring interface with patient list
- **Category Indicators:** Color-coded status display with emojis

---

## 3. System Architecture

### 3.1 High-Level Architecture Diagram

```
                              ┌─────────────────────────────────────┐
                              │         INPUT: CTG SIGNAL           │
                              │      (FHR @ 4Hz + UC @ 4Hz)         │
                              └─────────────────┬───────────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
                    v                           v                           v
        ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
        │   DATA LOADING    │       │   PREPROCESSING   │       │   RULE ENGINE     │
        │                   │       │                   │       │                   │
        │ • CTUDataLoader   │       │ • Gap Detection   │       │ • Baseline        │
        │ • CTGRecord       │       │ • 10-Second Rule  │       │ • Variability     │
        │ • pH Extraction   │       │ • Spike Removal   │       │ • Decelerations   │
        │                   │       │ • Median Filter   │       │ • Tachysystole    │
        │                   │       │                   │       │ • Sinusoidal      │
        └─────────┬─────────┘       └─────────┬─────────┘       └─────────┬─────────┘
                  │                           │                           │
                  └───────────────────────────┼───────────────────────────┘
                                              │
                                              v
                              ┌─────────────────────────────────────┐
                              │         MOMENT ENCODER              │
                              │   (AutonLab/MOMENT-1-large)         │
                              │                                     │
                              │   Input:  FHR Signal (10-min)       │
                              │   Output: 1024-dim Embedding        │
                              │   Mode:   Zero-Shot Inference       │
                              └─────────────────┬───────────────────┘
                                                │
                                                v
                              ┌─────────────────────────────────────┐
                              │         FEATURE FUSION              │
                              │                                     │
                              │   MOMENT Embedding (1024 dims)      │
                              │   + Rule Features (11 dims)         │
                              │   = Feature Vector (1035 dims)      │
                              └─────────────────┬───────────────────┘
                                                │
                                                v
                              ┌─────────────────────────────────────┐
                              │      XGBOOST CLASSIFIER             │
                              │   (Stratified K-Fold CV)            │
                              │                                     │
                              │   Output: Category 0/1/2            │
                              │           + Confidence Score        │
                              └─────────────────┬───────────────────┘
                                                │
                                                v
                              ┌─────────────────────────────────────┐
                              │      MEDICAL OVERRIDE               │
                              │   (Safety Net Logic)                │
                              │                                     │
                              │   • Sinusoidal → Force Cat 3        │
                              │   • Absent Var + Decels → Cat 3     │
                              │   • Safety Floor → At least Cat 2   │
                              └─────────────────┬───────────────────┘
                                                │
                                                v
                              ┌─────────────────────────────────────┐
                              │        ALERT ENGINE                 │
                              │   (Hebrew XAI Strings)              │
                              │                                     │
                              │   Output: Headline, Explanation,    │
                              │           Findings, Recommendations │
                              └─────────────────┬───────────────────┘
                                                │
                                                v
                              ┌─────────────────────────────────────┐
                              │      STREAMLIT DASHBOARD            │
                              │                                     │
                              │   • Patient List                    │
                              │   • CTG Plot (Plotly)               │
                              │   • Alert Display                   │
                              │   • Findings Panel                  │
                              └─────────────────────────────────────┘
```

### 3.2 Feature Vector Structure (1035 dimensions)

```
Index Range     | Feature                    | Normalization
----------------|----------------------------|---------------
0-1023          | MOMENT Embedding           | Unit norm
1024            | Baseline FHR               | / 160
1025            | Variability Value          | / 25
1026            | Variability: Absent        | One-hot (0/1)
1027            | Variability: Minimal       | One-hot (0/1)
1028            | Variability: Moderate      | One-hot (0/1)
1029            | Variability: Marked        | One-hot (0/1)
1030            | Late Deceleration Count    | / 10
1031            | Variable Deceleration Count| / 10
1032            | Recurrent Decelerations    | Binary (0/1)
1033            | Tachysystole Flag          | Binary (0/1)
1034            | Sinusoidal Flag            | Binary (0/1)
```

### 3.3 Medical Override Rules

```
HARD OVERRIDE RULES (Force Category 3):
┌────────────────────────────────────────────────────────────────────┐
│ 1. Sinusoidal Pattern Detected → ALWAYS Category 3                 │
│    "Critical finding indicating potential severe fetal anemia"     │
│                                                                    │
│ 2. Absent Variability + (Recurrent Late OR Variable Decels         │
│    OR Bradycardia) → ALWAYS Category 3                             │
│    "Highly predictive of fetal acidemia"                           │
└────────────────────────────────────────────────────────────────────┘

SAFETY FLOOR RULE (Minimum Category 2):
┌────────────────────────────────────────────────────────────────────┐
│ 3. Absent Variability + ML predicts Normal → Upgrade to Cat 2      │
│    "Absent variability requires closer monitoring"                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. Directory Structure

```
SentinelFetal/
│
├── .git/                           # Git repository
├── .claude/                        # Claude workspace configuration
├── .gitignore                      # Git ignore rules
├── .pytest_cache/                  # Pytest cache
├── .venv/                          # Python virtual environment
│
├── data/                           # Dataset storage
│   ├── ctu-chb-intrapartum-cardiotocography-database-1.0.0/
│   │   └── [552 CTG recordings: *.hea, *.dat files]
│   └── processed/                  # Processed datasets (NEW)
│       ├── X.npy                   # Feature matrix (N, 1035)
│       └── y.npy                   # Labels (N,)
│
├── docs/                           # Documentation
│   ├── SentinelFetal_PRD.docx.txt
│   ├── SentinelFetal_TechSpec.docx.txt
│   └── SentinelFetal_Gen35_Spec.docx.txt
│
├── models/                         # Saved models (NEW)
│   ├── xgb_demo.json              # Trained XGBoost model
│   └── xgb_demo.config.json       # Model configuration
│
├── notebooks/                      # Jupyter notebooks
│
├── src/                            # Main source code (6,130 lines)
│   ├── __init__.py                 # Package init (version 0.1.0)
│   │
│   ├── data/                       # Data loading & preprocessing (921 lines)
│   │   ├── __init__.py
│   │   ├── loader.py               # CTU-UHB dataset loader (425 lines) ← UPDATED
│   │   └── preprocess.py           # Signal preprocessing (490 lines)
│   │
│   ├── rules/                      # Clinical rule engine (1,538 lines)
│   │   ├── __init__.py
│   │   ├── baseline.py             # Baseline FHR calculation (241 lines)
│   │   ├── variability.py          # Variability analysis (283 lines)
│   │   ├── decelerations.py        # Deceleration detection (474 lines)
│   │   ├── tachysystole.py         # Tachysystole detection (202 lines)
│   │   └── sinusoidal.py           # Sinusoidal pattern detection (338 lines)
│   │
│   ├── models/                     # AI models & classifier (1,305 lines)
│   │   ├── __init__.py
│   │   ├── moment_encoder.py       # MOMENT foundation model (477 lines)
│   │   ├── fusion.py               # Feature fusion utilities (371 lines)
│   │   └── classifier.py           # XGBoost classifier (457 lines) ← NEW
│   │
│   ├── analysis/                   # Analysis & alerts (641 lines) ← NEW MODULE
│   │   ├── __init__.py
│   │   ├── alerts.py               # Alert Engine with Hebrew XAI (331 lines)
│   │   └── override.py             # Medical Override safety logic (310 lines)
│   │
│   ├── training/                   # Training scripts (531 lines) ← NEW MODULE
│   │   ├── __init__.py
│   │   ├── prepare_data.py         # Dataset preparation pipeline (344 lines)
│   │   └── train_demo.py           # Demo model training (187 lines)
│   │
│   ├── ui/                         # User interface (765 lines) ← NEW MODULE
│   │   ├── __init__.py
│   │   ├── app.py                  # Streamlit dashboard (450 lines)
│   │   └── plots.py                # Plotly visualizations (315 lines)
│   │
│   └── visualize_preprocessing.py  # Visualization tools (251 lines)
│
├── tests/                          # Unit tests (2,387 lines)
│   ├── __init__.py
│   ├── test_preprocessing.py       # Preprocessing tests (273 lines)
│   ├── test_rules.py               # Rule engine tests (563 lines)
│   ├── test_pipeline_integration.py # Integration tests (697 lines)
│   ├── test_phase4.py              # Classifier & override tests (515 lines) ← NEW
│   └── test_ui_components.py       # UI & alerts tests (389 lines) ← NEW
│
├── README.md                       # Project documentation
├── DEVELOPMENT_PLAN.md             # Development roadmap
├── requirements.txt                # Python dependencies
└── preprocessing_validation.png    # Visualization output
```

---

## 5. Module Documentation

---

### 5.1 Data Module

**Location:** `src/data/`
**Total Lines:** 921
**Purpose:** Load and preprocess CTG recordings from the CTU-UHB database

---

#### 5.1.1 loader.py (425 lines)

**Purpose:** Load CTG recordings and extract pH-based labels from the CTU-UHB database

##### Exceptions

```python
class DataLoaderError(Exception):
    """Base exception for data loader errors."""

class RecordNotFoundError(DataLoaderError):
    """Raised when a requested record cannot be found."""

class InvalidRecordError(DataLoaderError):
    """Raised when a record has invalid or missing data."""
```

##### Classes

###### CTGRecord (Dataclass)

```python
@dataclass
class CTGRecord:
    record_id: str                    # Unique identifier (e.g., '1001')
    fhr1: np.ndarray                  # Primary fetal heart rate signal
    fhr2: Optional[np.ndarray]        # Secondary FHR signal (may be None)
    uc: np.ndarray                    # Uterine contractions signal
    sampling_rate: float              # Sampling frequency (4 Hz)
    duration_seconds: float           # Total recording duration
    metadata: dict[str, Any]          # Additional info
```

###### CTUDataLoader (Class)

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `list_records()` | None | `list[str]` | Get sorted list of all available record IDs |
| `load_record(record_id)` | `record_id: str` | `CTGRecord` | Load single CTG record |
| `load_record_as_dataframe(record_id)` | `record_id: str` | `pd.DataFrame` | Load as pandas DataFrame |
| `iter_records(limit)` | `limit: Optional[int]` | `Generator[CTGRecord]` | Iterate over records |
| `extract_ph(record_id)` | `record_id: str` | `Optional[float]` | **NEW** Extract pH from header file |
| `get_outcome_label(record_id)` | `record_id: str` | `int` | **NEW** Get category label (0/1/2) based on pH |
| `get_all_ph_values()` | None | `dict[str, Optional[float]]` | **NEW** Extract pH for all records |

##### pH Label Mapping (NEW)

```python
def get_outcome_label(self, record_id: str) -> int:
    """
    Map pH to category labels per Israeli Position Paper:
        - pH < 7.15 → Label 2 (Category 3 - Pathological)
        - 7.15 ≤ pH < 7.20 → Label 1 (Category 2 - Intermediate)
        - pH ≥ 7.20 (or unknown) → Label 0 (Category 1 - Normal)
    """
```

---

#### 5.1.2 preprocess.py (490 lines)

**Purpose:** CTG signal preprocessing implementing the 10-second rule

*(No changes from previous documentation)*

---

### 5.2 Rules Module

**Location:** `src/rules/`
**Total Lines:** 1,538
**Purpose:** Clinical rule engine implementing Israeli Position Paper guidelines

*(No significant changes from previous documentation - baseline.py, variability.py, decelerations.py, tachysystole.py, sinusoidal.py remain the same)*

---

### 5.3 Models Module

**Location:** `src/models/`
**Total Lines:** 1,305
**Purpose:** AI models, feature fusion, and classification

---

#### 5.3.1 moment_encoder.py (477 lines)

*(No changes from previous documentation)*

---

#### 5.3.2 fusion.py (371 lines)

*(No changes from previous documentation)*

---

#### 5.3.3 classifier.py (457 lines) - NEW

**Purpose:** XGBoost classifier with Stratified K-Fold cross-validation

##### Classes

###### ClassifierConfig (Dataclass)

```python
@dataclass
class ClassifierConfig:
    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Regularization
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    # Training parameters
    n_folds: int = 5
    random_state: int = 42
    use_class_weights: bool = True
    tree_method: str = "auto"
```

###### TrainingResult (Dataclass)

```python
@dataclass
class TrainingResult:
    cv_scores: list[float]           # Per-fold F1 scores
    mean_cv_score: float             # Mean CV F1 score
    std_cv_score: float              # Standard deviation
    best_score: float                # Best fold score
    feature_importance: Optional[np.ndarray]  # Feature importances
    confusion_matrix: Optional[np.ndarray]    # Final confusion matrix
    classification_report: Optional[str]      # sklearn report
```

###### XGBClassifierWrapper (Class)

```python
class XGBClassifierWrapper:
    CLASS_NAMES = ['Normal (Cat 1)', 'Intermediate (Cat 2)', 'Pathological (Cat 3)']

    def __init__(self, config: Optional[ClassifierConfig] = None)
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `train(X, y, validate)` | `X: np.ndarray, y: np.ndarray, validate: bool` | `TrainingResult` | Train with Stratified K-Fold CV |
| `predict(X)` | `X: np.ndarray` | `np.ndarray` | Predict class labels |
| `predict_proba(X)` | `X: np.ndarray` | `np.ndarray` | Predict class probabilities |
| `get_feature_importance()` | None | `Optional[np.ndarray]` | Get feature importance scores |
| `save_model(path)` | `path: Union[str, Path]` | None | Save model to disk |
| `load_model(path)` | `path: Union[str, Path]` | None | Load model from disk |

**Usage Example:**

```python
from src.models.classifier import XGBClassifierWrapper, ClassifierConfig

# Configure and train
config = ClassifierConfig(n_estimators=100, max_depth=5, n_folds=5)
classifier = XGBClassifierWrapper(config)
result = classifier.train(X, y)

print(f"Mean CV F1: {result.mean_cv_score:.3f} ± {result.std_cv_score:.3f}")

# Save and load
classifier.save_model("models/my_model")
classifier.load_model("models/my_model")

# Predict
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
```

---

### 5.4 Analysis Module (NEW)

**Location:** `src/analysis/`
**Total Lines:** 641
**Purpose:** Alert generation and medical override logic

---

#### 5.4.1 alerts.py (331 lines)

**Purpose:** Generate explanatory alerts in Hebrew per Section 8 of the spec

##### Classes

###### Alert (Dataclass)

```python
@dataclass
class Alert:
    """
    מייצג התראה מלאה עם הסבר.
    Represents a complete alert with explanation in Hebrew.
    """
    category: int              # Classification category (1, 2, or 3)
    confidence: float          # Model confidence (0-1)
    headline: str              # Short headline (Hebrew)
    explanation: str           # Detailed explanation (Hebrew)
    findings: List[str]        # Medical findings supporting the decision
    recommendations: List[str] # Recommendations for medical team
    timestamp: str             # ISO format timestamp
```

##### Functions

###### generate_alert()

```python
def generate_alert(
    category: int,
    confidence: float,
    baseline: Union[float, BaselineResult],
    variability: Union[dict, VariabilityResult],
    decelerations: List[Deceleration],
    tachysystole: Union[dict, TachysystoleResult],
    sinusoidal: Union[dict, SinusoidalResult]
) -> Alert:
    """Generate an explanatory alert with Hebrew text."""
```

##### Hebrew Headlines by Category

| Category | Hebrew Headline |
|----------|-----------------|
| 1 | `סטטוס ירוק - קטגוריה 1 (תקין)` |
| 2 | `התראה כתומה - קטגוריה 2 (ביניים)` |
| 3 | `התראה אדומה - קטגוריה 3 (פתולוגי)` |

##### Category 3 Recommendations (Hebrew)

```python
recommendations = [
    'הערכה מיידית של הסיבות האפשריות',
    'שקילת החייאה תוך-רחמית',
    'היערכות ליילוד מיידי אם אין שיפור'
]
```

##### Helper Functions

```python
def get_category_color(category: int) -> str:
    """Get display color: green/orange/red"""

def get_category_emoji(category: int) -> str:
    """Get emoji indicator: 🟢/🟠/🔴"""
```

---

#### 5.4.2 override.py (310 lines)

**Purpose:** Medical override safety logic (Section 7 of the spec)

##### Enumeration

```python
class OverrideReason(Enum):
    NONE = auto()
    SINUSOIDAL_PATTERN = auto()
    ABSENT_VARIABILITY_WITH_DECELS = auto()
    BRADYCARDIA = auto()
    ABSENT_VARIABILITY_SAFETY_FLOOR = auto()
```

##### Classes

###### MedicalOverride (Dataclass)

```python
@dataclass
class MedicalOverride:
    should_override: bool      # Whether ML prediction should be overridden
    final_category: int        # Final category after override (0, 1, or 2)
    reason: OverrideReason     # Reason for override
    ml_prediction: int         # Original ML prediction
    explanation: str           # Human-readable explanation
```

##### Main Function

```python
def apply_medical_override(
    ml_prediction: int,
    baseline: BaselineResult,
    variability: VariabilityResult,
    decelerations: List[Deceleration],
    tachysystole: TachysystoleResult,
    sinusoidal: SinusoidalResult
) -> MedicalOverride:
    """
    Apply medical override rules to ML prediction.

    HARD OVERRIDE RULES (Force Category 3):
    1. Sinusoidal pattern detected → Force Category 3
    2. Absent variability + (recurrent late/variable decels OR bradycardia) → Force Category 3

    SAFETY FLOOR RULE (Force Category 2):
    3. ML predicts Normal BUT variability is Absent → Force Category 2
    """
```

**Usage Example:**

```python
from src.analysis.override import apply_medical_override

override = apply_medical_override(
    ml_prediction=0,  # ML says Normal
    baseline=baseline_result,
    variability=variability_result,  # Absent variability
    decelerations=decelerations,
    tachysystole=tachy_result,
    sinusoidal=sinusoidal_result  # Detected!
)

if override.should_override:
    print(f"OVERRIDE: {override.reason.name}")
    print(f"Final category: {override.final_category}")
```

---

### 5.5 Training Module (NEW)

**Location:** `src/training/`
**Total Lines:** 531
**Purpose:** Dataset preparation and model training scripts

---

#### 5.5.1 prepare_data.py (344 lines)

**Purpose:** Run the full pipeline to create training dataset

**Usage:**

```bash
# Test with 5 records (uses mock MOMENT)
python src/training/prepare_data.py --limit 5 --use-mock

# Process all records with real MOMENT
python src/training/prepare_data.py --device cuda
```

**Command Line Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/ctu-chb-...` | Path to CTU-UHB dataset |
| `--output-dir` | `data/processed` | Output directory for X.npy, y.npy |
| `--limit` | None | Limit number of records to process |
| `--device` | `auto` | Device for MOMENT model (auto/cuda/mps/cpu) |
| `--use-mock` | False | Use mock MOMENT embeddings (testing only) |

**Output:**

- `data/processed/X.npy` - Feature matrix (N_windows, 1035)
- `data/processed/y.npy` - Labels (N_windows,)

---

#### 5.5.2 train_demo.py (187 lines)

**Purpose:** Train XGBoost classifier on prepared dataset

**Usage:**

```bash
python src/training/train_demo.py
python src/training/train_demo.py --data-dir data/processed --output models/xgb_demo.json
```

**Command Line Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/processed` | Directory containing X.npy and y.npy |
| `--output` | `models/xgb_demo.json` | Output path for trained model |
| `--n-folds` | 3 | Number of cross-validation folds |

**Output:**

- `models/xgb_demo.json` - Trained XGBoost model
- `models/xgb_demo.config.json` - Model configuration

---

### 5.6 UI Module (NEW)

**Location:** `src/ui/`
**Total Lines:** 765
**Purpose:** Streamlit dashboard and Plotly visualizations

---

#### 5.6.1 app.py (450 lines)

**Purpose:** Main Streamlit dashboard application

**Usage:**

```bash
streamlit run src/ui/app.py
```

**Features:**

- Patient list sidebar with record selection
- Real-time CTG visualization with Plotly
- Hebrew alert display with color coding
- Findings and recommendations panels
- Technical details expandable section
- ML/Rule-based classification toggle

**Main Functions:**

```python
def run_full_pipeline(record: CTGRecord, use_ml: bool = True) -> Dict[str, Any]:
    """
    Run the complete SentinelFetal pipeline on a record.

    Returns:
        Dictionary with all pipeline results including:
        - fhr, uc: Preprocessed signals
        - baseline, variability, decelerations, tachysystole, sinusoidal: Rule results
        - ml_prediction, confidence: ML classification
        - final_category: After medical override
        - alert: Generated Alert object
    """
```

---

#### 5.6.2 plots.py (315 lines)

**Purpose:** Plotly-based CTG visualizations

##### Color Scheme

```python
FHR_COLOR = '#1E90FF'      # Dodger Blue for FHR
UC_COLOR = '#FF8C00'       # Dark Orange for UC
DECEL_COLOR = 'rgba(255, 0, 0, 0.3)'  # Red for decelerations
```

##### Main Functions

###### create_ctg_plot()

```python
def create_ctg_plot(
    fhr: np.ndarray,
    uc: np.ndarray,
    decelerations: Optional[List[Deceleration]] = None,
    sampling_rate: float = 4.0,
    title: str = "CTG Monitor",
    height: int = 600
) -> go.Figure:
    """
    Create an interactive CTG plot with FHR, UC, and deceleration markers.

    Features:
        - Blue line for FHR (top panel)
        - Orange line for UC (bottom panel)
        - Green band for normal FHR range (110-160 bpm)
        - Red/orange/purple highlighting for decelerations
        - Hebrew labels and hover text
    """
```

###### create_category_indicator()

```python
def create_category_indicator(
    category: int,
    confidence: float,
    size: int = 150
) -> go.Figure:
    """Create a category indicator gauge."""
```

###### create_findings_summary()

```python
def create_findings_summary(
    baseline: float,
    variability: dict,
    late_count: int,
    variable_count: int,
    tachysystole: bool
) -> go.Figure:
    """Create a visual summary of findings."""
```

---

### 5.7 Visualization Module

**Location:** `src/visualize_preprocessing.py`
**Lines:** 251
**Purpose:** Create visual validation of preprocessing

*(No changes from previous documentation)*

---

## 6. Data Flow & Pipeline

### 6.1 Complete Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          PHASE 1: DATA LOADING                           │
├──────────────────────────────────────────────────────────────────────────┤
│  Raw CTG File (*.hea, *.dat) + pH from header                           │
│         │                                                                │
│         v                                                                │
│  CTUDataLoader → CTGRecord + pH label                                   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌──────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: PREPROCESSING                            │
├──────────────────────────────────────────────────────────────────────────┤
│  CTGPreprocessor: Gap filling (10-second rule), spike removal           │
│         │                                                                │
│         v                                                                │
│  PreprocessingResult(processed_signal, masks, stats)                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           v                        v                        v
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ PHASE 3A: RULE ENGINE│  │ PHASE 3B: MOMENT     │  │  Uterine Contractions│
├──────────────────────┤  ├──────────────────────┤  │        (UC)          │
│ • calculate_baseline │  │ MomentFeatureExtractor│  └──────────┬───────────┘
│ • calculate_variability│ │ → 1024-dim Embedding │             │
│ • detect_decelerations│  └──────────────────────┘             │
│ • detect_tachysystole │<───────────────────────────────────────┘
│ • detect_sinusoidal   │
└──────────────────────┘
           │                        │
           └────────────┬───────────┘
                        v
┌──────────────────────────────────────────────────────────────────────────┐
│                       PHASE 4: FEATURE FUSION                            │
├──────────────────────────────────────────────────────────────────────────┤
│  build_feature_vector(): MOMENT (1024) + Rules (11) → 1035 dims         │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌──────────────────────────────────────────────────────────────────────────┐
│                      PHASE 5: CLASSIFICATION                             │
├──────────────────────────────────────────────────────────────────────────┤
│  XGBClassifierWrapper.predict() → Category 0/1/2 + Confidence            │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌──────────────────────────────────────────────────────────────────────────┐
│                     PHASE 6: MEDICAL OVERRIDE                            │
├──────────────────────────────────────────────────────────────────────────┤
│  apply_medical_override(): Safety net checks                             │
│  • Sinusoidal detected? → Force Cat 3                                    │
│  • Absent var + bad findings? → Force Cat 3                              │
│  • Absent var alone? → At least Cat 2                                    │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌──────────────────────────────────────────────────────────────────────────┐
│                      PHASE 7: ALERT GENERATION                           │
├──────────────────────────────────────────────────────────────────────────┤
│  generate_alert(): Hebrew headline, explanation, findings, recommendations│
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌──────────────────────────────────────────────────────────────────────────┐
│                     PHASE 8: DASHBOARD DISPLAY                           │
├──────────────────────────────────────────────────────────────────────────┤
│  Streamlit app: CTG plot, alert display, findings panel                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Usage Example (Complete Pipeline)

```python
from src.data.loader import CTUDataLoader
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig
from src.rules import (
    calculate_baseline, calculate_variability, detect_decelerations,
    detect_tachysystole, detect_sinusoidal_pattern
)
from src.models.moment_encoder import MomentFeatureExtractor
from src.models.fusion import build_feature_vector
from src.models.classifier import XGBClassifierWrapper
from src.analysis.override import apply_medical_override
from src.analysis.alerts import generate_alert

# 1. Load and preprocess
loader = CTUDataLoader("data/ctu-chb-...")
record = loader.load_record("1001")
preprocessor = CTGPreprocessor()
result = preprocessor.process(record.fhr1)
fhr = result.processed_signal

# 2. Rule Engine
baseline = calculate_baseline(fhr)
variability = calculate_variability(fhr)
decelerations = detect_decelerations(fhr, record.uc, baseline.value)
tachysystole = detect_tachysystole(record.uc)
sinusoidal = detect_sinusoidal_pattern(fhr)

# 3. MOMENT embedding
moment = MomentFeatureExtractor()
embedding = moment.extract(fhr[:2400])

# 4. Feature fusion
feature_vec = build_feature_vector(
    embedding=embedding, baseline=baseline, variability=variability,
    decelerations=decelerations, tachysystole=tachysystole, sinusoidal=sinusoidal
)

# 5. Classification
classifier = XGBClassifierWrapper()
classifier.load_model("models/xgb_demo")
ml_pred = classifier.predict(feature_vec.vector.reshape(1, -1))[0]
confidence = classifier.predict_proba(feature_vec.vector.reshape(1, -1)).max()

# 6. Medical override
override = apply_medical_override(
    ml_prediction=ml_pred, baseline=baseline, variability=variability,
    decelerations=decelerations, tachysystole=tachysystole, sinusoidal=sinusoidal
)
final_category = override.final_category + 1  # Convert to 1/2/3

# 7. Generate alert
alert = generate_alert(
    category=final_category, confidence=confidence, baseline=baseline,
    variability=variability, decelerations=decelerations,
    tachysystole=tachysystole, sinusoidal=sinusoidal
)

print(f"Category: {final_category}")
print(f"Headline: {alert.headline}")
print(f"Findings: {alert.findings}")
```

---

## 7. Clinical Rules Reference

### 7.1 Category Classification Matrix

| Parameter | Category 1 (Normal) | Category 2 (Intermediate) | Category 3 (Pathological) |
|-----------|---------------------|---------------------------|---------------------------|
| **Baseline** | 110-160 bpm | < 110 or > 160 bpm | Extreme values |
| **Variability** | 6-25 bpm (Moderate) | 3-5 bpm (Minimal) | 0-2 bpm (Absent) > 50 min |
| **Accelerations** | Present | Absent | N/A |
| **Early Decels** | Present or absent | N/A | N/A |
| **Variable Decels** | Absent | Present, no severity | With severity signs |
| **Late Decels** | Absent | Occasional | Recurrent (≥50% contractions) |
| **Sinusoidal** | **No** | **No** | **YES (SEVERE!)** |
| **Tachysystole** | No | Yes (>5/10min) | Extreme |

### 7.2 pH-Based Label Mapping

| pH Value | Category | Label | Clinical Significance |
|----------|----------|-------|----------------------|
| pH < 7.15 | Pathological | 2 | Fetal acidemia |
| 7.15 ≤ pH < 7.20 | Intermediate | 1 | Borderline |
| pH ≥ 7.20 | Normal | 0 | Healthy |
| Unknown | Normal | 0 | Default assumption |

*(Remaining clinical rules unchanged from previous documentation)*

---

## 8. Testing Documentation

### 8.1 Test Coverage Summary

| Module | Test File | Test Count | Lines | Coverage Focus |
|--------|-----------|------------|-------|----------------|
| Preprocessing | `test_preprocessing.py` | 14 | 273 | Gap filling, 10-second rule |
| Rules | `test_rules.py` | 24 | 563 | All rule engine components |
| Integration | `test_pipeline_integration.py` | 18 | 697 | End-to-end pipeline |
| Phase 4 | `test_phase4.py` | 18 | 515 | pH extraction, override, classifier |
| Phase 5 | `test_ui_components.py` | 24 | 389 | Alerts, UI, Hebrew strings |
| **Total** | | **98** | **2,387** | |

### 8.2 New Test Files

#### test_phase4.py (515 lines)

```python
class TestPHExtraction:
    def test_extract_ph_record_1001()
    def test_get_outcome_label_pathological()
    def test_get_outcome_label_unknown_returns_normal()
    def test_ph_label_boundaries()

class TestMedicalOverride:
    def test_sinusoidal_forces_category_3()
    def test_absent_variability_with_late_decels_forces_category_3()
    def test_absent_variability_with_bradycardia_forces_category_3()
    def test_safety_floor_absent_variability_upgrades_normal_to_intermediate()
    def test_no_override_when_all_normal()
    def test_ml_pathological_prediction_not_downgraded()

class TestDatasetPreparation:
    def test_processed_directory_exists_after_prepare()

class TestClassifier:
    def test_classifier_import()
    def test_classifier_instantiation()
    def test_classifier_training_synthetic()
```

#### test_ui_components.py (389 lines)

```python
class TestAlertGeneration:
    def test_category_3_headline_exact_string()
    def test_category_2_headline_exact_string()
    def test_category_1_headline_exact_string()
    def test_category_3_recommendations_exact_strings()
    def test_category_2_recommendations_exact_strings()
    def test_category_1_recommendations_exact_strings()
    def test_bradycardia_finding_hebrew()
    def test_tachycardia_finding_hebrew()
    def test_absent_variability_finding_hebrew()
    def test_sinusoidal_finding_hebrew()
    def test_late_deceleration_finding_hebrew()

class TestAlertDataclass:
    def test_alert_creation_with_valid_category()
    def test_alert_invalid_category_raises()
    def test_alert_invalid_confidence_raises()
    def test_alert_timestamp_auto_generated()

class TestCategoryHelpers:
    def test_get_category_color()
    def test_get_category_emoji()

class TestPlotFunctions:
    def test_create_ctg_plot_returns_figure()
    def test_create_ctg_plot_with_decelerations()

class TestTrainDemoScript:
    def test_train_demo_script_exists()
    def test_train_demo_imports()

class TestAppImports:
    def test_app_module_imports()
    def test_plots_module_imports()
    def test_alerts_module_imports()
```

### 8.3 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_phase4.py -v
pytest tests/test_ui_components.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test class
pytest tests/test_phase4.py::TestMedicalOverride -v

# Run specific test
pytest tests/test_ui_components.py::TestAlertGeneration::test_category_3_headline_exact_string -v
```

---

## 9. Configuration & Dependencies

### 9.1 Python Requirements

```
# requirements.txt

# Core dependencies
numpy>=1.24.0              # Numerical computing
pandas>=2.0.0              # Data manipulation
scipy>=1.10.0              # Scientific computing (signal processing, FFT)
wfdb>=4.1.0                # PhysioNet data loading

# Visualization
matplotlib>=3.7.0          # Static plots
plotly>=5.14.0             # Interactive plots

# Testing
pytest>=7.3.0              # Test framework
pytest-cov>=4.1.0          # Coverage reporting

# Utilities
scikit-learn>=1.2.0        # ML utilities
tqdm>=4.65.0               # Progress bars
python-dateutil>=2.8.0     # Date utilities

# Phase 3+ Dependencies
torch>=2.0.0               # PyTorch for MOMENT
momentfm>=0.1.0            # MOMENT foundation model
xgboost>=1.7.0             # Gradient boosting classifier

# Phase 5 Dependencies
streamlit>=1.28.0          # Dashboard framework
```

### 9.2 Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 9.3 Running the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/ui/app.py

# Access at http://localhost:8501
```

---

## 10. Development Status

### 10.1 Completed Phases

#### Phase 1: Data Pipeline ✅
- [x] CTU-UHB data loader
- [x] Signal preprocessing with 10-second rule
- [x] Gap detection and filling
- [x] Spike detection and removal

#### Phase 2: Rule Engine ✅
- [x] Baseline FHR calculator
- [x] Variability analyzer (4 categories)
- [x] Deceleration detector and classifier
- [x] Tachysystole detector
- [x] Sinusoidal pattern detector

#### Phase 3: MOMENT Integration ✅
- [x] MomentFeatureExtractor class
- [x] Mock mode for testing
- [x] Sliding window extraction
- [x] Feature fusion module
- [x] 1035-dim feature vector construction

#### Phase 4: Classifier & Medical Override ✅
- [x] pH extraction from header files
- [x] Label generation based on pH
- [x] XGBClassifierWrapper with Stratified K-Fold CV
- [x] Medical Override safety logic
- [x] Dataset preparation pipeline
- [x] Demo model training script

#### Phase 5: Alert Engine & Dashboard ✅
- [x] Alert dataclass with Hebrew XAI strings
- [x] Streamlit dashboard with CTG visualization
- [x] Plotly-based create_ctg_plot (blue FHR, orange UC, red decels)
- [x] Hebrew findings and recommendations
- [x] Category color/emoji indicators
- [x] 98 tests passing

### 10.2 Current Statistics

| Metric | Value |
|--------|-------|
| Total Source Lines | 6,130 |
| Total Test Lines | 2,387 |
| Total Lines | 8,517 |
| Total Tests | 98 |
| Test Pass Rate | 100% |

### 10.3 Future Work

#### Phase 6: Real-time System ⏳
- [ ] Streaming data handler
- [ ] Real-time inference pipeline
- [ ] WebSocket-based updates
- [ ] Clinical alert notifications

#### Additional Enhancements ⏳
- [ ] Fine-tuning MOMENT on CTU-UHB
- [ ] Clinical validation study
- [ ] Multi-hospital deployment
- [ ] Regulatory compliance (FDA/CE)

---

## Appendix A: Quick Reference

### Function Signatures

```python
# Data Loading
CTUDataLoader(data_dir: Union[str, Path])
loader.load_record(record_id: str) -> CTGRecord
loader.extract_ph(record_id: str) -> Optional[float]
loader.get_outcome_label(record_id: str) -> int

# Preprocessing
CTGPreprocessor(config: Optional[PreprocessingConfig])
preprocessor.process(fhr: np.ndarray) -> PreprocessingResult

# Rule Engine
calculate_baseline(fhr, sampling_rate=4.0) -> BaselineResult
calculate_variability(fhr, sampling_rate=4.0) -> VariabilityResult
detect_decelerations(fhr, uc, baseline) -> list[Deceleration]
detect_tachysystole(uc, sampling_rate=4.0) -> TachysystoleResult
detect_sinusoidal_pattern(fhr, sampling_rate=4.0) -> SinusoidalResult

# MOMENT Encoder
MomentFeatureExtractor(device=None, use_mock=False)
extractor.extract(fhr: np.ndarray) -> np.ndarray

# Feature Fusion
build_feature_vector(embedding, baseline, variability, ...) -> FeatureVector

# Classifier
XGBClassifierWrapper(config: Optional[ClassifierConfig])
classifier.train(X, y) -> TrainingResult
classifier.predict(X) -> np.ndarray
classifier.predict_proba(X) -> np.ndarray

# Medical Override
apply_medical_override(ml_prediction, baseline, variability, ...) -> MedicalOverride

# Alert Generation
generate_alert(category, confidence, baseline, ...) -> Alert
get_category_color(category: int) -> str
get_category_emoji(category: int) -> str

# Visualization
create_ctg_plot(fhr, uc, decelerations, ...) -> go.Figure
```

### Clinical Thresholds

| Parameter | Normal | Abnormal |
|-----------|--------|----------|
| Baseline | 110-160 bpm | < 110 or > 160 |
| Variability | 6-25 bpm | < 6 or > 25 |
| Decelerations | None | Any late/variable |
| Contractions | ≤ 5/10min | > 5/10min |
| Sinusoidal | Not present | **CATEGORY 3** |
| pH | ≥ 7.20 | < 7.15 (Pathological) |

---

**End of Documentation**

*Generated: January 2026*
*SentinelFetal Gen3.5 - Hybrid AI System for Fetal Monitoring*
*All 98 tests passing ✅*
