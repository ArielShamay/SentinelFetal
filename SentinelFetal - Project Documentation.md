# SentinelFetal - Project Documentation

**Version:** Gen3.5
**Last Updated:** January 2026
**Author:** Ariel Shamay
**Total Lines of Code:** ~5,100 lines (3,558 source + 1,533 tests)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Module Documentation](#4-module-documentation)
   - [4.1 Data Module](#41-data-module)
   - [4.2 Rules Module](#42-rules-module)
   - [4.3 Models Module](#43-models-module)
   - [4.4 Visualization Module](#44-visualization-module)
5. [Data Flow & Pipeline](#5-data-flow--pipeline)
6. [Clinical Rules Reference](#6-clinical-rules-reference)
7. [Testing Documentation](#7-testing-documentation)
8. [Configuration & Dependencies](#8-configuration--dependencies)
9. [Development Status](#9-development-status)

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
| **Category Classification** | Category 1 (Normal), 2 (Intermediate), 3 (Pathological) |

### 1.3 Classification Categories

| Category | Status | Color | Action Required |
|----------|--------|-------|-----------------|
| **Category 1** | Normal | Green | Routine monitoring |
| **Category 2** | Intermediate | Orange | Close monitoring, possible intervention |
| **Category 3** | Pathological | Red | Immediate intervention required |

### 1.4 Technology Stack

- **Language:** Python 3.9+
- **AI Model:** MOMENT (AutonLab/MOMENT-1-large) - 385M parameters
- **Dataset:** CTU-UHB Intrapartum Cardiotocography Database (552 recordings)
- **Clinical Basis:** Israeli Position Paper on CTG Interpretation

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

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
        │ • wfdb interface  │       │ • Spike Removal   │       │ • Decelerations   │
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
                              │        HYBRID CLASSIFIER            │
                              │      (XGBoost / MLP - Future)       │
                              │                                     │
                              │   Output: Category 1/2/3            │
                              │           + Confidence Score        │
                              └─────────────────────────────────────┘
```

### 2.2 Dual Feature Extraction Approach

The system uses a hybrid approach combining:

1. **MOMENT Foundation Model (AI Path)**
   - Pre-trained transformer model (385M parameters)
   - Zero-shot inference (no fine-tuning)
   - Captures: anomalies, trends, periodicity, variance changes
   - Output: 1024-dimensional semantic embeddings

2. **Rule Engine (Deterministic Path)**
   - Based on Israeli Position Paper clinical guidelines
   - Explicit clinical feature extraction
   - Output: 11 interpretable clinical features
   - Provides explainability for medical decisions

### 2.3 Feature Vector Structure (1035 dimensions)

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

---

## 3. Directory Structure

```
SentinelFetal/
│
├── .git/                           # Git repository
├── .claude/                        # Claude workspace configuration
├── .gitignore                      # Git ignore rules (65 lines)
├── .venv/ & venv/                  # Python virtual environments
├── .pytest_cache/                  # Pytest cache
│
├── data/                           # Dataset storage
│   └── ctu-chb-intrapartum-cardiotocography-database-1.0.0/
│       └── [552 CTG recordings: *.hea, *.dat, *.qrs files]
│
├── docs/                           # Documentation
│   ├── SentinelFetal_PRD.docx.txt           # Product Requirements
│   ├── SentinelFetal_TechSpec.docx.txt      # Technical Specification
│   └── SentinelFetal_Gen35_Spec.docx.txt    # Gen 3.5 Specification
│
├── models/                         # Saved/trained models (*.pt, *.pth, *.ckpt)
├── notebooks/                      # Jupyter notebooks for exploration
│
├── src/                            # Main source code (3,558 lines)
│   ├── __init__.py                 # Package init (version 0.1.0)
│   │
│   ├── data/                       # Data loading & preprocessing (821 lines)
│   │   ├── __init__.py
│   │   ├── loader.py               # CTU-UHB dataset loader (325 lines)
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
│   ├── models/                     # AI models & fusion (848 lines)
│   │   ├── __init__.py
│   │   ├── moment_encoder.py       # MOMENT foundation model (477 lines)
│   │   └── fusion.py               # Feature fusion utilities (371 lines)
│   │
│   └── visualize_preprocessing.py  # Visualization tools (251 lines)
│
├── tests/                          # Unit tests (1,533 lines)
│   ├── __init__.py
│   ├── test_preprocessing.py       # Preprocessing tests (273 lines)
│   ├── test_rules.py               # Rule engine tests (563 lines)
│   └── test_pipeline_integration.py # Integration tests (697 lines)
│
├── README.md                       # Project documentation (313 lines)
├── DEVELOPMENT_PLAN.md             # Development roadmap
├── requirements.txt                # Python dependencies (27 lines)
└── preprocessing_validation.png    # Visualization output
```

---

## 4. Module Documentation

---

### 4.1 Data Module

**Location:** `src/data/`
**Total Lines:** 821
**Purpose:** Load and preprocess CTG recordings from the CTU-UHB database

---

#### 4.1.1 loader.py (325 lines)

**Purpose:** Load CTG recordings from the CTU-UHB PhysioNet database

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

Container for a single CTG recording.

```python
@dataclass
class CTGRecord:
    record_id: str                    # Unique identifier (e.g., '1001')
    fhr1: np.ndarray                  # Primary fetal heart rate signal
    fhr2: Optional[np.ndarray]        # Secondary FHR signal (may be None)
    uc: np.ndarray                    # Uterine contractions signal
    sampling_rate: float              # Sampling frequency (4 Hz)
    duration_seconds: float           # Total recording duration
    metadata: dict[str, Any]          # Additional info (signal names, units, comments)
```

**Properties:**
| Property | Return Type | Description |
|----------|-------------|-------------|
| `timestamps` | `np.ndarray` | Array of timestamps from 0 to duration |
| `n_samples` | `int` | Number of samples in the recording |

###### CTUDataLoader (Class)

Main loader class for the CTU-UHB database.

```python
class CTUDataLoader:
    SAMPLING_RATE: int = 4  # Hz (class constant)

    def __init__(self, data_dir: Union[str, Path]) -> None
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `list_records()` | None | `list[str]` | Get sorted list of all available record IDs |
| `load_record(record_id)` | `record_id: str` | `CTGRecord` | Load single CTG record |
| `load_record_as_dataframe(record_id)` | `record_id: str` | `pd.DataFrame` | Load as pandas DataFrame |
| `iter_records(limit)` | `limit: Optional[int]` | `Generator[CTGRecord]` | Iterate over records |

**Usage Example:**
```python
from src.data.loader import CTUDataLoader

loader = CTUDataLoader("data/ctu-chb-intrapartum-cardiotocography-database-1.0.0")
records = loader.list_records()  # ['1001', '1002', ...]

record = loader.load_record("1001")
print(f"Duration: {record.duration_seconds / 60:.1f} minutes")
print(f"Mean FHR: {np.nanmean(record.fhr1):.1f} bpm")
```

---

#### 4.1.2 preprocess.py (490 lines)

**Purpose:** CTG signal preprocessing implementing the 10-second rule per Israeli Position Paper

##### Exceptions

```python
class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""

class InvalidSignalError(PreprocessingError):
    """Raised when input signal is invalid."""
```

##### Classes

###### PreprocessingConfig (Dataclass)

Configuration parameters for CTG preprocessing.

```python
@dataclass
class PreprocessingConfig:
    sampling_rate: float = 4.0        # Signal sampling frequency (Hz)
    fhr_min: float = 50.0             # Minimum valid FHR (BPM)
    fhr_max: float = 240.0            # Maximum valid FHR (BPM)
    max_gap_seconds: float = 10.0     # Maximum gap duration to fill (10-second rule)
    smoothing_window: int = 5         # Window size for median filter (samples)
    spike_threshold: float = 30.0     # Maximum valid BPM change between samples
```

**Properties:**
| Property | Return Type | Description |
|----------|-------------|-------------|
| `max_gap_samples` | `int` | Maximum gap size in samples (10s × 4Hz = 40) |

###### PreprocessingResult (Dataclass)

Result container for CTG signal preprocessing.

```python
@dataclass
class PreprocessingResult:
    processed_signal: np.ndarray      # Cleaned and gap-filled FHR signal
    original_signal: np.ndarray       # Original input signal (unchanged)
    nan_mask: np.ndarray              # Boolean mask - True = original invalid values
    filled_mask: np.ndarray           # Boolean mask - True = gaps that were filled
    unfilled_mask: np.ndarray         # Boolean mask - True = gaps that remain (too large)
    stats: dict                       # Preprocessing statistics
```

**Stats Dictionary Contents:**
| Key | Type | Description |
|-----|------|-------------|
| `total_samples` | `int` | Total number of samples |
| `duration_seconds` | `float` | Signal duration in seconds |
| `invalid_samples` | `int` | Count of originally invalid samples |
| `invalid_percent` | `float` | Percentage of invalid samples |
| `filled_samples` | `int` | Count of samples that were filled |
| `filled_percent` | `float` | Percentage of filled samples |
| `remaining_nan` | `int` | Count of remaining NaN values |
| `remaining_nan_percent` | `float` | Percentage of remaining NaN |
| `mean_fhr` | `float` | Mean FHR after processing |
| `std_fhr` | `float` | Standard deviation of FHR |

###### CTGPreprocessor (Class)

Main preprocessor class implementing the preprocessing pipeline.

```python
class CTGPreprocessor:
    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `process(fhr, apply_smoothing)` | `fhr: np.ndarray, apply_smoothing: bool = False` | `PreprocessingResult` | Process FHR through full pipeline |

**Internal Methods:**
| Method | Description |
|--------|-------------|
| `_mark_out_of_range(signal)` | Mark values outside 50-240 BPM as invalid |
| `_detect_spikes(signal)` | Detect >30 BPM changes (noise artifacts) |
| `_fill_gaps(signal)` | Apply 10-second rule for gap filling |
| `_find_gaps(nan_mask)` | Identify contiguous NaN regions |
| `_interpolate_gap(signal, start, end)` | Linear interpolation for gaps |
| `_apply_median_filter(signal)` | Smoothing (preserves edges) |
| `_calculate_stats(...)` | Generate preprocessing statistics |

**Processing Pipeline:**
```
1. Mark out-of-range values as NaN (< 50 or > 240 BPM, including 0)
2. Detect spikes (>30 BPM changes between consecutive samples)
3. Detect gaps (contiguous NaN regions)
4. Fill small gaps (≤10 seconds) with linear interpolation
5. Leave large gaps (>10 seconds) as NaN
6. Optional: Apply median filter smoothing
```

##### Helper Function

```python
def preprocess_fhr(
    fhr: np.ndarray,
    config: Optional[PreprocessingConfig] = None
) -> np.ndarray:
    """Convenience wrapper for quick preprocessing."""
```

**Usage Example:**
```python
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig

config = PreprocessingConfig(sampling_rate=4.0, max_gap_seconds=10.0)
preprocessor = CTGPreprocessor(config)
result = preprocessor.process(fhr_signal)

print(f"Filled {result.stats['filled_percent']:.1f}% of gaps")
print(f"Remaining NaN: {result.stats['remaining_nan_percent']:.1f}%")
clean_fhr = result.processed_signal
```

---

### 4.2 Rules Module

**Location:** `src/rules/`
**Total Lines:** 1,538
**Purpose:** Clinical rule engine implementing Israeli Position Paper guidelines

---

#### 4.2.1 baseline.py (241 lines)

**Purpose:** Calculate baseline FHR per Israeli Position Paper

**Definition:** Baseline FHR is the mean FHR rounded to increments of 5 bpm during a 10-minute segment, excluding periodic changes and periods of marked variability.

##### Exception

```python
class BaselineCalculationError(Exception):
    """Raised when baseline calculation fails."""
```

##### Classes

###### BaselineResult (Dataclass)

```python
@dataclass
class BaselineResult:
    value: float                      # Calculated baseline in bpm (rounded to nearest 5)
    is_normal: bool                   # True if 110-160 bpm
    is_bradycardia: bool              # True if < 110 bpm
    is_tachycardia: bool              # True if > 160 bpm
    stable_segment_found: bool        # True if stable segment was found
    segment_start_idx: Optional[int]  # Start index of stable segment
    segment_variability: Optional[float]  # Variability in stable segment
    confidence: float                 # Confidence score (0-1)
```

##### Main Function

```python
def calculate_baseline(
    fhr: np.ndarray,
    sampling_rate: float = 4.0,
    window_minutes: float = 2.0,
    variability_threshold: float = 25.0,
    step_seconds: float = 10.0
) -> BaselineResult:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fhr` | `np.ndarray` | - | FHR signal array in bpm |
| `sampling_rate` | `float` | 4.0 | Sampling frequency in Hz |
| `window_minutes` | `float` | 2.0 | Minimum stable segment duration |
| `variability_threshold` | `float` | 25.0 | Maximum variability for stable segment |
| `step_seconds` | `float` | 10.0 | Step size for sliding window |

**Algorithm:**
1. Slide a 2-minute window across the signal (10-second steps)
2. For each window, calculate variability (max - min)
3. Find the most stable segment (lowest variability < 25 bpm)
4. Calculate mean FHR in that segment
5. Round to nearest multiple of 5
6. Fallback to global mean if no stable segment found

**Clinical Reference:**
| Classification | Range |
|----------------|-------|
| Normal | 110-160 bpm |
| Bradycardia | < 110 bpm |
| Tachycardia | > 160 bpm |

---

#### 4.2.2 variability.py (283 lines)

**Purpose:** Analyze FHR variability per Israeli Position Paper

**Definition:** Variability is the fluctuation in baseline FHR, quantified as the amplitude range (max - min) within 1-minute windows.

##### Exception

```python
class VariabilityCalculationError(Exception):
    """Raised when variability calculation fails."""
```

##### Enumeration

```python
class VariabilityCategory(Enum):
    ABSENT = "Absent"      # 0-2 bpm (severe - associated with fetal acidemia)
    MINIMAL = "Minimal"    # 3-5 bpm (concerning - requires monitoring)
    MODERATE = "Moderate"  # 6-25 bpm (NORMAL - intact autonomic nervous system)
    MARKED = "Marked"      # > 25 bpm (elevated - may indicate hypoxia/infection)
    UNKNOWN = "Unknown"    # Unable to calculate
```

##### Classes

###### VariabilityResult (Dataclass)

```python
@dataclass
class VariabilityResult:
    value: float                      # Average variability amplitude (bpm)
    category: VariabilityCategory     # Classification
    is_normal: bool                   # True if MODERATE
    is_concerning: bool               # True if ABSENT or MINIMAL
    window_values: list[float]        # Per-window variability values
    min_value: float                  # Minimum variability across windows
    max_value: float                  # Maximum variability across windows
    n_windows: int                    # Number of valid 1-minute windows
```

##### Main Function

```python
def calculate_variability(
    fhr: np.ndarray,
    sampling_rate: float = 4.0,
    window_seconds: float = 60.0,
    overlap_ratio: float = 0.5,
    min_valid_ratio: float = 0.5
) -> VariabilityResult:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fhr` | `np.ndarray` | - | FHR signal array in bpm |
| `sampling_rate` | `float` | 4.0 | Sampling frequency in Hz |
| `window_seconds` | `float` | 60.0 | Window duration (1 minute) |
| `overlap_ratio` | `float` | 0.5 | Overlap between windows (50%) |
| `min_valid_ratio` | `float` | 0.5 | Minimum valid data per window |

**Algorithm:**
1. Divide signal into 1-minute windows (50% overlap)
2. For each window, calculate amplitude (max - min)
3. Require at least 50% valid data per window
4. Average the amplitudes across all windows
5. Classify according to categories

##### Helper Function

```python
def classify_variability(value: float) -> VariabilityCategory:
    """Classify a variability value into a category."""
```

---

#### 4.2.3 decelerations.py (474 lines)

**Purpose:** Detect and classify FHR decelerations

**Definition:** A deceleration is a decrease in FHR of ≥15 bpm below baseline lasting ≥15 seconds but less than 10 minutes.

##### Exception

```python
class DecelerationDetectionError(Exception):
    """Raised when deceleration detection fails."""
```

##### Enumeration

```python
class DecelerationType(Enum):
    EARLY = "Early"            # With contraction, symmetric (benign)
    LATE = "Late"              # After contraction, lag >15s (concerning)
    VARIABLE = "Variable"      # Abrupt onset, variable timing (cord compression)
    PROLONGED = "Prolonged"    # Duration > 2 minutes (severe stress)
    UNCLASSIFIED = "Unclassified"  # Cannot determine type
```

##### Classes

###### Deceleration (Dataclass)

```python
@dataclass
class Deceleration:
    start_idx: int                    # Start index of deceleration
    end_idx: int                      # End index (exclusive)
    nadir_idx: int                    # Index of minimum FHR (nadir)
    depth: float                      # Depth below baseline (bpm)
    duration_seconds: float           # Duration of event
    decel_type: DecelerationType      # Classification
    lag_seconds: float                # Time from contraction peak to nadir
    has_severity_signs: bool          # True if severity signs present
    severity_signs: list[str]         # List of detected severity signs
    descent_rate: float               # Rate of FHR decline (bpm/sample)
    nadir_value: float                # FHR value at nadir (bpm)
```

##### Main Function

```python
def detect_decelerations(
    fhr: np.ndarray,
    uc: np.ndarray,
    baseline: float,
    sampling_rate: float = 4.0,
    min_depth: float = 15.0,
    min_duration_seconds: float = 15.0,
    max_duration_seconds: float = 600.0
) -> list[Deceleration]:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fhr` | `np.ndarray` | - | FHR signal array |
| `uc` | `np.ndarray` | - | Uterine contraction signal |
| `baseline` | `float` | - | Baseline FHR value |
| `sampling_rate` | `float` | 4.0 | Sampling frequency |
| `min_depth` | `float` | 15.0 | Minimum depth below baseline |
| `min_duration_seconds` | `float` | 15.0 | Minimum duration |
| `max_duration_seconds` | `float` | 600.0 | Maximum duration (10 min) |

##### Classification Function

```python
def classify_deceleration(
    fhr: np.ndarray,
    uc: np.ndarray,
    nadir_idx: int,
    decel_start: int,
    decel_end: int,
    sampling_rate: float = 4.0
) -> tuple[DecelerationType, float, float]:  # (type, lag_seconds, descent_rate)
```

**Classification Algorithm (based on lag time):**
| Lag Time | Descent Rate | Classification |
|----------|--------------|----------------|
| < 5 seconds | - | Early |
| > 15 seconds | - | Late |
| - | > 0.5 bpm/sample | Variable |
| Duration > 2 min | - | Prolonged |

##### Severity Signs (for Variable Decelerations)

```python
def _check_severity_signs(
    fhr: np.ndarray,
    start: int,
    end: int,
    nadir: int,
    baseline: float,
    sampling_rate: float
) -> tuple[bool, list[str]]:
```

**5 Severity Signs Detected:**
1. **Severe Drop:** FHR < 70 bpm for > 60 seconds
2. **Absent Variability:** Internal variability < 5 bpm within deceleration
3. **Slow Recovery:** > 60 seconds to return to baseline
4. **Overshoot:** Rise > 10 bpm above baseline after recovery
5. **W-Shape:** Biphasic pattern (double dip)

##### Helper Function

```python
def count_recurrent_decelerations(
    decelerations: list[Deceleration],
    decel_type: DecelerationType,
    total_contractions: int,
    threshold_ratio: float = 0.5
) -> tuple[int, bool]:  # (count, is_recurrent)
```

**Definition:** Recurrent = occurs with ≥50% of contractions

---

#### 4.2.4 tachysystole.py (202 lines)

**Purpose:** Detect excessive uterine activity

**Definition:** Tachysystole = > 5 contractions per 10-minute window, averaged over 30 minutes

##### Exception

```python
class TachysystoleDetectionError(Exception):
    """Raised when tachysystole detection fails."""
```

##### Classes

###### TachysystoleResult (Dataclass)

```python
@dataclass
class TachysystoleResult:
    detected: bool                    # True if tachysystole present
    contractions_per_10min: float     # Average contraction rate
    total_contractions: int           # Count in analysis window
    analysis_duration_minutes: float  # Duration analyzed
    contraction_indices: list[int]    # Peak indices
    confidence: float                 # Detection confidence (0-1)
```

##### Main Function

```python
def detect_tachysystole(
    uc: np.ndarray,
    sampling_rate: float = 4.0,
    analysis_window_minutes: float = 30.0,
    threshold_per_10min: float = 5.0,
    min_contraction_distance_seconds: float = 60.0
) -> TachysystoleResult:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `uc` | `np.ndarray` | - | Uterine contraction signal |
| `sampling_rate` | `float` | 4.0 | Sampling frequency |
| `analysis_window_minutes` | `float` | 30.0 | Duration to analyze |
| `threshold_per_10min` | `float` | 5.0 | Threshold for detection |
| `min_contraction_distance_seconds` | `float` | 60.0 | Min time between contractions |

**Algorithm:**
1. Take last 30 minutes of UC signal
2. Peak detection with 75th percentile threshold
3. Require minimum 60 seconds between contractions
4. Calculate contractions per 10 minutes
5. If > 5 contractions/10min → Tachysystole detected

##### Helper Function

```python
def count_contractions(
    uc: np.ndarray,
    sampling_rate: float = 4.0,
    min_distance_seconds: float = 60.0
) -> int:
```

---

#### 4.2.5 sinusoidal.py (338 lines)

**Purpose:** Detect sinusoidal pattern - **SEVERE FINDING**

**Definition:** A smooth, sine wave-like oscillating pattern with:
- Frequency: 3-5 cycles per minute (0.05-0.083 Hz)
- Amplitude: 5-15 bpm
- Duration: > 20 minutes
- Absent short-term variability (smooth waves)

**Clinical Significance:**
- **ALWAYS Category 3 (Pathological)**
- Associated with fetal anemia, severe hypoxia
- May indicate imminent fetal death
- **REQUIRES IMMEDIATE ACTION**

##### Exception

```python
class SinusoidalDetectionError(Exception):
    """Raised when sinusoidal pattern detection fails."""
```

##### Classes

###### SinusoidalResult (Dataclass)

```python
@dataclass
class SinusoidalResult:
    detected: bool                            # True = SEVERE FINDING
    confidence: float                         # Detection confidence (0-1)
    dominant_frequency: Optional[float]       # Frequency in Hz
    frequency_cycles_per_min: Optional[float] # Frequency in cycles/minute
    amplitude: float                          # Oscillation amplitude (bpm)
    amplitude_in_range: bool                  # True if 5-15 bpm
    dominance_ratio: float                    # Power ratio
    analysis_duration_minutes: float          # Duration analyzed
```

##### Main Function

```python
def detect_sinusoidal_pattern(
    fhr: np.ndarray,
    sampling_rate: float = 4.0,
    min_duration_minutes: float = 20.0,
    freq_min_cycles_per_min: float = 3.0,
    freq_max_cycles_per_min: float = 5.0,
    amp_min: float = 5.0,
    amp_max: float = 15.0,
    dominance_threshold: float = 0.3
) -> SinusoidalResult:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fhr` | `np.ndarray` | - | FHR signal array |
| `sampling_rate` | `float` | 4.0 | Sampling frequency |
| `min_duration_minutes` | `float` | 20.0 | Minimum duration required |
| `freq_min_cycles_per_min` | `float` | 3.0 | Minimum frequency |
| `freq_max_cycles_per_min` | `float` | 5.0 | Maximum frequency |
| `amp_min` | `float` | 5.0 | Minimum amplitude |
| `amp_max` | `float` | 15.0 | Maximum amplitude |
| `dominance_threshold` | `float` | 0.3 | Min power ratio in target range |

**Algorithm:**
1. Require minimum 20 minutes of signal
2. Perform FFT to analyze frequency content
3. Look for dominant peak in 3-5 cycles/min range
4. Check if amplitude is in 5-15 bpm range
5. If both criteria met → **Sinusoidal pattern detected**

##### Test Signal Generator

```python
def generate_sinusoidal_test_signal(
    duration_minutes: float = 25.0,
    sampling_rate: float = 4.0,
    baseline: float = 140.0,
    amplitude: float = 10.0,
    cycles_per_minute: float = 4.0
) -> np.ndarray:
```

---

### 4.3 Models Module

**Location:** `src/models/`
**Total Lines:** 848
**Purpose:** AI models and feature fusion

---

#### 4.3.1 moment_encoder.py (477 lines)

**Purpose:** MOMENT foundation model integration for embedding extraction

##### Technical Specifications

| Specification | Value |
|---------------|-------|
| Model | AutonLab/MOMENT-1-large |
| Parameters | 385 million |
| Embedding Dimension | 1024 |
| Patch Size | 64 samples (16 seconds @ 4Hz) |
| Maximum Input | 512 patches = 8,192 samples (~34 min) |
| VRAM Requirements | ~2GB (inference only) |
| Inference Time | ~100-200ms per 10-min window (CPU) |
| Mode | Zero-Shot (no fine-tuning) |

##### Exception

```python
class MomentEncoderError(Exception):
    """Raised when MOMENT encoding fails."""
```

##### Classes

###### EmbeddingResult (Dataclass)

```python
@dataclass
class EmbeddingResult:
    embedding: np.ndarray             # 1024-dimensional embedding
    start_idx: int                    # Start index in original signal
    end_idx: int                      # End index
    start_time_sec: float             # Start time in seconds
    end_time_sec: float               # End time in seconds
    is_mock: bool = False             # True if mock mode (MOMENT unavailable)
```

###### MomentFeatureExtractor (Class)

```python
class MomentFeatureExtractor:
    EMBEDDING_DIM: int = 1024
    DEFAULT_WINDOW_SIZE: int = 2400   # 10 minutes @ 4Hz

    def __init__(
        self,
        device: Optional[str] = None,  # 'cpu', 'cuda', 'mps', or auto-detect
        use_mock: bool = False          # Force mock mode for testing
    ) -> None
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `extract(fhr, window_size)` | `fhr: np.ndarray, window_size: int = None` | `np.ndarray (1024,)` | Extract embedding from FHR window |
| `is_available()` | None | `bool` | Check if real MOMENT model available |

**Internal Methods:**
| Method | Description |
|--------|-------------|
| `_load_model()` | Load MOMENT from HuggingFace |
| `_prepare_signal(fhr, window_size)` | Pad/truncate, handle NaN, normalize |
| `_model_extract(fhr)` | Extract using real MOMENT model |
| `_mock_extract(fhr)` | Generate deterministic mock embedding |

##### Sliding Window Function

```python
def extract_embeddings_sliding_window(
    fhr: np.ndarray,
    extractor: MomentFeatureExtractor,
    sampling_rate: float = 4.0,
    window_minutes: float = 10.0,
    step_minutes: float = 1.0
) -> List[EmbeddingResult]:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fhr` | `np.ndarray` | - | Complete FHR signal |
| `extractor` | `MomentFeatureExtractor` | - | Extractor instance |
| `sampling_rate` | `float` | 4.0 | Sampling frequency |
| `window_minutes` | `float` | 10.0 | Window size |
| `step_minutes` | `float` | 1.0 | Step size |

##### Device Info Function

```python
def get_device_info() -> dict:
    """Get information about available compute devices."""
    # Returns: torch_available, moment_available, default_device,
    #          cuda_available, mps_available, cuda_device_count, cuda_device_name
```

**Mock Mode Behavior:**
- Generates random 1024-dim embeddings for testing
- Embeddings are deterministic based on signal statistics
- Similar signals produce similar embeddings
- Allows pipeline testing without MOMENT dependencies

---

#### 4.3.2 fusion.py (371 lines)

**Purpose:** Fuse MOMENT embeddings with Rule Engine features

##### Constants

```python
FEATURE_VECTOR_DIM = 1035   # Total feature dimensions
EMBEDDING_DIM = 1024        # MOMENT embedding dimensions

FEATURE_INDICES = {
    'embedding': (0, 1024),
    'baseline': 1024,
    'variability_value': 1025,
    'variability_onehot': (1026, 1030),
    'late_decel_count': 1030,
    'variable_decel_count': 1031,
    'recurrent_decels': 1032,
    'tachysystole': 1033,
    'sinusoidal': 1034,
}
```

##### Classes

###### FeatureVector (Dataclass)

```python
@dataclass
class FeatureVector:
    vector: np.ndarray                # 1035-dimensional feature vector
    start_idx: int = 0                # Start index in original signal
    end_idx: int = 0                  # End index
    start_time_sec: float = 0.0       # Start time
    end_time_sec: float = 0.0         # End time

    # Component values for interpretability
    baseline: float = 0.0
    variability_value: float = 0.0
    variability_category: str = "Unknown"
    late_decel_count: int = 0
    variable_decel_count: int = 0
    recurrent_decels: bool = False
    tachysystole: bool = False
    sinusoidal: bool = False
```

**Methods:**
| Method | Returns | Description |
|--------|---------|-------------|
| `get_rule_features()` | `np.ndarray (11,)` | Extract only rule-based features |
| `get_embedding()` | `np.ndarray (1024,)` | Extract only MOMENT embedding |

##### Main Function

```python
def build_feature_vector(
    embedding: np.ndarray,
    baseline: Union[float, BaselineResult],
    variability: Union[dict, VariabilityResult],
    decelerations: List[Deceleration],
    tachysystole: Union[dict, TachysystoleResult],
    sinusoidal: Union[dict, SinusoidalResult],
    total_contractions: int = 0,
    start_idx: int = 0,
    end_idx: int = 0,
    start_time_sec: float = 0.0,
    end_time_sec: float = 0.0,
) -> FeatureVector:
```

##### Helper Functions

```python
def build_feature_matrix(feature_vectors: List[FeatureVector]) -> np.ndarray:
    """Stack multiple feature vectors into matrix (N, 1035)."""

def get_feature_names() -> List[str]:
    """Get human-readable names for all 1035 features."""
```

---

### 4.4 Visualization Module

**Location:** `src/visualize_preprocessing.py`
**Lines:** 251
**Purpose:** Create visual validation of preprocessing

##### Functions

```python
def create_synthetic_demo_data() -> np.ndarray:
    """Generate synthetic 10-minute CTG signal for demonstration."""

def visualize_preprocessing(
    original: np.ndarray,
    result: PreprocessingResult,
    output_path: str = "preprocessing_validation.png"
) -> None:
    """Create comparison visualization of raw vs processed signals."""
```

**Visualization Features:**
- Marks filled gaps (green)
- Marks unfilled gaps (red)
- Shows out-of-range regions
- Side-by-side raw vs processed comparison
- Generates matplotlib/plotly visualizations

---

## 5. Data Flow & Pipeline

### 5.1 Complete Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          PHASE 1: DATA LOADING                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Raw CTG File (*.hea, *.dat)                                            │
│         │                                                                │
│         v                                                                │
│  ┌─────────────────┐                                                     │
│  │  CTUDataLoader  │ ──> CTGRecord(fhr1, fhr2, uc, metadata)            │
│  └─────────────────┘                                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌──────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: PREPROCESSING                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Raw FHR Signal                                                          │
│         │                                                                │
│         v                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      CTGPreprocessor                             │    │
│  │  1. Mark out-of-range (< 50 or > 240 BPM) ─────────────> NaN    │    │
│  │  2. Detect spikes (> 30 BPM change) ───────────────────> NaN    │    │
│  │  3. Find gaps (contiguous NaN regions)                          │    │
│  │  4. Fill small gaps (≤ 10s) ──────────────> Linear interpolation│    │
│  │  5. Leave large gaps (> 10s) ─────────────> NaN (signal loss)   │    │
│  │  6. Optional: Median filter smoothing                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         v                                                                │
│  PreprocessingResult(processed_signal, masks, stats)                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           v                        v                        v
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ PHASE 3A: RULE ENGINE│  │ PHASE 3B: MOMENT     │  │  Uterine Contractions│
├──────────────────────┤  ├──────────────────────┤  │        (UC)          │
│                      │  │                      │  └──────────┬───────────┘
│ calculate_baseline() │  │ MomentFeatureExtractor│             │
│         │            │  │         │            │             │
│         v            │  │         v            │             │
│ BaselineResult       │  │ 1024-dim Embedding   │             │
│ (value, is_normal,   │  │                      │             │
│  bradycardia,        │  └──────────────────────┘             │
│  tachycardia)        │                                       │
│                      │                                       │
│ calculate_variability│                                       │
│         │            │                                       │
│         v            │                                       │
│ VariabilityResult    │                                       │
│ (value, category)    │                                       │
│                      │                                       │
│ detect_decelerations │<──────────────────────────────────────┘
│         │            │  (requires UC signal)
│         v            │
│ List[Deceleration]   │
│                      │
│ detect_tachysystole  │<──────────────────────────────────────┘
│         │            │  (requires UC signal)
│         v            │
│ TachysystoleResult   │
│                      │
│detect_sinusoidal_pattern
│         │            │
│         v            │
│ SinusoidalResult     │
│ (SEVERE if detected) │
│                      │
└──────────────────────┘
           │                        │
           └────────────┬───────────┘
                        v
┌──────────────────────────────────────────────────────────────────────────┐
│                       PHASE 4: FEATURE FUSION                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    build_feature_vector()                        │    │
│  │                                                                  │    │
│  │  MOMENT Embedding [0-1023]    +    Rule Features [1024-1034]    │    │
│  │       1024 dims                        11 dims                   │    │
│  │                                                                  │    │
│  │  ═══════════════════════════════════════════════════════════    │    │
│  │                                                                  │    │
│  │  [emb_0...emb_1023 | base | var | OH×4 | late | var | rec | tach | sin]│
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         v                                                                │
│  FeatureVector (1035 dimensions)                                         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌──────────────────────────────────────────────────────────────────────────┐
│                      PHASE 5: CLASSIFICATION (Future)                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  FeatureVector (1035 dims)                                               │
│         │                                                                │
│         v                                                                │
│  ┌─────────────────────┐                                                 │
│  │  Hybrid Classifier  │                                                 │
│  │  (XGBoost / MLP)    │                                                 │
│  └─────────────────────┘                                                 │
│         │                                                                │
│         v                                                                │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │                    OUTPUT                                │            │
│  │                                                          │            │
│  │  Category: 1 (Normal) / 2 (Intermediate) / 3 (Pathological)          │
│  │  Confidence: 0.0 - 1.0                                   │            │
│  │  Explanation: Rule-based features for interpretability   │            │
│  │                                                          │            │
│  └─────────────────────────────────────────────────────────┘            │
│                                                                          │
│  ⚠️ OVERRIDE: If sinusoidal.detected == True → Always Category 3        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Usage Example (Complete Pipeline)

```python
from src.data.loader import CTUDataLoader
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig
from src.rules import (
    calculate_baseline,
    calculate_variability,
    detect_decelerations,
    detect_tachysystole,
    detect_sinusoidal_pattern
)
from src.models.moment_encoder import MomentFeatureExtractor
from src.models.fusion import build_feature_vector

# 1. Load CTG record
loader = CTUDataLoader("data/ctu-chb-...")
record = loader.load_record("1001")

# 2. Preprocess signal
config = PreprocessingConfig(sampling_rate=4.0, max_gap_seconds=10.0)
preprocessor = CTGPreprocessor(config)
result = preprocessor.process(record.fhr1)
fhr = result.processed_signal

# 3. Rule Engine analysis
baseline_result = calculate_baseline(fhr, sampling_rate=4.0)
variability_result = calculate_variability(fhr, sampling_rate=4.0)
decelerations = detect_decelerations(fhr, record.uc, baseline_result.value)
tachysystole_result = detect_tachysystole(record.uc)
sinusoidal_result = detect_sinusoidal_pattern(fhr)

# 4. MOMENT embedding
extractor = MomentFeatureExtractor(device='cuda')
embedding = extractor.extract(fhr[:2400])  # 10-minute window

# 5. Feature fusion
feature_vec = build_feature_vector(
    embedding=embedding,
    baseline=baseline_result,
    variability=variability_result,
    decelerations=decelerations,
    tachysystole=tachysystole_result,
    sinusoidal=sinusoidal_result,
    total_contractions=tachysystole_result.total_contractions
)

print(f"Feature vector shape: {feature_vec.vector.shape}")  # (1035,)

# Check for critical findings
if sinusoidal_result.detected:
    print("⚠️ SEVERE: Sinusoidal pattern - CATEGORY 3!")
```

---

## 6. Clinical Rules Reference

### 6.1 Category Classification Matrix

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

### 6.2 Baseline FHR

| Classification | Range | Clinical Significance |
|----------------|-------|----------------------|
| Normal | 110-160 bpm | Healthy fetal status |
| Bradycardia | < 110 bpm | May indicate hypoxia, cord compression |
| Tachycardia | > 160 bpm | May indicate infection, maternal fever |

**Calculation Method:**
- Find most stable 2-minute segment (variability < 25 bpm)
- Calculate mean FHR in that segment
- Round to nearest multiple of 5

### 6.3 Variability Categories

| Category | Range (bpm) | Clinical Interpretation |
|----------|-------------|------------------------|
| **Absent** | 0-2 | SEVERE - Associated with fetal acidemia |
| **Minimal** | 3-5 | Concerning - Requires close monitoring |
| **Moderate** | 6-25 | NORMAL - Intact autonomic nervous system |
| **Marked** | > 25 | Elevated - May indicate hypoxia or infection |

**Clinical Note:** Moderate variability is the **single most reliable indicator** of fetal well-being.

### 6.4 Decelerations

#### Types and Timing

| Type | Lag Time | Descent Rate | Clinical Significance |
|------|----------|--------------|----------------------|
| Early | < 5 sec | Gradual | Benign (head compression) |
| Late | > 15 sec | Gradual | Concerning (uteroplacental insufficiency) |
| Variable | Any | > 0.5 bpm/sample | Cord compression |
| Prolonged | N/A | N/A | Duration > 2 minutes |

#### Criteria
- **Definition:** ≥15 bpm below baseline for 15 seconds to 10 minutes
- **Recurrent:** Occurs with ≥50% of contractions

#### Variable Deceleration Severity Signs
1. Drop to < 70 bpm for > 60 seconds
2. Absent variability within deceleration
3. Slow recovery (> 60 seconds)
4. Overshoot (> 10 bpm above baseline after recovery)
5. W-shape (biphasic pattern)

### 6.5 Tachysystole

| Parameter | Value |
|-----------|-------|
| Definition | > 5 contractions per 10 minutes |
| Analysis Window | 30 minutes |
| Clinical Action | Consider tocolytics, position change |

### 6.6 Sinusoidal Pattern

**⚠️ SEVERE FINDING - Always Category 3**

| Parameter | Criteria |
|-----------|----------|
| Frequency | 3-5 cycles per minute |
| Amplitude | 5-15 bpm |
| Duration | > 20 minutes |
| Variability | Absent (smooth waves) |

**Associated Conditions:**
- Fetal anemia (Rh isoimmunization, fetomaternal hemorrhage)
- Severe hypoxia
- May indicate imminent fetal death

**Action Required:** Immediate intervention regardless of other parameters

---

## 7. Testing Documentation

### 7.1 Test Coverage Summary

| Module | Test File | Test Count | Coverage |
|--------|-----------|------------|----------|
| Preprocessing | `test_preprocessing.py` | 4 | Gap filling, 10-second rule |
| Baseline | `test_rules.py` | 6 | Calculation, classification |
| Variability | `test_rules.py` | 5 | All categories |
| Decelerations | `test_rules.py` | 5 | Types, severity |
| Tachysystole | `test_rules.py` | 3 | Detection, counting |
| Sinusoidal | `test_rules.py` | 5 | FFT analysis, thresholds |
| Integration | `test_pipeline_integration.py` | 2+ | End-to-end flow |
| **Total** | | **26+** | |

### 7.2 Test Files

#### test_preprocessing.py (273 lines)

```python
class TestGapFilling:
    def test_5_second_gap_is_filled()      # Gap ≤10s filled
    def test_15_second_gap_remains_nan()   # Gap >10s remains NaN
    def test_exactly_10_second_gap_is_filled()  # Boundary condition
    def test_11_second_gap_remains_nan()   # Just over limit
```

#### test_rules.py (563 lines)

```python
class TestBaseline:
    def test_flat_signal_baseline()        # Known baseline
    def test_baseline_rounding()           # Round to nearest 5
    def test_bradycardia_detection()       # < 110 bpm
    def test_tachycardia_detection()       # > 160 bpm
    def test_short_signal_fallback()       # Uses global mean
    def test_confidence_calculation()       # Confidence scoring

class TestVariability:
    def test_absent_variability()          # 0-2 bpm
    def test_minimal_variability()         # 3-5 bpm
    def test_moderate_variability()        # 6-25 bpm (normal)
    def test_marked_variability()          # > 25 bpm
    def test_short_signal()                # Edge case

class TestDecelerations:
    def test_early_deceleration()          # < 5 sec lag
    def test_late_deceleration()           # > 15 sec lag
    def test_variable_deceleration()       # Abrupt onset
    def test_severity_signs_detection()    # 5 severity signs
    def test_recurrent_calculation()       # ≥50% of contractions

class TestTachysystole:
    def test_tachysystole_detected()       # >5/10min
    def test_normal_contraction_rate()     # ≤5/10min
    def test_duration_handling()           # Short signals

class TestSinusoidalPattern:
    def test_sinusoidal_detection()        # Synthetic pattern
    def test_frequency_range()             # 3-5 cycles/min
    def test_amplitude_range()             # 5-15 bpm
    def test_duration_requirement()        # >20 minutes
    def test_normal_signal_not_detected()  # No false positives
```

#### test_pipeline_integration.py (697 lines)

```python
class TestDataLoading:
    def test_load_record()
    def test_list_records()

class TestPreprocessing:
    def test_preprocessing_pipeline()

class TestRuleEngine:
    def test_complete_rule_analysis()

class TestMomentEncoder:
    def test_embedding_extraction()        # Mock mode
    def test_embedding_dimensions()        # 1024 dims

class TestFeatureFusion:
    def test_feature_vector_creation()     # 1035 dims

class TestPipelineFlow:
    def test_end_to_end_pipeline()
```

### 7.3 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_rules.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test class
pytest tests/test_rules.py::TestBaseline -v

# Run specific test
pytest tests/test_rules.py::TestBaseline::test_bradycardia_detection -v
```

---

## 8. Configuration & Dependencies

### 8.1 Python Requirements

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

# Phase 3+ (Optional)
# torch>=2.0.0             # PyTorch for MOMENT
# momentfm>=0.1.0          # MOMENT foundation model
# xgboost>=1.7.0           # Gradient boosting classifier
# tensorflow>=2.12.0       # Alternative neural networks
```

### 8.2 Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install MOMENT dependencies
pip install torch momentfm
```

### 8.3 Data Directory Configuration

Expected structure:
```
data/
└── ctu-chb-intrapartum-cardiotocography-database-1.0.0/
    └── ctu-chb-intrapartum-cardiotocography-database-1.0.0/
        ├── 1001.hea    # Header files
        ├── 1001.dat    # Signal data
        ├── 1001.qrs    # QRS annotations
        ├── 1002.hea
        ├── 1002.dat
        ...
```

**Dataset Download:**
```bash
wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/
```

---

## 9. Development Status

### 9.1 Completed Phases

#### Phase 1: Data Pipeline ✅
- [x] CTU-UHB data loader
- [x] Signal preprocessing with 10-second rule
- [x] Gap detection and filling
- [x] Spike detection and removal
- [x] Unit tests and validation

#### Phase 2: Rule Engine ✅
- [x] Baseline FHR calculator
- [x] Variability analyzer (4 categories)
- [x] Deceleration detector and classifier
- [x] Tachysystole detector
- [x] Sinusoidal pattern detector
- [x] 26 unit tests

#### Phase 3: MOMENT Integration ✅ (Framework Complete)
- [x] MomentFeatureExtractor class
- [x] Mock mode for testing
- [x] Sliding window extraction
- [x] Feature fusion module
- [x] 1035-dim feature vector construction

### 9.2 In Progress

#### Phase 4: Hybrid Classifier 🔄
- [ ] XGBoost implementation
- [ ] MLP alternative
- [ ] Training pipeline
- [ ] Category 1/2/3 classification
- [ ] Confidence scoring
- [ ] Model evaluation metrics

### 9.3 Future Work

#### Phase 5: Real-time System ⏳
- [ ] Streaming data handler
- [ ] Real-time inference pipeline
- [ ] Alert system
- [ ] Dashboard UI
- [ ] Clinical integration

#### Additional Enhancements ⏳
- [ ] Fine-tuning on CTU-UHB labels
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

# Preprocessing
CTGPreprocessor(config: Optional[PreprocessingConfig])
preprocessor.process(fhr: np.ndarray, apply_smoothing: bool = False) -> PreprocessingResult

# Rule Engine
calculate_baseline(fhr, sampling_rate=4.0, window_minutes=2.0) -> BaselineResult
calculate_variability(fhr, sampling_rate=4.0, window_seconds=60.0) -> VariabilityResult
detect_decelerations(fhr, uc, baseline, sampling_rate=4.0) -> list[Deceleration]
detect_tachysystole(uc, sampling_rate=4.0, analysis_window_minutes=30.0) -> TachysystoleResult
detect_sinusoidal_pattern(fhr, sampling_rate=4.0, min_duration_minutes=20.0) -> SinusoidalResult

# MOMENT Encoder
MomentFeatureExtractor(device=None, use_mock=False)
extractor.extract(fhr: np.ndarray, window_size: int = None) -> np.ndarray

# Feature Fusion
build_feature_vector(embedding, baseline, variability, decelerations,
                     tachysystole, sinusoidal, total_contractions) -> FeatureVector
```

### Clinical Thresholds

| Parameter | Normal | Abnormal |
|-----------|--------|----------|
| Baseline | 110-160 bpm | < 110 or > 160 |
| Variability | 6-25 bpm | < 6 or > 25 |
| Decelerations | None | Any late/variable |
| Contractions | ≤ 5/10min | > 5/10min |
| Sinusoidal | Not present | **CATEGORY 3** |

---

**End of Documentation**

*Generated: January 2026*
*SentinelFetal Gen3.5 - Hybrid AI System for Fetal Monitoring*
