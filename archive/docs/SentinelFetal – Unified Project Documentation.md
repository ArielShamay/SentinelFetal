# SentinelFetal – Unified Project Documentation

## 1. Project Overview

SentinelFetal is a real-time fetal distress detection system for Cardiotocography (CTG) signals that combines:

- **Rule-based engine** aligned with the Israeli Position Paper (baseline, variability, decelerations, tachysystole, sinusoidal patterns)
- **Foundation-model path** using MOMENT embeddings (1024-dim) fused with rule features (11 dims) into a 1035-dim vector for XGBoost classification (Categories 1/2/3)
- **Safety-net medical override** that enforces critical clinical findings regardless of ML output
- **Streamlit dashboards** for visualization, alerting (Hebrew explanations), and demo flows
- **Real-time simulation system** for training scenarios with 8 concurrent simulated patients and event injection capabilities

### Key Metrics

| Metric | Value |
|--------|-------|
| Training Samples | 3,240 |
| Records Used | 50 CTU-UHB recordings |
| Feature Dimensions | 1,035 (1024 MOMENT + 11 rules) |
| Cross-validation | 3-fold Stratified K-Fold |
| F1 Score | 1.00 (demo data) |
| Sampling Rate | 4 Hz |
| Output Language | Hebrew (XAI explanations) |

---

## 2. Change Log / Summary of Updates

- **Modular Architecture Refactoring (Jan 2026)**: Complete refactoring to Dependency Injection (DI) pattern using Python Protocols.
  - **New Interfaces Module**: `src/interfaces/` defining contracts for all components (Protocol-based).
  - **Adapters Layer**: `src/adapters/` wrapping existing logic to conform to interfaces.
  - **Pipeline Container**: `src/pipeline/container.py` for centrally managing dependencies.
  - **Modular Pipeline**: `src/pipeline/analysis_pipeline.py` orchestrating analysis via injected components.
  - **Zero Logic Changes**: Core clinical and ML algorithms remain untouched, only the structure changed.
- **Real-time simulation system**: Complete `src/simulation/` module with orchestrator, patient generators, FHR/UC signal synthesis, event injection, and ring buffer management
- **Simulation dashboard**: New `src/ui/simulation_app.py` with 8-patient real-time monitoring
- **Pipeline adapter**: Bridges simulation to existing Gen3.5 analysis pipeline
- **Event type system**: Comprehensive event definitions for decelerations, baseline changes, variability changes, sinusoidal patterns, and tachysystole
- **Event logging**: Lightweight audit trail without storing raw signals
- **New test suite**: `test_simulation_core.py` and `test_simulation_integration.py`
- **New documentation**: Real-time simulator PRD and specifications
- **Centralized configuration**: `src/config.py` with CTG, THRESHOLDS, COLORS, MODEL, PATHS, HEBREW dataclasses
- **Utils module**: `src/utils/signal_utils.py` with reusable signal helpers
- **Clean exports**: Updated `__init__.py` files for all modules
- **Model naming fix**: `models/__init__.py` exports `XGBClassifierWrapper`
- **Clinical thresholds**: Aligned to Position Paper; baseline/tachycardia/bradycardia in config

---

## 3. Directory and File Structure

```
SentinelFetal/
├── data/                                          # Dataset storage
│   ├── ctu-chb-intrapartum.../                   # CTU-UHB database (552 recordings)
│   └── processed/                                 # Preprocessed data
│       ├── X.npy                                  # Feature matrix (3240 × 1035)
│       └── y.npy                                  # Label vector
│
├── docs/                                          # Documentation
│   └── SentinelFetal – Unified Project Documentation.md  # This file
│
├── models/                                        # Saved trained models
│   ├── xgb_demo.json                             # XGBoost classifier
│   └── xgb_demo.config.json                      # Model configuration
│
├── notebooks/                                     # Jupyter notebooks
│
├── scripts/                                       # Executable scripts
│   ├── run_simulation.py                         # Launch simulation dashboard
│   ├── visualize_preprocessing.py                # Data preprocessing visualization
│   └── verify_system.py                          # System health verification script
│   └── README.md                                 # Script usage notes
│
├── src/                                           # Main source code
│   ├── __init__.py
│   ├── config.py                                 # Centralized configuration
│   │
│   ├── interfaces/                               # [NEW] Protocol definitions
│   │   ├── __init__.py
│   │   ├── protocols.py                          # Interface contracts
│   │   └── types.py                              # Shared type definitions
│   │
│   ├── adapters/                                 # [NEW] Implementation wrappers
│   │   ├── __init__.py
│   │   ├── data_adapters.py                      # Data layer adapters
│   │   ├── rule_adapters.py                      # Clinical rule adapters
│   │   ├── model_adapters.py                     # ML model adapters
│   │   └── analysis_adapters.py                  # Analysis components adapters
│   │
│   ├── pipeline/                                 # [NEW] Analysis orchestration
│   │   ├── __init__.py
│   │   ├── container.py                          # Dependency injection container
│   │   └── analysis_pipeline.py                  # Main analysis workflow
│   │
│   ├── analysis/                                 # Analysis & alert generation
│   │   ├── __init__.py
│   │   ├── alerts.py                             # Alert generation with Hebrew XAI
│   │   └── override.py                           # Medical safety override logic
│   │
│   ├── data/                                     # Data pipeline
│   │   ├── __init__.py
│   │   ├── loader.py                             # CTU-UHB dataset loader
│   │   └── preprocess.py                         # CTG signal preprocessing
│   │
│   ├── models/                                   # ML models & feature extraction
│   │   ├── __init__.py
│   │   ├── classifier.py                         # XGBoost classifier wrapper
│   │   ├── fusion.py                             # Feature fusion (MOMENT + rules)
│   │   └── moment_encoder.py                     # MOMENT foundation model integration
│   │
│   ├── rules/                                    # Clinical rule engine
│   │   ├── __init__.py
│   │   ├── baseline.py                           # Baseline FHR calculation
│   │   ├── variability.py                        # Variability analysis
│   │   ├── decelerations.py                      # Deceleration detection
│   │   ├── tachysystole.py                       # Tachysystole detection
│   │   └── sinusoidal.py                         # Sinusoidal pattern detection
│   │
│   ├── simulation/                               # Real-time simulation system
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py                   # Main simulation controller
│   │   │   └── ring_buffer.py                    # Circular buffer for streaming
│   │   ├── generators/
│   │   │   ├── __init__.py
│   │   │   ├── patient_generator.py              # Synthetic patient data
│   │   │   ├── fhr_generator.py                  # FHR signal synthesis
│   │   │   └── uc_generator.py                   # Uterine contraction generation
│   │   ├── events/
│   │   │   ├── __init__.py
│   │   │   └── event_types.py                    # Event definitions
│   │   ├── processing/
│   │   │   ├── __init__.py
│   │   │   └── pipeline_adapter.py               # Bridge to analysis pipeline
│   │   └── logging/
│   │       ├── __init__.py
│   │       └── event_logger.py                   # Event recording
│   │
│   ├── training/                                 # Model training & data prep
│   │   ├── __init__.py
│   │   ├── prepare_data.py                       # Dataset preparation
│   │   └── train_demo.py                         # Training script
│   │
│   ├── ui/                                       # User interfaces
│   │   ├── __init__.py
│   │   ├── app.py                                # Main Streamlit dashboard
│   │   ├── simulation_app.py                     # Real-time simulation dashboard
│   │   └── plots.py                              # Plotting utilities
│   │
│   └── utils/                                    # Utility functions
│       ├── __init__.py
│       └── signal_utils.py                       # Signal processing helpers
│
├── tests/                                         # Test suite
│   ├── __init__.py
│   ├── test_modular_pipeline.py                  # [NEW] Modular architecture tests
│   ├── test_preprocessing.py                     # Preprocessing tests
│   ├── test_rules.py                             # Rule engine tests
│   ├── test_phase4.py                            # Phase 4 integration tests
│   ├── test_pipeline_integration.py              # End-to-end pipeline tests
│   ├── test_simulation_core.py                   # Simulation core tests
│   ├── test_simulation_integration.py            # Simulation integration tests
│   └── test_ui_components.py                     # UI component tests
│
├── .gitignore
├── requirements.txt                               # Python dependencies
├── README.md                                      # Project overview
├── DEMO_GUIDE.md                                  # Demo instructions (Hebrew)
├── DEVELOPMENT_PLAN.md                            # Development roadmap
└── SentinelFetal – Unified Project Documentation.md  # This file
```

---

## 4. Architecture Overview

### 4.1 Hybrid Pipeline Architecture (Conceptual)

```
┌─────────────┐
│  CTG Signal │
│ (FHR + UC)  │
└──────┬──────┘
       │
       ├──────────────────┬───────────────────┐
       │                  │                   │
       v                  v                   v
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│Preprocessing│   │ MOMENT      │   │ Rule Engine │
│ - Gap Fill  │   │ Embeddings  │   │ - Baseline  │
│ - Spike Det │   │ (1024-dim)  │   │ - Variability│
│ - 10s Rule  │   │             │   │ - Decels    │
└──────┬──────┘   └──────┬──────┘   │ - Tachysyst │
       │                  │          │ - Sinusoidal│
       │                  │          └──────┬──────┘
       │                  │                 │
       └──────────────────┴─────────────────┘
                          │
                          v
                   ┌─────────────┐
                   │ Feature     │
                   │ Fusion      │
                   │ (1035-dim)  │
                   └──────┬──────┘
                          │
                          v
                   ┌─────────────┐
                   │  Hybrid     │
                   │  Classifier │
                   │  (XGBoost)  │
                   └──────┬──────┘
                          │
                          v
         ┌────────────────────────────┐
         │   Medical Override Safety  │
         │   (Can override ML pred)   │
         └──────┬─────────────────────┘
                │
                v
         ┌─────────────────┐
         │  Alert Engine   │
         │  (Hebrew XAI)   │
         └─────────────────┘
```

### 4.2 Modular Software Architecture (Implementation)

The system is implemented using a **Dependency Injection (DI)** pattern defined in `src/pipeline/container.py`.

```
┌───────────────────────────────────────────────┐
│              Pipeline Container               │
│ (Manages dependencies & life-cycle)           │
└───────────────────────┬───────────────────────┘
                        │ Injects
                        v
┌───────────────────────────────────────────────┐
│               AnalysisPipeline                │
│ (Orchestrates flow via Protocols)             │
└────┬──────────────┬───────────────┬───────┬───┘
     │              │               │       │
     │     Uses     │     Uses      │ Uses  │
     v              v               v       v
┌─────────┐   ┌────────────┐   ┌────────┐ ┌──────┐
│IPrepro- │   │IRule-      │   │IModel- │ │...   │
│cessor   │   │Calculators │   │Adapters│ │      │
└────▲────┘   └──────▲─────┘   └───▲────┘ └──────┘
     │               │             │
     │ Implements    │ Implements  │ Implements
     │               │             │
┌────┴────┐   ┌──────┴─────┐   ┌───┴────┐
│Adapter  │   │Adapter     │   │Adapter │
│(Wrapper)│   │(Wrapper)   │   │(Wrap)  │
└────┬────┘   └──────┬─────┘   └───┬────┘
     │ Calls         │ Calls       │ Calls
     v               v             v
┌─────────┐   ┌────────────┐   ┌────────┐
│Legacy   │   │Legacy Rules│   │Legacy  │
│Preproc  │   │(func/class)│   │Models  │
└─────────┘   └────────────┘   └────────┘
```

### 4.3 Layer Responsibilities

| Layer | Module | Purpose |
|-------|--------|---------|
| **Interfaces** | `interfaces/*` | **Protocol definitions** (Contracts) for all system components |
| **Adapters** | `adapters/*` | **Wrappers** that adapt legacy logic to new Protocols |
| **Pipeline** | `pipeline/*` | **Container** & **Orchestrator** using DI to run analysis |
| Data | `loader.py` | Reads CTU-UHB records, returns `CTGRecord` with FHR/UC arrays |
| Preprocessing | `preprocess.py` | Implementation of signal cleaning logic |
| Rule Engine | `rules/*` | Implementation of clinical algorithms |
| ML Core | `models/*` | Implementation of MOMENT and XGBoost logic |
| UI | `ui/*` | Streamlit dashboards (consumers of the pipeline) |


---

## 5. Detailed Component Documentation

### 5.1 Configuration Module (`src/config.py`)

Centralized configuration for the entire application with frozen dataclasses.

#### Classes

| Class | Purpose |
|-------|---------|
| `CTGConfig` | CTG signal processing constants |
| `ClinicalThresholds` | Clinical decision thresholds per Israeli Position Paper |
| `UIColors` | Color scheme for UI components |
| `ModelConfig` | XGBoost and ML model configuration |
| `DataPaths` | Default data paths |
| `HebrewStrings` | Hebrew explanatory strings for alerts |

#### Exports

```python
from src.config import CTG, THRESHOLDS, COLORS, MODEL, PATHS, HEBREW
```

#### Key Constants

**CTGConfig:**
- `SAMPLING_RATE`: 4.0 Hz
- `MOMENT_WINDOW_MINUTES`: 10.0 (2400 samples)
- `MOMENT_EMBEDDING_DIM`: 1024
- `RULE_FEATURES_DIM`: 11
- `TOTAL_FEATURES_DIM`: 1035

**ClinicalThresholds:**
- `PH_PATHOLOGICAL`: < 7.15 → Category 3
- `PH_INTERMEDIATE`: 7.15 - 7.20 → Category 2
- `BASELINE_NORMAL_MIN`: 110 bpm
- `BASELINE_NORMAL_MAX`: 160 bpm
- `VARIABILITY_ABSENT_MAX`: 2.0 bpm
- `VARIABILITY_MODERATE_MIN`: 6.0 bpm
- `VARIABILITY_MODERATE_MAX`: 25.0 bpm

---

### 5.2 Data Pipeline Modules (`src/data/`)

#### 5.2.1 `loader.py` - Dataset Loading

**Classes:**

- **`CTGRecord`** - Container for a single CTG recording
  - `record_id`: Unique identifier
  - `fhr1`, `fhr2`: FHR signals (numpy arrays)
  - `uc`: Uterine contractions signal
  - `sampling_rate`: 4 Hz
  - Properties: `fhr` (alias for fhr1), `timestamps`, `n_samples`, `duration_seconds`

- **`CTUDataLoader`** - Loads data from CTU-UHB database
  - `list_records()`: List all available record IDs
  - `load_record(record_id)`: Load single recording
  - `iter_records(limit)`: Iterator over records
  - `extract_ph(record_id)`: Extract pH from header
  - `get_outcome_label(record_id)`: Get category label (0/1/2) based on pH

**Errors:** `DataLoaderError`, `RecordNotFoundError`, `InvalidRecordError`

#### 5.2.2 `preprocess.py` - Signal Preprocessing

**Classes:**

- **`PreprocessingConfig`** - Configuration parameters
  - `sampling_rate`: 4.0 Hz
  - `fhr_min`: 50 bpm, `fhr_max`: 240 bpm
  - `max_gap_seconds`: 10 (10-second rule)
  - `spike_threshold`: 30 bpm
  - `smoothing_window`: 5 samples

- **`PreprocessingResult`** - Output container
  - `processed_signal`: Cleaned FHR
  - `original_signal`: Original input
  - `nan_mask`, `filled_mask`, `unfilled_mask`: Boolean masks
  - `stats`: Dictionary with statistics

- **`CTGPreprocessor`** - Main preprocessing class
  - `process(fhr, apply_smoothing=False)`: Run preprocessing pipeline

**Pipeline Steps:**
1. Mark out-of-range values (50-240 bpm) as NaN
2. Detect and remove spikes (>30 bpm changes)
3. Fill small gaps (≤10 seconds) with linear interpolation
4. Leave large gaps (>10 seconds) as NaN
5. Optional median filter smoothing

---

### 5.3 Clinical Rule Engine Modules (`src/rules/`)

#### 5.3.1 `baseline.py` - Baseline FHR Calculation

**Algorithm:**
1. Slide 2-minute window across signal (10-second steps)
2. Find segment with lowest variability (<25 bpm)
3. Calculate mean FHR in that segment
4. Round to nearest 5 bpm

**Classes:**

- **`BaselineResult`**
  - `value`: Baseline in bpm
  - `is_normal`: Within 110-160 bpm
  - `is_bradycardia`: < 110 bpm
  - `is_tachycardia`: > 160 bpm
  - `confidence`: 0-1 confidence score
  - `stable_segment_found`: Boolean

**Function:** `calculate_baseline(fhr, sampling_rate=4, window_minutes=2, variability_threshold=25, step_seconds=10)`

#### 5.3.2 `variability.py` - Variability Analysis

**Algorithm:**
1. Divide signal into 1-minute windows (50% overlap)
2. Calculate amplitude (max - min) per window
3. Average amplitudes across all windows
4. Classify into categories

**Enum: `VariabilityCategory`**
- `ABSENT`: 0-2 bpm (severe, fetal acidemia)
- `MINIMAL`: 3-5 bpm (concerning)
- `MODERATE`: 6-25 bpm (normal)
- `MARKED`: >25 bpm (may indicate hypoxia)
- `UNKNOWN`: Cannot calculate

**Classes:**

- **`VariabilityResult`**
  - `value`: Average variability in bpm
  - `category`: Classification enum
  - `is_normal`: True if MODERATE
  - `is_concerning`: True if ABSENT or MINIMAL
  - `window_values`: Per-window values

**Function:** `calculate_variability(fhr, sampling_rate=4, window_seconds=60, overlap_ratio=0.5, min_valid_ratio=0.5)`

#### 5.3.3 `decelerations.py` - Deceleration Detection

**Enum: `DecelerationType`**
- `EARLY`: With contraction (<5s lag), normal
- `LATE`: After contraction (>15s lag), uteroplacental insufficiency
- `VARIABLE`: Abrupt onset, variable timing (cord compression)
- `PROLONGED`: Duration >2 minutes
- `UNCLASSIFIED`: Cannot determine

**Deceleration Criteria:**
- Depth: ≥15 bpm below baseline
- Duration: 15 seconds to 10 minutes

**Classes:**

- **`Deceleration`** dataclass
  - `start_idx`, `end_idx`, `nadir_idx`: Positions
  - `depth`: Depth below baseline (bpm)
  - `duration_seconds`: Duration
  - `decel_type`: Classification
  - `lag_seconds`: Time to nadir from contraction
  - `has_severity_signs`: Boolean flag
  - `severity_signs`: List of warning signs

**Function:** `detect_decelerations(fhr, uc, baseline, sampling_rate=4, min_depth=15, min_duration_seconds=15, max_duration_seconds=600)`

**Severity Signs (Category 3 indicators):**
- Drops below 70 bpm
- Absent internal variability
- Slow recovery (>60 seconds)
- Overshoot (>10 bpm above baseline)
- Biphasic W-shape

#### 5.3.4 `tachysystole.py` - Tachysystole Detection

**Criteria:**
- Definition: >5 contractions per 10 minutes
- Analysis window: 30 minutes
- Minimum distance: 45 seconds between contractions

**Classes:**

- **`TachysystoleResult`**
  - `detected`: Boolean
  - `contractions_per_10min`: Rate
  - `total_contractions`: Count
  - `analysis_duration_minutes`: Window analyzed
  - `confidence`: 0-1 score

**Function:** `detect_tachysystole(uc, sampling_rate=4, analysis_window_minutes=30, threshold_per_10min=5, min_contraction_distance_seconds=60)`

#### 5.3.5 `sinusoidal.py` - Sinusoidal Pattern Detection

**Criteria (SEVERE finding - always Category 3):**
- Frequency: 2-6 cycles per minute
- Amplitude: 5-15 bpm
- Duration: ≥20 minutes
- Clinical significance: Fetal anemia, severe hypoxia

**Classes:**

- **`SinusoidalResult`**
  - `detected`: Boolean
  - `confidence`: 0-1 score
  - `dominant_frequency`: Hz
  - `frequency_cycles_per_min`: CPM
  - `amplitude`: bpm
  - `dominance_ratio`: Peak prominence

**Function:** `detect_sinusoidal_pattern(fhr, sampling_rate=4, min_duration_minutes=20, freq_min_cycles_per_min=3, freq_max_cycles_per_min=5, amp_min=5, amp_max=15, dominance_threshold=0.3)`

---

### 5.4 Model & Feature Extraction Modules (`src/models/`)

#### 5.4.1 `moment_encoder.py` - MOMENT Foundation Model

**Purpose:** Extract 1024-dimensional time series embeddings using the MOMENT-1-large model (385M parameters).

**Classes:**

- **`EmbeddingResult`**
  - `embedding`: 1024-dim numpy array
  - `window_start_idx`, `window_end_idx`: Window positions
  - `start_time_sec`, `end_time_sec`: Timestamps
  - `is_mock`: Boolean flag

- **`MomentFeatureExtractor`**
  - `__init__(device, use_mock)`: Initialize with optional mock mode
  - `extract(window)`: Extract embedding from signal window
  - `extract_embeddings_sliding_window(fhr, window_minutes=10, step_minutes=1)`: Sliding window extraction

**Mock Mode:** Generates deterministic embeddings from signal statistics when MOMENT library unavailable.

#### 5.4.2 `fusion.py` - Hybrid Feature Fusion

**Purpose:** Combine MOMENT embeddings with clinical rule features.

**Constants:**
- `FEATURE_VECTOR_DIM`: 1035
- `EMBEDDING_DIM`: 1024
- `FEATURE_INDICES`: Map of feature positions

**Feature Vector Composition (1035 dimensions):**
| Index Range | Feature |
|-------------|---------|
| 0-1023 | MOMENT embeddings |
| 1024 | Baseline value (normalized) |
| 1025 | Baseline status (-1/0/1) |
| 1026 | Variability value (normalized) |
| 1027 | Variability category (0-4) |
| 1028 | Deceleration count |
| 1029 | Has late decelerations (0/1) |
| 1030 | Has variable decelerations (0/1) |
| 1031 | Sinusoidal detected (0/1) |
| 1032 | Tachysystole detected (0/1) |
| 1033 | Maximum deceleration depth |
| 1034 | Absent variability flag (0/1) |

**Classes:**

- **`FeatureVector`** dataclass
  - `vector`: 1035-dim numpy array
  - `get_rule_features()`: Extract rule features
  - `get_embedding()`: Extract MOMENT embedding

**Functions:**
- `build_feature_vector(embedding, baseline, variability, decelerations, tachysystole, sinusoidal, ...)`
- `build_feature_matrix(feature_vectors)`: Stack vectors into ndarray

#### 5.4.3 `classifier.py` - XGBoost Classifier

**Classes:**

- **`ClassifierConfig`** - XGBoost hyperparameters
  - `n_estimators`: 100
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `cv_folds`: 3

- **`TrainingResult`**
  - `cv_scores`: Cross-validation scores
  - `mean_score`, `std_score`, `best_score`
  - `feature_importance`: Feature importances
  - `confusion_matrix`, `classification_report`

- **`XGBClassifierWrapper`**
  - `train(X, y, class_weights)`: Train with StratifiedKFold CV
  - `predict(X)`: Return predictions (0/1/2)
  - `predict_proba(X)`: Return probabilities
  - `save_model(path)`, `load_model(path)`: Persistence

**Output Classes:**
- 0: Category 1 (Normal)
- 1: Category 2 (Intermediate/Suspicious)
- 2: Category 3 (Pathological)

---

### 5.5 Analysis Modules (`src/analysis/`)

#### 5.5.1 `alerts.py` - Alert Generation

**Classes:**

- **`Alert`** dataclass
  - `category`: 1, 2, or 3
  - `confidence`: 0-1 model confidence
  - `headline`: Short description (Hebrew)
  - `explanation`: Detailed explanation (Hebrew)
  - `findings`: List of medical findings
  - `recommendations`: List of recommendations
  - `timestamp`: ISO format

**Function:** `generate_alert(category, confidence, baseline, variability, decelerations, tachysystole, sinusoidal)`

**Helpers:** `get_category_color(category)`, `get_category_emoji(category)`

#### 5.5.2 `override.py` - Medical Safety Override

**Purpose:** Safety net that can override ML prediction with medical rules.

**Enum: `OverrideReason`**
- `NONE`
- `SINUSOIDAL_PATTERN`
- `ABSENT_VARIABILITY_WITH_DECELS`
- `BRADYCARDIA`
- `ABSENT_VARIABILITY_SAFETY_FLOOR`

**Classes:**

- **`MedicalOverride`** dataclass
  - `should_override`: Boolean
  - `final_category`: 0-indexed category
  - `reason`: OverrideReason enum
  - `ml_prediction`: Original prediction
  - `explanation`: Text description

**Override Rules:**
1. **Sinusoidal** → Force Category 3
2. **Absent variability + (recurrent late/variable decels OR bradycardia)** → Force Category 3
3. **ML says Normal (0) but variability Absent** → Force Category 2 (safety floor)

**Function:** `apply_medical_override(ml_prediction, baseline, variability, decelerations, tachysystole, sinusoidal)`

---

### 5.6 Real-Time Simulation Module (`src/simulation/`)

The simulation module provides a complete real-time CTG simulation system for training scenarios.

#### 5.6.1 Core Components (`src/simulation/core/`)

##### `orchestrator.py` - Main Orchestrator

**Purpose:** Coordinates multiple patient generators and MOMENT processing.

**Classes:**

- **`OrchestratorConfig`**
  - `num_patients`: Number of simulated patients (default: 8)
  - `sampling_rate`: 4.0 Hz
  - `tick_interval_seconds`: 1.0 second
  - `moment_interval_seconds`: 30.0 seconds (full cycle)
  - `patient_names`: Hebrew names list
  - `moment_per_patient_interval`: Calculated property

- **`SimulationOrchestrator`**
  - **Control Methods:**
    - `start()`: Start simulation in background thread
    - `stop()`: Stop simulation
    - `pause()`, `resume()`: Pause/resume
    - `set_speed(multiplier)`: Speed 0.5x - 2.0x
    - `reset_all()`: Reset all patients

  - **Event Injection:**
    - `inject_event(patient_id, event_type, params, duration_seconds)`
    - `inject_event_all(event_type, params, duration_seconds)`

  - **Data Access:**
    - `get_all_patients_status()`: Sorted by category (highest first)
    - `get_patient_data(patient_id, duration_minutes)`
    - `get_patient(patient_id)`: Get PatientGenerator
    - `get_simulation_time()`, `get_simulation_time_formatted()`
    - `get_statistics()`: Tick count, MOMENT processes, etc.

  - **Logging:**
    - `export_log(filepath)`: Export to CSV
    - `get_event_log()`: Get EventLogger

**Staggered Processing Strategy:**
- With 8 patients and 30s MOMENT interval: each patient processed every 3.75 seconds
- Rule engine runs every tick (~10ms)
- MOMENT processing ~100-300ms per call
- Prevents CPU spikes by spreading load

**Example:**
```python
orchestrator = SimulationOrchestrator()
orchestrator.start()
orchestrator.inject_event('P1', EventType.SINUSOIDAL_PATTERN, params)
time.sleep(10)
statuses = orchestrator.get_all_patients_status()
orchestrator.stop()
```

##### `ring_buffer.py` - Circular Buffer

**Purpose:** Fixed-size circular buffer for CTG signals (~77KB for 8 patients).

**Classes:**

- **`RingBuffer`** dataclass
  - `max_samples`: 2400 (10 min at 4Hz)
  - `sampling_rate`: 4.0 Hz

  - **Methods:**
    - `append(fhr, uc, timestamp)`: Add single sample
    - `append_batch(fhr, uc, timestamps)`: Add multiple samples
    - `get_window(duration_seconds)`: Get data dictionary
    - `get_last_n_minutes(minutes)`: Convenience method
    - `get_latest()`: Get most recent sample
    - `clear()`: Clear all data

  - **Properties:**
    - `size`, `duration_seconds`, `duration_minutes`
    - `is_full`, `is_empty`

#### 5.6.2 Generators (`src/simulation/generators/`)

##### `patient_generator.py` - Patient Data Generation

**Purpose:** Generates and manages CTG data for a single simulated patient.

**Classes:**

- **`PatientConfig`**
  - `patient_id`: Unique identifier (e.g., 'P1')
  - `bed_number`: Bed/room number
  - `name`: Patient name (Hebrew)
  - `baseline_fhr`: Initial baseline (default: 140.0)
  - `baseline_variability`: Initial variability (default: 10.0)
  - `contractions_per_10min`: Initial rate (default: 4.0)
  - `buffer_duration_minutes`: Ring buffer size (default: 10.0)

- **`PatientGenerator`**
  - **Generation:**
    - `generate_tick(n_samples=4)`: Generate one tick of data
    - Returns `{'fhr', 'uc', 'timestamps', 'contraction_peaks'}`

  - **Event Management:**
    - `inject_event(event_type, params, duration_seconds)`: Inject event
    - `remove_event(event)`: Remove specific event
    - `clear_events()`: Remove all events
    - `get_active_events()`: Thread-safe copy

  - **Data Access:**
    - `get_buffer_data(duration_minutes)`: Get buffer with metadata
    - `get_status()`: Patient status summary
    - `reset()`: Reset to initial state

  - **State:**
    - `latest_category`: Current category (1/2/3)
    - `latest_alert`: Last generated Alert
    - `latest_findings`: Last findings dictionary
    - `buffer`: Access to RingBuffer

##### `fhr_generator.py` - FHR Signal Synthesis

**Purpose:** Generates realistic FHR signals with variability and event support.

**Classes:**

- **`FHRGeneratorConfig`**
  - `baseline_fhr`: 140.0 bpm
  - `baseline_variability`: 10.0 bpm
  - `high_freq_noise_std`: 2.0 bpm
  - `min_fhr`: 50, `max_fhr`: 240 bpm

- **`FHRGenerator`**
  - `generate_samples(n_samples, active_events, contraction_peaks)`
  - Applies variability oscillations (2-6 cycles/min)
  - Applies events: decelerations, baseline shifts, sinusoidal
  - `reset()`: Reset generator state

**Event Application:**
- Late decelerations: Gaussian shape after contraction peak
- Variable decelerations: Abrupt trapezoid shape with optional severity signs
- Early decelerations: Symmetric Gaussian at contraction peak
- Prolonged decelerations: Trapezoidal >2 minutes
- Bradycardia/Tachycardia: Baseline shift with ramp
- Sinusoidal: Pure sine wave replacing normal variability

##### `uc_generator.py` - Uterine Contraction Synthesis

**Purpose:** Generates realistic uterine contraction signals.

**Classes:**

- **`UCGeneratorConfig`**
  - `contractions_per_10min`: 4.0
  - `contraction_duration_sec`: 60.0
  - `contraction_amplitude`: 80 (0-100 scale)
  - `baseline_tonus`: 10
  - `noise_std`: 3.0

- **`UCGenerator`**
  - `generate_samples(n_samples, active_events)`: Generate UC + peak markers
  - Returns `(uc_array, contraction_peaks_bool_array)`
  - `get_recent_peaks(duration_seconds)`: Recent contraction peaks
  - `get_contraction_rate()`: Contractions per 10 min
  - `reset()`: Reset generator state

**Contraction Model:**
- Gaussian-shaped contractions
- Variable intervals with randomness
- Tachysystole event support

#### 5.6.3 Events (`src/simulation/events/`)

##### `event_types.py` - Event Definitions

**Enum: `EventType`**
- Decelerations: `LATE_DECELERATION`, `VARIABLE_DECELERATION`, `PROLONGED_DECELERATION`, `EARLY_DECELERATION`
- Baseline: `BRADYCARDIA`, `TACHYCARDIA`
- Variability: `ABSENT_VARIABILITY`, `MINIMAL_VARIABILITY`, `MARKED_VARIABILITY`
- Patterns: `SINUSOIDAL_PATTERN`
- Uterine: `TACHYSYSTOLE`

**Enum: `EventSeverity`**
- `MILD`, `MODERATE`, `SEVERE`

**Parameter Classes:**

| Class | Key Parameters |
|-------|----------------|
| `LateDecelerationParams` | depth_bpm, lag_seconds, recovery_seconds, recurrence_rate |
| `VariableDecelerationParams` | depth_bpm, duration_decel_seconds, has_shoulders, severity signs |
| `ProlongedDecelerationParams` | depth_bpm, duration_seconds (>2min) |
| `EarlyDecelerationParams` | depth_bpm |
| `BradycardiaParams` | target_fhr, onset_type |
| `TachycardiaParams` | target_fhr |
| `VariabilityParams` | target_variability_bpm |
| `SinusoidalParams` | frequency_cycles_per_min, amplitude_bpm |
| `TachysystoleParams` | contractions_per_10min |

Each has factory methods: `.mild()`, `.moderate()`, `.severe()`, `.typical()`

**Classes:**

- **`InjectedEvent`** dataclass
  - `event_type`, `params`, `patient_id`
  - `start_time`, `end_time`, `is_active`
  - Properties: `remaining_seconds`, `duration_seconds`
  - Methods: `is_expired(time)`, `progress(time)`, `to_dict()`

#### 5.6.4 Processing (`src/simulation/processing/`)

##### `pipeline_adapter.py` - Analysis Pipeline Bridge

**Purpose:** Bridges simulation to existing Gen3.5 analysis pipeline.

**Classes:**

- **`PipelineAdapterConfig`**
  - `use_real_moment`: True (CRITICAL: Use real MOMENT model)
  - `model_path`: "models/xgb_demo.json"
  - `sampling_rate`: 4.0 Hz
  - `min_data_seconds`: 60.0

- **`PipelineAdapter`**
  - `process_patient(patient_id, data, run_moment=True)`: Run full pipeline
  - `clear_cache(patient_id)`: Clear embedding cache
  - `get_stats()`: Processing statistics
  - `is_moment_real`: Check if using real MOMENT

**Pipeline Steps:**
1. **Preprocessing**: Spike removal, gap filling, smoothing
2. **Rule Engine**: Baseline, variability, decelerations, etc.
3. **MOMENT Embedding**: Real model or cached
4. **Feature Fusion**: Build 1035-dim vector
5. **Classification**: XGBoost prediction
6. **Medical Override**: Safety net
7. **Alert Generation**: Hebrew explanations

**Returns:**
```python
{
    'category': 1-3,
    'alert': Alert object,
    'findings': {...},
    'confidence': 0.0-1.0,
    'ml_prediction': 1-3,
    'was_overridden': bool,
    'insufficient_data': bool
}
```

#### 5.6.5 Logging (`src/simulation/logging/`)

##### `event_logger.py` - Event Recording

**Purpose:** Lightweight logging without storing raw signals (~100KB max).

**Classes:**

- **`LogEntry`** dataclass
  - `timestamp`: Real-world datetime
  - `simulation_time`: Simulation seconds
  - `event_type`: 'INJECTION', 'ALERT', etc.
  - `patient_id`: Patient identifier
  - `details`: Additional details dict

- **`EventLogger`**
  - `log_injection(event)`: Log event injection
  - `log_alert(patient_id, results)`: Log Category 2/3 alert
  - `log_custom(event_type, patient_id, simulation_time, details)`
  - `get_entries(event_type, patient_id, limit)`: Query entries
  - `get_alerts()`, `get_injections()`: Filtered queries
  - `export_csv(filepath)`, `export_json(filepath)`: Export
  - `get_summary()`: Statistics summary
  - `clear()`: Clear all entries

---

### 5.7 Training Modules (`src/training/`)

#### 5.7.1 `prepare_data.py` - Dataset Preparation

**Purpose:** Prepare training dataset from raw CTU-UHB records.

**Classes:**

- **`DatasetConfig`**
  - `limit`: Number of records to process
  - `use_mock`: Use mock MOMENT
  - `window_minutes`, `step_minutes`: Sliding window params

**Functions:**
- `run_rule_engine(fhr, uc, sampling_rate)`: Run all rule modules
- `process_record(record_id, ...)`: Process single record
- `prepare_dataset(config)`: Orchestrate full preparation

**CLI:**
```bash
python src/training/prepare_data.py --limit 50 --use_mock False
```

**Output:**
- `X.npy`: Feature matrix (N × 1035)
- `y.npy`: Labels (N,) with values 0/1/2

#### 5.7.2 `train_demo.py` - Training Script

**Purpose:** Train XGBoost classifier on prepared data.

**Process:**
1. Load `X.npy`, `y.npy`
2. Train with StratifiedKFold CV
3. Print CV scores
4. Save model to `models/xgb_demo.json`

**CLI:**
```bash
python src/training/train_demo.py
```

---

### 5.8 User Interface Modules (`src/ui/`)

#### 5.8.1 `app.py` - Main Streamlit Dashboard

**Purpose:** Visualize CTG records and display AI analysis.

**Features:**
- Patient list sidebar (select from available records)
- CTG visualization (FHR + UC)
- Real-time analysis display
- Alert display in Hebrew
- Medical findings panel
- Recommendations panel

**Caching:** Uses `@st.cache_resource` for loader, encoder, classifier.

**Run:**
```bash
streamlit run src/ui/app.py
```

#### 5.8.2 `simulation_app.py` - Real-Time Simulation Dashboard

**Purpose:** Demonstrate real-time simulation with 8 patients.

**Features:**
- **Control Panel:** Start/Stop/Pause/Reset buttons, speed slider (0.5x-2x)
- **Status Display:** Simulation time, tick count, MOMENT process count
- **Event Injection:** Patient selection, event type, severity, duration
- **Patient Overview:** 2×4 grid of patient cards with category colors
- **Patient Detail:** Full CTG plot, findings metrics, alert display
- **Event Log:** Recent injections and alerts

**UI Components:**
- `render_header()`: Application header with styling
- `render_control_panel(orchestrator)`: Control buttons
- `render_event_injection(orchestrator)`: Event injection form
- `render_patient_overview(orchestrator)`: Patient grid
- `render_patient_card(status, orchestrator)`: Individual patient card
- `render_patient_detail(orchestrator)`: Selected patient view
- `render_findings_panel(patient)`: Findings metrics
- `render_alert_panel(patient)`: Alert display
- `render_event_log(orchestrator)`: Event log

**Run:**
```bash
streamlit run src/ui/simulation_app.py
# or
python scripts/run_simulation.py
```

#### 5.8.3 `plots.py` - Visualization Utilities

**Functions:**
- `create_ctg_plot(fhr, uc, decelerations, sinusoidal, sampling_rate, colors)`: CTG with highlights
- `create_category_indicator(category, confidence, colors)`: Category gauge

**Features:**
- Plotly interactive charts
- Deceleration region highlights
- Normal range bands (110-160 bpm)
- Reference lines

---

### 5.9 Utilities Module (`src/utils/`)

#### `signal_utils.py` - Signal Processing Helpers

**Functions:**
- `get_valid_values(arr)`: Extract non-NaN values
- `get_valid_ratio(arr)`: Percentage of valid data
- `interpolate_nans(arr, max_gap)`: Fill NaN gaps
- `safe_nanmean(arr, default)`: Mean ignoring NaN
- `safe_nanstd(arr, default)`: Std ignoring NaN
- `safe_nanmedian(arr, default)`: Median ignoring NaN
- `detect_spikes(arr, threshold)`: Boolean spike mask
- `apply_median_filter(arr, window)`: Median smoothing
- `calculate_signal_quality(arr, sampling_rate)`: Quality metric
- `chunk_signal(arr, window, step)`: Sliding window generator

---

### 5.10 Interfaces & Protocols (`src/interfaces/`)

Defines the contract for all system components using Python's `typing.Protocol`.

**Key Protocols:**
- `IDataLoader`: Data loading contract
- `IPreprocessor`: Signal cleaning contract
- `IBaselineCalculator`, `IVariabilityCalculator`, etc.: Rule engine contracts
- `IFeatureExtractor`: MOMENT embedding contract
- `IClassifier`: machine learning model contract
- `IAlertGenerator`: Alert generation contract

This layer ensures that components are loosely coupled and easily swappable (e.g., replacing XGBoost with Random Forest requires only implementing `IClassifier`).

### 5.11 Adapters (`src/adapters/`)

Wrappers that adapt existing implementation classes/functions to the new Protocols.

**Modules:**
- `rule_adapters.py`: Wraps functional rule logic (`calculate_baseline`, etc.) into class-based adapters.
- `model_adapters.py`: Wraps `MomentFeatureExtractor` and `XGBClassifierWrapper`.
- `data_adapters.py`: Wraps `CTUDataLoader` and `CTGPreprocessor`.
- `analysis_adapters.py`: Wraps `apply_medical_override` and `generate_alert`.

### 5.12 Pipeline Orchestration (`src/pipeline/`)

The core of the modular architecture.

#### `container.py` - PipelineContainer
Dependency Injection container that holds references to all components.
- **Method**: `create_default()` instantiates the container with standard SentinelFetal adapters.
- **Usage**: Allows overriding specific components (e.g., `container.classifier = MockClassifier()`) before pipeline creation.

#### `analysis_pipeline.py` - AnalysisPipeline
Orchestrates the data flow:
1. Validates container completeness
2. Calls `preprocessor.process()`
3. Calls all rule calculators
4. Calls `feature_extractor` & `fusion`
5. Calls `classifier.predict()`
6. Calls `medical_override` & `alert_generator`
7. Returns comprehensive `AnalysisResult` dataclass

---

## 6. Test Suite (`tests/`)

| Test File | Coverage |
|-----------|----------|
| `test_modular_pipeline.py` | **New**: Pipeline container, DI, adapters, and end-to-end modular flow |
| `test_preprocessing.py` | Gap fill, spike detection, out-of-range handling |
| `test_rules.py` | 26 unit tests: baseline (6), variability (5), decelerations (5), tachysystole (3), sinusoidal (5), integration (2) |
| `test_phase4.py` | Phase 4 (model training) tests |
| `test_pipeline_integration.py` | End-to-end pipeline tests (legacy flow) |
| `test_simulation_core.py` | Orchestrator, ring buffer, patient generator |
| `test_simulation_integration.py` | Full simulation integration |
| `test_ui_components.py` | UI component tests |

**Run Tests:**
```bash
# Run all tests (including new modular tests)
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

---

## 7. Design Patterns and Relationships

### 7.1 Structural Patterns

- **Dataclasses** for structured results: `BaselineResult`, `VariabilityResult`, `Deceleration`, `TachysystoleResult`, `SinusoidalResult`, `FeatureVector`, `Alert`, `MedicalOverride`, `PreprocessingResult`, `EmbeddingResult`, `InjectedEvent`, `LogEntry`

- **Enums** for categorical logic: `VariabilityCategory`, `DecelerationType`, `OverrideReason`, `EventType`, `EventSeverity`

### 7.2 Architectural Patterns

- **Layered Architecture**: Preprocessing → Rules → Embeddings → Fusion → Classifier → Override → Alerts → UI

- **Safety Net Pattern**: Override always upgrades severity; ML cannot downgrade critical findings

- **Separation of Concerns**: Data loading vs preprocessing vs rule engine vs ML vs UI

- **Centralized Configuration**: All constants in `config.py`

- **Dependency Injection**: Sampling rate, thresholds passed as arguments

- **Observer Pattern**: Orchestrator callbacks for MOMENT processing

- **Producer-Consumer**: PatientGenerator produces data, PipelineAdapter consumes

### 7.3 Threading Model

- **Main Thread**: Streamlit UI
- **Simulation Thread**: Background daemon for tick generation
- **Thread Safety**: RLock for orchestrator, Lock for patient events
- **Staggered Processing**: Round-robin MOMENT calls to prevent CPU spikes

---

## 8. Usage Examples

### 8.1 Load and Preprocess a Record

```python
from src.data.loader import CTUDataLoader
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig

loader = CTUDataLoader("data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0")
record = loader.load_record("1001")

prep = CTGPreprocessor(PreprocessingConfig())
result = prep.process(record.fhr)
clean = result.processed_signal
```

### 8.2 Run Rule Engine

```python
from src.rules import (
    calculate_baseline,
    calculate_variability,
    detect_decelerations,
    detect_tachysystole,
    detect_sinusoidal_pattern
)

baseline = calculate_baseline(clean)
variability = calculate_variability(clean)
decels = detect_decelerations(clean, record.uc, baseline.value)
tachy = detect_tachysystole(record.uc)
sinus = detect_sinusoidal_pattern(clean)
```

### 8.3 Build Features and Predict

```python
from src.models.moment_encoder import MomentFeatureExtractor
from src.models.fusion import build_feature_vector, build_feature_matrix
from src.models.classifier import XGBClassifierWrapper

moment = MomentFeatureExtractor(use_mock=True)
embs = moment.extract_embeddings_sliding_window(clean, sampling_rate=4.0)

fvs = [build_feature_vector(e.embedding, baseline, variability, decels, tachy, sinus,
                            start_idx=e.start_idx, end_idx=e.end_idx,
                            start_time_sec=e.start_time_sec, end_time_sec=e.end_time_sec)
       for e in embs]

X = build_feature_matrix(fvs)

clf = XGBClassifierWrapper()
clf.load_model("models/xgb_demo.json")
preds = clf.predict(X)
```

### 8.4 Apply Override and Generate Alert

```python
from src.analysis.override import apply_medical_override
from src.analysis.alerts import generate_alert

override = apply_medical_override(
    ml_prediction=int(preds[0]),
    baseline=baseline,
    variability=variability,
    decelerations=decels,
    tachysystole=tachy,
    sinusoidal=sinus
)

alert = generate_alert(
    category=override.final_category + 1,  # Convert to 1-indexed
    confidence=0.9,
    baseline=baseline,
    variability=variability,
    decelerations=decels,
    tachysystole=tachy,
    sinusoidal=sinus
)

print(alert.headline)
```

### 8.5 Run Real-Time Simulation

```python
from src.simulation import (
    SimulationOrchestrator,
    OrchestratorConfig,
    PipelineAdapter,
    EventType,
    LateDecelerationParams
)
import time

# Create adapter and orchestrator
adapter = PipelineAdapter()

def callback(patient_id, data):
    return adapter.process_patient(patient_id, data, run_moment=True)

orchestrator = SimulationOrchestrator(
    OrchestratorConfig(num_patients=8),
    processing_callback=callback
)

# Start simulation
orchestrator.start()

# Inject event
orchestrator.inject_event(
    'P1',
    EventType.LATE_DECELERATION,
    LateDecelerationParams.severe(),
    duration_seconds=300
)

# Wait and check status
time.sleep(60)
statuses = orchestrator.get_all_patients_status()
for s in statuses:
    print(f"{s['patient_id']}: Category {s['category']}")

# Stop
orchestrator.stop()
```

### 8.6 Run Dashboards

```bash
# Main dashboard
streamlit run src/ui/app.py

# Simulation dashboard
streamlit run src/ui/simulation_app.py

# Or via script
python scripts/run_simulation.py
```

### 8.7 Prepare Data and Train

```bash
# Prepare dataset
python src/training/prepare_data.py --limit 50 --use_mock False

# Train model
python src/training/train_demo.py
```

### 8.8 Run Modular Analysis Pipeline (New Standard)

```python
from src.pipeline.container import PipelineContainer
from src.pipeline.analysis_pipeline import AnalysisPipeline

# 1. Initialize Container with default adapters
container = PipelineContainer.create_default()

# 2. (Optional) Swap a component
# container.classifier = MyCustomClassifier()

# 3. Create Pipeline
pipeline = AnalysisPipeline(container)

# 4. Run Analysis
# fhr and uc are numpy arrays
result = pipeline.analyze(fhr, uc)

print(f"Category: {result.category}")
print(f"Explanation: {result.alert.explanation}")
```

---

## 9. Internal Logic (End-to-End Flow)

### 9.1 Static Analysis Flow

1. **Data Ingest**: `CTUDataLoader` reads FHR/UC signals; outcome labels derived from pH thresholds

2. **Preprocessing**: `CTGPreprocessor` enforces valid ranges, removes spikes, fills short gaps (≤10s), leaves long gaps

3. **Rule Engine**: Baseline (stable 2-min window), variability (1-min windows), decelerations (depth/duration/lag vs contractions + severity), tachysystole (contraction rate), sinusoidal (FFT signature)

4. **Embeddings**: `MomentFeatureExtractor` generates 1024-dim vectors over sliding 10-min windows

5. **Feature Fusion**: `build_feature_vector` merges embedding + normalized rule features into 1035-dim vector

6. **Classification**: `XGBClassifierWrapper` predicts per-window Category (0/1/2)

7. **Safety Override**: `apply_medical_override` upgrades to Category 3 on sinusoidal or absent variability with ominous signs

8. **Alerting**: `generate_alert` creates Hebrew headline, explanation, findings, recommendations

9. **UI**: Streamlit app renders CTG/UC plots, highlights decels, shows category indicator and alert text

### 9.2 Real-Time Simulation Flow

1. **Orchestrator Start**: Creates background thread for simulation loop

2. **Per-Tick**: Generates 4 samples (1 second at 4Hz) for all 8 patients via `PatientGenerator`

3. **PatientGenerator**: Coordinates `FHRGenerator` and `UCGenerator`, applies active events, stores in `RingBuffer`

4. **Staggered MOMENT**: Every 3.75 seconds, one patient's buffer is sent to `PipelineAdapter`

5. **PipelineAdapter**: Runs full Gen3.5 pipeline (preprocess → rules → MOMENT → fusion → classify → override → alert)

6. **Results Update**: Patient's `latest_category`, `latest_alert`, `latest_findings` updated

7. **Alert Logging**: Category 2/3 alerts logged to `EventLogger`

8. **UI Refresh**: Streamlit auto-refreshes every 1 second when running

---

## 10. Notes for Contributors

### 10.1 Code Guidelines

- Keep clinical logic aligned with config thresholds; modify only in `src/config.py`
- Use dataclass/Enum types for new structured outputs; expose via `__init__.py`
- When adding features to fusion, update `FEATURE_INDICES` and downstream consumers/tests
- Prefer mock MOMENT only for tests/demos; production pipelines should set `use_mock=False`
- Maintain tests; add synthetic fixtures in `tests/` when modifying rule logic

### 10.2 Simulation Guidelines

- Event parameters should use factory methods (`.mild()`, `.moderate()`, `.severe()`)
- Ring buffer auto-discards old data; no manual cleanup needed
- Thread safety: Always use locks when accessing patient state
- MOMENT caching: Clear cache if patient data significantly changes

### 10.3 Testing Guidelines

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_rules.py -v

# Run specific test
pytest tests/test_rules.py::test_baseline_normal -v
```

---

## 11. Dependencies

**Core Data Processing:**
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

**Data Loading:**
- wfdb >= 4.1.0

**Visualization:**
- matplotlib >= 3.7.0
- plotly >= 5.14.0

**Machine Learning:**
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0

**Web Interface:**
- streamlit >= 1.22.0

**Testing:**
- pytest >= 7.3.0
- pytest-cov >= 4.1.0

**Utilities:**
- tqdm >= 4.65.0
- python-dateutil >= 2.8.0

**Optional (for real MOMENT embeddings):**
- momentfm
- torch

---

## 12. Single Source of Truth

This document supersedes all prior documentation files:
- DEMO_GUIDE.md (demo-specific instructions remain valid)
- DEVELOPMENT_PLAN.md (roadmap information)
- SentinelFetal Project Documentation.md
- SentinelFetal – Updated Project Documentation.md

Use this file as the canonical reference for the current codebase.

---

*Last updated: January 2026*
*Version: 4.0 (Real-Time Simulator)*
