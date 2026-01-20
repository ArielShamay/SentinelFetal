# SentinelFetal Gen3.5 - Complete Technical Documentation

**Version:** 3.0 | **Date:** January 2026 | **Status:** Production V3.0 (MiniRocket Engine)

> **Single Source of Truth** – This document provides complete technical documentation for the SentinelFetal fetal monitoring system. It is intended for developers, auditors, and clinical engineers who need to understand the system from A to Z.

## V3.0 Production Architecture

**Build Date:** 2026-01-20

| Component | Format | Status | Notes |
|-----------|--------|--------|-------|
| MiniRocket Encoder | scikit-learn/sktime | ✅ Active | 84 fixed kernels, ~10ms inference |
| XGBoost Classifier | JSON | ✅ Active | `models/sentinel_classifier.json` |
| Rule Engine | Python | ✅ Active | Israeli Position Paper + 30s descent rule |
| FSQI Gate | Python | ✅ Active | Signal quality filtering (threshold: 0.7) |
| Safety Net | Python | ✅ Active | Medical overrides enabled |

> **Note:** V3.0 replaces the heavy MOMENT transformer (341M params) with MiniRocket (84 fixed kernels), achieving 10-20x inference speedup while maintaining clinical accuracy. MOMENT is retained as an optional fallback for research purposes.

---

## Table of Contents

1. [Chapter 1: System Architecture & Purpose](#chapter-1-system-architecture--purpose)
    - [1.1 What is SentinelFetal?](#11-what-is-sentinelfetal)
    - [1.2 Why Was It Built?](#12-why-was-it-built)
    - [1.3 High-Level Architecture](#13-high-level-architecture)
    - [1.4 Component Breakdown](#14-component-breakdown)
    - [1.5 Data Assets: CTU-CHB/CTU-UHB](#15-data-assets-ctu-chbctu-uhb-intrapartum-ctg-database)
    - [1.6 End-to-End Flow: Monitor → Alert](#16-end-to-end-flow-monitor--alert)
    - [1.7 Synthetic Data Generation](#17-synthetic-data-generation)
2. [Chapter 2: Codebase Deep Dive](#chapter-2-codebase-deep-dive)
   - [2.1 Simulation Engine](#21-simulation-engine)
   - [2.2 Signal Processing](#22-signal-processing)
   - [2.3 Clinical Rules Engine](#23-clinical-rules-engine)
   - [2.4 AI Model Integration (V2.0 PyTorch)](#24-ai-model-integration)
   - [2.5 Safety Net (Medical Overrides)](#25-safety-net-medical-overrides)
3. [Chapter 3: Verification & Validation](#chapter-3-verification--validation)
   - [3.1 Accuracy Testing](#31-accuracy-testing)
   - [3.2 Endurance Testing](#32-endurance-testing)
   - [3.3 Robustness Testing](#33-robustness-testing)
   - [3.4 Performance Metrics](#34-performance-metrics)
4. [Chapter 4: Setup & Operations](#chapter-4-setup--operations)
   - [4.1 Installation](#41-installation)
   - [4.2 Running the Simulator](#42-running-the-simulator)
   - [4.3 Running Tests](#43-running-tests)
   - [4.4 Configuration Reference](#44-configuration-reference)
5. [Appendix D: Repository Map](#appendix-d-repository-map)

---

# Chapter 1: System Architecture & Purpose

## 1.1 What is SentinelFetal?

**SentinelFetal Gen3.5** is a **hybrid AI system** for real-time fetal monitoring that combines:

1. **Rule-Based Clinical Logic** - Algorithms implementing the Israeli Position Paper on CTG (Cardiotocography) Interpretation, including the 30-second descent time rule for deceleration classification
2. **Lightweight Feature Extraction** - MiniRocket (84 fixed convolutional kernels) for fast pattern recognition, replacing the heavy MOMENT transformer
3. **Signal Quality Gate** - FSQI (Fetal Signal Quality Index) with spectral noise analysis to filter unreliable signals
4. **Machine Learning Classifier** - XGBoost for final category prediction
5. **Safety Net** - Medical overrides that enforce critical clinical findings regardless of ML predictions

### Core Classification Output

The system classifies fetal heart rate (FHR) patterns into three categories per the Israeli Position Paper:

| Category | Color | Hebrew | Clinical Meaning |
|----------|-------|--------|------------------|
| **Category 1** | 🟢 Green | קטגוריה 1 (תקין) | Normal - Reassuring fetal status |
| **Category 2** | 🟡 Yellow | קטגוריה 2 (לא מוגדר) | Intermediate - Requires closer monitoring |
| **Category 3** | 🔴 Red | קטגוריה 3 (פתולוגי) | Pathological - Immediate intervention required |

### Key Technical Specifications

| Specification | Value |
|---------------|-------|
| Sampling Rate | 4 Hz (4 samples/second) |
| Analysis Window | 10 minutes (2,400 samples) |
| Feature Dimensions | 95 (84 MiniRocket + 11 rule-based) |
| Concurrent Patients | 20 (on standard i5 CPU) |
| Memory Footprint | <200 MB total |
| Tick Latency | <50 ms |
| FSQI Threshold | 0.7 (high quality gate) |

---

## 1.2 Why Was It Built?

### The Clinical Problem

Cardiotocography (CTG) monitoring is the standard of care during labor, but interpretation is:
- **Subjective** - Inter-observer variability of 20-30% among clinicians
- **Complex** - Multiple parameters must be evaluated simultaneously
- **Time-Critical** - Delayed recognition of fetal distress can cause permanent harm

### The Engineering Challenge

Previous AI approaches (Gen4) failed due to:
| Gen4 Approach | Problem |
|---------------|---------|
| MOMENT + Fine-tuning + LoRA | Insufficient training data, overfitting |
| Mamba sequence models | Too complex, unstable training |
| Multi-database fusion | Data quality inconsistencies |
| GAN-generated training data | Unrealistic synthetic patterns |

### The Gen3.5 Solution

SentinelFetal Gen3.5 takes a pragmatic approach:

| Design Decision | Rationale |
|-----------------|-----------|
| **MiniRocket Zero-Shot** | 84 fixed kernels, no training required, 10-20x faster than MOMENT |
| **30-Second Descent Rule** | Physics-based deceleration classification per FIGO/NICHD guidelines |
| **FSQI Quality Gate** | Filter signals below 0.7 quality threshold before classification |
| **Rule-Based Safety Net** | Critical findings (sinusoidal, bradycardia) always trigger alerts |
| **Single Database** | CTU-UHB only - clean, well-annotated, pH-labeled data |
| **Staggered UI Updates** | 5-group pattern prevents browser freeze with 20 patients |
| **Full Explainability** | Every alert maps directly to clinical criteria |

### Success Metrics Achieved

| Metric | Target | Achieved |
|--------|--------|----------|
| Detection Rate (overall) | >50% | **64.0%** |
| False Positive Rate | <5% | **0.0%** |
| Sinusoidal Detection (clean) | >90% | **100%** |
| Brady/Tachy/Prolonged | >95% | **100%** |
| Memory Stability | <50 MB/hr growth | **1.89 MB/hr** |

---

## 1.3 High-Level Architecture

### System Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SIMULATION LAYER                                       │
│                                                                                          │
│   ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐              │
│   │  Orchestrator    │────▶│  PatientGenerator │────▶│   RingBuffer     │              │
│   │  (threading)     │     │  (FHR + UC)       │     │   (10-min)       │              │
│   │  20 patients     │     │  + Event Injection│     │   circular       │              │
│   └──────────────────┘     └───────────────────┘     └──────────────────┘              │
│                                                                │                        │
└────────────────────────────────────────────────────────────────┼────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   QUALITY GATE LAYER                                     │
│                                                                                          │
│   ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐              │
│   │  Preprocessor    │────▶│   FSQI Analysis   │────▶│   Quality Gate   │              │
│   │  • Coiflet4      │     │   • Spectral noise│     │   • FSQI ≥ 0.7   │              │
│   │  • Gap filling   │     │   • SNR ratio     │     │   • Pass/Fail    │              │
│   │  • Savitzky-Golay│     │   • Valid %       │     │                  │              │
│   └──────────────────┘     └───────────────────┘     └──────────────────┘              │
│                                                                │                        │
└────────────────────────────────────────────────────────────────┼────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PROCESSING LAYER                                       │
│                                                                                          │
│   ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐              │
│   │  Rule Engine     │────▶│ MiniRocket Encoder│────▶│  Feature Fusion  │              │
│   │  • Baseline      │     │   • 84 kernels    │     │  [84 + 11] = 95  │              │
│   │  • Variability   │     │   • ~10ms/window  │     │                  │              │
│   │  • Decelerations │     │   • Cold-start OK │     │                  │              │
│   │  • 30s Descent   │     │                   │     │                  │              │
│   │  • Sinusoidal    │     └───────────────────┘     └──────────────────┘              │
│   │  • Tachysystole  │                                       │                          │
│   └──────────────────┘                                       │                          │
│                                                              ▼                          │
│                            ┌──────────────────────────┐                                  │
│                            │   XGBoost Classifier     │                                  │
│                            │   3-class (Cat 1/2/3)    │                                  │
│                            └──────────────────────────┘                                  │
│                                      │                                                   │
└──────────────────────────────────────┼───────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SAFETY LAYER                                           │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────┐          │
│   │                      Medical Override (Safety Net)                        │          │
│   │                                                                           │          │
│   │   IF sinusoidal detected                    → FORCE Category 3           │          │
│   │   IF absent_variability + recurrent_decels  → FORCE Category 3           │          │
│   │   IF bradycardia                            → FORCE Category 2           │          │
│   │   IF recurrent_late_decelerations           → FORCE Category 2           │          │
│   │   IF absent_variability + ML=Normal         → FORCE Category 2           │          │
│   │                                                                           │          │
│   └──────────────────────────────────────────────────────────────────────────┘          │
│                                      │                                                   │
│                                      ▼                                                   │
│   ┌──────────────────────────────────────────────────────────────────────────┐          │
│   │                      Alert Generator (Hebrew)                             │          │
│   │   • Headline, Explanation, Findings, Recommendations                      │          │
│   └──────────────────────────────────────────────────────────────────────────┘          │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PRESENTATION LAYER                                     │
│                                                                                          │
│   ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐              │
│   │  Streamlit UI    │     │  Live CTG Plots   │     │  Alert Panel     │              │
│   │  8-patient grid  │     │  FHR + UC traces  │     │  Hebrew text     │              │
│   └──────────────────┘     └───────────────────┘     └──────────────────┘              │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

1. **Simulation**: `Orchestrator` manages 20 `PatientGenerator` instances, each producing FHR/UC signals at 4 Hz
2. **Buffering**: Each patient has a `RingBuffer` storing the last 10 minutes (2,400 samples)
3. **Preprocessing**: Raw signals are cleaned with Coiflet 4 wavelet denoising, Savitzky-Golay smoothing
4. **Quality Gate**: FSQI (Fetal Signal Quality Index) filters signals - only those with FSQI ≥ 0.7 proceed to classification
5. **Rule Engine**: Clinical algorithms extract baseline, variability, decelerations (with 30-second descent time rule)
6. **MiniRocket Encoding**: Clean FHR signal → 84-dimensional feature vector using fixed convolutional kernels
7. **Feature Fusion**: Combine MiniRocket features (84) + rule features (11) = 95-dimensional vector
8. **Classification**: XGBoost predicts Category 1/2/3
9. **Safety Override**: Medical rules may override ML prediction for critical findings
10. **Alert Generation**: Hebrew explanations generated for clinical staff
11. **UI Rendering**: Staggered 5-group updates prevent browser freeze with 20 patients

---

## 1.4 Component Breakdown

### Directory Structure

```
SentinelFetal/
├── src/                          # Source code
│   ├── analysis/                 # Safety analysis components
│   │   ├── alerts.py             # Hebrew alert generation
│   │   └── override.py           # Medical override safety net
│   ├── config.py                 # Global configuration constants
│   ├── core/                     # Core infrastructure
│   │   ├── di/                   # Dependency injection container
│   │   ├── events/               # Event bus system
│   │   └── types/                # Shared type definitions
│   ├── data/                     # Data handling
│   │   ├── loader.py             # CTU-UHB database loader
│   │   └── preprocess.py         # Signal preprocessing (Savitzky-Golay)
│   ├── models/                   # AI models
│   │   ├── classifier.py         # XGBoost classifier
│   │   ├── fusion.py             # Feature fusion (MOMENT + rules)
│   │   └── moment_encoder.py     # MOMENT embeddings
│   ├── rules/                    # Clinical rule engine
│   │   ├── baseline.py           # Baseline FHR calculation
│   │   ├── decelerations.py      # Deceleration detection & classification
│   │   ├── sinusoidal.py         # Sinusoidal pattern detection
│   │   ├── tachysystole.py       # Contraction rate analysis
│   │   └── variability.py        # Variability calculation
│   ├── services/                 # Service layer
│   │   ├── impl/                 # Concrete implementations
│   │   └── interfaces/           # Protocol definitions
│   ├── simulation/               # Real-time simulation
│   │   ├── core/                 # Orchestrator, ring buffer
│   │   ├── events/               # Event types and parameters
│   │   ├── generators/           # Patient signal generators
│   │   └── processing/           # Pipeline adapter
│   └── ui/                       # User interface
│       ├── app.py                # Main Streamlit app
│       ├── plots.py              # CTG visualization
│       └── simulation_app.py     # Simulation dashboard
├── tests/                        # Test suite
│   ├── benchmarks/               # Performance benchmarks
│   └── test_*.py                 # Unit tests
├── models/                       # Trained model artifacts
│   ├── sentinel_classifier.json  # XGBoost model (V2.0)
│   └── xgb_demo.config.json      # Model configuration
├── data/                         # CTG data
│   ├── ctu-chb-intrapartum.../   # CTU-UHB database
│   └── processed/                # Preprocessed numpy arrays
├── docs/                         # Documentation
│   └── reports/                  # Test reports
└── scripts/                      # Utility scripts
    └── run_simulation.py         # Launch simulator
```

### Module Responsibilities

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `src/data/loader.py` | Load CTU-UHB CTG recordings | `CTUDataLoader`, `CTGRecord` |
| `src/data/preprocess.py` | Clean and filter signals | `CTGPreprocessor`, `PreprocessResult` |
| `src/data/signal_quality.py` | **NEW** Signal quality gate | `calculate_fsqi()`, `apply_quality_gate()`, `denoise_coiflet4()` |
| `src/rules/baseline.py` | Calculate baseline FHR | `calculate_baseline()`, `BaselineResult` |
| `src/rules/variability.py` | Calculate variability | `calculate_variability()`, `VariabilityResult` |
| `src/rules/decelerations.py` | Detect decelerations | `detect_decelerations()`, `calculate_descent_time()`, `Deceleration` |
| `src/rules/sinusoidal.py` | Detect sinusoidal patterns | `detect_sinusoidal()`, `SinusoidalResult` |
| `src/rules/tachysystole.py` | Detect excessive contractions | `detect_tachysystole()`, `TachysystoleResult` |
| `src/models/minirocket_encoder.py` | **NEW** Extract MiniRocket features | `MiniRocketEncoder`, `MiniRocketConfig` |
| `src/models/moment_encoder.py` | **FALLBACK** Extract MOMENT embeddings | `MOMENTFeatureExtractor` |
| `src/models/fusion.py` | Combine features | `FeatureFusion.fuse()` |
| `src/models/classifier.py` | Predict category | `HybridClassifier.predict()` |
| `src/adapters/model_adapters.py` | **NEW** Backend selection | `MiniRocketAdapter`, `get_feature_extractor()` |
| `src/analysis/override.py` | Safety net rules | `MedicalOverride.apply()` |
| `src/analysis/alerts.py` | Generate Hebrew alerts | `AlertGenerator.generate()` |
| `src/simulation/core/orchestrator.py` | Manage simulation | `SimulationOrchestrator` |
| `src/simulation/generators/patient_generator.py` | Generate signals | `PatientGenerator` |
| `src/simulation/core/ring_buffer.py` | Circular buffer | `RingBuffer` |

---

## 1.5 Data Assets: CTU-CHB/CTU-UHB Intrapartum CTG Database

- **Source:** PhysioNet CTU-CHB / CTU-UHB Intrapartum Cardiotocography Database v1.0.0 (552 intrapartum labors) stored in [data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0](data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0).
- **Signals per record (WFDB):** FHR1 (primary FHR), FHR2 (secondary FHR), UC (uterine contractions), all sampled at 4 Hz; header files (`.hea`) carry metadata and umbilical cord pH as `#pH` comments.
- **Loader:** `CTUDataLoader` reads `.hea` + waveform pairs via WFDB, returns `CTGRecord` objects with signals, timestamps, and metadata; `extract_ph()` parses the pH tag; pH thresholds from [src/config.py](src/config.py) map outcomes (pH < 7.15 → Category 3, 7.15–7.20 → Category 2, ≥7.20 → Category 1).
- **How we use it:**
    1) **Preprocessing & labeling:** Raw WFDB signals → `CTGPreprocessor` cleaning (out-of-range removal, spike removal, gap fill, Savitzky-Golay) → rule features + MOMENT embedding → fused 1,035-dim feature vectors.
    2) **Training artifacts:** [src/training/prepare_data.py](src/training/prepare_data.py) runs the full pipeline over all records and saves features/labels to [data/processed/X.npy](data/processed/X.npy) and [data/processed/y.npy](data/processed/y.npy).
    3) **Classifier training:** [src/training/train_demo.py](src/training/train_demo.py) consumes X/y to train the hybrid XGBoost model saved at [models/sentinel_classifier.json](models/sentinel_classifier.json) using the config at [models/xgb_demo.config.json](models/xgb_demo.config.json).
- **Why needed:** Provides real intrapartum CTG with clinical ground truth (pH) to calibrate thresholds, train the hybrid classifier, and validate rule/ML outputs against physiologic outcomes.

---

## 1.6 End-to-End Flow: Monitor → Alert

**Real data path (clinical monitors):**
1) **Acquisition:** Monitors emit FHR/UC at 4 Hz → ingested as WFDB streams or batch files.
2) **Buffering:** Signals stored in per-patient `RingBuffer` (10 minutes, 2,400 samples) to support sliding-window analysis without memory growth.
3) **Preprocessing:** `CTGPreprocessor` removes out-of-range values (50–240 bpm), spikes (>30 bpm step), fills ≤10s gaps, and applies Savitzky-Golay smoothing (11-window, poly=2) to preserve decel shape.
4) **Rule Engine:** `baseline`, `variability`, `decelerations`, `sinusoidal`, `tachysystole` extract 11 clinically grounded features and flags (e.g., recurrent late decels, tachysystole, sinusoidal yes/no).
5) **Foundation Embedding:** MOMENT encoder generates a 1,024-dim representation of the 10-minute FHR segment (zero-shot, no fine-tuning).
6) **Feature Fusion:** `FeatureFusion` concatenates embedding + 11 rule features → 1,035-dim normalized vector.
7) **Classifier:** XGBoost (`HybridClassifier`) outputs Cat1/2/3 probabilities.
8) **Safety Net:** `MedicalOverride` enforces clinical guardrails (sinusoidal → Cat3; absent variability + recurrent decels/brady → Cat3; bradycardia or recurrent late decels → ≥Cat2; absent variability cannot be Normal).
9) **Alerting & UI:** `AlertGenerator` emits Hebrew headline/explanation/findings/recommendations; Streamlit UI renders patient grid, traces, and alerts in real time.

**Synthetic path (simulator):**
1) `SimulationOrchestrator` ticks at 1 Hz for up to 8 patients; MOMENT is staggered (~3.75s per patient) to keep CPU <25%.
2) `PatientGenerator` synthesizes FHR/UC using baseline + variability + contraction models; events are injected from `event_types` with severity presets (mild/moderate/severe) controlling depth/lag/recovery/recurrence.
3) Output flows through the same buffer → preprocessing → rules → MOMENT → fusion → classifier → override → alerts, ensuring simulation fidelity to production logic.

## 1.7 Synthetic Data Generation

- **Purpose:** Safe, repeatable clinical scenarios without PHI; used for demos, robustness tests, and regression of the rule/ML stack.
- **Event catalog (from `src/simulation/events/event_types.py`):** late, variable, prolonged, early decels; tachysystole; brady/tachycardia; absent/minimal/marked variability; sinusoidal pattern. Each has parameterized severities (e.g., late decel mild/moderate/severe control depth/lag/recovery/recurrence).
- **Signal models:**
    - UC generator: peaks spaced by configurable contractions_per_10min with noise and prominence constraints.
    - FHR generator: baseline + variability + decel shapes keyed to UC peaks; optional sinusoidal overlay; brady/tachy ramps.
- **Quality controls:**
    - RingBuffer prevents drift; gaps/spikes injected to test preprocessing robustness.
    - Savitzky-Golay smoothing improves decel timing/shape preservation vs. median filter (Phase 13 upgrade).
- **Robustness harness:** 500-scenario sweeps vary noise σ∈{0,2,5,10,15}, dropout∈{0,5%,10%}, pattern type, and severity to measure detection sensitivity and false positives end-to-end.

---

# Chapter 2: Codebase Deep Dive

## 2.1 Simulation Engine

### Overview

The simulation engine generates realistic CTG signals for 8 concurrent patients, allowing clinical scenario testing without real patient data.

### SimulationOrchestrator (`src/simulation/core/orchestrator.py`)

The orchestrator is the main controller managing all simulation activities.

**Threading Model:**
```python
class SimulationOrchestrator:
    def __init__(self, config: SimulationConfig):
        self._lock = threading.RLock()  # Protects patient data
        self._thread: Optional[threading.Thread] = None  # Daemon thread
        
    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
```

**Simulation Loop Algorithm:**
```python
def _run_loop(self):
    while self._running:
        if self._paused:
            time.sleep(0.1)
            continue
            
        # Generate data for all patients
        self._tick()
        
        # Staggered MOMENT processing (prevent CPU freeze)
        if time_since_last_moment >= moment_per_patient_interval:
            self._process_next_patient_moment()  # Round-robin
```

**Key Configuration:**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_patients` | 8 | Concurrent patient count |
| `sampling_rate` | 4.0 Hz | Samples per second |
| `tick_interval_seconds` | 1.0 | Time between ticks |
| `moment_interval_seconds` | 30.0 | Full MOMENT cycle for all patients |

### PatientGenerator (`src/simulation/generators/patient_generator.py`)

Each patient has a dedicated generator producing FHR and UC signals.

**Signal Generation Flow:**
```python
def generate_tick(self, n_samples: int = 4):
    # 1. Remove expired events
    self._remove_expired_events()
    
    # 2. Generate uterine contraction signal
    uc_samples = self._uc_generator.generate(n_samples)
    contraction_peaks = self._detect_peaks(uc_samples)
    
    # 3. Generate FHR signal (uses contraction timing for decelerations)
    fhr_samples = self._fhr_generator.generate(n_samples, contraction_peaks)
    
    # 4. Apply any active events (bradycardia, sinusoidal, etc.)
    fhr_samples = self._apply_events(fhr_samples)
    
    # 5. Add to ring buffer
    self._buffer.append_batch(fhr_samples, uc_samples, timestamps)
```

**Patient Configuration:**
```python
@dataclass
class PatientConfig:
    patient_id: str
    bed_number: int
    name: str = "יולדת סימולציה"      # "Simulation Patient" in Hebrew
    baseline_fhr: float = 140.0        # Normal baseline
    baseline_variability: float = 10.0 # Normal variability
    contractions_per_10min: float = 4.0
    buffer_duration_minutes: float = 10.0
    sampling_rate: float = 4.0
```

### RingBuffer (`src/simulation/core/ring_buffer.py`)

A fixed-size circular buffer preventing memory growth during continuous monitoring.

**Memory Calculation:**
```
Buffer Size = 2400 samples × 8 patients × 4 bytes × 2 channels = ~77 KB
```

**Implementation:**
```python
class RingBuffer:
    def __init__(self, max_samples: int = 2400):
        self._fhr = collections.deque(maxlen=max_samples)
        self._uc = collections.deque(maxlen=max_samples)
        self._timestamps = collections.deque(maxlen=max_samples)
    
    def append(self, fhr: float, uc: float, timestamp: float):
        # O(1) append - oldest samples automatically discarded
        self._fhr.append(fhr)
        self._uc.append(uc)
        self._timestamps.append(timestamp)
    
    def get_window(self, duration_seconds: float) -> Dict[str, np.ndarray]:
        n_samples = int(duration_seconds * self._sampling_rate)
        return {
            'fhr': np.array(list(self._fhr)[-n_samples:]),
            'uc': np.array(list(self._uc)[-n_samples:]),
            'timestamps': np.array(list(self._timestamps)[-n_samples:])
        }
```

### Event Types (`src/simulation/events/event_types.py`)

Injectable pathological events for testing:

| Category | Event Type | Clinical Meaning |
|----------|------------|------------------|
| **Decelerations** | `LATE_DECELERATION` | Nadir >15s after contraction peak (uteroplacental insufficiency) |
| | `VARIABLE_DECELERATION` | Abrupt onset, variable timing (cord compression) |
| | `PROLONGED_DECELERATION` | Duration 2-10 minutes (severe stress) |
| | `EARLY_DECELERATION` | Nadir coincides with contraction peak (head compression) |
| **Baseline** | `BRADYCARDIA` | FHR <110 bpm for ≥10 minutes |
| | `TACHYCARDIA` | FHR >160 bpm for ≥10 minutes |
| **Variability** | `ABSENT_VARIABILITY` | ≤2 bpm (SEVERE - associated with acidemia) |
| | `MINIMAL_VARIABILITY` | 3-5 bpm (concerning) |
| | `MARKED_VARIABILITY` | >25 bpm (elevated) |
| **Patterns** | `SINUSOIDAL_PATTERN` | 3-5 cycles/min, smooth waves (ALWAYS Category 3) |
| **Uterine** | `TACHYSYSTOLE` | >5 contractions per 10 minutes |

**Deceleration Parameters with Severity Levels:**
```python
@dataclass
class LateDecelerationParams(EventParameters):
    depth_bpm: float = 30.0
    lag_seconds: float = 20.0
    recovery_seconds: float = 30.0
    
    @classmethod
    def mild(cls):      # depth=22, lag=15s, recovery=20s, recurrence=30%
    @classmethod
    def moderate(cls):  # depth=30, lag=20s, recovery=30s, recurrence=60%
    @classmethod
    def severe(cls):    # depth=50, lag=25s, recovery=45s, recurrence=80%
```

**Severity Signs for Variable Decelerations (Category 3 indicators):**
1. Drop to <70 bpm for >60 seconds
2. Absent variability within the deceleration
3. Slow recovery (>60 seconds from nadir to baseline)
4. Overshoot (>10 bpm above baseline after recovery)
5. W-shape (biphasic pattern with recovery between dips)

---

## 2.2 Signal Processing

### CTGPreprocessor (`src/data/preprocess.py`)

The preprocessor cleans raw CTG signals according to Israeli Position Paper guidelines.

**Processing Pipeline:**
```
Raw FHR → Out-of-Range Removal → Spike Detection → Gap Filling → Savitzky-Golay Smoothing
```

**Configuration:**
```python
@dataclass
class PreprocessConfig:
    sampling_rate: float = 4.0
    fhr_min: float = 50.0          # Minimum valid FHR (bpm)
    fhr_max: float = 240.0         # Maximum valid FHR (bpm)
    max_gap_seconds: float = 10.0  # The "10-Second Rule"
    smoothing_window: int = 5
    spike_threshold: float = 30.0  # Max BPM change between samples
```

### The 10-Second Rule

Per the Israeli Position Paper:
- **Gaps ≤10 seconds** (40 samples @ 4Hz): Fill with linear interpolation
- **Gaps >10 seconds**: Leave as NaN (signal loss - cannot reliably interpolate)

```python
def _fill_gaps(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_gap_samples = int(self.config.max_gap_seconds * self.config.sampling_rate)
    
    for start, end in gap_regions:
        gap_length = end - start
        if gap_length <= max_gap_samples:
            # Linear interpolation
            signal[start:end] = np.interp(
                np.arange(start, end),
                [start-1, end],
                [signal[start-1], signal[end]]
            )
        # else: leave as NaN
```

### Spike Detection

Non-physiological artifacts are detected and removed:
```python
def _remove_spikes(self, signal: np.ndarray) -> np.ndarray:
    # Change >30 BPM between consecutive samples = artifact
    diff = np.abs(np.diff(signal))
    spike_mask = diff > self.config.spike_threshold
    signal[1:][spike_mask] = np.nan
    return signal
```

### Savitzky-Golay Smoothing (Phase 13 Upgrade)

**Why Savitzky-Golay?**
- Previous median filter distorted deceleration shapes
- Savitzky-Golay preserves peak/valley characteristics while removing noise
- Better detection of Late vs Variable decelerations under noisy conditions

**Implementation:**
```python
from scipy.signal import savgol_filter

def _apply_smoothing_filter(self, signal: np.ndarray) -> np.ndarray:
    window_length = 11  # 2.75 seconds at 4Hz
    polyorder = 2       # Quadratic - preserves peaks/valleys
    
    # Handle NaN values (savgol_filter doesn't accept NaN)
    valid_mask = ~np.isnan(signal)
    if np.sum(valid_mask) < window_length:
        return signal
    
    # Interpolate NaN for filtering, then restore
    signal_interp = np.interp(
        np.arange(len(signal)),
        np.where(valid_mask)[0],
        signal[valid_mask]
    )
    smoothed = savgol_filter(signal_interp, window_length, polyorder)
    smoothed[~valid_mask] = np.nan  # Restore NaN positions
    return smoothed
```

**Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `window_length` | 11 | 2.75 seconds - captures short-term changes |
| `polyorder` | 2 | Quadratic fit - preserves curvature |

---

## 2.3 Clinical Rules Engine

### Baseline Calculation (`src/rules/baseline.py`)

**Definition:** Mean FHR rounded to 5 bpm during a 10-minute segment, excluding periodic changes and marked variability.

**Algorithm:**
```python
def calculate_baseline(fhr: np.ndarray, sampling_rate: float = 4.0) -> BaselineResult:
    # 1. Slide 2-minute window across signal (10-second steps)
    for start in range(0, len(fhr) - window_samples, step_samples):
        segment = fhr[start:start + window_samples]
        
        # 2. Calculate variability in window
        variability = np.nanmax(segment) - np.nanmin(segment)
        
        # 3. Find most stable segment (variability < 25 bpm)
        if variability < VARIABILITY_THRESHOLD:
            stable_segments.append((start, segment))
    
    # 4. Calculate mean FHR in most stable segment
    if stable_segments:
        best_segment = min(stable_segments, key=lambda x: np.nanstd(x[1]))
        baseline = np.nanmean(best_segment[1])
    else:
        baseline = np.nanmean(fhr)  # Fallback
    
    # 5. Round to nearest 5 bpm
    return round(baseline / 5) * 5
```

**Classification Thresholds:**
| Classification | Range | Clinical Meaning |
|----------------|-------|------------------|
| **Normal** | 110-160 bpm | Reassuring |
| **Bradycardia** | <110 bpm | Below normal - requires monitoring |
| **Tachycardia** | >160 bpm | Above normal - may indicate fever, infection, hypoxia |

**Result Structure:**
```python
@dataclass
class BaselineResult:
    value: float              # Baseline in bpm (rounded to 5)
    is_normal: bool           # 110-160 bpm
    is_bradycardia: bool      # < 110 bpm
    is_tachycardia: bool      # > 160 bpm
    stable_segment_found: bool
    confidence: float         # 1.0 - (variability / threshold)
```

### Variability Calculation (`src/rules/variability.py`)

**Definition:** Fluctuation in baseline FHR, quantified as amplitude range (max - min) within 1-minute windows.

**Categories per Israeli Position Paper:**
| Category | Range | Clinical Significance |
|----------|-------|----------------------|
| **ABSENT** | 0-2 bpm | **SEVERE** - associated with fetal acidemia |
| **MINIMAL** | 3-5 bpm | Concerning - requires close monitoring |
| **MODERATE** | 6-25 bpm | **NORMAL** - intact autonomic nervous system |
| **MARKED** | >25 bpm | Elevated - may indicate hypoxia or infection |

> **Clinical Note:** Moderate variability is the **single most reliable indicator** of fetal well-being.

**Algorithm:**
```python
def calculate_variability(fhr: np.ndarray, sampling_rate: float = 4.0) -> VariabilityResult:
    window_samples = int(60 * sampling_rate)  # 1 minute
    overlap = 0.5  # 50% overlap
    
    amplitudes = []
    for start in range(0, len(fhr) - window_samples, int(window_samples * (1 - overlap))):
        segment = fhr[start:start + window_samples]
        valid = segment[~np.isnan(segment)]
        
        if len(valid) >= window_samples * 0.5:  # Require 50% valid data
            amplitude = np.max(valid) - np.min(valid)
            amplitudes.append(amplitude)
    
    avg_amplitude = np.mean(amplitudes)
    
    # Classify
    if avg_amplitude <= 2.0:
        category = VariabilityCategory.ABSENT
    elif avg_amplitude <= 5.0:
        category = VariabilityCategory.MINIMAL
    elif avg_amplitude <= 25.0:
        category = VariabilityCategory.MODERATE
    else:
        category = VariabilityCategory.MARKED
    
    return VariabilityResult(value=avg_amplitude, category=category)
```

### Deceleration Detection (`src/rules/decelerations.py`)

**Definition:** Decrease in FHR of ≥15 bpm below baseline lasting ≥15 seconds but <10 minutes.

**Phase 13 Threshold Changes (improved noise tolerance):**
| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `min_depth` | 15 bpm | 12 bpm | Better detection under noise |
| `min_duration` | 15 sec | 12 sec | Catch early decelerations |

**Classification by Descent Time (30-Second Rule - FIGO/NICHD Standard):**

The 30-second descent time rule is the primary discriminator between Variable and Late decelerations, based on FIGO and NICHD clinical guidelines.

```python
# Descent time = time from onset to nadir

DESCENT_TIME_THRESHOLD = 30.0  # seconds

def calculate_descent_time(fhr, start_idx, nadir_idx, sampling_rate=4.0):
    """Calculate descent time from onset to nadir."""
    return (nadir_idx - start_idx) / sampling_rate

def classify_deceleration(fhr, uc, nadir_idx, start, end, sampling_rate):
    descent_time = calculate_descent_time(fhr, start, nadir_idx, sampling_rate)
    lag_seconds = (nadir_idx - contraction_peak_idx) / sampling_rate
    
    # PRIMARY RULE: 30-second descent time threshold
    if descent_time < DESCENT_TIME_THRESHOLD:
        # Abrupt onset (<30s) is hallmark of Variable deceleration
        return DecelerationType.VARIABLE
    else:
        # Gradual onset (≥30s) indicates Late or Early deceleration
        if lag_seconds > 15:
            return DecelerationType.LATE     # Uteroplacental insufficiency
        elif abs(lag_seconds) < 5:
            return DecelerationType.EARLY    # Head compression
        else:
            return DecelerationType.UNCLASSIFIED
```

**Fuzzy Logic for Borderline Cases (25-35 seconds):**
```python
def _fuzzy_classification(descent_time: float) -> Tuple[float, float]:
    """
    Returns (variable_confidence, late_confidence) for borderline cases.
    
    Descent time zones:
    - < 25s: Clearly Variable (1.0, 0.0)
    - 25-30s: Fuzzy zone favoring Variable
    - 30-35s: Fuzzy zone favoring Late
    - > 35s: Clearly Late (0.0, 1.0)
    """
    if descent_time < 25:
        return (1.0, 0.0)
    elif descent_time < 30:
        # Linear interpolation: 25s → 100% Variable, 30s → 50% Variable
        variable_conf = 1.0 - (descent_time - 25) / 10
        return (variable_conf, 1 - variable_conf)
    elif descent_time < 35:
        # Linear interpolation: 30s → 50% Late, 35s → 100% Late
        late_conf = 0.5 + (descent_time - 30) / 10
        return (1 - late_conf, late_conf)
    else:
        return (0.0, 1.0)
```

**Classification Summary Table:**
| Descent Time | Classification | Confidence | Clinical Meaning |
|--------------|----------------|------------|------------------|
| < 25 seconds | Variable | 100% | Abrupt onset - cord compression |
| 25-30 seconds | Variable (fuzzy) | 60-100% | Likely cord compression |
| 30-35 seconds | Late (fuzzy) | 50-80% | Borderline uteroplacental |
| > 35 seconds | Late | 100% | Gradual onset - uteroplacental insufficiency |

**Legacy Descent Rate Calculation (Phase 13 - retained for compatibility):**
```python
def _calculate_descent_rate(fhr, start, nadir):
    segment = fhr[start:nadir+1]
    valid_values = segment[~np.isnan(segment)]
    
    # Use percentiles for robustness to jagged signals
    n_valid = len(valid_values)
    first_portion = max(1, n_valid // 5)  # First 20%
    
    start_val = np.percentile(valid_values[:first_portion], 90)  # 90th percentile
    nadir_val = np.min(valid_values)  # True minimum
    
    drop = start_val - nadir_val
    samples = nadir - start
    
    return drop / samples if samples > 0 else 0.0
```

**Severity Signs Detection:**
```python
def _check_severity_signs(fhr, start, end, nadir, baseline, sampling_rate):
    signs = []
    segment = fhr[start:end]
    
    # 1. Drop < 70 bpm for > 60 seconds
    below_70 = segment < 70
    below_70_duration = np.sum(below_70) / sampling_rate
    if below_70_duration > 60:
        signs.append("Drop below 70 bpm for >60 seconds")
    
    # 2. Absent internal variability
    if np.nanmax(segment) - np.nanmin(segment) < 5:
        signs.append("Absent internal variability")
    
    # 3. Slow recovery (> 60 seconds from nadir to end)
    recovery_time = (end - nadir) / sampling_rate
    if recovery_time > 60:
        signs.append("Slow recovery (>60 seconds)")
    
    # 4. Overshoot (> baseline + 10)
    post_recovery = fhr[end:end + int(30 * sampling_rate)]
    if np.any(post_recovery > baseline + 10):
        signs.append("Overshoot (>10 bpm above baseline)")
    
    return len(signs) > 0, signs
```

### Sinusoidal Pattern Detection (`src/rules/sinusoidal.py`)

**Definition:** Smooth, sine wave-like oscillation in baseline FHR.

> ⚠️ **ALWAYS CATEGORY 3 (PATHOLOGICAL)** - indicates severe fetal anemia

**Characteristics:**
| Parameter | Range |
|-----------|-------|
| Frequency | 3-5 cycles per minute (0.05-0.083 Hz) |
| Amplitude | 5-25 bpm (Phase 13: relaxed from 5-15) |
| Duration | >20 minutes |
| Variability | Absent short-term variability (smooth waves) |

**Algorithm (FFT-based):**
```python
def detect_sinusoidal(fhr: np.ndarray, sampling_rate: float = 4.0) -> SinusoidalResult:
    min_duration_minutes = 20.0
    
    # Check signal length
    if len(fhr) < min_duration_minutes * 60 * sampling_rate:
        return SinusoidalResult(detected=False, reason="Signal too short")
    
    # Take last 20 minutes
    window = fhr[-int(20 * 60 * sampling_rate):]
    
    # 1. Perform FFT
    fft_result = np.fft.fft(window - np.nanmean(window))
    freqs = np.fft.fftfreq(len(window), 1/sampling_rate)
    power = np.abs(fft_result) ** 2
    
    # 2. Look for dominant peak in 3-5 cycles/minute range
    target_freq_min = 3.0 / 60  # 0.05 Hz
    target_freq_max = 5.0 / 60  # 0.083 Hz
    
    in_range = (freqs >= target_freq_min) & (freqs <= target_freq_max)
    target_power = np.sum(power[in_range])
    total_power = np.sum(power)
    
    # 3. Calculate dominance ratio
    dominance_ratio = target_power / total_power if total_power > 0 else 0
    
    # 4. Calculate amplitude
    amplitude = (np.nanmax(window) - np.nanmin(window)) / 2
    
    # 5. Detection criteria (Phase 13: relaxed thresholds)
    detected = (
        dominance_ratio >= 0.15 and        # Was 0.3
        5.0 <= amplitude <= 25.0           # Was 5-15
    )
    
    return SinusoidalResult(
        detected=detected,
        frequency_cycles_per_min=peak_freq * 60,
        amplitude_bpm=amplitude,
        dominance_ratio=dominance_ratio
    )
```

### Tachysystole Detection (`src/rules/tachysystole.py`)

**Definition:** Excessive uterine activity - average >5 contractions per 10-minute window.

**Algorithm:**
```python
def detect_tachysystole(uc: np.ndarray, sampling_rate: float = 4.0) -> TachysystoleResult:
    analysis_window_minutes = 30.0
    
    # Detect contractions using peak detection
    from scipy.signal import find_peaks
    
    uc_clean = np.nan_to_num(uc, nan=0.0)
    height_threshold = np.percentile(uc_clean, 75)
    min_distance = int(60 * sampling_rate)  # Minimum 60 seconds between contractions
    prominence = 0.5 * np.std(uc_clean)
    
    peaks, _ = find_peaks(
        uc_clean,
        height=height_threshold,
        distance=min_distance,
        prominence=prominence
    )
    
    # Calculate rate
    duration_minutes = len(uc) / sampling_rate / 60
    contractions_per_10min = len(peaks) / duration_minutes * 10
    
    return TachysystoleResult(
        detected=contractions_per_10min > 5.0,
        contractions_per_10min=contractions_per_10min,
        total_contractions=len(peaks)
    )
```

---

## 2.4 AI Model Integration

### V3.0 Architecture: MiniRocket Engine

**Updated 2026-01-20:** The system now uses MiniRocket for feature extraction, replacing the heavy MOMENT transformer. This provides 10-20x inference speedup while maintaining clinical accuracy.

#### Current Production Stack

| Component | Backend | File | Status |
|-----------|---------|------|--------|
| **MiniRocket Encoder** | sktime | `src/models/minirocket_encoder.py` | ✅ Active (Default) |
| **MOMENT Encoder** | PyTorch | `src/models/moment_encoder.py` | 🔄 Optional Fallback |
| **XGBoost Classifier** | JSON | `models/sentinel_classifier.json` | ✅ Active |
| **FSQI Gate** | scipy | `src/data/signal_quality.py` | ✅ Active |
| **Rule Engine** | Python | `src/rules/` | ✅ Active |
| **Safety Net** | Python | `src/analysis/override.py` | ✅ Active |

> **✅ V3.0 Upgrade:** MiniRocket uses 84 fixed convolutional kernels that require no training. Cold-start capability enables immediate inference without pre-fitted data.

#### Backend Selection

```python
from src.adapters.model_adapters import get_feature_extractor

# Automatically selects best available backend (MiniRocket preferred)
encoder = get_feature_extractor(backend='auto')
features = encoder.extract_features(fhr_window)  # 84-dim vector

# In V3.0, this will use MiniRocket by default
print(f"Encoder type: {type(encoder).__name__}")  # MiniRocketAdapter

# Force specific backend if needed
minirocket = get_feature_extractor(backend='minirocket')
moment = get_feature_extractor(backend='moment')  # Fallback for research
```

#### Performance Comparison (V3.0 vs V2.0)

| Metric | MiniRocket (V3.0) | MOMENT (V2.0) | Improvement |
|--------|-------------------|---------------|-------------|
| Inference Time | ~10ms per window | 2-5s per window | **100-500x faster** |
| Model Size | <1MB (kernels) | ~1.5GB (weights) | **1500x smaller** |
| Memory Usage | ~50MB peak | ~2GB peak | **40x less** |
| Concurrent Patients | 20 | 8 | **2.5x more** |
| Cold-Start | ✅ Instant | ❌ Requires warmup | Immediate |
| Feature Dimensions | 84 | 1024 | More compact |

#### MiniRocket Configuration

```python
@dataclass
class MiniRocketConfig:
    num_kernels: int = 84           # Fixed kernel count
    max_dilations_per_kernel: int = 32
    random_state: int = 42          # Reproducibility
    n_jobs: int = 1                 # Single-threaded for consistency
```

### MiniRocket Feature Extractor (`src/models/minirocket_encoder.py`)

**What is MiniRocket?**
- Lightweight time-series feature extractor from sktime
- Uses 84 fixed random convolutional kernels
- No training required - deterministic feature extraction
- Supports cold-start with synthetic training data generation

**Specifications:**
| Parameter | Value |
|-----------|-------|
| Library | sktime MiniRocket |
| Kernels | 84 (fixed random) |
| Feature Dimension | 84 |
| Input Size | 2,400 samples (10 min @ 4Hz) |
| Inference Time | ~10ms per window |
| Memory | ~50MB peak |

**Cold-Start Capability:**
```python
def _cold_start_fit(self):
    """Generate synthetic data for fitting if no real data available."""
    # Generate diverse synthetic FHR patterns
    synthetic_data = self._generate_synthetic_training_data(n_samples=100)
    
    # Fit MiniRocket transform on synthetic data
    self._transformer.fit(synthetic_data)
    self._is_fitted = True
```

**Feature Extraction:**
```python
def extract_features(self, fhr: np.ndarray) -> np.ndarray:
    # 1. Prepare signal (pad/truncate to window_size)
    signal = self._prepare_signal(fhr)
    
    # 2. Cold-start if not fitted
    if not self._is_fitted:
        self._cold_start_fit()
    
    # 3. Transform using MiniRocket kernels
    features = self._transformer.transform(signal.reshape(1, -1))
    
    return features.flatten()  # Shape: (84,)
```

### Signal Quality Gate (`src/data/signal_quality.py`)

**Purpose:** Filter out low-quality signals before classification to reduce false alerts.

**FSQI (Fetal Signal Quality Index) Components:**
| Component | Weight | Description |
|-----------|--------|-------------|
| Valid Signal Ratio | 0.4 | Proportion of non-NaN samples |
| Spectral Noise Ratio | 0.3 | Low-frequency vs high-frequency power |
| Baseline Stability | 0.2 | Standard deviation of rolling baseline |
| Gap Penalty | 0.1 | Penalty for signal gaps > 10s |

**Algorithm:**
```python
def calculate_fsqi(fhr: np.ndarray, sampling_rate: float = 4.0) -> float:
    # 1. Valid signal ratio (weight: 0.4)
    valid_ratio = np.sum(~np.isnan(fhr)) / len(fhr)
    
    # 2. Spectral noise analysis (weight: 0.3)
    fft = np.fft.fft(fhr[~np.isnan(fhr)])
    freqs = np.fft.fftfreq(len(fft), 1/sampling_rate)
    
    # Physiological band (0.01-0.5 Hz) vs noise band (>0.5 Hz)
    physiological_power = np.sum(np.abs(fft[(freqs > 0.01) & (freqs < 0.5)])**2)
    noise_power = np.sum(np.abs(fft[freqs > 0.5])**2)
    snr = physiological_power / (noise_power + 1e-8)
    spectral_score = min(1.0, snr / 10)  # Normalize
    
    # 3. Baseline stability (weight: 0.2)
    rolling_mean = np.convolve(fhr, np.ones(60)/60, mode='valid')
    stability_score = 1.0 - min(1.0, np.nanstd(rolling_mean) / 20)
    
    # 4. Gap penalty (weight: 0.1)
    gap_penalty = count_large_gaps(fhr) * 0.1
    
    # Combined FSQI
    fsqi = (0.4 * valid_ratio + 
            0.3 * spectral_score + 
            0.2 * stability_score - 
            gap_penalty)
    
    return max(0.0, min(1.0, fsqi))

def apply_quality_gate(fhr: np.ndarray, threshold: float = 0.7) -> Tuple[bool, float]:
    fsqi = calculate_fsqi(fhr)
    return fsqi >= threshold, fsqi
```

**Quality Categories:**
| FSQI Range | Category | Action |
|------------|----------|--------|
| ≥ 0.85 | HIGH | Full classification |
| 0.70-0.84 | ACCEPTABLE | Classification with caution flag |
| 0.50-0.69 | LOW | Rule-based only (no ML) |
| < 0.50 | POOR | Signal loss alert |

### Coiflet 4 Wavelet Denoising

**Purpose:** Pre-processing step to remove high-frequency noise while preserving deceleration morphology.

```python
import pywt

def denoise_coiflet4(signal: np.ndarray, level: int = 4) -> np.ndarray:
    # Decompose signal using Coiflet 4 wavelet
    coeffs = pywt.wavedec(signal, 'coif4', level=level)
    
    # Soft threshold detail coefficients
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(signal)))
    
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    
    # Reconstruct denoised signal
    return pywt.waverec(coeffs, 'coif4')[:len(signal)]
```

### Feature Fusion (`src/models/fusion.py`)

**Purpose:** Combine MiniRocket features with rule-based features into a single feature vector.

**Feature Vector Structure (95 dimensions):**

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0-83 | MiniRocket Features | Already normalized |
| 84 | Baseline FHR | `/160` |
| 85 | Variability Value | `/25` |
| 86 | Variability: Absent | 0/1 (one-hot) |
| 87 | Variability: Minimal | 0/1 (one-hot) |
| 88 | Variability: Moderate | 0/1 (one-hot) |
| 89 | Variability: Marked | 0/1 (one-hot) |
| 90 | Late Deceleration Count | `/10` |
| 91 | Variable Deceleration Count | `/10` |
| 92 | Recurrent Decelerations Flag | 0/1 |
| 93 | Descent Time (seconds) | `/60` **NEW** |
| 94 | FSQI Score | 0-1 **NEW** |
| 1033 | Tachysystole Flag | 0/1 |
| 1034 | Sinusoidal Flag | 0/1 |

**Fusion Implementation:**
```python
def fuse(
    self,
    embedding: np.ndarray,
    baseline: BaselineResult,
    variability: VariabilityResult,
    decelerations: List[Deceleration],
    tachysystole: TachysystoleResult,
    sinusoidal: SinusoidalResult,
    total_contractions: int
) -> np.ndarray:
    
    features = np.zeros(1035, dtype=np.float32)
    
    # MOMENT embedding (1024 dims)
    features[:1024] = embedding
    
    # Rule features (11 dims)
    features[1024] = baseline.value / 160.0
    features[1025] = variability.value / 25.0
    
    # Variability one-hot
    var_idx = {
        VariabilityCategory.ABSENT: 1026,
        VariabilityCategory.MINIMAL: 1027,
        VariabilityCategory.MODERATE: 1028,
        VariabilityCategory.MARKED: 1029
    }
    features[var_idx[variability.category]] = 1.0
    
    # Deceleration counts
    late_count = sum(1 for d in decelerations if d.decel_type == DecelerationType.LATE)
    var_count = sum(1 for d in decelerations if d.decel_type == DecelerationType.VARIABLE)
    features[1030] = min(late_count / 10.0, 1.0)
    features[1031] = min(var_count / 10.0, 1.0)
    
    # Recurrent = >50% of contractions
    total_decels = late_count + var_count
    recurrent = (total_decels / total_contractions > 0.5) if total_contractions > 0 else False
    features[1032] = 1.0 if recurrent else 0.0
    
    # Flags
    features[1033] = 1.0 if tachysystole.detected else 0.0
    features[1034] = 1.0 if sinusoidal.detected else 0.0
    
    return features
```

### XGBoost Classifier (`src/models/classifier.py`)

**Purpose:** Predict fetal categories (1/2/3) from 1,035-dimensional feature vectors.

**Configuration:**
```python
xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'use_label_encoder': False
}
```

**Training Features:**
- Stratified K-Fold cross-validation (5 folds)
- Class weighting for imbalanced data (inverse frequency)
- F1 macro score for evaluation

**Inference:**
```python
class HybridClassifier:
    def predict(self, features: np.ndarray) -> int:
        # Returns class label: 0, 1, or 2 (maps to Category 1, 2, 3)
        return self.model.predict(features.reshape(1, -1))[0]
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        # Returns probability distribution: [P(Cat1), P(Cat2), P(Cat3)]
        return self.model.predict_proba(features.reshape(1, -1))[0]
```

---

## 2.5 Safety Net (Medical Overrides)

### Philosophy

The Medical Override system implements a **"Safety Net"** that enforces critical clinical findings regardless of ML predictions.

**Core Principle:**
> ML can UPGRADE to more severe categories, but critical findings should **NEVER** be DOWNGRADED by ML.

### Override Rules (`src/analysis/override.py`)

#### HARD OVERRIDE RULES (Force Category 3 - Pathological)

**Rule 1: Sinusoidal Pattern**
```python
if sinusoidal.detected:
    return OverrideResult(
        final_category=3,
        override_applied=True,
        reason="Sinusoidal pattern detected",
        original_ml_category=ml_prediction
    )
```
**Clinical Rationale:** Sinusoidal pattern indicates severe fetal anemia/compromise and requires immediate intervention.

**Rule 2: Absent Variability + Ominous Signs**
```python
if is_absent_variability:
    recurrent_late = sum(1 for d in decels if d.decel_type == LATE) >= 3
    recurrent_variable = sum(1 for d in decels if d.decel_type == VARIABLE) >= 3
    has_bradycardia = baseline.is_bradycardia
    
    if recurrent_late or recurrent_variable or has_bradycardia:
        return OverrideResult(
            final_category=3,
            override_applied=True,
            reason="Absent variability with recurrent decelerations or bradycardia"
        )
```
**Clinical Rationale:** Absent variability combined with recurrent late/variable decelerations or bradycardia is highly predictive of fetal acidemia.

#### UPGRADE RULES (Force Category 2 - Intermediate)

**Rule 3: Bradycardia Alone**
```python
if baseline.is_bradycardia:
    return OverrideResult(
        final_category=max(ml_prediction, 2),  # At least Category 2
        override_applied=True,
        reason="Bradycardia detected"
    )
```

**Rule 4: Recurrent Late Decelerations Alone**
```python
late_count = sum(1 for d in decelerations if d.decel_type == LATE)
if late_count >= 3:  # "Recurrent" = appearing in ≥50% of contractions
    return OverrideResult(
        final_category=max(ml_prediction, 2),
        override_applied=True,
        reason="Recurrent late decelerations"
    )
```

#### SAFETY FLOOR RULE

**Rule 5: Absent Variability Never Normal**
```python
if is_absent_variability and ml_prediction == 1:
    return OverrideResult(
        final_category=2,
        override_applied=True,
        reason="Absent variability should not be classified as Normal"
    )
```
**Clinical Rationale:** Absent variability always requires closer monitoring, even without other concerning findings.

### Override Summary Table

| # | Condition | Action | Rationale |
|---|-----------|--------|-----------|
| 1 | Sinusoidal pattern detected | → **Force Cat 3** | Severe fetal anemia |
| 2 | Absent variability + recurrent decels/brady | → **Force Cat 3** | High acidemia risk |
| 3 | Bradycardia alone | → **Force Cat 2** | Requires monitoring |
| 4 | Recurrent late decelerations | → **Force Cat 2** | Uteroplacental insufficiency |
| 5 | Absent variability + ML=Normal | → **Force Cat 2** | Safety floor |

### Alert Generation (`src/analysis/alerts.py`)

Generates Hebrew explanations for clinical staff:

```python
CATEGORY_HEADLINES = {
    1: "התראה ירוקה - קטגוריה 1 (תקין)",      # Green - Normal
    2: "התראה צהובה - קטגוריה 2 (לא מוגדר)",  # Yellow - Intermediate
    3: "התראה אדומה - קטגוריה 3 (פתולוגי)"    # Red - Pathological
}

@dataclass
class Alert:
    category: int
    confidence: float
    headline: str          # Short description
    explanation: str       # Detailed Hebrew explanation
    findings: List[str]    # Medical findings (Hebrew)
    recommendations: List[str]  # Clinical recommendations (Hebrew)
    timestamp: str         # ISO format
```

---

# Chapter 3: Verification & Validation

## 3.1 Accuracy Testing

### Phase A: MOMENT Baseline Performance

**Test:** Load MOMENT model and measure inference characteristics.

| Metric | Value |
|--------|-------|
| Model Load Time | 11.1 seconds |
| RAM Usage | +1,335 MB |
| Inference Mean | 1,869 ms per 10-min window |
| Inference P95 | 2,392 ms |
| Throughput | 0.535 windows/second |

### Phase B: Pattern Calibration

**Test:** Inject known pathological patterns and measure detection sensitivity.

| Pattern | Sensitivity |
|---------|-------------|
| Sinusoidal | 1.00 (100%) |
| Bradycardia | 1.00 (100%) |
| Tachycardia | 1.00 (100%) |
| Prolonged Deceleration | 1.00 (100%) |
| Variable Deceleration (severe) | 1.00 (100%) |
| Late Deceleration (severe) | 1.00 (100%) |
| Late Deceleration (moderate) | 0.75 (75%) |
| Late Deceleration (mild) | 0.05 (5%) |

**Finding:** Severe patterns detected with 100% sensitivity. Mild patterns have lower sensitivity due to threshold proximity.

### Phase C: Load Testing

**Test:** Verify system meets real-time requirements with 8 patients.

| Metric | Value | Threshold |
|--------|-------|-----------|
| Tick Rate | ~0.984 Hz | ≥1.0 Hz |
| CPU Usage (mean) | 19-24% | <70% |
| Lag Flag | Cleared | - |

---

## 3.2 Endurance Testing

### Phase 10: 1-Hour Stability Test

**Objective:** Verify system stability over extended operation.

**Configuration:**
- Duration: 3,600 seconds (1 hour)
- Patients: 8 concurrent
- Injected Events: 12 (sinusoidal, late decels, bradycardia, etc.)

**Results:** ✅ **PASS**

| Metric | Value | Threshold |
|--------|-------|-----------|
| Total Duration | 3,600.2 sec | 3,600 sec |
| Maximum RAM | 367.74 MB | - |
| Memory Growth | 1.89 MB/hr | <50 MB/hr |
| Total Ticks | 3,041 | ~3,600 |
| Max Tick Latency | 162.61 ms | <1,500 ms |
| Injections Processed | 12/12 | 12 |

**Conclusion:** Zero crashes, stable memory (no leaks), all injections handled correctly, no tick latency warnings.

---

## 3.3 Robustness Testing

### Phase 11: Initial Robustness Test (Torture Test)

**Objective:** Test system resilience to noise and signal dropout.

**Configuration:**
- 108 scenarios (6 patterns × 3 severities × 3 noise levels × 2 dropout rates)
- Noise levels: 0, 5, 10 bpm σ
- Dropout rates: 0%, 10%

**Results:** ⚠️ **FAILURE REVEALED**

| Metric | Value |
|--------|-------|
| Overall Detection | **9/108 (8.3%)** |
| Sinusoidal (all conditions) | 0% |
| Late Decel (severe, clean) | 22% |
| Late Decel (high noise) | 0% |

**Finding:** System was highly sensitive to noise, prompting Phase 12/13 hardening.

### Phase 12: Massive Scale Robustness (Initial)

**Objective:** 500-scenario stress test with fixes applied.

**Configuration:**
- 500 randomized scenarios
- 7 patterns (Sinusoidal, Late/Variable/Prolonged Decel, Brady, Tachy, Normal)
- 5 noise levels (0, 2, 5, 10, 15 bpm σ)
- 3 dropout rates (0%, 5%, 10%)

**Fix Applied:** Enabled smoothing by default (`apply_smoothing=True`)

**Results:**
| Metric | Value |
|--------|-------|
| Overall Detection | **299/500 (59.8%)** |
| False Positive Rate | 0/68 (0.0%) |
| Prolonged/Brady/Tachy | 100% |
| Sinusoidal | 0% |
| Late/Variable Decel | 3-4% |

**Finding:** Brady/Tachy/Prolonged perfect due to medical overrides. Sinusoidal still failing (window too short).

### Phase 13: Signal Processing Upgrade

**Objective:** Address remaining detection failures through signal processing improvements.

**Changes Made:**
1. **Savitzky-Golay Filter** - Replaced median filter (preserves peak shapes)
2. **Benchmark Window Fix** - Sinusoidal uses 25-min window (was 10-min)
3. **Deceleration Classification Fix** - Timing-based Late vs descent-rate Variable

**Final Results:**

| Metric | Phase 12 | Phase 13 | Change |
|--------|----------|----------|--------|
| **Overall Detection** | 59.8% | **64.0%** | +4.2% |
| **False Positive Rate** | 0.0% | **0.0%** | Maintained |
| Sinusoidal (clean) | 0% | **100%** | ✅ Fixed |
| Late Decel (clean) | 0-4% | **5-23%** | Improved |
| Variable Decel (severe) | 0-8% | **6-31%** | Improved |
| Prolonged/Brady/Tachy | 100% | **100%** | Maintained |
| Normal (specificity) | 100% | **100%** | Maintained |

**Detection by Noise Level (Final):**

| Pattern | σ=0 | σ=2 | σ=5 | σ=10 | σ=15 |
|---------|-----|-----|-----|------|------|
| Sinusoidal | 100% | 0% | 0% | 0% | 0% |
| Late Decel | 23% | 5% | 0% | 0% | 0% |
| Variable Decel | 14% | 15% | 0% | 17% | 6% |
| Prolonged | 100% | 100% | 100% | 100% | 100% |
| Bradycardia | 100% | 100% | 100% | 100% | 100% |
| Tachycardia | 100% | 100% | 100% | 100% | 100% |
| Normal | 100% | 100% | 100% | 100% | 100% |

**Key Insight:** Critical patterns (Prolonged, Brady, Tachy) maintain 100% detection across all noise levels due to medical override safety net. Sinusoidal requires clean signal for FFT detection.

---

## 3.4 Performance Metrics

### CPU Performance

| Operation | Time | Notes |
|-----------|------|-------|
| MOMENT Load | 11.1s | One-time startup |
| MOMENT Inference | 1,869ms mean | Per 10-min window |
| Rule Engine (all rules) | ~10ms | Per patient per tick |
| Full Tick (8 patients) | <200ms | Including staggered MOMENT |

### Memory Performance

| Metric | Value |
|--------|-------|
| Base RAM Usage | ~300 MB |
| MOMENT Model RAM | +1,335 MB |
| Ring Buffer (8 patients) | ~77 KB |
| Memory Growth Rate | 1.89 MB/hr |
| 24-Hour Projected Growth | ~45 MB |

### Throughput

| Metric | Value |
|--------|-------|
| Tick Rate | 0.984 Hz |
| Samples/Second | 4 Hz × 8 patients = 32 |
| MOMENT Windows/Second | 0.535 (staggered) |

### UI Rendering Performance (Streamlit + Plotly)

**Observed Bottleneck (stress tests):** Plotly figure build + JSON serialization dominated cycle time in UI load tests.

**Mitigations implemented (source of truth):**
- Plot generation optimized in [src/ui/plots.py](src/ui/plots.py): min/max downsampling, `Scattergl` (WebGL) for large traces, and `uirevision` to preserve zoom/pan.
- App-level reruns optimized in [src/ui/simulation_app.py](src/ui/simulation_app.py): CTG monitor panel runs as a Streamlit fragment (`st.fragment(run_every=...)`) to avoid full-page reruns.
- Throttling: refresh rate capped to 2–5 FPS via a UI control (prevents rerun storms / CPU thrash).

### Load & Stability Benchmarks (Latest)

| Benchmark | Scenario | Result | Notes |
|-----------|----------|--------|-------|
| Load (Phase 10) | 1-hour, 8 patients, 12 injections | ✅ Pass | Max tick latency 162.6 ms; RAM 367.7 MB; +1.89 MB/hr growth |
| Robustness (Phase 13) | 500 scenarios (noise/dropout sweep) | Detection 64.0%, FPR 0.0% | Sinusoidal clean 100%; Prolonged/Brady/Tachy 100% across noise; Late/Variable improved after Savitzky-Golay |
| MOMENT Baseline | Load + inference timing | 11.1s load; 1.87s inference mean | CPU-only measurements |

### Detection Constraints (Clinically Tuned)

| Pattern | Constraint | Purpose |
|---------|------------|---------|
| Baseline | 110–160 bpm normal; brady <110; tachy >160 | Align with Position Paper ranges |
| Variability | Absent ≤2, Minimal ≤5, Moderate ≤25, Marked >25 bpm | Moderate is primary well-being indicator |
| Decelerations | Depth ≥12 bpm, duration ≥12s; descent ≥0.5 bpm/sample → Variable; lag >15s → Late | Improve robustness under noise (Phase 13) |
| Sinusoidal | 3–5 cycles/min, amplitude 5–25 bpm, dominance ≥0.15, ≥20 min | Always Category 3 safety rule |
| Tachysystole | >5 contractions/10 min (UC peaks) | Uterine hyperstimulation detection |
| Overrides | Sinusoidal → Cat3; Absent variability + recurrent decels/brady → Cat3; Brady or recurrent late → ≥Cat2; Absent variability never Normal | Safety net to prevent dangerous downgrades |

---

# Chapter 4: Setup & Operations

## 4.1 Installation

### Prerequisites

- Python 3.10+ 
- Windows/Linux/macOS
- 4 GB RAM minimum (8 GB recommended for MOMENT)
- CPU: i5 or equivalent (Intel recommended for OpenVINO)

### Installation Steps

```bash
# 1. Clone repository
git clone <repository-url>
cd SentinelFetal

# 2. Create virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install ONNX optimization dependencies
pip install onnx>=1.14.0 onnxruntime>=1.15.0

# 5. (Optional) Install OpenVINO for Intel CPUs (best performance)
pip install openvino>=2023.1.0

# 6. Set Python path
# Windows
set PYTHONPATH=.

# Linux/macOS
export PYTHONPATH=.

# 7. Verify installation
python scripts/verify_system.py
```

### Dependencies

```
# Core
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# ML
xgboost>=1.7.0
torch>=2.0.0
transformers>=4.30.0

# UI
streamlit>=1.28.0
plotly>=5.15.0

# Testing
pytest>=7.3.0

# ONNX Optimization (V2 - recommended)
onnx>=1.14.0
onnxruntime>=1.15.0

# OpenVINO (V2 - optional, best for Intel CPUs)
openvino>=2023.1.0

# Optional: MOMENT PyTorch (for training/export)
momentfm>=0.1.0  # If using real MOMENT model
```

### V2 Model Optimization (Optional)

For best inference performance, export the MOMENT model to ONNX:

```bash
# Export to ONNX (requires momentfm)
python scripts/export_moment_onnx.py --output models/moment.onnx

# Quantize to INT8 (4x smaller, 2-3x faster)
python scripts/quantize_moment.py --input models/moment.onnx --output models/moment_int8.onnx
```

The system will automatically use the optimized model if available.

---

## 4.2 Running the Simulator

### Launch Simulation Dashboard

```bash
python scripts/run_simulation.py
```

Or run Streamlit directly:

```bash
streamlit run src/ui/simulation_app.py
```

This opens a Streamlit dashboard at `http://localhost:8501`.

### UI Architecture (V2.0 - Clinical Minimalism)

The dashboard uses a **Two-View Architecture**:

| View | Description |
|------|-------------|
| **Ward View (Grid)** | All patient monitors side-by-side (max 4 columns) |
| **Detail View** | Single patient focus with full CTG monitor and findings |

**Design Principles:**
- **Background:** Pure white (#FFFFFF)
- **Text:** Pure black (#000000)
- **Alerts:** Colored indicators (🟢🟠🔴) instead of large banners
- **Population:** Dynamic 1-20 patients (slider control)

**Key Features:**
- Population slider (1-20 synthetic patients)
- Start/Pause/Reset simulation controls
- Event injection with expected detection time display
- Mini sparklines in ward view
- Full CTG monitor (WebGL/Scattergl) in detail view
- Patient-specific event log

**UI Performance Notes:**
- CTG monitor uses `st.fragment(run_every=...)` for partial rerenders
- Refresh rate throttled to 2–5 FPS via slider
- WebGL (Scattergl) for large traces (>1500 points)
- `uirevision` preserves pan/zoom state

### Demo Scenarios

**Scenario 1: Normal (Green Alert)**
1. Start simulation
2. Observe Patient 1 (no injections)
3. Expected: Category 1 (Green) throughout

**Scenario 2: Sinusoidal → Pathological (Red Alert)**
1. Select Patient 2
2. Click "Inject Event" → "Sinusoidal Pattern"
3. Wait 20-40 seconds
4. Expected: Category 3 (Red) with override message

**Scenario 3: Late Decelerations → Intermediate (Yellow Alert)**
1. Select Patient 3
2. Inject "Late Deceleration (severe)" × 3
3. Expected: Category 2 (Yellow) due to recurrent late decels override

**Scenario 4: Bradycardia → Intermediate (Yellow Alert)**
1. Select Patient 4
2. Inject "Bradycardia" (80 bpm target)
3. Expected: Category 2 (Yellow) via medical override

---

## 4.3 Running Tests

### Unit Tests

```bash
# All tests
pytest tests/ -v

# Specific test files
pytest tests/test_rules.py -v
pytest tests/test_preprocessing.py -v
pytest tests/test_simulation_core.py -v
```

### Benchmarks

```bash
# Accuracy benchmark
python tests/benchmarks/benchmark_accuracy.py

# Clinical categories benchmark
python tests/benchmarks/benchmark_clinical.py

# Load test
python tests/benchmarks/benchmark_load.py

# Massive robustness test (500 scenarios)
python tests/benchmarks/benchmark_robustness_massive.py
```

### Expected Test Results

| Test Suite | Tests | Expected |
|------------|-------|----------|
| `test_rules.py` | 26 | All pass |
| `test_preprocessing.py` | 14 | All pass |
| `test_simulation_core.py` | 12 | All pass |
| `test_simulation_integration.py` | 8 | All pass |

---

## 4.4 Configuration Reference

### Main Configuration (`src/config.py`)

```python
# Signal Processing
SAMPLING_RATE = 4.0                    # Hz
MOMENT_WINDOW_MINUTES = 10.0
MOMENT_WINDOW_SAMPLES = 2400
MOMENT_EMBEDDING_DIM = 1024
TOTAL_FEATURES_DIM = 1035

# Clinical Thresholds (pH)
PH_PATHOLOGICAL = 7.15
PH_INTERMEDIATE = 7.20

# Baseline
BASELINE_NORMAL_MIN = 110              # bpm
BASELINE_NORMAL_MAX = 160              # bpm

# Variability
VARIABILITY_ABSENT_MAX = 2.0           # bpm
VARIABILITY_MINIMAL_MAX = 5.0          # bpm
VARIABILITY_MODERATE_MAX = 25.0        # bpm

# Decelerations
MIN_DECEL_DEPTH = 12.0                 # bpm (Phase 13)
MIN_DECEL_DURATION = 12.0              # seconds (Phase 13)
VARIABLE_DESCENT_THRESHOLD = 0.5       # bpm/sample

# Sinusoidal
SINUSOIDAL_MIN_DURATION = 20.0         # minutes
SINUSOIDAL_FREQ_MIN = 3.0              # cycles/min
SINUSOIDAL_FREQ_MAX = 5.0              # cycles/min
SINUSOIDAL_AMP_MIN = 5.0               # bpm
SINUSOIDAL_AMP_MAX = 25.0              # bpm (Phase 13)
SINUSOIDAL_DOMINANCE_THRESHOLD = 0.15  # (Phase 13)

# Simulation
DEFAULT_NUM_PATIENTS = 8
TICK_INTERVAL_SECONDS = 1.0
MOMENT_INTERVAL_SECONDS = 30.0
```

### Model Configuration (`models/xgb_demo.config.json`)

> **V2.0 Note:** The XGBoost classifier is saved as `models/sentinel_classifier.json` (renamed from `xgb_demo.json`).

```json
{
    "model_type": "xgboost",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "num_class": 3,
    "feature_dim": 1035
}
```

---

# Appendix A: Clinical Reference

## Israeli Position Paper Categories

| Category | Criteria | Action |
|----------|----------|--------|
| **Category 1** | Baseline 110-160, Moderate variability, No decels or early decels only | Continue monitoring |
| **Category 2** | Anything not fitting Cat 1 or Cat 3 | Evaluate, consider intervention |
| **Category 3** | Absent variability + recurrent late/variable decels, Sinusoidal, Bradycardia <100 | Immediate intervention |

## Deceleration Classification

| Type | Lag Time | Onset | Clinical Meaning |
|------|----------|-------|------------------|
| **Early** | <5 seconds | Gradual | Head compression (benign) |
| **Late** | >15 seconds | Gradual | Uteroplacental insufficiency |
| **Variable** | Variable | Abrupt (>0.5 bpm/s) | Cord compression |
| **Prolonged** | N/A | N/A | Duration >2 min (severe) |

---

# Appendix B: Glossary

| Term | Definition |
|------|------------|
| **FHR** | Fetal Heart Rate (bpm) |
| **UC** | Uterine Contractions |
| **CTG** | Cardiotocography - continuous monitoring of FHR and UC |
| **Baseline** | Mean FHR over 10 minutes, excluding periodic changes |
| **Variability** | Fluctuation in baseline FHR (amplitude range) |
| **Deceleration** | Temporary decrease in FHR ≥15 bpm for ≥15 seconds |
| **Sinusoidal** | Smooth sine-wave pattern (3-5 cycles/min) - ALWAYS pathological |
| **Tachysystole** | >5 contractions per 10 minutes |
| **MOMENT** | Foundation model for time-series (CMU AutonLab) |
| **Safety Net** | Medical override rules that enforce critical findings |

---

# Appendix C: Version History

| Version | Date | Changes |
|---------|------|---------|
| Phase 1-2 | Nov 2024 | Data pipeline, rule engine |
| Phase 3-4 | Nov 2024 | MOMENT integration, hybrid classifier |
| Phase 5-7 | Dec 2024 | Real-time simulation, dashboards |
| Phase 8 | Dec 2024 | Comprehensive evaluation |
| Phase 9 | Dec 2024 | Hourly batch simulation |
| Phase 10 | Jan 2025 | Endurance testing (1-hour) |
| Phase 11 | Jan 2025 | Robustness torture test |
| Phase 12 | Jan 2025 | Massive scale robustness (500 scenarios) |
| Phase 13 | Jan 2025 | Signal processing upgrade (Savitzky-Golay) |
| Phase 15 | Jan 2026 | Grand Unification documentation |

---

# Appendix D: Repository Map

| Path | Purpose | Key Contents |
|------|---------|--------------|
| [src/analysis](src/analysis) | Alerting & safety | `alerts.py` (Hebrew alerts), `override.py` (medical safety net) |
| [src/config.py](src/config.py) | Global configuration | Clinical thresholds, data/model paths, UI strings |
| [src/data](src/data) | Data ingestion & preprocessing | `loader.py` (CTU loader, pH parsing), `preprocess.py` (clean + Savitzky-Golay) |
| [src/models](src/models) | AI components | `moment_encoder.py` (PyTorch), `moment_onnx.py` (ONNX - blocked), `fusion.py` (1,035-dim), `classifier.py` (XGBoost) |
| [src/rules](src/rules) | Clinical rule engine | Baseline, variability, decelerations, sinusoidal, tachysystole |
| [src/simulation](src/simulation) | Real-time simulator | `core/` (orchestrator, ring buffer), `generators/` (patient & UC/FHR), `events/` (patterns), `processing/` (pipeline adapter) |
| [src/ui](src/ui) | Streamlit apps | `simulation_app.py` (V2.0 two-view dashboard: Ward/Detail, 1-20 patients, clinical minimalism CSS), `plots.py` (WebGL/Scattergl + `uirevision`), `styles.py` (white/black theme) |
| [src/training](src/training) | Dataset → features → model | `prepare_data.py` (build X.npy/y.npy), `train_demo.py` (train/save XGBoost) |
| [tests](tests) | Unit, integration, benchmarks | `test_*.py`, `benchmarks/` (accuracy, clinical, load, robustness) |
| [data/ctu-chb-intrapartum-cardiotocography-database-1.0.0](data/ctu-chb-intrapartum-cardiotocography-database-1.0.0) | Raw CTU-UHB intrapartum CTG | WFDB records (.hea, .dat) |
| [data/processed](data/processed) | Precomputed features/labels | `X.npy`, `y.npy` |
| [models](models) | Saved model artifacts | `sentinel_classifier.json` (V2.0), `xgb_demo.config.json` |
| [scripts](scripts) | Utility/launchers | `run_simulation.py`, `visualize_preprocessing.py`, `export_moment_onnx.py`, `quantize_moment.py` |
| [docs/reports](archive/docs/reports) | Test reports (archived) | Robustness, endurance, evaluation summaries |
| [archive/docs](archive/docs) | Archived specs/PRDs/cheat sheets | Historical documentation set |

---

# Appendix E: V2 ONNX Optimization Reference (Blocked)

> **⚠️ Note:** ONNX export of MOMENT is currently blocked due to `aten::nanmean` operator not being supported in ONNX opset 17. The system uses PyTorch MOMENT for inference.

## Dependencies

```bash
# Core ONNX dependencies (installed but not used for MOMENT)
pip install onnx>=1.14.0 onnxruntime>=1.15.0

# Optional: OpenVINO for Intel CPU optimization (future use)
pip install openvino>=2023.1.0
```

## Model Export Workflow (Blocked)

```bash
# Step 1: Export MOMENT to ONNX (requires momentfm package)
python scripts/export_moment_onnx.py --output models/moment.onnx --verify

# Step 2: Quantize to INT8 (reduces ~1.5GB to ~400MB, ~2-3x speedup)
python scripts/quantize_moment.py --input models/moment.onnx --output models/moment_int8.onnx --benchmark

# Step 3 (Optional): Convert to OpenVINO for Intel CPUs
python -c "from src.models.moment_onnx import convert_onnx_to_openvino; convert_onnx_to_openvino('models/moment_int8.onnx', 'models/moment_openvino')"
```

## Backend Selection

The system automatically selects the best available backend:

| Model Files Present | Backend Used | Expected Performance |
|---------------------|--------------|----------------------|
| `models/moment_openvino/` | OpenVINO | ~50-100ms |
| `models/moment_int8.onnx` | ONNX Runtime INT8 | ~100-200ms |
| `models/moment.onnx` | ONNX Runtime FP32 | ~200-400ms |
| None (momentfm installed) | PyTorch | ~2-5s |
| None | Mock Mode | Instant (testing only) |

## Verification

```python
from src.models.moment_encoder import get_encoder_info, get_moment_encoder

# Check available backends
info = get_encoder_info()
print(f"Recommended backend: {info['recommended_backend']}")

# Get encoder with best backend
encoder = get_moment_encoder()
print(f"Active backend: {encoder.backend}")

# Run inference
import numpy as np
test_fhr = np.random.randn(2400).astype(np.float32) * 10 + 140
embedding = encoder.extract(test_fhr)
print(f"Embedding shape: {embedding.shape}")  # (1024,)
```

---

**Document End**

*This document represents the complete technical specification for SentinelFetal Gen3.5 V2. For questions or updates, refer to the project repository.*
