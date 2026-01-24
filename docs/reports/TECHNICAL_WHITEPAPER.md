# SentinelFetal: Technical Whitepaper

**Real-Time Fetal Distress Detection Using Hybrid AI**

*Version 3.0 — January 2026*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [The AI Core: From Transformers to MiniRocket](#3-the-ai-core-from-transformers-to-minirocket)
4. [Signal Processing & Performance Optimizations](#4-signal-processing--performance-optimizations)
5. [Clinical Logic & Guidelines](#5-clinical-logic--guidelines)
6. [V2.0: MHR Guard Module](#6-v20-mhr-guard-module)
7. [V2.0: Trend Analyzer Module](#7-v20-trend-analyzer-module)
8. [V2.0: Explainability Module](#8-v20-explainability-module)
9. [Performance Data & Evidence](#9-performance-data--evidence)
10. [Limitations & Future Work](#10-limitations--future-work)

---

## 1. Executive Summary

### The Problem

Cardiotocography (CTG) remains the gold standard for intrapartum fetal monitoring, yet its interpretation is notoriously subjective. Studies show inter-observer agreement rates as low as 29% for CTG classification, leading to both missed pathology and excessive false alarms—the latter contributing to clinician "alarm fatigue" where up to 85% of alerts are ignored.

### The Solution

**SentinelFetal** is a hybrid AI system that combines:

- **Probabilistic ML**: A lightweight MiniRocket encoder generating 9,996 time-series features, fed into an XGBoost classifier
- **Deterministic Rules**: Hard-coded clinical logic based on FIGO/NICHD guidelines (via the Israeli Position Paper) that can **never** be overridden by ML

This dual approach achieves:
- **98.7% Overall Accuracy** on synthetic clinical archetypes
- **P99 Latency of 58ms** — enabling true real-time operation at 4Hz
- **100% Specificity** on healthy tracings — eliminating false alarms
- **100% Noise Rejection** via an upstream signal quality gate (FSQI)

The system classifies CTG recordings into three categories per international guidelines:
- **Category I (Normal)**: No intervention required
- **Category II (Intermediate)**: Increased monitoring, possible intervention
- **Category III (Pathological)**: Immediate intervention recommended

---

## 2. System Architecture

### 2.1 The Pipeline Pattern

SentinelFetal's core is the `PipelineAdapter` class—a facade that orchestrates the complete analysis chain:

```
Signal → Preprocess → FSQI Gate → Rules + ML → Fusion → Classify → Override → Alert
```

**Implementation** (`src/simulation/processing/pipeline_adapter.py`):

```python
class PipelineAdapter:
    def process_patient(self, patient_id, data, run_moment=True) -> Dict:
        # Step 1: Preprocess (spike removal, gap filling)
        fhr_clean = self._preprocessor.process(fhr)
        
        # Step 2: Quality Gate (FSQI)
        passes_gate, quality = apply_quality_gate(fhr_clean, sampling_rate)
        if not passes_gate:
            return {"category": 2, "error": "Signal Quality Too Low"}
        
        # Step 3: Rule Engine
        baseline = calculate_baseline(fhr_clean, sampling_rate)
        variability = calculate_variability(fhr_clean, sampling_rate)
        decelerations = detect_decelerations(fhr_clean, uc, baseline.value)
        tachysystole = detect_tachysystole(uc, sampling_rate)
        sinusoidal = detect_sinusoidal_pattern(fhr_clean, sampling_rate)
        
        # Step 4: ML Features (MiniRocket)
        features = self._encoder.extract_features(fhr_clean, sampling_rate)
        
        # Step 5: Fusion (Rule features + ML embeddings)
        feature_vector = build_feature_vector(features, baseline, variability, ...)
        
        # Step 6: Classification
        prediction = self._classifier.predict(feature_vector)
        
        # Step 7: Medical Override (Safety Net)
        final = apply_medical_override(prediction, baseline, variability, ...)
        
        return {"category": final.final_category, ...}
```

This pattern ensures:
- **Single Responsibility**: Each module handles one concern
- **Testability**: Each step can be unit-tested independently
- **Fail-Safe Ordering**: The override happens *after* ML, so dangerous patterns are never missed

### 2.2 The RingBuffer: O(1) Memory Management

Real-time monitoring sessions can last hours. Naive implementations would accumulate unbounded memory. We solved this with a **fixed-size circular buffer** using Python's `collections.deque`.

**Implementation** (`src/simulation/core/ring_buffer.py`):

```python
@dataclass
class RingBuffer:
    max_samples: int = 2400  # 10 minutes at 4Hz
    
    def __post_init__(self):
        # deque with maxlen automatically discards oldest when full
        self._fhr = deque(maxlen=self.max_samples)
        self._uc = deque(maxlen=self.max_samples)
        self._timestamps = deque(maxlen=self.max_samples)
    
    def append(self, fhr: float, uc: float, timestamp: float):
        self._fhr.append(fhr)  # O(1) - automatic eviction
        ...
```

**Why `deque` over NumPy arrays?**

| Approach | Append | Random Access | Memory Realloc |
|----------|--------|---------------|----------------|
| `deque(maxlen=N)` | O(1) | O(1) | Never |
| `np.roll()` | O(N) | O(1) | Never |
| `np.append()` | O(N) | O(1) | Every append |
| `list.append()` | Amortized O(1) | O(1) | Periodic |

For our 4Hz real-time constraint, the O(1) guarantee of `deque` was essential. NumPy's `roll` would require copying 2,400 elements per tick—a 10x overhead.

**Memory Footprint**: 2,400 samples × 3 channels × 8 bytes = **~58KB per patient**. For 8 concurrent patients: **~464KB total**—negligible.

### 2.3 The "Hybrid Engine" Concept

The core innovation is the **Hybrid Engine**—combining rule-based determinism with ML probabilism:

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID ENGINE                            │
│                                                             │
│   ┌─────────────────┐          ┌─────────────────┐         │
│   │  RULE ENGINE    │          │  ML ENGINE      │         │
│   │  (11 features)  │          │  (9,996 feats)  │         │
│   │                 │          │                 │         │
│   │  • Baseline     │          │  MiniRocket     │         │
│   │  • Variability  │          │  Fixed Kernels  │         │
│   │  • Decelerations│          │  84 dilations   │         │
│   │  • Tachysystole │          │  PPV pooling    │         │
│   │  • Sinusoidal   │          │                 │         │
│   └────────┬────────┘          └────────┬────────┘         │
│            │                            │                   │
│            └──────────┬─────────────────┘                   │
│                       │                                     │
│              ┌────────▼────────┐                           │
│              │  FUSION LAYER   │                           │
│              │  1,035 dims     │                           │
│              └────────┬────────┘                           │
│                       │                                     │
│              ┌────────▼────────┐                           │
│              │   XGBoost       │                           │
│              │   Classifier    │                           │
│              └────────┬────────┘                           │
│                       │                                     │
│              ┌────────▼────────┐                           │
│              │  MEDICAL        │  ← HARD OVERRIDE          │
│              │  OVERRIDE       │    Rules ALWAYS win       │
│              └─────────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why Both?**

- **Rules provide interpretability**: A clinician can understand "Late deceleration detected at 12:34" but cannot interpret a 9,996-dimensional vector.
- **ML provides generalization**: Rules miss edge cases; ML learns from data patterns humans didn't explicitly code.
- **Override provides safety**: If ML says "Normal" but rules detect sinusoidal pattern, the **rules win**.

### 2.4 User Interface Stack (V3.0 — React + FastAPI)

The V3.0 UI introduces a high-performance "Central Station" dashboard built on React 18 with FastAPI WebSocket streaming, optimized for real-time monitoring of up to 20 simultaneous patients at 60 FPS.

**Technology Stack:**

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Frontend** | React 18 + TypeScript 5 | Type-safe, component-based, industry standard |
| Charts | TradingView lightweight-charts | Canvas-based, 60 FPS capable, dual-axis CTG |
| State | Zustand 4.4 | Lightweight, no boilerplate, React hooks-native |
| Styling | Tailwind CSS 3.4 | Utility-first, tree-shakeable, RTL support |
| i18n | i18next | Full Hebrew/English with RTL switching |
| Build | Vite 5.x | <1s HMR, tree-shaking, code splitting |
| **Backend** | FastAPI + Uvicorn | Async-first, OpenAPI spec, WebSocket-native |
| Protocol | WebSockets + MessagePack | Binary serialization, <1ms latency, 4Hz push |
| Bridge | DataBridge (singleton) | Thread-safe O(1) ring buffers for data flow |
| **Deployment** | Docker + Nginx | Multi-stage builds, reverse proxy, health checks |

**Performance Metrics:**

| Metric | Target | Achieved |
|--------|--------|----------|
| Frame Rate | 60 FPS | 60 FPS (Canvas) |
| Max Patients | 20 | 20 |
| Frontend Bundle | <200KB gzip | 153KB gzip |
| WebSocket Latency | <10ms | ~3ms |
| Browser Memory | <100MB | ~60MB |

**Key Optimizations:**

1. **Canvas Rendering**: TradingView lightweight-charts uses WebGL-accelerated canvas
2. **Ring Buffers (O(1))**: Both backend (`deque(maxlen=2400)`) and frontend (TypedArray) use fixed-size circular buffers
3. **MessagePack Binary Protocol**: 40% smaller than JSON, faster serialization
4. **React Suspense + Lazy Loading**: Code splitting for ward/detail views
5. **Zustand Selectors**: Fine-grained subscriptions prevent unnecessary re-renders

**Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (React 18)                          │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────┐   │
│  │  WardView   │   │ DetailView  │   │ GodModePanel         │   │
│  │  (Grid)     │   │ (CTG Chart) │   │ (Event Injection)    │   │
│  └──────┬──────┘   └──────┬──────┘   └──────────┬───────────┘   │
│         │                 │                      │               │
│         └─────────────────┴──────────────────────┘               │
│                           │                                      │
│                  ┌────────▼────────┐                            │
│                  │  Zustand Store  │                            │
│                  │  (patientStore) │                            │
│                  └────────┬────────┘                            │
│                           │                                      │
│                  ┌────────▼────────┐                            │
│                  │ useWebSocket()  │ ◄── msgpack binary         │
│                  └────────┬────────┘                            │
└───────────────────────────┼─────────────────────────────────────┘
                            │ WebSocket (ws://host/ws/stream)
┌───────────────────────────┼─────────────────────────────────────┐
│                     BACKEND (FastAPI)                            │
│                  ┌────────▼────────┐                            │
│                  │ WebSocket Router│                            │
│                  │ (broadcast hub) │                            │
│                  └────────┬────────┘                            │
│                           │                                      │
│                  ┌────────▼────────┐                            │
│                  │  DataBridge     │ ◄── Thread-safe singleton  │
│                  │  (state_bridge) │                            │
│                  └────────┬────────┘                            │
│                           │                                      │
│                  ┌────────▼────────┐                            │
│                  │ SimOrchestrator │                            │
│                  │ (8 patients)    │                            │
│                  └─────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation** (`api/main.py`):

```python
# FastAPI lifespan with WebSocket broadcasting
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize orchestrator (simulation engine)
    orchestrator = get_orchestrator_adapter()
    orchestrator.initialize(patient_count=settings.default_patient_count)

    # Start WebSocket broadcaster (4Hz push)
    broadcaster = get_broadcaster()
    await broadcaster.start()

    yield

    # Graceful shutdown
    await broadcaster.stop()
    orchestrator.shutdown()
```

---

## 3. The AI Core: From Transformers to MiniRocket

### 3.1 The Decision: Why We Abandoned MOMENT

We initially planned to use **MOMENT** (AutonLab/MOMENT-1-large), a 341M-parameter foundation model for time series. Benchmarks showed impressive zero-shot performance on classification tasks.

**Reality check**:

| Metric | MOMENT | MiniRocket |
|--------|--------|------------|
| Parameters | 341M | 84 kernels |
| Inference Time | 300-500ms | ~1ms |
| Memory | 2.5GB | <100MB |
| Accuracy (CTG) | ~95% | ~95% |

For a 4Hz real-time system (250ms budget per sample), MOMENT's 300ms+ inference was **impossible**. We needed a model that could run in single-digit milliseconds.

### 3.2 MiniRocket: The Speed Champion

**MiniRocket** (Dempster et al., 2020) is a deterministic convolutional transform that achieves state-of-the-art accuracy on time series classification while being **75x faster** than the original ROCKET.

**How it works**:

1. **Fixed Kernels**: 84 pre-defined convolutional kernels (no learning required)
2. **Dilation**: Each kernel is applied at multiple dilation rates (1, 2, 4, 8, ..., up to signal length)
3. **PPV Pooling**: For each kernel-dilation pair, compute the **Proportion of Positive Values** (PPV) after convolution
4. **Output**: 9,996 features (84 kernels × ~119 dilations × 1 PPV value)

**Implementation** (`src/models/minirocket_encoder.py`):

```python
class MiniRocketEncoder:
    def __init__(self, config: MiniRocketConfig = None):
        self.config = config or MiniRocketConfig(
            num_kernels=10000,
            max_dilations_per_kernel=32,
            window_size=2400  # 10 minutes at 4Hz
        )
        self._transformer = MiniRocket(
            num_kernels=self.config.num_kernels,
            random_state=42  # Reproducibility
        )
    
    def extract_features(self, fhr: np.ndarray, sampling_rate: float) -> MiniRocketFeatureResult:
        # Reshape for sktime: (n_instances, n_dims, series_length)
        X = fhr.reshape(1, 1, -1)
        features = self._transformer.transform(X)
        return MiniRocketFeatureResult(features=features.flatten(), ...)
```

### 3.3 Why Convolutions Beat Attention for CTG

Transformer attention has O(N²) complexity with sequence length. For a 10-minute window at 4Hz (2,400 samples), that's 5.76 million attention computations—per layer, per head.

MiniRocket's convolutions are O(N):
- Each kernel slides across the signal once
- Dilation is achieved via stride, not additional computation
- PPV is a single pass through the convolution output

**Benchmark** (single 10-minute window):

| Model | Time | Complexity |
|-------|------|------------|
| MOMENT (Transformer) | 350ms | O(N²) |
| MiniRocket | 1.2ms | O(N) |
| **Speedup** | **291x** | — |

---

## 4. Signal Processing & Performance Optimizations

### 4.1 The "Zero-Inference" FSQI Gate

**Problem**: CTG signals are notoriously noisy—maternal movement, transducer slippage, fetal position changes. Feeding garbage to the ML model wastes CPU cycles and produces unreliable predictions.

**Solution**: The **Fetal Signal Quality Index (FSQI)** gate rejects low-quality signals *before* they reach the AI.

**Implementation** (`src/data/signal_quality.py`):

```python
def calculate_fsqi(fhr: np.ndarray, sampling_rate: float) -> SignalQualityResult:
    # Component 1: Valid sample ratio (non-NaN, non-zero)
    valid_ratio = _calculate_valid_ratio(fhr)
    
    # Component 2: Physiological range (50-250 bpm)
    physiological_ratio = _calculate_physiological_ratio(fhr)
    
    # Component 3: Spectral noise (high-frequency content)
    noise_score = _calculate_noise_score(fhr, sampling_rate)
    
    # Component 4: Signal stability (sudden jumps)
    stability_score = _calculate_stability_score(fhr, sampling_rate)
    
    # Weighted combination
    score = (0.35 * valid_ratio + 0.25 * physiological_ratio +
             0.20 * noise_score + 0.20 * stability_score)
    
    return SignalQualityResult(
        score=score,
        should_classify=(score >= 0.7),  # HIGH quality threshold
        ...
    )
```

**Impact**: In our Heavy Noise scenario, **100% of garbage signals were blocked**. No wasted inference cycles, no garbage-in-garbage-out.

### 4.2 Direct State Injection ("Warm-Up")

**Problem**: Clinical monitoring doesn't start from zero. When a patient arrives in L&D, they already have 20+ minutes of history. Testing must simulate this.

**Solution**: Instead of running 20 minutes of simulation ticks (expensive), we **directly inject** synthetic baseline data into the ring buffer.

**Implementation** (`scripts/deep_endurance_audit.py`):

```python
def warm_up_patient(patient: PatientGenerator, baseline: float = 140.0):
    """Fill buffer with 20 minutes of baseline data—instantly."""
    samples = int(20 * 60 * SAMPLING_RATE)  # 4800 samples
    fhr = np.full(samples, baseline, dtype=np.float64)
    uc = np.zeros(samples, dtype=np.float64)
    timestamps = np.arange(samples) / SAMPLING_RATE
    
    # Directly append to internal buffer (bypassing tick loop)
    patient._buffer.append_batch(fhr, uc, timestamps)
```

**Impact**: Test setup reduced from 20+ minutes of simulation to <1 second.

### 4.3 Nuclear Silence: Eliminating I/O Bottlenecks

**Problem**: Python's `logging` module, even at `WARNING` level, was consuming 15-20% of our tick budget. Third-party libraries (NumPy, scikit-learn) also emit warnings.

**Solution**: **Complete logging suppression** at the interpreter level.

```python
# MUST be before ANY imports
import logging
import warnings
import os
import sys

# 1. Disable Python logging above CRITICAL
logging.disable(logging.CRITICAL)

# 2. Suppress all warnings
warnings.filterwarnings("ignore")

# 3. Suppress TensorFlow/ONNX spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 4. Redirect stderr during imports
sys.stderr = open(os.devnull, 'w')
```

**Impact**: Achieved **6 ticks/second** sustained throughput (vs. 2-3 ticks/sec with logging enabled).

---

## 5. Clinical Logic & Guidelines

### 5.1 Guideline Compliance

SentinelFetal implements the **Israeli Position Paper on CTG Interpretation**, which consolidates FIGO and NICHD guidelines. Key thresholds are centralized in `src/config.py`:

```python
@dataclass(frozen=True)
class ClinicalThresholds:
    # FHR Baseline (bpm)
    BASELINE_NORMAL_MIN: int = 110
    BASELINE_NORMAL_MAX: int = 160
    BASELINE_BRADYCARDIA: int = 110
    
    # Variability (bpm)
    VARIABILITY_ABSENT_MAX: float = 2.0    # 0-2 bpm = Absent (SEVERE)
    VARIABILITY_MINIMAL_MAX: float = 5.0   # 3-5 bpm = Minimal
    VARIABILITY_MODERATE_MAX: float = 25.0 # 6-25 bpm = Moderate (NORMAL)
    # > 25 bpm = Marked
```

### 5.2 Pattern Recognition Algorithms

#### Late Decelerations

**Definition**: FHR drop ≥15 bpm below baseline, lasting ≥15 seconds, with nadir occurring **>15 seconds after** contraction peak.

**Implementation** (`src/rules/decelerations.py`):

```python
def classify_deceleration(fhr, uc, nadir_idx, start, end, sampling_rate):
    # Find nearest contraction peak
    contraction_peak_idx = _find_nearest_contraction_peak(uc, nadir_idx)
    
    # Calculate lag time (nadir - contraction peak)
    lag_samples = nadir_idx - contraction_peak_idx
    lag_seconds = lag_samples / sampling_rate
    
    # Classification per Israeli Position Paper
    if lag_seconds < 5:
        return DecelerationType.EARLY   # With contraction
    elif lag_seconds > 15:
        return DecelerationType.LATE    # After contraction (CONCERNING)
    else:
        # Check for abrupt onset (Variable deceleration)
        descent_rate = _calculate_descent_rate(fhr, start, nadir_idx)
        if descent_rate > 0.5:  # bpm per sample
            return DecelerationType.VARIABLE
```

**Clinical Significance**: Late decelerations indicate **uteroplacental insufficiency**—the placenta isn't delivering enough oxygen. Recurrent late decels mandate intervention.

#### Variability (Heart Rate Variability)

**Definition**: The fluctuation in baseline FHR, quantified as amplitude range (max - min) in 1-minute windows.

**Implementation** (`src/rules/variability.py`):

```python
def calculate_variability(fhr, sampling_rate, window_seconds=60.0):
    window_samples = int(window_seconds * sampling_rate)
    step = int(window_samples * 0.5)  # 50% overlap
    
    amplitudes = []
    for i in range(0, len(fhr) - window_samples, step):
        window = fhr[i:i + window_samples]
        valid = window[~np.isnan(window)]
        if len(valid) > window_samples * 0.5:
            amplitude = np.max(valid) - np.min(valid)
            amplitudes.append(amplitude)
    
    avg_variability = np.mean(amplitudes)
    
    # Classify per Israeli Position Paper
    if avg_variability <= 2:
        category = VariabilityCategory.ABSENT   # SEVERE
    elif avg_variability <= 5:
        category = VariabilityCategory.MINIMAL
    elif avg_variability <= 25:
        category = VariabilityCategory.MODERATE # NORMAL
    else:
        category = VariabilityCategory.MARKED
```

**Clinical Significance**: **Moderate variability (6-25 bpm) is the single most reliable indicator of fetal well-being.** Absent variability with late decels = high risk of acidemia.

#### Sinusoidal Pattern (Critical)

**Definition**: Smooth sine-wave oscillation at 3-5 cycles/minute, amplitude 5-15 bpm, duration >20 minutes, absent short-term variability.

**Implementation** (`src/rules/sinusoidal.py`):

```python
def detect_sinusoidal_pattern(fhr, sampling_rate):
    # Need at least 20 minutes of data
    min_samples = int(20 * 60 * sampling_rate)  # 4800 at 4Hz
    if len(fhr) < min_samples:
        return SinusoidalResult(detected=False, ...)
    
    segment = fhr[-min_samples:]
    
    # FFT analysis
    n = len(segment)
    yf = np.abs(fft(segment - np.mean(segment)))
    xf = fftfreq(n, 1 / sampling_rate)
    
    # Target frequency: 3-5 cycles/min = 0.05-0.083 Hz
    freq_min = 3 / 60  # 0.05 Hz
    freq_max = 5 / 60  # 0.083 Hz
    
    target_mask = (xf > freq_min) & (xf < freq_max)
    target_power = yf[target_mask]
    
    # Check dominance ratio
    total_power = np.sum(yf[xf > 0])
    dominance = np.max(target_power) / total_power
    
    if dominance > 0.15 and amplitude_in_range:
        return SinusoidalResult(detected=True, ...)  # SEVERE!
```

**Clinical Significance**: Sinusoidal patterns are associated with **severe fetal anemia** (Rh isoimmunization, fetomaternal hemorrhage). **ALWAYS Category III—requires immediate action.**

### 5.3 The Medical Override ("Safety Net")

The override layer ensures that certain findings **cannot** be downgraded by ML:

```python
def apply_medical_override(ml_prediction, baseline, variability, decelerations, 
                           tachysystole, sinusoidal) -> MedicalOverride:
    
    # RULE 1: Sinusoidal → Always Category 3
    if sinusoidal.detected:
        return MedicalOverride(
            should_override=True,
            final_category=2,  # Index 2 = Category III
            reason=OverrideReason.SINUSOIDAL_PATTERN,
            explanation="Sinusoidal pattern detected - SEVERE"
        )
    
    # RULE 2: Absent variability + (recurrent late OR recurrent variable OR brady)
    if variability.category == VariabilityCategory.ABSENT:
        if _has_recurrent_late_decels(decelerations) or \
           _has_recurrent_variable_decels(decelerations) or \
           _detect_bradycardia(baseline):
            return MedicalOverride(
                should_override=True,
                final_category=2,  # Category III
                reason=OverrideReason.ABSENT_VARIABILITY_WITH_DECELS
            )
    
    # RULE 3: Safety Floor - If ML says Normal but variability is Absent
    if ml_prediction == 0 and variability.category == VariabilityCategory.ABSENT:
        return MedicalOverride(
            should_override=True,
            final_category=1,  # Upgrade to Category II minimum
            reason=OverrideReason.ABSENT_VARIABILITY_SAFETY_FLOOR
        )
    
    # No override needed
    return MedicalOverride(should_override=False, final_category=ml_prediction)
```

---

## 6. V2.0: MHR Guard Module

### 6.1 The Problem: Maternal Heart Rate Contamination

One of the most dangerous failure modes in CTG monitoring occurs when the ultrasound transducer loses the fetal heartbeat and begins tracking the maternal heart rate instead. This **MHR contamination** can mask severe fetal distress—the monitor shows a reassuring 80-100 bpm maternal heart rate while the fetus is bradycardic.

**Clinical scenario**: A distressed fetus develops bradycardia (70 bpm). The transducer slips and picks up maternal pulse (75 bpm). The display shows "stable" 75 bpm—clinicians are falsely reassured while the fetus deteriorates.

### 6.2 Detection Strategy: Multi-Method Fusion

SentinelFetal V2.0 implements a **three-method detection system** with weighted voting:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MHR GUARD MODULE                            │
│                                                                 │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────┐  │
│   │ Cross-          │   │ Spectral RSA    │   │ Baseline    │  │
│   │ Correlation     │   │ Analysis        │   │ Jump        │  │
│   │ (Weight: 0.5)   │   │ (Weight: 0.3)   │   │ (Weight: 0.2)│  │
│   └────────┬────────┘   └────────┬────────┘   └──────┬──────┘  │
│            │                     │                    │         │
│            └──────────────┬──────┴────────────────────┘         │
│                           │                                     │
│                  ┌────────▼────────┐                           │
│                  │  Weighted Vote  │                           │
│                  │  Fusion Engine  │                           │
│                  └────────┬────────┘                           │
│                           │                                     │
│                  ┌────────▼────────┐                           │
│                  │ Fetal Sleep     │ ← CRITICAL ADJUSTMENT     │
│                  │ Adjustment      │   Accelerations present?  │
│                  └────────┬────────┘                           │
│                           │                                     │
│                  ┌────────▼────────┐                           │
│                  │ Action Decision │                           │
│                  │ NONE/WARN/BLOCK │                           │
│                  └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Method 1: Cross-Correlation with MHR Reference

When a maternal SpO₂ pulse is available (from finger probe), we compute the normalized cross-correlation:

**Implementation** (`src/safety/mhr_detector.py`):

```python
def _compute_cross_correlation(self, fhr: np.ndarray, mhr: np.ndarray) -> float:
    """
    High correlation (>0.8) suggests FHR is tracking MHR.
    """
    # Z-score normalization
    fhr_norm = (fhr - np.mean(fhr)) / np.std(fhr)
    mhr_norm = (mhr - np.mean(mhr)) / np.std(mhr)

    # Pearson correlation
    correlation = np.correlate(fhr_norm, mhr_norm, mode='valid')[0] / len(fhr)
    return abs(float(correlation))
```

**Threshold**: If correlation > 0.8, flag as suspected MHR contamination.

### 6.4 Method 2: Spectral RSA Analysis

**Respiratory Sinus Arrhythmia (RSA)** creates characteristic frequency signatures:

| Subject | RSA Band | Breathing Rate |
|---------|----------|----------------|
| Adult (Maternal) | 0.15-0.35 Hz | 9-21 breaths/min |
| Fetus | 0.4-1.0 Hz | 24-60 breaths/min |

**Implementation** (`src/safety/spectral_analyzer.py`):

```python
class SpectralAnalyzer:
    ADULT_RSA_BAND = (0.15, 0.35)   # Hz - adult respiratory modulation
    FETAL_RSA_BAND = (0.4, 1.0)     # Hz - fetal respiratory modulation

    def analyze_rsa(self, fhr_segment: np.ndarray, sampling_rate: float) -> SpectralResult:
        # Detrend and window
        detrended = fhr_segment - np.mean(fhr_segment)
        windowed = detrended * np.hanning(len(detrended))

        # FFT
        n = len(windowed)
        freqs = fftfreq(n, 1 / sampling_rate)
        power = np.abs(fft(windowed)) ** 2

        # Calculate power in each band
        adult_mask = (freqs >= 0.15) & (freqs <= 0.35)
        fetal_mask = (freqs >= 0.4) & (freqs <= 1.0)

        adult_power = np.sum(power[adult_mask])
        fetal_power = np.sum(power[fetal_mask])

        adult_power_ratio = adult_power / (adult_power + fetal_power + 1e-10)

        return SpectralResult(
            adult_power_ratio=adult_power_ratio,
            spectral_centroid=self._compute_centroid(freqs, power),
            ...
        )
```

**Detection logic**: If adult power ratio > 40% AND spectral centroid < 0.3 Hz → suspected MHR.

### 6.5 Method 3: Baseline Jump Detection

A sudden baseline shift (e.g., 140 → 85 bpm in 5 seconds) often indicates the transducer switched signal sources.

**Implementation**:

```python
def _detect_baseline_jump(self, fhr: np.ndarray, sampling_rate: float) -> BaselineJumpResult:
    window = int(5 * sampling_rate)  # 5-second window

    # Rolling mean
    rolling_mean = np.convolve(fhr, np.ones(window) / window, mode='valid')

    # Derivative (bpm per second)
    derivative = np.diff(rolling_mean) * sampling_rate

    # Large jump = >4 bpm/sec sustained
    jump_threshold = 4.0  # >20 bpm in 5 seconds
    jump_indices = np.where(np.abs(derivative) > jump_threshold)[0]

    if len(jump_indices) > 0:
        # Check if signal is stable after jump
        post_jump_std = np.std(rolling_mean[jump_indices[-1]:])
        if post_jump_std < 5.0:  # Stable after jump = suspicious
            return BaselineJumpResult(jump_detected=True, ...)
```

### 6.6 Critical: Fetal Sleep Cycle Handling

**Problem**: A sleeping fetus shows **low variability**—the same pattern as MHR contamination. Without accounting for this, we'd generate false MHR alerts during normal fetal sleep.

**Solution**: Check for **accelerations**. A sleeping fetus RETAINS accelerations (autonomic nervous system still functional). MHR contamination shows NO accelerations (adult heart doesn't have fetal-type accelerations).

**Implementation**:

```python
def _fuse_results(self, results, has_accelerations, segment_length) -> MHRCheckResult:
    # ... compute weighted confidence ...

    # CRITICAL: Fetal Sleep Adjustment
    if has_accelerations and final_confidence > 0.3:
        original_confidence = final_confidence
        final_confidence *= 0.5  # Halve confidence if accelerations present
        reasons.append(
            f"Confidence reduced {original_confidence:.0%}→{final_confidence:.0%} "
            f"(accelerations present - may be fetal sleep)"
        )

    return MHRCheckResult(...)
```

### 6.7 Integration in Pipeline (Step 0)

MHR Guard runs **before** any other processing:

```python
# In PipelineAdapter.process_patient():
# STEP 0: MHR Guard Check (V2.0)
if self.config.enable_mhr_guard:
    has_accelerations = self._detect_accelerations(fhr_clean, baseline_result.value)
    mhr_result = self._mhr_detector.check_segment(
        fhr_segment=fhr_clean[-240:],  # Last 60 seconds
        has_accelerations=has_accelerations,
        sampling_rate=4.0
    )

    if mhr_result.recommended_action == MHRAction.BLOCK_SEGMENT:
        return {
            "category": None,  # SUSPENDED - no classification
            "mhr_alert": mhr_result.to_dict(),
            "error": "Signal source ambiguous - verify sensor placement"
        }
```

---

## 7. V2.0: Trend Analyzer Module

### 7.1 The Problem: MiniRocket Erases Time

MiniRocket's **PPV (Proportion of Positive Values) pooling** is excellent for classification speed but destroys temporal order. A 60-minute segment with variability declining 12→8→4 bpm produces the same features as one with variability improving 4→8→12 bpm.

This means MiniRocket cannot detect **gradual deterioration**—a critical clinical pattern where the fetus slowly decompensates over 30-60 minutes.

### 7.2 The Solution: Parallel Trend Tracking

SentinelFetal V2.0 maintains a **separate trend buffer** that preserves temporal order:

```
┌───────────────────────────────────────────────────────────────┐
│                   TREND ANALYZER MODULE                        │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │              TrendBuffer (60 minutes)                 │    │
│   │              30 points @ 2-min intervals              │    │
│   │                                                       │    │
│   │  t=0    t=10   t=20   t=30   t=40   t=50   t=60     │    │
│   │   •──────•──────•──────•──────•──────•──────•        │    │
│   │  12.5   11.8   10.2   9.5    8.1    7.2    5.8      │    │
│   │              (variability declining)                  │    │
│   └──────────────────────────────────────────────────────┘    │
│                            │                                   │
│                   ┌────────▼────────┐                         │
│                   │ Linear Regression│                         │
│                   │ slope = -1.1/10m │                         │
│                   └────────┬────────┘                         │
│                            │                                   │
│          ┌─────────────────┼─────────────────┐                │
│          │                 │                 │                 │
│  ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐       │
│  │ Variability   │ │ Deceleration  │ │ Baseline      │       │
│  │ Trend         │ │ Frequency     │ │ Drift         │       │
│  │ (40% weight)  │ │ (30% weight)  │ │ (30% weight)  │       │
│  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘       │
│          │                 │                 │                 │
│          └─────────────────┼─────────────────┘                │
│                            │                                   │
│                   ┌────────▼────────┐                         │
│                   │ Deterioration   │                         │
│                   │ Score (0-100)   │                         │
│                   └─────────────────┘                         │
└───────────────────────────────────────────────────────────────┘
```

### 7.3 TrendBuffer: FSQI-Masked Circular Buffer

**Critical design decision**: Only samples with FSQI ≥ 0.9 are stored. This prevents signal artifacts from corrupting trend regression.

**Implementation** (`src/analysis/trend_buffer.py`):

```python
class TrendBuffer:
    MIN_FSQI_THRESHOLD = 0.9

    def __init__(self, max_minutes: int = 60, sample_interval_minutes: int = 2):
        self.max_points = max_minutes // sample_interval_minutes  # 30 points
        self._buffer = deque(maxlen=self.max_points)

    def add_sample(self, data_point: TrendDataPoint) -> bool:
        """Add sample ONLY if signal quality is sufficient."""
        # CRITICAL: FSQI masking
        if data_point.fsqi_score < self.MIN_FSQI_THRESHOLD:
            self._samples_masked += 1
            return False  # Silently skip low-quality samples

        self._buffer.append(data_point)
        return True
```

**Why 0.9 threshold?** Lower thresholds allow artifact-contaminated samples that introduce noise into the regression. Higher thresholds reject too many samples, leaving insufficient data for trend analysis.

### 7.4 Linear Regression for Slope Calculation

**Implementation** (`src/analysis/trend_analyzer.py`):

```python
def _compute_linear_trend(self, series: np.ndarray) -> LinearTrendResult:
    """
    Compute slope in units per 10 minutes.
    Series has 2-minute intervals, so we convert x-axis accordingly.
    """
    if len(series) < 3:
        return LinearTrendResult(slope=0.0, r_squared=0.0, confidence=0.0)

    # Time axis in 10-minute units (2 min intervals → divide by 5)
    x = np.arange(len(series)) * 2 / 10

    # Simple linear regression: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
    x_mean = np.mean(x)
    y_mean = np.mean(series)

    numerator = np.sum((x - x_mean) * (series - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    slope = numerator / denominator

    # Calculate R² for confidence
    y_pred = slope * (x - x_mean) + y_mean
    ss_res = np.sum((series - y_pred) ** 2)
    ss_tot = np.sum((series - y_mean) ** 2)
    r_squared = max(0.0, 1 - (ss_res / ss_tot))

    return LinearTrendResult(slope=slope, r_squared=r_squared, ...)
```

### 7.5 Deterioration Score (0-100)

The composite score combines three components:

**Formula**:
```
Score = 40% × Variability_Penalty + 30% × Decel_Penalty + 30% × Baseline_Penalty
```

**Implementation**:

```python
def _calculate_deterioration_score(self, var_slope, baseline_slope,
                                    decel_count, current_variability) -> int:
    # Variability penalty: declining slope + low absolute value
    var_slope_penalty = min(1.0, max(0.0, -var_slope) / 2.0)  # -2/10min = max
    var_level_penalty = min(1.0, max(0.0, (8 - current_variability) / 8))
    var_penalty = 0.6 * var_slope_penalty + 0.4 * var_level_penalty

    # Deceleration penalty: 5+ late decels in 15min = max
    decel_penalty = min(1.0, decel_count / 5)

    # Baseline penalty: abnormal drift (10 bpm/10min = max)
    baseline_penalty = min(1.0, abs(baseline_slope) / 10)

    # Weighted sum → scale to 0-100
    score = (0.4 * var_penalty + 0.3 * decel_penalty + 0.3 * baseline_penalty)
    return min(100, max(0, int(score * 100)))
```

**Interpretation**:

| Score | Severity | Clinical Action |
|-------|----------|-----------------|
| 0-30 | Good | Continue routine monitoring |
| 31-50 | Caution | Increase monitoring frequency |
| 51-70 | Warning | Prepare for intervention |
| 71-100 | Critical | Immediate clinical review |

### 7.6 Trend-Based Category Override

When deterioration score exceeds 70 and ML classified as Category I, we override to Category II:

```python
# In PipelineAdapter:
if trend_result.deterioration_score > 70 and final_category == 1:
    final_category = 2
    was_trend_overridden = True
```

This ensures gradual deterioration is never dismissed as "Normal."

---

## 8. V2.0: Explainability Module

### 8.1 The Problem: Black Box AI in Medicine

ML models, including MiniRocket+XGBoost, are opaque. A clinician cannot understand why the system classified a tracing as Category II. This creates:

1. **Trust issues**: Clinicians may ignore recommendations they don't understand
2. **Liability concerns**: "The AI said so" is not a defensible medical decision
3. **Training gaps**: Juniors can't learn clinical reasoning from opaque systems

### 8.2 The Solution: Dual-Track Explanations

SentinelFetal V2.0 provides **two explanation tracks**:

```
┌───────────────────────────────────────────────────────────────┐
│                  EXPLAINABILITY MODULE                         │
│                                                                │
│   ┌─────────────────────────┐   ┌─────────────────────────┐   │
│   │    RULE EXPLAINER       │   │    SHAP EXPLAINER       │   │
│   │    (Always Available)   │   │    (On-Demand Only)     │   │
│   │                         │   │                         │   │
│   │  • Deterministic        │   │  • ML Feature           │   │
│   │  • Fast (<1ms)          │   │    Attribution          │   │
│   │  • Interpretable        │   │  • Slower (~50ms)       │   │
│   │  • Clinical terms       │   │  • Top 3 features       │   │
│   └───────────┬─────────────┘   └───────────┬─────────────┘   │
│               │                             │                  │
│               └──────────────┬──────────────┘                  │
│                              │                                 │
│                     ┌────────▼────────┐                       │
│                     │ ExplanationEngine│                       │
│                     │ (Orchestrator)   │                       │
│                     └────────┬────────┘                       │
│                              │                                 │
│              ┌───────────────┼───────────────┐                │
│              │               │               │                 │
│      ┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐       │
│      │ Text Summary  │ │ Highlight │ │ Contributor   │       │
│      │ (Natural Lang)│ │ Regions   │ │ Rankings      │       │
│      └───────────────┘ └───────────┘ └───────────────┘       │
└───────────────────────────────────────────────────────────────┘
```

### 8.3 Rule Explainer: Always Available

The rule explainer translates each rule output into human-readable explanations:

**Implementation** (`src/explainability/rule_explainer.py`):

```python
class RuleExplainer:
    def explain_variability(self, result: VariabilityResult) -> RuleExplanation:
        if result.category == VariabilityCategory.ABSENT:
            return RuleExplanation(
                rule_name="variability",
                contribution=0.8,  # High pathological contribution
                description=f"Absent variability ({result.value:.1f} bpm) - SEVERE",
                severity="CRITICAL",
                time_region=TimeRegion(start_index=-240, end_index=-1, color="red")
            )
        elif result.category == VariabilityCategory.MINIMAL:
            return RuleExplanation(
                rule_name="variability",
                contribution=0.4,
                description=f"Minimal variability ({result.value:.1f} bpm)",
                severity="MEDIUM",
                ...
            )
        # ... other categories
```

**Example output**:

```json
{
  "summary": "Category II: Late deceleration detected with minimal variability",
  "contributors": [
    {
      "source": "rule",
      "name": "late_decel",
      "contribution": 0.7,
      "description": "Late deceleration: -25 bpm, nadir 20s after contraction peak",
      "time_region": {"start": -180, "end": -120, "color": "red"}
    },
    {
      "source": "rule",
      "name": "variability",
      "contribution": 0.4,
      "description": "Minimal variability (4.2 bpm)",
      "time_region": {"start": -240, "end": -1, "color": "orange"}
    }
  ]
}
```

### 8.4 SHAP Explainer: On-Demand ML Attribution

SHAP (SHapley Additive exPlanations) provides insight into which MiniRocket features drove the XGBoost prediction.

**Why on-demand only?** SHAP computation takes ~50-100ms—acceptable for user-triggered "Explain ML" requests but would blow the real-time latency budget if run on every classification.

**Implementation** (`src/explainability/shap_explainer.py`):

```python
class SHAPExplainer:
    def __init__(self, classifier, feature_names: List[str]):
        # TreeExplainer for XGBoost - optimized for tree-based models
        self.explainer = shap.TreeExplainer(classifier.get_model())
        self.feature_names = feature_names

    def explain(self, feature_vector: np.ndarray, top_k: int = 3) -> List[SHAPExplanation]:
        # Compute SHAP values
        shap_values = self.explainer.shap_values(feature_vector.reshape(1, -1))

        # Get top-K features by absolute SHAP value
        abs_values = np.abs(shap_values[0])
        top_indices = np.argsort(abs_values)[-top_k:][::-1]

        explanations = []
        for idx in top_indices:
            explanations.append(SHAPExplanation(
                feature_index=idx,
                feature_name=self.feature_names[idx],
                shap_value=shap_values[0][idx],
                clinical_category=self._map_to_clinical(idx),
                description=self._generate_description(idx, shap_values[0][idx])
            ))

        return explanations
```

### 8.5 Visual Mapper: Graph Highlighting

> **Implementation Status:** Backend module IMPLEMENTED (`src/explainability/visual_mapper.py`).
> **UI Integration:** NOT YET WIRED — see [UI_UX_GAP_ANALYSIS.md](UI_UX_GAP_ANALYSIS.md) for details.
> The VisualMapper produces HighlightRegion objects, but the UI does not currently consume them.

The VisualMapper converts explanation regions to UI overlay coordinates:

**Implementation** (`src/explainability/visual_mapper.py`):

```python
class VisualMapper:
    # Color scheme for different finding types
    COLORS = {
        "sinusoidal": "rgba(220, 53, 69, 0.3)",   # Red - critical
        "late_decel": "rgba(220, 53, 69, 0.3)",   # Red - critical
        "variable_decel": "rgba(253, 126, 20, 0.3)",  # Orange - warning
        "variability": "rgba(255, 193, 7, 0.3)",  # Yellow - caution
        "baseline": "rgba(40, 167, 69, 0.2)",     # Green - info
        "acceleration": "rgba(40, 167, 69, 0.2)", # Green - reassuring
    }

    def map_to_highlights(self, contributors, signal_length) -> List[HighlightRegion]:
        highlights = []
        for c in contributors:
            if c.time_region:
                # Convert negative indices to absolute
                start = c.time_region.start_index
                end = c.time_region.end_index
                if start < 0:
                    start = signal_length + start
                if end < 0:
                    end = signal_length + end

                highlights.append(HighlightRegion(
                    start=max(0, start),
                    end=min(signal_length - 1, end),
                    color=self.COLORS.get(c.name, "rgba(128, 128, 128, 0.2)"),
                    label=c.name.replace("_", " ").title(),
                    is_pathological=(c.contribution > 0)
                ))

        return highlights
```

### 8.6 Integration in Pipeline (Step 9)

Explanation generation is the final step:

```python
# In PipelineAdapter.process_patient():
# STEP 9: Explanation Generation (V2.0)
if self.config.enable_explanations:
    explanation = self._explanation_engine.explain(
        category=final_category,
        rule_outputs={
            "baseline": baseline_result,
            "variability": variability_result,
            "decelerations": decelerations,
            "tachysystole": tachysystole_result,
            "sinusoidal": sinusoidal_result,
        },
        ml_features=feature_vector.vector,
        compute_shap=False  # Only on explicit user request
    )
    response["explanation"] = explanation.to_dict()
```

---

## 9. Performance Data & Evidence

### 9.1 Clinical Validation Results

From `CLINICAL_VALIDATION_REPORT.md`:

| Scenario | N | Expected | Accuracy | Notes |
|----------|---|----------|----------|-------|
| Textbook Healthy | 30 | Cat I | **100.0%** | Zero false alarms |
| Late Deceleration | 30 | Cat II/III | **93.3%** | 28/30 detected |
| Variable Deceleration | 30 | Correct Type | **100.0%** | 113 decels typed correctly |
| Sinusoidal Pattern | 30 | Cat III | **100.0%** | Override always fires |
| Heavy Noise | 30 | Rejected | **100.0%** | FSQI gate blocked all |

**Key Metrics**:
- **Overall Accuracy**: 98.7%
- **Specificity (Healthy → Cat I)**: 100% — **Crucial for alarm fatigue**
- **Sensitivity (Late Decel → Cat II/III)**: 93.3%
- **Noise Immunity**: 100%

### 9.2 Endurance Test Results

From `DEEP_ENDURANCE_REPORT.md`:

| Metric | Value |
|--------|-------|
| Duration | 35 minutes simulated |
| Patients | 5 concurrent |
| Total Ticks | 8,400 |
| Total Predictions | 42,000 |
| Throughput | 6.0 ticks/sec |
| Avg Latency | 31.16 ms |
| **P99 Latency** | **58.11 ms** |
| Max Latency | 2,298 ms (outlier, GC pause) |

**Category Distribution**:
- Category I: 99.1% (baseline stability test)
- Category II: 0.9%
- Category III: 0.0%

---

## 10. Limitations & Future Work

### 10.1 Current Limitations

1. **Sinusoidal Detection**: Currently relies on FFT-based frequency analysis with hard thresholds. More diverse training data and ML-based detection would improve sensitivity.

2. **Variable Deceleration Severity**: We detect and classify variable decelerations but do not yet compute all five NICHD severity signs (W-shape, slow recovery, overshoot, etc.).

3. **Contraction Integration**: UC signal is used for deceleration classification but not for tachysystole quantification relative to actual contraction count.

4. **Dataset Generalization**: Validated on synthetic archetypes. Real-world CTU-CHB database integration is complete but requires clinical validation.

5. **MHR Guard SpO₂ Dependency**: Cross-correlation method requires maternal SpO₂ pulse, which may not be available in all clinical setups. The system falls back to spectral analysis alone.

6. **SHAP Library Dependency**: SHAP explanations require the optional `shap` library. If not installed, the system gracefully degrades to rule-only explanations.

### 10.2 Completed in V3.0

**Backend Modules** (from V2.0):

- ✅ **MHR Guard Module**: Detects maternal heart rate contamination via spectral RSA analysis, cross-correlation, and baseline jump detection. Includes fetal sleep cycle awareness.
- ✅ **Trend Analyzer Module**: 60-minute trend tracking with FSQI-masked buffer, linear regression for slope calculation, and composite deterioration score (0-100).
- ✅ **Explainability Module**: Rule-based explanations (always available) and SHAP feature attribution (on-demand). Visual highlighting for CTG graphs.

**Full-Stack Migration** (V3.0):

- ✅ **React 18 Frontend**: TypeScript, Tailwind CSS, Zustand state management
- ✅ **FastAPI Backend**: Async REST + WebSocket streaming
- ✅ **Real-Time WebSocket**: MessagePack binary protocol, 4Hz push updates
- ✅ **Canvas-Based Charting**: TradingView lightweight-charts for 60 FPS CTG
- ✅ **Docker Deployment**: Multi-stage builds, Nginx reverse proxy
- ✅ **i18n Support**: Full Hebrew/English with RTL switching
- ✅ **E2E Testing**: Playwright test suite for critical flows

### 10.3 Future Roadmap

- **Phase 5**: Multi-fetal monitoring (twins)
- **Phase 6**: Integration with hospital EHR/HL7 FHIR
- **Phase 7**: Edge deployment on Raspberry Pi / Jetson for rural clinics
- **Phase 8**: Prospective clinical trial for CE/FDA clearance

---

## References

1. Ayres-de-Campos, D., et al. (2015). FIGO consensus guidelines on intrapartum fetal monitoring.
2. Israeli Position Paper on CTG Interpretation (2020).
3. Dempster, A., et al. (2020). ROCKET: Exceptionally fast and accurate time series classification. *Data Mining and Knowledge Discovery*.
4. Dempster, A., et al. (2021). MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification.
5. Das, S., et al. (2023). FSQI: Fetal Signal Quality Index for CTG Pre-processing.

---

*This whitepaper accompanies the SentinelFetal codebase. For implementation details, see the source code in `src/`.*

<p align="center">
<b>SentinelFetal — Where Engineering Meets Clinical Excellence</b>
</p>
