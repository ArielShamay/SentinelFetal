# SentinelFetal Development Plan

## 🎯 Project Vision
Real-time fetal distress detection system using CTG analysis with ML/Rule-based hybrid approach.

---

## 📅 Development Phases

### Phase 1: Data Pipeline (Current)
**Status**: 🔄 In Progress

| Task | Description | Status |
|------|-------------|--------|
| 1.1 | Create data loader for CTU-UHB dataset | ✅ |
| 1.2 | Implement preprocessing with gap filling | ✅ |
| 1.3 | Apply 10-second rule (40 samples @ 4Hz) | ✅ |
| 1.4 | Unit tests for preprocessing | ✅ |
| 1.5 | Visual validation script | ✅ |

**Key Rules:**
- FHR valid range: 50-240 BPM (out-of-range → NaN)
- Gap filling: Linear interpolation for gaps ≤ 10 seconds (40 samples)
- Gaps > 10 seconds remain as NaN

---

### Phase 2: Rule Engine
**Status**: ⏳ Pending

| Task | Description | Status |
|------|-------------|--------|
| 2.1 | Implement baseline calculator | ⏳ |
| 2.2 | Acceleration detector | ⏳ |
| 2.3 | Deceleration classifier (Early/Late/Variable) | ⏳ |
| 2.4 | Variability analyzer | ⏳ |
| 2.5 | NICHD/FIGO category assignment | ⏳ |

**Clinical Thresholds:**
- Baseline: 110-160 BPM (normal)
- Accelerations: ≥15 BPM above baseline for ≥15 seconds
- Decelerations: ≥15 BPM below baseline for ≥15 seconds
- Variability: 5-25 BPM (normal)

---

### Phase 3: ML Model
**Status**: ⏳ Pending

| Task | Description | Status |
|------|-------------|--------|
| 3.1 | Feature extraction pipeline | ⏳ |
| 3.2 | Model training (XGBoost/LSTM) | ⏳ |
| 3.3 | Model evaluation & validation | ⏳ |
| 3.4 | Hybrid scoring system | ⏳ |

---

### Phase 4: Real-time System
**Status**: ⏳ Pending

| Task | Description | Status |
|------|-------------|--------|
| 4.1 | Streaming data handler | ⏳ |
| 4.2 | Real-time alert system | ⏳ |
| 4.3 | Dashboard UI | ⏳ |
| 4.4 | API endpoints | ⏳ |

---

## 📊 Dataset: CTU-UHB

- **Source**: PhysioNet CTU-UHB Intrapartum CTG Database
- **Records**: 552 intrapartum recordings
- **Sampling Rate**: 4 Hz
- **Signals**: FHR1, FHR2, UC (Uterine Contractions)

---

## 🧪 Testing Strategy

1. **Unit Tests**: Individual function validation
2. **Integration Tests**: Pipeline flow validation
3. **Clinical Validation**: Comparison with expert annotations
4. **Performance Tests**: Real-time processing benchmarks

---

## 📁 Project Structure

```
SentinelFetal/
├── src/
│   ├── data/
│   │   ├── loader.py       # Data loading from CTU-UHB
│   │   └── preprocess.py   # Signal preprocessing
│   ├── rules/
│   │   ├── baseline.py     # Baseline calculation
│   │   ├── accelerations.py
│   │   ├── decelerations.py
│   │   └── variability.py
│   ├── models/
│   │   ├── features.py     # Feature extraction
│   │   └── classifier.py   # ML model
│   └── visualize_preprocessing.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_rules.py
│   └── test_models.py
├── data/
│   └── ctu-uhb/           # Dataset location
├── notebooks/
│   └── exploration.ipynb
└── DEVELOPMENT_PLAN.md
```

---

## 🔧 Configuration

```yaml
preprocessing:
  sampling_rate: 4  # Hz
  fhr_min: 50       # BPM
  fhr_max: 240      # BPM
  max_gap_seconds: 10
  max_gap_samples: 40

rules:
  baseline_window: 600  # 10 minutes in samples (2400)
  acceleration_threshold: 15  # BPM
  acceleration_duration: 15   # seconds
  deceleration_threshold: 15  # BPM
```

---

## ✅ Acceptance Criteria

### Phase 1 Complete When:
- [x] Can load any record from CTU-UHB dataset
- [x] Preprocessing handles gaps correctly (10-second rule)
- [x] Unit tests pass for gap filling logic
- [x] Visual comparison shows correct preprocessing
