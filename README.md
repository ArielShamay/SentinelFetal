# SentinelFetal Gen3.5

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Hybrid AI System for Fetal Monitoring** combining MOMENT foundation model with rule-based engine based on the Israeli Position Paper on CTG Interpretation.

---

## 🎯 Overview

SentinelFetal Gen3.5 is a real-time fetal distress detection system that uses a **Dual Feature Extraction** approach:

1. **MOMENT Model** (AutonLab/MOMENT-1-large): 385M parameter foundation model for zero-shot time series embeddings (1024-dim)
2. **Rule Engine**: Clinical rules from Israeli Position Paper detecting baseline, variability, decelerations, tachysystole, and sinusoidal patterns

The system classifies CTG recordings into three medical categories:
- **Category 1 (Normal)**: Green light - Normal fetal status
- **Category 2 (Intermediate)**: Orange light - Requires close monitoring
- **Category 3 (Pathological)**: Red light - Immediate intervention required

---

## 🏗️ Architecture

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
                   │   Hybrid    │
                   │ Classifier  │
                   │ (Cat 1/2/3) │
                   └─────────────┘
```

---

## 📦 Installation

### Requirements
- Python 3.9+
- NumPy, pandas, scipy
- wfdb (for CTU-UHB dataset)
- pytest (for testing)

### Setup

```bash
# Clone repository
git clone https://github.com/ArielShamay/SentinelFetal.git
cd SentinelFetal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

---

## 📊 Dataset

Uses the **CTU-UHB Intrapartum Cardiotocography Database** from PhysioNet:
- 552 intrapartum recordings
- Sampling rate: 4 Hz
- Signals: FHR1, FHR2, UC

Download from: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

---

## 🚀 Usage

### Data Loading & Preprocessing

```python
from src.data.loader import CTUDataLoader
from src.data.preprocess import CTGPreprocessor, PreprocessingConfig

# Load CTG record
loader = CTUDataLoader("data/ctu-chb-intrapartum-cardiotocography-database-1.0.0")
record = loader.load_record("1001")

# Preprocess signal
config = PreprocessingConfig(sampling_rate=4.0, max_gap_seconds=10.0)
preprocessor = CTGPreprocessor(config)
result = preprocessor.process(record.fhr1)

print(f"Filled {result.stats['filled_percent']:.1f}% of gaps")
print(f"Mean FHR: {result.stats['mean_fhr']:.1f} bpm")
```

### Rule Engine

```python
from src.rules import (
    calculate_baseline,
    calculate_variability,
    detect_decelerations,
    detect_tachysystole,
    detect_sinusoidal_pattern
)

# Calculate baseline
baseline = calculate_baseline(fhr, sampling_rate=4.0)
print(f"Baseline: {baseline.value} bpm")

# Analyze variability
variability = calculate_variability(fhr, baseline.value)
print(f"Variability: {variability.category.name}")

# Detect decelerations
decelerations = detect_decelerations(fhr, uc, baseline.value)
for decel in decelerations:
    print(f"{decel.type.name} deceleration: depth={decel.depth_bpm} bpm")

# Check for sinusoidal pattern (SEVERE)
sinusoidal = detect_sinusoidal_pattern(fhr)
if sinusoidal.detected:
    print("⚠️ SEVERE: Sinusoidal pattern detected - Category 3!")
```

---

## 🧪 Testing

The project includes 26 comprehensive unit tests covering all rule engine modules:

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_rules.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- ✅ Baseline calculation (6 tests)
- ✅ Variability analysis (5 tests)
- ✅ Deceleration detection (5 tests)
- ✅ Tachysystole detection (3 tests)
- ✅ Sinusoidal pattern detection (5 tests)
- ✅ Integration tests (2 tests)

---

## 📁 Project Structure

```
SentinelFetal/
├── data/                           # Dataset storage
│   └── ctu-chb.../                # CTU-UHB database
├── docs/                           # Documentation
│   ├── SentinelFetal_PRD.docx.txt
│   ├── SentinelFetal_TechSpec.docx.txt
│   └── SentinelFetal_Gen35_Spec.docx.txt
├── models/                         # Saved models
├── notebooks/                      # Jupyter notebooks
├── src/
│   ├── data/
│   │   ├── loader.py              # CTU-UHB data loader
│   │   └── preprocess.py          # Signal preprocessing
│   ├── rules/
│   │   ├── baseline.py            # Baseline FHR calculation
│   │   ├── variability.py         # Variability analysis
│   │   ├── decelerations.py       # Deceleration detection
│   │   ├── tachysystole.py        # Tachysystole detection
│   │   └── sinusoidal.py          # Sinusoidal pattern detection
│   └── visualize_preprocessing.py # Visualization tools
├── tests/
│   ├── test_preprocessing.py
│   └── test_rules.py              # 26 unit tests
├── .gitignore
├── DEVELOPMENT_PLAN.md
├── README.md
└── requirements.txt
```

---

## 🔬 Clinical Rules (Israeli Position Paper)

### Baseline FHR
- **Normal**: 110-160 bpm
- **Bradycardia**: <110 bpm
- **Tachycardia**: >160 bpm
- Calculation: Mean FHR in 2-min stable segment (variability <25 bpm), rounded to nearest 5

### Variability
- **Absent**: 0-2 bpm (Category 3 if >50 min)
- **Minimal**: 3-5 bpm (Category 2)
- **Moderate**: 6-25 bpm (Normal)
- **Marked**: >25 bpm

### Decelerations
Classification by lag time from contraction peak:
- **Early**: <5 seconds (benign)
- **Late**: >15 seconds (concerning - Category 2/3)
- **Variable**: Abrupt onset (>0.5 bpm/sample)

Criteria: ≥15 bpm below baseline, duration 15s-10min

### Tachysystole
- **Definition**: >5 contractions per 10 minutes
- **Window**: Averaged over 30 minutes

### Sinusoidal Pattern ⚠️
- **SEVERE FINDING** - Always Category 3
- Frequency: 3-5 cycles/minute
- Amplitude: 5-15 bpm
- Duration: >20 minutes
- Clinical significance: Fetal anemia, severe hypoxia

---

## 📈 Development Phases

- [x] **Phase 1**: Data Pipeline (Complete)
  - CTU-UHB loader
  - Preprocessing with 10-second rule
  - Gap filling & spike detection
  - Unit tests & validation

- [x] **Phase 2**: Rule Engine (Complete)
  - Baseline calculator
  - Variability analyzer
  - Deceleration classifier
  - Tachysystole detector
  - Sinusoidal pattern detector
  - 26 unit tests

- [ ] **Phase 3**: MOMENT Integration (In Progress)
  - Load MOMENT model (AutonLab/MOMENT-1-large)
  - Extract 1024-dim embeddings
  - Zero-shot inference

- [ ] **Phase 4**: Hybrid Classifier
  - Feature fusion (MOMENT + Rules)
  - Category 1/2/3 classification
  - Confidence scoring

- [ ] **Phase 5**: Real-time System
  - Streaming data handler
  - Alert system
  - Dashboard UI

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

- **Israeli Position Paper on CTG Interpretation** (Ministry of Health, Israel)
- **MOMENT Model**: [AutonLab/MOMENT](https://huggingface.co/AutonLab/MOMENT-1-large)
- **CTU-UHB Database**: [PhysioNet](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)
- **NICHD Guidelines**: National Institute of Child Health and Human Development

---

## 👨‍💻 Author

**Ariel Shamay**
- GitHub: [@ArielShamay](https://github.com/ArielShamay)

---

## 🙏 Acknowledgments

- PhysioNet for providing the CTU-UHB database
- AutonLab for the MOMENT foundation model
- Israeli Ministry of Health for clinical guidelines
