<p align="center">
  <img src="https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
</p>

# 🩺 SentinelFetal

### Real-Time Fetal Distress Detection Using Hybrid AI

> **One-liner:** A production-grade CTG monitoring system that combines lightweight ML (MiniRocket) with deterministic clinical rules to classify fetal status in **under 60ms** with **98.7% accuracy**.

---

## ⚡ Performance at a Glance

| Metric | Value | Status |
|--------|-------|--------|
| **Overall Accuracy** | 98.7% | ✅ |
| **P99 Latency** | 58 ms | ✅ |
| **Specificity (Healthy)** | 100% | ✅ |
| **Late Decel Sensitivity** | 93.3% | ✅ |
| **Noise Immunity (FSQI Gate)** | 100% | ✅ |
| **Endurance (35-min stress)** | Passed | ✅ |

---

## 🎯 What Problem Does This Solve?

Cardiotocography (CTG) is the standard for intrapartum fetal monitoring, but:
- **Manual interpretation is subjective** → High inter-observer variability
- **Alarm fatigue is deadly** → Clinicians ignore 85%+ of alerts
- **Heavy AI models are too slow** → Transformers can't run in real-time

**SentinelFetal** solves this with a **Hybrid Engine**: Fast ML embeddings + Hard clinical rules = Safe, interpretable, real-time classification.

---

## 🚀 Quick Start

```bash
# Clone & Setup
git clone https://github.com/ArielShamay/SentinelFetal.git
cd SentinelFetal
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Run Clinical Validation (Accuracy Test)
python scripts/clinical_validation_suite.py

# Run Deep Endurance Audit (Stability Test)
python scripts/deep_endurance_audit.py
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CTG Signal (FHR + UC)                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   FSQI Quality Gate   │  ← Blocks bad signals BEFORE AI
                    └───────────┬───────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Preprocessing  │   │  Rule Engine    │   │  MiniRocket     │
│  (Spike/Gap)    │   │  (FIGO/NICHD)   │   │  (9,996 feats)  │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                      │
         └─────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Feature Fusion    │  → 1,035-dim vector
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  XGBoost Classifier │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Medical Override  │  ← Hard safety rules
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Category 1/2/3     │  → Alert Generation
                    └─────────────────────┘
```

---

## 📊 Classification Categories

| Category | Name | Meaning | Action |
|----------|------|---------|--------|
| **I** | Normal | Healthy fetal status | Continue monitoring |
| **II** | Intermediate | Uncertain, needs attention | Increase monitoring |
| **III** | Pathological | Fetal distress suspected | **Immediate intervention** |

---

## 📁 Project Structure

```
SentinelFetal/
├── scripts/
│   ├── clinical_validation_suite.py  # Accuracy testing
│   └── deep_endurance_audit.py       # Stability testing
├── src/
│   ├── data/           # Preprocessing, signal quality (FSQI)
│   ├── models/         # MiniRocket, XGBoost, Fusion
│   ├── rules/          # Baseline, Variability, Decelerations, etc.
│   ├── analysis/       # Medical Override, Alert Generation
│   ├── simulation/     # Real-time patient generator, RingBuffer
│   └── pipeline/       # PipelineAdapter (orchestration)
├── docs/
│   ├── TECHNICAL_WHITEPAPER.md       # Deep technical documentation
│   └── reports/
│       ├── CLINICAL_VALIDATION_REPORT.md
│       └── DEEP_ENDURANCE_REPORT.md
└── tests/              # Unit & integration tests
```

---

## 📚 Documentation

- **[TECHNICAL_WHITEPAPER.md](docs/TECHNICAL_WHITEPAPER.md)** — Deep-dive into architecture, algorithms, and clinical logic
- **[CLINICAL_VALIDATION_REPORT.md](docs/reports/CLINICAL_VALIDATION_REPORT.md)** — Accuracy & sensitivity results
- **[DEEP_ENDURANCE_REPORT.md](docs/reports/DEEP_ENDURANCE_REPORT.md)** — 35-minute stability test results

---

## 🏆 Key Technical Highlights

- **Zero-Inference FSQI Gate**: Bad signals are rejected *before* reaching the ML model, saving CPU and preventing garbage-in-garbage-out.
- **MiniRocket over Transformers**: 9,996 fixed-kernel features in ~1ms vs 300ms+ for attention-based models. Same accuracy, 300x faster.
- **Medical Override Safety Net**: Hard clinical rules (sinusoidal → Cat III) can *never* be overridden by ML, ensuring patient safety.
- **O(1) Memory via RingBuffer**: `collections.deque` with fixed maxlen ensures constant memory regardless of session length.

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built for the Medical AI Hackathon 2026</b><br/>
  <i>Where Engineering Meets Clinical Excellence</i>
</p>
