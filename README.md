<p align="center">
  <img src="https://img.shields.io/badge/Backend-Production_Ready-success?style=for-the-badge" alt="Backend Status"/>
  <img src="https://img.shields.io/badge/Frontend_V3-React-blue?style=for-the-badge" alt="Frontend Status"/>
  <img src="https://img.shields.io/badge/Version-3.0-blue?style=for-the-badge" alt="Version"/>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Accuracy-97.0%25-success?style=for-the-badge" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Scale-20_Patients-blue?style=for-the-badge" alt="Scale"/>
</p>

# SentinelFetal

### Real-Time Fetal Distress Detection Using Hybrid AI

> **One-liner:** A production-grade CTG monitoring system that combines lightweight ML (MiniRocket) with deterministic clinical rules to classify fetal status in **under 60ms** with **98.7% accuracy**.

## 🆕 V3.0 - Modern React Frontend

V3.0 introduces a complete frontend rewrite with React, TypeScript, and real-time WebSocket streaming.

### Quick Start (V3 Development)

```bash
# Backend (Terminal 1)
cd SentinelFetal
python -m venv .venv && .venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (Terminal 2)
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

### Quick Start (Docker - Production)

```bash
docker-compose up --build
```

Open **http://localhost** in your browser (frontend: 80, API: 8000).

### V3 Tech Stack
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Zustand
- **Charts**: lightweight-charts for real-time CTG visualization  
- **i18n**: English & Hebrew with RTL support
- **Backend**: FastAPI + WebSocket streaming (msgpack binary protocol)
- **Testing**: Playwright E2E tests
- **Deployment**: Docker + Nginx

📄 Full migration status: [docs/V3_STATUS.md](docs/V3_STATUS.md)

---

## V2.0 New Features

| Feature | Description | Status |
|---------|-------------|--------|
| **MHR Guard** | Detects maternal heart rate contamination via spectral RSA analysis | Backend implemented — UI surfacing pending |
| **Trend Analyzer** | 60-minute trend tracking with deterioration scoring (0-100) | Backend implemented — UI trend view pending |
| **Explainability** | Rule-based explanations with visual graph highlighting | Backend implemented — front-end overlays pending |

---

## ⚡ Performance at a Glance

### Clinical Validation Suite (Precision Test)
| Metric | Value | Status |
|--------|-------|--------|
| **Overall Accuracy** | 98.7% | ✅ |
| **P99 Latency** | 58 ms | ✅ |
| **Specificity (Healthy)** | 100% | ✅ |
| **Late Decel Sensitivity** | 93.3% | ✅ |
| **Noise Immunity (FSQI Gate)** | 100% | ✅ |

### The Gauntlet V4 (Scale & Robustness Test)
| Metric | Value | Status |
|--------|-------|--------|
| **Total Events Processed** | 2,500 | ✅ |
| **Overall Accuracy** | 97.0% | ✅ |
| **Late Decel Sensitivity** | **100.0%** | ✅ ⭐ |
| **Variable Decel Sensitivity** | **100.0%** | ✅ ⭐ |
| **Sinusoidal Detection** | 99.6% | ✅ |
| **Healthy Detection** | 90.2% | ✅ |
| **Patients Tested** | 20 | ✅ |
| **Execution Time** | 3m 26s | ✅ |

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

## 🖥️ Running the UI

### Development Mode

```bash
# Backend (Terminal 1)
cd SentinelFetal
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (Terminal 2)
cd frontend && npm run dev
```

Then open **http://localhost:5173** in your browser.

### Production Mode (Docker)

```bash
docker-compose up --build
# Frontend: http://localhost (port 80)
# API: http://localhost:8000
# WebSocket: ws://localhost/ws/stream
```

### UI Features
- **Multi-Patient Grid**: Monitor up to 20 patients simultaneously
- **Real-Time ECharts**: Dual-track CTG (FHR + UC) with 4Hz updates
- **Category Badges**: Color-coded I/II/III classification
- **God Mode**: Inject clinical events (Late Decel, Sinusoidal, etc.) for testing
- **Detail View**: Click any patient for full 10-minute history with zoom

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
├── api/                      # FastAPI Backend (V3)
│   ├── main.py               # Application entry point
│   ├── routers/              # REST + WebSocket routes
│   ├── services/             # Business logic adapters
│   └── models/               # Pydantic schemas
├── frontend/                 # React Frontend (V3)
│   ├── src/
│   │   ├── components/       # React components (charts, panels)
│   │   ├── pages/            # WardView, DetailView, Settings
│   │   ├── store/            # Zustand state management
│   │   ├── hooks/            # Custom React hooks
│   │   └── i18n/             # Translations (EN/HE)
│   ├── e2e/                  # Playwright E2E tests
│   └── package.json
├── src/
│   ├── interfaces/           # Abstract protocols + state bridge
│   ├── data/                 # Preprocessing, FSQI quality gate
│   ├── models/               # MiniRocket encoder, XGBoost
│   ├── rules/                # FIGO/NICHD clinical rules
│   ├── analysis/             # Trend analyzer, alerts
│   ├── explainability/       # Rule + SHAP explainers
│   ├── safety/               # MHR Guard module
│   └── simulation/           # Patient generator, orchestrator
├── scripts/                  # Validation & testing scripts
├── docs/
│   ├── V3_STATUS.md          # V3 migration status
│   └── reports/              # Technical documentation
├── docker-compose.yml        # Full-stack deployment
├── Dockerfile.backend        # Python FastAPI image
├── Dockerfile.frontend       # React + Nginx image
└── tests/                    # Python unit tests
```

---

## 📚 Documentation

### Status & Reality Check
- **[STATUS.md](docs/STATUS.md)** — Current system state: Backend (Production Ready) vs. Frontend (Alpha/Broken)
- **[UI_UX_GAP_ANALYSIS.md](docs/reports/UI_UX_GAP_ANALYSIS.md)** — Forensic audit of all UI defects with line numbers

### Technical Documentation
- **[TECHNICAL_WHITEPAPER.md](docs/reports/TECHNICAL_WHITEPAPER.md)** — Deep-dive into architecture, algorithms, and clinical logic (V2.0)
- **[CLINICAL_VALIDATION_REPORT.md](docs/reports/CLINICAL_VALIDATION_REPORT.md)** — Accuracy & sensitivity results
- **[DEEP_ENDURANCE_REPORT.md](docs/reports/DEEP_ENDURANCE_REPORT.md)** — 35-minute stability test results
- **[SentinelFetal_V2_PRD_SPECS.md](docs/plan/SentinelFetal_V2_PRD_SPECS.md)** — V2.0 Product Requirements (Backend Implemented, UI Pending)

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
