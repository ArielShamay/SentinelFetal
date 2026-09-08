# SentinelFetal

Real-time CTG (cardiotocography) monitoring that flags fetal distress while the signal is still streaming.

The system combines a deterministic clinical rules engine (FIGO baseline, variability, decelerations) with a machine-learning classifier and a medical override layer that acts as a safety net.

## Model performance

### SentinelFetal V2 - PatchTST foundation model (current)

Two-stage pipeline: masked pre-training on raw CTG windows, then acidemia classification. Evaluated once on a held-out CTU-UHB test set of 55 recordings (11 acidemic) that was never touched during development.

| Metric | Result |
| --- | --- |
| AUC | 0.839 |
| Sensitivity | 0.818 |
| Specificity | 0.773 |
| Accuracy | 78.2% |

Training data: CTU-UHB (PhysioNet, 552 recordings) plus FHRMA (135, pre-training only), split patient-wise so no recording leaks between train, validation and test.

Model architecture: PatchTST, ~413K parameters, 1800-sample (7.5 min) windows, 48-sample patches with 50% stride, 3 transformer layers, 4 attention heads. Training code lives in a separate private repository; this repository is the serving system.

> The model bundled in this repo for the live demo is the earlier Gen3.5 hybrid (MiniRocket + XGBoost + rules engine).

### Serving

| Metric | Value |
| --- | --- |
| Inference latency | < 50 ms |
| Concurrent patients | up to 24 |

## Architecture

```
Frontend            Backend             Pipeline
React + TS   --WS-->  FastAPI   ------>  Analysis
port 3000            port 8000
```

Pipeline stages:

1. **Preprocessing** - signal cleaning and normalization
2. **Rules engine** - baseline, variability and deceleration detection
3. **Classifier** - ML scoring of the window
4. **Medical override** - clinical safety net over the model output

## Quick start

Requirements: Python 3.11+, Node.js 18+

Terminal 1 - backend:

```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --reload --port 8000
```

Terminal 2 - frontend:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. The simulation starts automatically. Click a patient card for the detailed CTG view, and use the God Mode panel to inject clinical events in real time.

## Project structure

```
api/          FastAPI backend (routers, services)
src/          Core logic: pipeline, analysis, adapters, simulation
frontend/     React application
models/       Trained model artifacts
config/       Configuration
docs/         Documentation
tests/        Test suites
```

## API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/ws/stream` | WebSocket | Real-time data stream |
| `/api/health` | GET | Health check |
| `/api/patients` | GET | Patient list |
| `/api/patients/{id}/inject` | POST | Inject a clinical event (God Mode) |

## Tests

```bash
pytest tests/                             # unit tests
python scripts/verify_v6_pipeline_e2e.py  # end-to-end pipeline
python scripts/load_test_suite.py         # load test
```

## Recognition

1st place, AI For Life Hackathon (March 2026). 2nd place, Ariel University Entrepreneurship Accelerator (June 2026).

## License

MIT - see LICENSE.
