# SentinelFetal 🩺

**מערכת חכמה לניטור CTG (קרדיוטוקוגרפיה) עוברי בזמן אמת**

מערכת היברידית המשלבת **בינה מלאכותית (MiniRocket + XGBoost)** עם **מנוע חוקים רפואיים** לסיווג מעקב עוברי לקטגוריות FIGO.

---

## 🚀 הרצה מהירה

### דרישות
- Python 3.11+
- Node.js 18+

### 3 צעדים בלבד:

**טרמינל 1 - Backend:**
```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --reload --port 8000
```

**טרמינל 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**זהו! פתחו את הדפדפן:** http://localhost:3000

> הסימולציה מתחילה אוטומטית. לחצו על כרטיס מטופל כדי לראות את ה-CTG המפורט.
> השתמשו בפאנל **God Mode** (בצד שמאל) כדי להזריק אירועים קליניים בזמן אמת.

---

## 🏗️ ארכיטקטורה

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────►│   Backend   │────►│  Pipeline   │
│  React+TS   │ WS  │  FastAPI    │     │  Analysis   │
│  Port 3000  │     │  Port 8000  │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Pipeline (4 שלבים):

1. **Preprocessing** - ניקוי ונרמול האות
2. **Rules Engine** - זיהוי Baseline, Variability, Decelerations
3. **AI (MiniRocket + XGBoost)** - סיווג ML
4. **Medical Override** - רשת ביטחון רפואית

---

## 📁 מבנה הפרויקט

```
SentinelFetal/
├── api/                 # FastAPI Backend
│   ├── main.py          # Entry point
│   ├── routers/         # API endpoints
│   └── services/        # Business logic
│
├── src/                 # Core Logic
│   ├── pipeline/        # Pipeline orchestration
│   ├── analysis/        # Rules & Override
│   ├── adapters/        # Model adapters
│   └── simulation/      # Synthetic data generator
│
├── frontend/            # React Application
│   └── src/
│       ├── components/  # UI components
│       └── hooks/       # Custom hooks
│
├── models/              # Trained ML models
│   ├── minirocket_encoder.joblib
│   └── ensemble_v5/xgboost_v5.pkl
│
├── config/              # Configuration files
├── docs/                # Documentation
└── tests/               # Test suites
```

---

## 📊 ביצועים (V5)

| מדד | ערך |
|-----|-----|
| **Sensitivity** | 89.2% |
| **Specificity** | 26.7% |
| **Latency** | <50ms |
| **מטופלים במקביל** | עד 24 |

---

## 🔧 API Endpoints

| Endpoint | Method | תיאור |
|----------|--------|--------|
| `/ws/stream` | WebSocket | נתונים בזמן אמת |
| `/api/health` | GET | Health check |
| `/api/patients` | GET | רשימת מטופלים |
| `/api/patients/{id}/inject` | POST | הזרקת אירוע (God Mode) |

---

## 📚 תיעוד נוסף

- [מצגת מפורטת](docs/PRESENTATION_SLIDES.md)
- [מצגת מתומצתת](docs/PRESENTATION_SIMPLIFIED.md)
- [SYSTEM_REFERENCE](docs/explain/SYSTEM_REFERENCE.md)
- [PRD](docs/plan/PRD.md)

---

## 🧪 בדיקות

```bash
# Unit tests
pytest tests/

# E2E Pipeline test
python scripts/verify_v6_pipeline_e2e.py

# Load test
python scripts/load_test_suite.py
```

---

## 📝 License

MIT License - See LICENSE file for details.

---

*Last updated: January 2026 | Version 6.0*
