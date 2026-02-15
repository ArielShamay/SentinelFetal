<div dir="rtl" style="text-align: right;">

# 🌐 Stage 6 — E2E Integration
## FastAPI/WS → Frontend + God-Mode

---

### 🎯 מטרה

**דמו "אמיתי"** — זרימה מלאה:

```
קלט סינתטי/Real-Time → Brain (Stage 2–5) → API → WebSocket → Frontend
```

כולל **God-Mode** להזרקת אירועים לבדיקה.

---

### 📥 קלט (Inputs)

| מקור | תיאור |
|------|-------|
| Frontend קיים | נשאר כמות שהוא |
| FastAPI/WS Server | שרת API + WebSocket |
| Orchestrator | מדמה 1–20 יולדות |
| Brain Pipeline | Stages 2–5 |

---

### 📤 פלט (Outputs)

| קובץ | תיאור |
|------|-------|
| **Contract** | פורמט Payload יציב ל־UI |
| **Logs** | לוגים מפורטים |
| **Latency Metrics** | זמני תגובה |
| `STAGE6_TRUTH.md` | מסמך אמת |

---

## 🔨 6A) Pipeline בנייה (Implement)

### שלב 6A.1 — Orchestrator

ניהול N יולדות במקביל:

| רכיב | תיאור |
|------|-------|
| Ring Buffers | שמירת נתונים אחרונים |
| Timestamps | סנכרון זמנים |
| Patient Slots | 1–20 יולדות בו־זמנית |

### שלב 6A.2 — Streaming Ingestion

```python
# עבור כל דגימה נכנסת:
if sample.fs != 4:
    if policy == "resample":
        sample = resample_to_4hz(sample)
    else:
        raise FailFast("Invalid fs")
```

### שלב 6A.3 — Window Trigger

כל **Stride** (5 דקות) מפעיל חישוב חלון 20 דקות:

```python
if current_time % stride == 0:
    window = get_last_20_minutes()
    process_window(window)
```

### שלב 6A.4 — Brain Call

| Stage | פעולה |
|-------|-------|
| Stage 2 | מדדי איכות (בזמן אמת) |
| Stage 3 | AI Inference |
| Stage 5 | Hybrid Decision + Explainability |

### שלב 6A.5 — API Endpoints

| Endpoint | תיאור |
|----------|-------|
| `POST /start` | התחלת מעקב |
| `POST /stop` | עצירת מעקב |
| `GET /status` | סטטוס נוכחי |
| `POST /inject` | God-Mode: הזרקת אירוע |

### שלב 6A.6 — WebSocket Stream (The Contract)

שידור Updates ל־UI לפי Contract מלא (מבוסס על `STAGE6_TRUTH`):

**Payload עבור 4Hz Update:**

```json
{
  "type": "patient_update",
  "patient_id": "P1",
  "category": 2, // 1=Normal, 2=Intermediate, 3=Pathological
  "confidence": 0.85,
  "fhr_latest": [140, 141, ...], // דגימות אחרונות להצגה רציפה
  "uc_latest": [10, 12, ...],
  "findings": {
    "baseline": {"value": 145, "is_normal": true},
    "variability": {"value": 8, "category": "MODERATE"},
    "decelerations": {"late": 1, "variable": 0}
  },
  "stage5_decision": {
    "tier": "tier_2",
    "summary": "AI suspicion (85%) + Rule confirmation",
    "reason_codes": ["TIER_2_AI_AND_RULES", "RULE_LATE_DECELERATION"]
  }
}
```

### שלב 6A.7 — רכיבי ארכיטקטורה (Component Flow)

1.  **Simulation Orchestrator:** מייצר אותות סינתטיים (4Hz).
2.  **Orchestrator Adapter:** הגשר שמקבל Snapshots ומפעיל את ה-Pipeline.
3.  **Pipeline Adapter:** מריץ את שרשרת הניתוח (Stage 2-5).
4.  **WebSocket Broadcaster:** משדר את ה-Payload הסופי ל-Frontend.

---

## 🔧 6C) Implementation (קוד בפועל - כבר ממומש ✅)

| רכיב | קובץ | סטטוס |
|------|------|-------|
| Orchestrator Core | `src/simulation/core/orchestrator.py` | ✅ מלא |
| Orchestrator Adapter | `api/services/orchestrator_adapter.py` (636 שורות) | ✅ מלא |
| **Pipeline Adapter (Integration!)** | `src/simulation/processing/pipeline_adapter.py` (744 שורות) | ✅ חיבור Stage 5 |
| WebSocket Routes | `api/routers/websocket.py` (208 שורות) | ✅ מלא |
| DataBridge | `src/interfaces/state_bridge.py` (581 שורות) | ✅ מלא |
| Broadcaster Service | `api/services/broadcaster.py` | ✅ מלא |
| Run Script | `scripts/pipeline/stage6_e2e.py` (153 שורות) | ✅ מלא |
| Verification | `scripts/validation/verify_stage6.py` (164 שורות) | ✅ מלא |

**🔴 קריטי: Stage 5 Integration**
PipelineAdapter משלב את Stage 5 ל-Orchestrator דרך `stage5_pipeline.process_window()` בתוך `_process_patient_moment()`.

**Walkthrough:** ראע [Stage_6_Walkthrough.md](../00_management/Stage_6_Walkthrough.md)

---

## ✅ 6B) Pipeline וולידציה (Demo PASS)

| בדיקה | תנאי הצלחה |
|-------|------------|
| **6B.1** | N=1..20 יולדות, ריצה יציבה ≥10 דקות |
| **6B.2** | Latency: UI מתעדכן כל Tick בלי Stutter |
| **6B.3** | God-Mode: Inject → שינוי ב־UI תוך שניות |
| **6B.4** | Fail-Fast: Input לא תקין → FAIL ברור |

---

### 🏆 קריטריון PASS

```
✓ דמו עובד End-to-End עם Frontend קיים
✓ חוזה Payload יציב + Explainability
✓ אין קריסות/תקיעות
```

---

</div>
