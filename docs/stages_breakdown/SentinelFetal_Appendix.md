<div dir="rtl" style="text-align: right;">

# 📎 נספח — Appendix

---

## 💡 הערה חשובה: מה "נוגע לפרונט" ומה לא

| שלבים | תפקיד |
|-------|-------|
| **Stage 2–5** | Brain / Offline / QA — עבודת רקע |
| **Stage 6** | חיבור לפרונט דרך Contract + WebSocket |

> **אם הפרונט מצוין — לא משנים אותו!**
> 
> **מיישרים את ה־Payload והאורקסטרציה אליו.**

---

## 📁 שמות סקריפטים מוצעים

| Stage | סקריפטים |
|-------|---------|
| **Stage 2** | `src/windowing/build_windows.py` |
|  | `src/windowing/verify_windows.py` |
| **Stage 3** | `src/ai/train_minirocket.py` |
|  | `src/ai/eval_ai.py` |
| **Stage 4** | `src/calibration/calibrate_thresholds.py` |
| **Stage 5** | `src/decision/validate_smart_logic.py` |
| **Stage 6** | `api/main.py` |
|  | `src/realtime/orchestrator.py` |
| **Stage 7** | `src/loaders/` |

---

</div>
