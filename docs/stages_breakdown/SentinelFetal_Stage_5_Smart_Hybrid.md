<div dir="rtl" style="text-align: right;">

# 🧠 Stage 5 — Smart Hybrid
## AI + Rules + Tiering + Boredom Gate + Explainability

---

### 🎯 מטרה

להפוך את ההחלטה ל**"חכמה" וקלינית**:

- שילוב `ai_score` עם `rule_score` באותו חלון
- Tiers להחלטה מדורגת
- Boredom Gate לדיכוי התראות מיותרות
- Persistence ל־K-of-N
- **Reason Codes** להסבר אנושי

---

### 📥 קלט (Inputs)

| מקור | תיאור |
|------|-------|
| Stage 2 | חלונות + מדדי איכות |
| Stage 3 | `ai_score` מ־LogisticRegression + `model_version` |
| Stage 4 | `thresholds`: `t_low`, `t_high`, `K`, `N` |
| Rules Engine | `rule_hits`, `rule_score`, `severity` |
| Masks | `quality_class` לכל חלון |

---

### 📤 פלט (Outputs)

| קובץ | תיאור |
|------|-------|
| `decision_outputs.parquet` | החלטה מלאה per-window |
| `rules_outputs.parquet` | `rule_hits`, `rule_score`, `severities` |
| `STAGE5_TRUTH.md` | מסמך אמת |
| `stage5_compare_report.json` | דוח השוואה |

---

## 🔨 5A) Pipeline בנייה (Build/Implement)

### שלב 5A.1 — חוקים per-window

החוקים עובדים על **אותו slicing בדיוק** של Stage 2 — אין Drift!

### שלב 5A.2 — הפקת Rule Outputs

לכל חלון:

| שדה | תיאור |
|-----|-------|
| `rule_hits` | רשימת חוקים שנפגעו |
| `rule_score` | סקור מצרפי מהחוקים |
| `severity` | חומרת הפגיעה |
| `reason_code` | קוד הסבר |

### שלב 5A.3 — Tiering (החלטה מדורגת)

| Tier | תנאי | תוצאה |
|------|------|-------|
| **Tier-1** | `ai_score >= t_high` | 🔴 Alert |
| **Tier-2** | `t_low <= ai_score < t_high` AND `rule_score >= r_min` | 🔴 Alert |
| **Tier-3** | `rule_severe == True` | 🔴 Alert (גם אם AI נמוך) |
| **Default** | לא עומד בתנאים | ✅ No Alert |

### שלב 5A.4 — Boredom Gate

מנגנון להשתקת התראות כאשר:
- איכות טובה (quality_class = GOOD)
- אין אירועים משמעותיים
- אין decels/hits חמורים

```
if quality == GOOD and no_significant_hits:
    suppress_alert()  # מדיניות מפורשת
```

### שלב 5A.5 — Persistence

החלטה סופית רק אם **K-of-N** מתקיים:

```python
# דוגמה: K=2, N=3
if sum(last_3_windows_above_threshold) >= 2:
    final_decision = ALERT
```

### שלב 5A.6 — Explainability

יצירת הסברים אנושיים:

```json
{
    "reason_codes": ["HIGH_AI_SCORE", "RULE_DECEL_DETECTED"],
    "summary": "זוהה AI Score גבוה עם האטה משמעותית",
    "confidence": 0.87
}
```

### שלב 5A.7 — Dump מלא per-window

```
record_id | win_start | win_end | quality_class | ai_score | 
t_low | t_high | rule_score | rule_hits | tier | 
persistence_state | decision | reasons
```

---
## 🔧 5C) Implementation (קוד בפועל - כבר ממומש ✅)

| רכיב | קובץ | סטטוס |
|------|------|-------|
| Tiering Logic (3-Tier) | `src/analysis/tiering.py` (193 שורות) | ✅ מלא |
| Boredom Gate | `src/analysis/boredom_gate.py` (121 שורות) | ✅ מלא |
| calculate_rule_score() | `src/analysis/override.py` (lines 363-493) | ✅ מלא |
| Stage5Pipeline Class | `src/pipeline/stage5_hybrid.py` (396 שורות) | ✅ מלא |
| WindowDecision Dataclass | `src/pipeline/stage5_hybrid.py` | ✅ מלא |
| Run Script | `scripts/pipeline/stage5_build.py` (220 שורות) | ✅ מלא |
| Verification | `scripts/validation/verify_stage5.py` (164 שורות) | ✅ מלא |

**Walkthrough:** ראע [Stage_5_Walkthrough.md](../00_management/Stage_5_Walkthrough.md)

---
## ✅ 5B) Pipeline וולידציה (Verify)

| בדיקה | תיאור |
|-------|-------|
| **5B.1** | Alignment Test: אותו חלון → אותו rule_score |
| **5B.2** | Comparison Test מול Baseline מיושר |
| **5B.3** | מדדים: Recall ≥ Guardrail, AlertRate ≤ Baseline |
| **5B.4** | Explainability Sanity: לכל Alert יש reasons |

---

### 🏆 קריטריון PASS

```
✓ אין עלייה ב־AlertRate ללא פגיעה ב־Recall
✓ Outputs per-window נשמרים ומאפשרים Debug מלא
```

---

</div>
