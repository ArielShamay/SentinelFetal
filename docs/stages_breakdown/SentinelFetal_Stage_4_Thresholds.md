<div dir="rtl" style="text-align: right;">

# ⚙️ Stage 4 — Threshold Calibration + Persistence
## כיול ספים + מנגנון התמדה

---

### 🎯 מטרה

להפוך **Score גולמי להחלטות יציבות**:

1. בחירת ספים `t_low` / `t_high` בצורה סיסטמית
2. הוספת מנגנון **התמדה K-of-N** להפחתת False Positives

---

### 📥 קלט (Inputs)

| מקור | תיאור |
|------|-------|
| `ai_scores` | סקורים מ־Stage 3 (`LogisticRegression.predict_proba[:, 1]`) |
| `ai_scores.parquet` | קובץ הסקורים עם metadata |
| Labels/Targets | לפחות negatives מסומנים |
| Runtime Config | פרמטרי persistence: `K`, `N` |

---

### 📤 פלט (Outputs)

| קובץ | תיאור |
|------|-------|
| `smart_logic_thresholds.yaml` | הספים: `t_low`, `t_high`, `K`, `N`, strict flags |
| `stage4_calibration_report.json` | דוח כיול |
| `STAGE4_TRUTH.md` | מסמך אמת |

---

## 🔨 4A) Pipeline בנייה (Calibration)

### שלב 4A.1 — בחירת סט לכיול

שימוש ב־Validation Set, במיוחד **Negatives** (רשומות ללא אירוע).

### שלב 4A.2 — חישוב התפלגות Scores

```python
# על Negatives בלבד
scores_neg = ai_scores[labels == 0]
distribution = np.histogram(scores_neg, bins=100)
```

### שלב 4A.3 — בחירת ספים

| סף | איך נקבע | דוגמה |
|----|----------|-------|
| `t_high` | קוונטיל גבוה על Negatives | 99th percentile |
| `t_low` | קוונטיל נמוך יותר / ROC tradeoff | 95th percentile |

```python
t_high = np.percentile(scores_neg, 99)
t_low  = np.percentile(scores_neg, 95)
```

### שלב 4A.4 — הוספת Persistence (K-of-N)

מנגנון שמחייב **K חלונות מתוך N עוקבים** לעבור את הסף לפני שמופקת התראה.

| פרמטר | משמעות | דוגמה |
|-------|--------|-------|
| `K` | מספר חלונות מינימלי מעל הסף | 2 |
| `N` | גודל החלון הנע | 3 |

```python
# דוגמה: K=2, N=3
# → צריך לפחות 2 מתוך 3 חלונות אחרונים מעל t_high
```

### שלב 4A.5 — Guardrails

| Guardrail | תנאי |
|-----------|------|
| **Recall** | לא יורד מתחת לסף מוגדר |
| **Alert Rate** | לא עולה חריג ביחס לבייסליין |

### שלב 4A.6 — שמירת Thresholds Artifact

```yaml
# smart_logic_thresholds.yaml
t_low: 0.45
t_high: 0.72
K: 2
N: 3
strict: true
```

---

## 🔧 4C) Implementation (קוד בפועל - כבר ממומש ✅)

| רכיב | קובץ | סטטוס |
|------|------|-------|
| Calibrator Module | `src/calibration/calibrator.py` (188 שורות) | ✅ מלא |
| Persistence Manager | `src/analysis/persistence.py` (173 שורות) | ✅ מלא |
| DynamicThresholds Dataclass | `src/config.py` (lines 239-247) | ✅ מלא |
| load_dynamic_thresholds() | `src/config.py` (lines 249-289) | ✅ מלא |
| Run Script | `scripts/pipeline/stage4_calibrate.py` (210 שורות) | ✅ מלא |
| Verification | `scripts/validation/verify_stage4.py` (180 שורות) | ✅ מלא |

**Walkthrough:** ראע [Stage_4_Walkthrough.md](../00_management/Stage_4_Walkthrough.md)

---

## ✅ 4B) Pipeline וולידציה (Verify)

| בדיקה | תיאור |
|-------|-------|
| **4B.1** | הספים נטענים ומופעלים בפועל |
| **4B.2** | K-of-N עובד על רצפים (Unit Tests סינתטיים) |
| **4B.3** | דוח מדדים: Recall, Alert-Rate, False Alerts |

---

### 🏆 קריטריון PASS

```
✓ Guardrail של Recall נשמר
✓ ירידה/אי־עלייה ב־Alert Rate (על Negatives)
✓ Thresholds Artifact קנוני ונקרא ע"י שלבים הבאים
```

---

</div>
