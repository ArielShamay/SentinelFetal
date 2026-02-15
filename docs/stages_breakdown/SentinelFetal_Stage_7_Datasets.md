<div dir="rtl" style="text-align: right;">

# 📊 Stage 7 — Datasets Integration
## שילוב מקורות דאטה נוספים

---

> ⚠️ **שלב אופציונלי/מומלץ** — לא חוסם את הדמו

---

### 🎯 מטרה

לאחד **מקורות דאטה נוספים** (WFDB, CSV, אחרים) לאותו Schema קנוני:

- שיפור אימון ומודלים
- כיול מדויק יותר
- שיפור חוקים

**בלי לשבור את הקונטרקט הקיים!**

---

### 📥 קלט (Inputs)

| מקור | תיאור |
|------|-------|
| מקורות דאטה נוספים | WFDB, CSV, פורמטים אחרים |
| Stage 1 Spec | מסכות + Normalize |
| Stage 2 Spec | חלונות |

---

### 📤 פלט (Outputs)

| קובץ | תיאור |
|------|-------|
| Loaders אחידים | טוענים לכל פורמט |
| Canonical Window Dataset | דאטאסט חלונות משותף |
| `STAGE7_TRUTH.md` | מסמך אמת |

---

## 🔨 7A) Pipeline בנייה

### שלב 7A.1 — Loader לכל מקור

```python
class WFDBLoader:
    def load(self, path) -> Tuple[FHR, UC, Timestamps, Masks]:
        ...

class CSVLoader:
    def load(self, path) -> Tuple[FHR, UC, Timestamps, Masks]:
        ...
```

### שלב 7A.2 — התאמה ל־fs קנוני

```python
if source_fs != 4:
    data = resample(data, source_fs, target_fs=4)
```

### שלב 7A.3 — יצירת Artifacts זהים

```
output/
├── X_norm_4hz.npy      # זהה ל־Stage 1
├── mask_fhr.npy
├── mask_uc.npy
├── manifest.csv
└── windows_index.parquet  # זהה ל־Stage 2
```

### שלב 7A.4 — QA: השוואת סטטיסטיקות

| מטריקה | מקור A | מקור B | סטייה |
|--------|--------|--------|-------|
| Invalid Rate FHR | 5.2% | 4.8% | 0.4% |
| Invalid Rate UC | 3.1% | 3.5% | 0.4% |
| Mean Duration | 45m | 42m | 3m |

---

## ✅ 7B) Pipeline וולידציה (Verify)

| בדיקה | תיאור |
|-------|-------|
| **7B.1** | Schema Tests: אותם שדות, אותם dtypes |
| **7B.2** | Window Consistency: חלונות תקינים לפי אותם כללים |
| **7B.3** | No Skew: אותו Preprocessing/Normalize |

---

### 🏆 קריטריון PASS

```
✓ שילוב דאטה חדש לא שובר שום שלב
✓ Metrics נשמרים או משתפרים
```

---

</div>
