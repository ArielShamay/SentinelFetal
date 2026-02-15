<div dir="rtl" style="text-align: right;">

# 🤖 Stage 3 — AI Baseline
## MiniRocket + StandardScaler + LogisticRegression

---

### 🎯 מטרה

להקים **Baseline מהיר ויציב** שרץ על CPU:

- **MiniRocket** כ־Feature Extractor על חלונות
- **StandardScaler** לנרמול ה־Features
- **LogisticRegression** כמסווג שמחזיר `ai_score`

---

### 📥 קלט (Inputs)

| מקור | תיאור |
|------|-------|
| Stage 2 Windows | רק חלונות **כשרים** לפי `valid_for_ai` |
| `X_norm_4hz.npy` | מערך הנתונים המנורמל |
| `windows_index.csv` | אינדקס החלונות עם metadata |
| `manifest.csv` | Labels (pH per record) |

---

### 📤 פלט (Outputs)

| קובץ | תיאור |
|------|-------|
| `stage3_ai_pipeline.joblib` | Pipeline מאומן (MiniRocket + Scaler + LR) |
| `stage3_eval_report.json` | דוח הערכה עם מטריקות |
| `ai_scores.parquet` | `ai_score` לכל חלון + metadata |
| `STAGE3_TRUTH.md` | מסמך אמת |

---

## 🔨 3A) Pipeline בנייה (Train/Build)

### שלב 3A.1 — הגדרת Split

חלוקת הדאטא ל־Train/Val/Test **לפי `record_id`** — למניעת Data Leakage בין חלונות מאותה רשומה.

```python
from sklearn.model_selection import GroupShuffleSplit
# 70% train / 15% val / 15% test
# groups = record_id
```

### שלב 3A.2 — הכנת קלט לחלונות

| תת־שלב | פעולה |
|--------|-------|
| **3A.2.1** | סינון חלונות כשרים (`valid_for_ai = True`) |
| **3A.2.2** | יצירת טנזור בפורמט MiniRocket |

```python
# פורמט נדרש:
X_tensor.shape = (n_windows, n_channels, n_timestamps)
# = (n_windows, 2, 4800)  # FHR + UC, 20min @ 4Hz
```

### שלב 3A.3 — אימון MiniRocket

```python
from sktime.transformations.panel.rocket import MiniRocket

minirocket = MiniRocket(random_state=42)
minirocket.fit(X_train)  # על TRAIN בלבד!
```

### שלב 3A.4 — הפקת Features

```python
features_train = minirocket.transform(X_train)
features_val   = minirocket.transform(X_val)
features_test  = minirocket.transform(X_test)
```

### שלב 3A.5 — נרמול עם StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(features_train)  # fit על TRAIN בלבד!
X_val_scaled   = scaler.transform(features_val)
X_test_scaled  = scaler.transform(features_test)
```

### שלב 3A.6 — אימון LogisticRegression

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    solver='lbfgs',
    max_iter=5000,
    class_weight='balanced',
    random_state=42
)
lr.fit(X_train_scaled, y_train)
```

| פרמטר | ערך | הסבר |
|-------|-----|------|
| solver | lbfgs | יציב ומהיר |
| max_iter | 5000 | הבטחת התכנסות |
| class_weight | balanced | טיפול באימבלנס |
| random_state | 42 | שחזוריות |

### שלב 3A.7 — הפקת AI Score

```python
ai_score = lr.predict_proba(features_scaled)[:, 1]
# ai_score ∈ [0, 1] — הסתברות לתוצאה שלילית (pH < 7.10)
```

### שלב 3A.8 — שמירת Artifacts

```python
import joblib
artifacts = {
    'minirocket': minirocket,
    'scaler': scaler,
    'lr': lr,
    'model_version': 'stage3_v1.0',
    'trained_at': datetime.now().isoformat()
}
joblib.dump(artifacts, 'stage3_ai_pipeline.joblib')
```

---

## ✅ 3B) Pipeline וולידציה (Verify)

| בדיקה | תיאור |
|-------|-------|
| **3B.1** | אין Leakage: חלונות מאותו record לא בסטים שונים |
| **3B.2** | ממדי Features עקביים (n_kernels × 2 = 20,000) |
| **3B.3** | טווח Scores תקין: `[0, 1]` |
| **3B.4** | מטריקות בסיס: ROC-AUC, PR-AUC |
| **3B.5** | שמירת Dump מלא לכל חלון |

### פורמט `ai_scores.parquet`:

| שדה | תיאור |
|-----|-------|
| `record_id` | מזהה רשומה |
| `window_idx` | אינדקס חלון |
| `window_start` | תחילת חלון (samples) |
| `window_end` | סוף חלון (samples) |
| `ai_score` | סקור מ־LogisticRegression |
| `y_true` | Label אמיתי |
| `split` | train/val/test |
| `quality_class` | מחלקת איכות מ־Stage 2 |

---

## 🚀 3C) Inference (Runtime)

בזמן אמת / Synthetic — **אין fit!**

```python
class AIScorer:
    def __init__(self, pipeline_path):
        p = joblib.load(pipeline_path)
        self.minirocket = p['minirocket']
        self.scaler = p['scaler']
        self.lr = p['lr']
        self.model_version = p['model_version']
    
    def score_window(self, x_window):
        # x_window: (2, 4800) or (4800, 2)
        features = self.minirocket.transform(x_window[np.newaxis])
        scaled = self.scaler.transform(features)
        return self.lr.predict_proba(scaled)[0, 1]
```

---

### 🏆 קריטריון PASS

```
✓ אימון והסקה רצים על CPU בזמן סביר
✓ אין Data Leakage בין סטים (GroupSplit)
✓ StandardScaler fit על TRAIN בלבד
✓ ai_score ∈ [0, 1] לכל החלונות
✓ Artifacts שמישים להרצה (load + predict עקבי)
```

---

</div>
