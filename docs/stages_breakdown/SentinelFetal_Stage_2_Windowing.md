<div dir="rtl" style="text-align: right;">

# 🪟 Stage 2 — Windowing + Window Quality Gate
## חלונות + שער איכות לחלון

---

### 🎯 מטרה

להמיר את הארטיפקטים של Stage 1 **לסט חלונות קנוני** עבור כל השלבים הבאים.

שלב זה מיישם **חוזה איכות לחלון** (Quality Gating) שמונע ממודלים וחוקים להתבסס על חלונות לא אמינים.

---

### 📥 קלט (Inputs)

| קובץ | תיאור |
|------|-------|
| `X_norm_4hz.npy` | מטריצת נתונים מנורמלת בגודל `(N, Max_T, 2)` |
| `mask_fhr.npy` | מסכת תקינות לערוץ FHR בגודל `(N, Max_T)` |
| `mask_uc.npy` | מסכת תקינות לערוץ UC בגודל `(N, Max_T)` |
| `manifest.csv` | מניפסט הרשומות, כולל `record_id` ו־`raw_len` |
| **Runtime Config** | פרמטרי חלון: `fs`, `window_minutes`, `stride_minutes`, ספים |

---

### 📤 פלט (Outputs)

| קובץ | תיאור |
|------|-------|
| `windows_index.parquet` | אינדקס חלונות — שורה לכל חלון עם מיקום ומטה־דאטא |
| `windows_metrics.parquet` | מדדי איכות לכל חלון: `inv_rate_fhr`, `inv_rate_uc`, `quality_class` |
| `windows_labels.parquet` | תוויות לחלונות (אם קיימות) |
| `X_windows.memmap` | (אופציונלי) נתוני החלונות עצמם בפורמט יעיל |
| `STAGE2_TRUTH.md` | מסמך אמת — עובדות בלבד |
| `stage2_qc_report.json` | דוח איכות מפורט |

---

## 🔨 2A) Pipeline בנייה (Build)

### שלב 2A.1 — טעינת קונפיגורציה

טעינת `runtime_config` הכולל:
- `fs` — תדר דגימה (לדוגמה: 4Hz)
- `window_minutes` — אורך חלון בדקות (לדוגמה: 20)
- `stride_minutes` — קפיצה בין חלונות (לדוגמה: 5)
- ספי איכות לחלון
- מצב `strict` (כן/לא)

### שלב 2A.2 — חישוב פרמטרים בדגימות

```python
W = window_minutes * 60 * fs   # ב־4Hz: 4800 דגימות
S = stride_minutes * 60 * fs   # ב־4Hz: 1200 דגימות
```

### שלב 2A.3 — טעינת נתונים

טעינת `X_norm` ו־masks באמצעות `mmap` או streaming — כדי למנוע קריסת זיכרון.

### שלב 2A.4 — יצירת חלונות לכל רשומה

עבור כל `record` בדאטאסט:

| תת־שלב | פעולה |
|--------|-------|
| **2A.4.1** | קריאת `raw_len` מהמניפסט |
| **2A.4.2** | קביעת טווח חוקי: `[0, raw_len)` — כל מעבר לזה הוא padding |
| **2A.4.3** | יצירת חלונות: `start = 0, S, 2S, ...` עד `raw_len - W` |
| **2A.4.4** | עיבוד כל חלון (ראה פירוט למטה) |

#### פירוט שלב 2A.4.4 — עיבוד חלון בודד

```
לכל חלון [start, start+W]:
```

| שלב | פעולה | קוד |
|-----|-------|-----|
| **(א)** | חיתוך מטריצות | `x_win = X_norm[i, start:start+W, :]` |
| **(ב)** | חיתוך masks | `m_f = mask_fhr[i, start:start+W]` |
| **(ג)** | חישוב inv_rate | `inv_rate_fhr = mean(m_f)` |
| **(ד)** | קביעת Quality Class | `GOOD / MED / LOW / FAIL` לפי ספים |
| **(ה)** | ציון תקינות | `window_valid_for_ai`, `window_valid_for_rules` |
| **(ו)** | שמירה באינדקס | `record_id, start, end, inv_rate_*, quality_class` |

### שלב 2A.5 — סיכומים גלובליים

- התפלגות `quality_class` על פני כל החלונות
- מספר חלונות שנפסלו
- סיבות הפסילה העיקריות

### שלב 2A.6 — כתיבת Artifacts

כתיבת כל הקבצים: `parquet`, `TRUTH.md`, `qc_report.json`.

---

## ✅ 2B) Pipeline וולידציה (Verify/Certify)

| בדיקה | תיאור |
|-------|-------|
| **2B.1** | אימות פרמטרים: `W` ו־`S` תואמים ל־`fs` ולקונפיג |
| **2B.2** | אימות חיתוך: כל חלון באורך `W` בדיוק |
| **2B.3** | אימות Padding Shield: חלון עם padding חייב להיות פסול |
| **2B.4** | אימות מדדי איכות: חישוב חוזר תואם ל־`windows_metrics` |
| **2B.5** | דטרמיניזם: אותו seed/config → אותה תוצאה |
| **2B.6** | בדיקות ביצועים: זמן סביר ליצירת חלונות |
| **2B.7** | הפקת `STAGE2_CERT.json` עם PASS/FAIL לכל בדיקה |

---

### 🏆 קריטריון PASS

```
✓ 0 חלונות "חצי־חלון" (פחות מ־20 דקות)
✓ 0 חלונות עם padding שמסומנים כ"כשרים"
✓ מדדי inv_rate נכונים ועקביים
✓ Artifacts נטענים בקלות לשלבים הבאים
```

---

</div>
