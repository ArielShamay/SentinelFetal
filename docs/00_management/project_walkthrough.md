<div dir="rtl" style="text-align: right;">

# יומן מסע (Walkthrough) - פייפליין הנתונים של SentinelFetal

## סקירה כללית
מסמך זה מתעד את היישום הטכני של פייפליין הנתונים עבור פרויקט SentinelFetal לניטור הריונות בסיכון.

## שלב 1: Ingest & Preflight (הוסמך)

**מטרה:** המרת נתוני WFDB גולמיים לפורמט קנוני בעל ביצועים גבוהים, עם בקרת איכות (QC) קפדנית.

### 1. יישום (Implementation)
- **סקריפט:** `scripts/build_dataset_v1.py`
- **לוגיקה:**
    - קריאת 552 רשומות WFDB בתדר 4Hz.
    - יישום **Disjoint QC Masking** (מיסוך מופרד):
        1.  **Padding (ריפוד):** (עדיפות 1) אזורים מעבר לאורך האות המקורי.
        2.  **Missing (חסר):** ערכי `NaN` או `0`.
        3.  **Range (טווח):** ערכים מחוץ לטווח `[50, 220]` (FHR) או `[0, 100]` (UC).
        4.  **Flatline (קו ישר):** בדיקת שונות (Variance) לערוץ הצירים (UC).
    - **נרמול:** סילום Min-Max לטווח `[0, 1]`.
    - **אימפיוטציה:** אינטרפולציה ליניארית לחורים בתוכן (אזורים מסומנים נשארים `True` במסיכות).

### 2. תהליך הסמכה (Certification Process)
- **סקריפט:** `src/data_pipeline/verify_stage1.py`
- **מתודולוגיה:** צ'ק-ליסט סרטיפיקציה בן 12 נקודות (A1-A12) המבטיח שלמות נתונים, אילוצים פיזיקליים ועקביות QC.
- **בדיקות מפתח:**
    - **A5 Padding Shielding:** וידוא שאזורי ריפוד מסומנים כלא-תקינים במסיכות.
    - **A10 QC Consistency:** וידוא שסטטיסטיקות ה-QC הגלובליות תואמות את סכימת המסיכות הבינאריות (בדיקת סכום).
    - **A12 Performance:** בדיקת מהירות IO.

### 3. תוצאות (V1.1)
- **סטטוס:** ✅ **עבר (PASS)**
- **תאריך:** 27/01/2026
- **ארטיפקטים:** `processed_data_v1/`
    - `X_norm_4hz.npy` (מוכן ל-Memmap)
    - `mask_fhr.npy`, `mask_uc.npy`
    - `manifest.csv`
    - `qc_report.json`
- **מקור האמת:** ראה [`TRUTH.md`](../processed_data_v1/TRUTH.md) למדדים מלאים ולוג אימות.

### 4. קטעי קוד (Code Snippets)

#### לוגיקת מיסוך מופרד (Disjoint Masking Logic)
```python
# העדיפות: Missing > Range
m_fhr_miss = (fhr_raw == 0) | np.isnan(fhr_raw)
m_fhr_range = (fhr_raw < FHR_MIN) | (fhr_raw > FHR_MAX)

# Range נחשב רק אם המידע לא חסר מלכתחילה
cnt_fhr_range = (m_fhr_range & ~m_fhr_miss).sum()
```

#### בדיקת עקביות QC (מתוך סקריפט האימות)
```python
# לוגיקת ולידציה:
calc_inv_total = mask_fhr.sum()
report_inv_total = qc["fhr_inv_total"]
diff = abs(calc_inv_total - report_inv_total)
```

### 5. שדרוג התיעוד (Documentation Upgrade)
**סטטוס: הושלם**
- **מיגרציה:** העברת כל התיעוד לתיקיית `docs/`.
- **דאשבורד חי:** הפיכת דוח ה-HTML לדאשבורד הטוען קבצי Markdown דינמית.

### 6. שיפורי חווית משתמש (Technical Docs UX)
**סטטוס: הושלם**
- **ניווט צדדי:** הוספת סרגל צד דביק (Sticky) למעבר מהיר בין קבצים.
- **שליטה בגופן:** הוספת כפתורי התאמת גודל טקסט (+/-) לקריאות משופרת.

</div>
