<div dir="rtl" style="text-align: right;">

### שלב 1: Ingest & Preflight (הוסמך)
**מטרה:** המרת נתוני WFDB גולמיים לפורמט NumPy קנוני עם בקרת איכות (QC) קפדנית.

**סטטוס:** ✅ **הוסמך (V1.1)**

**הישגים מרכזיים:**
- **לוגיקת QC מופרדת (Disjoint):** יושמה מערכת מסיכות מבוססת עדיפות (Padding > Missing > Range).
- **תוצרים מוסמכים:**
    - `X_norm_4hz.npy`: מערך מנורמל במימדים (552, 21620, 2).
    - `mask_fhr.npy` / `mask_uc.npy`: מסיכות בינאריות לתקינות דאטה.
    - `manifest.csv`: מטא-דאטה וסטטיסטיקה לכל רשומה.
    - `qc_report.json`: סיכום סטטיסטיקה גלובלי.
- **אימות (Verification):** בוצע מעבר בהצלחה של כל 12 הבדיקות (A1-A12), כולל Padding Shielding ו-QC Consistency.
- **תיעוד:** נוצר דוח [`TRUTH.md`](../processed_data_v1/TRUTH.md) המהווה את "מקור האמת" היחיד.

**צעדים הבאים:**
1.  **שלב 2 (Quality Gate):** סינון רשומות על בסיס שיעורי ה-invalid ב-`manifest.csv`.
2.  **חיתוך דאטה (Segmentation):** חלוקת הדאטה הרציף לחלונות של 20 דקות לאימון.

# רשימת משימות: הפייפליין של SentinelFetal

- [/] **שלב 2: Stage 2 - Windowing Implementation**
    - [ ] **יישום Quality Gate (סינון)**
    - [ ] **לוגיקת חיתוך (Windowing 20m)**

- [ ] **שלב 3: Stage 3 - AI Baseline (MiniRocket)**
    - [x] **מיגרציית תיעוד ל-`docs/`**
    - [x] **שדרוג דוח ה-HTML לתיעוד דינמי**

- [x] ניתוח פורמט ומבנה הנתונים הגולמיים <!-- id: 0 -->
- [x] יצירת דוח ניתוח ראשוני <!-- id: 1 -->
- [x] מחקר פורמט נתונים אופטימלי למודלי AI <!-- id: 2 -->
- [x] יישום V1 Data Pipeline (Robust) <!-- id: 3 -->
    - [x] יצירת `data_pipeline_v1_spec.md` <!-- id: 4 -->
    - [x] יצירת `build_dataset_v1.py` (מסכות, QC, מניפסט) <!-- id: 8 -->
    - [x] יישום `certify_stage1.py` עם צ'ק-ליסט קפדני (A1-A12)
    - [x] הרצת סרטיפיקציה ויצירת `TRUTH.md`
    - [x] **אבן דרך: הסמכת שלב 1 (Stage 1 Certified)**
    - [x] הרצת פייפליין ויצירת ארטיפקטים (X, masks, report) <!-- id: 9 -->
    - [x] אימות דוח QC ותוכן מניפסט <!-- id: 10 -->
- [ ] יצירת דוגמת Data Loader ל-PyTorch/TensorFlow (אופציונלי) <!-- id: 7 -->

</div>
