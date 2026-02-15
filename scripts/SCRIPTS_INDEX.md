<div dir="rtl" style="text-align: right;">

# אינדקס סקריפטים (Scripts Registry)

קובץ זה מתעד את כל הסקריפטים הביצועיים בפרויקט, ממוינים לפי קטגוריה.
**חובה לעדכן קובץ זה בכל הוספה של סקריפט חדש.**

## 🏭 Pipeline (עיבוד נתונים)
**מיקום:** `scripts/pipeline/`
*   **`build_dataset_v1.py`**: הסקריפט הראשי לביצוע Stage 1. קורא קבצי WFDB, מחיל מסיכות QC, מנרמל, ויוצר את תוצרי ה-NumPy והמניפסט.
*   **`stage2_build.py`**: הסקריפט הראשי לביצוע Stage 2. חותך את הנתונים לחלונות, מחשב מדדי איכות, ויוצר את `windows_index.parquet`.
*   **`train_stage3_minirocket_lr.py`**: אימון AI Baseline (Stage 3): MiniRocket + StandardScaler + LogisticRegression. מייצר `stage3_ai_pipeline.joblib` ו-`STAGE3_TRUTH.md`.
*   **`stage6_e2e.py`**: (משולב) הרצת המערכת המלאה (Headless Simulation) לבחינת אינטגרציית E2E. מאתחל סימולטור, מזריק אירועים, ובוחן תגובת Pipeline.

## 🧪 Validation (בדיקות והסמכה)
**מיקום:** `scripts/validation/`
*   **`clinical_validation_suite.py`**: (לשעבר `verify_stage1.py`) מריץ את סט הבדיקות המלא (A1-A12) כדי להסמיך את הדאטה-סט. מחייב מעבר מלא כדי לקבל חותמת CERTIFIED.
*   **`verify_stage2.py`**: מריץ את סט הבדיקות (V1-V6) להסמכת שלב 2. מוודא גודל חלונות, Padding Shield, ומדדי איכות.
*   **`eval_stage3_minirocket_lr.py`**: הערכת Stage 3 על כל ה-splits. מייצר `ai_scores.parquet` עם סקורים לכל חלון.
*   **`verify_stage6.py`**: בדיקת תקינות אינטגרציית Stage 6. מוודא קיום רכיבים קריטיים, ייבוא נכון של Stage 5, וזמינות WebSocket.


## 📊 Reporting (דוחות וויזואליזציה)
**מיקום:** `scripts/reporting/`
*   **`generate_report_data.py`**: אוסף את נתוני המניפסט וה-QC ומייצר את קבצי ה-JSON שדוח ה-HTML (בדפדפן) צורך.

## 🛠️ Setup & Ops (תשתית)
**מיקום:** `scripts/setup/`
*   **`setup-dev.bat` / `setup-dev.sh`**: סקריפטים להתקנת סביבת הפיתוח (Virtualenv + Requirements) באופן אוטומטי בווינדוס/לינוקס/מק.

---

### הנחיות לפיתוח (Developer Guidelines)
1.  **הרצה:** יש להריץ את כל הסקריפטים מתיקיית השורש של הפרויקט.
    *   דוגמה: `python scripts/pipeline/build_dataset_v1.py`
2.  **מיקום:** אין לשמור קבצי `.py` בתיקיית `scripts/` הראשית. יש לשייך לתת-תיקייה מתאימה.

</div>
