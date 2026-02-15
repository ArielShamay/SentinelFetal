<div dir="rtl" style="text-align: right;">

# מפרט טכני: Data Pipeline V1 (Robust)

מסמך זה מגדיר את הסטנדרט לבניית הדאטה-סט הקנוני (Canonical Dataset), בהתאם להנחיות ה-Review האחרון.

## עקרונות ליבה (Core Principles)
1.  **שמירת האמת (Ground Truth):** לא מוחקים מידע חסר. שומרים מסיכה (`mask`) שמציינת איפה המידע היה לא תקין.
2.  **ניקוי מבוקר:** האימפיוטציה (השלמת נתונים) נעשית רק לטובת הקלט למודל, אך המסיכה נשמרת בנפרד.
3.  **נרמול דומיין:** נרמול מתבצע רק אחרי סינון ערכים לא פיזיולוגיים, וביחס לטווחים קליניים קבועים.
4.  **עקיבות (Traceability):** הפקת דוח QC (בקרת איכות) ומניפסט (Manifest) לכל רשומה.

## שלבי העיבוד (Pipeline Steps)

### 1. קריאה (Read & Decode)
*   **מקור:** קבצי WFDB (`.dat` + `.hea`).
*   **פעולה:** קריאת ערוצי FHR ו-UC.
*   **תדר:** שמירה על 4Hz (ללא דילול).

### 2. זיהוי חריגים ויצירת מסיכות (Invalid Detection & Masking)

לכל ערוץ תיווצר מסיכה בינארית (`1` = לא תקין, `0` = תקין).

**ערוץ דופק (FHR Masks):**
*   **Missing:** ערך `0` או `NaN`.
*   **Out of Physiology:** ערך מתחת ל-50 bpm או מעל 220 bpm.
*   **Spikes:** (Not implemented in V1).

**ערוץ צירים (UC Masks):**
*   **Missing:** ערך `NaN`.
*   **Flatline:** זיהוי מקטעים שטוחים לחלוטין (רעש חיישן).

### 3. מילוי חוסרים (Controlled Imputation)
*   מילוי ערכים חסרים (איפה שהמסיכה דולקת) באמצעות **אינטרפולציה ליניארית**.
*   **מטרה:** יצירת רצף נקי עבור רשת הנוירונים (Conv1D לא יודעת לאכול NaN).

### 4. נרמול (Normalization)
*   חיתוך (Clip) לטווחים הקליניים:
    *   FHR: [50, 220]
    *   UC: [0, 100]
*   נרמול לטווח [0, 1] (Min-Max Scaling).

### 5. שמירת תוצרים (Artifact Generation)

הקבצים יישמרו בתיקייה `processed_data_v1`:

1.  **`X_norm_4hz.npy`**: המידע המעובד.
    *   Shape: `(N_Records, Max_Time, 2)`
    *   Type: `float32`
2.  **`mask_fhr.npy`**: מסיכת תקינות דופק.
    *   Shape: `(N_Records, Max_Time)`
    *   Type: `bool` / `uint8`
3.  **`mask_uc.npy`**: מסיכת תקינות צירים.
    *   Shape: `(N_Records, Max_Time)`
    *   Type: `bool` / `uint8`
4.  **`manifest.csv`**: טבלת מטא-דאטה.
    *   עמודות: `record_id`, `duration_sec`, `fhr_missing_pct`, `uc_missing_pct`, `label_ph`...
5.  **`qc_report.json`**: סיכום סטטיסטי גלובלי (כמה רשומות נזרקו, התפלגות איכות).

</div>
