<div dir="rtl" style="text-align: right;">

# תוכנית סרטיפיקציה (Certification Plan) - Stage 1

מטרת התוכנית: אישור סופי ומוחלט לשלב הנתונים הראשון, תוך הוכחה מתמטית שהנתונים תקינים, עקביים ומייצגים את המציאות (Ground Truth).

## שלב 1: שדרוג ה-Build Script
**סקריפט:** `scripts/build_dataset_v1.py`
**שינויים נדרשים:**
*   **Disjoint Masks:** לוגיקה שמבטיחה שכל "דגימה פסולה" משויכת לסיבה אחת בלבד לפי סדר עדיפות:
    1.  **Padding:** (הכי חשוב, אזור "מחוץ לקלט")
    2.  **Missing:** (ערך 0 או NaN)
    3.  **Out of Range:** (ערך לא פיזיולוגי)
    4.  **Flatline** (עבור UC)
*   **QC Breakdown:** חישוב המונים (Counters) לפי הקטגוריות הנ"ל בדיוק, כך שסכומם יהיה שווה בדיוק ל-Total Invalid.

## שלב 2: פיתוח סקריפט הסרטיפיקציה
**סקריפט:** `src/data_pipeline/certify_stage1.py`
**בדיקות (Checks A-I):**
*   **A - Integrity:** קיום קבצים, Shapes, חוסר NaN/Inf, טווחים תקינים.
*   **B - Manifest:** עקביות מזהים, אורכים.
*   **C - Padding:** בדיקה קריטית - האם כל מה שמעבר ל-`raw_len` מסומן כ-`True` במסיכה.
*   **D - QC Breakdown:** וידוא שסכום הסיבות שווה לסה"כ הפסולים.
*   **E - UC Missing:** בדיקה סטטיסטית על רצפי אפסים ב-UC.
*   **F - Flatline Consistecy:** בדיקה מדגמית (Sampling) של מקטעים שסומנו.
*   **G - Raw Cross-check:** טעינה מחדש של WFDB והשוואה "ביט-לביט" מול המסיכה השמורה.
*   **H - Windowing:** הוכחת היתכנות לחלוקה לחלונות (20 דק').
*   **I - Performance:** מדידת זמנים אמיתית.

## שלב 3: הרצה והפקת תוצרים
1.  הרצת `build_dataset_v1.py` המשופר.
2.  הרצת `certify_stage1.py`.
3.  בחינת תיקיית ההוכחות `processed_data_v1/cert/`.
4.  עדכון אוטומטי של `TRUTH.md` עם חותמת "CERTIFIED" או רשימת כשלי בדיקה.

</div>
