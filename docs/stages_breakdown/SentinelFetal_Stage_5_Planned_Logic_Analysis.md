<div dir="rtl" style="text-align: right;">

# 🚫 DEPRECATED ❌

## ⚠️ קובץ זה אינו בשימוש עוד

**STATUS**: 🚫 **DEPRECATED** (use consolidated design docs instead)

הקובץ הזה היה ניתוח ביניים מתוך פיתוח. עם סיום ההטמעה:

✅ **תכנון Stage 4** → ראע סימן בחזרה ל [SentinelFetal_Stage_4_Thresholds.md](SentinelFetal_Stage_4_Thresholds.md)  
✅ **תכנון Stage 5** → ראע סימן בחזרה ל [SentinelFetal_Stage_5_Smart_Hybrid.md](SentinelFetal_Stage_5_Smart_Hybrid.md)  
✅ **Walkthrough יישום** → ראע [../../00_management/Stage_5_Walkthrough.md](../../00_management/Stage_5_Walkthrough.md)  
✅ **סטטוס הטמעה** → ראע [../../00_management/IMPLEMENTATION_STATUS.md](../../00_management/IMPLEMENTATION_STATUS.md)

---

# ~~ניתוח חוקים על פי מסמכי התכנון (Stage 4-5 Docs)~~

> **⚠️ מסמך ניתוח (Analysis) [DEPRECATED]:** מסמך זה הינו ניתוח לוגי של התכנון. למפרט המחייב (Authoritative Spec) של הלוגיקה ההיברידית, ראה **[`SentinelFetal_Stage_5_Smart_Hybrid.md`](SentinelFetal_Stage_5_Smart_Hybrid.md)**.

דוח זה עונה על שאלותיך בהתבסס **אך ורק** על קבצי התיעוד בתיקיית `docs/` (בפרט שלבי 4 ו-5), תוך התעלמות מהמימוש בקוד.

## 1. איפה בדיוק מוגדרים ה-Rules במסמכים?

על פי מסמכי התכנון, ה"חוקים" מפוצלים לשני רבדים שונים המוגדרים בקבצים נפרדים:

1.  **חוקי האינטגרציה וההחלטה (Hybrid Logic):**
    *   מוגדרים במסמך: **`docs/stages_breakdown/SentinelFetal_Stage_5_Smart_Hybrid.md`**.
    *   מסמך זה מתאר כיצד המערכת משלבת בין ה-AI לבין ה"חוקים" (באמצעות מנגנון Tiers), אבל **לא** מגדיר את החוקים הקליניים עצמם. הוא מתייחס אליהם כ-"Rules Engine" חיצוני ("Black Box").

2.  **חוקי הסף וההתמדה (Thresholds & Persistence):**
    *   מוגדרים במסמך: **`docs/stages_breakdown/SentinelFetal_Stage_4_Thresholds.md`**.
    *   מסמך זה מגדיר את הכיול הסטטיסטי של ספי ה-AI ואת מנגנון ה-K-of-N.

**הערה חשובה:** ההגדרות הקליניות המפורטות (כגון: "מהי בדיוק האטה משתנה?" או "מהו סף טכיכרדיה?") **אינן** מופיעות בצורה מפורשת (IF-THEN) במסמכי ה-Stage 4/5 שנסקרו. המסמכים מסתמכים על "נייר עמדה ישראלי" ועל ה-Spec הכללי כהנחת יסוד.

---

## 2. מהם בדיוק הבדיקות שהמערכת מבצעת (על פי המסמכים בלבד)?

להלן הלוגיקה שתוארה במסמכים אלו:

### א. לוגיקה היברידית - דירוג החלטה (Tiering Logic)
*   **מקור:** Stage 5 Docs (סעיף 5A.3).
*   **הבדיקות (על כל חלון):**
    1.  **Tier-1 (אדום):** `IF ai_score >= t_high`. (ה-AI בטוח מאוד).
    2.  **Tier-2 (אדום):** `IF t_low <= ai_score < t_high` **וגם** `rule_score >= r_min`. (ה-AI בחשד בינוני + יש תמיכה מהחוקים).
    3.  **Tier-3 (אדום):** `IF rule_severe == True`. (החוקים זיהו מצב חירום, גם אם ה-AI רגוע).
    4.  **Default (ירוק):** אחרת -> אין התראה.

### ב. מנגנון Boredom Gate (סינון רעשים)
*   **מקור:** Stage 5 Docs (סעיף 5A.4).
*   **הבדיקה:**
    *   `IF quality_class == GOOD` **וגם** `no_significant_hits` -> השתקת התראה (Suppress).

### ג. מנגנון התמדה (Persistence / K-of-N)
*   **מקור:** Stage 4 & 5 Docs.
*   **הבדיקה:**
    *   האם ב-**N** החלונות האחרונים (למשל 3), לפחות **K** (למשל 2) חצו את הסף?
    *   רק אם התנאי מתקיים -> מופקת התראה סופית למשתמש.

### ד. ספי AI (כיול סטטיסטי)
*   **מקור:** Stage 4 Docs.
*   **הבדיקה:** הספים `t_low` ו-`t_high` נקבעים סטטיסטית לפי אחוזונים (95% ו-99%) של רשומות בריאות (Negatives), ולא לפי ערך קליני קבוע מראש.

</div>
