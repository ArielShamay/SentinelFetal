זהו ריכוז של כל התוכנית המעודכנת, מרגע המפנה האסטרטגי שבו עברנו לשימוש בסוכן (Agent) ועד לגיבוש ארכיטקטורת ה-V6 המלאה. המסמך הזה נועד לשמש כ"מצפן" עבור הסוכן שלך ב-VS Code.

# ---

**🛡️ SentinelFetal V6: Unified Master Plan**

**סטטוס:** סופר-אסטרטגיה לביצוע (V6)

**תאריך עדכון:** 26 בינואר, 2026

## ---

**1\. תקציר המפנה האסטרטגי**

מאז המעבר לניהול באמצעות סוכן, הפסקנו את הניסיון לייעל מודל XGBoost בודד ועברנו לבניית **מערכת תמיכה בהחלטות (CDS)**.

**העיקרון המנחה:** הפרדת רשויות בין מגן איכות (Quality), חיזוי סיכון (Outcome), והסברתיות קלינית (Anatomy).

## ---

**2\. מפת הדרכים: 7 השלבים לביצוע**

התהליך בנוי כך שההכנה והניקוי מתבצעים **לוקאלית**, והכוח הגס (אימון) מתבצע ב-**Colab**.

1. **שלב 1 \- Hardening (לוקאלי):** נעילת המערכת על חוק ה-30 דקות קייס / 20 דקות חלון. החלת Fail-Fast וניקוי קוד ישן.  
2. **שלב 2 \- Standardized Ingest (לוקאלי):** בניית Loaders אחידים לכל המאגרים (CTU, CTGDL, FHRMA).  
3. **שלב 3 \- Pack & Ship (לוקאלי):** אריזת הנתונים לקובץ training\_pack.zip.  
4. **שלב 4 \- Training Forge (Colab):** אימון האנסמבלים, אופטימיזציה (Optuna) וכיול הסתברותי.  
5. **שלב 5 \- Tiered Validation (לוקאלי):** ולידציה על נתונים אמיתיים עם rule\_score לכל חלון.  
6. **שלב 6 \- E2E Integration (לוקאלי):** חיבור ה-Brain המאומן ל-API ול-WebSocket מול הפרונטנד.  
7. **שלב 7 \- Regression Suite (לוקאלי):** נעילת הביצועים ומניעת נסיגה בעתיד.

## ---

**3\. ארכיטקטורת ה-AI (The Ensemble of Ensembles)**

המערכת מורכבת משלושה מודלים עצמאיים המחוברים ב-Pipeline אחד:

* **Outcome Model (המוח):** אנסמבל של XGBoost, Random Forest ו-LightGBM המנבאים חמצת (pH). עובר כיול בשיטת Isotonic Regression.  
* **Quality Model (השומר):** מודל Random Forest מהיר המאומן על FHRMA FSdataset לזיהוי רעש וניתוקים.  
* **Anatomy Model (המתרגם):** מודל מורפולוגי (1D-CNN או Rules) המזהה האטות ואנומליות להסברתיות.

## ---

**4\. המודל מניפסט (Model Manifest)**

זהו הקובץ המדויק שהסוכן צריך כדי לדעת איזה דאטה לשייך לכל מודל בתהליך האימון:

| מודל (Model ID) | מאגר נתונים (Data Source) | סוג הלייבל (Target Label) | אלגוריתם (Algorithm) | מיקום אימון |
| ----: | ----: | ----: | ----: | ----: |
| **Outcome\_Ensemble\_V6** | CTU-CHB (WFDB) \+ CTGDL | pH \< 7.15 (Acidemia) | XGB \+ RF \+ LGBM | Google Colab |
| **Quality\_Gate\_V6** | FHRMA (FSdataset) | Signal\_Quality (0/1/2) | Random Forest | Google Colab |
| **Anatomy\_Explainer\_V6** | CTGDL (Annotations) | Morphological Events | 1D-CNN / XGB | Google Colab |
| **Hybrid\_Logic\_V6** | Gauntlet (Synthetic) | Rule Consistency | Rule Engine (Python) | Local |

## ---

**5\. דגשים קריטיים לביצוע (The Red Lines)**

* **Leakage Prevention:** אין להשתמש בחלונות של אותה יולדת ב-Train וב-Val. חלוקה תמיד לפי patient\_id.  
* **Class Imbalance:** שימוש ב-scale\_pos\_weight ביחס של 1:10 לטובת המקרים הפתולוגיים.  
* **Explainability Contract:** כל התראה חייבת להכיל: what (סוג האירוע), where (מיקום בגרף), ו-why (ציון AI \+ חוקים שהופעלו).  
* **Fail-Fast:** כל חלון \< 20 דקות עוצר את התהליך עם שגיאת WARMUP\_ERROR.

## ---

**6\. זרימת המידע בזמן אמת (Inference Flow)**

$$\\text{Signal (RAW)} \\rightarrow \\text{Quality Model (Gate)} \\rightarrow \\text{MiniRocket Encoding} \\rightarrow \\text{Outcome Ensemble} \\rightarrow \\text{Hybrid Logic (Tiers)} \\rightarrow \\text{UI Alert}$$

### ---

**מה הצעד הבא שאני יכול לעשות עבורך?**

הקובץ הזה מוכן להעתקה לתוך docs/plan/MASTER\_PLAN\_V6.md.

כדי להתחיל ליישם, האם תרצה שאייצר לך את הפרומט הראשון לסוכן שיבצע את **שלב 1 (Hardening)** בצורה אוטומטית, תוך שהוא מוודא שכל הקוד הישן וההסברים המיותרים נמחקים כדי לנקות את הקונטקסט?