<div dir="rtl" style="text-align: right;">

# 📌 דאשבורד פרויקט SentinelFetal

**מקור האמת (Single Source of Truth) לסטטוס הפרויקט, מסמכים ותוצרים.**
נכון לתאריך: 27/01/2026

---

## 🚦 סטטוס נוכחי: Stage 1 Completed & Certified

הפרויקט סיים בהצלחה את שלב ה-Ingest וה-Preflight. הדאטה עבר נרמול, ניקוי וסרטיפיקציה מחמירה.

> **עדכון חשוב (Max_T Strategy):** בניגוד לתכנון המקורי לאורך קבוע, הוחלט שבעת האימון (Training) נשתמש ב-**Dynamic Max_T** המבוסס על הרשומה הארוכה ביותר בדאטה (נקבע בזמן ריצה ע"י `build_dataset_v1.py`). בריל-טיים, נעבוד עם Fixed Ring Buffer של 20 דקות.

| שלב הארכיטקטורה | סטטוס | ארטיפקטים מרכזיים |
| :--- | :--- | :--- |
| **1. Ingest & Preflight** | ✅ **CERTIFIED (V1.1)** | `X_norm_4hz.npy`, `mask_fhr/uc.npy`, `manifest.csv` |
| **2. Windowing** | 🚧 **Ready for Dev** | (מתוכנן: חיתוך לחלונות 20 דקות) |
| **3. AI Baseline (MiniRocket)** | 🚧 **Ready for Dev** | (מתוכנן: אימון מודל Baseline) |
| **4. Thresholds** | 📜 **Planned** | (מתוכנן: כיול ספים) |

---

## 📂 אינדקס מסמכים (Documentation Structure)

כל הידע בפרויקט מרוכז בתיקיית `docs/` במבנה הבא:

### 0. קונפיגורציה מרכזית (Single Source of Truth)
*   **[`src/config.py`](../src/config.py)**: הקובץ היחיד המגדיר ספים קליניים (pH, Baseline), צבעי UI, הגדרות מודל וקבועי זמנים. כל שינוי בפרמטרים חייב להיעשות כאן.

### 1. אפיונים טכניים (`docs/01_specs`)
*   **[ממשק קלט למודל (AI Input)](01_specs/01_ai_input_specification.md)**: הגדרת הטנזורים.
*   **[מפרט Data Pipeline V1](01_specs/02_data_pipeline_v1_spec.md)**: החוקים העסקיים לניקוי ונרמול הדאטה.

### 2. מחקר ואסטרטגיה (`docs/02_research`)
*   **[אסטרטגיית דאטה מודרנית](02_research/01_modern_data_strategy.md)**: למה בחרנו ב-Edge AI ו-CPU Only.
*   **[פורמט אופטימלי](02_research/02_optimal_data_format.md)**: ההחלטה על `npy` ו-Memory Mapping.
*   **[שאלות ותשובות אסטרטגיות](02_research/03_comprehensive_strategy_qa.md)**: הרחבה על ההחלטות.
*   **[ניתוח דאטה ראשוני](02_research/05_initial_data_analysis.md)**: הכרת ה-WFDB.
*   **[שימור אינווריאנטים](02_research/signal_invariants_analysis.md)**: (מתוכנן) ניתוח חוקי הברזל של האות (טיוטה - קישור עתידי).

### 3. דוחות Stage 1 (`docs/03_stage1_reports`)
*   **[סיכום מנהלים](03_stage1_reports/01_summary.md)**: מה בוצע ב-Stage 1.
*   **[תוכנית סרטיפיקציה](03_stage1_reports/02_certification_plan.md)**: מה תכננו לבדוק.
*   **[רציונל בדיקות (מומלץ!)](03_stage1_reports/03_certification_rationale.md)**: הסבר מעמיק על 12 בדיקות האיכות (A1-A12).

### 4. דוחות ניהול (`docs/00_management`)
*   **[היסטוריית משימות](00_management/task_history.md)**: לוג משימות טכני.
*   **[Walkthrough](00_management/project_walkthrough.md)**: יומן מסע טכני של הפיתוח.
*   **[Stage 4 Walkthrough](00_management/Stage_4_Walkthrough.md)**: Calibration & Persistence - תוצאות ובדיקות ✅
*   **[Stage 5 Walkthrough](00_management/Stage_5_Walkthrough.md)**: Smart Hybrid - Tiering, Boredom Gate ✅
*   **[Stage 6 Walkthrough](00_management/Stage_6_Walkthrough.md)**: E2E Integration - תוצאות ובדיקות ✅
*   **[Implementation Status](00_management/IMPLEMENTATION_STATUS.md)**: סטטוס עדכני של Stages 4-6 ✅
*   **[מצגת Stage 2](00_management/SentinelFetal_Stage_2_Slides.md)**: שקפים לתכנון שלב החילון.

### 5. פירוק שלבים עתידיים (`docs/stages_breakdown`)
פירוט מלא של ארכיטקטורת ההמשך, מפורק לקבצים אטומיים:
*   [Stage 0: Overview](stages_breakdown/SentinelFetal_Stage_0_Overview.md)
*   [Stage 2: Windowing](stages_breakdown/SentinelFetal_Stage_2_Windowing.md)
*   [Stage 3: AI Baseline](stages_breakdown/SentinelFetal_Stage_3_AI_Baseline.md)
*   [Stage 4: Thresholds](stages_breakdown/SentinelFetal_Stage_4_Thresholds.md)
*   [Stage 5: Smart Hybrid](stages_breakdown/SentinelFetal_Stage_5_Smart_Hybrid.md)
*   [Stage 6: E2E Integration](stages_breakdown/SentinelFetal_Stage_6_E2E.md)
*   [Stage 7: Datasets](stages_breakdown/SentinelFetal_Stage_7_Datasets.md)
*   [Appendix](stages_breakdown/SentinelFetal_Appendix.md)


---

## 💎 תוצרים ונתונים (Data Artifacts)

הקבצים המעובדים (האמת בשטח) נמצאים בתיקייה `processed_data_v1/`:

*   📄 **[TRUTH.md](../processed_data_v1/TRUTH.md)** - **דוח הסטטוס החי**. נוצר אוטומטית ע"י הסקריפט, מכיל את תוצאות כל הבדיקות העדכניות.
*   `qc_report.json`: סיכום סטטיסטי JSON.
*   `manifest.csv`: טבלת מטא-דאטה לכל רשומה.

</div>
