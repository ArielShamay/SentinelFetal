

להלן גרסה “מסודרת להעברת שרביט” — מסמך Markdown מלא, שמרכז את הארכיטקטורה, הדאטה, מה שבוצע עד עכשיו (Step 1, Step 2, Step 2.1, Step 2.2), ומה הם השלבים הבאים עד מערכת רצה עם JSON ל-UI. שילבתי ותיקנתי נקודות מהטיוטה שלך, והוספתי פרטים שחסרו (במיוחד: מקור־אמת, סדר הצנרת, מבנה ה-NPZ/manifest, וכללי Fail-Fast).

SentinelFetal V6 — Handoff Summary (תשתית → דאטה → אימון → אינטגרציה)

תאריך: 2026-01-27

סטטוס: PASS מלא של Step 1 (Hardening) + Step 2/2.1/2.2 (Ingest+Pack+Extraction) ברמת Pre-AI; מוכנים ל-Training Forge (Colab) לפי ה-Master Plan.

0) עקרונות יסוד (החוקים שאסור לשבור)

Steel Wall: הפרדה מוחלטת Pre-AI ↔ AI

כל מה שב־Pre-AI חייב להיות “סטרילי” וקל משקל: בלי Torch, בלי momentfm, בלי מודלים כבדים. ה-Pre-AI הוא הצנרת שצריכה לרוץ גם בחדר לידה בלי הפתעות.



Runtime Invariants (Single Source of Truth)

הפרמטרים המחייבים נמצאים ב־config/runtime.yaml, ונקראים ע״י src/utils/runtime_config.py.

חובה ליישר לכל המערכת (לוקאלי + קולאב):

fs_hz = 4

window_minutes = 20, stride_minutes = 5

min_window_minutes = 20 (אין ניתוח מתחת ל-20 דק׳)

recommended_case_minutes = 30 (ל-Gauntlet/Packaging: קייסים “מלאים”)

strict_mode = true (Fail-Fast, בלי “להסתדר”)

Fail-Fast & WARMUP_ERROR

כל חלון קצר מדי (או נתון RAW לא תקין) חייב לזרוק שגיאה מפורשת WARMUP_ERROR ולעצור.

1) מפת דרכים כללית (Master Plan — 7 שלבים)

המעבר החשוב: הכנה וניקוי לוקאלית, אימון “כוח גס” ב-Colab.



Step 1 — Hardening (Local): נעילת 30m/20m, Strict Mode, ניקוי Legacy.

Step 2 — Standardized Ingest (Local): Loaders אחידים ל-CTU/CTGDL/FHRMA.

Step 3 — Pack & Ship (Local): אריזה ל-training_pack.zip.

Step 4 — Training Forge (Colab): אימון + Optuna + Calibration.

Step 5 — Tiered Validation (Local): ולידציה עם rule_score לכל חלון.

Step 6 — E2E Integration (Local): חיבור Brain מאומן ל-API/WS מול UI.

Step 7 — Regression Suite (Local): מניעת נסיגות, CI, “Gauntlet Gate”.

2) מבנה הקוד (איך הפרויקט “מחולק” כיום)

2.1 Pre-AI Core (הקיר הפלדה)

נתיב: src/v6/pre_ai/

ה־Pipeline המחייב הוא: RAW → Invariants → Quality Gate → Windowing.

קבצי מפתח:



pipeline.py — הזרימה הרשמית: קודם assert_raw_invariants, אחרי זה quality_gate, על גבי חלונות מ־window_iter.

invariants.py — בדיקות RAW (אורך, numeric, alignment) + WARMUP_ERROR (מוזכר ב-SPECS).

quality_gate.py — איכות RAW פר חלון: NaNs, zeros, jumps, out-of-range, flatline וכו׳.

windowing.py — חיתוך לחלונות 20 דק׳ ו-stride 5 דק׳ (הגדרה רשמית ב-SPECS).

2.2 Ingest (ה”מפעל” שמכין דאטה לאימון)

נתיב: src/v6/pre_ai/ingest/



schema.py — StandardizedRecord + חישוב איכות ברמת record (valid_frac, nan_frac, out_of_range וכו׳).

ctu_loader.py — טעינת CTU-CHB (WFDB) לרקורד אחיד

ctgd_loader.py — טעינת CTGDL (CSV/metadata) לרקורד אחיד

fhrma_loader.py — טעינת FHRMA (FSdataset; כולל שלב decoding ב-Step 2.2 לפי סיכום הסוכן)

2.3 Packager + Verifier (האריזה שמייצרת ZIP מאומן)

scripts/build_v6_training_pack.py — בונה zip, יוצר manifest, כותב NPZ לכל מטופלת, ומחשב סטטיסטיקות.

scripts/verify_v6_pipeline_e2e.py — מוודא “Green Light” שהדאטה יוצא מה-zip ועובר Pre-AI.

3) נכסי דאטה (מה יש לנו ולמה כל אחד משמש)

ל-V6 יש 3 מאגרי אמת + מאגר סינתטי לבדיקות:



CTU-CHB (WFDB)

מה מכיל: רשומות CTG עם תוצאה קלינית (pH/Acidemia).

למה משמש: Outcome Model (המוח) — חיזוי חמצת.

CTGDL (CSV/Annotations)

מה מכיל: הרבה רשומות CTG + אנוטציות/אירועים מורפולוגיים (כמו האטות/האצות, תלוי בתת-סט).

למה משמש:

תיוג מורפולוגי → Anatomy Model (“המסביר”).

בנוסף, לפי ה-Manifest בטופס התכנון: משמש גם להרחבת אימון ה-Outcome יחד עם CTU.

FHRMA / FSdataset (Binary/Noise heavy)

מה מכיל: דאטה שמדגיש רעשים/ניתוקים/False Signals (זה “הדלק” למודל האיכות).

למה משמש: Quality Model (“השומר”).

Gauntlet (Synthetic)

מה מכיל: סיגנלים סינתטיים + הזרקות תקלה (רעש, קפיצות, ניתוקים) כדי לבדוק “מה נשבר”.

למה משמש: בדיקות מערכת + עקביות Rule Engine / Hybrid Logic (בהמשך).

4) מה ביצענו עד עכשיו (Step 1 + Step 2 + 2.1 + 2.2)

Step 1 — Hardening (PASS)

מטרה: “לנעול” את המערכת על החוקים, ולמנוע זליגות AI למסלולי Pre-AI.

ההגדרות המחייבות מופיעות ב-SPECS (Strict Mode, windowing, WARMUP_ERROR, RAW-first).

Step 2 — Standardized Ingest + Pack (PASS)

מטרה: Loaders אחידים + Packager שמייצר zip אחד לאימון.



Step 2.1 — Quality Recalibration (PASS)

כאן עשינו שינוי חשוב: במקום “לזרוק חצי עולם ל-LOW”, עברנו למדיניות מרכזית אחת (policy) + gap-fill קטן בלבד.

המדיניות וה-guardrails מתועדים בדוח Step2.1:

thresholds לשמירה על record (למשל keep עד nan_frac גבוה),

ו-interpolation רק לפערים קטנים: max_gap_samples_to_fill: 10 (2.5 שניות ב-4Hz).

Step 2.2 — Deep Extraction & Decoding (PASS לפי סיכום הסוכן)

מטרה: לפתור bottleneck בפורמט הקבצים:



CTGDL היה בארכיונים (tar.gz) → חילוץ ל-data/_extracted/...

FHRMA היה בפורמטים בינאריים → decoding והפקת רשומות

בסוף Step2.2 הסוכן מדווח שה-pack v6.2 כולל CTU+CTGDL+FHRMA, ואימות E2E עבר לכל dataset (Green Light). (זה מידע מהסיכום שלך/של הסוכן — לא מצורף כאן כקובץ מקור לקריאה ישירה).

5) Pre-AI Pipeline — הסבר מדויק “מה קורה ומתי”

5.1 RAW Invariants (לפני כל דבר)

הסדר המחייב: קודם RAW invariants, ורק אחר כך איכות וחלונות.

הבדיקות המחייבות:

אורך מינימלי + חישוב מול min_window_minutes/recommended_case_minutes

numeric-only (לא strings)

alignment: len(fhr)==len(uc) + אותו fs

Fail behavior:

<20m → WARMUP_ERROR (אין “ניתוח בכאילו”).

בפועל, run_pre_ai עושה זאת ראשון, לפני windowing.



5.2 Quality Gate (Iron Dome) — על RAW בלבד

ה-Quality Gate רץ על חלון RAW ומחזיר:



quality_class (HIGH/MED/LOW)

hard_low

diagnostics/metrics

החישובים כוללים, בין היתר:



nan_frac / valid_frac

zeros_frac

max_nan_run

std, mad

jumps: jump_count_gt25, max_abs_jump

out_of_range 60..220

unique_ratio / flatline_ratio

הגיון “Fail-Fast” של איכות:



חלון קצר מדי → WARMUP_ERROR

אי-יישור FHR/UC בחלון → ValueError

איך וידאנו שזה עובד?



SPECS קובע טלמטריה מינימלית (warnings/fallback/invariants/warmup) ו-Strict Mode (warnings-as-errors).

בפועל run_pre_ai מפיק לכל חלון רשומה עם quality_class + indices, ומוודא שאם אין חלונות בכלל זה FAIL.

Step2.1 הוסיף מדיניות מרכזית + gap-fill קטן בלבד (לא מזייף RAW), עם תיעוד ספים.

5.3 Windowing — 20 דקות חלון, stride 5 דקות

ההגדרה הרשמית:



window_samples = 20*60*4 = 4800

stride_samples = 5*60*4 = 1200

וכל חלון קצר מזה הוא WARMUP_ERROR.

בפועל run_pre_ai משתמש ב־window_iter(...) עם הפרמטרים מה-runtime config.

6) מה עשינו עם הדאטה עד שהוא “מוכן לאימון”

6.1 StandardizedRecord (השפה האחידה של כל הדאטהסטים)

כל loader מחזיר StandardizedRecord שמכיל:



patient_id

fhr_raw, uc_raw (1D)

fs_hz

source (CTU/CTGDL/FHRMA)

labels (תלוי dataset)

meta (סטטיסטיקות, decoding info וכו׳)

record_quality (מדדי איכות ברמת record)

בנוסף, מחשבים record_quality (valid/nan/zeros/out_of_range/duration).



6.2 Training Pack (ZIP) — איך הוא בנוי

ה-Packager שומר לכל מטופלת קובץ NPZ דחוס עם שדות קבועים:



fhr_raw, uc_raw

fhr_filled, uc_filled (אם לא קיים fill → שומר raw)

fs_hz, patient_id, source

labels_json, meta_json, record_quality_json

בנוסף, נוצר manifest עם:



runtime_config (fs/window/stride/min_window)

quality_policy

patients[] (כולל included/skip_reason/avg_quality וכו׳)

stats (counts, skipped_by_reason, duration stats, quality_class_counts)

המשמעות הפרקטית: ב-Colab אפשר לטעון manifest פעם אחת, לבחור subset לפי dataset/label_type, ואז לטעון NPZ ספציפיים — בלי “לסרוק את הכל”.

7) איך משתמשים בדאטה לכל מודל (ומה כל מודל אמור לעשות)

ה-Master Plan מגדיר 3 מודלים עצמאיים + Hybrid Logic:



7.1 Outcome Model (“המוח”) — חיזוי מצוקה/חמצת

מטרה: הסתברות ל-Acidemia (למשל pH < 7.15).

מקור דאטה: CTU-CHB + CTGDL (לפי המניפסט).

אלגוריתמים (אנסמבל): XGBoost + RandomForest + LightGBM, עם כיול Isotonic Regression.

מה נכנס: חלון 20 דק׳ (4800 דגימות ב-4Hz) אחרי בדיקות Pre-AI.

7.2 Quality Model (“השומר”) — איכות חיישן/אות ורעש

מטרה: לזהות signal quality / false signals / ניתוקים.

מקור דאטה: FHRMA (FSdataset).

אלגוריתם (לפי master plan): Random Forest מהיר.

הערה קריטית לתכנון אימון: בניגוד ל-Outcome, פה רוצים גם דוגמאות “גרועות”, כי זה בדיוק מה שהמודל צריך ללמוד.

7.3 Anatomy Model (“המסביר”) — זיהוי מורפולוגיה להסברתיות

מטרה: לזהות אירועים כמו האטות/האצות/אנומליות כדי להוציא “הסבר קליני”.

מקור דאטה: CTGDL Annotations.

אלגוריתם: 1D-CNN או XGB/Rules (לפי master plan).

7.4 Hybrid Logic (Local) — שכבת בטיחות

מטרה: לקבל החלטה סופית מ-(Outcome + Quality + Anatomy + Rules), עם persistence/tiers וכו׳ (זה בעיקר Step 5–6).

8) מה השלב הבא (מהיום עד מערכת מלאה)

Step 4 — Training Forge (Colab)

מה עושים:



מעלים training_pack_v6_2.zip ל-Drive/Colab.

טוענים manifest, בוחרים subset לפי dataset/label_type (Outcome/Quality/Anatomy).

מייצרים חלונות (אותו windowing כמו לוקאלי — חייב להיות 20m/5m/4Hz).

Feature extraction (ל-Outcome כנראה MiniRocket/TimeSeries features).

אימון אנסמבלים + CV ברמת patient_id (Zero leakage).

Calibration (Isotonic) + שמירת Model Artifacts + Model Manifest.

דגש: המטרה של Step 4 היא “להוציא מוח” יציב עם כיול נכון, לא רק AUC גבוה.

Step 5 — Tiered Validation (Local)

מריצים את המודלים המאומנים על data אמיתי + משווים מול Rule Engine.

בונים “Gauntlet Gate” ו-metrics: רעש FP, boring FP=0, recall guardrail.

Step 6 — E2E Integration (Local)

מחברים מודלים ל-FastAPI/WS.

מייצרים JSON עשיר: tier_decision, quality_class, rule_hits, highlight_regions, ו-what/where/why.

Step 7 — Regression Suite

טסטים שמונעים חזרה אחורה (CI), כולל stress ל-1–20 מטופלות.

9) Guardrails (מה להיזהר ממנו במיוחד)

Leakage לפי חלונות: חלונות מאותה יולדת אסור שיזלגו בין train/val — חייב split לפי patient_id.

Alignment: כל resampling/decoding חייב לשמור alignment FHR↔UC (אורך זהה).

Interpolation מוגבל: gap-fill קטן בלבד; מעבר לזה — מסמנים signal loss ולא “ממציאים מציאות”.

Strict Mode תמיד: warnings → errors; fallback_count חייב להיות 0.

Single Source of Truth: runtime config + quality policy חייבים להיות זהים לוקאלי וקולאב.

10) מחיקות וניקוי (מה כן/לא למחוק)

מה לא למחוק

src/v6/pre_ai/** (זה הקיר הפלדה)

scripts/build_v6_training_pack.py, verify_v6_pipeline_e2e.py

config/runtime.yaml + policy (למשל config/v6_quality_policy.yaml)

docs/plan/PRD.md, docs/plan/SPECS.md, וה-Master Plan (כמקור־אמת)

מה כן למחוק “רק אחרי הוכחה”

כל Legacy שאינו referenced ע״י קוד/סקריפטים/טסטים קריטיים.

כלל עבודה: rg -n "<filename|symbol>" → אם 0 אז למחוק.

לגבי moment_encoder.py: אם עדיין referenced (לפי הסוכן) — לא מוחקים עד שמנתקים references בקוד/טסטים (או מסגירים אותם מאחורי flag).

11) Checklist: מה לוקחים לענן (Colab)

מינימום חובה:



 training_pack_v6_2.zip (או השם המדויק אצלך)

 scripts/ שקשורים לטעינת pack/manifest (לפחות: loader ל-manifest + NPZ reader)

 src/v6/pre_ai/ (כדי לשחזר בדיוק windowing/quality logic אם צריך בקולאב)

 config/runtime.yaml (fs/window/stride/min_window)

 config/v6_quality_policy.yaml (כדי שתיעוד המדיניות יהיה מסונכרן עם ה-pack)

 docs/plan/SPECS.md + Master Plan (כדי שהסוכן/אתה לא תסטו מהחוקים)

מומלץ:



 requirements.txt/pyproject.toml וגרסת Python (3.11)

 snapshot של git commit שה-pack נבנה עליו (ה-manifest כבר שומר commit כשאפשר)

