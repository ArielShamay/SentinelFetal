<div dir="rtl" style="text-align: right;">

# מסמך מאוחד: כל בדיקות העומס/Stress/Endurance בפרויקט SentinelFetal

מסמך זה מרכז **את כל בדיקות העומס שמצאתי בריפו** (סקריפטים + בנצ'מרקים + דוחות תוצאות), ומסביר:
- איזה בדיקות קיימות
- איפה הקוד שמריץ אותן
- איזה קבצי פלט/דוחות הן יוצרות
- מה בדיוק נמדד בכל בדיקה
- מה יצא בפועל לפי הדוחות שנשמרו בפרויקט

> הערה חשובה: חלק מהדוחות עצמם כוללים שדה "Failure reason" שהוא לפעמים תוצאה של עצירה ידנית/חריגה, ולכן במסמך הזה אני מפריד בין **תוצאה שנובעת ממדידה תקינה** לבין **תוצאה שעלולה להיות ארטיפקט (למשל pause של המחשב/עצירה ידנית)**.

---

## 1) אינדקס בדיקות עומס שמצאתי (Inventory)

| # | שם הבדיקה | קוד שמריץ | קובץ תוצאות/דוח | סוג עומס | MOMENT | הערה מרכזית |
|---|-----------|-----------|------------------|----------|--------|------------|
| 1 | Full System Stress (כולל UI) | `scripts/full_system_stress.py` | `docs/reports/FULL_SYSTEM_STRESS_REPORT.md` | UI Rendering + Serialization | לא (חיקוי Rules) | Plotly הוא צוואר הבקבוק (~95%) |
| 2 | Stress Test Suite (Phase 4) | `scripts/stress_test_suite.py` | `docs/reports/STRESS_TEST_RESULTS.md` | מערכת מלאה (Orchestrator + Pipeline) | כן (PyTorch) | עבר (PASS) עם 3 מטופלות |
| 3 | Endurance Benchmark (Phase 10) | `tests/benchmarks/benchmark_endurance.py` | `docs/reports/ENDURANCE_TEST_REPORT.md` + `tests/benchmarks/results/endurance_log.csv` | יציבות ארוכת טווח (1 שעה) | לא (Mock) | PASS לשעה עם 8 מטופלות + 12 injections |
| 4 | Load Benchmark (Phased 1/2/4/8) | `tests/benchmarks/benchmark_load.py` | `tests/benchmarks/results/load_results.json` | capacity בסיסי (tick loop) | לא (Mock) | tick_rate ~0.984Hz עד 8 |
| 5 | Hourly Offline Simulation (Phase 9) | `tests/benchmarks/benchmark_hourly_simulation.py` | `docs/reports/hourly_simulation_summary.md` + `tests/benchmarks/results/hourly_simulation_results.json` | סימולציה לוגית (איכות/override תחת עומס זמן) | לא (Mock) | סינוסואידלי לא זוהה בגלל חלון קצר |
| 6 | Breaking Point Finder (Synthetic backend) | `scripts/find_breaking_point.py` | `docs/reports/LIMIT_TEST_RESULTS.md` | “נקודת שבירה” (Latency/CPU/Accuracy) | לא (מודל חיקוי) | ריצה אחרונה מראה ארטיפקט חריג מאוד ב-latency max |
| 7 | Performance Benchmark כללי | `benchmark_performance.py` | פלט למסך בלבד | מיקרו-בנצ'מרק לרכיבים | לפעמים כן | שימושי לכיוונון, לא “Stress test” מלא |

---

## 2) פירוט מלא לכל בדיקה

### 2.1 Full System Stress Test (כולל UI Rendering)

**קוד:** `scripts/full_system_stress.py`

**מה הבדיקה מדמה?**
לולאת Streamlit מלאה בלי Streamlit בפועל:
1) יצירת נתונים סינתטיים (FHR)
2) “Rules Engine” קל (proxy)
3) **Plotly**: יצירת Figure + `fig.to_json()` (מדמה serialization של Streamlit)

**מה נמדד?**
- זמן מחזור כולל (ms)
- חלוקת זמן לפי רכיבים: Data gen / rules / plotting
- CPU/RAM
- יציבות generator (NaN/flatline)

**תנאי עצירה בקוד:**
- `total_cycle_ms > 1000` → FPS Drop
- `nan_count > 10` / `flatline_count > 50` → כשל generator
- `KeyboardInterrupt` → User interrupted

**מה יצא בפועל (מהדוח `docs/reports/FULL_SYSTEM_STRESS_REPORT.md`):**
- **UI Rendering (Plotly)** = **128.4ms** בממוצע = **94.8%** מהמחזור
- total avg = **135.5ms**
- max cycle = **260ms**
- CPU max = **40.1%**, RAM max = **56MB**
- NaN=0, Flatline=0

**מסקנה פרקטית:**
העומס המרכזי הוא **Plotly + JSON serialization**, לא חישוב/כללים. הדוח מציין “Max Concurrent Patients = 1” אבל גם מציין “User interrupted” ולכן זה **לא** בהכרח הגבול האמיתי — פשוט זו נקודת המדידה שבה נעצר.

---

### 2.2 Stress Test Suite (Phase 4 – Capacity & Stability)

**קוד:** `scripts/stress_test_suite.py`

**מה הבדיקה עושה?**
- מריצה `SimulationOrchestrator` עם `PipelineAdapter`
- מפעילה `run_moment=True` בתוך callback (כלומר **MOMENT אמיתי**)
- אוספת metrics כל שניה: CPU/RAM/latency/drift

**קונפיג חשוב (ברירת מחדל בקוד):**
- מטופלות: 3
- sampling: 4Hz
- tick interval: 0.25s
- moment interval: 30s
- ספים:
  - CPU < 500%
  - RAM < 4096MB
  - latency < 1500ms
  - drift < 3s

**תוצאה בפועל מהדוח `docs/reports/STRESS_TEST_RESULTS.md`:**
- Overall: ✅ **PASSED**
- CPU avg/max/p95: **25.5% / 398.4% / 325.3%** (threshold 500%)
- RAM avg/max: **1750MB / 1782MB** (threshold 4096MB)
- latency avg/max/p95: **155ms / 624ms / 624ms** (threshold 1500ms)
- drift avg/max: **0.08s / 1.06s** (threshold 3s)
- Total ticks: 356
- Total MOMENT calls: 8

**תובנה מרכזית:**
הפעלת MOMENT אמיתי גורמת ל-RAM להיות גבוה (בסקאלה של ~1.7GB) אבל עדיין בתוך הסף. זה תואם גם לבנצ'מרק MOMENT (סעיף 2.7).

**ONNX:**
הדוח מציין שייצוא ONNX נכשל בגלל `Unfold` עם dynamic shapes → fallback ל-PyTorch.

---

### 2.3 Endurance Benchmark (Phase 10 – 1 hour stability)

**קוד:** `tests/benchmarks/benchmark_endurance.py`

**מה הבדיקה עושה?**
- 8 מטופלות
- tick סינכרוני ב-1Hz (כל tick מפעיל `orchestrator._tick()` ישירות)
- כל 5 דקות הזרקת אירוע פתולוגי אקראי (sinusoidal או late)
- לוג כל דקה ל-CSV
- בסוף מוסיף שורה/סיכום לדוח Markdown

**קבצי פלט:**
- `tests/benchmarks/results/endurance_log.csv`
- `docs/reports/ENDURANCE_TEST_REPORT.md`

**תוצאות מרכזיות מהדוח `docs/reports/ENDURANCE_TEST_REPORT.md`:**
- Status: **PASS** (שעה מלאה)
- Max tick latency: **162.61ms** (threshold 1500ms)
- Max RAM: **367.74MB**
- Memory growth: **1.89MB/hr** (pass, <50MB/hr)
- Injections: **12/12** תקין
- “Memory leak: NONE DETECTED”

**תובנה מרכזית:**
המערכת יציבה לאורך זמן, ובפרט אין דליפת זיכרון משמעותית. ההזרקות בזמן אמת עובדות ועוברות אימות.

---

### 2.4 Load Benchmark (Phased 1/2/4/8)

**קוד:** `tests/benchmarks/benchmark_load.py`

**מה הבדיקה עושה?**
- מריצה 4 שלבים: 1, 2, 4, 8 מטופלות
- 60 שניות לכל שלב
- tick interval = 1s
- callback קל (חיקוי עבודה) ללא MOMENT אמיתי

**קובץ תוצאות:** `tests/benchmarks/results/load_results.json`

**תוצאות מהקובץ `load_results.json`:**
- tick_rate_hz ≈ **0.9839** כמעט בכל השלבים
- worst_tick_sec ≈ **1.0163s**
- lag מוגדר כ-`worst_tick_delay > 1.02` ולכן סווג כ-**false** בכל השלבים
- CPU mean עולה מ~19% (1–4) ל~24% ב-8 מטופלות

**תובנה מרכזית:**
לולאת הטיקים עצמה “בקצה” של 1Hz (יש jitter ~16ms), אבל תחת הקריטריון (1.02s) זה עדיין נחשב תקין עד 8 מטופלות (במצב Mock processing).

---

### 2.5 Hourly Offline Simulation (Phase 9)

**קוד:** `tests/benchmarks/benchmark_hourly_simulation.py`

**מה הבדיקה עושה?**
- סימולציה של שעה (3600s) ב-4Hz
- 4 מטופלות עם לוח אירועים מתוזמן:
  - control
  - sinusoidal (10 דקות)
  - late decels (פעמיים)
  - variable decel + bradycardia
- ניתוח כל 2 דקות על חלון 10 דקות

**פלט:**
- JSON מפורט: `tests/benchmarks/results/hourly_simulation_results.json`
- סיכום אנושי: `docs/reports/hourly_simulation_summary.md`

**ממצאים מהסיכום:**
- P1_control: Cat1 בכל 26/26
- P2_sinusoidal: Cat1 בכל 26/26 → **לא זוהה** (דרישת detector ≥20 דקות)
- P3_late: Cat2 ב-7 חלונות, מתאים ל-overrides של recurrent late
- P4_variable_brady: Cat2 ב-3 חלונות בעקבות bradycardia; variable decel לא הפך קטגוריה

**תובנה מרכזית:**
הבדיקה הזו פחות “עומס מחשובי” ויותר “עומס זמן/תרחישים” שמאמת את ה-overrides והלוגיקה בתנאים ריאליסטיים של התקדמות לאורך שעה.

---

### 2.6 Breaking Point Finder (Synthetic backend – Progressive load)

**קוד:** `scripts/find_breaking_point.py`

**מה הבדיקה עושה?**
- backend סינתטי בלבד (ללא UI)
- מתחיל ב-5 מטופלות ומוסיף +5 כל 60 שניות
- לכל מטופלת מייצר הרבה רעש/ארטיפקטים, ומזריק late decels ל-20% מהמטופלות
- מודד latency לכל “batch”, CPU, RAM, ודיוק של detector heuristic (TP/FP/FN) ביחס לאירועים שהוזרקו

**תנאי עצירה בקוד:**
- latency > 2000ms
- CPU > 95% רציף 30 שניות
- accuracy < 70% אחרי לפחות 100 בדיקות אירוע
- MemoryError

**תוצאה מהדוח `docs/reports/LIMIT_TEST_RESULTS.md` (ריצה אחרונה שנמצאת בפרויקט כרגע):**
- Failure point: **150**
- Failure mode: **Latency exceeded 2000ms (observed 764672ms)**
- Safe operational limit (80%): **120**
- Avg latency: **43ms**
- Max latency: **764672ms** (כמעט 12.7 דקות)
- CPU avg/max: **43.0% / 312.5%**
- RAM max: **53MB**
- Accuracy final/min: **0.96 / 0.00**
- TP/FP/FN: **76165/421334/2973**

**הערת איכות נתונים (חשוב):**
הפער בין “Avg latency = 43ms” לבין “Max latency = 764672ms” הוא קיצוני מאוד. זה בדרך כלל מעיד על:
- pause/hibernate של המחשב
- עצירה של process (breakpoint / debugger / מעבר חלון + throttling)
- או אירוע מערכת חריג שתקע את הלולאה רגעית

כלומר: יש כאן **מדד max חריג** שכדאי להתייחס אליו בזהירות לפני שמסיקים “נקודת שבירה אמיתית”. מצד שני, עצם ההגעה ל-150 מטופלות עם RAM ~53MB ו-CPU avg ~43% מראה שהסימולטור עצמו מאוד קל – מה שמחזק את ההשערה שה-max הוא ארטיפקט.

---

### 2.7 MOMENT Performance Benchmark (תומך בכל בדיקות העומס)

זה לא stress test בפני עצמו, אבל הוא קריטי להבנת הסקייל: כמה מהר אפשר להריץ MOMENT ב-CPU.

**קובץ תוצאות:** `tests/benchmarks/results/moment_results.json`

**תוצאות:**
- load_time ≈ **11.1s**
- RAM delta ≈ **+1335MB**
- inference per 10-min window:
  - mean ≈ **1869ms**
  - median ≈ **1798ms**
  - p95 ≈ **2392ms**
  - max ≈ **2720ms**
- throughput ≈ **0.535 windows/sec**

**תובנה מרכזית:**
ב-CPU בלבד, MOMENT הוא “הפיל שבחדר” מבחינת זמן + זיכרון. זה מסביר למה בדיקות שמשתמשות ב-real MOMENT מוגבלות יחסית, ולמה חלק מהבנצ'מרקים משתמשים ב-Mock.

---

## 3) תמונת מצב מאוחדת (מה באמת מגביל סקייל?)

### מגבלות לפי שכבות
- **UI/Streamlit/Plotly:** צוואר הבקבוק הראשי במצב “Full System UI” (serialization של Plotly)
- **MOMENT אמיתי ב-CPU:** מגביל throughput ו-RAM (≈1.3GB תוספת רק בטעינה)
- **Orchestrator + Mock:** מסוגל ל-1–8 מטופלות ב-1Hz עם jitter קטן
- **Endurance:** יציבות גבוהה לאורך שעה, ללא דליפת זיכרון

### הפער בין “Mock” ל-“Real”
בחלק מהבדיקות (load/endurance/hourly) MOMENT הוא Mock כדי לבודד יציבות/לוגיקה. ב-stress_test_suite הוא Real, ולכן זו אינדיקציה חשובה יותר לקצה היכולת של pipeline מלא.

---

## 4) איך להריץ את כל הבדיקות (רשימת פקודות)

מהשורש של הפרויקט:

```powershell
# 1) UI + Plotly full-system stress
.\.venv\Scripts\python.exe scripts\full_system_stress.py

# 2) Stress Test Suite (REAL MOMENT)
.\.venv\Scripts\python.exe scripts\stress_test_suite.py --patients 3 --duration 300

# 3) Endurance (ברירת מחדל 1 שעה)
.\.venv\Scripts\python.exe tests\benchmarks\benchmark_endurance.py --duration 3600

# 4) Load benchmark (1/2/4/8)
.\.venv\Scripts\python.exe tests\benchmarks\benchmark_load.py

# 5) Hourly offline simulation
.\.venv\Scripts\python.exe tests\benchmarks\benchmark_hourly_simulation.py

# 6) Breaking point finder (עשוי לקחת זמן)
.\.venv\Scripts\python.exe scripts\find_breaking_point.py
```

---

## 5) המלצות “מה לעשות עכשיו” (אם המטרה היא להגיע למספרים קליניים אמיתיים)

1) **להחליט מה יעד הסקייל**: UI (מסכים), או Engine (כמה מטופלות), או Endurance (כמה זמן).
2) **למדוד Full system עם Real MOMENT + UI**: כרגע אין בדיקה אחת שמחברת גם UI וגם real MOMENT באופן ישיר.
3) **לתקן/לאמת את LIMIT_TEST_RESULTS**:
   - להריץ שוב ולוודא שהמחשב לא נכנס לשינה
   - לשקול לכתוב לוג פר אבן דרך (כל דקה) כדי לאבחן spikes
4) **UI optimization ל-Plotly**: Scattergl/caching/עדכון data בלבד.

---

## נספח: איפה עוד יש אזכורים לבדיקות עומס

- `docs/reports/COMPREHENSIVE_EVALUATION_REPORT.md` מרכז גם דיוק וגם בנצ'מרקים של load + MOMENT.
- `README.md` מצביע על חלק מהדוחות תחת “reports”.

</div>
