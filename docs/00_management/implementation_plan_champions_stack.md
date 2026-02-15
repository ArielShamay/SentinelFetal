<div dir="rtl" style="text-align: right;">

# חזון טכנולוגי: "דוקר של אלופים" (Champions Stack) 🏆

## המטרה החדשה
בניית דוקר "היברידי" המשלב את שני הכלים המהירים ביותר בתעשייה כיום: **uv** (לבקנד) ו-**Bun** (לפרונט).

---

# תוכנית עבודה מעודכנת

### שלב 0: ניקיון עמוק (Deep Cleanup) 🧹
לפני שמתחילים, נמחק הכל כדי להבטיח שאין "זבל" היסטורי:
1.  **פרונט:** מחיקת `frontend/node_modules` ו-`frontend/dist`.
2.  **דאטה:** מחיקת `processed_data_v1` (נבנה הכל מחדש בצורה נקייה).
3.  **בקנד:** מחיקת `.pytest_cache`, קבצי `__pycache__`, והקובץ הישן [requirements.txt](file:///Users/tzoharlary/Documents/Projects/Hackathon/SentinelFetal/requirements.txt).
4.  **כללי:** מחיקת קבצים זמניים (כמו `vite.config.ts.timestamp`).

### שלב 1: Backend (uv) 🐍
נבצע תהליך איטרטיבי לייצוב ה-Backend:
1.  מחיקת `.venv` הישן.
2.  הרצת `uv run src.main:app` (או הסקריפט הראשי).
3.  בכל קריסה (`ModuleNotFoundError`) -> הוספה עם `uv add`.
4.  התוצאה: קובץ [pyproject.toml](file:///Users/tzoharlary/Documents/Projects/Hackathon/SentinelFetal/pyproject.toml) מזוקק.

### שלב 2: Frontend (Bun) 🥟
נעבור לניהול מהיר עם Bun:
1.  התקנת `bun` (אם חסר).
2.  הרצת `bun install` בתיקיית [frontend](file:///Users/tzoharlary/Documents/Projects/Hackathon/SentinelFetal/Dockerfile.frontend) (במקום npm).
3.  הרצת `bun run build`.
4.  וידוא שהבנייה עוברת בהצלחה.

### שלב 3: Dockerfile משודרג 🐳
בניית Dockerfile חדש ומודרני:
*   `FROM oven/bun` -> לבניית ה-Frontend.
*   `FROM ghcr.io/astral-sh/uv` -> להריץ את ה-Python.
*   חיבור התוצרים ל-Image סופי קטן ויעיל.

</div>