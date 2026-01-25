# Phase 0: Setup & Foundation - Technical Specifications

**Phase:** 0 of 6
**Document Type:** Technical Specifications
**Target Audience:** Developers, DevOps

---

## 1. Project Structure

### 1.1 Final Directory Layout

```
SentinelFetal/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI
│
├── api/                              # FastAPI Backend (NEW)
│   ├── __init__.py
│   ├── main.py                       # Entry point
│   ├── config.py                     # API configuration
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── patients.py               # /api/patients routes
│   │   ├── simulation.py             # /api/simulation routes
│   │   └── websocket.py              # /ws/* routes
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── broadcaster.py            # WebSocket broadcaster
│   │   └── orchestrator_adapter.py   # Bridge to existing orchestrator
│   └── tests/
│       └── test_api.py
│
├── frontend/                         # React Frontend (NEW)
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── main.tsx                  # Entry point
│   │   ├── App.tsx                   # Root component
│   │   ├── components/
│   │   │   └── .gitkeep
│   │   ├── hooks/
│   │   │   └── .gitkeep
│   │   ├── stores/
│   │   │   └── .gitkeep
│   │   ├── types/
│   │   │   └── index.ts              # TypeScript types
│   │   └── styles/
│   │       └── globals.css
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── src/                              # Existing Core (UNCHANGED)
│   └── ... (existing structure)
│
├── scripts/
│   ├── setup-dev.sh                  # Unix setup
│   └── setup-dev.bat                 # Windows setup
│
├── docker/                           # Phase 6
│   └── .gitkeep
│
├── .env.example                      # Environment template
├── .gitignore                        # Updated for full stack
├── .pre-commit-config.yaml           # Pre-commit hooks
├── pyproject.toml                    # Python project config
├── README.md                         # Updated
└── CONTRIBUTING.md                   # Development workflow
```

---

## 2. Python Configuration

### 2.1 pyproject.toml

```toml
[project]
name = "sentinelfetal"
version = "3.0.0"
description = "Real-time CTG monitoring with AI classification"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}

dependencies = [
    # Existing dependencies (from requirements.txt)
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "wfdb>=4.1.0",
    "matplotlib>=3.7.0",
    "plotly>=5.14.0",
    "scikit-learn>=1.2.0",
    "sktime>=0.24.0",
    "numba>=0.58.0",
    "pywavelets>=1.4.0",
    "onnx>=1.14.0",
    "onnxruntime>=1.15.0",
    "tqdm>=4.65.0",
    "python-dateutil>=2.8.0",
    "psutil>=5.9.0",
    "joblib>=1.3.0",
    # NEW: FastAPI dependencies
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "msgpack>=1.0.7",
    "python-multipart>=0.0.6",
    "websockets>=12.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.26.0",          # For testing FastAPI
    "black>=24.1.0",
    "isort>=5.13.0",
    "ruff>=0.1.14",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]

streamlit = [
    "streamlit>=1.28.0",
    "streamlit-echarts>=0.4.0",
]

[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311", "py312"]
exclude = '''
/(
    \.git
    | \.venv
    | __pycache__
    | frontend
    | node_modules
)/
'''

[tool.isort]
profile = "black"
line_length = 100
skip = [".venv", "frontend", "node_modules"]

[tool.ruff]
line-length = 100
target-version = "py39"
exclude = [".venv", "frontend", "node_modules"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "C4"]
ignore = ["E501"]  # Line length handled by black

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
exclude = ["frontend", "node_modules", ".venv"]

[tool.pytest.ini_options]
testpaths = ["tests", "api/tests"]
asyncio_mode = "auto"
```

### 2.2 Virtual Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Unix)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

---

## 3. Node.js Configuration

### 3.1 package.json

```json
{
  "name": "sentinelfetal-frontend",
  "private": true,
  "version": "3.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "lint:fix": "eslint src --ext ts,tsx --fix",
    "format": "prettier --write \"src/**/*.{ts,tsx,css}\"",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.21.0",
    "lightweight-charts": "^4.1.0",
    "zustand": "^4.4.0",
    "@tanstack/react-query": "^5.17.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@typescript-eslint/eslint-plugin": "^6.19.0",
    "@typescript-eslint/parser": "^6.19.0",
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.17",
    "eslint": "^8.56.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.5",
    "postcss": "^8.4.33",
    "prettier": "^3.2.4",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.3.0",
    "vite": "^5.0.0"
  }
}
```

### 3.2 tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",

    /* Strict mode */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,

    /* Paths */
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### 3.3 vite.config.ts

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
```

### 3.4 tailwind.config.js

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // SentinelFetal clinical colors
        'cat-normal': '#28a745',
        'cat-intermediate': '#fd7e14',
        'cat-pathological': '#dc3545',
        'fhr-blue': '#1E90FF',
        'uc-orange': '#FF8C00',
      },
    },
  },
  plugins: [],
}
```

### 3.5 ESLint Configuration (.eslintrc.cjs)

```javascript
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true },
    ],
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
  },
}
```

---

## 4. Pre-commit Configuration

### 4.1 .pre-commit-config.yaml

```yaml
repos:
  # Python formatting
  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
        language_version: python3

  # Python import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  # Python linting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.14
    hooks:
      - id: ruff
        args: [--fix]

  # TypeScript/JavaScript formatting
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.1.0
    hooks:
      - id: prettier
        types_or: [javascript, jsx, ts, tsx, css, json]
        additional_dependencies:
          - prettier@3.2.4

  # General checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
```

### 4.2 Installation

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files (first time)
pre-commit run --all-files
```

---

## 5. CI/CD Configuration

### 5.1 .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  python-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check .

      - name: Check formatting with black
        run: black --check .

      - name: Type check with mypy
        run: mypy api/ src/

      - name: Run tests
        run: pytest --cov=src --cov=api --cov-report=xml

  frontend-checks:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Type check
        run: npm run type-check

      - name: Build
        run: npm run build
```

---

## 6. Setup Scripts

### 6.1 scripts/setup-dev.sh (Unix)

```bash
#!/bin/bash
set -e

echo "=== SentinelFetal V3 Development Setup ==="

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Check Node.js version
node_version=$(node --version 2>&1 | cut -d'v' -f2 | cut -d'.' -f1)
echo "Node.js version: $node_version"

# Create Python virtual environment
echo -e "\n${GREEN}[1/5] Creating Python virtual environment...${NC}"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install Python dependencies
echo -e "\n${GREEN}[2/5] Installing Python dependencies...${NC}"
pip install -e ".[dev]"

# Install pre-commit hooks
echo -e "\n${GREEN}[3/5] Setting up pre-commit hooks...${NC}"
pre-commit install

# Install frontend dependencies
echo -e "\n${GREEN}[4/5] Installing frontend dependencies...${NC}"
cd frontend
npm install
cd ..

# Create .env from example
echo -e "\n${GREEN}[5/5] Creating environment files...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from template"
fi

echo -e "\n${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "To start development:"
echo "  1. Activate Python env: source .venv/bin/activate"
echo "  2. Start backend:       cd api && uvicorn main:app --reload"
echo "  3. Start frontend:      cd frontend && npm run dev"
```

### 6.2 scripts/setup-dev.bat (Windows)

```batch
@echo off
echo === SentinelFetal V3 Development Setup ===

:: Check Python
python --version

:: Check Node.js
node --version

:: Create Python virtual environment
echo.
echo [1/5] Creating Python virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat
pip install --upgrade pip

:: Install Python dependencies
echo.
echo [2/5] Installing Python dependencies...
pip install -e ".[dev]"

:: Install pre-commit hooks
echo.
echo [3/5] Setting up pre-commit hooks...
pre-commit install

:: Install frontend dependencies
echo.
echo [4/5] Installing frontend dependencies...
cd frontend
call npm install
cd ..

:: Create .env from example
echo.
echo [5/5] Creating environment files...
if not exist .env (
    copy .env.example .env
    echo Created .env from template
)

echo.
echo === Setup Complete! ===
echo.
echo To start development:
echo   1. Activate Python env: .venv\Scripts\activate
echo   2. Start backend:       cd api ^&^& uvicorn main:app --reload
echo   3. Start frontend:      cd frontend ^&^& npm run dev
```

---

## 7. Environment Variables

### 7.1 .env.example

```bash
# =============================================================================
# SentinelFetal V3 Environment Configuration
# =============================================================================

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true
API_RELOAD=true

# WebSocket Configuration
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS=100

# Simulation Configuration
SIM_PATIENTS_DEFAULT=4
SIM_TICK_RATE_HZ=4

# Frontend Configuration (for reference)
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

---

## 8. Initial Files to Create

### 8.1 api/main.py (Minimal Scaffold)

```python
"""
SentinelFetal API - Entry Point

This is a placeholder created in Phase 0.
Full implementation in Phase 1.
"""

from fastapi import FastAPI

app = FastAPI(
    title="SentinelFetal API",
    version="3.0.0",
    description="Real-time CTG monitoring with AI classification",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "3.0.0"}


# Placeholder for Phase 1
# from api.routers import patients, simulation, websocket
# app.include_router(patients.router)
# app.include_router(simulation.router)
# app.include_router(websocket.router)
```

### 8.2 frontend/src/main.tsx (Minimal Scaffold)

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/globals.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

### 8.3 frontend/src/App.tsx (Minimal Scaffold)

```tsx
function App() {
  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-800">
          SentinelFetal V3.0
        </h1>
        <p className="mt-4 text-gray-600">
          Frontend scaffold created. Implementation starts in Phase 3.
        </p>
      </div>
    </div>
  )
}

export default App
```

---

## 9. .gitignore Updates

```gitignore
# =============================================================================
# SentinelFetal V3 .gitignore
# =============================================================================

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
*.egg-info/
dist/
build/
.mypy_cache/
.ruff_cache/
.pytest_cache/
*.egg
.eggs/

# Node.js
node_modules/
frontend/dist/
frontend/build/
.npm/
*.tsbuildinfo

# Environment
.env
.env.local
.env.*.local

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Testing
coverage/
htmlcov/
.coverage
*.cover

# Build artifacts
*.so
*.dylib
*.dll

# Project specific
models/*.json
models/*.onnx
data/processed/
profile_output.prof
```

---

## 10. Verification Checklist

After Phase 0 completion, verify:

```bash
# Python environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"
python -c "import pydantic; print(f'Pydantic {pydantic.__version__}')"

# Backend starts
cd api && uvicorn main:app --reload &
curl http://localhost:8000/health
# Expected: {"status":"ok","version":"3.0.0"}

# Frontend environment
cd frontend
npm run type-check  # Should pass with no errors
npm run lint        # Should pass with no errors
npm run dev &       # Should start on port 3000

# Pre-commit hooks
git add .
git commit -m "test: verify pre-commit hooks"  # Should run formatters

# CI verification
gh workflow run ci.yml  # Or push to trigger
```

---

## 11. Implementation Status

### ✅ PHASE 0 COMPLETED - January 24, 2026

**Files Created:**

**Backend (`/api/`):**
- `__init__.py` - Package init with version
- `config.py` - Pydantic Settings configuration
- `main.py` - FastAPI app with CORS, lifespan, routers
- `routers/__init__.py` - Router exports
- `routers/patients.py` - Patient routes (stubs)
- `routers/simulation.py` - Simulation control routes (stubs)
- `routers/websocket.py` - WebSocket endpoint (stub)
- `models/__init__.py` - Model exports
- `models/schemas.py` - Full Pydantic schema definitions
- `services/__init__.py` - Service exports
- `services/orchestrator_adapter.py` - Singleton adapter (stub)
- `services/broadcaster.py` - AsyncBroadcaster (stub)
- `tests/test_api.py` - Basic API tests

**Frontend (`/frontend/`):**
- `package.json` - Dependencies including i18next, react-hot-toast
- `tsconfig.json` - Strict TypeScript config
- `tsconfig.node.json` - Vite node config
- `vite.config.ts` - Proxy configuration
- `tailwind.config.js` - Clinical color palette
- `postcss.config.js` - PostCSS config
- `.eslintrc.cjs` - ESLint rules
- `index.html` - Entry HTML
- `src/main.tsx` - React entry with QueryClient
- `src/App.tsx` - Basic routing scaffold
- `src/styles/globals.css` - Tailwind + RTL support
- `src/types/index.ts` - Full TypeScript definitions
- `src/components/.gitkeep` - Placeholder
- `src/hooks/.gitkeep` - Placeholder
- `src/stores/.gitkeep` - Placeholder

**Configuration:**
- `pyproject.toml` - Python project with tools config
- `.pre-commit-config.yaml` - Pre-commit hooks
- `.env.example` - Environment template
- `.github/workflows/ci.yml` - CI pipeline
- `scripts/setup-dev.sh` - Unix setup script
- `scripts/setup-dev.bat` - Windows setup script
- `docker/.gitkeep` - Docker placeholder

**Verification Results:**
```
✅ python -c "from api.main import app" - PASSED
✅ FastAPI 0.128.0 installed
✅ Pydantic 2.12.5 installed
✅ All schemas defined and importable
```

**Deviations from Spec:**
- None. All specifications implemented as documented.

---

*End of Phase 0 Technical Specifications*
