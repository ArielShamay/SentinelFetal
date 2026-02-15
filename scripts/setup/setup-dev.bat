@echo off
REM SentinelFetal V3 Development Setup Script (Windows)
REM ====================================================

echo 🚀 SentinelFetal V3 Development Setup
echo ======================================

REM Check Python version
python --version

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📁 Creating Python virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
pip install --upgrade pip

REM Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -e ".[dev]"

REM Install pre-commit hooks
echo 🪝 Installing pre-commit hooks...
pre-commit install

REM Check Node.js
where node >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    node --version
) else (
    echo ⚠️  Node.js not found. Please install Node.js 18+ for frontend development.
)

REM Install frontend dependencies
if exist "frontend" (
    echo 📦 Installing frontend dependencies...
    cd frontend
    where npm >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        npm install
    ) else (
        echo ⚠️  npm not found. Skipping frontend setup.
    )
    cd ..
)

REM Copy .env.example to .env if not exists
if not exist ".env" (
    echo 📋 Creating .env from template...
    copy .env.example .env
)

echo.
echo ✅ Setup complete!
echo.
echo To start development:
echo   1. Activate venv: .venv\Scripts\activate
echo   2. Start backend: uvicorn api.main:app --reload
echo   3. Start frontend: cd frontend ^&^& npm run dev
echo.
