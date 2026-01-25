#!/bin/bash
# SentinelFetal V3 Development Setup Script (Unix/macOS/Linux)
# ============================================================

set -e

echo "🚀 SentinelFetal V3 Development Setup"
echo "======================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📦 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📁 Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "📦 Node.js version: $NODE_VERSION"
else
    echo "⚠️  Node.js not found. Please install Node.js 18+ for frontend development."
fi

# Install frontend dependencies
if [ -d "frontend" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    if command -v npm &> /dev/null; then
        npm install
    else
        echo "⚠️  npm not found. Skipping frontend setup."
    fi
    cd ..
fi

# Copy .env.example to .env if not exists
if [ ! -f ".env" ]; then
    echo "📋 Creating .env from template..."
    cp .env.example .env
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start development:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Start backend: uvicorn api.main:app --reload"
echo "  3. Start frontend: cd frontend && npm run dev"
echo ""
