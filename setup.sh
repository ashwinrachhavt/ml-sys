#!/bin/bash
# Quick setup script for ml-sys

set -e

echo "🚀 Setting up ml-sys..."

# Deactivate any active venv
if [ -n "$VIRTUAL_ENV" ]; then
    echo "⚠️  Deactivating current virtual environment: $VIRTUAL_ENV"
    deactivate 2>/dev/null || true
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed"
else
    echo "✅ uv already installed"
fi

# Sync dependencies
echo "📦 Installing dependencies..."
uv sync --dev --extra ui

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
uv run pre-commit install

# Check if model exists
if [ ! -f "artifacts/xgb_lead_scoring.joblib" ]; then
    echo "🤖 Training initial model..."
    mkdir -p artifacts
    uv run python scripts/train.py --config config/base_config.yaml --output artifacts/xgb_lead_scoring.joblib
    echo "✅ Model trained"
else
    echo "✅ Model already exists"
fi

# Run tests
echo "🧪 Running tests..."
if uv run pytest -q; then
    echo "✅ All tests passed"
else
    echo "⚠️  Some tests failed (this may be expected)"
fi

echo ""
echo "✨ Setup complete! ✨"
echo ""
echo "Quick start commands:"
echo "  uv run pytest                           # Run tests"
echo "  uv run uvicorn mlsys.inference.service:app --reload  # Start API"
echo "  uv run python scripts/train.py --config config/base_config.yaml    # Train model"
echo "  docker-compose up -d                    # Start full stack"
echo ""
echo "Documentation:"
echo "  README.md           - Full documentation"
echo "  QUICKSTART.md       - 5-minute quick start"
echo "  SETUP.md            - Detailed setup guide"
echo "  TROUBLESHOOTING.md  - Common issues and solutions"
echo ""
echo "Visit http://localhost:8000/docs after starting the API!"
