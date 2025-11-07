#!/bin/bash
# Fix virtual environment conflicts

echo "🔧 Fixing virtual environment setup..."
echo ""

# Check current environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "❌ You currently have a virtual environment activated:"
    echo "   $VIRTUAL_ENV"
    echo ""
    echo "⚠️  This conflicts with uv's project environment."
    echo ""
    echo "Please run these commands:"
    echo ""
    echo "   deactivate"
    echo "   ./fix_env.sh"
    echo ""
    exit 1
fi

echo "✅ No virtual environment currently activated"
echo ""

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "🗑️  Removing old .venv directory..."
    rm -rf .venv
fi

# Sync with uv
echo "📦 Installing dependencies with uv..."
uv sync --dev --extra ui

echo ""
echo "✅ Environment fixed!"
echo ""
echo "Now you can run:"
echo "  uv run pytest                    # Run tests"
echo "  uv run uvicorn mlsys.inference.service:app --reload  # Start API"
echo "  ./test_e2e.sh                    # Run end-to-end tests"
