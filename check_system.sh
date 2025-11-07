#!/bin/bash
# Quick system status check

echo "🔍 ML-Sys System Status Check"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check virtual environment
echo "1. Virtual Environment:"
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "   ${RED}✗ Active venv detected: $VIRTUAL_ENV${NC}"
    echo -e "   ${YELLOW}⚠  This will cause issues with uv${NC}"
    echo -e "   ${YELLOW}   Run: deactivate${NC}"
else
    echo -e "   ${GREEN}✓ No conflicting venv${NC}"
fi

# Check uv
echo ""
echo "2. UV Package Manager:"
if command -v uv > /dev/null 2>&1; then
    UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
    echo -e "   ${GREEN}✓ uv installed: $UV_VERSION${NC}"
else
    echo -e "   ${RED}✗ uv not found${NC}"
    echo -e "   ${YELLOW}   Install: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
fi

# Check Python
echo ""
echo "3. Python Environment:"
PYTHON_PATH=$(uv run which python 2>/dev/null)
if [ $? -eq 0 ]; then
    PYTHON_VERSION=$(uv run python --version 2>&1)
    echo -e "   ${GREEN}✓ Python: $PYTHON_VERSION${NC}"
    echo -e "   ${GREEN}✓ Location: $PYTHON_PATH${NC}"
else
    echo -e "   ${RED}✗ Python not accessible via uv${NC}"
fi

# Check dependencies
echo ""
echo "4. Key Dependencies:"
for pkg in pandas scikit-learn xgboost fastapi uvicorn pytest; do
    if uv run python -c "import $pkg" 2>/dev/null; then
        VERSION=$(uv run python -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null)
        echo -e "   ${GREEN}✓ $pkg ($VERSION)${NC}"
    else
        echo -e "   ${RED}✗ $pkg not installed${NC}"
    fi
done

# Check optional dependencies
echo ""
echo "5. Optional Dependencies:"
if uv run python -c "import prometheus_client" 2>/dev/null; then
    echo -e "   ${GREEN}✓ prometheus-client${NC}"
else
    echo -e "   ${YELLOW}⊘ prometheus-client (optional)${NC}"
fi

if uv run python -c "import mlflow" 2>/dev/null; then
    echo -e "   ${GREEN}✓ mlflow${NC}"
else
    echo -e "   ${YELLOW}⊘ mlflow (optional)${NC}"
fi

# Check model file
echo ""
echo "6. Model Artifacts:"
if [ -f "artifacts/xgb_lead_scoring.joblib" ]; then
    SIZE=$(du -h artifacts/xgb_lead_scoring.joblib | cut -f1)
    echo -e "   ${GREEN}✓ Model exists ($SIZE)${NC}"
else
    echo -e "   ${YELLOW}⊘ No trained model found${NC}"
    echo -e "   ${YELLOW}   Train: uv run python scripts/train.py --config config/base_config.yaml${NC}"
fi

# Check Docker
echo ""
echo "7. Docker:"
if command -v docker > /dev/null 2>&1; then
    if docker info > /dev/null 2>&1; then
        echo -e "   ${GREEN}✓ Docker running${NC}"
    else
        echo -e "   ${YELLOW}⊘ Docker installed but not running${NC}"
    fi
else
    echo -e "   ${YELLOW}⊘ Docker not installed (optional)${NC}"
fi

# Check ports
echo ""
echo "8. Port Availability:"
for port in 8000 5000 9090 3000; do
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "   ${YELLOW}⊘ Port $port in use${NC}"
    else
        echo -e "   ${GREEN}✓ Port $port available${NC}"
    fi
done

# Summary
echo ""
echo "=============================="
echo "📋 Summary & Next Steps"
echo "=============================="
echo ""

if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${RED}❌ Issues detected${NC}"
    echo ""
    echo "Fix by running:"
    echo "  1. deactivate"
    echo "  2. ./fix_env.sh"
    echo "  3. ./test_e2e.sh"
elif ! command -v uv > /dev/null 2>&1; then
    echo -e "${RED}❌ UV not installed${NC}"
    echo ""
    echo "Install uv:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
else
    echo -e "${GREEN}✅ System looks good!${NC}"
    echo ""
    echo "Run end-to-end tests:"
    echo "  ./test_e2e.sh"
    echo ""
    echo "Or start using the system:"
    echo "  uv run pytest                           # Run tests"
    echo "  uv run uvicorn mlsys.inference.service:app --reload  # Start API"
    echo "  docker-compose up -d                    # Full stack"
fi

echo ""
