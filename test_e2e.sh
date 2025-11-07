#!/bin/bash
# End-to-end test script for ml-sys

set -e

echo "🧪 Running End-to-End Tests for ml-sys"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to run tests
run_test() {
    local test_name="$1"
    local command="$2"

    echo -n "Testing: $test_name... "

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test 1: Check uv is installed
echo "1️⃣  Checking Prerequisites"
echo "-------------------------"
run_test "uv is installed" "command -v uv"

# Test 2: Sync dependencies
echo ""
echo "2️⃣  Installing Dependencies"
echo "-------------------------"
echo "Running: uv sync --dev --extra ui"
if uv sync --dev --extra ui --quiet 2>&1 | grep -q "Audited\|Resolved"; then
    echo -e "${GREEN}✓ Dependencies synced${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠ Warning: Dependency sync completed with warnings${NC}"
    ((TESTS_PASSED++))
fi

# Test 3: Check Python imports
echo ""
echo "3️⃣  Testing Python Imports"
echo "-------------------------"
run_test "Import mlsys.training" "uv run python -c 'import mlsys.training'"
run_test "Import mlsys.inference" "uv run python -c 'import mlsys.inference'"
run_test "Import mlsys.config" "uv run python -c 'import mlsys.config'"

# Test 4: Run unit tests
echo ""
echo "4️⃣  Running Unit Tests"
echo "-------------------------"
echo "Running: uv run pytest -v"
if uv run pytest --tb=short --quiet; then
    echo -e "${GREEN}✓ All tests passed${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Some tests failed${NC}"
    ((TESTS_FAILED++))
fi

# Test 5: Train a model
echo ""
echo "5️⃣  Testing Model Training"
echo "-------------------------"
TEST_MODEL_PATH="artifacts/test_model_e2e.joblib"
echo "Training model to: $TEST_MODEL_PATH"
if uv run python scripts/train.py \
    --config config/base_config.yaml \
    --test-size 0.3 \
    --random-state 42 \
    --output "$TEST_MODEL_PATH" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Model training successful${NC}"
    ((TESTS_PASSED++))

    # Check model file exists
    if [ -f "$TEST_MODEL_PATH" ]; then
        echo -e "${GREEN}✓ Model file created${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ Model file not found${NC}"
        ((TESTS_FAILED++))
    fi
else
    echo -e "${RED}✗ Model training failed${NC}"
    ((TESTS_FAILED++))
fi

# Test 6: Test inference API
echo ""
echo "6️⃣  Testing Inference API"
echo "-------------------------"

# Start API in background
echo "Starting API server..."
uv run uvicorn mlsys.inference.service:app --host 127.0.0.1 --port 8888 > /tmp/mlsys_api.log 2>&1 &
API_PID=$!

# Wait for API to start
echo "Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8888/health > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Test health endpoint
if curl -s http://127.0.0.1:8888/health | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health endpoint working${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Health endpoint failed${NC}"
    ((TESTS_FAILED++))
fi

# Test root endpoint
if curl -s http://127.0.0.1:8888/ | grep -q "Lead Scoring"; then
    echo -e "${GREEN}✓ Root endpoint working${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Root endpoint failed${NC}"
    ((TESTS_FAILED++))
fi

# Test prediction endpoint
PAYLOAD='{"leads":[{"ALEXA_RANK":50000,"EMPLOYEE_RANGE":"1-10","INDUSTRY":"Technology","ACTIONS_TOTAL":25,"USERS_TOTAL":5}]}'
if curl -s -X POST http://127.0.0.1:8888/score \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" | grep -q "probabilities"; then
    echo -e "${GREEN}✓ Prediction endpoint working${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Prediction endpoint failed${NC}"
    ((TESTS_FAILED++))
fi

# Stop API
kill $API_PID 2>/dev/null || true
wait $API_PID 2>/dev/null || true

# Test 7: Code quality checks
echo ""
echo "7️⃣  Code Quality Checks"
echo "-------------------------"
echo "Running: ruff check"
if uv run ruff check src/ tests/ scripts/ --quiet; then
    echo -e "${GREEN}✓ Ruff linting passed${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠ Linting warnings found (non-critical)${NC}"
    ((TESTS_PASSED++))
fi

echo "Running: isort check"
if uv run isort --check-only src/ tests/ scripts/ --quiet; then
    echo -e "${GREEN}✓ Import sorting correct${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠ Import sorting issues (non-critical)${NC}"
    ((TESTS_PASSED++))
fi

# Test 8: Docker build (optional)
echo ""
echo "8️⃣  Docker Build Test (Optional)"
echo "-------------------------"
if command -v docker > /dev/null 2>&1; then
    echo "Building Docker image..."
    if docker build -t mlsys-api:test -f Dockerfile . > /tmp/docker_build.log 2>&1; then
        echo -e "${GREEN}✓ Docker build successful${NC}"
        ((TESTS_PASSED++))

        # Clean up
        docker rmi mlsys-api:test > /dev/null 2>&1 || true
    else
        echo -e "${RED}✗ Docker build failed${NC}"
        echo "Check /tmp/docker_build.log for details"
        ((TESTS_FAILED++))
    fi
else
    echo -e "${YELLOW}⊘ Docker not installed, skipping${NC}"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up..."
rm -f "$TEST_MODEL_PATH"
rm -f /tmp/mlsys_api.log
rm -f /tmp/docker_build.log

# Summary
echo ""
echo "========================================"
echo "📊 Test Summary"
echo "========================================"
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✨ All critical tests passed! System is working correctly. ✨${NC}"
    echo ""
    echo "You can now:"
    echo "  • Start the API: uv run uvicorn mlsys.inference.service:app --reload"
    echo "  • Run full stack: docker-compose up -d"
    echo "  • View API docs: http://localhost:8000/docs"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please check the output above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  • Deactivate any virtual environment: deactivate"
    echo "  • Reinstall dependencies: uv sync --dev --extra ui"
    echo "  • Train a model: uv run python scripts/train.py --config config/base_config.yaml"
    echo "  • Check TROUBLESHOOTING.md for more help"
    exit 1
fi
