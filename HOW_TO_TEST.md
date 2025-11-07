# How to Test the System End-to-End

## Quick Fix (If you see errors)

You have a global virtual environment active. Here's how to fix it:

```bash
# Step 1: Deactivate the global venv
deactivate

# Step 2: Run the fix script
./fix_env.sh

# Step 3: Run end-to-end tests
./test_e2e.sh
```

## Complete Testing Guide

### 1. Check System Status

First, see what's working and what needs fixing:

```bash
./check_system.sh
```

This will show you:
- ✅ What's working correctly
- ⚠️  What needs attention
- ❌ What's broken and how to fix it

### 2. Fix Any Issues

If the status check shows problems:

```bash
# If you have a venv active
deactivate

# Run the fix script
./fix_env.sh
```

### 3. Run End-to-End Tests

This comprehensive test will verify everything works:

```bash
./test_e2e.sh
```

The script tests:
1. ✅ Prerequisites (uv installation)
2. ✅ Dependencies installation
3. ✅ Python imports
4. ✅ Unit tests
5. ✅ Model training
6. ✅ API endpoints (health, root, predictions)
7. ✅ Code quality (linting, formatting)
8. ✅ Docker build (optional)

## Manual Testing Steps

If you prefer to test manually:

### A. Test Dependencies

```bash
uv sync --dev --extra ui
uv run python -c "import mlsys; print('✓ Imports working')"
```

### B. Test Training

```bash
uv run python scripts/train.py \
  --config config/base_config.yaml \
  --test-size 0.3 \
  --output artifacts/test_model.joblib

# Verify model was created
ls -lh artifacts/test_model.joblib
```

### C. Test Unit Tests

```bash
uv run pytest -v
```

### D. Test API

Start the API:
```bash
uv run uvicorn mlsys.inference.service:app --reload
```

In another terminal, test endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Root endpoint
curl http://localhost:8000/

# Prediction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "leads": [{
      "ALEXA_RANK": 50000,
      "EMPLOYEE_RANGE": "1-10",
      "INDUSTRY": "Technology",
      "ACTIONS_TOTAL": 25,
      "USERS_TOTAL": 5
    }]
  }'
```

Or visit http://localhost:8000/docs for interactive testing.

### E. Test Docker

```bash
# Build image
docker build -t mlsys-api:test .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts:ro \
  mlsys-api:test

# Test in another terminal
curl http://localhost:8000/health
```

### F. Test Full Stack

```bash
# Start all services
docker-compose up -d

# Check services
docker-compose ps

# Test API
curl http://localhost:8000/health

# Open services in browser
open http://localhost:8000/docs    # API docs
open http://localhost:5000         # MLflow
open http://localhost:9090         # Prometheus
open http://localhost:3000         # Grafana (admin/admin)

# Stop services
docker-compose down
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'prometheus_client'"

**Fix:**
```bash
deactivate  # if in venv
uv sync --dev --extra ui
```

### Issue: "VIRTUAL_ENV does not match"

**Fix:**
```bash
deactivate
./fix_env.sh
```

### Issue: "Model not found"

**Fix:**
```bash
uv run python scripts/train.py --config config/base_config.yaml
```

### Issue: Tests fail

**Fix:**
```bash
# Full reset
deactivate  # if in venv
rm -rf .venv
uv sync --dev --extra ui
uv run pytest
```

## Success Criteria

Your system is working correctly when:

1. ✅ `./check_system.sh` shows all green checkmarks
2. ✅ `./test_e2e.sh` reports "All critical tests passed"
3. ✅ `uv run pytest` completes without errors
4. ✅ API responds at http://localhost:8000/health
5. ✅ You can make predictions via the API
6. ✅ Docker build succeeds

## Quick Reference

```bash
# Check status
./check_system.sh

# Fix environment
deactivate && ./fix_env.sh

# Run all tests
./test_e2e.sh

# Run unit tests only
uv run pytest

# Train model
uv run python scripts/train.py --config config/base_config.yaml

# Start API
uv run uvicorn mlsys.inference.service:app --reload

# Full stack
docker-compose up -d

# Code quality
uv run ruff check --fix src/
uv run isort src/
```

## What Each Script Does

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `check_system.sh` | Check what's working | First step, diagnostics |
| `fix_env.sh` | Fix environment issues | When venv conflicts exist |
| `setup.sh` | Complete setup from scratch | Initial setup |
| `test_e2e.sh` | Run all tests end-to-end | Verify everything works |
| `run.sh` | Run commands with uv | Quick command wrapper |

## Expected Results

### Successful Test Output

```
🧪 Running End-to-End Tests for ml-sys
========================================

1️⃣  Checking Prerequisites
-------------------------
Testing: uv is installed... ✓ PASS

2️⃣  Installing Dependencies
-------------------------
✓ Dependencies synced

3️⃣  Testing Python Imports
-------------------------
Testing: Import mlsys.training... ✓ PASS
Testing: Import mlsys.inference... ✓ PASS
Testing: Import mlsys.config... ✓ PASS

4️⃣  Running Unit Tests
-------------------------
✓ All tests passed

5️⃣  Testing Model Training
-------------------------
✓ Model training successful
✓ Model file created

6️⃣  Testing Inference API
-------------------------
✓ Health endpoint working
✓ Root endpoint working
✓ Prediction endpoint working

7️⃣  Code Quality Checks
-------------------------
✓ Ruff linting passed
✓ Import sorting correct

8️⃣  Docker Build Test (Optional)
-------------------------
✓ Docker build successful

========================================
📊 Test Summary
========================================
Tests Passed: 15+
Tests Failed: 0

✨ All critical tests passed! System is working correctly. ✨
```

## Next Steps After Testing

Once all tests pass:

1. **Development**: Start API and make changes
   ```bash
   uv run uvicorn mlsys.inference.service:app --reload
   ```

2. **Training**: Experiment with models
   ```bash
   uv run python scripts/train.py --config config/base_config.yaml --tune
   ```

3. **Production**: Deploy with Docker
   ```bash
   docker-compose up -d
   ```

4. **Monitoring**: View dashboards
   - API: http://localhost:8000/docs
   - MLflow: http://localhost:5000
   - Grafana: http://localhost:3000

## Getting Help

If tests still fail after following this guide:

1. Check `TROUBLESHOOTING.md` for common issues
2. Review error messages carefully
3. Run `./check_system.sh` for diagnostics
4. Ensure you've deactivated any global venv
5. Try a fresh install with `./fix_env.sh`

## CI/CD Testing

The same tests run in GitHub Actions:

- Every push: Linting, tests, Docker build
- On main: Deploy to staging
- On tags: Deploy to production
- Weekly: Model retraining

See `.github/workflows/` for CI/CD definitions.
