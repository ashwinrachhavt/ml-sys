# Troubleshooting Guide

## Issue: Wrong Virtual Environment

### Problem
```
warning: `VIRTUAL_ENV=/Users/ashwin/Applications/Master/venv` does not match the project environment path `.venv`
```

### Solution

You have a global virtual environment activated. You need to either:

**Option 1: Deactivate and use uv (Recommended)**
```bash
# Deactivate current venv
deactivate

# Use uv to run commands (uv manages its own environment)
uv run pytest
uv run uvicorn mlsys.inference.service:app --reload
```

**Option 2: Use the project's .venv**
```bash
# Deactivate current venv
deactivate

# Activate project venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run commands normally
pytest
uvicorn mlsys.inference.service:app --reload
```

**Option 3: Use the helper script**
```bash
# This automatically uses uv
./run.sh pytest
./run.sh uvicorn mlsys.inference.service:app --reload
```

## Issue: ModuleNotFoundError: No module named 'prometheus_client'

### Problem
Dependencies not installed in the active environment.

### Solution

```bash
# Make sure you're using uv or the correct venv
deactivate  # if in wrong venv

# Let uv sync dependencies
uv sync --dev --extra ui

# Or install manually in .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"
```

## Issue: FastAPI factory error

### Problem
```
ERROR: Error loading ASGI app factory: FastAPI.__call__() missing 3 required positional arguments
```

### Solution

Don't use `--factory` flag. The app is already instantiated:

```bash
# Correct
uv run uvicorn mlsys.inference.service:app --reload

# Wrong
uv run uvicorn mlsys.inference.service:app --factory --reload
```

## Issue: Tests failing

### Common Causes

1. **Wrong environment**: Use `uv run pytest` instead of just `pytest`
2. **Missing model**: Train a model first with `uv run python scripts/train.py --config config/base_config.yaml`
3. **Missing dependencies**: Run `uv sync --dev`

### Solutions

```bash
# Full reset
deactivate  # if in venv
uv sync --dev --extra ui
uv run pytest
```

## Issue: Import errors in tests

### Problem
```
ImportError while importing test module
```

### Solution

Make sure PYTHONPATH is set (handled automatically by uv):

```bash
# With uv (automatic)
uv run pytest

# Manual (if not using uv)
PYTHONPATH=src pytest
```

## Issue: Docker build fails

### Problem
Model artifact missing

### Solution

```bash
# Train model first
uv run python scripts/train.py --config config/base_config.yaml --output artifacts/xgb_lead_scoring.joblib

# Then build
docker build -t mlsys-api:latest .
```

## Issue: Pre-commit hooks failing

### Solutions

```bash
# Clean and reinstall
pre-commit clean
pre-commit uninstall
pre-commit install

# Run manually to see errors
pre-commit run --all-files

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

## Issue: MLflow not working

### Problem
MLflow tracking not logging

### Solution

```bash
# Start MLflow server first
uv run mlflow ui --port 5000

# In another terminal, set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Then run training
uv run python scripts/train.py --config config/base_config.yaml
```

## Best Practices

### Always use uv

Instead of activating venvs, use uv:

```bash
# Bad (requires venv activation)
source .venv/bin/activate
pytest

# Good (works anywhere)
uv run pytest
```

### Check your environment

```bash
# See what environment you're in
echo $VIRTUAL_ENV

# Check Python being used
which python

# With uv
uv run which python
```

### Clean slate

If everything is broken:

```bash
# Deactivate any venv
deactivate 2>/dev/null || true

# Remove old venv
rm -rf .venv

# Sync fresh environment
uv sync --dev --extra ui

# Run tests
uv run pytest
```

## Common Commands Quick Reference

```bash
# Install/sync dependencies
uv sync --dev --extra ui

# Run tests
uv run pytest
uv run pytest --cov=src/mlsys

# Train model
uv run python scripts/train.py --config config/base_config.yaml

# Start API
uv run uvicorn mlsys.inference.service:app --reload

# Code quality
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run isort src/ tests/

# Pre-commit
uv run pre-commit run --all-files

# MLflow
uv run mlflow ui --port 5000

# Evaluate model
uv run python scripts/evaluate.py --model-path artifacts/xgb_lead_scoring.joblib
```

## Getting Help

If you're still stuck:

1. Check the full error message
2. Verify your environment: `uv run which python`
3. Check dependencies are installed: `uv run pip list`
4. Look at recent commits/changes
5. Try a clean install (see "Clean slate" above)

## Environment Variable Issues

### Set MLflow tracking

```bash
# In .env file
MLFLOW_TRACKING_URI=http://localhost:5000

# Or export
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Load .env automatically

```bash
# Install python-dotenv (already in dependencies)
# Then in your shell rc file (.bashrc, .zshrc):
if [ -f .env ]; then
    export $(cat .env | xargs)
fi
```

## Permission Issues

### Can't write to global Python

This happens when pip tries to install globally. Always use uv or a venv:

```bash
# Don't do this
pip install some-package

# Do this
uv pip install some-package
# or
source .venv/bin/activate && pip install some-package
```
