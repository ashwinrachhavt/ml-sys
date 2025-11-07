# Setup Guide for ml-sys

This guide will walk you through setting up the ml-sys project for local development.

## Prerequisites

- Python 3.11 or higher
- Git
- Docker and Docker Compose (optional, for containerized deployment)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ml-sys.git
cd ml-sys
```

### 2. Set Up Python Environment

#### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --dev --extra ui
```

#### Using pip

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with dev dependencies
pip install -e ".[dev,ui]"
```

### 3. Install Pre-commit Hooks

```bash
# Activate virtual environment if using pip
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install
```

This will automatically run code quality checks before each commit.

### 4. Verify Installation

```bash
# Run tests to verify everything is working
uv run pytest  # or just `pytest` if using pip

# Check code quality
uv run ruff check src/ tests/
uv run isort --check-only src/ tests/
```

### 5. Train Initial Model

```bash
# Train a model with default settings
uv run python scripts/train.py --config config/base_config.yaml

# Or with hyperparameter tuning
uv run python scripts/train.py --config config/base_config.yaml --tune
```

The trained model will be saved to `artifacts/xgb_lead_scoring.joblib`.

### 6. Start the API Server

```bash
# Start the FastAPI inference service
uv run uvicorn mlsys.inference.service:app --reload
```

Visit http://localhost:8000/docs to see the interactive API documentation.

### 7. (Optional) Set Up MLflow

```bash
# Start MLflow tracking server
uv run mlflow ui --port 5000
```

Visit http://localhost:5000 to view experiment tracking.

Set the tracking URI before training:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python scripts/train.py --config config/base_config.yaml --tune
```

### 8. (Optional) Set Up Full Stack with Docker Compose

This will start the API, MLflow, Prometheus, and Grafana:

```bash
docker-compose up -d
```

Services will be available at:
- API: http://localhost:8000
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (login: admin/admin)

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/mlsys --cov-report=html

# Run specific test file
uv run pytest tests/training/test_pipeline.py -v
```

### Code Quality

```bash
# Lint code
uv run ruff check src/ tests/ scripts/

# Auto-fix linting issues
uv run ruff check --fix src/ tests/ scripts/

# Format code
uv run ruff format src/ tests/ scripts/

# Sort imports
uv run isort src/ tests/ scripts/
```

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes

3. Run tests and linting:
   ```bash
   uv run pytest
   uv run ruff check src/ tests/
   uv run isort --check-only src/ tests/
   ```

4. Commit (pre-commit hooks will run automatically):
   ```bash
   git add .
   git commit -m "Add my feature"
   ```

5. Push and create a pull request:
   ```bash
   git push origin feature/my-feature
   ```

## Troubleshooting

### ImportError: No module named 'mlsys'

Make sure you've installed the package in editable mode and set PYTHONPATH:
```bash
pip install -e .
export PYTHONPATH=src
```

### Model Not Found Error

Train a model first:
```bash
uv run python scripts/train.py --config config/base_config.yaml
```

### Docker Build Fails

Make sure you have the model artifact:
```bash
mkdir -p artifacts
uv run python scripts/train.py --config config/base_config.yaml --output artifacts/xgb_lead_scoring.joblib
docker build -t mlsys-api:latest .
```

### Pre-commit Hooks Failing

Update hooks and try again:
```bash
pre-commit clean
pre-commit install
pre-commit run --all-files
```

## Environment Variables

Create a `.env` file for local development:

```env
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Model paths
MODEL_PATH=artifacts/xgb_lead_scoring.joblib

# API configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
```

## Next Steps

- Read the [README.md](README.md) for usage examples
- Check out the [API documentation](http://localhost:8000/docs)
- Explore the [MLflow UI](http://localhost:5000) for experiment tracking
- Set up [Grafana dashboards](http://localhost:3000) for monitoring

## Getting Help

- Check [existing issues](https://github.com/yourusername/ml-sys/issues)
- Read the documentation
- Ask questions in [Discussions](https://github.com/yourusername/ml-sys/discussions)
