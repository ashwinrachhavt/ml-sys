# Quick Start Guide

Build, evaluate, and serve a lead-scoring model in minutes.

## 1. Install Dependencies
```bash
git clone https://github.com/yourusername/ml-sys.git
cd ml-sys
uv sync --dev
uv run pre-commit install
```

## 2. Train a Model
```bash
uv run python scripts/train.py   --config config/base_config.yaml   --tune   --calibration-size 0.1
```
- The CLI merges command-line flags into the YAML config and records metrics, artifacts, and config snapshots under `artifacts/`.
- Every run registers the model under `serving.model_registry_path`; the latest champion is served automatically.

## 3. Evaluate the Champion
```bash
uv run python scripts/evaluate.py   --config config/base_config.yaml   --use-registry   --output artifacts/eval.json
```
This loads the best model from the registry, evaluates it on the configured dataset, prints metrics, and stores them as JSON.

## 4. Serve Predictions
```bash
uv run uvicorn mlsys.inference.service:app --reload
```
Open http://localhost:8000/docs to interact with the API. Responses include the artifact path used for the prediction.

## 5. Optional: Manual Registration
```bash
uv run python scripts/register_model.py   artifacts/xgb_model.joblib   --config config/base_config.yaml   --metrics artifacts/eval.json
```
This copies the artifact into the registry (updating the champion if it performs better). Add `--mlflow` to log the model to MLflow using the tracking settings in the config.

## 6. Common Commands
```bash
uv run pytest                  # test suite
uv run ruff check              # lint
uv run ruff format             # format
uv run python scripts/evaluate.py --help
uv run python scripts/train.py --help
```

See [`README.md`](README.md) for a deeper tour, [`START_HERE.md`](START_HERE.md) for troubleshooting, and [`MLOPS_ARCHITECTURE.md`](MLOPS_ARCHITECTURE.md) for the system design.
