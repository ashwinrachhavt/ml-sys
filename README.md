# ml-sys

Production-ready ML framework for HubSpot's Finance MLE assessment. The repository demonstrates how to enable data scientists to iterate quickly while preserving reproducibility, governance, and operational rigor.

## Why ml-sys
- **Config-driven runs**: `FrameworkConfig` captures data paths, model settings, tracking, and serving parameters so experiments are reproducible and auditable.
- **Modular architecture**: Data loading, feature engineering, modeling, tracking, and serving are cleanly separated to make extensions safe and obvious.
- **Model registry built-in**: Every training run is versioned locally; the inference API automatically serves the current champion and exposes its metadata.
- **Observability ready**: FastAPI surface with Prometheus metrics, optional MLflow tracking, and structured evaluation outputs.
- **Developer ergonomics**: CLI wrappers, comprehensive docs, Ruff/mypy/pytest in CI, and uv-based dependency management keep the workflow tight.

## Project Layout
```
ml-sys/
├── config/                # YAML configs for reproducible runs
├── scripts/               # CLI entry points (train, evaluate, register)
├── src/mlsys/
│   ├── core/              # Config models & shared protocols
│   ├── data/              # Raw table loaders
│   ├── features/          # Feature engineering utilities
│   ├── models/            # Estimator factory wrappers
│   ├── tracking/          # MLflow tracker abstraction
│   ├── inference/
│   │   ├── registry.py    # Lightweight file-based model registry
│   │   └── service.py     # FastAPI inference surface
│   └── training/
│       ├── pipeline.py    # End-to-end training orchestration
│       └── cli.py         # Config-aware training CLI
├── tests/                 # Pytest suites for pipeline and API
├── monitoring/            # Prometheus / Grafana scaffolding
├── data/                  # Assessment CSVs (ignored by registry locally)
└── docs/                  # Onboarding guides (START_HERE, QUICKSTART, etc.)
```

## Quick Start

### 1. Install dependencies
```bash
uv sync --dev
uv run pre-commit install
```

### 2. Train a model
```bash
uv run python scripts/train.py   --config config/base_config.yaml   --tune   --calibration-size 0.1   --output artifacts/xgb_model.joblib
```
- CLI overrides (`--data-dir`, `--registry-path`, `--random-state`) layer on top of the YAML configuration.
- Every run saves metrics, config snapshot, classification report, and model artifact under `artifacts/` and registers it in the local registry defined by `serving.model_registry_path`.

### 3. Inspect results
```bash
uv run python scripts/evaluate.py   --config config/base_config.yaml   --use-registry   --output artifacts/eval.json
```
This loads the current champion from the registry, evaluates on the configured dataset, and writes a metrics JSON alongside console output.

### 4. Serve predictions
```bash
uv run uvicorn mlsys.inference.service:app --reload
```
- Health: `GET /health`
- Predict: `POST /score`
- Docs: `GET /docs`
- Metrics: `GET /metrics` (if `prometheus-client` installed)

The API reports which artifact produced the response so monitoring can tie back to registry versions.

### 5. Register models manually (optional)
```bash
uv run python scripts/register_model.py   artifacts/xgb_model.joblib   --config config/base_config.yaml   --metrics artifacts/eval.json   --primary-metric roc_auc_calibrated
```
This copies the artifact into the registry, updates the champion if the primary metric improves, and (when `--mlflow` is passed) logs it to MLflow using the tracking settings from the config.

## Configuration Surface
`FrameworkConfig` (YAML) drives every step:
- `data`: paths, random seeds, split ratios, calibration fraction
- `features`: toggles for SMOTE and feature engineering options
- `model`: algorithm, hyperparameters, calibration strategy, tuning flag
- `tracking`: MLflow URI, experiment name, artifact logging toggle
- `serving`: host/port for inference, registry location, Prometheus toggle
- `artifacts_path`: where training outputs are stored by default

CLI overrides or environment variables simply mutate this schema, preserving reproducibility in the persisted config snapshots.

## Automation & Quality Gates
- **CI** (`.github/workflows/ci.yml`): Ruff lint/format, mypy type-checking, pytest suites.
- **Scheduled Retraining** (`retrain.yml`): demonstrates how nightly retraining would run the CLI and update the registry.
- **CD Stub** (`cd.yml`): shows where deployment packaging/tests would live.
- **Pre-commit**: Ruff (lint+format), mypy, isort, YAML/TOML/JSON validators.

## Testing
```bash
uv run pytest -q                    # full suite
uv run pytest tests/training -q     # training pipeline smoke tests
uv run pytest tests/inference -q    # API tests (loads registry champion)
```

## Monitoring & Observability
```bash
uv run uvicorn mlsys.inference.service:app
```
- Prometheus metrics automatically expose prediction counts, latency, and batch sizes.
- Optional MLflow tracker logs params/metrics/artifacts for each run if `MLFLOW_TRACKING_URI` is configured.

## Next Steps
- Extend `src/mlsys/models/` with additional estimators (e.g., CatBoost, LogisticRegression).
- Integrate drift detection and scheduled registry promotions in `retrain.yml`.
- Enrich docs (`docs/`) with architecture diagrams, troubleshooting, and on-call runbooks.

The repo is deliberately opinionated but lightweight: everything is plain Python packaged for clarity, making it easy to swap components or scale into a fuller production stack.
