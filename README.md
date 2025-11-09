# HubSpot Prospect ML Framework

A fast, reproducible template for building, evaluating, and serving prospect conversion models. The project packages data loading, feature engineering, experiment tracking, and a FastAPI inference surface so that data scientists can iterate on tabular problems without rebuilding infrastructure from scratch.

## Why this framework

- **Config driven runs** – `mlsys.config.Settings` loads a single YAML file, validates it, and propagates settings into data loaders, feature transformers, model search, and serving defaults. Environment variables prefixed with `MLSYS_` override any value, making experiments reproducible and repeatable. 【F:src/mlsys/config/settings.py†L67-L113】
- **Extensible registries** – Data loaders, feature transformers, and models plug into registries. Adding a Snowflake loader or a CatBoost wrapper is a two-line change registered next to the existing CSV loader, categorical encoder, and scikit-learn models. 【F:src/mlsys/data/registry.py†L11-L58】【F:src/mlsys/features/transformer.py†L19-L87】【F:src/mlsys/models/registry.py†L10-L60】
- **One orchestration path** – `Trainer` wires together dataset ingestion, feature pipeline, cross-validated model selection, evaluation, callback hooks, and artifact creation. The CLI, tests, and API all reuse the same entrypoint. 【F:src/mlsys/training/trainer.py†L35-L136】
- **Deployment ready** – `PredictorService` loads either a local joblib bundle or an MLflow registered model and exposes predict-one/batch helpers consumed by FastAPI routes. Prometheus middleware and Docker assets are ready for production hardening. 【F:src/mlsys/serving/predictor.py†L16-L86】【F:scripts/serve.py†L4-L33】【F:docker/docker-compose.yml†L1-L64】
- **Tooling included** – `pyproject.toml` defines runtime and dev dependencies, Ruff, mypy, and pytest configuration. The `Makefile` wraps common tasks, and `.github/workflows/ci.yml` runs lint + tests on every push. 【F:pyproject.toml†L1-L89】【F:Makefile†L1-L27】【F:.github/workflows/ci.yml†L1-L41】

## Repository layout

```
ml-sys/
├── README.md                  # you are here
├── config/config.yaml         # default project configuration
├── config/config.schema.json  # JSON schema describing valid config keys
├── scripts/                   # CLI entrypoints (train/evaluate/serve)
├── src/mlsys/                 # installable Python package
│   ├── api/                   # FastAPI app and routes
│   ├── data/                  # loaders + registry abstraction
│   ├── features/              # feature transformers + pipeline
│   ├── models/                # model registry and wrappers
│   ├── training/              # trainer, callbacks, evaluation helpers
│   ├── serving/               # model loader + predictor service
│   └── tracking/              # MLflow-backed experiment tracker
├── tests/                     # unit + integration coverage
├── docker/                    # docker-compose + service Dockerfile
└── monitoring/                # Prometheus + Grafana stubs
```

## Quick start

1. **Set up a Python environment**
   ```bash
   make install  # installs mlsys with dev extras inside the active virtualenv
   ```

2. **Create datasets**
   Place CSV, Parquet, or JSON tables for `customers`, `noncustomers`, and `usage_actions` under a `data/` directory (see the configuration section for exact expectations).

3. **Review configuration**
   Edit `config/config.yaml` (sample below) to point to your data files, tune feature transformers, and pick the models you want to evaluate.

4. **Train models**
   ```bash
   make train               # runs scripts/train.py using config/config.yaml
   make train NO_TRACKING=1 # disable MLflow tracking if needed
   ```
   The training script prints a leaderboard, persists the best artifact, and, if MLflow is enabled, logs metrics + parameters.

5. **Evaluate or inspect artifacts**
   ```bash
   make evaluate
   ```
   The evaluator compares hold-out metrics and can be extended to compute business KPIs.

6. **Launch the API**
   ```bash
   make serve
   ```
   Visit `http://localhost:8000/docs` for interactive Swagger documentation. Predictions hit the same artifact used during training via `PredictorService`.

7. **Quality gates**
   ```bash
   make lint      # Ruff
   make typecheck # mypy
   make test      # pytest
   ```

8. **Container + monitoring (optional)**
   ```bash
   make docker-build
   make up        # docker-compose with API + MLflow + Prometheus/Grafana
   ```
   Tear everything down with `make down` and tail logs using `make logs`.

## Configuration guide

Configuration is centralized in `config/config.yaml` and validated by both Pydantic (`Settings`) and `config/config.schema.json`. Environment variables prefixed with `MLSYS_` (e.g. `MLSYS_TRAINING__TEST_SIZE=0.25`) override any value at runtime.

### Sample

```yaml
project:
  name: hubspot-ml-framework
  version: "1.0.0"
  description: "Prospect conversion modeling sandbox"

data:
  loader_type: csv
  sources:
    customers: data/raw/customers.csv
    noncustomers: data/raw/noncustomers.csv
    usage_actions: data/raw/usage_actions.csv
  id_column: company_id
  target_column: is_customer

features:
  transformers:
    - type: fillna
      strategy: median
    - type: categorical
      columns: [EMPLOYEE_RANGE, INDUSTRY]
      encoding: onehot
    - type: datetime
      columns: [LAST_ACTIVE_AT]
      reference_date: "2024-12-31"

training:
  test_size: 0.2
  val_size: 0.1
  random_state: 42
  models:
    - name: logistic_baseline
      type: sklearn.logistic_regression
      param_grid:
        C: [0.1, 1.0, 10.0]
        penalty: ["l2"]
    - name: random_forest
      type: sklearn.random_forest
      param_grid:
        n_estimators: [200, 300]
        max_depth: [10, 20]

tracking:
  backend: mlflow
  tracking_uri: "file:./mlflow"
  experiment_name: hubspot_conversion
  run_name_prefix: leadscore
  registry:
    model_name: hubspot_conversion_model
    stage: Production

serving:
  host: 0.0.0.0
  port: 8000
  model_name: hubspot_conversion_model
  model_stage: Production
  local_model_path: models/best.joblib
```

### Key sections

- **project** – human readable metadata (used in logs, MLflow tags, and presentations).
- **data** – loader type and a map of dataset names to file paths. Dataset names `customers`, `noncustomers`, and `usage_actions` receive special handling inside `FeaturePipeline`. 【F:src/mlsys/features/pipeline.py†L36-L80】
- **features.transformers** – ordered list of feature transformers applied prior to splitting. Built-in options include `fillna`, `categorical`, `datetime`, and `aggregation`; extend the registry to add custom feature builders. 【F:src/mlsys/features/transformer.py†L68-L163】
- **training** – split sizes, random seed, and enabled models. `Trainer` performs a parameter grid search for each enabled model using stratified K-fold cross-validation, tracks metrics via the configured tracker, and records leaderboard comparisons. 【F:src/mlsys/training/trainer.py†L70-L135】
- **tracking** – choose `mlflow`, `wandb`, or `none`. The MLflow tracker also handles optional registry pushes to a named model and target stage. 【F:src/mlsys/tracking/mlflow_tracker.py†L14-L71】
- **serving** – network config and artifact lookup. Point `local_model_path` at a joblib bundle for offline serving, or leave it empty to always resolve the latest model from the tracker’s registry. 【F:src/mlsys/serving/predictor.py†L36-L86】

## Presenting the project

- Start with the **problem framing**: prioritizing sales outreach requires rapid experimentation and trustworthy deployment. Use the config-driven workflow to show how analysts can tweak features and models without code changes.
- Walk through the **training lifecycle** using `scripts/train.py` and the evaluation summary printed to the console. Highlight how callbacks or experiment trackers can hook into the same flow for W&B dashboards or Slack alerts.
- Demo the **API** via `make serve`, cURL a `/predict` request, and mention how the same predictor can batch-score CSVs.
- Close with the **backlog** in `docs/improvement-plan.md` to show you already identified hardening opportunities (e.g., leakage guards, better artifact metadata, serving error handling).

For a longer presentation outline and suggested narrative beats, check `docs/presentation-script.md`.

## Getting unstuck

- Validate YAML with `python -m jsonschema -i config/config.yaml config/config.schema.json`.
- List available transformers or models using the registries:
  ```python
  from mlsys.features import TransformerRegistry
  from mlsys.models import ModelRegistry
  print(TransformerRegistry.list())
  print(ModelRegistry.list())
  ```
- Use the unit tests in `tests/` as working examples of the public APIs.

## Additional references

- `notebooks/` contains exploratory notebooks that mirror the production pipeline for storytelling during the presentation.
- `monitoring/` seeds Prometheus + Grafana dashboards for latency/error metrics once the API is containerized.
- `docs/` (added in this commit) includes a speaking script and prioritized improvement plan to guide interviews and follow-up work.

