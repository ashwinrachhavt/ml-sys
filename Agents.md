You’re basically asking for a **FastAPI-first MLOps template** that you can reuse for any tabular ML project. Let’s design one that hits all 8 of your requirements.

Below is a concrete **repo structure + responsibilities**; after that I’ll map each requirement to the right modules.

---

## 🧱 Top-level layout

```text
mlops_fastapi_service/
│
├── README.md
├── pyproject.toml           # or setup.py + requirements.txt
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml           # pytest + lint + type-check
│
├── data/
│   ├── raw/                 # optional, mainly for local dev
│   ├── processed/
│   └── artifacts/           # local models, profiles, etc (or S3)
│
├── mlflow/                  # tracking store if run locally
│   └── mlruns/
│
├── app/                     # FastAPI application package (recommended pattern) :contentReference[oaicite:0]{index=0}
│   ├── __init__.py
│   ├── main.py              # create FastAPI() and include routers
│   │
│   ├── core/                # “infrastructure”: settings, logging, deps
│   │   ├── __init__.py
│   │   ├── config.py        # Pydantic Settings: paths, mlflow URI, etc.
│   │   ├── logging.py       # struct log config
│   │   └── mlflow_utils.py  # helpers to start runs, log artifacts
│   │
│   ├── api/                 # all FastAPI routers
│   │   ├── __init__.py
│   │   ├── deps.py          # dependencies (DB / model loader / auth)
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── eda.py       # /eda/profile
│   │   │   ├── training.py  # /admin/train, /admin/models
│   │   │   ├── inference.py # /predict, /predict/batch
│   │   │   └── health.py    # /health, /metrics
│   │   └── routers.py       # include_router(...) wiring
│   │
│   ├── schemas/             # Pydantic models
│   │   ├── __init__.py
│   │   ├── eda.py           # request/response for profile endpoint
│   │   ├── training.py      # train request (model_name, params, run_name)
│   │   └── inference.py     # ProspectFeatures, Prediction, BatchPrediction
│   │
│   ├── data/                # data loading & validation
│   │   ├── __init__.py
│   │   ├── base_loader.py   # abstract DataLoader
│   │   ├── pandas_loader.py # current implementation
│   │   ├── validators.py    # schema checks, type checks (pydantic + pandera)
│   │   └── transforms.py    # small reusable transforms (e.g. recency calc)
│   │
│   ├── features/            # feature pipelines (offline)
│   │   ├── __init__.py
│   │   ├── base_transformer.py
│   │   ├── feature_builder.py  # your final feature-engineering logic
│   │   ├── pipelines.py        # compose loaders + builder into a DAG-ish flow
│   │   └── store.py            # simple feature “registry” / versioning
│   │
│   ├── ml/                  # modeling code (training + inference)
│   │   ├── __init__.py
│   │   ├── base_model.py       # unified interface: fit/predict/predict_proba
│   │   ├── models/
│   │   │   ├── logistic.py
│   │   │   ├── xgboost.py
│   │   │   ├── catboost.py
│   │   │   └── lgbm.py
│   │   ├── trainer.py          # training orchestration, MLflow logging
│   │   ├── registry.py         # load/save best model from MLflow or disk
│   │   └── evaluation.py       # metrics, cross-validation helpers
│   │
│   ├── serving/             # runtime prediction logic
│   │   ├── __init__.py
│   │   ├── predictor.py      # load preprocessor + model; single/batch predict
│   │   ├── batch_scorer.py   # offline scoring job (can be called via CLI)
│   │   └── monitoring.py     # log prediction metrics, request timing, etc.
│   │
│   └── monitoring/          # observability hooks
│       ├── __init__.py
│       ├── prometheus.py     # Prometheus metrics endpoint & middleware
│       └── drift.py          # simple PSI / KS tests for input drift, etc.
│
├── scripts/
│   ├── build_features.py     # CLI wrapper: run feature pipeline once
│   ├── train.py              # CLI training entrypoint (calls app.ml.trainer)
│   ├── evaluate.py           # offline eval / comparison against test set
│   └── serve.py              # uvicorn entrypoint (optional)
│
└── tests/
    ├── __init__.py
    ├── test_data_loaders.py
    ├── test_feature_builder.py
    ├── test_trainer.py
    ├── test_inference_api.py
    └── fixtures/             # sample CSVs, stub models, etc.
```

This mirrors patterns from common FastAPI project-structure guides and templates (separating `core`, `api`, `schemas`, `services/ml` etc.) while staying simple enough to navigate. ([Medium][1])

---

## 🔌 How each of your requirements fits

### 1. EDA profile report endpoint (pandas/ydata-profiling)

* **Files:**

  * `app/api/v1/eda.py` – FastAPI router for `/eda/profile`
  * `app/schemas/eda.py` – request/response models
* **Implementation idea:**

  * Accept a file upload (`UploadFile`) or S3 URL.
  * Use `pandas.read_csv(...)` then `ydata_profiling.ProfileReport(df, minimal=True)` to generate HTML. (Pandas Profiling is now `ydata-profiling` but the API is the same. ([DataCamp][2]))
  * Store the HTML in:

    * local `data/artifacts/profiles/` **and**
    * log it as an artifact to MLflow (via `mlflow.log_text(html, "eda/profile_<timestamp>.html")`).
  * Return either:

    * A direct HTMLResponse, or
    * A JSON response with a link to the saved artifact.

This endpoint becomes a generic “EDA microservice” – any CSV → quick profile.

---

### 2. Composable Dataloaders & feature transforms

* **Files:**

  * `app/data/base_loader.py` – abstract `load_raw()` method.
  * `app/data/pandas_loader.py` – current implementation using CSVs.
  * `app/features/feature_builder.py` – merges tables, engineers features.
  * `app/features/pipelines.py` – orchestrates loader + builder into one flow.

Pattern:

```python
# app/data/base_loader.py
class BaseDataLoader(ABC):
    @abstractmethod
    def load_raw(self) -> Dict[str, pd.DataFrame]:
        ...

# app/data/pandas_loader.py
class PandasCSVLoader(BaseDataLoader):
    def load_raw(self) -> Dict[str, pd.DataFrame]:
        # read customers.csv, noncustomers.csv, usage_actions.csv
        ...

# app/features/pipelines.py
def build_training_features(loader: BaseDataLoader) -> FeatureBuildResult:
    raw = loader.load_raw()
    return FeatureBuilder().build(**raw)
```

Later, when a real ETL exists, you just add a new loader (e.g. `SnowflakeLoader`) implementing the same interface.

---

### 3. Training pipeline inside FastAPI (and CLI)

* **Files:**

  * `app/ml/trainer.py`
  * `app/ml/registry.py`
  * `app/api/v1/training.py`
  * `scripts/train.py`

**Core idea:**

* `trainer.py` exposes a function like:

  ```python
  def train_and_register_model(
      model_name: str,
      params: dict,
      experiment_name: str,
  ) -> TrainedModelInfo:
      # 1. build features (or reuse pre-saved features)
      # 2. run KFold/holdout training
      # 3. log metrics, params, plots to MLflow
      # 4. register “best” run in MLflow Model Registry or local registry
      ...
  ```

* FastAPI **admin endpoint** `/admin/train` (with auth!) calls that function to kick off training jobs.

* `scripts/train.py` is a thin CLI wrapper that calls the same trainer, so notebooks / CLI / API all share the **same training logic**.

MLflow orchestration using the same patterns you already implemented (model signature + input_example, log_figure/log_text instead of writing local files). ([Medium][3])

---

### 4. Online and batch inference endpoints

* **Files:**

  * `app/serving/predictor.py` – core logic: load preprocessor + model; `predict_one` and `predict_batch`.
  * `app/api/v1/inference.py` – FastAPI router.
  * `app/schemas/inference.py` – `ProspectFeatures`, `Prediction`, `BatchPredictionRequest`, etc.
  * `app/serving/batch_scorer.py` – offline scoring job used by `/predict/batch` and scripts.

**Flow (online):**

1. `/predict` receives JSON matching `ProspectFeatures` (Pydantic).
2. API converts to pandas/DataFrame or dict of features.
3. `predictor.load_model()` returns (cached) preprocessor + model (from MLflow or local artifact).
4. `predictor.predict_one(features)` returns score + label.
5. Endpoint logs prediction latency and basic stats to Prometheus + MLflow (for monitoring).

**Flow (batch):**

* `/predict/batch` accepts:

  * uploaded CSV, or
  * a list of `ProspectFeatures` items.
* Uses `predictor.predict_batch(df)` → array of probabilities + labels.
* Optionally stores full prediction CSV to `data/artifacts/predictions/` and logs as MLflow artifact.

---

### 5. MLflow for training *and* inference metrics

You already have the training side; you can:

* Set the tracking URI & experiment in `app/core/mlflow_utils.py`.
* For **training**, `trainer.py` starts runs, logs params, metrics, artifacts.
* For **inference**, you can:

  * either reuse a “monitoring” experiment (e.g. `prospect_conversion_inference`),
  * or push aggregated metrics (like rolling AUC once ground truth arrives).

MLflow docs describe model packaging + deployment, and in v3 they even include a built-in FastAPI inference server; your version is a customized version of that inside your own app. ([MLflow][4])

---

### 6–7. Docker, Docker Compose, monitoring

* **Files:**

  * `Dockerfile` – multi-stage build (install deps, copy app, run uvicorn).
  * `docker-compose.yml` – FastAPI service + MLflow server + Prometheus + Grafana.
  * `monitoring/prometheus.yml`, `monitoring/grafana_dashboard.json`.

Use FastAPI’s recommended Docker patterns (non-root user, gunicorn/uvicorn worker, health checks) and general best practices like small base image, env-driven config. ([FastAPI][5])

Typical Dockerfile skeleton:

```dockerfile
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml` then wires:

* `web` → FastAPI container
* `mlflow` → MLflow tracking server (with mounted volume for `mlruns/`)
* `prometheus` & `grafana` → scrape `/metrics` from `web`

---

### 8. GitHub Actions CI + pytest + stubs

* **Files:**

  * `.github/workflows/ci.yml`
  * `tests/` with fixtures and unit tests.

CI pipeline ideas:

* **Jobs:**

  * `lint` – run `ruff` or `flake8`
  * `type-check` – `mypy app`
  * `test` – `pytest tests/`
* Pin Python version matrix if you want (e.g. 3.10, 3.11).
* Use fixtures + stub objects:

  * fake DataLoader that returns a tiny DataFrame
  * fake Model that returns constant probabilities
  * so tests don’t depend on heavy ML frameworks.

Example minimal `ci.yml` outline:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest -q
```

---

## Why this structure works for what you want

* **Composable**: data loaders, feature builders, and models all hang off abstract base classes – easy to swap implementations without touching API code.
* **Single source of truth**: training logic in `app/ml/trainer.py` is used by notebooks, CLI, and admin endpoints.
* **FastAPI-centric**: everything lives in `app/` (as per common FastAPI templates) with clear separation between `api`, `schemas`, `core`, and `ml/serving`. ([GitHub][6])
* **Production-minded**: Docker, MLflow, Prometheus/Grafana, CI – all wired from day one.

If you’d like, next step we can:

* Pick **one** model family (say CatBoost or GradientBoosting from your comparison table),
* Design the exact `trainer.train_and_register_model()` signature and its MLflow logging,
* Then draft the FastAPI router for `/admin/train` and `/predict`.

[1]: https://medium.com/%40amirm.lavasani/how-to-structure-your-fastapi-projects-0219a6600a8f?utm_source=chatgpt.com "How to Structure Your FastAPI Projects"
[2]: https://www.datacamp.com/tutorial/pandas-profiling-ydata-profiling-in-python-guide?utm_source=chatgpt.com "Pandas Profiling (ydata-profiling) in Python: A Guide for ..."
[3]: https://tuanatran.medium.com/end-to-end-ml-deployment-with-mlflow-fastapi-and-aws-fargate-2fa554925792?utm_source=chatgpt.com "End-to-End ML Deployment with MLflow, FastAPI, and ..."
[4]: https://mlflow.org/docs/3.2.0/ml/deployment/?utm_source=chatgpt.com "MLflow Serving"
[5]: https://fastapi.tiangolo.com/deployment/docker/?utm_source=chatgpt.com "FastAPI in Containers - Docker"
[6]: https://github.com/99sbr/fastapi-template?utm_source=chatgpt.com "99sbr/fastapi-template"
