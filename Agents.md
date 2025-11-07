# Agents Mission Brief: HubSpot Finance MLE Assessment

## Why This Repo Exists
We are designing **ml-sys**, a production-minded ML platform that helps HubSpot data scientists turn raw go-to-market data into deployable conversion models quickly, safely, and reproducibly. The assessment is less about model accuracy and more about our ability to build a trustworthy system that other scientists love to use.

**North Star:** deliver a miniature ML platform that demonstrates senior-level software engineering discipline, thoughtful UX for data scientists, and a clear growth path toward a full MLOps stack.

---

## Success Criteria
- **Production posture**: deterministic environments (uv + lock files), CI gates (lint, type-checks, tests), automated artifact/version management, fast inference surface.
- **Reproducible experimentation**: every run is driven by config/CLI, writes metrics-artifacts-config snapshots, and can be replayed identically.
- **Modular architecture**: clean protocols/ABCs separate data ingest, feature engineering, training, evaluation, tracking, and serving. Engineers can swap any layer without touching others.
- **Scientist empathy**: quickstarts, opinionated defaults, and CLI helpers make iteration delightful. Documentation reads like we care about our peers’ time.
- **Transparent storytelling**: README, diagrams, and code comments clearly communicate trade-offs, assumptions, and how to extend the system.

---

## Target System Overview
```
ml-sys/
├── config/               # YAML configs for data, models, tracking, serving
├── data/                 # Assessment CSVs + schema docs (read-only in prod)
├── scripts/              # User-facing CLI entrypoints (train, evaluate, serve, register)
├── src/mlsys/
│   ├── core/             # Config loading, protocols, pipeline orchestration
│   ├── data/             # Loaders, validators, synthetic fallbacks
│   ├── features/         # Feature transformers, encoders, imbalance handlers
│   ├── models/           # Estimator wrappers (XGBoost, CatBoost, SKLearn)
│   ├── evaluation/       # Metrics calculators, reports, calibration
│   ├── tracking/         # MLflow tracker + metadata writers
│   └── inference/        # FastAPI service, model registry, schemas
├── tests/                # Pytest suite, fixtures, smoke/e2e hooks
├── api/                  # FastAPI entry module for uvicorn
├── notebooks/            # Optional demo walkthroughs (not the system of record)
├── infra/                # Dockerfile, docker-compose, monitoring scaffolds
├── .github/workflows/    # CI (lint/type/test) + CD stubs (train, register)
└── docs/                 # README, START_HERE, QUICKSTART, diagrams
```

### Core Components
1. **Configuration Layer** (`core.config`)
   - Typed Pydantic models map YAML into runtime objects.
   - Config captures data paths, feature toggles, model params, tracking URIs, serving ports.
   - Every training run saves its resolved config alongside artifacts for provenance.

2. **Data & Features Layer** (`data` + `features`)
   - CSV loaders with schema validation fallback to synthetic stubs if files missing.
   - Feature transformers implement a `FeatureTransformer` protocol (fit, transform, fit_transform).
   - Categorical handling via OneHot or CatBoost native support; SMOTE/SMOTENC optional for imbalance.
   - Strict split ordering: train/test split precedes any resampling or scaling to avoid leakage.

3. **Model Layer** (`models`)
   - BaseModel class enforces train/predict/predict_proba/save/load contract and logs core metrics.
   - Implementations: XGBoost (with early stopping), CatBoost (categorical aware), fallback Scikit-learn options.
   - Model hyperparameters supplied from config; defaults set for stability.

4. **Pipeline Orchestrator** (`core.pipeline`)
   - Coordinates data loading → feature engineering → model training → calibration → evaluation → artifact logging.
   - Emits a `TrainingResult` dataclass containing metrics, classification report, artifact paths, and metadata.
   - Supports optional hyperparameter search and calibration splits.

5. **Tracking & Registry** (`tracking`, `inference`)
   - MLflow Tracker logs params, metrics, artifacts; can register models under experiment name.
   - Lightweight file-based model registry keeps semantic versions, metrics snapshot, and symlink to best model (`best_model.joblib`).
   - Registry integrates with CI/CD to gate promotion when holdout metrics improve.

6. **Inference Surface** (`api`, `inference.service`)
   - FastAPI app loads the best model on startup, exposes `/health`, `/predict`, `/predict/batch`, `/model/info`, `/models`, `/model/reload`.
   - Prometheus optional metrics for latency, batch size, request outcomes.
   - Pydantic schemas guarantee contract with upstream consumers.

7. **DevEx & Ops**
   - `uv` manages virtualenv + lockstep deps; Dockerfile uses uv for reproducible containers.
   - CI (`ci.yml`) runs `ruff`, `ruff-format`, `isort`, `mypy`, `pytest`.
   - Additional workflows (`cd.yml`, `retrain.yml`) show how automated retraining and deployments could happen.
   - Shell helpers (`run.sh`, `setup.sh`, `fix_env.sh`, `test_e2e.sh`) streamline local workflows.

---

## End-to-End Workflows
### 1. Experimentation
1. Scientist clones repo, runs `uv sync`, and follows `START_HERE.md`.
2. Creates/edits a config under `config/experiments/exp_<id>.yaml`.
3. Executes `uv run python scripts/train.py --config <path>` or `uv run ml-train --config <path>`.
4. Pipeline logs metrics, saves model, config snapshot, and classification report under `artifacts/<run_id>/`.
5. MLflow UI (`uv run mlflow ui`) provides visualization; CLI prints ROC-AUC, PR-AUC, precision@k etc.
6. Registry decides if the new model beats the previous champion; symlink updates automatically.

### 2. Evaluation & Reporting
- `scripts/evaluate.py` can re-score a model on any dataset, output JSON metrics, confusion matrix, classification report.
- `scripts/register_model.py` registers artifacts or triggers manual promotions.
- Documentation guides how to interpret metrics and extend evaluation logic.

### 3. Deployment & Serving
- Run-time server: `uv run python scripts/serve.py --config config/base_config.yaml`.
- Uvicorn serves FastAPI with auto OpenAPI docs at `/docs`.
- Health probes ensure readiness; optional `docker-compose.yml` brings up API + monitoring stack.
- `monitoring/` contains starter Grafana/Prometheus configs.

### 4. Automation Hooks
- `retrain.yml`: orchestrates nightly retraining using latest data, logs to MLflow, updates registry if metrics improve.
- `cd.yml`: deploys new best model by building Docker image, pushing to registry, running smoke tests.
- Shell scripts encapsulate these flows for local repro.

---

## Implementation Principles
- **Deterministic runs**: set `random_state` everywhere; persist raw/train/val/test splits or seeds for reproducibility.
- **No leakage**: all encoding/SMOTE/scaling objects fit only on training data; calibration uses separate holdout or CV.
- **Typed interfaces**: Pydantic models + Python typing + mypy keep contracts explicit.
- **Composable design**: new data source? add loader + config entry. New model? subclass BaseModel and register in factory.
- **Artifact hygiene**: each run stores model (`.joblib`), metrics (`.json`), reports (`.txt`), config (`.yaml`), plus MLflow pointers.
- **Observability**: logging (structured), Prometheus counters/histograms, saved evaluation summaries.
- **Docs-first mindset**: README tells the story, QUICKSTART shows first five minutes, HOW_TO_TEST explains validation strategy, TROUBLESHOOTING covers common issues.

---

## Talking Points for Presentation
1. **Trade-offs**: uv vs poetry, custom registry vs MLflow registry, file-based artifacts, why FastAPI, why XGBoost/CatBoost.
2. **Scientist UX**: config-driven flows, CLI commands, quick iteration loops, clear logs.
3. **Production readiness**: CI gates, Docker, monitoring hooks, model versioning, reproducible runs.
4. **Integration**: artifacts consumable by other teams (joblib, JSON metrics), REST API with consistent schemas, MLflow for cross-team reporting.
5. **Future roadmap**: feature store integration, automated hyperparameter tuning, drift detection, streaming inference, A/B testing harness.

---

## Stretch Goals (if time allows)
- Optuna-driven hyperparameter sweeps stored in MLflow.
- Data quality checks with Great Expectations.
- Model explainability (SHAP) artifactory.
- CLI plugin to scaffold new experiments from templates.
- Terraform/IaC stubs showing how infra would evolve.

---

## Definition of Done Checklist
- [ ] Config-driven pipeline runs end-to-end on assessment data with deterministic outputs.
- [ ] CI passes: lint, format, import order, type-check, unit tests.
- [ ] Model artifacts saved with metrics + config snapshots; registry selects champion.
- [ ] FastAPI server returns predictions using latest champion and passes smoke tests.
- [ ] Documentation suite updated (README, QUICKSTART, START_HERE, HOW_TO_TEST, TROUBLESHOOTING).
- [ ] Presentation narrative prepared covering design decisions, trade-offs, future work.

Stay focused on clarity, reproducibility, and the developer experience. If every decision answers “how would this help a HubSpot data scientist succeed under real constraints?”, we win.
