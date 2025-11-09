# Hardening Backlog

These are the improvements I would call out during an interview or follow-up iteration. They focus on scientific rigor, reliability, and usability without over-engineering the stack.

## 1. Data preparation discipline

- **Defensive copies and leakage guards** – `FeaturePipeline._combine_datasets` modifies the `noncustomers` frame in-place when aligning columns, potentially leaking state across runs in notebooks or tests. Clone inputs and log schema mismatches before filling defaults. `FeaturePipeline._ensure_numeric` silently drops categorical columns; emit warnings and surface a hook for categorical encoders to avoid accidental information loss. 【F:src/mlsys/features/pipeline.py†L64-L111】
- **Custom imputers** – Promote `FillNATransformer` to handle column-specific strategies so feature pipelines do not rely on the global numeric fill with zeros. This pairs nicely with drift monitoring because imputations become auditable. 【F:src/mlsys/features/transformer.py†L137-L191】

## 2. Model evaluation rigor

- **Probability handling** – Evaluation assumes binary classifiers with `predict_proba()[:, 1]`. Add adapter logic to support pandas DataFrame probabilities and multiclass settings so metrics remain correct when analysts plug in different estimators. 【F:src/mlsys/training/trainer.py†L109-L131】
- **Threshold persistence** – The `TrainingResult` returns an optimal threshold but the persisted joblib artifact does not store it, so serving always falls back to raw scores. Serialize the threshold alongside model metadata for consistent offline and online behavior. 【F:src/mlsys/training/trainer.py†L131-L155】【F:src/mlsys/serving/predictor.py†L16-L86】
- **Full retrain of best model** – After cross-validation, retrain the winning estimator on train+validation data before final evaluation to prevent underfitting caused by withholding the validation fold.

## 3. Tracking and observability

- **Atomic artifact logging** – `MLflowTracker.log_dict` writes to a deterministic filename in the working directory. Concurrent runs can collide. Write to a `NamedTemporaryFile` under a per-run directory to avoid race conditions. 【F:src/mlsys/tracking/mlflow_tracker.py†L42-L57】
- **Prediction monitoring** – Extend `PredictorService` to record request/response metrics (latency, error counts) via Prometheus counters/histograms and emit optional audit logs for scored entities. That makes the `/metrics` endpoint meaningful once deployed.

## 4. Serving resilience

- **Clear failure modes** – If no local artifact or registry model is available, `_ensure_model` leaves `_predictor` as `None`, leading to attribute errors on the first request. Raise a dedicated `ModelNotLoaded` exception and convert it into a 503 response inside the FastAPI dependency. 【F:src/mlsys/serving/predictor.py†L52-L86】【F:src/mlsys/api/routes/inference.py†L1-L40】
- **Schema enforcement** – Leverage `FeatureMetadata` to validate incoming payloads before scoring and return informative errors when fields are missing or extra.

## 5. Testing roadmap

- **API integration tests** – Add coverage for `/predict` and `/predict/batch` using the HTTPX test client with a stub predictor service. This guards dependency wiring and serialization. 【F:tests/integration/test_health.py†L1-L55】
- **Tracker stubs** – Introduce lightweight fake trackers in unit tests to assert metric/parameter logging without requiring a live MLflow server.

Each bullet is scoped to a presentation-friendly improvement: describe the problem, demonstrate awareness of the responsible code, and outline the implementation path.

