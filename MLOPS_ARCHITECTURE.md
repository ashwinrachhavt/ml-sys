# MLOps Architecture and Framework Design

## Executive Summary

This document describes the production-grade MLOps framework implemented for the ml-sys lead scoring system. The framework encompasses the entire machine learning lifecycle from data versioning through model deployment, monitoring, and continuous retraining.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data & Model Lifecycle                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐ │
│  │   Data   │───▶│  Training │───▶│   Model  │───▶│ Inference│ │
│  │Ingestion │    │  Pipeline │    │ Registry │    │  Service │ │
│  └──────────┘    └───────────┘    └──────────┘    └──────────┘ │
│       │               │                 │               │        │
│       ▼               ▼                 ▼               ▼        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              MLflow Experiment Tracking                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Automation & Orchestration                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CI Pipeline         CD Pipeline         Retrain Pipeline        │
│  ┌──────────┐       ┌──────────┐        ┌──────────┐           │
│  │  Lint &  │       │  Build & │        │  Drift   │           │
│  │   Test   │──────▶│  Deploy  │◀───────│ Detection│           │
│  └──────────┘       └──────────┘        └──────────┘           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Monitoring & Observability                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Prometheus  ──────▶  Grafana  ──────▶  Alerts                  │
│  (Metrics)           (Dashboards)      (Notifications)           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Training Pipeline with SMOTE Integration

**Location**: `src/mlsys/training/pipeline.py`

**Key Features**:
- **SMOTE for Class Imbalance**: Automatically applies SMOTE when imbalance ratio > 2.0
- **Pipeline Isolation**: Uses `imblearn.pipeline.Pipeline` to ensure SMOTE only applies during fit()
- **Configurable**: Can be disabled via `use_smote=False` parameter
- **Safe Defaults**: Automatically adjusts k_neighbors based on minority class size

**Implementation**:
```python
if use_smote and imbalance_ratio > 2.0:
    pipeline = ImbPipeline(
        steps=[
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=42, k_neighbors=min(5, positive - 1))),
            ("model", estimator),
        ]
    )
```

**Benefits**:
- Prevents data leakage (SMOTE only on training set)
- Improves model performance on minority class
- No manual intervention required

### 2. FastAPI Inference Service

**Location**: `src/mlsys/inference/service.py`

**Architecture**:
- **Pydantic Validation**: Type-safe request/response models
- **Health Checks**: `/health` endpoint for orchestrators
- **Prometheus Integration**: Automatic metrics collection
- **Logging Middleware**: Request/response tracking
- **Error Handling**: Graceful degradation with proper HTTP status codes

**Endpoints**:
- `GET /`: API information
- `GET /health`: Health check with model status
- `POST /score`: Lead scoring predictions
- `GET /metrics`: Prometheus metrics
- `GET /docs`: Interactive API documentation

**Metrics Tracked**:
- `mlsys_predictions_total`: Prediction count by status (success/error)
- `mlsys_prediction_duration_seconds`: Latency histogram
- `mlsys_prediction_batch_size`: Batch size distribution

### 3. MLflow Experiment Tracking

**Integration Points**:
- Automatic logging during training
- Parameters: All hyperparameters and config
- Metrics: ROC AUC, PR AUC, Brier score, precision@k
- Artifacts: Model files, classification reports
- Model Registry: Versioned model storage with staging

**Usage**:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/train.py --config config/base_config.yaml --tune
```

**Benefits**:
- Complete reproducibility
- Model comparison across experiments
- Centralized model storage
- Version control for ML artifacts

### 4. Docker Containerization

**Multi-Stage Build**:
1. **Builder Stage**: Install dependencies and build
2. **Runtime Stage**: Minimal production image

**Security Features**:
- Non-root user (`mlsys`)
- Minimal base image (python:3.11-slim)
- No unnecessary packages
- Health check configuration

**Size Optimization**:
- Layer caching
- .dockerignore for build context
- Multi-stage build reduces final image size

### 5. CI/CD Pipeline

#### Continuous Integration (.github/workflows/ci.yml)

**Stages**:
1. **Lint and Format**: Ruff + isort validation
2. **Test**: Pytest with coverage reporting
3. **Docker Build**: Container validation
4. **Security Scan**: Trivy vulnerability scanning

**Quality Gates**:
- All tests must pass
- Code coverage threshold
- No linting errors
- Successful Docker build
- No critical vulnerabilities

#### Continuous Deployment (.github/workflows/cd.yml)

**Flow**:
1. Build and push Docker image to GHCR
2. Deploy to staging (automatic on main)
3. Deploy to production (manual approval or tag)
4. Notifications and status updates

**Environments**:
- **Staging**: Auto-deploy on main branch
- **Production**: Requires manual approval or version tag

### 6. Scheduled Retraining Workflow

**Location**: `.github/workflows/retrain.yml`

**Schedule**: Every Sunday at midnight UTC (configurable)

**Process**:
1. **Drift Detection**: Check for data distribution changes
2. **Retraining**: Train new model if drift detected
3. **Evaluation**: Compare with current production model
4. **Model Comparison**: Validate performance improvement
5. **Auto-deployment**: Deploy if metrics improve

**Manual Triggers**:
- Force retrain: Override drift detection
- Deploy after retrain: Automatic deployment flag

**Artifacts**:
- Trained model (.joblib)
- Evaluation metrics (JSON)
- MLflow experiment logs

### 7. Monitoring Stack

#### Prometheus
- Scrapes `/metrics` endpoint every 10s
- Stores time-series data
- Configured in `monitoring/prometheus.yml`

#### Grafana
- Pre-configured dashboards
- Data source: Prometheus
- Visualizations: Request rate, latency, errors
- Accessible at http://localhost:3000

#### Logging
- Structured logging throughout
- Request/response tracking
- Error logging with stack traces
- Correlation IDs for distributed tracing

## Code Quality and Standards

### Pre-commit Hooks

**Tools**:
- **Ruff**: Fast Python linter (replaces Flake8, pylint)
- **isort**: Import sorting
- **mypy**: Static type checking
- **Standard hooks**: Trailing whitespace, EOF fixer, YAML validation

**Configuration**: `.pre-commit-config.yaml`

### Testing Strategy

**Test Types**:
1. **Unit Tests**: Individual function testing
2. **Integration Tests**: API endpoint testing with TestClient
3. **Smoke Tests**: End-to-end training pipeline validation

**Coverage**:
- Minimum 80% code coverage
- Critical paths: 100% coverage
- Excludes: Test files, conftest.py

**Test Execution**:
```bash
pytest --cov=src/mlsys --cov-report=html
```

## Deployment Architecture

### Local Development

```bash
docker-compose up -d
```

**Services**:
- **API** (port 8000): Inference service
- **MLflow** (port 5000): Experiment tracking
- **Prometheus** (port 9090): Metrics collection
- **Grafana** (port 3000): Visualization

### Production Deployment

**Options**:
1. **Kubernetes**: Deployment + Service + HPA
2. **Cloud Run** (GCP): Serverless container
3. **Azure Container Apps**: Managed containers
4. **AWS ECS/Fargate**: Elastic container service

**Requirements**:
- Load balancer for HTTPS
- Persistent storage for models
- Environment variables for config
- Health check configuration

### Scaling Strategy

**Horizontal Scaling**:
- Stateless API design
- Load balancer distribution
- Auto-scaling based on CPU/requests

**Vertical Scaling**:
- Increase container resources
- Optimize model loading
- Batch prediction support

## Data and Model Versioning

### Dataset Versioning

**Recommended**: DVC (Data Version Control)

**Benefits**:
- Git-like versioning for data
- S3/GCS/Azure Blob storage
- Reproducible data pipelines
- Data lineage tracking

**Alternative**: Timestamped snapshots in cloud storage

### Model Versioning

**MLflow Model Registry**:
- Automatic version increment
- Stage management (None, Staging, Production)
- Model lineage and metadata
- Download by version or stage

**Artifact Storage**:
- Local: `artifacts/` directory
- MLflow: S3/GCS/Azure backend
- Git: Small models only (< 1MB)

## Security Considerations

### Container Security
- Non-root user in containers
- Minimal base images
- Regular security scans (Trivy)
- No secrets in images

### API Security
- Input validation (Pydantic)
- Rate limiting (recommended)
- HTTPS in production
- CORS configuration

### Secrets Management
- Environment variables
- GitHub Secrets for CI/CD
- Cloud secret managers (AWS Secrets Manager, GCP Secret Manager)
- Never commit secrets to Git

## Continuous Learning and Drift Detection

### Drift Detection

**Current Implementation**:
- Simple threshold-based detection
- Scheduled weekly checks

**Production Recommendations**:
- Statistical tests (Kolmogorov-Smirnov, Chi-square)
- Evidently AI or Alibi Detect libraries
- Population Stability Index (PSI)
- Model performance degradation tracking

### Retraining Strategy

**Triggers**:
1. **Scheduled**: Weekly retraining
2. **Drift-based**: Automatic when drift > threshold
3. **Performance-based**: When metrics degrade
4. **Manual**: On-demand via workflow dispatch

**Validation**:
- Compare new model with current production
- Require performance improvement
- A/B testing before full rollout

## Observability and Alerting

### Metrics to Monitor

**System Metrics**:
- CPU/Memory utilization
- Request rate and latency
- Error rate (4xx, 5xx)
- Container health

**ML Metrics**:
- Prediction latency
- Batch size distribution
- Model confidence distribution
- Feature drift

### Alerting Rules

**Critical**:
- API down (health check fails)
- Error rate > 5%
- Latency p99 > 2s

**Warning**:
- Error rate > 1%
- Latency p95 > 1s
- Model drift detected

## Best Practices Implemented

1. **Version Everything**: Code, data, models, configs
2. **Automate Everything**: Testing, deployment, retraining
3. **Monitor Everything**: Metrics, logs, traces
4. **Document Everything**: Code, APIs, decisions
5. **Test Everything**: Unit, integration, end-to-end
6. **Secure Everything**: Containers, API, data
7. **Reproducible Everything**: Seeds, versions, environments

## Future Enhancements

### Short Term
- [ ] Add A/B testing framework
- [ ] Implement feature store
- [ ] Add model explainability (SHAP)
- [ ] Create Grafana dashboard templates

### Medium Term
- [ ] Multi-model deployment
- [ ] Canary deployments
- [ ] Advanced drift detection
- [ ] Data validation (Great Expectations)

### Long Term
- [ ] Federated learning support
- [ ] AutoML integration
- [ ] Real-time feature engineering
- [ ] Model performance benchmarking

## Conclusion

This MLOps framework provides a production-ready foundation for machine learning systems with:

- **Automated Workflows**: From training to deployment
- **Quality Assurance**: Testing, linting, security scanning
- **Observability**: Metrics, logs, and monitoring
- **Continuous Learning**: Automatic retraining and drift detection
- **Scalability**: Docker, Kubernetes-ready
- **Maintainability**: Clean code, documentation, versioning

The framework follows industry best practices and is designed to scale from local development to production deployment across cloud platforms.

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
