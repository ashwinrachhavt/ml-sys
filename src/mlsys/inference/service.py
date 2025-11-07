"""FastAPI service exposing the trained lead-scoring model."""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from mlsys.config.paths import MODEL_PATH
from mlsys.inference.registry import ModelRegistry
from mlsys.inference.schemas import ScoreRequest, ScoreResponse

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(os.environ.get("MLSYS_MODEL_REGISTRY_PATH", "models/registry"))
MODEL_REGISTRY = ModelRegistry(REGISTRY_PATH)
_CURRENT_MODEL_PATH: Path | None = None

# Optional Prometheus integration
try:
    from prometheus_client import Counter, Histogram, make_asgi_app

    PROMETHEUS_AVAILABLE = True

    # Prometheus metrics
    PREDICTION_COUNT = Counter("mlsys_predictions_total", "Total number of predictions made", ["status"])
    PREDICTION_LATENCY = Histogram(
        "mlsys_prediction_duration_seconds",
        "Time spent processing predictions",
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    )
    PREDICTION_BATCH_SIZE = Histogram(
        "mlsys_prediction_batch_size", "Number of leads scored per request", buckets=[1, 5, 10, 25, 50, 100, 500]
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-client not available. Metrics endpoint will be disabled.")


def _resolve_model_path(path: Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path)
    else:
        try:
            candidate = MODEL_REGISTRY.get_best_model_path()
        except FileNotFoundError:
            candidate = MODEL_PATH
    if not candidate.exists():
        raise FileNotFoundError(f"Model artifact not found at {candidate}")
    return candidate


def load_model(path: Path | None = None):
    resolved_path = _resolve_model_path(path)
    logger.info("Loading model from %s", resolved_path)
    global _CURRENT_MODEL_PATH
    _CURRENT_MODEL_PATH = resolved_path
    return joblib.load(resolved_path)


@lru_cache(maxsize=1)
def get_model():
    return load_model()


def current_model_path() -> Path | None:
    return _CURRENT_MODEL_PATH


def create_app() -> FastAPI:
    """Create and configure the FastAPI application with monitoring."""
    app = FastAPI(
        title="Lead Scoring Inference API",
        description="Production ML system for lead scoring with monitoring and observability",
        version="0.1.0",
    )

    # Mount Prometheus metrics endpoint if available
    if PROMETHEUS_AVAILABLE:
        try:
            metrics_app = make_asgi_app()
            app.mount("/metrics", metrics_app)
            logger.info("Prometheus metrics endpoint enabled at /metrics")
        except Exception as e:
            logger.warning(f"Could not mount metrics endpoint: {e}")
    else:
        logger.info("Prometheus metrics disabled (prometheus-client not installed)")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Middleware to log all requests and track metrics."""
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Health check endpoint that verifies model is loaded."""
        try:
            model = get_model()
            path = current_model_path() or MODEL_PATH
            return {"status": "healthy", "model_loaded": model is not None, "model_path": str(path)}
        except Exception as exc:  # pragma: no cover - we want to log unexpected failures
            logger.exception("Model health check failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/score", response_model=ScoreResponse)
    def score(payload: ScoreRequest) -> ScoreResponse:
        """Score leads and return probability predictions."""
        start_time = time.time()

        try:
            model = get_model()
            if not payload.leads:
                if PROMETHEUS_AVAILABLE:
                    PREDICTION_COUNT.labels(status="error").inc()
                raise HTTPException(status_code=400, detail="No leads provided")

            # Track batch size
            batch_size = len(payload.leads)
            if PROMETHEUS_AVAILABLE:
                PREDICTION_BATCH_SIZE.observe(batch_size)

            df = pd.DataFrame([lead.model_dump() for lead in payload.leads])

            if "ALEXA_RANK" in df.columns and "ALEXA_RANK_LOG1P" not in df.columns:
                df["ALEXA_RANK_LOG1P"] = pd.Series(df["ALEXA_RANK"]).apply(
                    lambda x: None if pd.isna(x) else np.log1p(x)
                )

            proba = model.predict_proba(df)[:, 1]
            predictions = proba.tolist()

            # Record successful predictions
            if PROMETHEUS_AVAILABLE:
                PREDICTION_COUNT.labels(status="success").inc(batch_size)
                PREDICTION_LATENCY.observe(time.time() - start_time)

            path = current_model_path() or MODEL_PATH

            return ScoreResponse(
                probabilities=predictions,
                meta={
                    "model_path": str(path),
                    "batch_size": batch_size,
                    "processing_time_seconds": round(time.time() - start_time, 4),
                },
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error during scoring")
            if PROMETHEUS_AVAILABLE:
                PREDICTION_COUNT.labels(status="error").inc()
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}") from exc

    @app.get("/")
    def root() -> dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "service": "Lead Scoring Inference API",
            "version": "0.1.0",
            "endpoints": {"health": "/health", "score": "/score", "metrics": "/metrics", "docs": "/docs"},
        }

    return app


# Create app instance for uvicorn
app = create_app()
