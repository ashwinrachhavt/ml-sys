from __future__ import annotations

import time
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from prometheus_client import Counter, Histogram

from app.api.deps import get_predictor
from app.schemas.inference import InferencePayload, PredictionResponse
from app.schemas.model import ModelInfo
from app.serving.predictor import ModelPredictor

router = APIRouter(prefix="/v1", tags=["inference"])

prediction_counter = Counter("api_predictions_total", "Total predictions served")
prediction_latency = Histogram("api_prediction_latency_seconds", "Prediction latency")


def _build_model_info(predictor: ModelPredictor) -> ModelInfo:
    metadata = predictor.metadata
    features = metadata.feature_names if metadata else []
    return ModelInfo(
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        model_stage=predictor.model_stage,
        features=features,
        last_trained=datetime.utcnow().isoformat() if metadata else None,
        metrics=predictor.metrics,
        threshold=predictor.best_threshold,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: InferencePayload,
    predictor: Annotated[ModelPredictor, Depends(get_predictor)],
) -> PredictionResponse:
    start = time.perf_counter()

    try:
        result = predictor.predict(payload.data, threshold=payload.threshold)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        prediction_latency.observe(time.perf_counter() - start)
        prediction_counter.inc()

    return PredictionResponse(
        prediction=result.prediction,
        probability=result.probability,
        model_version=result.model_version,
    )


@router.get("/model/info", response_model=ModelInfo)
def model_info(
    predictor: Annotated[ModelPredictor, Depends(get_predictor)],
) -> ModelInfo:
    metadata = predictor.metadata
    features = metadata.feature_names if metadata else []
    return ModelInfo(
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        model_stage=predictor.model_stage,
        features=features,
        last_trained=None,
        metrics=predictor.metrics,
        threshold=predictor.best_threshold,
    )
