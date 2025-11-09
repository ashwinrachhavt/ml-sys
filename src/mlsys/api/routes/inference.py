"""Inference endpoints using PredictorService."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from mlsys.serving import PredictionResponse, PredictorService

router = APIRouter()


class PredictionRequest(BaseModel):
    payload: dict[str, Any]


class BatchPredictionRequest(BaseModel):
    payload: list[dict[str, Any]]


class PredictionResult(BaseModel):
    predictions: list[dict[str, Any]]


def get_predictor() -> PredictorService:  # pragma: no cover - injected at runtime
    raise RuntimeError("Predictor dependency not wired")


@router.post("/predict", response_model=PredictionResult, tags=["inference"])
def predict_one(
    request: PredictionRequest,
    predictor: Annotated[PredictorService, Depends(get_predictor)],
) -> PredictionResult:
    response: PredictionResponse = predictor.predict_one(request.payload)
    return PredictionResult(predictions=response.predictions)


@router.post("/predict/batch", response_model=PredictionResult, tags=["inference"])
def predict_batch(
    request: BatchPredictionRequest,
    predictor: Annotated[PredictorService, Depends(get_predictor)],
) -> PredictionResult:
    response = predictor.predict_batch(request.payload)
    return PredictionResult(predictions=response.predictions)


__all__ = ["router", "get_predictor"]
