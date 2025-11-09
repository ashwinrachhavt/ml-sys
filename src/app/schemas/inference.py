from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class InferencePayload(BaseModel):
    """Generic payload for prediction requests."""

    data: dict[str, Any] = Field(..., description="Feature dictionary for the model")
    threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional decision threshold override",
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None = None
    model_version: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
