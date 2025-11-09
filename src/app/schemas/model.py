from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    model_name: str
    model_version: str | None = None
    model_stage: str
    features: list[str]
    last_trained: str | None
    metrics: dict[str, float] = Field(default_factory=dict)
    threshold: float | None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    uptime_seconds: float
