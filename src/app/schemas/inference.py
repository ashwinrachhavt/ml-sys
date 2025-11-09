from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

_INFERENCE_EXAMPLE: dict[str, Any] = {
    "data": {
        "CLOSEDATE": "2019-06-20",
        "MRR": 290,
        "ALEXA_RANK": 309343,
        "EMPLOYEE_RANGE": "201 to 1000",
        "INDUSTRY": "Other",
        "id": 199,
        "ACTIONS_CRM_CONTACTS": 278,
        "ACTIONS_CRM_COMPANIES": 0,
        "ACTIONS_CRM_DEALS": 34,
        "ACTIONS_EMAIL": 0,
        "USERS_CRM_CONTACTS": 2,
        "USERS_CRM_COMPANIES": 0,
        "USERS_CRM_DEALS": 2,
        "USERS_EMAIL": 0,
    },
    "threshold": 0.5,
}


class InferencePayload(BaseModel):
    """Generic payload for prediction requests."""

    data: dict[str, Any] = Field(..., description="Feature dictionary for the model")
    threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional decision threshold override",
    )

    model_config = {
        "json_schema_extra": {
            "example": _INFERENCE_EXAMPLE,
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None = None
    model_version: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
