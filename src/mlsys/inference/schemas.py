"""Pydantic schemas for inference requests/responses."""
from __future__ import annotations

from typing import Dict, Optional, Union

from pydantic import BaseModel, Field, ConfigDict


class LeadFeatures(BaseModel):
    """Incoming feature payload matching training inputs."""

    ALEXA_RANK: Optional[float] = Field(None, ge=0)
    EMPLOYEE_RANGE: Optional[str] = None
    INDUSTRY: Optional[str] = None
    ALEXA_RANK_LOG1P: Optional[float] = None
    # Usage features kept flexible: allow extra keys for actions/users totals

    model_config = ConfigDict(extra="allow")


class ScoreRequest(BaseModel):
    leads: list[LeadFeatures]


class ScoreResponse(BaseModel):
    probabilities: list[float]
    meta: Dict[str, Union[str, float]]
