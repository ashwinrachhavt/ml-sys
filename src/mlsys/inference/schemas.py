"""Pydantic schemas for inference requests/responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LeadFeatures(BaseModel):
    """Incoming feature payload matching training inputs."""

    ALEXA_RANK: float | None = Field(None, ge=0)
    EMPLOYEE_RANGE: str | None = None
    INDUSTRY: str | None = None
    ALEXA_RANK_LOG1P: float | None = None
    # Usage features kept flexible: allow extra keys for actions/users totals

    model_config = ConfigDict(extra="allow")


class ScoreRequest(BaseModel):
    leads: list[LeadFeatures]


class ScoreResponse(BaseModel):
    probabilities: list[float]
    meta: dict[str, str | float | int]
