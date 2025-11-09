"""Basic health endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", tags=["health"])
def readiness() -> dict[str, str]:
    return {"status": "ready"}


__all__ = ["router"]
