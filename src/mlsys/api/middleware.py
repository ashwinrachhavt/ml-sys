"""Optional middleware utilities (metrics, logging)."""

from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware


def install_default_middleware(app: FastAPI) -> None:
    """Apply opinionated default middleware choices."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


__all__ = ["install_default_middleware"]
