"""Unified FastAPI-first MLOps system package."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - best effort metadata lookup
    __version__ = version("mlsys")
except PackageNotFoundError:  # pragma: no cover - local development fallback
    __version__ = "0.0.0"

from . import models  # noqa: F401

__all__ = ["__version__"]
