"""Model registry exports."""

from . import sklearn_models  # noqa: F401 - trigger registrations
from .base import BaseModel, ModelSpec
from .registry import ModelRegistry

__all__ = ["BaseModel", "ModelSpec", "ModelRegistry"]
