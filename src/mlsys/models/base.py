"""Common model interfaces used by the training pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """Unified interface across estimators."""

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> BaseModel:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.Series:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Model does not implement probabilistic predictions")

    @property
    def name(self) -> str:
        return self.__class__.__name__


@dataclass
class ModelSpec:
    """Configuration used by the trainer to instantiate models."""

    name: str
    constructor: type[BaseModel]
    params: dict[str, Any]
    enabled: bool = True
