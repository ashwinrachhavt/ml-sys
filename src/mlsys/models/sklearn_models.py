"""Concrete sklearn-based models implementing the BaseModel interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .base import BaseModel
from .registry import ModelRegistry


@dataclass
class SklearnClassifier(BaseModel):
    """Adapter over scikit-learn classifiers."""

    estimator: ClassifierMixin
    name_override: str | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> SklearnClassifier:
        self.estimator.fit(x, y, **kwargs)
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        return pd.Series(self.estimator.predict(x), index=x.index)

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        proba = self.estimator.predict_proba(x)
        return pd.DataFrame(proba, index=x.index, columns=getattr(self.estimator, "classes_", None))

    @property
    def name(self) -> str:
        return self.name_override or self.estimator.__class__.__name__


@ModelRegistry.register("sklearn.logistic_regression")
class LogisticRegressionModel(SklearnClassifier):
    """Logistic regression convenience wrapper."""

    def __init__(self, **kwargs: Any):
        estimator = LogisticRegression(**kwargs)
        super().__init__(estimator=estimator, name_override="logistic_regression")


@ModelRegistry.register("sklearn.random_forest")
class RandomForestModel(SklearnClassifier):
    """Random forest classifier wrapper."""

    def __init__(self, **kwargs: Any):
        estimator = RandomForestClassifier(**kwargs)
        super().__init__(estimator=estimator, name_override="random_forest")


__all__ = ["LogisticRegressionModel", "RandomForestModel", "SklearnClassifier"]
