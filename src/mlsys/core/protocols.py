"""Protocols describing extension points in the framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class DataLoader(Protocol):
    """Contract for loading raw datasets into DataFrames."""

    def load(self, *, data_dir: Path | None = None) -> pd.DataFrame:
        """Return a pandas DataFrame containing raw records."""
        ...


class FeatureTransformer(Protocol):
    """Contract for feature engineering components."""

    def fit(self, data: pd.DataFrame, target: pd.Series | None = None) -> FeatureTransformer: ...

    def transform(self, data: pd.DataFrame) -> pd.DataFrame: ...

    def fit_transform(
        self,
        data: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> pd.DataFrame: ...


class Model(Protocol):
    """Contract for estimator wrappers used in training and inference."""

    @property
    def name(self) -> str: ...

    def train(
        self,
        features: pd.DataFrame | np.ndarray,
        target: pd.Series | np.ndarray,
        validation_features: pd.DataFrame | np.ndarray | None = None,
        validation_target: pd.Series | np.ndarray | None = None,
    ) -> dict[str, float]: ...

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray: ...

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


class MetricsTracker(Protocol):
    """Contract for experiment tracking backends (e.g., MLflow)."""

    def start_run(self, *, run_name: str | None = None) -> None: ...

    def end_run(self) -> None: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None: ...

    def log_artifact(self, artifact_path: Path) -> None: ...

    def log_model(self, model: Any, artifact_path: str) -> None: ...
