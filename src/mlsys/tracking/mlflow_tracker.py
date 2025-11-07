"""Thin wrapper around MLflow to make logging optional."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlsys.core.config import TrackingConfig

try:  # pragma: no cover - optional dependency
    import mlflow
    import mlflow.sklearn

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False


class MLflowTracker:
    """Handles MLflow lifecycle while allowing graceful opt-out."""

    def __init__(self, config: TrackingConfig):
        self._config = config
        self._active = False
        self._available = _MLFLOW_AVAILABLE

    @property
    def active(self) -> bool:
        return self._active

    @property
    def available(self) -> bool:
        return self._available

    def start_run(self, *, run_name: str | None = None) -> bool:
        if not self._available:
            return False
        try:
            mlflow.set_tracking_uri(self._config.tracking_uri)
            mlflow.set_experiment(self._config.experiment_name)
            mlflow.start_run(run_name=run_name)
            self._active = True
        except Exception as exc:  # pragma: no cover - defensive logging
            warnings.warn(
                f"Unable to initialise MLflow tracking ({exc}). Continuing without tracking.",
                stacklevel=2,
            )
            self._active = False
        return self._active

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self._active:
            mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        if self._active:
            mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: Path) -> None:
        if self._active and self._config.log_artifacts:
            mlflow.log_artifact(str(path))

    def log_model(self, model: Any, artifact_path: str, *, registered_model_name: str | None = None) -> None:
        if self._active and self._config.log_artifacts:
            mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_model_name)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if self._active:
            mlflow.set_tags(dict(tags))

    def end_run(self) -> None:
        if self._active:
            mlflow.end_run()
            self._active = False


__all__ = ["MLflowTracker"]
