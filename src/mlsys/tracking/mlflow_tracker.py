"""MLflow implementation of the tracking abstraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

from .tracker import ExperimentTracker


class MLflowTracker(ExperimentTracker):
    """Track experiments using MLflow."""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name_prefix: str = "run",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name_prefix = run_name_prefix
        self._active_run_id: str | None = None

    def _ensure_experiment(self) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)  # type: ignore[attr-defined]
        mlflow.set_experiment(self.experiment_name)  # type: ignore[attr-defined]

    def _on_start(self, run_name: str, tags: dict[str, str] | None = None) -> None:
        self._ensure_experiment()
        run = mlflow.start_run(  # type: ignore[attr-defined]
            run_name=f"{self.run_name_prefix}-{run_name}"
        )
        self._active_run_id = run.info.run_id
        if tags:
            mlflow.set_tags(tags)  # type: ignore[attr-defined]

    def _on_end(self) -> None:
        if self._active_run_id is not None:
            mlflow.end_run()  # type: ignore[attr-defined]
            self._active_run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)  # type: ignore[attr-defined]

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)  # type: ignore[attr-defined]

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(path, artifact_path=artifact_path)  # type: ignore[attr-defined]

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        temp_path = Path(artifact_file)
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            mlflow.log_artifact(str(temp_path))  # type: ignore[attr-defined]
        finally:
            temp_path.unlink(missing_ok=True)

    def register_model(self, model_uri: str, name: str, stage: str | None = None) -> None:
        mlflow.register_model(model_uri=model_uri, name=name)  # type: ignore[attr-defined]
        if stage:
            client = mlflow.tracking.MlflowClient()  # type: ignore[attr-defined]
            latest = client.get_latest_versions(name, stages=["None"])
            for version in latest:
                client.transition_model_version_stage(
                    name=name,
                    version=version.version,
                    stage=stage,
                    archive_existing_versions=True,
                )


__all__ = ["MLflowTracker"]
