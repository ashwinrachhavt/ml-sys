from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn  # type: ignore[attr-defined]
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

from app.core.config import load_config
from app.data.base_loader import BaseDataLoader
from app.features.pipeline import FeatureMatrix, FeaturePipeline
from app.ml.evaluation import classification_metrics


@dataclass
class ModelSpec:
    name: str
    estimator: ClassifierMixin
    param_grid: Mapping[str, Iterable[Any]] | None = None


@dataclass
class TrainingResult:
    best_model_name: str
    best_estimator: ClassifierMixin
    best_params: Mapping[str, Any]
    metrics: Mapping[str, Any]
    feature_names: list[str]


class TrainingPipeline:
    """Orchestrate end-to-end model training with configurable components."""

    def __init__(
        self,
        loader: BaseDataLoader,
        feature_pipeline: FeaturePipeline,
        model_specs: list[ModelSpec],
        *,
        enable_mlflow: bool = False,
    ) -> None:
        self.loader = loader
        self.feature_pipeline = feature_pipeline
        self.model_specs = model_specs
        self.enable_mlflow = enable_mlflow

    def run(
        self,
        *,
        config_path: Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> TrainingResult:
        cfg = load_config(config_path) if config is None else config

        mlflow_run: mlflow.ActiveRun | None = None
        if self.enable_mlflow:
            experiment = cfg.get("mlflow", {}).get("experiment_name", "default")
            mlflow.set_experiment(experiment)
            mlflow_run = mlflow.start_run(run_name=cfg.get("mlflow", {}).get("run_name_prefix", "training"))

        try:
            datasets = self.loader.load(config=cfg, config_path=config_path)
            feature_matrix = self.feature_pipeline.build_feature_matrix(datasets)

            best_spec, grid_search = self._train_models(cfg, feature_matrix)
            metrics = self._evaluate(grid_search, feature_matrix)

            if mlflow_run is not None:
                self._log_mlflow(best_spec, grid_search, metrics, feature_matrix)

            return TrainingResult(
                best_model_name=best_spec.name,
                best_estimator=grid_search.best_estimator_,
                best_params=dict(grid_search.best_params_),
                metrics=metrics,
                feature_names=feature_matrix.feature_names,
            )
        finally:
            if mlflow_run is not None:
                mlflow.end_run()

    def _train_models(
        self,
        cfg: Mapping[str, Any],
        feature_matrix: FeatureMatrix,
    ) -> tuple[ModelSpec, GridSearchCV]:
        evaluation_cfg = cfg.get("evaluation", {})
        primary_metric = evaluation_cfg.get("primary_metric", "roc_auc")
        cv_folds = cfg.get("training", {}).get("cv_folds", 5)

        best_spec: ModelSpec | None = None
        best_grid: GridSearchCV | None = None
        best_score = -np.inf

        for spec in self.model_specs:
            estimator = spec.estimator
            param_grid = spec.param_grid or {}
            grid = GridSearchCV(
                estimator,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=primary_metric,
                n_jobs=-1,
            )
            grid.fit(feature_matrix.x_train, feature_matrix.y_train)

            mean_score = grid.best_score_
            if mean_score > best_score:
                best_score = mean_score
                best_spec = spec
                best_grid = grid

        if best_spec is None or best_grid is None:
            raise RuntimeError("Training failed to produce a model")

        return best_spec, best_grid

    def _evaluate(self, grid: GridSearchCV, feature_matrix: FeatureMatrix) -> Mapping[str, Any]:
        estimator: BaseEstimator = grid.best_estimator_
        y_val_pred = estimator.predict(feature_matrix.x_val)

        y_val_proba: np.ndarray | None = None
        if hasattr(estimator, "predict_proba"):
            y_val_proba = estimator.predict_proba(feature_matrix.x_val)[:, 1]

        return classification_metrics(
            feature_matrix.y_val.values,
            y_val_pred,
            y_val_proba,
        )

    def _log_mlflow(
        self,
        spec: ModelSpec,
        grid: GridSearchCV,
        metrics: Mapping[str, Any],
        feature_matrix: FeatureMatrix,
    ) -> None:
        mlflow.log_params({f"{spec.name}_{k}": v for k, v in grid.best_params_.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        mlflow.log_text(
            "\n".join(feature_matrix.feature_names),
            artifact_file="feature_names.txt",
        )
        mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")
