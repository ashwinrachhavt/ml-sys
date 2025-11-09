from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover
    SMOTE = None  # type: ignore[assignment]
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV

try:
    import mlflow
    import mlflow.sklearn  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback when mlflow is optional
    mlflow = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    import mlflow  # noqa: F401

from app.core.config import load_config
from app.data.base_loader import BaseDataLoader
from app.features.pipeline import FeatureMatrix, FeatureMetadata, FeaturePipeline
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
    best_threshold: float | None
    feature_importance: pd.DataFrame | None
    metadata: FeatureMetadata


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

        mlflow_run = None
        if self.enable_mlflow:
            if mlflow is None:
                raise RuntimeError("MLflow is not installed but enable_mlflow=True")

            experiment = cfg.get("mlflow", {}).get("experiment_name", "default")
            mlflow.set_experiment(experiment)
            mlflow_run = mlflow.start_run(run_name=cfg.get("mlflow", {}).get("run_name_prefix", "training"))

        try:
            datasets = self.loader.load(config=cfg, config_path=config_path)
            feature_matrix = self.feature_pipeline.build_feature_matrix(datasets)

            x_fit, y_fit = self._prepare_training_inputs(cfg, feature_matrix)

            best_spec, grid_search = self._train_models(cfg, feature_matrix, x_fit, y_fit)

            estimator: ClassifierMixin = grid_search.best_estimator_
            metrics = self._collect_metrics(cfg, estimator, feature_matrix)
            best_threshold = metrics.pop("best_threshold", None)

            feature_importance = self._extract_feature_importance(estimator, feature_matrix.feature_names)

            if mlflow_run is not None:
                self._log_mlflow(
                    best_spec,
                    grid_search,
                    metrics,
                    feature_matrix,
                    best_threshold,
                    feature_importance,
                )

            return TrainingResult(
                best_model_name=best_spec.name,
                best_estimator=estimator,
                best_params=dict(grid_search.best_params_),
                metrics=metrics,
                feature_names=feature_matrix.feature_names,
                best_threshold=best_threshold,
                feature_importance=feature_importance,
                metadata=feature_matrix.metadata,
            )
        finally:
            if mlflow_run is not None:
                mlflow.end_run()

    def _train_models(
        self,
        cfg: Mapping[str, Any],
        feature_matrix: FeatureMatrix,
        x_train_fit: pd.DataFrame,
        y_train_fit: pd.Series,
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
            grid.fit(x_train_fit, y_train_fit)

            mean_score = grid.best_score_
            if mean_score > best_score:
                best_score = mean_score
                best_spec = spec
                best_grid = grid

        if best_spec is None or best_grid is None:
            raise RuntimeError("Training failed to produce a model")

        return best_spec, best_grid

    def _log_mlflow(
        self,
        spec: ModelSpec,
        grid: GridSearchCV,
        metrics: Mapping[str, Any],
        feature_matrix: FeatureMatrix,
        best_threshold: float | None,
        feature_importance: pd.DataFrame | None,
    ) -> None:
        if mlflow is None:
            return

        mlflow.log_params({f"{spec.name}_{k}": v for k, v in grid.best_params_.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, int | float)})
        mlflow.log_text("\n".join(feature_matrix.feature_names), artifact_file="feature_names.txt")
        mlflow.log_text(
            json.dumps(feature_matrix.metadata.to_dict()),
            artifact_file="feature_metadata.json",
        )
        if best_threshold is not None:
            mlflow.log_param("best_threshold", best_threshold)

        if feature_importance is not None:
            from io import StringIO

            buffer = StringIO()
            feature_importance.to_csv(buffer, index=False)
            mlflow.log_text(buffer.getvalue(), artifact_file="feature_importance.csv")

        mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")

    def _prepare_training_inputs(
        self,
        cfg: Mapping[str, Any],
        feature_matrix: FeatureMatrix,
    ) -> tuple[pd.DataFrame, pd.Series]:
        training_cfg = cfg.get("training", {})
        imbalance_cfg = training_cfg.get("handle_imbalance", {})
        method = imbalance_cfg.get("method")

        x_train = feature_matrix.x_train
        y_train = feature_matrix.y_train

        if method == "smote":
            if SMOTE is None:
                raise RuntimeError("imblearn is required for SMOTE sampling but is not installed")
            smote_kwargs = {k: v for k, v in imbalance_cfg.items() if k != "method"}
            sampler = SMOTE(**smote_kwargs)
            x_res, y_res = sampler.fit_resample(x_train, y_train)
            return x_res, y_res

        return x_train, y_train

    def _collect_metrics(
        self,
        cfg: Mapping[str, Any],
        estimator: ClassifierMixin,
        feature_matrix: FeatureMatrix,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        metrics.update(self._evaluate_split(estimator, feature_matrix.x_train, feature_matrix.y_train, prefix="train"))
        metrics.update(self._evaluate_split(estimator, feature_matrix.x_val, feature_matrix.y_val, prefix="val"))
        metrics.update(self._evaluate_split(estimator, feature_matrix.x_test, feature_matrix.y_test, prefix="test"))

        best_threshold = self._find_optimal_threshold(cfg, estimator, feature_matrix.x_val, feature_matrix.y_val)
        metrics["best_threshold"] = best_threshold

        if best_threshold is not None:
            metrics.update(
                self._evaluate_with_threshold(
                    estimator,
                    feature_matrix.x_val,
                    feature_matrix.y_val,
                    best_threshold,
                    prefix="val_opt",
                )
            )
            metrics.update(
                self._evaluate_with_threshold(
                    estimator,
                    feature_matrix.x_test,
                    feature_matrix.y_test,
                    best_threshold,
                    prefix="test_opt",
                )
            )

        return metrics

    def _evaluate_split(
        self,
        estimator: ClassifierMixin,
        x: pd.DataFrame,
        y: pd.Series,
        *,
        prefix: str,
    ) -> dict[str, float]:
        y_pred = estimator.predict(x)

        y_proba: np.ndarray | None = None
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(x)[:, 1]

        metrics = classification_metrics(
            y.values,
            y_pred,
            y_proba,
        )

        return {f"{prefix}_{key}": value for key, value in metrics.items()}

    def _evaluate_with_threshold(
        self,
        estimator: ClassifierMixin,
        x: pd.DataFrame,
        y: pd.Series,
        threshold: float,
        *,
        prefix: str,
    ) -> dict[str, float]:
        if not hasattr(estimator, "predict_proba"):
            return {}

        proba = estimator.predict_proba(x)[:, 1]
        preds = (proba >= threshold).astype(int)
        metrics = classification_metrics(y.values, preds, proba)
        return {f"{prefix}_{key}": value for key, value in metrics.items()}

    def _find_optimal_threshold(
        self,
        cfg: Mapping[str, Any],
        estimator: ClassifierMixin,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float | None:
        if not hasattr(estimator, "predict_proba"):
            return None

        evaluation_cfg = cfg.get("evaluation", {})
        metric_name = evaluation_cfg.get("threshold_metric", "f1")
        thresholds = np.linspace(0.1, 0.9, 50)

        proba = estimator.predict_proba(x_val)[:, 1]

        best_score = -np.inf
        best_threshold: float | None = None

        for threshold in thresholds:
            preds = (proba >= threshold).astype(int)

            metrics = classification_metrics(y_val.values, preds, proba)

            if metric_name == "precision":
                score = metrics["precision"]
            elif metric_name == "recall":
                score = metrics["recall"]
            else:
                score = metrics["f1"]

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def _extract_feature_importance(
        self,
        estimator: ClassifierMixin,
        feature_names: list[str],
    ) -> pd.DataFrame | None:
        values: np.ndarray | None = None

        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if isinstance(coef, np.ndarray):
                if coef.ndim == 1:
                    values = np.abs(coef)
                else:
                    values = np.abs(coef[0])

        if values is None:
            return None

        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": values,
            }
        ).sort_values("importance", ascending=False)

        return importance
