"""Training orchestration tying loaders, features and models together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from mlsys.config import Settings
from mlsys.data import DatasetLoader
from mlsys.evaluation import Evaluator
from mlsys.features import FeaturePipeline, TransformerRegistry
from mlsys.models import ModelRegistry
from mlsys.tracking import ExperimentTracker, NullTracker

from .callbacks import NullCallback, TrainingCallback, TrainingContext


@dataclass
class ModelComparison:
    """Summary for each trained model."""

    name: str
    cv_score: float
    best_params: dict[str, Any]


@dataclass
class TrainingResult:
    """Overall training output."""

    best_model_name: str
    best_threshold: float | None
    metrics: dict[str, float]
    comparisons: list[ModelComparison]
    metadata: dict[str, Any]


class Trainer:
    """High-level orchestration entrypoint used by CLI and API."""

    def __init__(
        self,
        settings: Settings,
        tracker: ExperimentTracker | None = None,
        callbacks: list[TrainingCallback] | None = None,
    ) -> None:
        self.settings = settings
        self.tracker = tracker or NullTracker()
        self.callbacks = callbacks or [NullCallback()]

    def _build_transformers(self) -> list[Any]:
        transformers = []
        for cfg in self.settings.features.transformers:
            params = {k: v for k, v in cfg.model_dump(exclude={"type"}).items() if v is not None}
            transformer = TransformerRegistry.create(cfg.type, **params)
            transformers.append(transformer)
        return transformers

    def train(self) -> TrainingResult:
        loader, datasets = DatasetLoader.from_config(
            loader_type=self.settings.data.loader_type,
            sources=self.settings.data_paths(),
        )

        pipeline = FeaturePipeline(transformers=self._build_transformers())
        features, target = pipeline.build(
            datasets,
            id_column=self.settings.data.id_column,
            target_column=self.settings.data.target_column,
        )

        categorical_features = [
            cfg.columns for cfg in self.settings.features.transformers if getattr(cfg, "encoding", None)
        ]
        categorical_flat = [c for cols in categorical_features if cols for c in cols]
        datetime_columns = [
            c
            for cfg in self.settings.features.transformers
            if getattr(cfg, "reference_date", None)
            for c in (cfg.columns or [])
        ]
        reference_date = next(
            (cfg.reference_date for cfg in self.settings.features.transformers if getattr(cfg, "reference_date", None)),
            None,
        )

        splits = pipeline.split(
            features,
            target,
            test_size=self.settings.training.test_size,
            val_size=self.settings.training.val_size,
            random_state=self.settings.training.random_state,
            stratify=self.settings.training.stratify,
            categorical_features=categorical_flat,
            datetime_columns=datetime_columns,
            reference_date=reference_date,
        )

        context = TrainingContext(
            x_train=splits.x_train,
            y_train=splits.y_train,
            feature_metadata=splits.metadata.to_dict(),
        )

        for callback in self.callbacks:
            callback.on_train_start(context)

        evaluator = Evaluator(
            metric_names=self.settings.evaluation.metrics,
            threshold_metric=self.settings.evaluation.threshold_metric,
        )

        primary_metric = self.settings.evaluation.primary_metric
        scoring_map = {
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "pr_auc": "average_precision",
        }
        scoring = scoring_map.get(primary_metric, "roc_auc")

        comparisons: list[ModelComparison] = []
        best_model_name: str | None = None
        best_metric = -float("inf")
        best_metrics: dict[str, float] = {}
        best_threshold: float | None = None
        best_model_obj = None

        for model_cfg in self.settings.training.models:
            if not model_cfg.enabled:
                continue

            grid = model_cfg.param_grid or {}
            parameter_grid = ParameterGrid(grid) if grid else [{}]

            best_cv_score = -float("inf")
            best_params: dict[str, Any] = {}

            for params in parameter_grid:
                # instantiate underlying estimator for CV using clone to avoid state bleed
                model = ModelRegistry.create(model_cfg.type, **params)
                estimator = getattr(model, "estimator", model)
                kfold = StratifiedKFold(
                    n_splits=self.settings.training.cv_folds,
                    shuffle=True,
                    random_state=self.settings.training.random_state,
                )
                cv_scores = []
                for train_idx, val_idx in kfold.split(splits.x_train, splits.y_train):
                    x_train_fold = splits.x_train.iloc[train_idx]
                    y_train_fold = splits.y_train.iloc[train_idx]
                    x_val_fold = splits.x_train.iloc[val_idx]
                    y_val_fold = splits.y_train.iloc[val_idx]
                    estimator_fold = clone(estimator)
                    estimator_fold.fit(x_train_fold, y_train_fold)
                    if scoring == "average_precision":
                        scores = estimator_fold.predict_proba(x_val_fold)[:, 1]
                        metric_value = evaluator.metric_fns["pr_auc"](y_val_fold, scores)
                    elif scoring == "roc_auc":
                        scores = estimator_fold.predict_proba(x_val_fold)[:, 1]
                        metric_value = evaluator.metric_fns["roc_auc"](y_val_fold, scores)
                    else:
                        preds = estimator_fold.predict(x_val_fold)
                        metric_value = evaluator.metric_fns[primary_metric](y_val_fold, preds)
                    cv_scores.append(metric_value)
                cv_score = float(np.mean(cv_scores))
                if cv_score > best_cv_score:
                    best_cv_score = cv_score
                    best_params = params

            comparisons.append(
                ModelComparison(
                    name=model_cfg.name,
                    cv_score=best_cv_score,
                    best_params=best_params,
                )
            )

            model = ModelRegistry.create(model_cfg.type, **best_params)
            model.fit(splits.x_train, splits.y_train)
            probs = model.predict_proba(splits.x_val)
            positive_scores = probs.iloc[:, 1] if isinstance(probs, pd.DataFrame) else probs

            result = evaluator.evaluate(
                splits.y_val,
                positive_scores.to_numpy() if isinstance(positive_scores, pd.Series) else positive_scores,
                thresholds=[i / 100 for i in range(10, 90, 5)],
            )

            metrics = result.metrics
            for callback in self.callbacks:
                callback.on_model_trained(context, model_cfg.name, metrics)

            with self.tracker.start_run(run_name=model_cfg.name, tags={"model": model_cfg.type}):
                self.tracker.log_params(best_params)
                self.tracker.log_metrics(metrics)
                if pipeline.metadata:
                    self.tracker.log_dict(pipeline.metadata.to_dict(), "feature_metadata.json")

            metric_value = metrics.get(primary_metric, 0.0)
            if metric_value > best_metric:
                best_metric = metric_value
                best_model_name = model_cfg.name
                best_metrics = metrics
                best_threshold = result.threshold
                best_model_obj = model

        for callback in self.callbacks:
            callback.on_train_end(context)

        if best_model_name is None:
            raise RuntimeError("No model was successfully trained")

        if self.settings.serving.local_model_path and best_model_obj is not None:
            output_path = self.settings.resolve_path(self.settings.serving.local_model_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            artifact = {
                "model": best_model_obj,
                "transformers": pipeline.transformers,
                "metadata": splits.metadata,
                "id_column": self.settings.data.id_column,
                "target_column": self.settings.data.target_column,
            }
            joblib.dump(artifact, output_path)

        return TrainingResult(
            best_model_name=best_model_name,
            best_threshold=best_threshold,
            metrics=best_metrics,
            comparisons=comparisons,
            metadata=pipeline.metadata.to_dict() if pipeline.metadata else {},
        )


__all__ = ["Trainer", "TrainingResult", "ModelComparison"]
