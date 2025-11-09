"""Training orchestration tying loaders, features and models together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from mlsys.config import Settings
from mlsys.data import DatasetLoader
from mlsys.evaluation import Evaluator
from mlsys.features import FeaturePipeline, SplitData, TransformerRegistry
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


@dataclass
class _BestModelState:
    """Internal container describing the strongest performing model."""

    name: str
    metrics: dict[str, float]
    threshold: float | None
    model: Any


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
        datasets = self._load_datasets()
        pipeline = self._build_pipeline()
        features, target = pipeline.build(
            datasets,
            id_column=self.settings.data.id_column,
            target_column=self.settings.data.target_column,
        )

        categorical_features, datetime_columns, reference_date = self._feature_metadata_from_settings()
        splits = pipeline.split(
            features,
            target,
            test_size=self.settings.training.test_size,
            val_size=self.settings.training.val_size,
            random_state=self.settings.training.random_state,
            stratify=self.settings.training.stratify,
            categorical_features=categorical_features,
            datetime_columns=datetime_columns,
            reference_date=reference_date,
        )

        context = self._create_training_context(splits)
        self._notify_start(context)

        evaluator = self._build_evaluator()
        scoring = self._resolve_scoring(self.settings.evaluation.primary_metric)

        comparisons, best_state = self._train_models(
            splits=splits,
            pipeline=pipeline,
            evaluator=evaluator,
            context=context,
            scoring=scoring,
        )

        self._notify_end(context)

        if best_state is None:
            raise RuntimeError("No model was successfully trained")

        self._persist_best_model(best_state, pipeline, splits)

        return TrainingResult(
            best_model_name=best_state.name,
            best_threshold=best_state.threshold,
            metrics=best_state.metrics,
            comparisons=comparisons,
            metadata=pipeline.metadata.to_dict() if pipeline.metadata else {},
        )

    def _load_datasets(self) -> dict[str, pd.DataFrame]:
        _, datasets = DatasetLoader.from_config(
            loader_type=self.settings.data.loader_type,
            sources=self.settings.data_paths(),
        )
        return datasets

    def _build_pipeline(self) -> FeaturePipeline:
        return FeaturePipeline(transformers=self._build_transformers())

    def _feature_metadata_from_settings(self) -> tuple[list[str], list[str], str | None]:
        categorical_sources = [
            cfg.columns for cfg in self.settings.features.transformers if getattr(cfg, "encoding", None)
        ]
        categorical_features = [column for columns in categorical_sources if columns for column in columns]

        datetime_columns = [
            column
            for cfg in self.settings.features.transformers
            if getattr(cfg, "reference_date", None)
            for column in (cfg.columns or [])
        ]

        reference_date = next(
            (cfg.reference_date for cfg in self.settings.features.transformers if getattr(cfg, "reference_date", None)),
            None,
        )
        return categorical_features, datetime_columns, reference_date

    def _create_training_context(self, splits: SplitData) -> TrainingContext:
        return TrainingContext(
            x_train=splits.x_train,
            y_train=splits.y_train,
            feature_metadata=splits.metadata.to_dict(),
        )

    def _notify_start(self, context: TrainingContext) -> None:
        for callback in self.callbacks:
            callback.on_train_start(context)

    def _notify_end(self, context: TrainingContext) -> None:
        for callback in self.callbacks:
            callback.on_train_end(context)

    def _build_evaluator(self) -> Evaluator:
        return Evaluator(
            metric_names=self.settings.evaluation.metrics,
            threshold_metric=self.settings.evaluation.threshold_metric,
        )

    def _resolve_scoring(self, primary_metric: str) -> str:
        scoring_map = {
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "pr_auc": "average_precision",
        }
        return scoring_map.get(primary_metric, "roc_auc")

    def _train_models(
        self,
        *,
        splits: SplitData,
        pipeline: FeaturePipeline,
        evaluator: Evaluator,
        context: TrainingContext,
        scoring: str,
    ) -> tuple[list[ModelComparison], _BestModelState | None]:
        comparisons: list[ModelComparison] = []
        best_state: _BestModelState | None = None
        best_metric = -float("inf")
        primary_metric = self.settings.evaluation.primary_metric

        for model_cfg in self.settings.training.models:
            if not model_cfg.enabled:
                continue

            comparison, state = self._train_single_model(
                model_cfg=model_cfg,
                splits=splits,
                evaluator=evaluator,
                context=context,
                pipeline=pipeline,
                scoring=scoring,
            )
            comparisons.append(comparison)

            metric_value = state.metrics.get(primary_metric, 0.0)
            if metric_value > best_metric:
                best_metric = metric_value
                best_state = state

        return comparisons, best_state

    def _train_single_model(
        self,
        *,
        model_cfg: Any,
        splits: SplitData,
        evaluator: Evaluator,
        context: TrainingContext,
        pipeline: FeaturePipeline,
        scoring: str,
    ) -> tuple[ModelComparison, _BestModelState]:
        best_cv_score, best_params = self._select_best_params(model_cfg, splits, evaluator, scoring)

        model = ModelRegistry.create(model_cfg.type, **best_params)
        model.fit(splits.x_train, splits.y_train)

        scores = self._extract_positive_scores(model.predict_proba(splits.x_val))
        result = evaluator.evaluate(
            splits.y_val,
            scores,
            thresholds=[i / 100 for i in range(10, 90, 5)],
        )

        metrics = result.metrics
        for callback in self.callbacks:
            callback.on_model_trained(context, model_cfg.name, metrics)

        self._record_run(model_cfg, best_params, metrics, pipeline)

        state = _BestModelState(
            name=model_cfg.name,
            metrics=metrics,
            threshold=result.threshold,
            model=model,
        )

        comparison = ModelComparison(
            name=model_cfg.name,
            cv_score=best_cv_score,
            best_params=best_params,
        )
        return comparison, state

    def _select_best_params(
        self,
        model_cfg: Any,
        splits: SplitData,
        evaluator: Evaluator,
        scoring: str,
    ) -> tuple[float, dict[str, Any]]:
        grid = model_cfg.param_grid or {}
        parameter_grid: Iterable[dict[str, Any]] = ParameterGrid(grid) if grid else [{}]

        best_cv_score = -float("inf")
        best_params: dict[str, Any] = {}

        for params in parameter_grid:
            model = ModelRegistry.create(model_cfg.type, **params)
            estimator = getattr(model, "estimator", model)
            cv_score = self._cross_validate(estimator, splits, evaluator, scoring)
            if cv_score > best_cv_score:
                best_cv_score = cv_score
                best_params = params

        return best_cv_score, best_params

    def _cross_validate(
        self,
        estimator: Any,
        splits: SplitData,
        evaluator: Evaluator,
        scoring: str,
    ) -> float:
        kfold = StratifiedKFold(
            n_splits=self.settings.training.cv_folds,
            shuffle=True,
            random_state=self.settings.training.random_state,
        )

        scores = []
        for train_idx, val_idx in kfold.split(splits.x_train, splits.y_train):
            x_train_fold = splits.x_train.iloc[train_idx]
            y_train_fold = splits.y_train.iloc[train_idx]
            x_val_fold = splits.x_train.iloc[val_idx]
            y_val_fold = splits.y_train.iloc[val_idx]

            estimator_fold = clone(estimator)
            estimator_fold.fit(x_train_fold, y_train_fold)
            score = self._score_estimator(estimator_fold, x_val_fold, y_val_fold, evaluator, scoring)
            scores.append(score)

        return float(np.mean(scores))

    def _score_estimator(
        self,
        estimator: Any,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        evaluator: Evaluator,
        scoring: str,
    ) -> float:
        primary_metric = self.settings.evaluation.primary_metric
        if scoring in {"average_precision", "roc_auc"}:
            probabilities = estimator.predict_proba(x_val)
            positive = self._extract_positive_scores(probabilities)
            metric_key = "pr_auc" if scoring == "average_precision" else "roc_auc"
            metric_fn = evaluator.metric_fns[metric_key]
            return float(metric_fn(y_val, positive))

        predictions = estimator.predict(x_val)
        metric_fn = evaluator.metric_fns[primary_metric]
        return float(metric_fn(y_val, predictions))

    def _extract_positive_scores(self, probabilities: Any) -> np.ndarray:
        if isinstance(probabilities, pd.DataFrame):
            return probabilities.iloc[:, 1].to_numpy()
        if isinstance(probabilities, pd.Series):
            return probabilities.to_numpy()

        array = np.asarray(probabilities)
        if array.ndim == 2 and array.shape[1] > 1:
            return array[:, 1]
        return array

    def _record_run(
        self,
        model_cfg: Any,
        params: dict[str, Any],
        metrics: dict[str, float],
        pipeline: FeaturePipeline,
    ) -> None:
        with self.tracker.start_run(run_name=model_cfg.name, tags={"model": model_cfg.type}):
            self.tracker.log_params(params)
            self.tracker.log_metrics(metrics)
            if pipeline.metadata:
                self.tracker.log_dict(pipeline.metadata.to_dict(), "feature_metadata.json")

    def _persist_best_model(self, state: _BestModelState, pipeline: FeaturePipeline, splits: SplitData) -> None:
        if not self.settings.serving.local_model_path or state.model is None:
            return

        output_path = self.settings.resolve_path(self.settings.serving.local_model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": state.model,
            "transformers": pipeline.transformers,
            "metadata": splits.metadata,
            "id_column": self.settings.data.id_column,
            "target_column": self.settings.data.target_column,
        }
        joblib.dump(artifact, output_path)


__all__ = ["Trainer", "TrainingResult", "ModelComparison"]
