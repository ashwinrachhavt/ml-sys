#!/usr/bin/env python3
"""Evaluate a production model against the holdout set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mlsys.config import Settings
from mlsys.data import DatasetLoader
from mlsys.evaluation import Evaluator
from mlsys.features import FeaturePipeline, TransformerRegistry
from mlsys.serving import ModelLoader

DEFAULT_CONFIG = Path("config/config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the production model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-uri", type=str, default=None, help="Optional explicit MLflow model URI")
    return parser.parse_args()


def build_pipeline(settings: Settings) -> FeaturePipeline:
    transformers = [
        TransformerRegistry.create(
            cfg.type,
            **{k: v for k, v in cfg.model_dump(exclude={"type"}).items() if v is not None},
        )
        for cfg in settings.features.transformers
    ]
    return FeaturePipeline(transformers=transformers)


def main() -> None:
    args = parse_args()
    settings = Settings.from_yaml(args.config)

    _, datasets = DatasetLoader.from_config(
        loader_type=settings.data.loader_type,
        sources=settings.data_paths(),
    )

    pipeline = build_pipeline(settings)
    features, target = pipeline.build(datasets, settings.data.id_column, settings.data.target_column)
    splits = pipeline.split(
        features,
        target,
        test_size=settings.training.test_size,
        val_size=settings.training.val_size,
        random_state=settings.training.random_state,
        stratify=settings.training.stratify,
        categorical_features=[],
        datetime_columns=[],
        reference_date=None,
    )

    loader = ModelLoader(tracking_uri=settings.tracking.tracking_uri)
    local_path: Path | None = None
    if getattr(settings.serving, "local_model_path", None):
        try:
            local_path = settings.resolve_path(settings.serving.local_model_path)  # type: ignore[attr-defined]
        except Exception:
            local_path = Path(settings.serving.local_model_path)

    if local_path and local_path.exists():
        print(f"Loading model from local path: {local_path}")
        loaded = loader.load_from_local(local_path)
    elif args.model_uri:
        loaded = loader.load_from_uri(args.model_uri)
    else:
        loaded = loader.load_from_registry(settings.serving.model_name, settings.serving.model_stage)

    predictor = loaded.model
    predictions = predictor.predict(splits.x_test)

    if isinstance(predictions, pd.DataFrame):
        if predictions.shape[1] == 1:
            scores = predictions.iloc[:, 0].to_numpy()
        elif "score" in predictions.columns:
            scores = predictions["score"].to_numpy()
        elif "probability" in predictions.columns:
            scores = predictions["probability"].to_numpy()
        else:
            scores = predictions.iloc[:, -1].to_numpy()
    elif isinstance(predictions, pd.Series):
        scores = predictions.to_numpy()
    else:
        scores = np.asarray(predictions)

    evaluator = Evaluator(settings.evaluation.metrics, settings.evaluation.threshold_metric)
    result = evaluator.evaluate(splits.y_test, scores)

    print("Evaluation metrics:")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
