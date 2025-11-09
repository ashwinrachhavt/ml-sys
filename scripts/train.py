#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    _ensure_src_on_path()
    from app.ml.trainer import ModelSpec
else:
    ModelSpec = Any  # type: ignore[assignment]


def _import_training_components() -> SimpleNamespace:
    _ensure_src_on_path()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from app.core.config import DEFAULT_CONFIG_PATH, load_config
    from app.data.dataframe_loader import DataFrameLoader
    from app.features.pipeline import FeaturePipeline
    from app.ml.models.factory import build_model_specs
    from app.ml.trainer import ModelSpec, TrainingPipeline

    return SimpleNamespace(
        RandomForestClassifier=RandomForestClassifier,
        LogisticRegression=LogisticRegression,
        DEFAULT_CONFIG_PATH=DEFAULT_CONFIG_PATH,
        load_config=load_config,
        DataFrameLoader=DataFrameLoader,
        FeaturePipeline=FeaturePipeline,
        build_model_specs=build_model_specs,
        ModelSpec=ModelSpec,
        TrainingPipeline=TrainingPipeline,
    )


def parse_args(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tabular training pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging even if enabled in config",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["all"],
        help="Subset of model names to train (default: all registered models)",
    )
    return parser.parse_args()


def main() -> None:
    components = _import_training_components()
    load_config = components.load_config
    dataframe_loader_cls = components.DataFrameLoader
    feature_pipeline_cls = components.FeaturePipeline
    training_pipeline_cls = components.TrainingPipeline
    build_model_specs = components.build_model_specs

    args = parse_args(components.DEFAULT_CONFIG_PATH)

    config_path = args.config.resolve()
    cfg = load_config(config_path)

    loader = dataframe_loader_cls()
    feature_pipeline = feature_pipeline_cls(cfg)
    requested_models = None if "all" in args.models else args.models
    model_specs = build_model_specs(cfg, requested_models)

    enable_mlflow = cfg.get("mlflow", {}).get("enabled", True) and not args.no_mlflow

    pipeline = training_pipeline_cls(
        loader=loader,
        feature_pipeline=feature_pipeline,
        model_specs=model_specs,
        enable_mlflow=enable_mlflow,
    )

    result = pipeline.run(config=cfg, config_path=config_path)

    print("Training completed")
    print(f"Best model: {result.best_model_name}")
    if result.best_threshold is not None:
        print(f"Best threshold: {result.best_threshold:.3f}")
    print(f"Feature count: {len(result.metadata.feature_names)}")
    print("\nModel comparison (CV score)")
    for comparison in result.comparisons:
        print(f"  {comparison.name:<25} {comparison.cv_score:.4f}")
    for metric, value in sorted(result.metrics.items()):
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
