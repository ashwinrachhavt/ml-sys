#!/usr/bin/env python
"""Register a model artifact in the local ml-sys registry (and optionally MLflow)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from mlsys.core.config import FrameworkConfig
from mlsys.inference.registry import ModelRegistry
from mlsys.tracking.mlflow_tracker import MLflowTracker


def load_metrics(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    metrics: dict[str, float] = {}
    for key, value in data.items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def load_config(config_path: Path | None) -> FrameworkConfig:
    if config_path is None:
        return FrameworkConfig()
    return FrameworkConfig.from_yaml(config_path)


def register_local(
    model_path: Path,
    *,
    registry_path: Path,
    metrics: dict[str, float],
    config: FrameworkConfig,
    primary_metric: str,
) -> tuple[Path, bool]:
    registry = ModelRegistry(registry_path)
    stored_path, is_best = registry.register_model(
        model_path,
        model_name=config.model.type,
        metrics=metrics,
        config=config.model_dump(mode="json"),
        primary_metric=primary_metric,
    )
    return stored_path, is_best


def register_mlflow(model_path: Path, tracker: MLflowTracker | None) -> None:
    if tracker is None or not tracker.available:
        return
    tracker.start_run(run_name="manual-register")
    try:
        model = joblib.load(model_path)
        tracker.log_model(model, "model", registered_model_name="lead-scoring-model")
        tracker.log_artifact(model_path)
    finally:
        tracker.end_run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a model artifact with the local registry")
    parser.add_argument("model_path", type=Path, help="Path to the model artifact (.joblib)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Framework configuration file for metadata (defaults to built-in config)",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Override the registry directory (defaults to config.serving.model_registry_path)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional JSON file containing evaluation metrics",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="roc_auc_calibrated",
        help="Metric key used to determine the best model",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Also log the model to MLflow using tracker settings from the config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    metrics = load_metrics(args.metrics)
    registry_path = args.registry_path or config.serving.model_registry_path
    registry_path.mkdir(parents=True, exist_ok=True)

    stored_path, is_best = register_local(
        args.model_path,
        registry_path=registry_path,
        metrics=metrics,
        config=config,
        primary_metric=args.primary_metric,
    )

    print(f"Model stored at {stored_path}")
    if is_best:
        print("This model is now the registry champion.")

    tracker = MLflowTracker(config.tracking) if args.mlflow else None
    if tracker:
        register_mlflow(args.model_path, tracker)


if __name__ == "__main__":
    main()
