#!/usr/bin/env python3
"""CLI entrypoint for running the training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlsys.config import Settings
from mlsys.tracking import MLflowTracker, NullTracker
from mlsys.training import Trainer

DEFAULT_CONFIG = Path("config/config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tabular training pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to configuration YAML")
    parser.add_argument("--no-tracking", action="store_true", help="Disable experiment tracking")
    return parser.parse_args()


def build_tracker(settings: Settings, disabled: bool) -> MLflowTracker | NullTracker:
    if disabled or settings.tracking.backend == "none":
        return NullTracker()
    if settings.tracking.backend == "mlflow":
        return MLflowTracker(
            tracking_uri=settings.tracking.tracking_uri,
            experiment_name=settings.tracking.experiment_name,
            run_name_prefix=settings.tracking.run_name_prefix,
        )
    return NullTracker()


def main() -> None:
    args = parse_args()
    settings = Settings.from_yaml(args.config)
    tracker = build_tracker(settings, disabled=args.no_tracking)

    trainer = Trainer(settings=settings, tracker=tracker)
    result = trainer.train()

    print("Training completed")
    print(f"Best model: {result.best_model_name}")
    if result.best_threshold is not None:
        print(f"Best threshold: {result.best_threshold:.3f}")
    for name, value in sorted(result.metrics.items()):
        print(f"{name}: {value:.4f}")
    print("\nModel comparisons:")
    for comparison in result.comparisons:
        print(f"  {comparison.name:<25} CV={comparison.cv_score:.4f} params={comparison.best_params}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
