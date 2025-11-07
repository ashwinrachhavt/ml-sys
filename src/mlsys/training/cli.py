"""CLI entry-point for triggering continuous training runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlsys.core.config import FrameworkConfig
from mlsys.training.pipeline import TrainingResult, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lead-scoring model using configuration overrides.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a FrameworkConfig YAML file (defaults to in-code defaults).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing customers.csv, noncustomers.csv, usage_actions.csv",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for evaluation",
    )
    parser.add_argument(
        "--calibration-size",
        type=float,
        default=0.0,
        help="Fraction of training split reserved for holdout-based calibration (0 for CV calibration)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducibility",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable randomized hyperparameter tuning for supported models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit path for saving the trained model artifact",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Directory for the local model registry override",
    )
    return parser.parse_args()


def main() -> TrainingResult:
    args = parse_args()
    if args.config:
        base_config = FrameworkConfig.from_yaml(args.config)
    else:
        base_config = FrameworkConfig()

    data_update = {
        "test_size": args.test_size,
        "calibration_size": args.calibration_size,
        "random_state": args.random_state,
    }
    if args.data_dir is not None:
        data_dir = args.data_dir
        data_update.update(
            {
                "customers_path": data_dir / "customers.csv",
                "noncustomers_path": data_dir / "noncustomers.csv",
                "usage_path": data_dir / "usage_actions.csv",
            }
        )

    data_config = base_config.data.model_copy(update=data_update)
    model_config = base_config.model.model_copy(update={"tune_hyperparameters": args.tune})

    serving_update = {}
    if args.registry_path is not None:
        serving_update["model_registry_path"] = args.registry_path
    serving_config = base_config.serving.model_copy(update=serving_update)

    config = base_config.model_copy(update={"data": data_config, "model": model_config, "serving": serving_config})

    result = train_and_evaluate(config=config, output_model_path=args.output)

    print("Training complete!")
    for metric in sorted(result.metrics):
        value = result.metrics[metric]
        print(f"  {metric}: {value:.4f}")

    print("\nClassification report (threshold=0.5):\n")
    print(result.classification_report)
    print(f"Model saved to: {result.model_path}")
    return result


if __name__ == "__main__":
    main()
