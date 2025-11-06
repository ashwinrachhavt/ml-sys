"""CLI entry-point for triggering continuous training runs."""
from __future__ import annotations

import argparse
from pathlib import Path

from mlsys.training.pipeline import TrainingResult, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGB lead-scoring model.")
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
        "--tune",
        action="store_true",
        help="Enable randomized hyperparameter tuning for the XGB model",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit path for saving the trained model artifact",
    )
    return parser.parse_args()


def main() -> TrainingResult:
    args = parse_args()
    result = train_and_evaluate(
        test_size=args.test_size,
        calibration_size=args.calibration_size,
        tune_hyperparameters_flag=args.tune,
        random_state=args.random_state,
        data_dir=args.data_dir,
        output_model_path=args.output,
    )

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
