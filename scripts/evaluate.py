#!/usr/bin/env python
"""Script to evaluate a trained model on test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlsys.core.config import FrameworkConfig
from mlsys.features.transformers import build_feature_matrix
from mlsys.inference.registry import ModelRegistry


def evaluate_model(
    model_path: Path,
    *,
    config: FrameworkConfig,
    data_config_overrides: dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a trained model and return metrics.

    Args:
        model_path: Path to the trained model file
        data_dir: Directory containing test data
        output_path: Path to save evaluation results (JSON)

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print("Loading test data...")
    data_config = config.data.model_copy(update=data_config_overrides or {})
    features, labels = build_feature_matrix(data_config=data_config)

    print("Generating predictions...")
    y_proba = model.predict_proba(features)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("Computing metrics...")
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, y_pred)),
        "precision": float(precision_score(labels, y_pred)),
        "recall": float(recall_score(labels, y_pred)),
        "f1_score": float(f1_score(labels, y_pred)),
        "roc_auc": float(roc_auc_score(labels, y_proba)),
        "pr_auc": float(average_precision_score(labels, y_proba)),
    }

    # Confusion matrix
    cm = confusion_matrix(labels, y_pred)
    metrics["confusion_matrix"] = {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }

    # Classification report
    clf_report = classification_report(labels, y_pred, output_dict=True)
    metrics["classification_report"] = clf_report

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR AUC:    {metrics['pr_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(
        f"  TN: {metrics['confusion_matrix']['true_negatives']:>5}  FP: {metrics['confusion_matrix']['false_positives']:>5}"
    )
    print(
        f"  FN: {metrics['confusion_matrix']['false_negatives']:>5}  TP: {metrics['confusion_matrix']['true_positives']:>5}"
    )
    print("=" * 60)

    return metrics


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a trained lead scoring model")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Framework configuration file (YAML)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Explicit path to model artifact (overrides registry lookup)",
    )
    parser.add_argument(
        "--use-registry",
        action="store_true",
        help="Load the best model from the configured model registry",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Override the model registry root directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing data files (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results (JSON format)",
    )

    args = parser.parse_args()

    config = FrameworkConfig.from_yaml(args.config) if args.config else FrameworkConfig()

    data_overrides: dict[str, Any] | None = None
    if args.data_dir is not None:
        data_dir = args.data_dir
        data_overrides = {
            "customers_path": data_dir / "customers.csv",
            "noncustomers_path": data_dir / "noncustomers.csv",
            "usage_path": data_dir / "usage_actions.csv",
        }

    if args.registry_path is not None:
        registry_path = args.registry_path
    else:
        registry_path = config.serving.model_registry_path

    model_path = args.model_path
    if model_path is None and args.use_registry:
        registry = ModelRegistry(registry_path)
        model_path = registry.get_best_model_path()

    if model_path is None:
        model_path = config.resolve_artifact_path("models", f"{config.model.type}_model.joblib")

    evaluate_model(
        model_path=model_path,
        config=config,
        data_config_overrides=data_overrides,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
