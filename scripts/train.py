from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import load_config
from app.data.dataframe_loader import DataFrameLoader
from app.features.pipeline import FeaturePipeline
from app.ml.trainer import ModelSpec, TrainingPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ML training pipeline.")
    parser.add_argument("--config", type=Path, default=None, help="Path to configuration YAML file")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    return parser.parse_args()


def build_model_specs(random_state: int) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
            param_grid={"C": [0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=random_state),
            param_grid={"n_estimators": [100, 200], "max_depth": [None, 6]},
        ),
    ]
    return specs


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    loader = DataFrameLoader()
    feature_pipeline = FeaturePipeline(config)

    random_state = config.get("training", {}).get("random_state", 42)
    specs = build_model_specs(random_state)

    pipeline = TrainingPipeline(
        loader=loader,
        feature_pipeline=feature_pipeline,
        model_specs=specs,
        enable_mlflow=args.mlflow,
    )

    result = pipeline.run(config_path=args.config, config=config)

    print("Best model:", result.best_model_name)
    print("Metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
