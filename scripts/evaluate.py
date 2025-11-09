#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _load_components() -> SimpleNamespace:
    from app.core.config import DEFAULT_CONFIG_PATH, load_config
    from app.data.dataframe_loader import DataFrameLoader
    from app.features.pipeline import FeaturePipeline
    from app.ml.evaluation import classification_metrics
    from app.serving.predictor import ModelPredictor

    return SimpleNamespace(
        DEFAULT_CONFIG_PATH=DEFAULT_CONFIG_PATH,
        load_config=load_config,
        DataFrameLoader=DataFrameLoader,
        FeaturePipeline=FeaturePipeline,
        classification_metrics=classification_metrics,
        ModelPredictor=ModelPredictor,
    )


def parse_args(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the production model on the hold-out test split")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to configuration file",
    )
    return parser.parse_args()


def main() -> None:
    components = _load_components()
    args = parse_args(components.DEFAULT_CONFIG_PATH)

    load_config = components.load_config
    data_loader_cls = components.DataFrameLoader
    feature_pipeline_cls = components.FeaturePipeline
    classification_metrics = components.classification_metrics
    predictor_cls = components.ModelPredictor

    config_path = args.config.resolve()
    cfg = load_config(config_path)

    loader = data_loader_cls()
    datasets = loader.load(config=cfg, config_path=config_path)
    pipeline = feature_pipeline_cls(cfg)
    matrix = pipeline.build_feature_matrix(datasets)

    predictor = predictor_cls(config_path=config_path)
    model = predictor.model
    if model is None:
        print("No production model is registered yet. Train and register a model before running evaluation.")
        return

    x_test = matrix.x_test
    y_test = matrix.y_test

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)[:, 1]
    else:
        proba = None
    preds = model.predict(x_test)

    metrics = classification_metrics(y_test.values, preds, proba)
    threshold = predictor.best_threshold or cfg.get("evaluation", {}).get("classification_threshold", 0.5)
    tuned_metrics = metrics
    if proba is not None:
        tuned_preds = (proba >= threshold).astype(int)
        tuned_metrics = classification_metrics(y_test.values, tuned_preds, proba)

    print("Evaluation against test set")
    print(f"Model version: {predictor.model_version}")
    print(f"Threshold: {threshold:.3f}")
    print("")
    for key, value in metrics.items():
        print(f"  base_{key}: {value:.4f}")

    if proba is not None:
        print("")
        for key, value in tuned_metrics.items():
            print(f"  tuned_{key}: {value:.4f}")


if __name__ == "__main__":
    main()
