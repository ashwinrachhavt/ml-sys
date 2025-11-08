from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from app.data.base_loader import BaseDataLoader
from app.features.pipeline import FeatureMatrix, FeaturePipeline
from app.ml.trainer import ModelSpec, TrainingPipeline


class StubLoader(BaseDataLoader):
    def __init__(self, features: pd.DataFrame, usage: pd.DataFrame) -> None:
        self.features = features
        self.usage = usage

    def load(self, *, config=None, config_path=None):  # type: ignore[override]
        return {
            "customers": self.features,
            "usage_actions": self.usage,
        }


class StubFeaturePipeline(FeaturePipeline):
    def __init__(self, matrix: FeatureMatrix) -> None:
        super().__init__({})
        self.matrix = matrix

    def build_feature_matrix(self, datasets):  # type: ignore[override]
        return self.matrix


@pytest.fixture
def feature_matrix() -> FeatureMatrix:
    x, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=3,
        random_state=42,
    )
    x_train, x_val = x[:60], x[60:]
    y_train, y_val = y[:60], y[60:]

    return FeatureMatrix(
        x_train=pd.DataFrame(x_train, columns=[f"f{i}" for i in range(5)]),
        x_val=pd.DataFrame(x_val, columns=[f"f{i}" for i in range(5)]),
        y_train=pd.Series(y_train),
        y_val=pd.Series(y_val),
        feature_names=[f"f{i}" for i in range(5)],
    )


@pytest.fixture
def trainer(feature_matrix: FeatureMatrix) -> TrainingPipeline:
    loader = StubLoader(
        features=pd.DataFrame(),
        usage=pd.DataFrame(),
    )
    pipeline = StubFeaturePipeline(feature_matrix)

    specs = [
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
            param_grid={"C": [0.1, 1.0]},
        ),
        ModelSpec(
            name="dummy",
            estimator=DummyClassifier(strategy="most_frequent"),
        ),
    ]

    return TrainingPipeline(
        loader=loader,
        feature_pipeline=pipeline,
        model_specs=specs,
    )


def test_training_pipeline_selects_best_model(trainer: TrainingPipeline) -> None:
    result = trainer.run(config={"training": {"cv_folds": 3}, "evaluation": {"primary_metric": "roc_auc"}})

    assert result.best_model_name == "logistic_regression"
    assert "roc_auc" in result.metrics
