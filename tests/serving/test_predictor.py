from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import app.serving.predictor as predictor_module
from app.features.pipeline import FeatureMetadata
from app.serving.predictor import FeatureAligner, ModelPredictor


def test_feature_aligner_reindexes_columns() -> None:
    metadata = FeatureMetadata(
        feature_names=["a", "b", "cat_blue"],
        categorical_features=["cat"],
        reference_date=None,
        datetime_columns=[],
    )
    aligner = FeatureAligner(metadata)
    payload = {"a": 1, "cat": "blue"}

    transformed = aligner.transform(payload)

    assert list(transformed.columns) == ["a", "b", "cat_blue"]
    assert transformed.at[0, "b"] == 0
    assert transformed.at[0, "cat_blue"] == 1


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mlflow:\n"
        "  tracking_uri: 'file:./mlflow'\n"
        "  registry:\n"
        "    model_name: demo\n"
        "api:\n"
        "  serving:\n"
        "    model_stage: Production\n",
        encoding="utf-8",
    )
    return config_path


class DummyModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.ones(len(features))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([[0.3, 0.7]]), (len(features), 1))


def test_model_predictor_uses_threshold(monkeypatch: pytest.MonkeyPatch, temp_config: Path) -> None:
    dummy_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        sklearn=SimpleNamespace(load_model=lambda uri: DummyModel()),
    )

    monkeypatch.setattr(predictor_module, "mlflow", dummy_mlflow)

    def fake_load_text(path: str) -> str:
        if path.endswith("feature_metadata.json"):
            return json.dumps(
                {
                    "feature_names": ["f1", "f2"],
                    "categorical_features": [],
                    "reference_date": None,
                    "datetime_columns": [],
                }
            )
        return "f1\nf2"

    monkeypatch.setattr(predictor_module, "mlflow_artifacts", SimpleNamespace(load_text=fake_load_text))

    dummy_run_data = SimpleNamespace(
        params={"best_threshold": "0.6"},
        metrics={"val_accuracy": 0.92},
    )
    dummy_client = SimpleNamespace(
        get_latest_versions=lambda name, stages: [SimpleNamespace(version="1", run_id="abc")],
        get_run=lambda run_id: SimpleNamespace(data=dummy_run_data),
    )
    monkeypatch.setattr(predictor_module, "MlflowClient", lambda: dummy_client)

    predictor = ModelPredictor(config_path=temp_config)

    result = predictor.predict({"f2": 1.0})

    assert result.prediction == 1  # probability 0.7 > threshold 0.6
    assert result.probability == pytest.approx(0.7)
    assert predictor.best_threshold == pytest.approx(0.6)
    assert predictor.metrics["val_accuracy"] == pytest.approx(0.92)
