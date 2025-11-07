from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from mlsys.core.config import FrameworkConfig
from mlsys.inference import service as inference_service
from mlsys.inference.registry import ModelRegistry
from mlsys.inference.service import create_app
from mlsys.training.pipeline import build_feature_matrix, train_and_evaluate


@pytest.fixture(scope="module", autouse=True)
def ensure_model_artifact(tmp_path_factory):
    """Ensure a model artifact exists for inference tests."""
    tmp_dir = tmp_path_factory.mktemp("model_training")
    tmp_model_path = tmp_dir / "model.joblib"
    registry_path = tmp_dir / "registry"

    base_config = FrameworkConfig()
    serving_config = base_config.serving.model_copy(update={"model_registry_path": registry_path})
    config = base_config.model_copy(
        update={
            "serving": serving_config,
            "artifacts_path": tmp_dir / "artifacts",
        }
    )

    train_and_evaluate(config=config, output_model_path=tmp_model_path)

    inference_service.get_model.cache_clear()
    inference_service._CURRENT_MODEL_PATH = None
    inference_service.REGISTRY_PATH = registry_path
    inference_service.MODEL_REGISTRY = ModelRegistry(registry_path)


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True


def test_score_endpoint_happy_path(client):
    features, _ = build_feature_matrix()
    subset = features.head(2)
    payload_df = subset.astype(object).where(~subset.isna(), None)
    payload = {"leads": payload_df.to_dict(orient="records")}
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probabilities" in data
    assert len(data["probabilities"]) == 2


def test_score_empty_payload(client):
    response = client.post("/score", json={"leads": []})
    assert response.status_code == 400


def test_health_missing_model(monkeypatch):
    inference_service.get_model.cache_clear()
    with mock.patch("mlsys.inference.service.MODEL_PATH", new=Path("/tmp/nonexistent.joblib")):
        with mock.patch("mlsys.inference.service.load_model", side_effect=FileNotFoundError("missing")):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 500
    inference_service.get_model.cache_clear()
