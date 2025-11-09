from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.deps import get_predictor
from app.main import create_app


class StubPredictor:
    model_name = "demo"
    model_version = "1"
    model_stage = "Production"
    best_threshold = 0.7
    metadata = None
    metrics = {"val_accuracy": 0.9}

    def predict(self, payload, threshold=None):  # pragma: no cover - not used here
        raise NotImplementedError


def test_model_info_endpoint():
    app = create_app(warm_model=False)
    client = TestClient(app)
    app.dependency_overrides[get_predictor] = lambda: StubPredictor()

    response = client.get("/v1/model/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "demo"
    assert payload["threshold"] == 0.7
    assert payload["metrics"]["val_accuracy"] == 0.9
