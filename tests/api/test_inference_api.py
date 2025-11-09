from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.deps import get_predictor
from app.main import create_app


class StubPredictor:
    model_version = "1"

    def predict(self, payload, threshold=None):
        class Result:
            prediction = 1
            probability = 0.9
            model_version = "1"
            raw_output = None

        return Result()


def test_predict_endpoint():
    app = create_app(warm_model=False)

    client = TestClient(app)
    app.dependency_overrides[get_predictor] = lambda: StubPredictor()
    response = client.post("/v1/predict", json={"data": {"foo": 1}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == 1
    assert payload["probability"] == 0.9
    assert payload["model_version"] == "1"
