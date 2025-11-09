from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from mlsys.api import create_app
from mlsys.config import Settings


def make_settings(tmp_path: Path) -> Settings:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: api

data:
  loader_type: csv
  sources:
    customers: customers.csv
  id_column: ID
  target_column: target

features:
  transformers: []

training:
  models:
    - name: logistic
      type: sklearn.logistic_regression
  cv_folds: 2
  test_size: 0.2
  val_size: 0.2
  stratify: false

tracking:
  backend: none

serving:
  model_name: api
  model_stage: Staging
""",
        encoding="utf-8",
    )
    data = pd.DataFrame({"ID": [1, 2], "target": [0, 1]})
    data.to_csv(tmp_path / "customers.csv", index=False)
    return Settings.from_yaml(config_path)


def test_health_routes(tmp_path: Path) -> None:
    client = TestClient(create_app(settings=make_settings(tmp_path)))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
