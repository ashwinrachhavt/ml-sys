from pathlib import Path

import pytest

from mlsys.config.loader import load_settings


def test_load_settings(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
project:
  name: test

data:
  loader_type: csv
  sources:
    customers: data/customers.csv
  id_column: ID
  target_column: is_customer

features:
  transformers: []

training:
  models:
    - name: logistic
      type: sklearn.logistic_regression

tracking:
  backend: none

serving:
  model_name: test
  model_stage: Staging
""",
        encoding="utf-8",
    )

    settings = load_settings(config)
    assert settings.project.name == "test"
    assert settings.data.loader_type == "csv"
    assert settings.serving.model_name == "test"


def test_missing_sources_raises(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
project:
  name: test

data:
  loader_type: csv
  sources: {}
  id_column: ID
  target_column: is_customer

training:
  models: []

tracking:
  backend: none

serving:
  model_name: test
  model_stage: Staging
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_settings(config)
