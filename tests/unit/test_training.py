from pathlib import Path

import pandas as pd

from mlsys.config import Settings
from mlsys.tracking import NullTracker
from mlsys.training import Trainer


def make_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: test

data:
  loader_type: csv
  sources:
    customers: customers.csv
  id_column: ID
  target_column: is_customer

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
  model_name: test
  model_stage: Staging
  local_model_path: model.joblib
""",
        encoding="utf-8",
    )
    customers = pd.DataFrame({"ID": range(20), "is_customer": [0, 1] * 10, "value": range(20)})
    customers.to_csv(tmp_path / "customers.csv", index=False)
    return config_path


def test_trainer_runs(tmp_path: Path) -> None:
    config_path = make_config(tmp_path)
    settings = Settings.from_yaml(config_path)
    trainer = Trainer(settings=settings, tracker=NullTracker())
    result = trainer.train()
    assert result.best_model_name == "logistic"
    assert result.metrics
    assert (tmp_path / "model.joblib").exists()
