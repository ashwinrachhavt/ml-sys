from pathlib import Path

import pandas as pd

from mlsys.config import Settings
from mlsys.training import Trainer


def create_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: integration

data:
  loader_type: csv
  sources:
    customers: customers.csv
  id_column: ID
  target_column: target

features:
  transformers:
    - type: fillna

training:
  models:
    - name: logistic
      type: sklearn.logistic_regression
  cv_folds: 2
  test_size: 0.3
  val_size: 0.2
  stratify: false

tracking:
  backend: none

serving:
  model_name: integration
  model_stage: Staging
""",
        encoding="utf-8",
    )

    data = pd.DataFrame(
        {
            "ID": range(30),
            "target": [0, 1] * 15,
            "feature": range(30),
        }
    )
    data.to_csv(tmp_path / "customers.csv", index=False)
    return config_path


def test_training_pipeline_end_to_end(tmp_path: Path) -> None:
    config_path = create_config(tmp_path)
    settings = Settings.from_yaml(config_path)
    trainer = Trainer(settings=settings)
    result = trainer.train()
    assert result.best_model_name == "logistic"
