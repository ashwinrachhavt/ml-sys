from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from app.data.dataframe_loader import DataFrameLoader


@pytest.fixture
def config_path(tmp_path: Path, tmp_raw_dir: Path) -> Path:
    config = {
        "data": {
            "customers_file": str(tmp_raw_dir / "customers.csv"),
            "noncustomers_file": str(tmp_raw_dir / "noncustomers.csv"),
            "usage_file": str(tmp_raw_dir / "usage_actions.csv"),
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture
def tmp_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "customers.csv").write_text(
        "id,CLOSEDATE,MRR,ALEXA_RANK,EMPLOYEE_RANGE,INDUSTRY\n1,2019-01-01,100,1000,1 to 10,TECH\n"
    )

    (raw_dir / "noncustomers.csv").write_text("id,ALEXA_RANK,EMPLOYEE_RANGE,INDUSTRY\n2,2000,11 to 25,OTHER\n")

    (raw_dir / "usage_actions.csv").write_text("id,WHEN_TIMESTAMP,ACTIONS_CRM_CONTACTS\n1,2020-01-01,10\n")

    return raw_dir


def test_dataframe_loader_loads_all_expected_frames(config_path: Path) -> None:
    loader = DataFrameLoader()
    data = loader.load(config_path=config_path)

    assert isinstance(data, dict)
    assert set(data.keys()) == {"customers", "noncustomers", "usage_actions"}

    for name, df in data.items():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, f"{name} DataFrame should not be empty"


def test_dataframe_loader_raises_if_file_missing(tmp_path: Path) -> None:
    missing_file_config: dict[str, Any] = {
        "data": {
            "customers_file": str(tmp_path / "raw" / "customers.csv"),
            "noncustomers_file": str(tmp_path / "raw" / "noncustomers.csv"),
            "usage_file": str(tmp_path / "raw" / "usage_actions.csv"),
        }
    }
    broken_config_path = tmp_path / "broken_config.yaml"
    broken_config_path.write_text(yaml.safe_dump(missing_file_config))

    loader = DataFrameLoader()

    with pytest.raises(FileNotFoundError):
        loader.load(config_path=broken_config_path)
