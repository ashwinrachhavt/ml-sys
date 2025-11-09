from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.features.pipeline import FeaturePipeline


@pytest.fixture
def config() -> dict:
    return {
        "data": {
            "id_column": "id",
            "target_column": "is_customer",
        },
        "features": {
            "categorical": ["INDUSTRY"],
            "reference_date": "2025-11-08",
        },
        "training": {
            "val_size": 0.2,
            "random_state": 42,
            "stratify": True,
        },
    }


@pytest.fixture
def datasets(tmp_path: Path) -> dict[str, pd.DataFrame]:
    customers = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "INDUSTRY": ["TECH", "FIN", "TECH", "HEALTH", "FIN"],
        }
    )
    usage = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 3, 3, 4],
            "WHEN_TIMESTAMP": [
                "2025-11-01",
                "2025-11-05",
                "2025-10-20",
                "2025-09-15",
                "2025-11-01",
                "2025-11-07",
                "2025-06-10",
            ],
            "ACTIONS_CRM_CONTACTS": [5, 7, 3, 1, 1, 2, 0],
        }
    )
    noncustomers = pd.DataFrame(
        {
            "id": [6, 7, 8],
            "INDUSTRY": ["TECH", "RETAIL", "FIN"],
        }
    )

    return {
        "customers": customers,
        "noncustomers": noncustomers,
        "usage_actions": usage,
    }


def test_feature_pipeline_creates_expected_columns(config: dict, datasets: dict[str, pd.DataFrame]) -> None:
    pipeline = FeaturePipeline(config)
    matrix = pipeline.build_feature_matrix(datasets)

    assert not matrix.x_train.empty
    assert "INDUSTRY_FIN" in matrix.feature_names
    assert "days_since_last_action" in matrix.feature_names
    assert set(matrix.y_train.unique()) <= {0, 1}
    assert matrix.metadata.feature_names == matrix.feature_names

    # Ensure target remains aligned after train/val split
    assert matrix.x_train.shape[0] == matrix.y_train.shape[0]
    assert matrix.x_val.shape[0] == matrix.y_val.shape[0]
    assert matrix.x_test.shape[0] == matrix.y_test.shape[0]


def test_feature_pipeline_requires_noncustomers_when_target_missing(
    config: dict, datasets: dict[str, pd.DataFrame]
) -> None:
    datasets.pop("noncustomers")
    config["data"]["target_column"] = "custom_target"
    pipeline = FeaturePipeline(config)

    with pytest.raises(KeyError):
        pipeline.build_feature_matrix(datasets)
