import pandas as pd

from mlsys.features import FeaturePipeline, TransformerRegistry
from tests.fixtures.sample_data import make_customer_data, make_usage_data


def test_feature_pipeline_combines_datasets() -> None:
    pipeline = FeaturePipeline(
        transformers=[
            TransformerRegistry.create("datetime", columns=["CLOSEDATE"], reference_date="2024-01-10"),
            TransformerRegistry.create("fillna"),
        ]
    )

    datasets = {
        "customers": make_customer_data(),
        "usage_actions": make_usage_data(),
    }
    features, target = pipeline.build(datasets, id_column="ID", target_column="is_customer")
    assert not features.empty
    assert target.shape[0] == features.shape[0]


def test_feature_pipeline_split() -> None:
    df = pd.DataFrame({"ID": range(10), "is_customer": [0, 1] * 5, "value": range(10)})
    pipeline = FeaturePipeline()
    features, target = pipeline.build({"customers": df}, id_column="ID", target_column="is_customer")
    splits = pipeline.split(
        features,
        target,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        stratify=True,
        categorical_features=[],
        datetime_columns=[],
        reference_date=None,
    )
    assert splits.x_train.shape[0] > 0
    assert splits.x_val.shape[0] > 0
    assert splits.x_test.shape[0] > 0
