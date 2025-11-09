import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlsys.features.transformer import FeatureMetadata
from mlsys.serving.predictor import LocalArtifactPredictor


class _AddBiasTransformer:
    """Simple transformer used for testing LocalArtifactPredictor."""

    def fit(self, df: pd.DataFrame) -> "_AddBiasTransformer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["bias"] = 1
        return result


def _make_artifact() -> dict:
    base = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})
    target = pd.Series([0, 1, 0, 1], name="target")

    transformer = _AddBiasTransformer().fit(base)
    features = transformer.transform(base)

    model = LogisticRegression()
    model.fit(features, target)

    metadata = FeatureMetadata(
        feature_names=list(features.columns),
        categorical_features=[],
        datetime_columns=[],
        reference_date=None,
        n_samples=len(features),
    )

    return {
        "model": model,
        "transformers": [transformer],
        "metadata": metadata,
        "id_column": "id",
        "target_column": "target",
    }


def test_local_artifact_predict_one_adds_missing_features() -> None:
    artifact = _make_artifact()
    predictor = LocalArtifactPredictor(artifact)

    response = predictor.predict_one({"a": 1, "b": 0})
    assert "score" in response
    assert isinstance(response["score"], float)


def test_local_artifact_predict_batch_vectorises() -> None:
    artifact = _make_artifact()
    predictor = LocalArtifactPredictor(artifact)

    batch = predictor.predict_batch(
        [
            {"a": 1, "b": 0},
            {"a": 0, "b": 1},
        ]
    )

    assert len(batch) == 2
    assert all("score" in row for row in batch)
