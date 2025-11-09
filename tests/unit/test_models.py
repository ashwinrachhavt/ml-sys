import pandas as pd

from mlsys.models import ModelRegistry


def test_model_registry_instantiates_logistic() -> None:
    model = ModelRegistry.create("sklearn.logistic_regression")
    df = pd.DataFrame({"x": [0, 1, 2, 3]})
    target = pd.Series([0, 0, 1, 1])
    model.fit(df, target)
    preds = model.predict(df)
    assert len(preds) == len(df)
