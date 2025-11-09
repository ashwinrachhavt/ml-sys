"""Model loading utilities for serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import mlflow
import mlflow.pyfunc
import pandas as pd


@dataclass
class LoadedModel:
    """Wrapper around an MLflow model artifact."""

    model: Any
    signature: Any | None = None
    input_example: Any | None = None


class ModelLoader:
    """Load models from MLflow registry or direct URI."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)  # type: ignore[attr-defined]

    def load_from_registry(self, name: str, stage: str) -> LoadedModel:
        model_uri = f"models:/{name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        signature = (
            mlflow.models.signature_inference.infer_signature(  # type: ignore[attr-defined]
                model.metadata.signature.inputs, model.metadata.signature.outputs
            )
            if model.metadata.signature
            else None
        )
        return LoadedModel(model=model, signature=signature, input_example=model.metadata.saved_input_example_info)

    def load_from_uri(self, uri: str) -> LoadedModel:
        model = mlflow.pyfunc.load_model(uri)
        return LoadedModel(
            model=model, signature=model.metadata.signature, input_example=model.metadata.saved_input_example_info
        )


class PandasPredictor:
    """Simple wrapper using MLflow pyfunc for inference."""

    def __init__(self, loaded: LoadedModel):
        self.loaded = loaded

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        df = pd.DataFrame([payload])
        predictions = self.loaded.model.predict(df)
        return self._normalise(predictions)[0]

    def predict_batch(self, payload: list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
        if isinstance(payload, pd.DataFrame):
            df = payload
        else:
            df = pd.DataFrame(payload)
        predictions = self.loaded.model.predict(df)
        return self._normalise(predictions)

    def _normalise(self, predictions: Any) -> list[dict[str, Any]]:
        if isinstance(predictions, pd.DataFrame):
            return cast(list[dict[str, Any]], predictions.to_dict(orient="records"))
        if isinstance(predictions, pd.Series):
            return [{"score": float(val)} for val in predictions]
        if isinstance(predictions, list):
            return [{"score": float(val)} for val in predictions]
        raise TypeError(f"Unsupported prediction output type: {type(predictions)!r}")


__all__ = ["ModelLoader", "LoadedModel", "PandasPredictor"]
