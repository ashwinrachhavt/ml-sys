"""Serving orchestrator used by FastAPI endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from mlsys.config import Settings

from .loader import ModelLoader, PandasPredictor


@dataclass
class PredictionResponse:
    """Structured prediction output."""

    predictions: list[dict[str, Any]]


class LocalArtifactPredictor:
    """Predictor backed by joblib artifacts saved during training."""

    def __init__(self, artifact: dict[str, Any]):
        self.model = artifact["model"]
        self.transformers = artifact.get("transformers", [])
        self.metadata = artifact.get("metadata")

    def _prepare_features(self, payload: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(payload, pd.DataFrame):
            df = payload.copy()
        else:
            df = pd.DataFrame(payload)

        for transformer in self.transformers:
            df = transformer.transform(df)

        if self.metadata:
            ordered = {}
            for column in self.metadata.feature_names:
                if column in df.columns:
                    ordered[column] = df[column]
                else:
                    ordered[column] = 0
            # Ensure a single-row DataFrame when values are scalars
            df = pd.DataFrame(
                {
                    k: ([v] if not hasattr(v, "__len__") or isinstance(v, (str, bytes)) else v)
                    for k, v in ordered.items()
                }
            )
        return df

    def predict_batch(self, payload: list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
        features = self._prepare_features(payload)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)
            if isinstance(proba, pd.DataFrame):
                scores = proba.iloc[:, -1].to_numpy()
            else:
                scores = proba[:, -1]
            return [{"score": float(value)} for value in scores]

        preds = self.model.predict(features)
        if isinstance(preds, pd.Series):
            values = preds.to_numpy()
        else:
            values = preds
        return [{"label": float(value)} for value in values]

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch([payload])[0]

    # Simple predict for evaluate/CLI usage
    def predict(self, x: pd.DataFrame | dict[str, Any]) -> pd.Series:
        if isinstance(x, dict):
            batch = self.predict_batch([x])
            # Normalise to Series of scores/labels
            key = "score" if "score" in batch[0] else "label"
            return pd.Series([row[key] for row in batch])
        batch = self.predict_batch(x)
        key = "score" if (batch and "score" in batch[0]) else "label"
        return pd.Series([row[key] for row in batch], index=x.index)


class PredictorService:
    """Load model artifacts and expose batch/online inference helpers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.loader = ModelLoader(tracking_uri=settings.tracking.tracking_uri)
        self._predictor: Any | None = None

    def _load_local_artifact(self) -> Any | None:
        local_path = self.settings.serving.local_model_path
        if not local_path:
            return None
        path = self.settings.resolve_path(local_path)
        if not path.exists():
            return None
        artifact = joblib.load(path)
        return LocalArtifactPredictor(artifact)

    def _ensure_model(self) -> Any:
        if self._predictor is None:
            predictor = self._load_local_artifact()
            if predictor is None:
                loaded = self.loader.load_from_registry(
                    name=self.settings.serving.model_name,
                    stage=self.settings.serving.model_stage,
                )
                predictor = PandasPredictor(loaded)
            self._predictor = predictor
        return self._predictor

    def predict_one(self, payload: dict[str, Any]) -> PredictionResponse:
        predictor = self._ensure_model()
        prediction = predictor.predict_one(payload)
        return PredictionResponse(predictions=[prediction])

    def predict_batch(self, payload: list[dict[str, Any]] | pd.DataFrame) -> PredictionResponse:
        predictor = self._ensure_model()
        predictions = predictor.predict_batch(payload)
        return PredictionResponse(predictions=predictions)


__all__ = ["PredictorService", "PredictionResponse", "LocalArtifactPredictor"]
