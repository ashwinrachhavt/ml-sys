from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow  # type: ignore[import]
import mlflow.sklearn  # type: ignore[attr-defined]
import numpy as np
import pandas as pd
from mlflow import artifacts as mlflow_artifacts  # type: ignore[attr-defined]
from mlflow.tracking import MlflowClient  # type: ignore[attr-defined]

from app.core.config import load_config
from app.features.pipeline import FeatureMetadata

if TYPE_CHECKING:  # pragma: no cover - assists static analysis
    import mlflow  # noqa: F401


@dataclass
class PredictionResult:
    prediction: int
    probability: float | None
    model_version: str | None
    raw_output: Any


class FeatureAligner:
    """Align incoming feature dictionaries with the trained feature schema."""

    def __init__(self, metadata: FeatureMetadata) -> None:
        self.metadata = metadata
        self.feature_names = list(metadata.feature_names)

    def transform(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        frame = pd.DataFrame([payload])

        reference_date = self.metadata.reference_date
        reference = datetime.fromisoformat(reference_date) if reference_date else None

        for column in list(frame.columns):
            if column in self.metadata.datetime_columns:
                if pd.api.types.is_numeric_dtype(frame[column]):
                    continue
                parsed = pd.to_datetime(frame[column], errors="coerce")
                ref = reference or parsed.max()
                frame[column] = (ref - parsed).dt.days.fillna(0)
            elif frame[column].dtype == object:
                parsed = pd.to_datetime(frame[column], errors="coerce")
                if parsed.notna().any():
                    ref = reference or parsed.max()
                    frame[column] = (ref - parsed).dt.days.fillna(0)

        encoded = pd.get_dummies(frame, drop_first=False)

        aligned = encoded.reindex(columns=self.feature_names, fill_value=0) if self.feature_names else encoded
        return aligned.astype(float)


class ModelPredictor:
    """Load trained model artifacts and serve predictions."""

    def __init__(self, *, config_path: Path | None = None) -> None:
        cfg = load_config(config_path)
        mlflow_cfg = cfg.get("mlflow", {})

        self.tracking_uri = mlflow_cfg.get("tracking_uri", "file:./mlflow")
        self.model_name = mlflow_cfg.get("registry", {}).get("model_name", "model")
        self.model_stage = cfg.get("api", {}).get("serving", {}).get("model_stage", "Production")

        self._model = None
        self._aligner: FeatureAligner | None = None
        self._metadata: FeatureMetadata | None = None
        self._metrics: dict[str, float] = {}
        self._model_version: str | None = None
        self._best_threshold: float | None = None

        self._load_artifacts()

    @property
    def model_version(self) -> str | None:
        return self._model_version

    @property
    def best_threshold(self) -> float | None:
        return self._best_threshold

    @property
    def metadata(self) -> FeatureMetadata | None:
        return self._metadata

    @property
    def metrics(self) -> dict[str, float]:
        return self._metrics

    @property
    def model(self):  # type: ignore[override]
        return self._model

    def _load_artifacts(self) -> None:
        if MlflowClient is None:
            raise RuntimeError("mlflow must be installed to load model artifacts")

        logging.info("Setting MLflow tracking URI to %s", self.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)  # type: ignore[attr-defined]
        client = MlflowClient()

        latest_versions = client.get_latest_versions(self.model_name, [self.model_stage])
        if not latest_versions:
            raise RuntimeError(f"No model version found for {self.model_name} in stage {self.model_stage}")

        model_version = latest_versions[0]
        self._model_version = model_version.version

        model_uri = f"models:/{self.model_name}/{self.model_stage}"
        logging.info("Loading model from %s", model_uri)
        self._model = mlflow.sklearn.load_model(model_uri)  # type: ignore[attr-defined]

        run_id = model_version.run_id

        metadata = self._load_feature_metadata(run_id)
        self._metadata = metadata
        self._aligner = FeatureAligner(metadata)

        run_info = client.get_run(run_id)
        threshold_param = run_info.data.params.get("best_threshold")
        if threshold_param is not None:
            try:
                self._best_threshold = float(threshold_param)
            except ValueError:  # pragma: no cover - defensive parsing
                logging.warning("Unable to parse best_threshold param: %s", threshold_param)

        for key, value in run_info.data.metrics.items():
            self._metrics[key] = value

    def predict(self, payload: Mapping[str, Any], *, threshold: float | None = None) -> PredictionResult:
        if self._model is None or self._aligner is None:
            raise RuntimeError("Predictor is not initialised")

        features = self._aligner.transform(payload)
        raw_output = self._model.predict(features)

        probability: float | None = None
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(features)
            probability = float(proba[0, 1])
            decision = probability
        elif isinstance(raw_output, np.ndarray) and raw_output.ndim > 0:
            decision = float(raw_output[0])
        else:
            decision = float(raw_output)

        selected_threshold = threshold if threshold is not None else self._best_threshold or 0.5
        prediction = int(decision >= selected_threshold)

        return PredictionResult(
            prediction=prediction,
            probability=probability,
            model_version=self._model_version,
            raw_output=raw_output,
        )

    def _load_feature_metadata(self, run_id: str) -> FeatureMetadata:
        feature_names: list[str] = []
        categorical: list[str] = []
        datetime_columns: list[str] = []
        reference_date: str | None = None

        if mlflow_artifacts is None:
            raise RuntimeError("mlflow artifacts module unavailable")

        try:
            metadata_json = mlflow_artifacts.load_text(f"runs:/{run_id}/feature_metadata.json")
            metadata = FeatureMetadata.from_dict(json.loads(metadata_json))
            return metadata
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to load feature metadata, falling back to feature names: %s", exc)

        try:
            feature_text = mlflow_artifacts.load_text(f"runs:/{run_id}/feature_names.txt")
            feature_names = [line.strip() for line in feature_text.splitlines() if line.strip()]
        except Exception as exc:  # noqa: BLE001
            logging.error("Unable to load feature names artifact: %s", exc)

        return FeatureMetadata(
            feature_names=feature_names,
            categorical_features=categorical,
            reference_date=reference_date,
            datetime_columns=datetime_columns,
        )
