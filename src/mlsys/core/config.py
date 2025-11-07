"""Configuration models for the ml-sys framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration controlling how raw data is located and split."""

    customers_path: Path = Field(default=Path("data/customers.csv"))
    noncustomers_path: Path = Field(default=Path("data/noncustomers.csv"))
    usage_path: Path = Field(default=Path("data/usage_actions.csv"))
    target_column: str = Field(default="is_customer")
    stratify: bool = Field(default=True)
    test_size: float = Field(default=0.2, ge=0.05, le=0.4)
    calibration_size: float = Field(default=0.1, ge=0.0, le=0.4)
    random_state: int = Field(default=42)


class FeatureConfig(BaseModel):
    """Configuration for feature engineering and preprocessing."""

    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    log_transform: list[str] = Field(default_factory=list)
    interactions: list[tuple[str, str]] = Field(default_factory=list)
    apply_smote: bool = Field(default=True)


class ModelConfig(BaseModel):
    """Model selection and hyperparameters."""

    type: Literal["xgboost", "catboost", "random_forest", "logistic_regression"] = "xgboost"
    params: dict[str, Any] = Field(default_factory=dict)
    calibration_method: Literal["none", "sigmoid", "isotonic"] = "sigmoid"
    tune_hyperparameters: bool = Field(default=False)


class TrackingConfig(BaseModel):
    """Experiment tracking configuration (MLflow or similar)."""

    tracking_uri: str = Field(default="mlruns")
    experiment_name: str = Field(default="lead_scoring")
    run_name: str | None = Field(default=None)
    log_artifacts: bool = Field(default=True)


class ServingConfig(BaseModel):
    """Inference service configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False)
    model_registry_path: Path = Field(default=Path("models/registry"))
    prometheus_enabled: bool = Field(default=True)


class FrameworkConfig(BaseModel):
    """Top-level configuration for the ml-sys framework."""

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    artifacts_path: Path = Field(default=Path("artifacts"))

    @classmethod
    def from_yaml(cls, path: Path | str) -> FrameworkConfig:
        """Load configuration from a YAML file."""

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cast("FrameworkConfig", cls.model_validate(data))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> FrameworkConfig:
        """Build configuration from an in-memory mapping (useful for tests)."""

        return cast("FrameworkConfig", cls.model_validate(mapping))

    def save_yaml(self, path: Path | str) -> None:
        """Persist configuration to disk for experiment provenance."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False, allow_unicode=False)

    def resolve_artifact_path(self, *parts: str) -> Path:
        """Resolve a subpath within the artifacts directory."""

        return (self.artifacts_path / Path(*parts)).resolve()


__all__ = [
    "DataConfig",
    "FeatureConfig",
    "ModelConfig",
    "TrackingConfig",
    "ServingConfig",
    "FrameworkConfig",
]
