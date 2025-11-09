"""Type-safe configuration management leveraging Pydantic Settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    """Project metadata and bookkeeping."""

    name: str
    version: str = "1.0.0"
    description: str = ""


class DataConfig(BaseModel):
    """Configuration concerning data ingestion."""

    loader_type: str = "csv"
    sources: dict[str, str]
    id_column: str = "id"
    target_column: str = "target"

    @field_validator("sources")
    @classmethod
    def _ensure_sources(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("At least one data source must be provided")
        return value


class FeatureTransformerConfig(BaseModel):
    """Config block for a single transformer entry."""

    type: str
    columns: list[str] | None = None
    reference_date: str | None = None
    encoding: str | None = None
    group_by: str | None = None
    agg_columns: dict[str, str] | None = None

    class Config:
        extra = "allow"  # allow transformer specific kwargs


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    transformers: list[FeatureTransformerConfig] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Single model entry plus hyper-parameters."""

    name: str
    type: str
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)
    enabled: bool = True


class TrainingConfig(BaseModel):
    """Training orchestration settings."""

    models: list[ModelConfig]
    cv_folds: int = 5
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    handle_imbalance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("test_size", "val_size")
    @classmethod
    def _validate_fraction(cls, value: float) -> float:
        if not 0 < value < 1:
            raise ValueError("Split ratios must be in (0, 1)")
        return value


class EvaluationConfig(BaseModel):
    """Evaluation and metric selection."""

    primary_metric: str = "roc_auc"
    threshold_metric: str = "f1"
    metrics: list[str] = Field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
    )


class TrackingRegistryConfig(BaseModel):
    """Model registry specifics for the tracker backend."""

    model_name: str
    auto_register: bool = True
    stage: Literal["Staging", "Production", "Archived"] = "Production"


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""

    backend: Literal["mlflow", "wandb", "none"] = "mlflow"
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "default"
    run_name_prefix: str = "run"
    registry: TrackingRegistryConfig | None = None


class ServingConfig(BaseModel):
    """Prediction service configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str
    model_stage: Literal["Staging", "Production", "Archived"] = "Production"
    enable_metrics: bool = True
    workers: int = 1
    local_model_path: str | None = None


class Settings(BaseSettings):
    """Application wide settings.

    Values load from YAML first and can be overridden using env vars prefixed
    with ``MLSYS_``. Nested values can be overridden using ``__`` as delimiter.
    """

    model_config = SettingsConfigDict(
        env_prefix="MLSYS_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    _config_path: Path | None = None

    project: ProjectConfig
    data: DataConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    training: TrainingConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    serving: ServingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        if not isinstance(payload, dict):
            raise ValueError("Configuration file must contain a mapping at the root")

        settings = cls(**payload)
        settings._config_path = config_path  # pyright: ignore[attr-defined]
        return settings

    def resolve_path(self, raw: str, relative_to: Path | None = None) -> Path:
        """Resolve a path relative to the config file or provided base."""

        base = relative_to
        if base is None:
            base = getattr(self, "_config_path", None)
            if base is not None:
                base = base.parent
            else:
                base = Path.cwd()
        return (base / raw).resolve()

    def data_paths(self, relative_to: Path | None = None) -> dict[str, Path]:
        """Return fully resolved data source paths."""

        return {name: self.resolve_path(path, relative_to) for name, path in self.data.sources.items()}


__all__ = ["Settings"]
