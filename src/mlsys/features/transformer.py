"""Composable feature transformers with registry support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

import pandas as pd


class FeatureTransformer(ABC):
    """Interface for all feature transformers."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> FeatureTransformer:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class TransformerRegistry:
    """Name based registry for transformer classes."""

    _registry: ClassVar[dict[str, type[FeatureTransformer]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[FeatureTransformer]], type[FeatureTransformer]]:
        def decorator(transformer_cls: type[FeatureTransformer]) -> type[FeatureTransformer]:
            if name in cls._registry:
                raise ValueError(f"Transformer '{name}' already registered")
            cls._registry[name] = transformer_cls
            return transformer_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> FeatureTransformer:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<empty>"
            raise KeyError(f"Unknown transformer '{name}'. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry)


@dataclass
class FeatureMetadata:
    """Simple metadata container for downstream artefacts."""

    feature_names: list[str]
    categorical_features: list[str]
    datetime_columns: list[str]
    reference_date: str | None
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "datetime_columns": self.datetime_columns,
            "reference_date": self.reference_date,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureMetadata:
        return cls(
            feature_names=list(payload.get("feature_names", [])),
            categorical_features=list(payload.get("categorical_features", [])),
            datetime_columns=list(payload.get("datetime_columns", [])),
            reference_date=payload.get("reference_date"),
            n_samples=int(payload.get("n_samples", 0)),
        )


@TransformerRegistry.register("datetime")
class DateTimeTransformer(FeatureTransformer):
    """Convert datetime columns into age (days)."""

    def __init__(self, columns: list[str], reference_date: str | None = None):
        self.columns = columns
        self.reference_date = reference_date
        self._pivot: datetime | None = None

    def fit(self, df: pd.DataFrame) -> DateTimeTransformer:
        if self.reference_date:
            self._pivot = datetime.fromisoformat(self.reference_date)
        else:
            candidates = []
            for column in self.columns:
                if column in df.columns:
                    dates = pd.to_datetime(df[column], errors="coerce")
                    if dates.notna().any():
                        candidates.append(dates.max())
            self._pivot = max(candidates) if candidates else datetime.now()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        for column in self.columns:
            if column not in transformed.columns:
                continue
            series = pd.to_datetime(transformed[column], errors="coerce")
            transformed[column] = (self._pivot - series).dt.days.fillna(0).astype(int)
        return transformed


@TransformerRegistry.register("categorical")
class CategoricalTransformer(FeatureTransformer):
    """Encode categorical columns using one-hot or label encoding."""

    def __init__(self, columns: list[str], encoding: str = "onehot", drop_first: bool = False):
        self.columns = columns
        self.encoding = encoding
        self.drop_first = drop_first
        self._categories: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> CategoricalTransformer:
        for column in self.columns:
            if column in df.columns:
                self._categories[column] = df[column].dropna().astype(str).unique().tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        available = [column for column in self.columns if column in transformed.columns]
        if not available:
            return transformed

        if self.encoding == "onehot":
            transformed = pd.get_dummies(transformed, columns=available, drop_first=self.drop_first)
        elif self.encoding == "label":
            for column in available:
                transformed[column] = pd.Categorical(
                    transformed[column].astype(str), categories=self._categories.get(column, [])
                ).codes
        else:
            raise ValueError(f"Unsupported encoding strategy '{self.encoding}'")
        return transformed


@TransformerRegistry.register("aggregation")
class AggregationTransformer(FeatureTransformer):
    """Group-based aggregations."""

    def __init__(self, group_by: str, agg_columns: dict[str, str]):
        self.group_by = group_by
        self.agg_columns = agg_columns

    def fit(self, df: pd.DataFrame) -> AggregationTransformer:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.group_by not in df.columns:
            return df
        applicable = {col: func for col, func in self.agg_columns.items() if col in df.columns}
        if not applicable:
            return df
        aggregated = df.groupby(self.group_by, as_index=False).agg(applicable)
        return aggregated


@TransformerRegistry.register("fillna")
class FillNATransformer(FeatureTransformer):
    """Missing value imputation."""

    def __init__(self, strategy: str = "mean", fill_value: Any | None = None, columns: list[str] | None = None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self._values: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> FillNATransformer:
        target_columns = self.columns or df.select_dtypes(include=["number"]).columns.tolist()
        for column in target_columns:
            if column not in df.columns:
                continue
            series = df[column]
            if self.strategy == "mean":
                self._values[column] = float(series.mean())
            elif self.strategy == "median":
                self._values[column] = float(series.median())
            elif self.strategy == "mode":
                self._values[column] = series.mode().iloc[0] if not series.mode().empty else 0
            elif self.strategy == "constant":
                self._values[column] = self.fill_value
            else:
                raise ValueError(f"Unsupported fill strategy '{self.strategy}'")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        for column, value in self._values.items():
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna(value)
        return transformed


__all__ = [
    "FeatureTransformer",
    "TransformerRegistry",
    "FeatureMetadata",
    "DateTimeTransformer",
    "CategoricalTransformer",
    "AggregationTransformer",
    "FillNATransformer",
]
