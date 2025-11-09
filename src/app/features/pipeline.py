from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class FeatureMetadata:
    feature_names: list[str]
    categorical_features: list[str]
    reference_date: str | None
    datetime_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureMetadata:
        return cls(
            feature_names=list(payload.get("feature_names", [])),
            categorical_features=list(payload.get("categorical_features", [])),
            reference_date=payload.get("reference_date"),
            datetime_columns=list(payload.get("datetime_columns", [])),
        )


@dataclass
class FeatureMatrix:
    """Train/validation feature matrices produced by the pipeline."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    metadata: FeatureMetadata


class FeaturePipeline:
    """Compose raw tabular inputs into model-ready feature matrices."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.data_config = config.get("data", {})
        self.features_config = config.get("features", {})
        self.training_config = config.get("training", {})

    def build_feature_matrix(self, datasets: dict[str, pd.DataFrame]) -> FeatureMatrix:
        customers = datasets.get("customers")
        noncustomers = datasets.get("noncustomers")
        usage = datasets.get("usage_actions")

        if customers is None or usage is None:
            raise KeyError("customers and usage_actions datasets are required")

        id_column = self.data_config.get("id_column", "id")
        target_column = self.data_config.get("target_column", "is_customer")

        base = self._prepare_base_dataframe(customers, noncustomers, target_column, id_column)
        usage_processed = self._prepare_usage_features(usage, id_column)
        merged = base.merge(usage_processed, on=id_column, how="left").fillna(0)

        if target_column not in merged.columns:
            raise KeyError(f"Target column '{target_column}' not found in merged dataset")

        y = merged[target_column]
        x = merged.drop(columns=[target_column])

        x = self._encode_categories(x)
        x, datetime_columns = self._finalize_features(x)

        random_state = self.training_config.get("random_state", 42)
        val_size = self.training_config.get("val_size", 0.2)
        test_size = self.training_config.get("test_size", 0.2)

        stratify = self.training_config.get("stratify", True)
        stratify_target = y if stratify else None

        # first split: hold-out test set
        x_temp, x_test, y_temp, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            stratify=stratify_target,
            random_state=random_state,
        )

        # update stratify target for validation split
        stratify_val = y_temp if stratify else None
        remaining = 1 - test_size
        if remaining <= 0:
            raise ValueError("training.val_size + training.test_size must be less than 1")

        val_ratio = min(1.0, max(0.0, val_size / remaining))

        x_train, x_val, y_train, y_val = train_test_split(
            x_temp,
            y_temp,
            test_size=val_ratio,
            stratify=stratify_val,
            random_state=random_state,
        )

        feature_names = list(x.columns)
        metadata = FeatureMetadata(
            feature_names=feature_names,
            categorical_features=self.features_config.get("categorical", []),
            reference_date=self.features_config.get("reference_date"),
            datetime_columns=datetime_columns,
        )

        return FeatureMatrix(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=feature_names,
            metadata=metadata,
        )

    def _prepare_base_dataframe(
        self,
        customers: pd.DataFrame,
        noncustomers: pd.DataFrame | None,
        target_column: str,
        id_column: str,
    ) -> pd.DataFrame:
        base = customers.copy()

        if id_column not in base.columns:
            raise KeyError(f"ID column '{id_column}' not found in customers dataset")

        if target_column not in base.columns:
            if noncustomers is None:
                raise KeyError(
                    f"Target column '{target_column}' missing and noncustomers dataset not provided to infer labels"
                )
            base[target_column] = 1

        frames: list[pd.DataFrame] = [base]
        if noncustomers is not None:
            noncustomers_frame = noncustomers.copy()
            if id_column not in noncustomers_frame.columns:
                raise KeyError(f"ID column '{id_column}' not found in noncustomers dataset")
            if target_column not in noncustomers_frame.columns:
                noncustomers_frame[target_column] = 0
            frames.append(noncustomers_frame)

        combined = pd.concat(frames, ignore_index=True, sort=False)
        if target_column not in combined.columns:
            raise KeyError(f"Target column '{target_column}' not found after combining datasets")

        return combined

    def _prepare_usage_features(self, usage: pd.DataFrame, id_column: str) -> pd.DataFrame:
        usage_features = usage.copy()

        if "WHEN_TIMESTAMP" in usage_features.columns:
            usage_features["WHEN_TIMESTAMP"] = pd.to_datetime(usage_features["WHEN_TIMESTAMP"], errors="coerce")
            reference_date_str = self.features_config.get("reference_date")
            reference_date = (
                datetime.fromisoformat(reference_date_str)
                if reference_date_str
                else usage_features["WHEN_TIMESTAMP"].max()
            )
            usage_features["days_since_last_action"] = (
                reference_date - usage_features["WHEN_TIMESTAMP"]
            ).dt.days.fillna(0)

        aggregate_columns = {col: "sum" for col in usage_features.columns if col not in {id_column, "WHEN_TIMESTAMP"}}

        grouped = usage_features.groupby(id_column, as_index=False).agg(aggregate_columns)
        return grouped

    def _encode_categories(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        categorical_features = self.features_config.get("categorical", [])
        if not categorical_features:
            return dataframe

        existing = [col for col in categorical_features if col in dataframe.columns]
        if not existing:
            return dataframe

        encoded = pd.get_dummies(dataframe, columns=existing, drop_first=False)
        return encoded

    def _finalize_features(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Ensure all features are numeric and handle temporal columns."""

        frame = dataframe.copy()
        converted_datetime: list[str] = []

        reference_date_str = self.features_config.get("reference_date")
        reference_date = datetime.fromisoformat(reference_date_str) if reference_date_str else None

        for column in list(frame.columns):
            series = frame[column]

            if pd.api.types.is_datetime64_any_dtype(series):
                ref = reference_date or series.max()
                frame[column] = (ref - series).dt.days.fillna(0)
                converted_datetime.append(column)
                continue

            if series.dtype == object:
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().any():
                    ref = reference_date or parsed.max()
                    frame[column] = (ref - parsed).dt.days.fillna(0)
                    converted_datetime.append(column)
                else:
                    frame.drop(columns=[column], inplace=True)

        return frame, converted_datetime
