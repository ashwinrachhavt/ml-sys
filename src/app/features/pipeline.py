from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class FeatureMatrix:
    """Train/validation feature matrices produced by the pipeline."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    feature_names: list[str]


class FeaturePipeline:
    """Compose raw tabular inputs into model-ready feature matrices."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.data_config = config.get("data", {})
        self.features_config = config.get("features", {})
        self.training_config = config.get("training", {})

    def build_feature_matrix(self, datasets: dict[str, pd.DataFrame]) -> FeatureMatrix:
        customers = datasets.get("customers")
        usage = datasets.get("usage_actions")

        if customers is None or usage is None:
            raise KeyError("customers and usage_actions datasets are required")

        id_column = self.data_config.get("id_column", "id")
        target_column = self.data_config.get("target_column", "converted")

        base = customers.copy()
        usage_processed = self._prepare_usage_features(usage, id_column)
        merged = base.merge(usage_processed, on=id_column, how="left").fillna(0)

        if target_column not in merged.columns:
            raise KeyError(f"Target column '{target_column}' not found in merged dataset")

        y = merged[target_column]
        x = merged.drop(columns=[target_column])

        x = self._encode_categories(x)

        random_state = self.training_config.get("random_state", 42)
        val_size = self.training_config.get("val_size", 0.2)
        stratify_flag = None
        if self.training_config.get("stratify", True):
            min_class = y.value_counts().min()
            val_count = max(1, int(round(len(y) * val_size)))
            if min_class >= 2 and val_count >= y.nunique():
                stratify_flag = y

        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=val_size,
            stratify=stratify_flag,
            random_state=random_state,
        )

        return FeatureMatrix(
            x_train=x_train,
            x_val=x_val,
            y_train=y_train,
            y_val=y_val,
            feature_names=list(x.columns),
        )

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
