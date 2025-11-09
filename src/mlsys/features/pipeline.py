"""Feature pipeline orchestration for tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .transformer import FeatureMetadata, FeatureTransformer


@dataclass
class SplitData:
    """Convenience container for train/validation/test data."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    metadata: FeatureMetadata


class FeaturePipeline:
    """Apply a series of transformers and create data splits."""

    def __init__(self, transformers: list[FeatureTransformer] | None = None):
        self.transformers = transformers or []
        self.metadata: FeatureMetadata | None = None

    def add(self, transformer: FeatureTransformer) -> FeaturePipeline:
        self.transformers.append(transformer)
        return self

    def build(
        self, datasets: dict[str, pd.DataFrame], id_column: str, target_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        base = self._combine_datasets(datasets, id_column=id_column, target_column=target_column)
        if target_column not in base.columns:
            raise ValueError(f"Target column '{target_column}' missing after combination")

        target = base[target_column]
        features = base.drop(columns=[target_column])

        for transformer in self.transformers:
            features = transformer.fit_transform(features)

        features = self._ensure_numeric(features)
        return features, target

    def split(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        test_size: float,
        val_size: float,
        random_state: int,
        stratify: bool,
        categorical_features: list[str],
        datetime_columns: list[str],
        reference_date: str | None,
    ) -> SplitData:
        stratify_target = target if stratify else None
        features_temp, features_test, target_temp, target_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )

        remaining = 1 - test_size
        val_ratio = val_size / remaining
        stratify_temp = target_temp if stratify else None
        x_train, x_val, y_train, y_val = train_test_split(
            features_temp,
            target_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=stratify_temp,
        )

        metadata = FeatureMetadata(
            feature_names=list(features.columns),
            categorical_features=categorical_features,
            datetime_columns=datetime_columns,
            reference_date=reference_date,
            n_samples=len(features),
        )
        self.metadata = metadata

        return SplitData(
            x_train=x_train,
            x_val=x_val,
            x_test=features_test,
            y_train=y_train,
            y_val=y_val,
            y_test=target_test,
            metadata=metadata,
        )

    def _combine_datasets(self, datasets: dict[str, pd.DataFrame], id_column: str, target_column: str) -> pd.DataFrame:
        customers = datasets.get("customers")
        noncustomers = datasets.get("noncustomers")
        usage = datasets.get("usage_actions")

        if customers is None:
            raise ValueError("'customers' dataset is mandatory")

        base = customers.copy()
        if target_column not in base.columns:
            base[target_column] = 1

        if noncustomers is not None:
            missing = set(base.columns) - set(noncustomers.columns)
            for column in missing:
                noncustomers[column] = None
            noncustomers[target_column] = 0
            base = pd.concat([base, noncustomers], ignore_index=True, sort=False)

        if usage is not None and id_column in usage.columns:
            usage_numeric = usage.select_dtypes(include=["number"]).groupby(id_column, as_index=False).sum()
            base = base.merge(usage_numeric, on=id_column, how="left")

        numeric_cols = base.select_dtypes(include=["number"]).columns
        base[numeric_cols] = base[numeric_cols].fillna(0)
        return base

    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column in list(result.columns):
            if result[column].dtype == object:
                coerced = pd.to_numeric(result[column], errors="coerce")
                if coerced.notna().all():
                    result[column] = coerced
                else:
                    datetime_series = pd.to_datetime(result[column], errors="coerce")
                    if datetime_series.notna().any():
                        reference = datetime_series.max()
                        result[column] = (reference - datetime_series).dt.days.fillna(0).astype(int)
                    else:
                        result = result.drop(columns=[column])
        return result


__all__ = ["FeaturePipeline", "SplitData"]
