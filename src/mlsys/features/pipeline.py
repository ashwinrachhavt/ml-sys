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
        """Append a transformer to the pipeline."""

        self.transformers.append(transformer)
        return self

    def build(
        self, datasets: dict[str, pd.DataFrame], id_column: str, target_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Combine datasets, apply transformers, and return features/target."""

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
        """Compose the model matrix from the expected dataset inputs."""

        base = self._load_base_dataset(datasets, target_column)
        base = self._append_negative_samples(base, datasets.get("noncustomers"), target_column)
        base = self._merge_usage_features(base, datasets.get("usage_actions"), id_column)
        return self._fill_numeric_gaps(base)

    def _load_base_dataset(self, datasets: dict[str, pd.DataFrame], target_column: str) -> pd.DataFrame:
        """Return a copy of the mandatory customers dataset with target defaults."""

        customers = datasets.get("customers")
        if customers is None:
            raise ValueError("'customers' dataset is mandatory")

        base = customers.copy()
        if target_column not in base.columns:
            base[target_column] = 1
        else:
            base[target_column] = base[target_column].fillna(1)
        return base

    def _append_negative_samples(
        self, base: pd.DataFrame, noncustomers: pd.DataFrame | None, target_column: str
    ) -> pd.DataFrame:
        """Append non-customer records as negative training examples."""

        if noncustomers is None:
            return base

        aligned = noncustomers.copy()
        missing = set(base.columns) - set(aligned.columns)
        for column in missing:
            aligned[column] = None

        aligned = aligned.reindex(columns=base.columns, fill_value=None)
        aligned[target_column] = 0
        combined = pd.concat([base, aligned], ignore_index=True, sort=False)
        return combined

    def _merge_usage_features(self, base: pd.DataFrame, usage: pd.DataFrame | None, id_column: str) -> pd.DataFrame:
        """Merge aggregated usage metrics into the base dataset."""

        if usage is None or id_column not in usage.columns:
            return base

        numeric_columns = usage.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_columns:
            return base

        grouped = usage.loc[:, [id_column] + numeric_columns].groupby(id_column, as_index=False).sum()
        return base.merge(grouped, on=id_column, how="left")

    def _fill_numeric_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure numeric columns have zero-filled missing values."""

        result = df.copy()
        numeric_cols = result.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            result.loc[:, numeric_cols] = result.loc[:, numeric_cols].fillna(0)
        return result

    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce object columns into numeric representations when possible."""

        result = df.copy()
        for column in list(result.columns):
            if result[column].dtype != object:
                continue

            coerced = pd.to_numeric(result[column], errors="coerce")
            if coerced.notna().all():
                result[column] = coerced
                continue

            datetime_series = pd.to_datetime(result[column], errors="coerce")
            if datetime_series.notna().any():
                reference = datetime_series.max()
                result[column] = (reference - datetime_series).dt.days.fillna(0).astype(int)
                continue

            result = result.drop(columns=[column])
        return result


__all__ = ["FeaturePipeline", "SplitData"]
