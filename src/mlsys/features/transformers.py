"""Feature engineering helpers for lead scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlsys.core.config import DataConfig
from mlsys.data.loaders import RawTables, load_raw_tables

USAGE_PREFIXES: tuple[str, ...] = ("ACTIONS_", "USERS_")


def merge_raw_tables(raw: RawTables) -> pd.DataFrame:
    """Merge customers, noncustomers, and usage tables into a single frame."""

    customers = raw.customers.copy()
    noncustomers = raw.noncustomers.copy()

    customers["is_customer"] = 1
    noncustomers["is_customer"] = 0

    data = pd.concat([customers, noncustomers], ignore_index=True, sort=False)

    data = data.drop(columns=["CLOSEDATE", "MRR"], errors="ignore")

    usage_feature_cols = [col for col in raw.usage.columns if any(col.startswith(prefix) for prefix in USAGE_PREFIXES)]

    usage_agg = raw.usage.groupby("id")[usage_feature_cols].sum(min_count=1).reset_index()

    action_cols = [col for col in usage_feature_cols if col.startswith("ACTIONS_")]
    user_cols = [col for col in usage_feature_cols if col.startswith("USERS_")]

    usage_agg["ACTIONS_TOTAL"] = usage_agg[action_cols].sum(axis=1)
    usage_agg["USERS_TOTAL"] = usage_agg[user_cols].sum(axis=1)

    data = data.merge(usage_agg, on="id", how="left")

    usage_cols = [col for col in data.columns if any(col.startswith(prefix) for prefix in USAGE_PREFIXES)]
    data[usage_cols] = data[usage_cols].fillna(0)

    for cat_col in ["EMPLOYEE_RANGE", "INDUSTRY"]:
        if cat_col in data.columns:
            data[cat_col] = data[cat_col].fillna("UNKNOWN")

    data["ALEXA_RANK"] = data["ALEXA_RANK"].replace(16000001, np.nan)
    data["ALEXA_RANK_LOG1P"] = np.log1p(data["ALEXA_RANK"])

    return data


def build_feature_matrix(
    data_config: DataConfig | None = None,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and target series ready for modeling."""

    raw_tables = load_raw_tables(data_config=data_config, data_dir=data_dir)
    merged = merge_raw_tables(raw_tables)
    target = merged.pop("is_customer")
    features = merged.drop(columns=["id"], errors="ignore")
    return features, target


def infer_categorical_features(features: pd.DataFrame) -> list[str]:
    """Identify columns that should be treated as categorical."""

    categorical_cols: list[str] = []
    for col in features.columns:
        dtype = features[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            categorical_cols.append(col)
    return categorical_cols


__all__ = [
    "USAGE_PREFIXES",
    "build_feature_matrix",
    "infer_categorical_features",
    "merge_raw_tables",
]
