"""XGBoost estimator construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xgboost import XGBClassifier

from .base import merge_params


def build_xgboost_classifier(
    *,
    params: Mapping[str, Any] | None,
    random_state: int,
    scale_pos_weight: float,
) -> XGBClassifier:
    """Construct an ``XGBClassifier`` with sensible defaults."""

    defaults = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_jobs": -1,
        "random_state": random_state,
        "scale_pos_weight": scale_pos_weight,
        "use_label_encoder": False,
    }

    final_params = merge_params(defaults, params)
    return XGBClassifier(**final_params)


__all__ = ["build_xgboost_classifier"]
