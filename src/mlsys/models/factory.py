"""Factory for selecting estimators based on configuration."""

from __future__ import annotations

from typing import Any

from mlsys.core.config import ModelConfig

from .xgboost import build_xgboost_classifier


def build_estimator(model_cfg: ModelConfig, *, random_state: int, scale_pos_weight: float) -> Any:
    """Return an estimator instance matching the requested model type."""

    if model_cfg.type == "xgboost":
        return build_xgboost_classifier(
            params=model_cfg.params,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )

    raise NotImplementedError(f"Model type '{model_cfg.type}' is not yet supported.")


__all__ = ["build_estimator"]
