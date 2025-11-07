"""Common helpers for estimator construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def merge_params(defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge user-supplied hyperparameters with defaults."""

    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


__all__ = ["merge_params"]
