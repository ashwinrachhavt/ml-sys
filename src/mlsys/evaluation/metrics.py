"""Metric utilities for classification tasks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn import metrics

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def available_metrics() -> dict[str, MetricFn]:
    return {
        "accuracy": metrics.accuracy_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
        "f1": metrics.f1_score,
        "roc_auc": metrics.roc_auc_score,
        "pr_auc": lambda y_true, y_score: metrics.average_precision_score(y_true, y_score),
    }


__all__ = ["available_metrics", "MetricFn"]
