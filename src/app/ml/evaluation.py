from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, Any]:
    """Compute core classification metrics for binary problems."""

    results: dict[str, Any] = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        results.update(
            {
                "roc_auc": metrics.roc_auc_score(y_true, y_proba),
                "pr_auc": metrics.average_precision_score(y_true, y_proba),
                "log_loss": metrics.log_loss(y_true, y_proba),
            }
        )

    return results
