"""Model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .metrics import available_metrics


@dataclass
class EvaluationResult:
    """Container for evaluation metrics and threshold information."""

    metrics: dict[str, float]
    threshold: float | None


class Evaluator:
    """Compute metrics based on evaluation config."""

    def __init__(self, metric_names: list[str], threshold_metric: str):
        self.metric_names = metric_names
        self.threshold_metric = threshold_metric
        self.metric_fns = available_metrics()

    def evaluate(
        self,
        y_true: pd.Series,
        y_scores: np.ndarray,
        *,
        thresholds: list[float] | None = None,
    ) -> EvaluationResult:
        metrics: dict[str, float] = {}
        y_pred = (y_scores >= 0.5).astype(int)

        for name in self.metric_names:
            metric_fn = self.metric_fns.get(name)
            if metric_fn is None:
                continue
            if name in {"roc_auc", "pr_auc"}:
                metrics[name] = float(metric_fn(y_true, y_scores))
            else:
                metrics[name] = float(metric_fn(y_true, y_pred))

        best_threshold = None
        if thresholds:
            best_score = -float("inf")
            metric_fn = self.metric_fns.get(self.threshold_metric, f1_score)
            for threshold in thresholds:
                preds = (y_scores >= threshold).astype(int)
                score = float(metric_fn(y_true, preds))
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        return EvaluationResult(metrics=metrics, threshold=best_threshold)


__all__ = ["Evaluator", "EvaluationResult"]
