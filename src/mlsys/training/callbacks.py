"""Training callback system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class TrainingCallback(Protocol):
    """Callback interface triggered during the training lifecycle."""

    def on_train_start(self, context: TrainingContext) -> None: ...  # pragma: no cover - interface

    def on_model_trained(self, context: TrainingContext, model_name: str, metrics: dict[str, float]) -> None: ...

    def on_train_end(self, context: TrainingContext) -> None: ...


@dataclass
class TrainingContext:
    """Context propagated to callbacks containing artefacts of interest."""

    x_train: pd.DataFrame
    y_train: pd.Series
    feature_metadata: dict[str, object]


class NullCallback:
    """No-op callback used as default."""

    def on_train_start(self, context: TrainingContext) -> None:  # pragma: no cover - trivial
        return

    def on_model_trained(self, context: TrainingContext, model_name: str, metrics: dict[str, float]) -> None:
        return

    def on_train_end(self, context: TrainingContext) -> None:
        return


__all__ = ["TrainingCallback", "TrainingContext", "NullCallback"]
