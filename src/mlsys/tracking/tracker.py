"""Abstract tracking interface so the trainer is backend agnostic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class ExperimentTracker(ABC):
    """Abstract experiment tracker."""

    @contextmanager
    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
        self._on_start(run_name=run_name, tags=tags)
        try:
            yield
        finally:
            self._on_end()

    @abstractmethod
    def _on_start(self, run_name: str, tags: dict[str, str] | None = None) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def _on_end(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        """Optional artifact logging."""
        return None

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        """Optional logging for dictionaries."""
        return None

    def register_model(self, model_uri: str, name: str, stage: str | None = None) -> None:
        """Optional registry integration."""
        return None


class NullTracker(ExperimentTracker):
    """Fallback tracker that ignores all logging calls."""

    def _on_start(self, run_name: str, tags: dict[str, str] | None = None) -> None:  # pragma: no cover - trivial
        return

    def _on_end(self) -> None:  # pragma: no cover - trivial
        return

    def log_params(self, params: dict[str, Any]) -> None:  # pragma: no cover - trivial
        return

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:  # pragma: no cover - trivial
        return


__all__ = ["ExperimentTracker", "NullTracker"]
