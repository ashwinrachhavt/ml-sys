"""Base classes and high level dataset loader orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from .base import DataLoader
from .registry import DataLoaderRegistry


class DatasetLoader:
    """High-level loader that can ingest multiple named datasets."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._logger = logging.getLogger(__name__)

    def load_many(self, sources: dict[str, Path], **kwargs: Any) -> dict[str, pd.DataFrame]:
        """Load each configured dataset and provide targeted error messages."""

        datasets: dict[str, pd.DataFrame] = {}
        for name, path in sources.items():
            resolved = Path(path)
            if not resolved.exists():
                message = f"Dataset '{name}' not found at path: {resolved}"
                raise FileNotFoundError(message)

            try:
                self._logger.debug("Loading dataset '%s' from %s", name, resolved)
                frame = self.loader.load(resolved, **kwargs)
            except EmptyDataError as exc:  # pragma: no cover - exercised in tests
                raise RuntimeError(f"Dataset '{name}' at '{resolved}' is empty") from exc
            except ParserError as exc:  # pragma: no cover - exercised in tests
                raise RuntimeError(f"Failed to parse dataset '{name}' from '{resolved}'") from exc
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.exception("Unexpected error loading dataset '%s'", name)
                raise RuntimeError(f"Failed loading dataset '{name}' from '{resolved}'") from exc

            datasets[name] = frame
            self._logger.info("Loaded %s rows for dataset '%s'", len(frame), name)
        return datasets

    @classmethod
    def from_config(
        cls, loader_type: str, sources: dict[str, Path], **loader_kwargs: Any
    ) -> tuple[DatasetLoader, dict[str, pd.DataFrame]]:
        loader = DataLoaderRegistry.create(loader_type, **loader_kwargs)
        instance = cls(loader)
        return instance, instance.load_many(sources)


__all__ = ["DataLoader", "DatasetLoader"]
