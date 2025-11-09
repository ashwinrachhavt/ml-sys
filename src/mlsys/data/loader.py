"""Base classes and high level dataset loader orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import DataLoader
from .registry import DataLoaderRegistry


class DatasetLoader:
    """High-level loader that can ingest multiple named datasets."""

    def __init__(self, loader: DataLoader):
        self.loader = loader

    def load_many(self, sources: dict[str, Path], **kwargs: Any) -> dict[str, pd.DataFrame]:
        datasets: dict[str, pd.DataFrame] = {}
        for name, path in sources.items():
            try:
                datasets[name] = self.loader.load(path, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"Failed loading dataset '{name}' from '{path}'") from exc
        return datasets

    @classmethod
    def from_config(
        cls, loader_type: str, sources: dict[str, Path], **loader_kwargs: Any
    ) -> tuple[DatasetLoader, dict[str, pd.DataFrame]]:
        loader = DataLoaderRegistry.create(loader_type, **loader_kwargs)
        instance = cls(loader)
        return instance, instance.load_many(sources)


__all__ = ["DataLoader", "DatasetLoader"]
