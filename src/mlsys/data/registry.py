"""Registry for pluggable data loader implementations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from .base import DataLoader


class DataLoaderRegistry:
    """Simple name -> loader class registry."""

    _registry: ClassVar[dict[str, type[DataLoader]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[DataLoader]], type[DataLoader]]:
        def decorator(loader_cls: type[DataLoader]) -> type[DataLoader]:
            if name in cls._registry:
                raise ValueError(f"Loader '{name}' already registered")
            cls._registry[name] = loader_cls
            return loader_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> DataLoader:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<empty>"
            raise KeyError(f"Unknown data loader '{name}'. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry)


@DataLoaderRegistry.register("csv")
class CSVDataLoader(DataLoader):
    """CSV backed loader supporting typical pandas options."""

    def __init__(self, sep: str = ",", encoding: str = "utf-8", **read_kwargs: Any):
        self.sep = sep
        self.encoding = encoding
        self.read_kwargs = read_kwargs

    def load(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        merged = {**self.read_kwargs, **kwargs}
        return pd.read_csv(path, sep=self.sep, encoding=self.encoding, **merged)


@DataLoaderRegistry.register("parquet")
class ParquetDataLoader(DataLoader):
    """Parquet backed loader."""

    def __init__(self, **read_kwargs: Any):
        self.read_kwargs = read_kwargs

    def load(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        merged = {**self.read_kwargs, **kwargs}
        return pd.read_parquet(path, **merged)


@DataLoaderRegistry.register("json")
class JSONDataLoader(DataLoader):
    """Semi-structured JSON loader."""

    def __init__(self, orient: str = "records", **read_kwargs: Any):
        self.orient = orient
        self.read_kwargs = read_kwargs

    def load(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        merged = {**self.read_kwargs, **kwargs}
        return pd.read_json(path, orient=self.orient, **merged)


__all__ = ["DataLoaderRegistry"]
