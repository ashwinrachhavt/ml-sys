"""Base class for data loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class DataLoader(ABC):
    """Abstract loader definition."""

    @abstractmethod
    def load(self, path: Path, **kwargs: Any) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError


__all__ = ["DataLoader"]
