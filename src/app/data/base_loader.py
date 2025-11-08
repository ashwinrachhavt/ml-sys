from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


class BaseDataLoader(ABC):
    """Abstract interface for loading raw tabular datasets."""

    @abstractmethod
    def load(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        config_path: Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load all required raw datasets."""
        raise NotImplementedError
