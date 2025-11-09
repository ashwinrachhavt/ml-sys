from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import extract_data_paths, load_config
from app.data.base_loader import BaseDataLoader


class DataFrameLoader(BaseDataLoader):
    """Load CSV datasets defined in the configuration file."""

    def _read_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.read_csv(path)

    def load(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        config_path: Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load configured CSV datasets into pandas DataFrames."""

        resolved_config = dict(config) if config is not None else load_config(config_path)
        data_paths = extract_data_paths(resolved_config, config_path=config_path)

        return {name: self._read_csv(path) for name, path in data_paths.items()}
