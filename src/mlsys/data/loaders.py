"""Raw data loading helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mlsys.config.paths import DATA_DIR
from mlsys.core.config import DataConfig
from mlsys.training.stub_data import load_stub_tables


@dataclass
class RawTables:
    """Container for the three raw CSV tables used in lead scoring."""

    customers: pd.DataFrame
    noncustomers: pd.DataFrame
    usage: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV with UTF-8 encoding and helpful error context."""

    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Required CSV file not found: {path}") from exc


def load_raw_tables(
    data_config: DataConfig | None = None,
    data_dir: Path | None = None,
    *,
    allow_stub: bool = True,
) -> RawTables:
    """Load customers, noncustomers, and usage tables.

    Preference order:
    1. Explicit paths provided by ``data_config``.
    2. Directory provided via ``data_dir``.
    3. Project default data directory.
    4. Synthetic stub tables (if ``allow_stub`` is True).
    """

    if data_config is not None:
        try:
            return RawTables(
                customers=_read_csv(data_config.customers_path),
                noncustomers=_read_csv(data_config.noncustomers_path),
                usage=_read_csv(data_config.usage_path),
            )
        except FileNotFoundError:
            warnings.warn(
                "Configured data files were not found; using fallback sources instead.",
                stacklevel=2,
            )

    candidate_dir = data_dir or DATA_DIR
    if candidate_dir is not None and candidate_dir.exists():
        try:
            return RawTables(
                customers=_read_csv(candidate_dir / "customers.csv"),
                noncustomers=_read_csv(candidate_dir / "noncustomers.csv"),
                usage=_read_csv(candidate_dir / "usage_actions.csv"),
            )
        except FileNotFoundError:
            warnings.warn(
                "Raw CSV files missing in provided directory; using fallback sources instead.",
                stacklevel=2,
            )

    if not allow_stub:
        raise FileNotFoundError("No valid data sources available and stub data disabled.")

    customers, noncustomers, usage = load_stub_tables()
    return RawTables(customers=customers, noncustomers=noncustomers, usage=usage)


__all__ = ["RawTables", "load_raw_tables"]
