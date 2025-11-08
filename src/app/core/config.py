from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the framework configuration from YAML."""

    target_path = config_path or DEFAULT_CONFIG_PATH
    with target_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)

    if not isinstance(loaded, dict):
        raise TypeError("Configuration root must be a mapping")

    return cast(dict[str, Any], loaded)


def resolve_path(value: str, *, config_path: Path | None = None) -> Path:
    """Resolve a possibly relative path from the configuration file."""

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate

    reference_dir = (config_path or DEFAULT_CONFIG_PATH).resolve().parent.parent
    return (reference_dir / candidate).resolve()


def extract_data_paths(config: Mapping[str, Any], *, config_path: Path | None = None) -> dict[str, Path]:
    """Return absolute paths for the required data sources."""

    data_section = config.get("data", {})
    required_keys = {
        "customers": data_section.get("customers_file"),
        "noncustomers": data_section.get("noncustomers_file"),
        "usage_actions": data_section.get("usage_file"),
    }

    missing = [name for name, path in required_keys.items() if not path]
    if missing:
        raise KeyError(f"Missing data path configuration for: {', '.join(missing)}")

    return {name: resolve_path(path_str, config_path=config_path) for name, path_str in required_keys.items()}
