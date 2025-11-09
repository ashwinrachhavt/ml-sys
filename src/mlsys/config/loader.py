"""Helpers around loading and validating configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import yaml  # type: ignore[import-untyped]

from .settings import Settings


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file returning raw dictionary."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping")
    return payload


def validate_config(payload: dict[str, Any], schema_path: str | Path) -> None:
    """Validate payload against a JSON schema if present."""

    schema_file = Path(schema_path)
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    with schema_file.open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle) if schema_file.suffix in {".yml", ".yaml"} else yaml.safe_load(handle)

    jsonschema.validate(instance=payload, schema=schema)


def load_settings(config_path: str | Path, schema_path: str | None = None) -> Settings:
    """Load Settings from YAML optionally validating with JSON schema."""

    payload = load_yaml(config_path)

    if schema_path:
        validate_config(payload, schema_path)

    return Settings(**payload)


__all__ = ["load_settings"]
