"""Registry for model implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .base import BaseModel, ModelSpec


class ModelRegistry:
    """Name -> constructor registry."""

    _registry: ClassVar[dict[str, type[BaseModel]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
        def decorator(model_cls: type[BaseModel]) -> type[BaseModel]:
            if name in cls._registry:
                raise ValueError(f"Model '{name}' already registered")
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseModel:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<empty>"
            raise KeyError(f"Unknown model '{name}'. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def build_specs(cls, config_models: list[dict[str, Any]]) -> list[ModelSpec]:
        specs: list[ModelSpec] = []
        for entry in config_models:
            model_name = entry["name"]
            model_type = entry["type"]
            params = entry.get("param_grid", {})
            enabled = entry.get("enabled", True)
            specs.append(
                ModelSpec(name=model_name, constructor=cls._registry[model_type], params=params, enabled=enabled)
            )
        return specs

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry)


__all__ = ["ModelRegistry"]
