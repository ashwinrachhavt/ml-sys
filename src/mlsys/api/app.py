"""FastAPI application factory wiring routes, dependencies and middleware."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI

from mlsys.config import Settings
from mlsys.serving import PredictorService

from .routes import health, inference

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def get_settings(config_path: Path | str | None = None) -> Settings:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return Settings.from_yaml(path)


@lru_cache(maxsize=1)
def _settings_cached() -> Settings:
    return get_settings()


def settings_dep() -> Settings:
    return _settings_cached()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or _settings_cached()
    app = FastAPI(title="mlsys", version=resolved_settings.project.version)

    if settings:

        @lru_cache(maxsize=1)
        def _settings_override() -> Settings:
            return settings

        app.dependency_overrides[settings_dep] = _settings_override

    app.include_router(health.router, prefix="/api")
    app.include_router(inference.router, prefix="/api")

    @lru_cache(maxsize=1)
    def _predictor_cached() -> PredictorService:
        return PredictorService(settings=resolved_settings)

    app.dependency_overrides[inference.get_predictor] = _predictor_cached

    return app


__all__ = ["create_app"]
