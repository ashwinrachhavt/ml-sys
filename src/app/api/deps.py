from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.serving.predictor import ModelPredictor


@lru_cache(maxsize=1)
def _get_predictor(config_path: Path | None = None) -> ModelPredictor:
    return ModelPredictor(config_path=config_path)


def get_predictor() -> ModelPredictor:
    return _get_predictor()
