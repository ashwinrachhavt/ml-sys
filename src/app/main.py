from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import REGISTRY, generate_latest

from app.api.v1.inference import router as inference_router
from app.serving.predictor import ModelPredictor


def create_app(*, warm_model: bool = True) -> FastAPI:
    app = FastAPI(title="ML Inference API", version="1.0.0")

    app.include_router(inference_router)

    if warm_model:

        @app.on_event("startup")
        def _warm_model() -> None:  # pragma: no cover - executed in runtime
            # instantiate predictor once to warm cache
            ModelPredictor()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "timestamp": str(time.time())}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(REGISTRY))

    return app


app = create_app()
