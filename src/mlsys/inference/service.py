"""FastAPI service exposing the trained lead-scoring model."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from mlsys.config.paths import MODEL_PATH
from mlsys.inference.schemas import ScoreRequest, ScoreResponse

logger = logging.getLogger(__name__)


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    logger.info("Loading model from %s", path)
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_model():
    return load_model()


def create_app() -> FastAPI:
    app = FastAPI(title="Lead Scoring Inference API")

    @app.get("/health")
    def health() -> Dict[str, str]:
        try:
            get_model()
        except Exception as exc:  # pragma: no cover - we want to log unexpected failures
            logger.exception("Model health check failed")
            raise HTTPException(status_code=500, detail=str(exc))
        return {"status": "ok"}

    @app.post("/score", response_model=ScoreResponse)
    def score(payload: ScoreRequest) -> ScoreResponse:
        model = get_model()
        if not payload.leads:
            raise HTTPException(status_code=400, detail="No leads provided")

        df = pd.DataFrame([lead.model_dump() for lead in payload.leads])

        if "ALEXA_RANK" in df.columns and "ALEXA_RANK_LOG1P" not in df.columns:
            df["ALEXA_RANK_LOG1P"] = pd.Series(df["ALEXA_RANK"]).apply(lambda x: None if pd.isna(x) else np.log1p(x))

        proba = model.predict_proba(df)[:, 1]
        predictions = proba.tolist()

        return ScoreResponse(probabilities=predictions, meta={"model_path": str(MODEL_PATH)})

    return app


app = create_app()
