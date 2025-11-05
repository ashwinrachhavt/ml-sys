"""Centralised project paths for training and inference artifacts."""
from pathlib import Path

# Root of the repository (assume this file lives under src/mlsys/config/)
REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "xgb_lead_scoring.joblib"

__all__ = ["REPO_ROOT", "DATA_DIR", "ARTIFACTS_DIR", "MODEL_PATH"]
