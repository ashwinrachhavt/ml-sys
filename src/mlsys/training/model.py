"""Model wrapper that combines preprocessing pipeline and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CalibratedPipelineModel:
    pipeline: Any
    calibrator: Any | None = None
    calibrator_requires_features: bool = False

    def predict_proba_raw(self, features):
        return self.pipeline.predict_proba(features)[:, 1]

    def predict_proba(self, features):
        base_proba = self.predict_proba_raw(features)
        if self.calibrator is not None:
            if self.calibrator_requires_features:
                calibrated = self.calibrator.predict_proba(features)[:, 1]
            elif hasattr(self.calibrator, "predict_proba"):
                calibrated = self.calibrator.predict_proba(base_proba.reshape(-1, 1))[:, 1]
            else:
                calibrated = self.calibrator.transform(base_proba)
        else:
            calibrated = base_proba
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, features, threshold: float = 0.5):
        proba = self.predict_proba(features)[:, 1]
        return (proba >= threshold).astype(int)
