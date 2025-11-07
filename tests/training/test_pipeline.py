from pathlib import Path

import joblib
import numpy as np

from mlsys.training import pipeline
from mlsys.training.model import CalibratedPipelineModel


def test_build_feature_matrix_shapes():
    X, y = pipeline.build_feature_matrix()
    assert len(X) == len(y)
    assert "ALEXA_RANK_LOG1P" in X.columns
    usage_cols = [col for col in X.columns if col.startswith("ACTIONS_") or col.startswith("USERS_")]
    if usage_cols:
        assert np.isfinite(X[usage_cols].to_numpy()).all()


def test_train_and_evaluate_smoke(tmp_path: Path):
    model_path = tmp_path / "model.joblib"
    result = pipeline.train_and_evaluate(
        test_size=0.3,
        calibration_size=0.0,
        random_state=123,
        data_dir=None,
        output_model_path=model_path,
    )

    expected_metrics = {
        "roc_auc_raw",
        "pr_auc_raw",
        "roc_auc_calibrated",
        "pr_auc_calibrated",
        "baseline_positive_rate",
    }
    assert expected_metrics.issubset(result.metrics)
    assert "brier_calibrated" in result.metrics
    assert "log_loss_calibrated" in result.metrics
    assert result.metrics["pr_auc_calibrated"] >= result.metrics["pr_auc_raw"]
    for key in ("precision_at_25", "precision_at_50", "precision_at_100"):
        assert key in result.metrics
    assert "precision" in result.classification_report

    assert isinstance(result.pipeline, CalibratedPipelineModel)
    assert result.pipeline.calibrator_requires_features is True
    assert model_path.exists()
    loaded = joblib.load(model_path)
    assert isinstance(loaded, CalibratedPipelineModel)
    assert hasattr(loaded, "predict_proba")
