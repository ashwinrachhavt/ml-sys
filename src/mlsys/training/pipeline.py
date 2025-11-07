"""Reusable training pipeline for the lead-scoring XGB model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, classification_report, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlsys.core.config import DataConfig, FrameworkConfig, ModelConfig
from mlsys.features.transformers import build_feature_matrix as build_feature_matrix_impl
from mlsys.features.transformers import infer_categorical_features
from mlsys.inference.registry import ModelRegistry
from mlsys.models import build_estimator
from mlsys.tracking import MLflowTracker
from mlsys.training.model import CalibratedPipelineModel


@dataclass
class TrainingResult:
    """Container for objects/results produced by training run."""

    pipeline: CalibratedPipelineModel
    metrics: dict[str, float]
    classification_report: str
    model_path: Path


def build_feature_matrix(
    data_config: DataConfig | None = None,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compatibility wrapper around the feature engineering module."""

    return cast(
        tuple[pd.DataFrame, pd.Series],
        build_feature_matrix_impl(data_config=data_config, data_dir=data_dir),
    )


def build_model(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    model_cfg: ModelConfig,
    random_state: int,
    use_smote: bool = True,
) -> Pipeline:
    """Build model pipeline with optional SMOTE for handling class imbalance.

    Args:
        features: Feature matrix
        target: Target labels
        use_smote: If True, apply SMOTE to balance classes during training

    Returns:
        Pipeline with preprocessing, optional SMOTE, and XGBoost classifier
    """
    categorical_features = infer_categorical_features(features)
    numeric_features = [col for col in features.columns if col not in categorical_features]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]),
                numeric_features,
            ),
        ]
    )

    positive = int((target == 1).sum())
    negative = int((target == 0).sum())

    if positive == 0:
        raise ValueError("No positive samples available for training.")

    scale_pos_weight = negative / positive
    imbalance_ratio = negative / positive if positive > 0 else 1.0

    estimator = build_estimator(
        model_cfg,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    )

    # Use SMOTE if imbalance ratio is significant (> 2.0) and use_smote is True
    if use_smote and imbalance_ratio > 2.0:
        # Use imblearn Pipeline to ensure SMOTE is applied only during fit
        pipeline = ImbPipeline(
            steps=[
                ("preprocess", preprocess),
                ("smote", SMOTE(random_state=42, k_neighbors=min(5, positive - 1) if positive > 1 else 1)),
                ("model", estimator),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", estimator),
            ]
        )

    return pipeline


def tune_hyperparameters(
    base_pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    model_cfg: ModelConfig,
    random_state: int,
) -> Pipeline:
    if model_cfg.type != "xgboost":
        raise NotImplementedError("Hyperparameter tuning is currently supported only for XGBoost models.")

    param_distributions = {
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.05, 0.07, 0.1],
        "model__n_estimators": [300, 400, 500, 600],
        "model__subsample": [0.6, 0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "model__min_child_weight": [1, 2, 3, 5],
        "model__gamma": [0.0, 0.1, 0.2],
    }

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="average_precision",
        cv=3,
        verbose=0,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(features, target)
    return search.best_estimator_


def train_and_evaluate(config: FrameworkConfig, output_model_path: Path | None = None) -> TrainingResult:
    """Train the model, evaluate on a holdout set, and persist artifacts."""

    data_cfg = config.data
    feature_cfg = config.features
    model_cfg = config.model
    tracking_cfg = config.tracking

    if data_cfg.calibration_size < 0 or data_cfg.calibration_size >= 1:
        raise ValueError("data.calibration_size must be in [0, 1)")

    tracker = MLflowTracker(tracking_cfg)
    tracker_active = tracker.start_run(run_name=tracking_cfg.run_name)
    if tracker_active:
        tracker.log_params(
            {
                "test_size": data_cfg.test_size,
                "calibration_size": data_cfg.calibration_size,
                "tune_hyperparameters": model_cfg.tune_hyperparameters,
                "use_smote": feature_cfg.apply_smote,
                "random_state": data_cfg.random_state,
                "model_type": model_cfg.type,
            }
        )

    try:
        features, target = build_feature_matrix(data_config=data_cfg)

        stratify_target = target if data_cfg.stratify else None
        x_train_full, x_test, y_train_full, y_test = train_test_split(
            features,
            target,
            test_size=data_cfg.test_size,
            random_state=data_cfg.random_state,
            stratify=stratify_target,
        )

        base_pipeline = build_model(
            x_train_full,
            y_train_full,
            model_cfg=model_cfg,
            random_state=data_cfg.random_state,
            use_smote=feature_cfg.apply_smote,
        )

        if model_cfg.tune_hyperparameters:
            tuned_pipeline = tune_hyperparameters(
                base_pipeline,
                x_train_full,
                y_train_full,
                model_cfg,
                data_cfg.random_state,
            )
        else:
            tuned_pipeline = base_pipeline

        calibrate = model_cfg.calibration_method != "none"
        calibration_size = data_cfg.calibration_size

        if calibrate and calibration_size > 0:
            x_train, x_cal, y_train, y_cal = train_test_split(
                x_train_full,
                y_train_full,
                test_size=calibration_size,
                random_state=data_cfg.random_state,
                stratify=y_train_full if data_cfg.stratify else None,
            )
            tuned_pipeline.fit(x_train, y_train)
            raw_proba = tuned_pipeline.predict_proba(x_test)[:, 1]
            calibrator = None
            if len(np.unique(y_cal)) >= 2 and model_cfg.calibration_method == "sigmoid":
                cal_proba = tuned_pipeline.predict_proba(x_cal)[:, 1].reshape(-1, 1)
                calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
                calibrator.fit(cal_proba, y_cal)
            calibrated_model = CalibratedPipelineModel(
                pipeline=tuned_pipeline,
                calibrator=calibrator,
                calibrator_requires_features=False,
            )
        elif calibrate:
            tuned_pipeline.fit(x_train_full, y_train_full)
            raw_proba = tuned_pipeline.predict_proba(x_test)[:, 1]
            _, class_counts = np.unique(y_train_full, return_counts=True)
            calibration_cv = int(class_counts.min()) if len(class_counts) > 0 else 0
            calibration_cv = min(5, calibration_cv)
            calibrator_model = None
            calibrator_requires_features = False
            if calibration_cv >= 2:
                calibrator_model = CalibratedClassifierCV(
                    estimator=clone(tuned_pipeline),
                    cv=calibration_cv,
                    method="sigmoid" if model_cfg.calibration_method == "sigmoid" else model_cfg.calibration_method,
                    n_jobs=1,
                )
                calibrator_model.fit(x_train_full, y_train_full)
                calibrator_requires_features = True
            else:
                warnings.warn(
                    "Insufficient samples per class for cross-validated calibration; using raw model probabilities instead.",
                    stacklevel=2,
                )
            calibrated_model = CalibratedPipelineModel(
                pipeline=tuned_pipeline,
                calibrator=calibrator_model,
                calibrator_requires_features=calibrator_requires_features,
            )
        else:
            tuned_pipeline.fit(x_train_full, y_train_full)
            raw_proba = tuned_pipeline.predict_proba(x_test)[:, 1]
            calibrated_model = CalibratedPipelineModel(
                pipeline=tuned_pipeline,
                calibrator=None,
                calibrator_requires_features=False,
            )

        calibrated_proba = calibrated_model.predict_proba(x_test)[:, 1]
        preds = (calibrated_proba >= 0.5).astype(int)

        metrics = {
            "roc_auc_raw": float(roc_auc_score(y_test, raw_proba)),
            "pr_auc_raw": float(average_precision_score(y_test, raw_proba)),
            "roc_auc_calibrated": float(roc_auc_score(y_test, calibrated_proba)),
            "pr_auc_calibrated": float(average_precision_score(y_test, calibrated_proba)),
            "brier_calibrated": float(brier_score_loss(y_test, calibrated_proba)),
            "log_loss_calibrated": float(log_loss(y_test, calibrated_proba, labels=[0, 1])),
            "baseline_positive_rate": float(y_test.mean()),
        }

        for k in (25, 50, 100):
            if k <= len(calibrated_proba):
                order = np.argsort(calibrated_proba)[::-1]
                topk_precision = float(y_test.iloc[order[:k]].mean())
                metrics[f"precision_at_{k}"] = topk_precision

        clf_report = classification_report(y_test, preds, digits=3)

        model_filename = f"{model_cfg.type}_model.joblib"
        resolved_model_path = output_model_path or config.resolve_artifact_path("models", model_filename)
        resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated_model, resolved_model_path)

        if tracker_active:
            tracker.log_metrics(metrics)
            tracker.log_params(
                {
                    "n_samples_train": len(x_train_full),
                    "n_samples_test": len(x_test),
                    "n_features": features.shape[1],
                    "positive_class_ratio": float(target.mean()),
                }
            )
            if tracking_cfg.log_artifacts:
                tracker.log_model(
                    calibrated_model,
                    "model",
                    registered_model_name="lead-scoring-model",
                )
                report_path = resolved_model_path.parent / "classification_report.txt"
                report_path.write_text(clf_report, encoding="utf-8")
                tracker.log_artifact(report_path)
                tracker.log_artifact(resolved_model_path)
                if report_path.exists():
                    report_path.unlink()
            tracker.set_tags(
                {
                    "model_type": model_cfg.type,
                    "calibrated": str(calibrate),
                    "smote_applied": str(feature_cfg.apply_smote),
                }
            )

        registry = ModelRegistry(config.serving.model_registry_path)
        registry.register_model(
            resolved_model_path,
            model_name=model_cfg.type,
            metrics=metrics,
            config=config.model_dump(mode="json"),
            primary_metric="roc_auc_calibrated",
        )

        return TrainingResult(
            pipeline=calibrated_model,
            metrics=metrics,
            classification_report=clf_report,
            model_path=resolved_model_path,
        )
    finally:
        tracker.end_run()
