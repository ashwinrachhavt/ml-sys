"""Reusable training pipeline for the lead-scoring XGB model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             log_loss, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from mlsys.config.paths import DATA_DIR, MODEL_PATH
from mlsys.training.model import CalibratedPipelineModel
from mlsys.training.stub_data import load_stub_tables


@dataclass
class TrainingResult:
    """Container for objects/results produced by training run."""

    pipeline: CalibratedPipelineModel
    metrics: Dict[str, float]
    classification_report: str
    model_path: Path


USAGE_PREFIXES = ("ACTIONS_", "USERS_")


def _load_raw_tables(data_dir: Path | None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read raw CSV tables, falling back to synthetic stubs if missing."""

    if data_dir is not None:
        try:
            customers = pd.read_csv(data_dir / "customers.csv")
            noncustomers = pd.read_csv(data_dir / "noncustomers.csv")
            usage = pd.read_csv(data_dir / "usage_actions.csv")
            return customers, noncustomers, usage
        except FileNotFoundError:
            warnings.warn(
                "Raw CSV files were not found in the provided data directory; "
                "using synthetic stub data instead.",
                stacklevel=2,
            )

    return load_stub_tables()


def _merge_datasets(customers: pd.DataFrame, noncustomers: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    customers = customers.copy()
    noncustomers = noncustomers.copy()

    customers["is_customer"] = 1
    noncustomers["is_customer"] = 0

    data = pd.concat([customers, noncustomers], ignore_index=True, sort=False)

    # Remove leaky / customer-only columns.
    data = data.drop(columns=["CLOSEDATE", "MRR"], errors="ignore")

    usage_feature_cols = [
        col
        for col in usage.columns
        if any(col.startswith(prefix) for prefix in USAGE_PREFIXES)
    ]

    usage_agg = (
        usage.groupby("id")[usage_feature_cols]
        .sum(min_count=1)
        .reset_index()
    )

    action_cols = [col for col in usage_feature_cols if col.startswith("ACTIONS_")]
    user_cols = [col for col in usage_feature_cols if col.startswith("USERS_")]

    usage_agg["ACTIONS_TOTAL"] = usage_agg[action_cols].sum(axis=1)
    usage_agg["USERS_TOTAL"] = usage_agg[user_cols].sum(axis=1)

    data = data.merge(usage_agg, on="id", how="left")

    usage_cols = [
        col
        for col in data.columns
        if any(col.startswith(prefix) for prefix in USAGE_PREFIXES)
    ]

    data[usage_cols] = data[usage_cols].fillna(0)
    for cat_col in ["EMPLOYEE_RANGE", "INDUSTRY"]:
        if cat_col in data.columns:
            data[cat_col] = data[cat_col].fillna("UNKNOWN")

    data["ALEXA_RANK"] = data["ALEXA_RANK"].replace(16000001, np.nan)
    data["ALEXA_RANK_LOG1P"] = np.log1p(data["ALEXA_RANK"])

    return data


def build_feature_matrix(data_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load raw CSVs and produce feature matrix (X) and target (y)."""

    resolved_dir = data_dir or DATA_DIR
    lookup_dir = resolved_dir
    if data_dir is None and not resolved_dir.exists():
        lookup_dir = None

    customers, noncustomers, usage = _load_raw_tables(lookup_dir)
    merged = _merge_datasets(customers, noncustomers, usage)

    y = merged.pop("is_customer")
    X = merged.drop(columns=["id"], errors="ignore")
    return X, y


def build_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    categorical_features = [
        col for col in ["EMPLOYEE_RANGE", "INDUSTRY"] if col in X.columns
    ]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    positive = int((y == 1).sum())
    negative = int((y == 0).sum())

    if positive == 0:
        raise ValueError("No positive samples available for training.")

    scale_pos_weight = negative / positive

    estimator = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )

    return pipeline


def tune_hyperparameters(base_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, random_state: int) -> Pipeline:
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
    search.fit(X, y)
    return search.best_estimator_


def train_and_evaluate(
    test_size: float = 0.2,
    calibration_size: float = 0.0,
    tune_hyperparameters_flag: bool = False,
    random_state: int = 42,
    data_dir: Path | None = None,
    output_model_path: Path | None = None,
) -> TrainingResult:
    """Train the model, evaluate on a holdout set, and persist artifacts."""

    X, y = build_feature_matrix(data_dir=data_dir)

    if calibration_size < 0 or calibration_size >= 1:
        raise ValueError("calibration_size must be in [0, 1)")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    base_pipeline = build_model(X_train_full, y_train_full)

    if tune_hyperparameters_flag:
        tuned_pipeline = tune_hyperparameters(base_pipeline, X_train_full, y_train_full, random_state)
    else:
        tuned_pipeline = base_pipeline

    if calibration_size > 0:
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_full,
            y_train_full,
            test_size=calibration_size,
            random_state=random_state,
            stratify=y_train_full,
        )
        tuned_pipeline.fit(X_train, y_train)
        raw_proba = tuned_pipeline.predict_proba(X_test)[:, 1]
        calibrator = None
        if len(np.unique(y_cal)) >= 2:
            cal_proba = tuned_pipeline.predict_proba(X_cal)[:, 1].reshape(-1, 1)
            calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
            calibrator.fit(cal_proba, y_cal)
        calibrated_model = CalibratedPipelineModel(
            pipeline=tuned_pipeline,
            calibrator=calibrator,
            calibrator_requires_features=False,
        )
    else:
        tuned_pipeline.fit(X_train_full, y_train_full)
        raw_proba = tuned_pipeline.predict_proba(X_test)[:, 1]
        calibrator_model = CalibratedClassifierCV(
            estimator=clone(tuned_pipeline),
            cv=5,
            method='sigmoid',
            n_jobs=-1,
        )
        calibrator_model.fit(X_train_full, y_train_full)
        calibrated_model = CalibratedPipelineModel(
            pipeline=tuned_pipeline,
            calibrator=calibrator_model,
            calibrator_requires_features=True,
        )

    calibrated_proba = calibrated_model.predict_proba(X_test)[:, 1]
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

    output_model_path = output_model_path or MODEL_PATH
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, output_model_path)

    return TrainingResult(
        pipeline=calibrated_model,
        metrics=metrics,
        classification_report=clf_report,
        model_path=output_model_path,
    )
