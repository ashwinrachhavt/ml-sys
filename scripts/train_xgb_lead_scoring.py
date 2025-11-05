from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def load_and_prepare_features(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    customers_path = data_dir / "customers.csv"
    noncustomers_path = data_dir / "noncustomers.csv"
    usage_path = data_dir / "usage_actions.csv"

    customers = pd.read_csv(customers_path)
    customers["is_customer"] = 1

    noncustomers = pd.read_csv(noncustomers_path)
    noncustomers["is_customer"] = 0

    data = pd.concat([customers, noncustomers], ignore_index=True, sort=False)

    # Guard against label leakage: drop columns that only exist for customers or contain future info.
    data = data.drop(columns=["CLOSEDATE", "MRR"], errors="ignore")

    # Read and aggregate product usage signals to the company (id) level.
    usage = pd.read_csv(usage_path)
    usage_feature_cols = [
        col
        for col in usage.columns
        if col.startswith("ACTIONS_") or col.startswith("USERS_")
    ]

    usage_agg = (
        usage.groupby("id")[usage_feature_cols]
        .sum(min_count=1)
        .reset_index()
    )

    # Total intensity features help tree models pick up overall engagement quickly.
    action_cols = [col for col in usage_feature_cols if col.startswith("ACTIONS_")]
    user_cols = [col for col in usage_feature_cols if col.startswith("USERS_")]

    usage_agg["ACTIONS_TOTAL"] = usage_agg[action_cols].sum(axis=1)
    usage_agg["USERS_TOTAL"] = usage_agg[user_cols].sum(axis=1)

    data = data.merge(usage_agg, on="id", how="left")

    # Missing usage implies no tracked activity.
    usage_fill_cols = [col for col in data.columns if col.startswith("ACTIONS_") or col.startswith("USERS_")]
    data[usage_fill_cols] = data[usage_fill_cols].fillna(0)

    # Categorical clean-up: treat missing as explicit "UNKNOWN" bucket.
    for cat_col in ["EMPLOYEE_RANGE", "INDUSTRY"]:
        data[cat_col] = data[cat_col].fillna("UNKNOWN")

    # Alexa rank has a sentinel 16000001 for "unranked"; replace with NaN and add log-scaled variant.
    data["ALEXA_RANK"] = data["ALEXA_RANK"].replace(16000001, np.nan)
    data["ALEXA_RANK_LOG1P"] = np.log1p(data["ALEXA_RANK"])

    # Prepare feature matrix and target vector.
    y = data.pop("is_customer")
    X = data.drop(columns=["id"], errors="ignore")

    return X, y


def build_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    categorical_features = [
        col for col in ["EMPLOYEE_RANGE", "INDUSTRY"] if col in X_train.columns
    ]
    numeric_features = [
        col
        for col in X_train.columns
        if col not in categorical_features
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    # Handle class imbalance using the empirical ratio from the training split.
    positive = (y_train == 1).sum()
    negative = (y_train == 0).sum()
    if positive == 0:
        raise ValueError("Training split contains no positive examples; adjust stratification or data.")

    scale_pos_weight = negative / positive

    model = XGBClassifier(
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
        use_label_encoder=False,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("xgb", model),
        ]
    )

    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("Baseline positive rate:", y_test.mean())
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, preds, digits=3))

    # Show a few precision@k style metrics that sales stakeholders care about.
    order = np.argsort(proba)[::-1]
    for k in (25, 50, 100):
        if k <= len(order):
            topk_precision = y_test.iloc[order[:k]].mean()
            print(f"Precision@{k:>3}: {topk_precision:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBClassifier for predictive lead scoring.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing customers.csv, noncustomers.csv, usage_actions.csv",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for the evaluation split",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X, y = load_and_prepare_features(args.data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_pipeline(X_train, y_train)
    pipeline.fit(X_train, y_train)

    evaluate_model(pipeline, X_test, y_test)


if __name__ == "__main__":
    main()
