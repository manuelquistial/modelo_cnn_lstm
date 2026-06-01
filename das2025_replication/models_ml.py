"""
Traditional machine learning classifiers (Das et al. 2025).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .config import RANDOM_STATE
from .evaluation import build_predictions_dataframe, compute_classification_metrics


def get_ml_models(random_state: int = RANDOM_STATE) -> dict:
    """Return sklearn classifiers reported in the article."""
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "NaiveBayes": GaussianNB(),
    }


def train_evaluate_ml_models(
    X_train_features: np.ndarray,
    y_train: np.ndarray,
    X_test_features: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train and evaluate all ML models.

    Returns metrics DataFrame and concatenated predictions DataFrame.
    """
    models = get_ml_models()
    metrics_rows = []
    pred_dfs = []

    for name, model in models.items():
        print(f"  Training ML model: {name}...")
        t0 = time.perf_counter()
        model.fit(X_train_features, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test_features)
        inference_time = time.perf_counter() - t1

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_features)

        metrics = compute_classification_metrics(
            y_test, y_pred, y_proba=y_proba, n_classes=len(class_names)
        )
        metrics["model"] = name
        metrics["model_name"] = name
        metrics["fit_time"] = fit_time
        metrics["total_inference_time"] = inference_time
        metrics["inference_time_per_trial"] = inference_time / max(len(y_test), 1)
        metrics_rows.append(metrics)

        pred_df = build_predictions_dataframe(
            y_test, y_pred, y_proba, meta_test, class_names, model_name=name
        )
        pred_dfs.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(pred_dfs, ignore_index=True)
    return metrics_df, predictions_df
