"""
Model evaluation metrics, predictions, and error analysis.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_classes: int = 2,
) -> dict[str, Any]:
    """Compute standard classification metrics."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if y_proba is not None and n_classes == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def build_predictions_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    meta_test: pd.DataFrame,
    class_names: list[str],
    model_name: str,
    positive_class: str = "right_hand",
) -> pd.DataFrame:
    """Build per-trial predictions DataFrame with error types."""
    n = len(y_true)
    records = []
    for i in range(n):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])
        true_class = class_names[true_label] if true_label < len(class_names) else str(true_label)
        pred_class = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
        confidence = float(np.max(y_proba[i])) if y_proba is not None else np.nan

        row: dict[str, Any] = {
            "subject": meta_test.iloc[i]["subject"] if "subject" in meta_test.columns else -1,
            "run": meta_test.iloc[i].get("run", -1),
            "trial_id": meta_test.iloc[i].get("trial_id", i),
            "true_label": true_label,
            "true_class": true_class,
            "predicted_label": pred_label,
            "predicted_class": pred_class,
            "confidence": confidence,
            "is_correct": true_label == pred_label,
            "model_name": model_name,
        }
        if y_proba is not None:
            for c in range(y_proba.shape[1]):
                cname = class_names[c] if c < len(class_names) else f"class_{c}"
                row[f"prob_{cname}"] = float(y_proba[i, c])

        # Error type
        if true_label == pred_label:
            row["error_type"] = "correct"
        else:
            row["error_type"] = f"{true_class}_as_{pred_class}"

        # Binary FP/FN (positive class = right_hand / right_fist)
        row["is_false_positive"] = False
        row["is_false_negative"] = False
        if len(class_names) == 2:
            pos_candidates = [positive_class, "right_hand", "right_fist"]
            pos_idx = next(
                (class_names.index(c) for c in pos_candidates if c in class_names),
                1,
            )
            if true_label != pos_idx and pred_label == pos_idx:
                row["is_false_positive"] = True
            if true_label == pos_idx and pred_label != pos_idx:
                row["is_false_negative"] = True

        records.append(row)
    return pd.DataFrame(records)


def evaluate_deep_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    class_names: list[str],
    model_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate a trained Keras model on test data."""
    t0 = time.perf_counter()
    y_proba = model.predict(X_test, verbose=0)
    inference_time = time.perf_counter() - t0

    if y_proba.ndim == 1:
        y_proba = np.column_stack([1 - y_proba, y_proba])
    y_pred = np.argmax(y_proba, axis=1)

    metrics = compute_classification_metrics(
        y_test, y_pred, y_proba=y_proba, n_classes=len(class_names)
    )
    metrics["model"] = model_name
    metrics["model_name"] = model_name
    metrics["total_inference_time"] = inference_time
    metrics["inference_time_per_trial"] = inference_time / max(len(y_test), 1)
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )

    pred_df = build_predictions_dataframe(
        y_test, y_pred, y_proba, meta_test, class_names, model_name=model_name
    )
    return metrics, pred_df


def analyze_errors_by_subject(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate errors by subject."""
    rows = []
    for (subject, model_name), grp in predictions_df.groupby(["subject", "model_name"]):
        n = len(grp)
        correct = grp["is_correct"].sum()
        errors = n - correct
        fp = grp.get("is_false_positive", pd.Series([False] * n)).sum()
        fn = grp.get("is_false_negative", pd.Series([False] * n)).sum()
        false_left = grp[
            (grp["true_class"] == "left_hand") & (grp["predicted_class"] == "right_hand")
        ].shape[0] if "true_class" in grp.columns else 0
        false_right = grp[
            (grp["true_class"] == "right_hand") & (grp["predicted_class"] == "left_hand")
        ].shape[0] if "true_class" in grp.columns else 0

        err_mask = ~grp["is_correct"]
        rows.append({
            "subject": subject,
            "model_name": model_name,
            "total_trials": n,
            "correct_trials": int(correct),
            "error_trials": int(errors),
            "accuracy": correct / n if n else 0.0,
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "false_left_as_right": int(false_left),
            "false_right_as_left": int(false_right),
            "mean_confidence_errors": float(grp.loc[err_mask, "confidence"].mean()) if err_mask.any() else np.nan,
            "mean_confidence_correct": float(grp.loc[~err_mask, "confidence"].mean()) if (~err_mask).any() else np.nan,
        })
    return pd.DataFrame(rows)


def get_misclassified_trials(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Return trial-level table of misclassified samples."""
    mis = predictions_df[~predictions_df["is_correct"]].copy()
    cols = [
        "subject", "run", "trial_id", "true_class", "predicted_class",
        "confidence", "error_type", "model_name",
    ]
    prob_cols = [c for c in mis.columns if c.startswith("prob_")]
    return mis[cols + prob_cols]


def save_evaluation_outputs(
    predictions_df: pd.DataFrame,
    output_dir: Any,
) -> None:
    """Save predictions and error analysis CSVs."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(out / "predictions_by_trial.csv", index=False)
    errors_subj = analyze_errors_by_subject(predictions_df)
    errors_subj.to_csv(out / "errors_by_subject.csv", index=False)
    misclassified = get_misclassified_trials(predictions_df)
    misclassified.to_csv(out / "misclassified_trials.csv", index=False)
