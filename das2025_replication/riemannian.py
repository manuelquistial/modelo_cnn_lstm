"""
Riemannian geometry baselines using pyriemann (if available).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .evaluation import build_predictions_dataframe, compute_classification_metrics

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM, FgMDM
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False


def run_riemannian_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Riemannian classifiers: TangentSpace+LR, MDM, FgMDM.

    X shape: (n_trials, n_channels, n_samples)
    """
    if not HAS_PYRIEMANN:
        print(
            "WARNING: pyriemann is not installed. Skipping Riemannian analysis.\n"
            "Install with: pip install pyriemann"
        )
        return pd.DataFrame(), pd.DataFrame()

    metrics_rows = []
    pred_dfs = []

    models = {
        "TangentSpace_LR": Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("ts", TangentSpace()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "MDM": Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("mdm", MDM()),
        ]),
        "FgMDM": Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("fgmdm", FgMDM()),
        ]),
    }

    for name, model in models.items():
        print(f"  Training Riemannian model: {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception:
                    pass

            metrics = compute_classification_metrics(
                y_test, y_pred, y_proba=y_proba, n_classes=len(class_names)
            )
            metrics["model"] = name
            metrics["model_name"] = name
            metrics_rows.append(metrics)

            pred_df = build_predictions_dataframe(
                y_test, y_pred, y_proba, meta_test, class_names, model_name=name
            )
            pred_dfs.append(pred_df)
        except Exception as exc:
            warnings.warn(f"Riemannian model {name} failed: {exc}")

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(pred_dfs, ignore_index=True) if pred_dfs else pd.DataFrame()
    return metrics_df, predictions_df
