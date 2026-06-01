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


def _tangent_space_dim(n_channels: int) -> int:
    """Symmetric SPD tangent-space feature count for n_channels x n_channels covariances."""
    return n_channels * (n_channels + 1) // 2


def _fgmdm_applicable(n_channels: int, n_classes: int) -> tuple[bool, str]:
    """
    FgMDM uses FGDA with LDA (n_components = n_classes - 1).

    Requires tangent-space dimension >= n_classes - 1. With paper_input (2 channels)
    tangent dim = 3, so FgMDM fails for 5-class problems (needs 4 components).
    """
    ts_dim = _tangent_space_dim(n_channels)
    lda_components = n_classes - 1
    if lda_components <= 0:
        return False, "FgMDM skipped: need at least 2 classes."
    if lda_components > ts_dim:
        return False, (
            f"FgMDM skipped: tangent dim {ts_dim} ({n_channels} ch) < "
            f"n_classes-1 ({lda_components}). "
            "Use full ROI channels (not paper_input 640×2) or binary mode."
        )
    return True, ""


def run_riemannian_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Riemannian classifiers: TangentSpace+LR, MDM, FgMDM (when mathematically valid).

    X shape: (n_trials, n_channels, n_samples)
    """
    if not HAS_PYRIEMANN:
        print(
            "WARNING: pyriemann is not installed. Skipping Riemannian analysis.\n"
            "Install with: pip install pyriemann"
        )
        return pd.DataFrame(), pd.DataFrame()

    n_channels = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    print(
        f"  Riemannian input: {X_train.shape[0]} train trials, "
        f"{n_channels} channels, {n_classes} classes "
        f"(tangent dim={_tangent_space_dim(n_channels)})",
        flush=True,
    )

    metrics_rows = []
    pred_dfs = []

    models: dict[str, Pipeline] = {
        "TangentSpace_LR": Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("ts", TangentSpace()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "MDM": Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("mdm", MDM()),
        ]),
    }

    ok_fgmdm, fgmdm_reason = _fgmdm_applicable(n_channels, n_classes)
    if ok_fgmdm:
        models["FgMDM"] = Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("fgmdm", FgMDM()),
        ])
    else:
        print(f"  Note: {fgmdm_reason}", flush=True)

    for name, model in models.items():
        print(f"  Training Riemannian model: {name}...", flush=True)
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
