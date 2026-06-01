"""
Complete experiment orchestration for Das et al. (2025) replication.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import (
    BINARY_CLASS_NAMES,
    BINARY_RUNS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEGMENT_LENGTH,
    DL_BATCH_SIZE,
    DL_EPOCHS,
    MULTICLASS_CLASS_NAMES,
    MULTICLASS_RUNS,
    RANDOM_STATE,
    get_output_paths,
    get_paper_subjects,
    save_experiment_config,
)
from .data_loading import (
    build_dataset,
    ensure_channel_first,
    prepare_deep_learning_input,
    print_dataset_summary,
)
from .evaluation import (
    analyze_errors_by_subject,
    evaluate_deep_model,
    get_misclassified_trials,
    save_evaluation_outputs,
)
from .features import build_ml_feature_matrix
from .models_deep import (
    HAS_TF,
    extract_deep_embeddings,
    get_deep_model,
    set_reproducibility,
    train_deep_model,
)
from .models_ml import train_evaluate_ml_models
from .preprocessing import (
    apply_channelwise_standardizer,
    apply_sklearn_scaler,
    fit_channelwise_standardizer,
    fit_sklearn_scaler,
    make_subjectwise_split,
    make_trialwise_split,
    make_validation_split_from_train,
)
from .riemannian import run_riemannian_baselines
from .rois import list_all_rois
from .visualization import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_real_vs_synthetic_eeg,
    plot_roi_comparison,
    plot_roc_curve_binary,
    plot_training_history,
    plot_tsne_embeddings,
    plot_tsne_raw_features,
    run_pca_visualization,
)

SplitStrategy = Literal["trialwise", "subjectwise"]


def _get_split_fn(strategy: SplitStrategy):
    if strategy == "trialwise":
        return make_trialwise_split
    return make_subjectwise_split


def _get_class_names(mode: str) -> list[str]:
    return BINARY_CLASS_NAMES if mode == "binary" else MULTICLASS_CLASS_NAMES


def _get_runs(mode: str) -> list[int]:
    return BINARY_RUNS if mode == "binary" else MULTICLASS_RUNS


def _prepare_dl_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    subjectwise_val: bool = True,
    meta_train: pd.DataFrame | None = None,
    y_train: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert to DL format and normalize using train statistics only."""
    X_tr = prepare_deep_learning_input(X_train)
    X_te = prepare_deep_learning_input(X_test)
    mean, std = fit_channelwise_standardizer(X_tr)
    X_tr = apply_channelwise_standardizer(X_tr, mean, std)
    X_te = apply_channelwise_standardizer(X_te, mean, std)

    if meta_train is not None and y_train is not None:
        X_fit, X_val, y_fit, y_val, _, _ = make_validation_split_from_train(
            X_tr, y_train, meta_train, subjectwise=subjectwise_val
        )
        return X_fit, X_val, X_te, y_fit, y_val, X_te

    # Simple stratified val split on normalized train
    from sklearn.model_selection import train_test_split

    if y_train is None:
        raise ValueError("y_train required for validation split")
    idx = np.arange(len(y_train))
    tr_i, val_i = train_test_split(
        idx, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    return X_tr[tr_i], X_tr[val_i], X_te, y_train[tr_i], y_train[val_i], X_te


def run_single_deep_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    class_names: list[str],
    model_name: str = "cnn_lstm_attention",
    epochs: int = DL_EPOCHS,
    batch_size: int = DL_BATCH_SIZE,
    split_strategy: str = "subjectwise",
    output_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> tuple[dict, pd.DataFrame, dict, Any]:
    """Train and evaluate one deep model."""
    if not HAS_TF:
        warnings.warn("TensorFlow not available; skipping deep model.")
        return {}, pd.DataFrame(), {}, None

    set_reproducibility()
    subjectwise_val = split_strategy == "subjectwise"
    X_fit, X_val, X_te, y_fit, y_val, _ = _prepare_dl_data(
        X_train, X_test, subjectwise_val, meta_train, y_train
    )
    input_shape = (X_fit.shape[1], X_fit.shape[2])
    num_classes = len(class_names)

    model = get_deep_model(model_name, input_shape, num_classes)
    print(f"\n{model_name} summary:")
    model.summary()

    model, history, fit_time = train_deep_model(
        model, X_fit, y_fit, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_name=model_name,
        output_dir=output_dir,
    )

    metrics, pred_df = evaluate_deep_model(
        model, X_te, y_test, meta_test, class_names, model_name
    )
    metrics["fit_time"] = fit_time

    if figures_dir is not None:
        plot_training_history(
            history,
            title=f"{model_name} training",
            save_path=figures_dir / f"{model_name}_history.png",
        )
        y_pred = pred_df["predicted_label"].values
        plot_confusion_matrix(
            y_test, y_pred, class_names,
            title=f"{model_name} confusion matrix",
            save_path=figures_dir / f"{model_name}_cm.png",
        )
        if len(class_names) == 2:
            proba_cols = [c for c in pred_df.columns if c.startswith("prob_")]
            if len(proba_cols) >= 2:
                y_proba = pred_df[proba_cols].values
                plot_roc_curve_binary(
                    y_test, y_proba,
                    title=f"{model_name} ROC",
                    save_path=figures_dir / f"{model_name}_roc.png",
                )

    return metrics, pred_df, history, model


def run_ml_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    class_names: list[str],
    fs: float = 160.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract features and train ML models."""
    X_train_ml = ensure_channel_first(X_train)
    X_test_ml = ensure_channel_first(X_test)
    print("Extracting ML features (train)...")
    X_train_feat, _ = build_ml_feature_matrix(X_train_ml, fs=fs)
    print("Extracting ML features (test)...")
    X_test_feat, _ = build_ml_feature_matrix(X_test_ml, fs=fs)
    scaler = fit_sklearn_scaler(X_train_feat)
    X_train_feat = apply_sklearn_scaler(scaler, X_train_feat)
    X_test_feat = apply_sklearn_scaler(scaler, X_test_feat)
    return train_evaluate_ml_models(
        X_train_feat, y_train, X_test_feat, y_test, meta_test, class_names
    )


def run_roi_experiments(
    subjects: list[int],
    mode: str = "binary",
    segment_length: float = DEFAULT_SEGMENT_LENGTH,
    split_strategy: SplitStrategy = "subjectwise",
    model_names: list[str] | None = None,
    epochs: int = DL_EPOCHS,
    batch_size: int = DL_BATCH_SIZE,
    output_dir: Path | None = None,
    paper_input: bool = False,
    use_paper_roi_epochs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run deep models across all six ROIs."""
    if model_names is None:
        model_names = ["cnn", "lstm", "cnn_lstm_attention"]
    paths = get_output_paths(output_dir)
    figures_dir = paths["figures"]
    class_names = _get_class_names(mode)
    split_fn = _get_split_fn(split_strategy)
    runs = _get_runs(mode)

    metrics_rows = []
    all_preds = []
    histories: dict[str, Any] = {}

    for roi in list_all_rois():
        print(f"\n{'=' * 60}\nROI EXPERIMENT: {roi}\n{'=' * 60}")
        X, y, metadata, ch_names = build_dataset(
            subjects, runs, mode=mode, segment_length=segment_length,
            roi_name=roi, paper_input=paper_input,
        )
        print(f"Selected channels ({len(ch_names)}): {ch_names}")
        X_train, X_test, y_train, y_test, meta_train, meta_test = split_fn(X, y, metadata)

        roi_epochs = epochs
        if use_paper_roi_epochs:
            from .paper_input import ROI_TRAINING_EPOCHS_PAPER
            roi_epochs = ROI_TRAINING_EPOCHS_PAPER.get(roi, epochs)

        for model_name in model_names:
            metrics, pred_df, history, _ = run_single_deep_experiment(
                X_train, y_train, X_test, y_test,
                meta_train, meta_test, class_names,
                model_name=model_name,
                epochs=roi_epochs,
                batch_size=batch_size,
                split_strategy=split_strategy,
                figures_dir=figures_dir,
            )
            if not metrics:
                continue
            metrics.update({
                "roi": roi,
                "mode": mode,
                "split_strategy": split_strategy,
                "segment_length": segment_length,
                "augmentation": "none",
            })
            metrics_rows.append(metrics)
            pred_df["roi"] = roi
            all_preds.append(pred_df)
            histories[f"{roi}_{model_name}"] = history

    roi_results_df = pd.DataFrame(metrics_rows)
    all_predictions_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return roi_results_df, all_predictions_df, histories


def run_segment_length_experiment(
    subjects: list[int],
    roi_name: str = "ROI_6",
    segment_lengths: list[float] | None = None,
    epochs_list: list[int] | None = None,
    mode: str = "binary",
    split_strategy: SplitStrategy = "subjectwise",
    model_name: str = "cnn_lstm_attention",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Compare segment lengths and training epochs."""
    if segment_lengths is None:
        segment_lengths = [1.0, 4.0, 5.0]
    if epochs_list is None:
        epochs_list = [25, 50]

    class_names = _get_class_names(mode)
    split_fn = _get_split_fn(split_strategy)
    runs = _get_runs(mode)
    rows = []

    for seg_len in segment_lengths:
        print(f"\n--- Segment length: {seg_len}s ---")
        X, y, metadata, _ = build_dataset(
            subjects, runs, mode=mode, segment_length=seg_len, roi_name=roi_name
        )
        X_train, X_test, y_train, y_test, meta_train, meta_test = split_fn(X, y, metadata)

        for n_epochs in epochs_list:
            print(f"  Epochs: {n_epochs}")
            metrics, _, _, _ = run_single_deep_experiment(
                X_train, y_train, X_test, y_test,
                meta_train, meta_test, class_names,
                model_name=model_name,
                epochs=n_epochs,
                split_strategy=split_strategy,
                output_dir=output_dir,
            )
            if metrics:
                metrics.update({
                    "segment_length": seg_len,
                    "epochs": n_epochs,
                    "roi": roi_name,
                    "model": model_name,
                    "mode": mode,
                    "split_strategy": split_strategy,
                })
                rows.append(metrics)

    return pd.DataFrame(rows)


def run_gan_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    class_names: list[str],
    split_strategy: str = "subjectwise",
    num_synthetic_per_class: int = 50,
    gan_config: dict | None = None,
    figures_dir: Path | None = None,
) -> pd.DataFrame:
    """Compare CNN-LSTM-Attention with and without GAN augmentation."""
    from .gan import augment_training_data_with_gan, compare_real_vs_synthetic_stats

    rows = []
    for aug_label, X_tr, y_tr in [
        ("none", X_train, y_train),
    ]:
        metrics, _, _, _ = run_single_deep_experiment(
            X_tr, y_tr, X_test, y_test,
            meta_train, meta_test, class_names,
            model_name="cnn_lstm_attention",
            split_strategy=split_strategy,
        )
        metrics["augmentation"] = aug_label
        rows.append(metrics)

    # GAN augmentation
    X_dl = prepare_deep_learning_input(X_train)
    mean, std = fit_channelwise_standardizer(X_dl)
    X_dl_norm = apply_channelwise_standardizer(X_dl, mean, std)

    try:
        X_aug, y_aug = augment_training_data_with_gan(
            X_dl_norm, y_train, num_synthetic_per_class, gan_config
        )
        syn_only = X_aug[len(X_dl_norm):]
        stats = compare_real_vs_synthetic_stats(X_dl_norm, syn_only)
        print(f"GAN quality check: {stats}")

        if figures_dir is not None:
            plot_real_vs_synthetic_eeg(
                X_dl_norm, syn_only,
                save_path=figures_dir / "real_vs_synthetic_eeg.png",
            )

        # Convert back to (trials, channels, samples) for pipeline consistency
        X_aug_chfirst = prepare_deep_learning_input(
            np.transpose(X_aug, (0, 2, 1))
        )
        # Actually X_aug is already (trials, samples, channels) — use directly
        metrics_gan, _, _, _ = run_single_deep_experiment(
            np.transpose(X_aug, (0, 2, 1)), y_aug,
            X_test, y_test,
            meta_train, meta_test, class_names,
            model_name="cnn_lstm_attention",
            split_strategy=split_strategy,
        )
        metrics_gan["augmentation"] = "gan"
        rows.append(metrics_gan)
    except Exception as exc:
        warnings.warn(f"GAN experiment failed: {exc}")

    return pd.DataFrame(rows)


def run_complete_das2025_replication(
    mode: str = "binary",
    split_strategy: SplitStrategy = "subjectwise",
    run_trialwise_comparison: bool = True,
    run_multiclass: bool = False,
    run_ml: bool = True,
    run_deep: bool = True,
    run_roi_experiments: bool = True,
    run_segment_length_experiment: bool = True,
    run_riemannian: bool = True,
    run_gan: bool = False,
    max_subjects: int | None = None,
    dl_epochs: int = DL_EPOCHS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    paper_input: bool = False,
    segment_length: float | None = None,
    use_paper_roi_epochs: bool = False,
) -> dict[str, Any]:
    """
    Execute the complete Das et al. (2025) replication pipeline.

  Parameters
    ----------
    max_subjects : int, optional
        Limit subjects for quick testing (None = all 103 paper subjects).
    """
    t_start = time.perf_counter()
    paths = get_output_paths(output_dir)
    out_dir = paths["base"]
    figures_dir = paths["figures"]

    np.random.seed(RANDOM_STATE)
    set_reproducibility()

    subjects = get_paper_subjects()
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
        print(f"NOTE: Using first {max_subjects} subjects for quick run.")

    class_names = _get_class_names(mode)
    runs = _get_runs(mode)

    seg_len = segment_length if segment_length is not None else DEFAULT_SEGMENT_LENGTH
    if paper_input and segment_length is None:
        from .paper_input import get_paper_segment_length
        seg_len = get_paper_segment_length(use_640_samples=True)

    config = {
        "mode": mode,
        "split_strategy": split_strategy,
        "subjects": len(subjects),
        "runs": runs,
        "dl_epochs": dl_epochs,
        "segment_length": seg_len,
        "paper_input": paper_input,
        "use_paper_roi_epochs": use_paper_roi_epochs,
        "run_ml": run_ml,
        "run_deep": run_deep,
        "run_roi": run_roi_experiments,
        "run_segment": run_segment_length_experiment,
        "run_riemannian": run_riemannian,
        "run_gan": run_gan,
        "random_state": RANDOM_STATE,
    }
    save_experiment_config(config, out_dir)

    results: dict[str, Any] = {}

    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIG")
    print("=" * 70)
    print(f"  mode:              {mode}")
    print(f"  split:             {split_strategy}")
    print(f"  subjects:          {len(subjects)}")
    print(f"  paper_input:       {paper_input}")
    print(f"  paper_roi_epochs:  {use_paper_roi_epochs}")
    print(f"  segment_length:    {seg_len}s")
    print(f"  dl_epochs:         {dl_epochs}")
    print(f"  run_roi:           {run_roi_experiments}")
    print("=" * 70)

    # --- Primary dataset (ROI_6, 5s) ---
    print("\n" + "=" * 70)
    print("BUILDING PRIMARY DATASET (ROI_6, 5s)")
    print("=" * 70)
    X, y, metadata, roi_channels = build_dataset(
        subjects, runs, mode=mode, segment_length=seg_len,
        roi_name="ROI_6", paper_input=paper_input,
    )
    print_dataset_summary(X, y, metadata, class_names)

    summary_rows = [{
        "n_trials": len(y),
        "n_subjects": metadata["subject"].nunique(),
        "n_channels": X.shape[1],
        "n_samples": X.shape[2],
        "mode": mode,
        "segment_length": DEFAULT_SEGMENT_LENGTH,
        "roi": "ROI_6",
    }]
    pd.DataFrame(summary_rows).to_csv(out_dir / "dataset_summary.csv", index=False)

    # --- Subject-wise split (thesis-grade) ---
    print("\n--- Subject-wise split (primary) ---")
    X_train, X_test, y_train, y_test, meta_train, meta_test = make_subjectwise_split(
        X, y, metadata
    )

    # --- Trial-wise split (paper-like comparison) ---
    if run_trialwise_comparison:
        print("\n--- Trial-wise split (paper-like) ---")
        make_trialwise_split(X, y, metadata)

    # --- ML models ---
    if run_ml:
        print("\n" + "=" * 70)
        print("TRADITIONAL ML MODELS")
        print("=" * 70)
        ml_metrics, ml_preds = run_ml_pipeline(
            X_train, y_train, X_test, y_test, meta_test, class_names
        )
        ml_metrics["split_strategy"] = split_strategy
        ml_metrics["mode"] = mode
        ml_metrics.to_csv(out_dir / "ml_metrics.csv", index=False)
        results["ml_metrics"] = ml_metrics
        print(ml_metrics[["model", "accuracy", "balanced_accuracy", "f1_macro", "mcc"]])

        # PCA / t-SNE on ML features
        X_train_feat, _ = build_ml_feature_matrix(X_train)
        plot_tsne_raw_features(
            X_train_feat[:500], y_train[:500], class_names,
            title="t-SNE raw ML features (train subset)",
            save_path=figures_dir / "tsne_ml_features.png",
        )
        run_pca_visualization(
            X_train_feat[:500], y_train[:500], class_names,
            title="PCA ML features",
            save_path=figures_dir / "pca_ml_features.png",
        )

    # --- Deep models ---
    deep_metrics_all = []
    all_preds = []
    if run_deep:
        print("\n" + "=" * 70)
        print("DEEP LEARNING MODELS")
        print("=" * 70)
        for model_name in ["cnn", "lstm", "cnn_lstm_attention"]:
            metrics, pred_df, history, model = run_single_deep_experiment(
                X_train, y_train, X_test, y_test,
                meta_train, meta_test, class_names,
                model_name=model_name,
                epochs=dl_epochs,
                split_strategy=split_strategy,
                output_dir=out_dir,
                figures_dir=figures_dir,
            )
            if metrics:
                metrics["split_strategy"] = split_strategy
                metrics["mode"] = mode
                metrics["roi"] = "ROI_6"
                metrics["segment_length"] = DEFAULT_SEGMENT_LENGTH
                metrics["augmentation"] = "none"
                deep_metrics_all.append(metrics)
                all_preds.append(pred_df)

                # t-SNE on embeddings for hybrid model
                if model_name == "cnn_lstm_attention" and model is not None:
                    X_tr_dl = prepare_deep_learning_input(X_train)
                    mean_e, std_e = fit_channelwise_standardizer(X_tr_dl)
                    X_te_dl = apply_channelwise_standardizer(
                        prepare_deep_learning_input(X_test), mean_e, std_e
                    )
                    n_emb = min(300, len(y_test))
                    emb = extract_deep_embeddings(model, X_te_dl[:n_emb])
                    plot_tsne_embeddings(
                        emb, y_test[:300], class_names,
                        title="t-SNE CNN-LSTM-Attention embeddings",
                        save_path=figures_dir / "tsne_deep_embeddings.png",
                    )

        if deep_metrics_all:
            deep_df = pd.DataFrame(deep_metrics_all)
            deep_df.to_csv(out_dir / "deep_metrics.csv", index=False)
            results["deep_metrics"] = deep_df
            plot_model_comparison(
                deep_df, metric="accuracy",
                save_path=figures_dir / "deep_model_comparison.png",
            )

    # --- ROI experiments ---
    if run_roi_experiments:
        print("\n" + "=" * 70)
        print("ROI EXPERIMENTS (1-6)")
        print("=" * 70)
        roi_df, roi_preds, _ = run_roi_experiments(
            subjects, mode=mode, segment_length=seg_len,
            split_strategy=split_strategy, epochs=dl_epochs, output_dir=out_dir,
            paper_input=paper_input, use_paper_roi_epochs=use_paper_roi_epochs,
        )
        roi_df.to_csv(out_dir / "roi_results.csv", index=False)
        results["roi_results"] = roi_df
        plot_roi_comparison(roi_df, save_path=figures_dir / "roi_comparison.png")
        if not roi_preds.empty:
            all_preds.append(roi_preds)

    # --- Segment length experiment ---
    if run_segment_length_experiment:
        print("\n" + "=" * 70)
        print("SEGMENT LENGTH EXPERIMENT")
        print("=" * 70)
        seg_df = run_segment_length_experiment(
            subjects, roi_name="ROI_6",
            segment_lengths=[1.0, 4.0, 5.0],
            epochs_list=[25, 50],
            mode=mode, split_strategy=split_strategy,
            output_dir=out_dir,
        )
        seg_df.to_csv(out_dir / "segment_length_results.csv", index=False)
        results["segment_length"] = seg_df

    # --- Riemannian ---
    if run_riemannian:
        print("\n" + "=" * 70)
        print("RIEMANNIAN GEOMETRY BASELINES")
        print("=" * 70)
        riem_metrics, riem_preds = run_riemannian_baselines(
            ensure_channel_first(X_train), y_train,
            ensure_channel_first(X_test), y_test,
            meta_test, class_names,
        )
        if not riem_metrics.empty:
            riem_metrics.to_csv(out_dir / "riemannian_results.csv", index=False)
            results["riemannian"] = riem_metrics

    # --- GAN ---
    if run_gan:
        print("\n" + "=" * 70)
        print("GAN AUGMENTATION COMPARISON")
        print("=" * 70)
        gan_df = run_gan_comparison(
            X_train, y_train, X_test, y_test,
            meta_train, meta_test, class_names,
            split_strategy=split_strategy,
            figures_dir=figures_dir,
        )
        gan_df.to_csv(out_dir / "gan_comparison_results.csv", index=False)
        results["gan"] = gan_df

    # --- Multiclass ---
    if run_multiclass:
        print("\n" + "=" * 70)
        print("MULTICLASS (5-class) EXPERIMENT")
        print("=" * 70)
        run_complete_das2025_replication(
            mode="multiclass",
            split_strategy=split_strategy,
            run_trialwise_comparison=False,
            run_multiclass=False,
            run_ml=run_ml,
            run_deep=run_deep,
            run_roi_experiments=False,
            run_segment_length_experiment=False,
            run_riemannian=run_riemannian,
            run_gan=False,
            max_subjects=max_subjects,
            dl_epochs=dl_epochs,
            output_dir=str(out_dir / "multiclass"),
        )

    # --- Save predictions and errors ---
    if all_preds:
        preds = pd.concat(all_preds, ignore_index=True)
        save_evaluation_outputs(preds, out_dir)
        results["predictions"] = preds

    # --- Classification reports JSON ---
    reports = {}
    for key, df in results.items():
        if isinstance(df, pd.DataFrame) and "classification_report" in df.columns:
            pass
    with open(out_dir / "classification_reports.json", "w") as f:
        json.dump(reports, f, indent=2, default=str)

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — elapsed {elapsed / 60:.1f} min")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Das et al. (2025) EEG MI classification replication"
    )
    parser.add_argument("--mode", default="binary", choices=["binary", "multiclass"])
    parser.add_argument(
        "--split", default="subjectwise", choices=["subjectwise", "trialwise"]
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=DL_EPOCHS)
    parser.add_argument("--quick", action="store_true", help="Fast test with 5 subjects")
    parser.add_argument("--gan", action="store_true", help="Enable GAN augmentation")
    parser.add_argument("--no-roi", action="store_true", help="Skip ROI loop")
    parser.add_argument("--no-segment", action="store_true", help="Skip segment length exp")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--paper-input",
        action="store_true",
        help="Use paper 640x2 contralateral pair input (Das et al. 2025)",
    )
    parser.add_argument(
        "--paper-roi-epochs",
        action="store_true",
        help="Use per-ROI training epochs from paper Table 6",
    )
    parser.add_argument(
        "--trialwise",
        action="store_true",
        help="Trial-wise split (closer to undocumented paper protocol)",
    )
    args = parser.parse_args()

    max_subj = 5 if args.quick else args.max_subjects

    run_complete_das2025_replication(
        mode=args.mode,
        split_strategy="trialwise" if args.trialwise else args.split,
        run_gan=args.gan,
        run_roi_experiments=not args.no_roi,
        run_segment_length_experiment=not args.no_segment,
        max_subjects=max_subj,
        dl_epochs=args.epochs,
        output_dir=args.output_dir,
        paper_input=args.paper_input,
        use_paper_roi_epochs=args.paper_roi_epochs,
    )
