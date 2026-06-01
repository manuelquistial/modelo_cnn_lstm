"""
Time-domain, frequency-domain, and wavelet feature extraction for traditional ML.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.signal import welch

from .config import SFREQ

try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def _bandpower(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _spectral_features(psd: np.ndarray, freqs: np.ndarray) -> dict[str, float]:
    psd = np.maximum(psd, 1e-12)
    total = np.sum(psd)
    if total <= 0:
        return {
            "mean_freq": 0.0,
            "median_freq": 0.0,
            "freq_variance": 0.0,
            "freq_std": 0.0,
            "total_power": 0.0,
        }
    mean_freq = float(np.sum(freqs * psd) / total)
    cumsum = np.cumsum(psd)
    median_idx = np.searchsorted(cumsum, 0.5 * cumsum[-1])
    median_freq = float(freqs[min(median_idx, len(freqs) - 1)])
    freq_variance = float(np.sum(((freqs - mean_freq) ** 2) * psd) / total)
    return {
        "mean_freq": mean_freq,
        "median_freq": median_freq,
        "freq_variance": freq_variance,
        "freq_std": float(np.sqrt(freq_variance)),
        "total_power": float(total),
    }


def extract_time_frequency_features(
    X: np.ndarray,
    fs: float = SFREQ,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract time- and frequency-domain features per trial and channel.

    X shape: (n_trials, n_channels, n_samples)
    """
    n_trials, n_channels, n_samples = X.shape
    base_time = [
        "mean", "median", "variance", "std", "min", "max",
        "skewness", "kurtosis", "rms", "zcr",
    ]
    base_freq = [
        "mean_freq", "median_freq", "freq_variance", "freq_std", "total_power",
        "delta", "theta", "alpha", "beta", "gamma",
    ]
    feature_names = [
        f"ch{ch}_{feat}"
        for ch in range(n_channels)
        for feat in base_time + base_freq
    ]

    feat_list: list[np.ndarray] = []
    for ch in range(n_channels):
        ch_data = X[:, ch, :]
        feat_list.extend([
            np.mean(ch_data, axis=1),
            np.median(ch_data, axis=1),
            np.var(ch_data, axis=1),
            np.std(ch_data, axis=1),
            np.min(ch_data, axis=1),
            np.max(ch_data, axis=1),
            stats.skew(ch_data, axis=1),
            stats.kurtosis(ch_data, axis=1),
            np.sqrt(np.mean(ch_data ** 2, axis=1)),
            np.mean(np.abs(np.diff(np.sign(ch_data), axis=1)), axis=1) / 2.0,
        ])
        for trial in range(n_trials):
            freqs, psd = welch(
                ch_data[trial], fs=fs, nperseg=min(256, n_samples)
            )
            spec = _spectral_features(psd, freqs)
            if trial == 0:
                freq_arrays = {name: [] for name in base_freq}
            freq_arrays["mean_freq"].append(spec["mean_freq"])
            freq_arrays["median_freq"].append(spec["median_freq"])
            freq_arrays["freq_variance"].append(spec["freq_variance"])
            freq_arrays["freq_std"].append(spec["freq_std"])
            freq_arrays["total_power"].append(spec["total_power"])
            freq_arrays["delta"].append(_bandpower(freqs, psd, 0.5, 4))
            freq_arrays["theta"].append(_bandpower(freqs, psd, 4, 8))
            freq_arrays["alpha"].append(_bandpower(freqs, psd, 8, 13))
            freq_arrays["beta"].append(_bandpower(freqs, psd, 13, 30))
            freq_arrays["gamma"].append(_bandpower(freqs, psd, 30, 50))
        for name in base_freq:
            feat_list.append(np.array(freq_arrays[name]))

    features = np.column_stack(feat_list)
    return features, feature_names


def extract_wavelet_features(
    X: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """Extract wavelet coefficient statistics per channel."""
    if not HAS_PYWT:
        warnings.warn("PyWavelets not installed; returning empty wavelet features.")
        return np.zeros((X.shape[0], 0)), []

    n_trials, n_channels, _ = X.shape
    names = ["w_energy", "w_mean", "w_std", "w_entropy"]
    feature_names = [f"ch{ch}_{n}" for ch in range(n_channels) for n in names]
    all_ch_feats: list[np.ndarray] = []

    for ch in range(n_channels):
        energies, means, stds, entropies = [], [], [], []
        for trial in range(n_trials):
            coeffs = pywt.wavedec(X[trial, ch, :], wavelet, level=level)
            flat = np.concatenate([c.ravel() for c in coeffs])
            energies.append(np.sum(flat ** 2))
            means.append(np.mean(flat))
            stds.append(np.std(flat))
            p = np.abs(flat) / (np.sum(np.abs(flat)) + 1e-12)
            entropies.append(-np.sum(p * np.log(p + 1e-12)))
        all_ch_feats.extend([
            np.array(energies),
            np.array(means),
            np.array(stds),
            np.array(entropies),
        ])

    return np.column_stack(all_ch_feats), feature_names


def build_ml_feature_matrix(
    X: np.ndarray,
    fs: float = SFREQ,
    use_time_freq: bool = True,
    use_wavelet: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Concatenate selected feature sets for ML classifiers."""
    parts: list[np.ndarray] = []
    names: list[str] = []
    if use_time_freq:
        tf, tf_names = extract_time_frequency_features(X, fs=fs)
        parts.append(tf)
        names.extend(tf_names)
    if use_wavelet:
        if HAS_PYWT:
            wv, wv_names = extract_wavelet_features(X)
            if wv.shape[1] > 0:
                parts.append(wv)
                names.extend(wv_names)
        else:
            warnings.warn("Skipping wavelet features: PyWavelets not installed.")
    if not parts:
        raise ValueError("No features extracted.")
    return np.hstack(parts), names
