"""Feature extraction and logistic models used for Figure 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FEATURE_NAMES = ("sum", "gradient", "motion")


def as_bilateral_traces(samples: np.ndarray) -> np.ndarray:
    """Normalize bilateral samples to shape ``(sample, time, antenna)``."""

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 4 and samples.shape[2] == 1 and samples.shape[3] == 2:
        samples = samples[:, :, 0, :]
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(
            "Expected samples shaped (N, T, 2) or legacy (N, T, 1, 2); "
            f"got {samples.shape}."
        )
    if samples.shape[1] < 2:
        raise ValueError(
            "At least two time points are required for the motion feature."
        )
    return samples


def log_threshold_transform(samples: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Apply the preprint's ``log(1 + max(I - threshold, 0))`` transform."""

    samples = np.asarray(samples, dtype=np.float64)
    return np.log1p(np.maximum(samples - threshold, 0.0))


def centerline_features(
    samples: np.ndarray,
    *,
    delay: int = 1,
    convention: str = "legacy",
) -> np.ndarray:
    """Return temporal-mean sum, gradient, and Reichardt motion features.

    Antenna index 0 is ``R`` and index 1 is ``L``, matching the legacy data
    arrays.  The signs therefore follow ``R-L`` and
    ``R(t)L(t+d)-R(t+d)L(t)``.
    """

    traces = as_bilateral_traces(samples)
    if delay < 1 or delay >= traces.shape[1]:
        raise ValueError(f"delay must be in [1, {traces.shape[1] - 1}]; got {delay}.")
    right = traces[..., 0]
    left = traces[..., 1]
    summed = 0.5 * (right + left).mean(axis=1)
    gradient = (right - left).mean(axis=1)
    motion = (
        right[:, :-delay] * left[:, delay:] - right[:, delay:] * left[:, :-delay]
    ).mean(axis=1)
    if convention == "manuscript":
        gradient = -gradient
        motion = -motion
    elif convention != "legacy":
        raise ValueError("convention must be 'legacy' or 'manuscript'.")
    return np.column_stack((summed, gradient, motion))


def extract_centerline_features(
    samples: Any,
    *,
    delay: int = 1,
    convention: str = "legacy",
    apply_log_transform: bool = True,
    threshold: float = 5.0,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Extract the three features without materializing a full movie copy.

    Legacy refits contain millions of samples.  Processing slices preserves
    memory mapping and also reproduces the notebook's ``np.nan_to_num``
    sanitation before the optional intensity transform.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    if not hasattr(samples, "shape") or len(samples.shape) not in (3, 4):
        raise ValueError("Samples must be an array-like object with 3 or 4 axes.")
    sample_count = int(samples.shape[0])
    output = np.empty((sample_count, len(FEATURE_NAMES)), dtype=np.float64)
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        chunk = np.array(samples[start:stop], copy=True)
        chunk = np.nan_to_num(chunk, copy=False)
        if apply_log_transform:
            chunk = log_threshold_transform(chunk, threshold=threshold)
        output[start:stop] = centerline_features(
            chunk, delay=delay, convention=convention
        )
    return output


def standardize_from_training(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_features = np.asarray(train_features, dtype=np.float64)
    test_features = np.asarray(test_features, dtype=np.float64)
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("Feature arrays must be two-dimensional.")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Train and test feature arrays must have the same columns.")
    means = train_features.mean(axis=0)
    stds = train_features.std(axis=0)
    if np.any(stds == 0):
        zero_names = [str(index) for index in np.flatnonzero(stds == 0)]
        raise ValueError(
            f"Cannot standardize constant training feature columns: {zero_names}"
        )
    return (train_features - means) / stds, (test_features - means) / stds, means, stds


@dataclass(frozen=True)
class Figure2Fit:
    auc: dict[str, float]
    weights: dict[str, float]
    means: np.ndarray
    stds: np.ndarray
    all_feature_auc: float


def fit_figure2_models(
    train_samples: np.ndarray,
    train_labels: np.ndarray,
    test_samples: np.ndarray,
    test_labels: np.ndarray,
    *,
    delay: int = 1,
    apply_log_transform: bool = True,
    threshold: float = 5.0,
    chunk_size: int = 100_000,
) -> Figure2Fit:
    """Fit the three single-feature GLMs and the combined Figure 2 GLM."""

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Fitting Figure 2 models requires scikit-learn.") from exc

    raw_train = extract_centerline_features(
        train_samples,
        delay=delay,
        apply_log_transform=apply_log_transform,
        threshold=threshold,
        chunk_size=chunk_size,
    )
    raw_test = extract_centerline_features(
        test_samples,
        delay=delay,
        apply_log_transform=apply_log_transform,
        threshold=threshold,
        chunk_size=chunk_size,
    )
    train_x, test_x, means, stds = standardize_from_training(raw_train, raw_test)
    y_train = np.asarray(train_labels).reshape(-1)
    y_test = np.asarray(test_labels).reshape(-1)
    if len(y_train) != len(train_x) or len(y_test) != len(test_x):
        raise ValueError("Signal and label arrays have incompatible sample counts.")

    auc: dict[str, float] = {}
    for column, name in enumerate(FEATURE_NAMES):
        model = LogisticRegression(random_state=0, max_iter=1_000_000, penalty=None)
        model.fit(train_x[:, column : column + 1], y_train)
        auc[name] = float(
            roc_auc_score(
                y_test, model.predict_proba(test_x[:, column : column + 1])[:, 1]
            )
        )

    combined = LogisticRegression(random_state=0, max_iter=1_000_000, penalty=None)
    combined.fit(train_x, y_train)
    all_auc = float(roc_auc_score(y_test, combined.predict_proba(test_x)[:, 1]))
    weights = {
        name: float(combined.coef_[0, column])
        for column, name in enumerate(FEATURE_NAMES)
    }
    return Figure2Fit(
        auc=auc, weights=weights, means=means, stds=stds, all_feature_auc=all_auc
    )


def aggregate_figure2_fits(fits: list[Figure2Fit]) -> list[dict[str, Any]]:
    if not fits:
        raise ValueError("At least one fit is required.")
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_NAMES:
        auc_values = np.asarray([fit.auc[feature] for fit in fits])
        weight_values = np.asarray([fit.weights[feature] for fit in fits])
        rows.append(
            {
                "feature": feature,
                "auc_mean": float(auc_values.mean()),
                "auc_sd": float(auc_values.std()),
                "weight_mean": float(weight_values.mean()),
                "weight_sd": float(weight_values.std()),
                "repeats": len(fits),
            }
        )
    return rows
