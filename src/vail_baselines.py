"""Fixed statistical baselines for Vail hourly-input / daily-output experiments.

All fitting, imputation and standardization use training rows only. This module
does not split dates or tune parameters: the caller must supply disjoint winter
groups. The flattened daily targets remain conditional on their issue's past
observations, known future calendar features and deterministic lead times.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SUMMARY_STATISTICS = (
    "latest", "last6_mean", "last24_mean", "last72_mean", "last168_mean",
    "last24_std", "last24_missing_fraction", "last168_missing_fraction",
    "last_valid_value", "hours_since_last_valid",
)

BASELINE_CONFIG = {
    "Ridge": {"alpha": 100.0},
    "HGB": {
        "loss": "squared_error", "max_iter": 120, "max_leaf_nodes": 7,
        "min_samples_leaf": 40, "l2_regularization": 10.0,
        "learning_rate": 0.05, "early_stopping": False,
        "random_state": 20261104,
    },
    "Persistence": {"fallback": "mean of observed training targets"},
}


def tabular_summary(xraw: np.ndarray) -> np.ndarray:
    """Summarize strictly past context [issue, hours, channel] as [issue, 10*C].

    The context must end at issue time. Missing values must remain NaN, rather
    than already being filled with zero. Features are statistic-major, in the
    order ``SUMMARY_STATISTICS``; each statistic contains every input channel.
    An entirely missing channel has missing numerical summaries, missingness
    fractions of one, and last-valid age equal to the context length. Age zero
    denotes a valid final hourly reading.
    """
    x = np.asarray(xraw, dtype=np.float32).copy()
    if x.ndim != 3 or x.shape[1] < 1:
        raise ValueError("Expected nonempty [issue, hours, channel] context")
    x[~np.isfinite(x)] = np.nan
    valid = np.isfinite(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        parts = [x[:, -1, :]]
        parts.extend(np.nanmean(x[:, -window:, :], axis=1)
                     for window in (6, 24, 72, 168))
        parts.append(np.nanstd(x[:, -24:, :], axis=1))
    parts.extend([1.0 - valid[:, -24:, :].mean(axis=1),
                  1.0 - valid[:, -168:, :].mean(axis=1)])
    has_value = valid.any(axis=1)
    reverse_age = np.argmax(valid[:, ::-1, :], axis=1)
    last_index = x.shape[1] - 1 - reverse_age
    last_value = np.take_along_axis(x, last_index[:, None, :], axis=1)[:, 0, :]
    last_value[~has_value] = np.nan
    age = np.where(has_value, reverse_age, x.shape[1]).astype(np.float32)
    parts.extend([last_value, age])
    return np.concatenate(parts, axis=1).astype(np.float32)


def _design(summary, future, issue_snow, fallback):
    summary = np.asarray(summary, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    snow = np.asarray(issue_snow, dtype=np.float32).reshape(-1)
    if summary.ndim != 2 or future.ndim != 3:
        raise ValueError("Expected summary[N,S] and future[N,H,F]")
    n, h, _ = future.shape
    if summary.shape[0] != n or snow.shape[0] != n or h < 1:
        raise ValueError("Issue dimensions differ, or forecast horizon is empty")
    missing = ~np.isfinite(snow)
    snow = np.where(missing, fallback, snow)
    leads = np.arange(1, h + 1, dtype=np.float32)
    # Shared explicit lead interactions give the linear baseline the ability
    # to let current conditions' influence decay. No realized future weather.
    decay = np.stack([np.ones(h), np.exp(-(leads - 1) / 3.0),
                      np.exp(-(leads - 1) / 10.0)], axis=-1).astype(np.float32)
    snow_features = snow[:, None, None] * decay[None, :, :]
    missing_feature = np.broadcast_to(missing[:, None, None], (n, h, 1))
    return np.concatenate([
        np.broadcast_to(summary[:, None, :], (n, h, summary.shape[1])),
        future, snow_features, missing_feature,
    ], axis=-1).reshape(n * h, -1).astype(np.float32)


def fit_predict_baselines(train_summary, train_future, train_y, train_mask,
                          test_summary, test_future, train_issue_snow,
                          test_issue_snow, *, return_models=False):
    """Fit fixed Ridge/HGB and return daily means for three baseline families.

    Required shapes: summary[N,S], future[N,H,F], y/mask[N,H], issue_snow[N].
    ``train_issue_snow`` and ``test_issue_snow`` are the most recent observed
    reported endpoint measurement at each issue, not a future target. Missing readings
    fall back to the training-label mean. Invalid/masked training targets are
    excluded, never made into zero. Predictions are clipped below at zero.

    Returns ``{'Ridge': [Ntest,H], 'HGB': ..., 'Persistence': ...}`` by default.
    With ``return_models=True``, returns ``(predictions, bundle)``; the bundle
    contains fitted estimators, training-only imputation/scaling and metadata,
    and can be serialized with joblib. No model selection or subsampling occurs.
    """
    y = np.asarray(train_y, dtype=np.float32)
    mask = np.asarray(train_mask, dtype=bool)
    if y.ndim != 2 or mask.shape != y.shape:
        raise ValueError("Training labels and mask must have equal [N,H] shape")
    if np.asarray(train_future).shape[:2] != y.shape:
        raise ValueError("Training future features and labels differ in shape")
    eligible = mask & np.isfinite(y) & (y >= 0)
    if not eligible.any():
        raise ValueError("No nonnegative observed training target")
    fallback = float(y[eligible].mean())
    fit_x = _design(train_summary, train_future, train_issue_snow, fallback)
    test_x = _design(test_summary, test_future, test_issue_snow, fallback)
    fit_x = fit_x[eligible.reshape(-1)]
    fit_y = y[eligible]
    if fit_x.shape[1] != test_x.shape[1]:
        raise ValueError("Training and test feature dimensions differ")

    # Explicit fitting also handles entirely missing channels without dropping
    # them, keeping a stable schema across winter folds and restored models.
    fit_x[~np.isfinite(fit_x)] = np.nan
    test_x[~np.isfinite(test_x)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        imputation_mean = np.nanmean(fit_x, axis=0)
    imputation_mean = np.nan_to_num(imputation_mean, nan=0.0)
    fit_x = np.where(np.isfinite(fit_x), fit_x, imputation_mean)
    test_x = np.where(np.isfinite(test_x), test_x, imputation_mean)
    scaler = StandardScaler()
    ridge = Ridge(**BASELINE_CONFIG["Ridge"])
    hgb = HistGradientBoostingRegressor(**BASELINE_CONFIG["HGB"])
    with threadpool_limits(limits=2):
        ridge.fit(scaler.fit_transform(fit_x), fit_y)
        ridge_prediction = ridge.predict(scaler.transform(test_x))
        hgb.fit(fit_x, fit_y)
        hgb_prediction = hgb.predict(test_x)
    shape = np.asarray(test_future).shape[:2]
    latest = np.asarray(test_issue_snow, dtype=np.float32).reshape(-1)
    latest = np.where(np.isfinite(latest), latest, fallback)
    predictions = {
        "Ridge": np.maximum(ridge_prediction.reshape(shape), 0),
        "HGB": np.maximum(hgb_prediction.reshape(shape), 0),
        "Persistence": np.broadcast_to(np.maximum(latest, 0)[:, None], shape).copy(),
    }
    if not return_models:
        return predictions
    bundle = {
        "Ridge": ridge, "HGB": hgb, "scaler": scaler,
        "imputation_mean": imputation_mean, "issue_snow_fallback": fallback,
        "config": BASELINE_CONFIG, "training_pairs": int(eligible.sum()),
        "training_issues": int(y.shape[0]), "feature_count": int(fit_x.shape[1]),
        "summary_statistics": SUMMARY_STATISTICS,
        "design": "summary + future + issue_snow * [1,exp(-(lead-1)/3),exp(-(lead-1)/10)] + issue_snow_missing",
        "hyperparameter_selection": "Fixed before held-out scoring; no search",
        "subsampling": None,
    }
    return predictions, bundle


def predict_baselines(bundle, summary, future, issue_snow):
    """Restore the exact prediction pipeline from a serialized fitted bundle."""
    x = _design(summary, future, issue_snow, bundle["issue_snow_fallback"])
    x = np.where(np.isfinite(x), x, bundle["imputation_mean"])
    shape = np.asarray(future).shape[:2]
    latest = np.asarray(issue_snow, dtype=np.float32).reshape(-1)
    latest = np.where(np.isfinite(latest), latest, bundle["issue_snow_fallback"])
    with threadpool_limits(limits=2):
        return {
            "Ridge": np.maximum(bundle["Ridge"].predict(bundle["scaler"].transform(x)).reshape(shape), 0),
            "HGB": np.maximum(bundle["HGB"].predict(x).reshape(shape), 0),
            "Persistence": np.broadcast_to(np.maximum(latest, 0)[:, None], shape).copy(),
        }
