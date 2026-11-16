"""Date-indexed regional feasibility test, not a resort or operational forecast.

Predict the daily SNOW record at KEYSTONE 0.7 SW from Dillon observations dated
before the target record. Four winters are held out in turn. Observation windows
and historical publication latency have not been aligned, so these are statistical
cross-validation results rather than verified issue-time forecast performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
TARGET = "US1COSU0052"
FEATURES = [
    "snow_lag1_inches", "snow_prev3_inches", "snow_prev7_inches",
    "lag1_tmax_celsius", "lag1_tmin_celsius", "lag1_precipitation_mm",
    "lag1_snow_depth_cm", "calendar_sin", "calendar_cos",
]
GRIDS = {"ridge": [10.0, 100.0], "hist_gradient_boosting": [7, 15]}


def calendar_prediction(train, test):
    """Local 31-calendar-day target-site mean; training fold observations only."""
    delta = np.abs(test.calendar_day.to_numpy()[:, None] - train.calendar_day.to_numpy()[None, :])
    mask = np.minimum(delta, 366 - delta) <= 15
    counts = mask.sum(axis=1)
    values = mask @ train.target_inches.to_numpy()
    return np.divide(values, counts, out=np.full(len(test), train.target_inches.mean()), where=counts > 0)


def make_model(family, parameter):
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=parameter))
    return HistGradientBoostingRegressor(
        max_leaf_nodes=parameter, min_samples_leaf=30, max_iter=80,
        learning_rate=0.05, l2_regularization=10.0, early_stopping=False,
        random_state=42,
    )


def predict(model, test):
    return np.maximum(0.0, model.predict(test[FEATURES]))


def metric(y, p):
    error = np.asarray(p) - np.asarray(y)
    return {"mse_inches_squared": float(np.mean(error ** 2)),
            "rmse_inches": float(np.sqrt(np.mean(error ** 2))),
            "mae_inches": float(np.mean(np.abs(error))),
            "bias_inches": float(np.mean(error))}


def tune(train, family):
    results = []
    for parameter in GRIDS[family]:
        losses = []
        for season in sorted(train.winter_start_year.unique()):
            inner_train = train.loc[train.winter_start_year != season]
            inner_test = train.loc[train.winter_start_year == season]
            model = make_model(family, parameter).fit(inner_train[FEATURES], inner_train.target_inches)
            losses.append(metric(inner_test.target_inches, predict(model, inner_test))["mse_inches_squared"])
        results.append({"parameter": parameter, "mean_inner_winter_mse": float(np.mean(losses))})
    # The grids and objective are fixed before inspecting any outer-fold result.
    selected = min(results, key=lambda entry: entry["mean_inner_winter_mse"])["parameter"]
    return selected, results


def main():
    daily = pd.read_csv(ROOT / "outputs/dillon_daily_features.csv", parse_dates=["date"]).set_index("date")
    target = pd.read_csv(ROOT / "data/target_nearby_snow_daily.csv", parse_dates=["date"])
    target = target.loc[target.station_id == TARGET].set_index("date")
    assert target.index.is_unique and daily.index.is_unique
    target["target_inches"] = target.snowfall_inches.where(target.usable == 1)
    joined = daily[FEATURES].join(target[["target_inches"]], how="left")
    joined["winter_start_year"] = np.where(joined.index.month >= 10, joined.index.year, joined.index.year - 1)
    # Use a common leap-year calendar so month/day matches across leap years.
    joined["calendar_day"] = [pd.Timestamp(2000, d.month, d.day).dayofyear for d in joined.index]
    seasonal = joined.loc[joined.index.month.isin([10, 11, 12, 1, 2, 3, 4]) & joined.winter_start_year.isin(range(2012, 2016))]
    cohort = seasonal.dropna(subset=FEATURES + ["target_inches"]).copy()
    assert set(cohort.winter_start_year) == {2012, 2013, 2014, 2015}
    assert all(name.startswith(("snow_lag", "snow_prev", "lag1_", "calendar_")) for name in FEATURES)
    coverage = []
    for season, frame in seasonal.groupby("winter_start_year"):
        usable = cohort.loc[cohort.winter_start_year == season]
        coverage.append({"winter": f"{season}/{season+1}", "calendar_days": len(frame),
                         "usable_target_days": int(frame.target_inches.notna().sum()),
                         "shared_evaluation_days": len(usable),
                         "nonzero_target_days": int((usable.target_inches > 0).sum())})
    monthly = []
    for month, frame in seasonal.groupby(seasonal.index.to_period("M")):
        monthly.append({"month": str(month), "calendar_days": len(frame),
                        "usable_target_days": int(frame.target_inches.notna().sum()),
                        "shared_evaluation_days": int(frame[FEATURES + ["target_inches"]].notna().all(axis=1).sum())})

    predictions, folds = [], []
    for season in sorted(cohort.winter_start_year.unique()):
        train = cohort.loc[cohort.winter_start_year != season]
        test = cohort.loc[cohort.winter_start_year == season]
        # Whole winter removal applies to every station's rows in the model.
        assert not (set(train.index) & set(test.index))
        out = test[["target_inches", "winter_start_year"]].copy()
        out["zero"] = 0.0
        out["target_station_mean"] = train.target_inches.mean()
        out["target_calendar_31day"] = calendar_prediction(train, test)
        fold = {"held_out_winter": f"{season}/{season+1}", "training_days": len(train),
                "test_days": len(test), "models": {}}
        for family in GRIDS:
            selected, inner_results = tune(train, family)
            model = make_model(family, selected).fit(train[FEATURES], train.target_inches)
            out[family] = predict(model, test)
            fold["models"][family] = {"selected_parameter": selected,
                                      "inner_validation": inner_results,
                                      "training_metrics": metric(train.target_inches, predict(model, train))}
        for family in ["zero", "target_station_mean", "target_calendar_31day", *GRIDS]:
            fold["models"].setdefault(family, {})["test_metrics"] = metric(out.target_inches, out[family])
        predictions.append(out)
        folds.append(fold)
    out = pd.concat(predictions).sort_index()
    assert len(out) == len(cohort) and out.index.is_unique
    assert not out.isna().any().any()
    summary = {}
    baseline = metric(out.target_inches, out.target_calendar_31day)["mse_inches_squared"]
    for family in ["zero", "target_station_mean", "target_calendar_31day", *GRIDS]:
        summary[family] = metric(out.target_inches, out[family])
        summary[family]["mse_skill_vs_calendar"] = 1.0 - summary[family]["mse_inches_squared"] / baseline
        summary[family]["mean_winter_mse_inches_squared"] = float(np.mean([
            fold["models"][family]["test_metrics"]["mse_inches_squared"] for fold in folds]))
    report = {
        "as_of": "2026-11-03", "experiment": "Date-indexed regional feasibility test",
        "target_station": {"id": TARGET, "name": "KEYSTONE 0.7 SW", "is_resort": False,
                           "latitude": 39.6011, "longitude": -105.9773, "elevation_m": 2819.4,
                           "source_url": f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{TARGET}.dly"},
        "feature_station": {"id": "USC00052281", "name": "DILLON 1 E"},
        "horizon": "Target date t daily SNOW from observations dated t-1 or earlier, subject to observation-window caveat",
        "features": FEATURES, "target_unit": "inches of reported new snowfall",
        "cohort_policy": "Only October-April 2012/13 through 2015/16, usable observed target, and all requested lagged Dillon features present. Every model uses the same complete-case evaluation rows; missing targets are never zero-filled.",
        "validation": "Four outer leave-one-winter-out folds. Three inner leave-one-winter-out folds select model hyperparameters using equal-weight winter MSE. All scaling is fit on training folds. Entire held-out winter is excluded, including Dillon covariate rows as training examples. Historical folds may train on later winters, under an exchangeable-year assumption; this is not chronological deployment simulation.",
        "hyperparameters": {"ridge_alphas": GRIDS["ridge"], "boosting_max_leaf_nodes": GRIDS["hist_gradient_boosting"],
                            "boosting_fixed": {"min_samples_leaf": 30, "max_iter": 80, "learning_rate": 0.05,
                                               "l2_regularization": 10.0, "early_stopping": False},
                            "prediction_postprocessing": "Clip negative predictions to zero before tuning and evaluation"},
        "coverage_by_winter": coverage, "coverage_by_month": monthly, "n_test_days": len(out),
        "pooled_test_metrics": summary, "outer_folds": folds,
        "limitations": [
            "This target is a lower-elevation community station, not Keystone Resort or A-Basin.",
            "Observation windows and original publication timestamps are not aligned; dated lags avoid same-date predictors but cannot guarantee operational forecast availability.",
            "Only four winters are available. Results cannot establish a reliable best model, climate regime response, or future mountain forecast skill.",
            "Missing target reports can bias the evaluated subset. Coverage is reported and no missing snowfall is assigned zero.",
            "Daily temporal persistence is different from month-ahead predictability. This experiment does not demonstrate improved monthly forecasts.",
            "No archived numerical weather forecasts are used, so this is a lagged-observation benchmark rather than a full weather forecasting system.",
        ],
        "checks": ["All models score identical target dates", "No same-date Dillon observations used as predictors",
                   "Whole-winter outer and inner partitions", "Fold-local feature scaling", "Nonnegative finite predictions"],
    }
    out.to_csv(ROOT / "outputs/keystone_station_transfer_predictions.csv", float_format="%.8f")
    (ROOT / "outputs/keystone_station_transfer_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"coverage": coverage, "metrics": summary}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
