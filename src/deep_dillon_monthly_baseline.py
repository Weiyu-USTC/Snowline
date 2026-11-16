"""Direct monthly comparator for the Oct31 daily-network benchmark.

One fixed train/validation/test split; no test-based tuning. Complete monthly
snowfall targets only. October observations are the only inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'outputs/dillon_deep_daily_input.csv'
FEATURES = ['oct_snow_inches', 'oct_precip_mm', 'oct_tmax_celsius',
            'oct_tmin_celsius', 'oct_mean_depth_cm']
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
MONTHS = [11, 12]


def build_season_table(frame):
    rows = []
    for year in range(2005, 2026):
        october = frame.reindex(pd.date_range(f'{year}-10-01', f'{year}-10-31'))
        row = {'season_start_year': year}
        for feature, column, mode in [
            ('oct_snow_inches', 'snowfall_inches', 'sum'),
            ('oct_precip_mm', 'precipitation_mm', 'sum'),
            ('oct_tmax_celsius', 'tmax_celsius', 'mean'),
            ('oct_tmin_celsius', 'tmin_celsius', 'mean'),
            ('oct_mean_depth_cm', 'snow_depth_cm', 'mean'),
        ]:
            values = october[column]
            # Do not pass a partial month's sum/mean as a full monthly measure.
            row[feature] = float(getattr(values, mode)()) if values.notna().all() else np.nan
            row[feature + '_usable_days'] = int(values.notna().sum())
        for month in MONTHS:
            period = pd.Period(f'{year}-{month:02}', freq='M')
            days = pd.date_range(period.start_time, period.end_time.normalize())
            snow = frame.reindex(days).snowfall_inches
            row[f'target_{month:02}_inches'] = float(snow.sum()) if snow.notna().all() else np.nan
            row[f'target_{month:02}_usable_days'] = int(snow.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index('season_start_year')


def pipeline(alpha):
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                         StandardScaler(), Ridge(alpha=alpha))


def predictions(train, test, alpha=None):
    rows = []
    fitted = {}
    for month in MONTHS:
        target = f'target_{month:02}_inches'
        tr = train.loc[train[target].notna()]
        te = test.loc[test[target].notna()]
        if alpha is None:
            prediction = np.full(len(te), tr[target].mean())
            model = None
        else:
            model = pipeline(alpha).fit(tr[FEATURES], tr[target])
            prediction = np.maximum(0, model.predict(te[FEATURES]))
        fitted[month] = {'n_train_complete_months': len(tr),
                         'training_years': [int(y) for y in tr.index],
                         'training_target_mean_inches': float(tr[target].mean())}
        for year, actual, predicted in zip(te.index, te[target], prediction):
            rows.append({'season_start_year': int(year), 'month': month,
                         'actual_inches': float(actual), 'predicted_inches': float(predicted),
                         'error_inches': float(predicted - actual)})
    return pd.DataFrame(rows).sort_values(['season_start_year', 'month']), fitted


def metrics(rows):
    error = rows.error_inches.to_numpy()
    by_month = {}
    for month, group in rows.groupby('month'):
        e = group.error_inches.to_numpy()
        by_month[str(month)] = {'n': len(e), 'rmse_inches': float(np.sqrt(np.mean(e * e))),
                                'mae_inches': float(np.mean(np.abs(e)))}
    return {'n_monthly_targets': len(rows), 'n_seasons': int(rows.season_start_year.nunique()),
            'mse_inches_squared': float(np.mean(error * error)),
            'rmse_inches': float(np.sqrt(np.mean(error * error))),
            'mae_inches': float(np.mean(np.abs(error))), 'per_month': by_month}


def main():
    frame = pd.read_csv(INPUT, parse_dates=['date']).set_index('date')
    seasons = build_season_table(frame)
    train = seasons.loc[2005:2016]
    validation = seasons.loc[2017:2020]
    refit = seasons.loc[2005:2020]
    test = seasons.loc[2021:2025]
    tuning = []
    for alpha in ALPHAS:
        pred, _ = predictions(train, validation, alpha)
        tuning.append({'alpha': alpha, **metrics(pred)})
    selected_alpha = min(tuning, key=lambda row: (row['mse_inches_squared'], -row['alpha']))['alpha']
    clim_val, _ = predictions(train, validation)
    ridge_val, _ = predictions(train, validation, selected_alpha)
    selected_family = 'direct_monthly_ridge' if metrics(ridge_val)['mse_inches_squared'] < metrics(clim_val)['mse_inches_squared'] else 'direct_monthly_climatology'
    clim_test, clim_fit = predictions(refit, test)
    ridge_test, ridge_fit = predictions(refit, test, selected_alpha)
    output = {
        'input': str(INPUT.relative_to(ROOT)),
        'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'forecast_origin': 'October31 after that day observations, each test winter',
        'targets': ['November snowfall total (inches)', 'December snowfall total (inches)'],
        'train_season_start_years': [2005, 2016],
        'validation_season_start_years': [2017, 2020],
        'final_fit_season_start_years': [2005, 2020],
        'heldout_season_start_years': [2021, 2025],
        'features': FEATURES,
        'missing_input_policy': 'Any incomplete October aggregate is missing. Median imputation plus missing indicators fit only within the training partition; never replace a missing target or missing day by zero.',
        'target_policy': 'Fit each target month separately using only complete daily snowfall months; incomplete Nov2005 excluded from November fit but valid Dec2005 retained for December fit.',
        'predeclared_alpha_grid': ALPHAS,
        'tuning_metric': 'Combined November/December validation mean squared error, same alpha for both target-specific models. Nonnegative predictions clipped at zero.',
        'tuning': tuning,
        'selected_ridge_alpha': selected_alpha,
        'selected_family_using_validation_only': selected_family,
        'validation': {'direct_monthly_climatology': metrics(clim_val), 'direct_monthly_ridge': metrics(ridge_val)},
        'heldout': {
            'direct_monthly_climatology': {'metrics': metrics(clim_test), 'fitted': clim_fit, 'predictions': clim_test.to_dict(orient='records')},
            'direct_monthly_ridge': {'metrics': metrics(ridge_test), 'fitted': ridge_fit, 'predictions': ridge_test.to_dict(orient='records')},
        },
        'limitations': ['Only five untouched test winters (ten monthly targets): model differences are descriptive, not strong evidence of superiority.',
                        'October reporting-date inputs are matched to the neural hindcast; archived publication timestamps are unavailable.',
                        'This benchmark tests direct monthly versus sum-of-daily target prediction at the same issue date, but networks may exploit more rolling within-winter training windows.',
                        'The neural model uses a longer daily sequence while Ridge uses five October aggregates. This is an operational model comparison, not an isolated causal test of daily versus monthly target resolution.',
                        'No ENSO input was added to this direct-monthly comparator.'],
    }
    assert test.index.min() > refit.index.max()
    assert len(clim_test) == len(ridge_test) == 10
    assert np.isfinite(ridge_test.predicted_inches).all()
    seasons.to_csv(ROOT / 'outputs/deep_dillon_direct_monthly_inputs.csv', float_format='%.8f')
    path = ROOT / 'outputs/deep_dillon_direct_monthly_baseline.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'selected_alpha': selected_alpha, 'validation_selected_family': selected_family,
                      'heldout': {name: data['metrics'] for name, data in output['heldout'].items()}}, indent=2))


if __name__ == '__main__':
    main()
