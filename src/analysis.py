"""Seasonal snowfall demo. Run from any directory; all inputs are snapshotted.

Forecast issue: November 1. Target: November-April snowfall in inches.
JAS index is lagged; revised historical ENSO makes this a pseudo-real-time study.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
SEED = 20261103
SPECS = {
    'climatology': [],
    'october_linear': ['oct'],
    'enso_linear': ['oni_jas'],
    'october_enso_ridge': ['oct', 'oni_jas'],
}


def add_enso(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for name in ['oni', 'roni']:
        table = pd.read_csv(DATA / f'{name}_noaa.csv').set_index('year')
        if not table.index.is_unique:
            raise ValueError('Duplicate ENSO years')
        frame[f'{name}_jas'] = frame.start_year.map(table.JAS)
        frame[f'{name}_djf'] = (frame.start_year + 1).map(table.DJF)
    return frame


def load_copper() -> pd.DataFrame:
    frame = pd.read_csv(DATA / 'copper_monthly_user.csv')
    months = ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr']
    if frame.season.duplicated().any() or frame[months].isna().any().any():
        raise ValueError('Duplicate seasons or missing primary months')
    if (frame[months] < 0).any().any():
        raise ValueError('Negative snowfall')
    frame['nov_apr'] = frame[months[1:]].sum(axis=1)
    frame['oct_apr'] = frame[months].sum(axis=1)
    # This sums recorded cells only; blank May is never interpreted as known zero.
    frame['observed_month_sum'] = frame[months + ['may']].sum(axis=1)
    frame['reported_minus_observed'] = frame.reported_total - frame.observed_month_sum
    frame['primary_included'] = frame.start_year != 2025
    frame['exclusion_reason'] = np.where(frame.primary_included, '', '60-inch unresolved total discrepancy')
    return add_enso(frame)


def load_station(station_id: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / f'ghcn_{station_id}_snow_seasonal.csv')
    frame = frame.loc[frame.complete_oct_nov_apr == 1].copy()
    frame = frame.rename(columns={'season_start': 'start_year',
                                  'october_snowfall_inches': 'oct',
                                  'nov_apr_snowfall_inches': 'nov_apr'})
    if frame[['oct', 'nov_apr']].isna().any().any():
        raise ValueError('Incomplete season marked complete')
    frame['oct_apr'] = frame.oct + frame.nov_apr
    return add_enso(frame.sort_values('start_year').reset_index(drop=True))


def association(x, y, permutations=19999) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    result = stats.pearsonr(x, y)
    ci = result.confidence_interval()
    spearman = stats.spearmanr(x, y)
    rng = np.random.default_rng(SEED)
    # Fixed seed, two-sided label permutations; assumes exchangeable seasons.
    xc, yc = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    perm = np.array([np.dot(xc, rng.permutation(yc)) / denominator for _ in range(permutations)])
    loo = [float(stats.pearsonr(np.delete(x, i), np.delete(y, i)).statistic) for i in range(len(x))]
    return {'n': len(x), 'pearson_r': float(result.statistic),
            'r_squared': float(result.statistic ** 2),
            'pearson_p_two_sided': float(result.pvalue),
            'pearson_ci95': [float(ci.low), float(ci.high)],
            'permutation_p_two_sided': float((np.sum(np.abs(perm) >= abs(result.statistic)) + 1) / (permutations + 1)),
            'spearman_rho': float(spearman.statistic), 'spearman_p': float(spearman.pvalue),
            'leave_one_out_r_range': [min(loo), max(loo)]}


def strong_contrast(frame, index='oni_djf', threshold=1.5, drop_boundary=False) -> dict:
    flag = (frame[index] >= threshold).to_numpy()
    if drop_boundary:
        # NOAA raw ONI DJF2010=1.47, published table=1.5.
        flag = flag & (frame.start_year.to_numpy() != 2009)
    y = frame.nov_apr.to_numpy(float)
    a, b = y[flag], y[~flag]
    if len(a) == 0 or len(b) == 0:
        return {'n_strong': len(a), 'n_other': len(b)}
    delta = float(a.mean() - b.mean())
    # Exact for Copper n=20,k=3; Monte Carlo for larger station cohorts.
    count = math.comb(len(y), len(a))
    if count <= 200000:
        permutations = itertools.combinations(range(len(y)), len(a))
        mode = 'exact'
        n_perm = count
    else:
        rng = np.random.default_rng(SEED)
        n_perm = 19999
        permutations = (rng.choice(len(y), len(a), replace=False) for _ in range(n_perm))
        mode = 'Monte Carlo'
    exceed = 0
    for indices in permutations:
        total = y[list(indices)].sum()
        diff = total / len(a) - (y.sum() - total) / len(b)
        exceed += abs(diff) >= abs(delta) - 1e-10
    pvalue = exceed / n_perm if mode == 'exact' else (exceed + 1) / (n_perm + 1)
    interval = None
    if len(a) >= 2 and len(b) >= 2:
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        degrees = (va + vb) ** 2 / (va ** 2 / (len(a)-1) + vb ** 2 / (len(b)-1))
        half = stats.t.ppf(.975, degrees) * np.sqrt(va + vb)
        interval = [float(delta - half), float(delta + half)]
    leave_event = []
    if len(a) > 1:
        for i, row in frame.loc[flag].reset_index(drop=True).iterrows():
            leave_event.append({'omitted_season': row.season, 'difference_inches': float(np.delete(a, i).mean()-b.mean())})
    return {'index': index, 'threshold': threshold, 'n_strong': len(a), 'n_other': len(b),
            'strong_seasons': frame.loc[flag, 'season'].tolist(),
            'strong_mean_inches': float(a.mean()), 'other_mean_inches': float(b.mean()),
            'difference_inches': delta, 'welch_ci95': interval,
            'permutation_p_two_sided': pvalue, 'permutation_mode': mode,
            'leave_one_strong_event_out': leave_event}


def fit_model(train, name, enso_column='oni_jas') -> dict:
    features = [enso_column if v == 'oni_jas' else v for v in SPECS[name]]
    y = train.nov_apr.to_numpy(float)
    if not features:
        intercept, coefficients = float(y.mean()), []
    else:
        X = train[features].to_numpy(float)
        if name.endswith('ridge'):
            # Fixed a priori. Standardization is fitted separately in each training fold.
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)
            scale, regression = estimator.steps[0][1], estimator.steps[1][1]
            coefficients = regression.coef_ / scale.scale_
            intercept = float(regression.intercept_ - np.dot(coefficients, scale.mean_))
        else:
            regression = LinearRegression().fit(X, y)
            coefficients, intercept = regression.coef_, float(regression.intercept_)
    return {'name': name, 'features': features, 'intercept': intercept,
            'coefficients': list(map(float, coefficients)), 'n_train': len(train),
            'train_first_year': int(train.start_year.min()), 'train_last_year': int(train.start_year.max())}


def predict(model, frame):
    prediction = np.full(len(frame), model['intercept'], dtype=float)
    for feature, coefficient in zip(model['features'], model['coefficients']):
        prediction += frame[feature].to_numpy(float) * coefficient
    return prediction


def baseline_interval(train, level=.8):
    y = train.nov_apr.to_numpy(float)
    half = float(stats.t.ppf((1 + level) / 2, len(y)-1) * y.std(ddof=1) * np.sqrt(1 + 1/len(y)))
    return [max(0., float(y.mean() - half)), float(y.mean() + half)]


def backtest(frame, min_train=8, enso_column='oni_jas'):
    frame = frame.sort_values('start_year').reset_index(drop=True)
    records = []
    for stop in range(min_train, len(frame)):
        train, test = frame.iloc[:stop], frame.iloc[stop:stop+1]
        assert train.start_year.max() < test.start_year.min()
        for name in SPECS:
            model = fit_model(train, name, enso_column)
            value = float(predict(model, test)[0])
            lo, hi = baseline_interval(train)
            records.append({'season': test.season.iloc[0], 'start_year': int(test.start_year.iloc[0]),
                            'model': name, 'n_train': len(train),
                            'train_last_year': int(train.start_year.max()),
                            'actual': float(test.nov_apr.iloc[0]), 'prediction': value,
                            'error': value-float(test.nov_apr.iloc[0]),
                            'baseline_pi80_low': lo, 'baseline_pi80_high': hi})
    predictions = pd.DataFrame(records)
    if predictions.empty:
        raise ValueError('Too few complete seasons for requested training window')
    metrics = []
    base = predictions.loc[predictions.model == 'climatology', 'error'].to_numpy()
    for name, group in predictions.groupby('model', sort=False):
        errors = group.error.to_numpy()
        # Paired uncertainty in improvement; exploratory iid season bootstrap.
        improvement = np.abs(base)-np.abs(errors)
        rng = np.random.default_rng(SEED)
        boot = improvement[rng.integers(len(errors), size=(10000, len(errors)))].mean(axis=1)
        metrics.append({'model': name, 'n_test': len(errors), 'mae_inches': float(np.abs(errors).mean()),
                        'rmse_inches': float(np.sqrt(np.mean(errors**2))),
                        'mae_skill_vs_climatology': float(1-np.abs(errors).mean()/np.abs(base).mean()),
                        'mse_skill_vs_climatology': float(1-np.mean(errors**2)/np.mean(base**2)),
                        'mean_mae_improvement_inches': float(improvement.mean()),
                        'mae_improvement_bootstrap_ci95': [float(v) for v in np.quantile(boot, [.025,.975])]})
    base_rows = predictions.loc[predictions.model == 'climatology']
    coverage = float(((base_rows.actual >= base_rows.baseline_pi80_low) &
                      (base_rows.actual <= base_rows.baseline_pi80_high)).mean())
    return predictions, metrics, coverage


def evaluate(frame, label, min_train):
    frame = frame.sort_values('start_year').reset_index(drop=True)
    if frame[['oct', 'nov_apr', 'oni_jas', 'oni_djf', 'roni_jas', 'roni_djf']].isna().any().any():
        raise ValueError(f'Missing required features in {label}')
    preds, metrics, coverage = backtest(frame, min_train)
    preds.to_csv(OUT / f'{label}_backtest_predictions.csv', index=False)
    frame.to_csv(OUT / f'{label}_analysis_data.csv', index=False)
    _, roni_metrics, _ = backtest(frame, min_train, 'roni_jas')
    return {'dataset': label, 'n': len(frame), 'first_season': frame.season.iloc[0], 'last_season': frame.season.iloc[-1],
            'mean_nov_apr_inches': float(frame.nov_apr.mean()),
            'october_vs_future': association(frame.oct, frame.nov_apr),
            'october_vs_oct_apr': association(frame.oct, frame.oct_apr),
            'jas_oni_vs_future': association(frame.oni_jas, frame.nov_apr),
            'djf_oni_vs_future_retrospective': association(frame.oni_djf, frame.nov_apr),
            'strong_oni_djf': strong_contrast(frame),
            'strong_oni_without_rounding_boundary': strong_contrast(frame, drop_boundary=True),
            'strong_roni_djf': strong_contrast(frame, index='roni_djf'),
            'backtest_min_train': min_train, 'backtest_metrics': metrics,
            'roni_backtest_metrics': roni_metrics,
            'climatology_pi80_coverage': coverage,
            'climatology_prediction_interval80': baseline_interval(frame),
            'models': {name: fit_model(frame, name) for name in SPECS}}


def plots(copper, dillon, results):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/copper-snowfall-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#f7f9fc', 'axes.facecolor': '#f7f9fc'})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    ax = axes[0]
    ax.scatter(copper.oct, copper.nov_apr, s=55, c='#176b87')
    b, a = np.polyfit(copper.oct, copper.nov_apr, 1)
    grid = np.linspace(copper.oct.min(), copper.oct.max(), 100)
    ax.plot(grid, a+b*grid, c='#176b87', alpha=.7)
    r = results['copper']['october_vs_future']
    ax.set(title=f"October vs the following winter\nr = {r['pearson_r']:.2f}; 95% CI {r['pearson_ci95'][0]:.2f} to {r['pearson_ci95'][1]:.2f}",
           xlabel='October snowfall (inches)', ylabel='November–April snowfall (inches)')
    ax = axes[1]
    strong = copper.oni_djf >= 1.5
    ax.scatter(copper.oni_djf, copper.nov_apr, s=55, c=np.where(strong, '#c35e2a', '#78909b'))
    for _, row in copper.loc[strong].iterrows():
        ax.annotate(row.season, (row.oni_djf,row.nov_apr), xytext=(4,6), textcoords='offset points', fontsize=9)
    ax.axvline(1.5, ls='--', color='#c35e2a', alpha=.6)
    ax.set(title='Strong El Niño: only three seasons\nWinter index shown retrospectively',
           xlabel='DJF ONI (°C; following calendar year)', ylabel='November–April snowfall (inches)')
    ax = axes[2]
    metrics = results['copper']['backtest_metrics']
    names = ['Historical mean', 'October', 'JAS ENSO', 'October + ENSO']
    vals = [m['rmse_inches'] for m in metrics]
    bars = ax.barh(names[::-1], vals[::-1], color=['#a6b7c2','#a6b7c2','#a6b7c2','#176b87'])
    ax.bar_label(bars, fmt='%.1f', padding=3)
    ax.set(xlim=(0,max(vals)*1.2), xlabel='Forecast RMSE (inches; lower is better)',
           title=f"Forecasts tested on later seasons\n{metrics[0]['n_test']} Copper test seasons")
    fig.suptitle('Copper Mountain seasonal snowfall demo · 20 seasons, 2005/06–2024/25', fontsize=16, weight='bold')
    fig.savefig(OUT / 'copper_findings.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, frame, label in [(axes[0], copper, 'Copper resort'), (axes[1], dillon, 'NOAA Dillon 1 E')]:
        ax.scatter(frame.oct,frame.nov_apr,s=45,color='#176b87')
        ax.set(xlabel='October snowfall (inches)', ylabel='November–April snowfall (inches)',
               title=f'{label} · {len(frame)} complete seasons')
    fig.suptitle('Independent station experiment: different sites, separate models',fontsize=15)
    fig.savefig(OUT / 'station_comparison.png', dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_copper()
    copper = raw.loc[raw.primary_included].copy()
    dillon = load_station('USC00052281')
    climax = load_station('USC00051660')
    result = {'analysis_date': '2026-11-03', 'seed': SEED,
              'target': 'November-April snowfall in inches; season starts in October',
              'issue_date': 'November 1; complete October assumed reported by issuance',
              'status': 'Exploratory pseudo-real-time hindcast; revised historical indices',
              'copper': evaluate(copper, 'copper', 8),
              'dillon': evaluate(dillon, 'dillon', 15),
              'climax': evaluate(climax, 'climax', 8)}
    recent = dillon.loc[dillon.start_year >= 2000]
    result['dillon_recent'] = evaluate(recent, 'dillon_recent', 8)
    result['copper_sensitivity'] = {
        'include_unresolved_monthly_2025': evaluate(raw, 'copper_all_monthly', 8),
        'october_vs_all_21_reported_totals': association(raw.oct, raw.reported_total),
        'october_vs_20_reported_totals': association(copper.oct, copper.reported_total),
        'october_vs_reported_minus_oct_20': association(copper.oct, copper.reported_total-copper.oct),
        'exclude_both_discrepant_rows': evaluate(copper.loc[copper.start_year != 2019], 'copper_without_2019', 8),
        'start_with_10_training_seasons_metrics': backtest(copper,10)[1],
        'start_with_12_training_seasons_metrics': backtest(copper,12)[1],
    }
    # Holm correction for the two named primary exploratory Copper hypotheses.
    ps = [result['copper']['october_vs_future']['permutation_p_two_sided'],
          result['copper']['strong_oni_djf']['permutation_p_two_sided']]
    order = np.argsort(ps)
    adjusted = np.empty(2)
    adjusted[order[0]] = min(1.,2*ps[order[0]])
    adjusted[order[1]] = max(adjusted[order[0]],ps[order[1]])
    result['primary_holm_adjusted_p'] = dict(zip(['october','strong_oni'],map(float,adjusted)))
    raw.to_csv(OUT / 'copper_data_audit.csv', index=False)
    result['input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    artifact = {'format_version': 1, 'target': result['target'], 'issue_date': result['issue_date'],
                'default_model': 'climatology', 'reason': 'No convincing improvement over the historical mean in the Copper hindcast.',
                'warning': 'Exploratory. Not a validated operational forecast. Inputs must be available at issue time.',
                'datasets': {key: {'models': result[key]['models'],
                                   'baseline_prediction_interval80': result[key]['climatology_prediction_interval80']}
                             for key in ['copper','dillon','climax']}}
    (OUT / 'trained_models.json').write_text(json.dumps(artifact,indent=2,allow_nan=False))
    plots(copper,dillon,result)
    for key in ['copper','dillon','climax','dillon_recent']:
        r = result[key]
        print(f"{key}: n={r['n']}, r={r['october_vs_future']['pearson_r']:.3f}, p={r['october_vs_future']['pearson_p_two_sided']:.3f}")
        for m in r['backtest_metrics']:
            print(f"  {m['model']}: MAE={m['mae_inches']:.2f}, RMSE={m['rmse_inches']:.2f}")


if __name__ == '__main__':
    main()
