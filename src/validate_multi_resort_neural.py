"""Read-only audit of saved daily-model inputs, splits, and forecast artifacts.

Rebuild contexts from source dates rather than the fitting window helper. Final
neural checkpoints are restored and their exact marginals and seeded coherent
paths are compared against the published CSV/NPZ artifacts. No models are fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from deep_dillon import calendar_features, lagged_roni
import multi_resort_neural as neural

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/multi-resort'
SEED = 20261104


def weather_from_source(site, plan):
    raw = pd.read_csv(ROOT / 'data/multi_resort_daily/current_inputs' / plan['weather'],
                      keep_default_na=False, low_memory=False).set_index('date')
    raw.index = pd.to_datetime(raw.index)
    frame = pd.DataFrame(index=raw.index)
    for col in ['tmax_c', 'tmin_c', 'precipitation_inches']:
        frame[col] = pd.to_numeric(raw[col], errors='coerce').where(raw[col + '_quality_flag'].eq(''))
    frame['precipitation_inches'] = np.log1p(frame['precipitation_inches'].clip(lower=0))
    return frame


def context_from_dates(weather, origin):
    last = pd.Timestamp(origin) - pd.Timedelta(days=2)
    return weather.reindex(pd.date_range(last - pd.Timedelta(days=59), last)).to_numpy('float32')


def future_from_dates(origin):
    origin = pd.Timestamp(origin)
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=60)
    roni = lagged_roni(origin)
    lead = np.arange(1, 61) / 60
    known = [roni if np.isfinite(roni) else 0., float(roni >= 1.5),
             float(roni <= -1.5), float(not np.isfinite(roni))]
    return np.column_stack([calendar_features(dates), lead, lead ** 2,
                            np.tile(known, (60, 1))]).astype('float32')


def assert_close(actual, expected, name, tolerance=3e-6):
    try:
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance,
                                   equal_nan=True)
    except AssertionError as error:
        raise AssertionError(name + ': ' + str(error)) from error


def audit_split(dest, tag, meta, arrays):
    record_path = dest / f'{tag}_training.json'
    predictions_path = dest / f'{tag}_predictions.csv'
    if not record_path.exists() or not predictions_path.exists():
        return {'status': 'pending'}
    record = json.loads(record_path.read_text())
    rows = pd.read_csv(predictions_path)
    train_years = set(record['training_years'])
    test_years = set(rows.winter.unique())
    assert not train_years & test_years, f'{tag}: held-out winter enters training'
    assert set(record['validation_years']) < train_years
    keys = ['winter', 'origin', 'date', 'lead']
    reference = rows[rows.model == 'Climatology'].set_index(keys).sort_index()
    assert reference.index.is_unique
    for model, group in rows.groupby('model'):
        group = group.set_index(keys).sort_index()
        assert group.index.equals(reference.index), f'{tag}/{model}: different evaluation cohort'
        assert_close(group.actual_inches, reference.actual_inches, f'{tag}/{model}: truth')
        assert np.isfinite(group[['predicted_inches', 'p_active', 'p_powder', 'crps']]).all().all()
        assert (group.predicted_inches >= 0).all()
        assert (group.p_powder >= 0).all() and (group.p_powder <= group.p_active + 1e-6).all()
        assert (group.p_active <= 1 + 1e-6).all()
    checkpoint = torch.load(dest / f'{tag}_neural.pt', map_location='cpu', weights_only=False)
    _, scaler = neural.restore_neural(checkpoint)
    fit = np.flatnonzero(meta.winter.isin(train_years))
    expected = neural.DailyScaler().fit(arrays['x'][fit]).fit_future(arrays['f'][fit])
    for name in ['mean', 'std', 'future_min', 'future_max', 'observed_counts']:
        # Saving a non-contiguous float32 context can change reduction order.
        tolerance = 1e-4 if name in ['mean', 'std'] else 3e-6
        assert_close(getattr(scaler, name), getattr(expected, name), f'{tag}: fit-only {name}', tolerance)
    return {'status': 'passed', 'train_winters': sorted(train_years),
            'test_winters': sorted(int(x) for x in test_years),
            'matched_observed_pairs_per_model': len(reference)}


def audit_site(site):
    dest = OUT / site
    plan = json.loads((dest / 'plan.json').read_text())
    meta = pd.read_csv(dest / 'origins.csv')
    arrays = np.load(dest / 'training_arrays.npz')
    weather = weather_from_source(site, plan)
    target_raw = pd.read_csv(dest / 'target_daily.csv')
    target = pd.Series(target_raw.snowfall_inches.to_numpy(),
                       index=pd.to_datetime(target_raw.date))
    for path, expected in plan['source_hashes'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, f'Source changed: {path}'
    assert arrays['x'].shape == (len(meta), 60, 3)
    assert arrays['f'].shape == (len(meta), 60, 10)
    assert arrays['y'].shape == (len(meta), 60)
    assert np.isnan(arrays['initial'][:, 0]).all()
    assert (arrays['initial'][:, 1:] == 0).all()
    for i, row in meta.iterrows():
        origin = pd.Timestamp(row.origin)
        assert_close(arrays['x'][i], context_from_dates(weather, origin), f'{site} context origin {origin}')
        assert_close(arrays['f'][i], future_from_dates(origin), f'{site} future origin {origin}')
        expected_y = target.reindex(pd.date_range(origin + pd.Timedelta(days=1), periods=60)).to_numpy('float32')
        assert_close(arrays['y'][i], expected_y, f'{site} labels origin {origin}')
    report = {'site': site, 'prepared_inputs': 'passed', 'checked_origins': len(meta),
              'unknown_initial_state_every_origin': True,
              'daily_weather_context_ends_two_days_before_origin': True,
              'source_hashes_unchanged': True,
              'splits': {tag: audit_split(dest, tag, meta, arrays)
                         for tag in ['fold0', 'fold1', 'fold2', 'chronological']}}
    artifacts = ['final_neural.pt', 'forecast_daily.csv', 'forecast_paths.npz', 'forecast_totals.csv', 'forecast_manifest.json']
    if not all((dest / artifact).exists() for artifact in artifacts):
        report['final_forecast'] = {'status': 'pending'}
        return report

    checkpoint = torch.load(dest / 'final_neural.pt', map_location='cpu', weights_only=False)
    model, scaler = neural.restore_neural(checkpoint)
    expected = neural.DailyScaler().fit(arrays['x']).fit_future(arrays['f'])
    for name in ['mean', 'std', 'future_min', 'future_max', 'observed_counts']:
        tolerance = 1e-4 if name in ['mean', 'std'] else 3e-6
        assert_close(getattr(scaler, name), getattr(expected, name), f'final: fit-only {name}', tolerance)
    assert scaler.context_days == 60 and scaler.raw_features == 3 and model.channels == 7
    manifest = json.loads((dest / 'forecast_manifest.json').read_text())
    origin = pd.Timestamp(manifest['issue_date'])
    x = context_from_dates(weather, origin)[None]
    f = future_from_dates(origin)[None]
    initial = np.array([[np.nan, 0, 0, 0]], dtype='float32')
    prediction = neural.forecast_neural(model, scaler, x, f, initial, return_components=True)
    powder = neural.exceedance_probability(prediction, 5.)
    paths = neural.simulate_neural(model, scaler, x, f, initial, n_paths=2048, seed=SEED)[0, :, :57]
    saved_paths = np.load(dest / 'forecast_paths.npz')['Neural events']
    assert_close(paths, saved_paths, 'checkpoint-restored paths')
    frame = pd.read_csv(dest / 'forecast_daily.csv')
    rows = frame[frame.model == 'Neural events'].sort_values('date')
    dates = pd.date_range('2026-11-05', '2026-12-31')
    assert pd.DatetimeIndex(rows.date).equals(dates)
    assert (rows.lead.to_numpy() == np.arange(1, 58)).all()
    assert_close(rows.mean_inches, prediction['mean'][0, :57], 'restored daily means')
    assert_close(rows.p_active, prediction['p_active'][0, :57], 'restored active probabilities')
    assert_close(rows.p_powder, powder[0, :57], 'restored powder probabilities')
    assert_close(rows.cumulative_mean_inches, prediction['mean'][0, :57].cumsum(), 'cumulative expected means')
    low, high = np.quantile(paths, [.1, .9], axis=0)
    assert_close(rows.q10_inches, low, 'daily lower quantiles')
    assert_close(rows.q90_inches, high, 'daily upper quantiles')
    assert np.isfinite(paths).all() and (paths >= 0).all()
    assert (rows.p_powder >= 0).all() and (rows.p_powder <= rows.p_active + 1e-6).all()
    assert_close(neural.exceedance_probability(prediction, 1.), prediction['p_active'], 'threshold boundary')
    totals = pd.read_csv(dest / 'forecast_totals.csv')
    for period, mask in [('Nov5–Dec31', np.ones(57, dtype=bool)),
                         ('Nov5–30', dates.month == 11), ('December', dates.month == 12)]:
        row = totals[(totals.model == 'Neural events') & (totals.period == period)].iloc[0]
        assert_close(row.expected_snowfall_inches, rows.mean_inches.to_numpy()[mask].sum(), f'{period} mean total')
        assert_close(row.expected_powder_days, rows.p_powder.to_numpy()[mask].sum(), f'{period} expected count')
        assert_close([row.q10_inches, row.q90_inches], np.quantile(paths[:, mask].sum(1), [.1, .9]), f'{period} joint total quantiles')
        counts = (paths[:, mask] >= 5).sum(1)
        assert_close([row.powder_count_q10, row.powder_count_q90],
                     np.quantile(counts, [.1, .9], method='inverted_cdf'), f'{period} joint count quantiles')
    # Monte Carlo error is diagnostic only; expected means are exact marginals,
    # while 2,048 joint samples supply predictive quantiles, not mean forecasts.
    total_samples = paths.sum(1)
    exact_total = float(rows.mean_inches.sum())
    mc_se = float(total_samples.std(ddof=1) / np.sqrt(len(total_samples)))
    report['final_forecast'] = dict(status='passed', checkpoint_reproduces_marginals=True,
        checkpoint_reproduces_seeded_joint_paths=True, forecast_days=57, joint_paths=2048,
        exact_expected_snowfall=exact_total, sampled_total_mean=float(total_samples.mean()),
        sampled_total_mean_standard_error=mc_se,
        mean_discrepancy_in_mc_standard_errors=float((total_samples.mean() - exact_total) / mc_se),
        daily_and_monthly_quantiles_verified=True, threshold_probabilities_verified=True,
        missing_weather_fraction=float((~np.isfinite(x)).mean()),
        transformed_current_roni=float(scaler.transform_future(f)[0, 0, 6]))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', choices=['alta', 'mansfield', 'jay', 'all'], default='all')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = {site: audit_site(site) for site in ['alta', 'mansfield', 'jay']
                  if args.site == 'all' or site == args.site}
    dest = OUT / 'validation'; dest.mkdir(parents=True, exist_ok=True)
    path = dest / f'neural_audit_{args.site}.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({'audit': str(path), 'sites': {site: data['final_forecast']['status']
                                                  for site, data in result.items()}}))
