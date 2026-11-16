"""Aggregate quality-controlled Dillon daily observations into monthly prefixes.

Each season has independent Oct--Apr fields. A future or incomplete month never
invalidates an earlier month. These are observed context features, not forecasts;
a caller must select only months completed by its forecast issue time.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONTHS = [('oct', 10), ('nov', 11), ('dec', 12), ('jan', 1),
          ('feb', 2), ('mar', 3), ('apr', 4)]
FEATURES = ['snow_total_inches', 'snow_days', 'max_daily_snow_inches',
            'mean_tmax_celsius', 'mean_tmin_celsius', 'precipitation_total_mm',
            'freezing_days', 'late7_snow_inches']
OBSERVATIONS = ['snowfall_inches', 'tmax_celsius', 'tmin_celsius',
                'precipitation_mm']


def complete_stat(values, operation):
    """Never let pandas silently turn a partial month into a full-month total."""
    return float(operation(values)) if values.notna().all() else np.nan


def aggregate(daily, start_years):
    rows = []
    for start_year in start_years:
        row = {'start_year': start_year, 'season': f'{start_year}/{start_year+1}',
               'station_id': 'USC00052281'}
        for name, month in MONTHS:
            year = start_year + int(month < 10)
            expected = calendar.monthrange(year, month)[1]
            dates = pd.date_range(f'{year}-{month:02d}-01', periods=expected)
            frame = daily.reindex(dates)
            snow, tmax, tmin, precip = (frame[col] for col in OBSERVATIONS)
            late7 = snow.iloc[-7:]
            values = {
                'snow_total_inches': complete_stat(snow, lambda x: x.sum()),
                'snow_days': complete_stat(snow, lambda x: (x > 0).sum()),
                'max_daily_snow_inches': complete_stat(snow, lambda x: x.max()),
                'mean_tmax_celsius': complete_stat(tmax, lambda x: x.mean()),
                'mean_tmin_celsius': complete_stat(tmin, lambda x: x.mean()),
                'precipitation_total_mm': complete_stat(precip, lambda x: x.sum()),
                'freezing_days': complete_stat(tmax, lambda x: (x <= 0).sum()),
                'late7_snow_inches': complete_stat(late7, lambda x: x.sum()),
            }
            row.update({f'{name}_{key}': value for key, value in values.items()})
            row[f'{name}_expected_days'] = expected
            for label, values_ in [('snow', snow), ('tmax', tmax), ('tmin', tmin),
                                   ('precipitation', precip), ('late7_snow', late7)]:
                row[f'{name}_{label}_valid_days'] = int(values_.notna().sum())
            row[f'{name}_all8_complete'] = int(all(pd.notna(v) for v in values.values()))
        rows.append(row)
    return pd.DataFrame(rows)


def validate(daily, result):
    checks = []
    # Independently compare every complete aggregate to its daily observations.
    for row in result.itertuples():
        for name, month in MONTHS:
            year = row.start_year + int(month < 10)
            n = calendar.monthrange(year, month)[1]
            month_daily = daily.reindex(pd.date_range(f'{year}-{month:02d}-01', periods=n))
            snow = month_daily.snowfall_inches
            actual = getattr(row, f'{name}_snow_total_inches')
            assert pd.notna(actual) == snow.notna().all()
            if pd.notna(actual):
                assert np.isclose(actual, snow.to_numpy().sum())
                assert getattr(row, f'{name}_snow_days') == sum(snow.to_numpy() > 0)
            assert getattr(row, f'{name}_snow_valid_days') == snow.notna().sum()
    checks.append('Every full-month snowfall total and measurable-snow day count independently reconciles to daily observations; incomplete-month totals are missing.')
    # Feature construction must be invariant to all observations after the origin.
    test_year = int(result.loc[result.oct_all8_complete == 1, 'start_year'].min())
    changed = daily.copy()
    after_october = changed.index > pd.Timestamp(test_year, 10, 31)
    assert after_october.any(), 'Invariance check needs actual later observations'
    changed.loc[after_october, OBSERVATIONS] = np.nan
    other = aggregate(changed, [test_year])
    october = [c for c in result if c.startswith('oct_')]
    np.testing.assert_allclose(result.loc[result.start_year == test_year, october],
                               other[october], equal_nan=True)
    assert other.nov_all8_complete.iloc[0] == 0
    assert other.nov_snow_total_inches.isna().all()
    checks.append('October features are invariant to changing later observations; later missing months do not remove a season.')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-season', type=int, default=2005)
    parser.add_argument('--last-season', type=int, default=2026)
    args = parser.parse_args()
    source = ROOT / 'outputs/dillon_daily_features.csv'
    daily = pd.read_csv(source, parse_dates=['date']).set_index('date')
    if not daily.index.is_unique:
        raise ValueError('Daily source has duplicate reporting dates')
    if not set(OBSERVATIONS).issubset(daily.columns):
        raise ValueError('Daily source is missing required cleaned observations')
    result = aggregate(daily, range(args.first_season, args.last_season + 1))
    checks = validate(daily, result)
    destination = ROOT / 'data/dillon_season_daily_features.csv'
    result.to_csv(destination, index=False, float_format='%.8f')
    audit_source = ROOT / 'outputs/dillon_daily_audit.json'
    audit = json.loads(audit_source.read_text())
    cohort = result.loc[result.start_year.between(2005, 2024)].copy()
    counts = []
    for i, (name, _) in enumerate(MONTHS):
        prefix = [m for m, _ in MONTHS[:i+1]]
        columns = [f'{m}_{f}' for m in prefix for f in FEATURES]
        eligible = cohort.loc[cohort[columns].notna().all(axis=1)]
        counts.append({'observed_through': name, 'complete_prefix_seasons': len(eligible),
                       'start_years': eligible.start_year.tolist()})
    metadata = {
        'station_id': 'USC00052281', 'station_name': 'DILLON 1 E',
        'source_file': str(source.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'raw_source_url': audit['source_url'], 'raw_source_sha256': audit['source_sha256'],
        'daily_audit_file': str(audit_source.relative_to(ROOT)),
        'first_reporting_date': str(daily.index.min().date()),
        'last_reporting_date': str(daily.index.max().date()),
        'season_start_years': [args.first_season, args.last_season],
        'rows': len(result), 'feature_suffixes': FEATURES,
        'feature_definitions': {
            'snow_total_inches': 'Sum of measured new snowfall, requiring all calendar dates.',
            'snow_days': 'Count of dates with measurable snowfall > 0; reported traces count as zero.',
            'max_daily_snow_inches': 'Maximum daily new snowfall, requiring all calendar dates.',
            'mean_tmax_celsius': 'Arithmetic mean daily maximum temperature, requiring all calendar dates.',
            'mean_tmin_celsius': 'Arithmetic mean daily minimum temperature, requiring all calendar dates.',
            'precipitation_total_mm': 'Total precipitation liquid-water equivalent, requiring all calendar dates.',
            'freezing_days': 'Count of ice days with daily maximum temperature <= 0 Celsius; requires all dates.',
            'late7_snow_inches': 'Sum on the final seven calendar dates of a month; only these seven dates must be valid.'},
        'quality_policy': audit['quality_policy'],
        'missingness_policy': 'Independent by month and element. All calendar dates required for full-month features; seven final dates required for late7. Unknown or future values are missing, never zero. all8_complete describes only the eight features in that month.',
        'model_use': 'Use only completed observed months at each forecast origin; observed-prefix cohort eligibility must not depend on future feature completeness. Daily records are local reporting dates, not exact issuance timestamps; operational publication latency still needs treatment.',
        'copper_2005_2024_complete_prefix_counts': counts,
        'validation_checks': checks,
    }
    (ROOT / 'data/dillon_season_daily_features_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'file': str(destination.relative_to(ROOT)), 'rows': len(result),
                      'prefix_counts': {v['observed_through']: v['complete_prefix_seasons'] for v in counts},
                      'current_october': result.loc[result.start_year == args.last_season,
                          ['start_year'] + [f'oct_{f}' for f in FEATURES]].to_dict('records')}, indent=2))


if __name__ == '__main__':
    main()
