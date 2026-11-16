"""Build strict daily observations and causal forecasting features for Dillon.

Rows are indexed by target day t. All columns named lag/prev use dates before t.
Same-day weather observations are retained for analysis but are NOT available
early-day predictors. Forward target windows exist only when every day is usable.
The calendar day is the station reporting day, not a guaranteed UTC interval or
publication timestamp; operational forecasting must also audit reporting latency.
"""
from __future__ import annotations

import argparse
import calendar
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATION = 'USC00052281'
ELEMENTS = {
    'SNOW': ('snowfall', 'snowfall_inches', 25.4, 'mm'),
    'SNWD': ('snow_depth', 'snow_depth_cm', 10.0, 'mm'),
    'PRCP': ('precipitation', 'precipitation_mm', 10.0, 'tenths_mm'),
    'TMAX': ('tmax', 'tmax_celsius', 10.0, 'tenths_celsius'),
    'TMIN': ('tmin', 'tmin_celsius', 10.0, 'tenths_celsius'),
}


def parse_records(path):
    records = {}
    for line in path.read_text().splitlines():
        if len(line) < 269 or line[:11] != STATION:
            raise ValueError('Malformed record or unexpected station')
        element = line[17:21]
        if element not in ELEMENTS:
            continue
        year, month = int(line[11:15]), int(line[15:17])
        prefix, cleaned, divisor, unit = ELEMENTS[element]
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            observed = date(year, month, day)
            fields = records.setdefault(observed, {})
            if f'{prefix}_record_present' in fields:
                raise ValueError(f'Duplicate {element} record for {observed}')
            block = line[21 + 8 * (day-1):29 + 8 * (day-1)]
            raw = int(block[:5])
            mflag, qflag, sflag = [v.strip() for v in block[5:8]]
            usable = (raw != -9999 and not qflag and mflag != 'P' and
                      (element in {'TMAX','TMIN'} or raw >= 0))
            if element == 'PRCP' and mflag in {'A','C','E','I'}:
                # Explicit partial-day or absent-report precipitation flags are
                # not treated as complete daily precipitation totals.
                usable = False
            fields.update({f'{prefix}_record_present': 1,
                           f'{prefix}_raw_{unit}': raw,
                           f'{prefix}_measurement_flag': mflag,
                           f'{prefix}_quality_flag': qflag,
                           f'{prefix}_source_flag': sflag,
                           f'{prefix}_usable': int(usable),
                           cleaned: raw / divisor if usable else np.nan})
    frame = pd.DataFrame.from_dict(records, orient='index').sort_index()
    frame.index = pd.DatetimeIndex(frame.index, name='date')
    return frame


def longest_missing_runs(series):
    missing = series.isna()
    starts = (missing != missing.shift(fill_value=False)).cumsum()
    runs = []
    for _, positions in missing.loc[missing].groupby(starts):
        runs.append({'start': positions.index.min().date().isoformat(),
                     'end': positions.index.max().date().isoformat(),
                     'days': len(positions)})
    return sorted(runs, key=lambda x: (-x['days'], x['start']))[:10]


def build_features(raw, start, end):
    # Include a prior-week buffer for the first target day's legitimate lags.
    calendar_index = pd.date_range(pd.Timestamp(start)-pd.Timedelta(days=7), end, name='date')
    frame = raw.reindex(calendar_index).copy()
    for prefix, cleaned, divisor, unit in ELEMENTS.values():
        for suffix in ['record_present','usable']:
            frame[f'{prefix}_{suffix}'] = frame[f'{prefix}_{suffix}'].fillna(0).astype(int)
        for suffix in ['measurement_flag','quality_flag','source_flag']:
            frame[f'{prefix}_{suffix}'] = frame[f'{prefix}_{suffix}'].fillna('')
    snow = frame.snowfall_inches
    for days in [1,3,7]:
        lagged = snow.shift(1)
        feature_name = 'snow_lag1_inches' if days == 1 else f'snow_prev{days}_inches'
        frame[feature_name] = lagged.rolling(days, min_periods=days).sum()
        frame[f'snow_prev{days}_usable_days'] = lagged.notna().astype(int).rolling(days, min_periods=days).sum()
        # Targets cover [t, t+days-1]. Do not replace unknown future days by 0.
        frame[f'target_snow_next{days}d_inches'] = snow.rolling(days, min_periods=days).sum().shift(-(days-1))
    for column in ['tmax_celsius','tmin_celsius','precipitation_mm','snow_depth_cm']:
        frame[f'lag1_{column}'] = frame[column].shift(1)
    frame['day_of_year'] = frame.index.dayofyear
    frame['calendar_sin'] = np.sin(2*np.pi*(frame.index.dayofyear-1)/365.2425)
    frame['calendar_cos'] = np.cos(2*np.pi*(frame.index.dayofyear-1)/365.2425)
    frame['station_id'] = STATION
    return frame.loc[start:end].copy()


def count_period(raw, start, end):
    expected = pd.date_range(start,end)
    period = raw.reindex(expected)
    return {'start':start,'end':end,'calendar_days':len(expected),
            'usable_days':{element:int(period[cleaned].notna().sum())
                           for element,(_,cleaned,_,_) in ELEMENTS.items()}}


def multiday_audit(source, start, end, frame):
    counts = {key:0 for key in ['MDSF','DASF','MDPR','DAPR']}
    for line in source.read_text().splitlines():
        element = line[17:21]
        if element not in counts:
            continue
        year,month = int(line[11:15]),int(line[15:17])
        for day in range(1,calendar.monthrange(year,month)[1]+1):
            observed = date(year,month,day).isoformat()
            raw_value = int(line[21+8*(day-1):26+8*(day-1)])
            if start <= observed <= end and raw_value != -9999:
                counts[element] += 1
    return {'nonmissing_records_in_export_period':counts,
            'handling':'Multiday elements are not parsed into single-day measurements. No multiday total is assigned to an individual day.',
            'mflag_A_definition':'The saved NOAA readme defines A as precipitation total from one report of 6-hour precipitation, not a multiday accumulation.',
            'mflag_A_values_in_export_period':sum(int((frame[f'{prefix}_measurement_flag']=='A').sum())
                                                 for prefix,_,_,_ in ELEMENTS.values())}


def validate_features(raw, frame, start, end):
    checks=[]
    sample_dates=[pd.Timestamp(start)+pd.Timedelta(days=10),pd.Timestamp('2005-11-24'),pd.Timestamp(end)]
    for dt in sample_dates:
        if dt not in frame.index:
            continue
        for count in [1,3,7]:
            dates=pd.date_range(dt-pd.Timedelta(days=count),dt-pd.Timedelta(days=1))
            observations=raw.reindex(dates).snowfall_inches
            expected=float(observations.sum()) if observations.notna().all() else np.nan
            key='snow_lag1_inches' if count==1 else f'snow_prev{count}_inches'
            actual=frame.loc[dt,key]
            assert (pd.isna(expected) and pd.isna(actual)) or np.isclose(actual,expected)
    checks.append('Lag windows independently checked across a normal day, a missing-observation boundary and the last day.')
    assert frame.iloc[-1].target_snow_next3d_inches != frame.iloc[-1].target_snow_next3d_inches
    assert frame.iloc[-1].target_snow_next7d_inches != frame.iloc[-1].target_snow_next7d_inches
    checks.append('Unavailable future-window labels remain missing at the snapshot boundary.')
    mutated=raw.copy()
    future=pd.Timestamp(end)
    mutated.loc[future,'snowfall_inches']=999.
    changed=build_features(mutated,start,end)
    feature_cols=[c for c in frame if c.startswith(('snow_lag','snow_prev','lag1_'))]
    np.testing.assert_allclose(frame[feature_cols].to_numpy(),changed[feature_cols].to_numpy(),equal_nan=True)
    checks.append('Changing the final target-day snowfall cannot alter any earlier or same-day lag feature.')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2005-10-01')
    parser.add_argument('--end', default='2026-10-31')
    args = parser.parse_args()
    source = ROOT/f'data/ghcn_{STATION}.dly'
    raw = parse_records(source)
    feature_frame = build_features(raw,args.start,args.end)
    validation_checks=validate_features(raw,feature_frame,args.start,args.end)
    output = ROOT/'outputs/dillon_daily_features.csv'
    feature_frame.to_csv(output, float_format='%.8f')
    all_elements = {}
    for element,(prefix,cleaned,_,_) in ELEMENTS.items():
        good = raw.loc[raw[cleaned].notna()]
        all_elements[element] = {'first_usable':good.index.min().date().isoformat(),
                                 'last_usable':good.index.max().date().isoformat(),
                                 'usable_days':len(good),
                                 'nonblank_quality_flags':int((raw[f'{prefix}_quality_flag'].fillna('')!='').sum()),
                                 'presumed_zero_flags':int((raw[f'{prefix}_measurement_flag']=='P').sum())}
    winter = feature_frame.loc[feature_frame.index.month.isin([10,11,12,1,2,3,4])]
    monthly = []
    for period,g in winter.groupby(winter.index.to_period('M')):
        expected = period.days_in_month
        usable = int(g.snowfall_inches.notna().sum())
        monthly.append({'month':str(period),'expected_days':expected,'usable_days':usable,
                        'complete':len(g)==expected and usable==expected})
    seasons = []
    for year in range(pd.Timestamp(args.start).year,pd.Timestamp(args.end).year):
        dates = pd.date_range(f'{year}-10-01',f'{year+1}-04-30')
        g = feature_frame.reindex(dates)
        count = int(g.snowfall_inches.notna().sum())
        seasons.append({'season':f'{year}/{year+1}','expected_days':len(dates),'usable_days':count,
                        'complete':count==len(dates)})
    feature_cols = [c for c in feature_frame if c.startswith(('snow_lag','snow_prev','lag1_','calendar_'))]
    report = {
        'as_of':'2026-11-03','station_id':STATION,'station_name':'DILLON 1 E',
        'latitude':39.6244,'longitude':-106.0336,'elevation_m':2767.9,
        'source_url':f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{STATION}.dly',
        'format_url':'https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'whole_record':all_elements,
        'periods':[count_period(raw,'1950-01-01',args.end),count_period(raw,args.start,args.end),
                   count_period(raw,'2010-01-01',args.end)],
        'quality_policy':'Blank QFLAG, MFLAG not P, raw not -9999; precipitation/snow/depth nonnegative. PRCP MFLAG A/C/E (partial-day reports) and I (no precipitation report) excluded. Temperatures may be negative. Trace is retained at its reported value. No missing value is imputed.',
        'multiday_audit':multiday_audit(source,args.start,args.end,feature_frame),
        'measurement_flags_in_export':{element:feature_frame[f'{prefix}_measurement_flag'].value_counts().to_dict()
                                       for element,(prefix,_,_,_) in ELEMENTS.items()},
        'validation_checks':validation_checks,
        'units':{'snowfall_inches':'SNOW raw millimetres /25.4',
                 'snow_depth_cm':'SNWD raw millimetres /10',
                 'precipitation_mm':'PRCP raw tenths millimetres /10',
                 'tmax_celsius':'TMAX raw tenths Celsius /10','tmin_celsius':'TMIN raw tenths Celsius /10'},
        'feature_table':{'file':str(output.relative_to(ROOT)),'rows':len(feature_frame),
                         'start':args.start,'end':args.end,'features':feature_cols,
                         'usable_feature_rows':{name:int(feature_frame[name].notna().sum()) for name in feature_cols},
                         'target_rows':{f'target_snow_next{k}d_inches':int(feature_frame[f'target_snow_next{k}d_inches'].notna().sum()) for k in [1,3,7]},
                         'time_contract':'Row t is a forecast target day. lag1/prev features use only dates t-1 and earlier. Forward target windows include t through t+k-1 only when all those observations exist and pass quality checks. Same-day weather columns are retained as observations and must not be passed as early-day forecast covariates.',
                         'publication_latency_caveat':'Station reporting day and record date do not guarantee publication before a forecast cutoff. Archived issue timestamps are needed for an operational hindcast.'},
        'winter_oct_apr':{'calendar_days':len(winter),'usable_snowfall_days':int(winter.snowfall_inches.notna().sum()),
                          'complete_months':sum(m['complete'] for m in monthly),'total_months':len(monthly),
                          'complete_seasons':sum(s['complete'] for s in seasons),'total_finished_seasons':len(seasons),
                          'month_completeness':monthly,'season_completeness':seasons},
        'largest_snowfall_gaps_in_export':longest_missing_runs(feature_frame.snowfall_inches),
        'site_and_training_limitations':[
            'Dillon is a separate lower-elevation station, not an A-Basin or Keystone snowfall label.',
            'Thousands of daily observations add short-horizon examples but do not create thousands of independent seasonal climate years.',
            'Nearby sites share storms. Daily validation should hold out whole seasons or contiguous storm/time blocks; lag/rolling-window overlap must be purged at train/test boundaries.',
            'Daily station values do not establish future resort forecast skill. Target-mountain snowfall observations and archived issue-time weather forecasts are needed for a validated mapping.',
            'Historical station moves and observation practices have not been homogenized in this export.'
        ]
    }
    (ROOT/'outputs/dillon_daily_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'export_rows':len(feature_frame),'winter_days':report['winter_oct_apr']['calendar_days'],
                      'winter_usable_snow_days':report['winter_oct_apr']['usable_snowfall_days'],
                      'complete_winter_months':report['winter_oct_apr']['complete_months'],
                      'winter_months':report['winter_oct_apr']['total_months'],
                      'complete_winter_seasons':report['winter_oct_apr']['complete_seasons'],
                      'finished_winter_seasons':len(seasons),
                      'largest_gaps':report['largest_snowfall_gaps_in_export'][:3]},indent=2))


if __name__=='__main__':main()
