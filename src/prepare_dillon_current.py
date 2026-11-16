"""Create an isolated, auditable current Dillon input without changing history files."""
from __future__ import annotations

import argparse
from datetime import datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from prepare_dillon_daily import (
    ROOT, STATION, ELEMENTS, parse_records, build_features, multiday_audit,
    validate_features,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', default='2026-11-03')
    parser.add_argument('--start', default='2005-10-01')
    parser.add_argument('--source', default='data/dillon_latest_20261103.dly')
    parser.add_argument('--headers', default='data/dillon_latest_20261103.headers.txt')
    args = parser.parse_args()
    source = ROOT / args.source
    header_file = ROOT / args.headers
    headers = {}
    if header_file.exists():
        for line in header_file.read_text().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.lower()] = value.strip()
    asof = pd.Timestamp(args.as_of)
    # Conservatively exclude the current station reporting day and later.
    permitted_end = asof - pd.Timedelta(days=1)
    parsed = parse_records(source)
    raw = parsed.loc[:permitted_end].copy()
    columns = [cleaned for _, cleaned, _, _ in ELEMENTS.values()]
    last_any = raw.index[raw[columns].notna().any(axis=1)].max()
    last_complete = raw.index[raw[columns].notna().all(axis=1)].max()
    end = last_any.date().isoformat()
    frame = build_features(raw, args.start, end)
    checks = validate_features(raw, frame, args.start, end)
    assert frame.index.max() <= permitted_end
    assert not frame.index.duplicated().any()
    old_source = ROOT / f'data/ghcn_{STATION}.dly'
    prior = parse_records(old_source)
    common = prior.index.intersection(raw.index)
    revisions = {}
    for column in columns:
        left, right = prior.loc[common, column], raw.loc[common, column]
        changed = ~(left.eq(right) | (left.isna() & right.isna()))
        previously_observed = left.notna()
        revisions[column] = {
            'changed_previously_observed_values': int((changed & previously_observed).sum()),
            'newly_available_values_on_preexisting_calendar_rows': int((left.isna() & right.notna()).sum()),
        }
    output = ROOT / 'outputs/dillon_deep_daily_input.csv'
    frame.to_csv(output, float_format='%.8f')
    utc_now = datetime.now(ZoneInfo('UTC'))
    local = ZoneInfo('America/New_York')
    retrieved = parsedate_to_datetime(headers['date']) if 'date' in headers else utc_now
    modified = parsedate_to_datetime(headers['last-modified']) if 'last-modified' in headers else None
    daily_recent = []
    for dt, row in raw.reindex(pd.date_range('2026-10-25', permitted_end)).iterrows():
        daily_recent.append({'date': dt.date().isoformat(), **{
            c: float(row[c]) if pd.notna(row[c]) else None for c in columns}})
    report = {
        'as_of_local_date': args.as_of,
        'timezone': 'America/New_York',
        'retrieved_http_date_utc': retrieved.isoformat(),
        'retrieved_http_date_local': retrieved.astimezone(local).isoformat(),
        'source_last_modified_utc': modified.isoformat() if modified else None,
        'source_last_modified_local': modified.astimezone(local).isoformat() if modified else None,
        'station_id': STATION,
        'station_name': 'DILLON 1 E',
        'source_url': f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{STATION}.dly',
        'format_url': 'https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt',
        'raw_file': str(source.relative_to(ROOT)),
        'raw_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'raw_bytes': source.stat().st_size,
        'http_headers_file': str(header_file.relative_to(ROOT)),
        'previous_raw_file_preserved': str(old_source.relative_to(ROOT)),
        'previous_raw_sha256': hashlib.sha256(old_source.read_bytes()).hexdigest(),
        'permitted_latest_reporting_date': permitted_end.date().isoformat(),
        'latest_any_usable_date': end,
        'latest_all_five_usable_date': last_complete.date().isoformat(),
        'latest_usable_by_element': {e: raw[c].last_valid_index().date().isoformat()
                                   for e, (_, c, _, _) in ELEMENTS.items()},
        'feature_table': str(output.relative_to(ROOT)),
        'feature_table_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'table_start': args.start,
        'table_end': end,
        'table_rows': len(frame),
        'table_columns_excluding_date': len(frame.columns),
        'usable_days': {e: int(frame[c].notna().sum()) for e, (_, c, _, _) in ELEMENTS.items()},
        'new_days_after_previous_clean_cutoff': int((frame.index > pd.Timestamp('2026-10-31')).sum()),
        'previous_raw_comparison': revisions,
        'raw_flags_preserved': True,
        'quality_policy': 'Uses prepare_dillon_daily.parse_records unchanged. Blank QFLAG, MFLAG not P, raw not -9999; nonnegative precipitation/snow/depth. PRCP A/C/E/I excluded. Trace retained at reported value. Missing values remain missing.',
        'multiday_audit': multiday_audit(source, args.start, end, frame),
        'validation_checks': checks + ['No reporting dates on or after the client as-of date enter the table.'],
        'recent_observations': daily_recent,
        'forecast_contract': {
            'latest_input_day': end,
            'first_unobserved_day': (last_any + pd.Timedelta(days=1)).date().isoformat(),
            'display_future_start': (asof + pd.Timedelta(days=1)).date().isoformat(),
            'display_future_end': '2026-12-31',
            'unobserved_days_to_bridge_before_display': [dt.date().isoformat() for dt in pd.date_range(last_any + pd.Timedelta(days=1), asof)],
            'november_total': 'Observed Nov1 plus simulated Nov2–30. Nov2–3 are unobserved bridge days, not verified observations.',
            'availability_caveat': 'The .dly file has a snapshot-level Last-Modified time but no per-observation publication timestamp. The retrieved file was available late on Nov3 NewYork time. Historical reporting dates alone do not establish real-time availability for operational hindcasts.',
        },
    }
    status = ROOT / 'data/dillon_current_observation_status.json'
    status.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['table_rows', 'table_end', 'latest_all_five_usable_date', 'usable_days', 'raw_sha256', 'feature_table']}, indent=2))


if __name__ == '__main__':
    main()
