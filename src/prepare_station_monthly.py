#!/usr/bin/env python3
"""Prepare strict monthly NOAA snowfall observations without filling missing days.

Run offline against the saved public GHCN snapshots. Each season is a row in the
wide file, with October--April columns, and each month is a row in the long file.
Incomplete or not-yet-ended months have a blank target, never an assumed zero.
Neighboring sites are distinct targets; any shared-model validation must hold
the entire calendar snow season out across all sites together.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

from fetch_ghcn import STATION_IDS, parse_metadata, parse_snow

MONTHS = [('oct', 10), ('nov', 11), ('dec', 12), ('jan', 1),
          ('feb', 2), ('mar', 3), ('apr', 4)]


def monthly_rows(records, station, first_season, last_season, as_of):
    long_rows, wide_rows = [], []
    for season_start in range(first_season, last_season + 1):
        season = f'{season_start}/{season_start + 1}'
        wide = {'station_id': station['id'], 'station_name': station['name'],
                'start_year': season_start, 'season': season}
        for label, month in MONTHS:
            year = season_start if month >= 10 else season_start + 1
            expected = calendar.monthrange(year, month)[1]
            month_end = date(year, month, expected)
            dates = [date(year, month, day) for day in range(1, expected + 1)]
            valid = [records[day] for day in dates
                     if day <= as_of and day in records and records[day]['usable']]
            observed = sum(value['snowfall_mm'] for value in valid) / 25.4
            ended = month_end <= as_of
            complete = ended and len(valid) == expected
            total = round(observed, 6) if complete else ''
            wide[label] = total
            wide[f'{label}_usable_days'] = len(valid)
            wide[f'{label}_expected_days'] = expected
            long_rows.append({'station_id': station['id'], 'station_name': station['name'],
                              'start_year': season_start, 'season': season,
                              'year': year, 'month': month, 'month_label': label,
                              'snowfall_inches': total,
                              'observed_partial_sum_inches': round(observed, 6),
                              'usable_days': len(valid), 'expected_days': expected,
                              'month_ended_by_as_of': int(ended),
                              'complete': int(complete), 'as_of': as_of.isoformat()})
        wide['complete_oct_apr'] = int(all(wide[label] != '' for label, _ in MONTHS))
        wide_rows.append(wide)
    return long_rows, wide_rows


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--first-season', type=int, default=1950)
    parser.add_argument('--last-season', type=int, default=2026)
    parser.add_argument('--as-of', type=date.fromisoformat, default=date(2026, 11, 3))
    args = parser.parse_args()
    if args.first_season > args.last_season:
        parser.error('first-season must not exceed last-season')
    root = args.data_dir
    stations = parse_metadata((root / 'ghcnd-stations.txt').read_text())
    long_rows, wide_rows, hashes = [], [], {}
    for sid in STATION_IDS:
        path = root / f'ghcn_{sid}.dly'
        records = parse_snow(path.read_text(), sid)
        long, wide = monthly_rows(records, stations[sid], args.first_season,
                                  args.last_season, args.as_of)
        long_rows.extend(long)
        wide_rows.extend(wide)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_csv(root / 'ghcn_monthly_long.csv', long_rows)
    write_csv(root / 'ghcn_monthly_wide.csv', wide_rows)
    provenance = {'as_of': args.as_of.isoformat(), 'input_sha256': hashes,
                  'sources': {sid: f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{sid}.dly' for sid in STATION_IDS},
                  'policy': 'Accept only nonnegative SNOW, blank quality flag, measurement flag not P. Retain recorded trace as zero. Require every calendar day and month ended by as_of for a monthly total.',
                  'missing': 'Blank snowfall_inches/season-month cells mean incomplete or not yet ended; partial observed sums are diagnostic only.',
                  'units': 'Fresh snowfall inches = GHCN SNOW millimetres / 25.4.',
                  'grouping': 'Separate sites, sharing regional weather. Hold out complete seasons across sites when pooling; do not treat same-season stations/months as independent validation examples.',
                  'current_october_is_not_copper': 'October2026 station observations are independent site measurements, not the resort Cache Site total.',
                  'complete_seasons_through_2025': {sid: sum(row['station_id'] == sid and row['start_year'] <= 2025 and row['complete_oct_apr'] for row in wide_rows) for sid in STATION_IDS}}
    (root / 'ghcn_monthly_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'Prepared {len(long_rows)} site-month rows and {len(wide_rows)} site-season rows.')
    print(json.dumps(provenance['complete_seasons_through_2025']))
    for row in wide_rows:
        if row['start_year'] == 2026:
            print(row['station_name'], 'October2026:', repr(row['oct']), 'inches,', row['oct_usable_days'], '/31 usable days')


if __name__ == '__main__':
    main()
