"""Acquire routine hourly airport observations for a separate liquid target.

These are airport gauge precipitation reports, not mountain fresh snowfall.
Raw observation minutes, trace flags, missingness and METAR text are retained.
Special reports are excluded to avoid summing overlapping partial-hour amounts.
IEM documents a site-dependent precipitation-reset interval, so even contiguous
hourly reports do not establish exact gauge-reset endpoints independently.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/neural48_training/asos'
API = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
STATIONS = {
    'SLC': dict(icao='KSLC', name='Salt Lake City International Airport',
        associated_site='alta', nominal_report_minute=54,
        qualification='Low-elevation regional airport; not Alta-Collins precipitation'),
    'LXV': dict(icao='KLXV', name='Leadville Lake County Airport',
        associated_site='vail', nominal_report_minute=53,
        qualification='High-elevation regional airport south of Vail; not the Vail resort station'),
    'MVL': dict(icao='KMVL', name='Morrisville-Stowe State Airport',
        associated_site='mansfield,jay', nominal_report_minute=54,
        qualification='Town-level regional airport; no independent Jay Peak hourly snowfall target'),
}


def fetch_winter(winter, out=OUT, refresh=False):
    raw = out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f'SLC_LXV_MVL_{winter}_{winter+1}.csv.gz'
    meta = path.with_suffix('.json')
    params = dict(station=','.join(STATIONS),
        data='p01i,tmpf,dwpf,drct,sknt,metar,wxcodes',
        sts=f'{winter}-10-31T00:00:00Z', ets=f'{winter+1}-05-03T00:00:00Z',
        tz='UTC', format='onlycomma', latlon='yes', elev='yes',
        missing='M', trace='0.0001', report_type=3)
    url = API + '?' + urlencode(params)
    if refresh or not path.exists():
        # IEM requests a maximum of one request per second. Each request contains
        # all three stations and an entire winter, with a bounded response size.
        time.sleep(1.2)
        with urlopen(Request(url, headers={'User-Agent': 'Snowline research'}), timeout=90) as r:
            data = r.read(20_000_001)
        if len(data) > 20_000_000:
            raise ValueError('Station response exceeded bounded download size')
        source = pd.read_csv(io.BytesIO(data), dtype=str)
        if not {'station', 'valid', 'p01i', 'metar'}.issubset(source.columns):
            raise ValueError('Unexpected ASOS response schema')
        if set(source.station) - set(STATIONS):
            raise ValueError('Unexpected station in ASOS response')
        path.write_bytes(gzip.compress(data, mtime=0))
        meta.write_text(json.dumps(dict(url=url,
            retrieved_utc=pd.Timestamp.now(tz='UTC').isoformat(),
            bytes_uncompressed=len(data), raw_sha256=hashlib.sha256(data).hexdigest(),
            compressed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            report_type='3: routine hourly only', rows=len(source)), indent=2))
    return path


def prepare(frame):
    d = frame.copy()
    d['observation_time_utc'] = pd.to_datetime(d.valid, utc=True)
    d = d.sort_values(['station', 'observation_time_utc']).reset_index(drop=True)
    d['duplicate_observation_clock'] = d.duplicated(['station', 'observation_time_utc'], keep=False)
    d['report_hour_utc'] = d.observation_time_utc.dt.floor('h')
    d['multiple_routine_reports_in_hour'] = d.duplicated(['station', 'report_hour_utc'], keep=False)
    d['previous_report_time_utc'] = d.groupby('station').observation_time_utc.shift()
    d['hours_since_previous_report'] = (d.observation_time_utc - d.previous_report_time_utc).dt.total_seconds() / 3600
    d['routine_reports_exactly_60_minutes_apart'] = d.hours_since_previous_report.eq(1)
    d['nominal_report_minute'] = d.station.map({k:v['nominal_report_minute'] for k,v in STATIONS.items()})
    d['report_minute_matches_nominal'] = d.observation_time_utc.dt.minute.eq(d.nominal_report_minute)
    d['trace_precipitation'] = d.p01i.eq('0.0001')
    amount = pd.to_numeric(d.p01i, errors='coerce')
    d['raw_gauge_precipitation_inches'] = amount
    # The API's 0.0001 is a trace encoding, not an actual measured quantity.
    # Use its zero lower bound for amount metrics while preserving the flag.
    d['gauge_precipitation_inches'] = amount.mask(d.trace_precipitation, 0.)
    d['precipitation_sensor_unavailable_metar'] = d.metar.fillna('').str.contains(r'\bPNO\b', regex=True)
    valid = amount.ge(0) & ~d.precipitation_sensor_unavailable_metar & ~d.duplicate_observation_clock
    valid &= ~d.multiple_routine_reports_in_hour
    d['valid_gauge_amount'] = valid
    d['gauge_precipitation_inches'] = d.gauge_precipitation_inches.where(valid)
    d['gauge_precipitation_mm'] = d.gauge_precipitation_inches * 25.4
    d['eligible_contiguous_hourly_report'] = valid & d.routine_reports_exactly_60_minutes_apart & d.report_minute_matches_nominal
    d['gauge_reset_endpoints_independently_verified'] = False
    d['temperature_c'] = (pd.to_numeric(d.tmpf, errors='coerce') - 32) * 5 / 9
    d['dewpoint_c'] = (pd.to_numeric(d.dwpf, errors='coerce') - 32) * 5 / 9
    d['wind_speed_mps'] = pd.to_numeric(d.sknt, errors='coerce') * .514444
    d['wind_direction_degrees'] = pd.to_numeric(d.drct, errors='coerce')
    d['has_snow_weather_code'] = d.wxcodes.fillna('').str.contains('SN', regex=False)
    d['season_end_year'] = d.observation_time_utc.dt.year + d.observation_time_utc.dt.month.ge(7).astype(int)
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--winters', nargs='+', type=int, default=[2020, 2021, 2022, 2023])
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--output', type=Path, default=OUT)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames, sources = [], []
    for winter in args.winters:
        path = fetch_winter(winter, args.output, args.refresh)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame['raw_file'] = str(path.relative_to(ROOT))
        frames.append(frame)
        sources.append(json.loads(path.with_suffix('.json').read_text()))
        print(json.dumps(dict(winter=winter, downloaded_rows=len(frame))), flush=True)
    d = prepare(pd.concat(frames, ignore_index=True))
    d.to_csv(args.output / 'routine_hourly_observations.csv', index=False)
    coverage = d.groupby(['station', 'season_end_year']).agg(
        routine_reports=('valid', 'size'), valid_gauge_amounts=('valid_gauge_amount', 'sum'),
        eligible_contiguous_hourly_reports=('eligible_contiguous_hourly_report', 'sum'),
        measurable_wet_reports=('gauge_precipitation_mm', lambda x: int(x.gt(0).sum())),
        trace_reports=('trace_precipitation', 'sum'),
        nominal_minute_reports=('report_minute_matches_nominal', 'sum'),
        duplicate_hour_reports=('multiple_routine_reports_in_hour', 'sum'),
        first_report=('observation_time_utc', 'min'), last_report=('observation_time_utc', 'max'))
    coverage.to_csv(args.output / 'coverage.csv')
    for station, meta in STATIONS.items():
        rows = d[d.station.eq(station)]
        meta.update(latitude=float(rows.lat.iloc[0]), longitude=float(rows.lon.iloc[0]),
                    elevation_m=float(rows.elevation.iloc[0]))
    manifest = dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(), stations=STATIONS,
        sources=sources, rows=len(d),
        documents={
            'provider_variable_definitions': 'https://mesonet.agron.iastate.edu/request/download.phtml',
            'provider_request_schema': 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=',
            'noaa_asos_manual': 'https://www.weather.gov/media/asos/aum-toc.pdf'},
        interpretation=[
            'p01i is liquid precipitation since the prior hourly gauge reset, in inches, at the reported timestamp. Reset timing varies by site.',
            'Only report_type=3 routine reports are requested. No specials or five-minute reports are summed.',
            'Observation minutes are preserved (:54 SLC/MVL and :53 LXV nominal); none are moved to :00.',
            'Contiguous 60-minute reports are an eligibility check, not independent proof of exact gauge-reset boundaries.',
            'IEM states frozen precipitation may or may not be captured, depending on station instruments; it does not provide an authoritative station sensor inventory.',
            'ASOS liquid-gauge errors must be separate from fresh-snow depth errors; no conversion with an assumed snow-to-liquid ratio creates observed snowfall.',
            'Missing and ambiguous duplicate reports remain missing, never zero. Trace encoding 0.0001 is retained and uses a zero lower bound in amount metrics.',
            'METAR SN also occurs within blowing-snow codes; has_snow_weather_code is descriptive and not an unqualified falling-snow target.'
        ])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(coverage.to_string())


if __name__ == '__main__':
    main()
