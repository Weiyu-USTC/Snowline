"""Issued hourly HRRR guidance at the Jeff Dozier snow study site.

The forecast reader is stdlib-only. Data acquisition uses the project's HRRR
environment for the original Utah Zarr grids and the NOAA projection.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
import os
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('data/mammoth/weather')
UTC = timezone.utc
HOUR = timedelta(hours=1)
SITE = {'name': 'Jeff Dozier Snow Study Site', 'latitude': 37.6431, 'longitude': -119.0291,
        'elevation_m': 2936}
FIELDS = {'precip_mm': 'precip_mm', 'snow_water_mm': 'snow_water_mm', 'temperature_k': 'temperature_c'}
COLUMNS = ('run_utc', 'available_utc', 'start_utc', 'end_utc', 'lead',
           'precip_mm', 'snow_water_mm', 'temperature_c')


def utc(value=None):
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Supply a timezone-aware issue or reference time.')
    return value.astimezone(UTC)


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(data, f, allow_nan=False)
            f.flush()
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def fetch_run(root, run):
    import pandas as pd
    import hrrr_portfolio as hrrr
    run = utc(run)
    path = Path(root)/CACHE/'runs'/run.strftime('%Y%m%dT%HZ.json')
    if path.is_file():
        try:
            data = json.loads(path.read_text())
            if data['run_utc'] == run.isoformat() and not data['errors'] and data['hours']:
                return data
        except (OSError, ValueError, KeyError):
            pass
    sites = {'cues': (SITE['latitude'], SITE['longitude'], SITE['elevation_m'])}
    grid = hrrr.location(*sites['cues'][:2])
    values, sources, errors = {}, {}, {}
    for field, name in FIELDS.items():
        try:
            periods, points, source = hrrr.field(pd.Timestamp(run), field, sites)
            value = {}
            for lead, point in zip(periods, points['cues']):
                v = float(point)
                if not math.isfinite(v) or (name != 'temperature_c' and v < 0):
                    continue
                value[int(lead)] = round(v - 273.15 if name == 'temperature_c' else v, 6)
            if not value:
                raise ValueError('All forecast values are missing.')
            values[name] = value
            sources[name] = dict(field=source['field'], units=source['units'],
                                 sha256=source['sha256'], url=hrrr.BASE + run.strftime('sfc/%Y%m%d/%Y%m%d_%Hz_fcst.zarr/')
                                 + source['field'] + '/' + source['field'] + f"/0.{grid['chunk_y']}.{grid['chunk_x']}")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            errors[name] = str(exc)
    period = sorted({lead for v in values.values() for lead in v})
    if not period:
        raise RuntimeError('Mammoth HRRR unavailable: ' + json.dumps(errors))
    available = (run + 3*HOUR).isoformat()
    hours = [dict(lead=lead, start_utc=(run+(lead-1)*HOUR).isoformat(),
                  end_utc=(run+lead*HOUR).isoformat(),
                  **{name: values.get(name, {}).get(lead) for name in FIELDS.values()}) for lead in period]
    data = dict(site=SITE, grid=grid, run_utc=run.isoformat(), available_utc=available,
                retrieved_utc=utc().isoformat(), hours=hours, errors=errors, sources=sources,
                source='NOAA operational HRRR via University of Utah forecast Zarr',
                documentation='https://mesowest.utah.edu/html/hrrr/',
                availability_note='Historical issue time uses a 3-hour availability allowance. Actual historical first-publication times are unavailable.')
    atomic_json(path, data)
    return data


def read_current(root=ROOT, now=None):
    now = utc(now)
    try:
        data = json.loads((Path(root)/CACHE/'current.json').read_text())
        if utc(data['available_utc']) > now:
            raise ValueError('Forecast is not yet available at requested time.')
        data['status'] = 'ready' if now-utc(data['available_utc']) < 9*HOUR else 'stale'
        return data
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return dict(status='unavailable', site=SITE, hours=[], message=str(exc), run_utc=None, available_utc=None)


def refresh_current(root=ROOT, now=None):
    now = utc(now)
    permitted = now - 3*HOUR
    latest = permitted.replace(hour=permitted.hour//6*6, minute=0, second=0, microsecond=0)
    messages = []
    for offset in (0, 6, 12):
        try:
            data = fetch_run(root, latest-offset*HOUR)
            if len(data['hours']) != 48 or data['errors']:
                raise ValueError('The run lacks all 48 leads for the three required HRRR fields.')
            if not all(all(row[k] is not None for k in FIELDS.values()) for row in data['hours']):
                raise ValueError('The run contains missing fields at the station.')
            atomic_json(Path(root)/CACHE/'current.json', data)
            return dict(status='partial' if offset else 'succeeded', run_utc=data['run_utc'],
                        available_utc=data['available_utc'], message='Mammoth HRRR guidance updated.',
                        errors=messages)
        except (RuntimeError, OSError, ValueError) as exc:
            messages.append(str(exc))
    return dict(status='failed', errors=messages, message='The latest Mammoth HRRR runs could not be collected.')


def history_runs(stride, cycles=(6,)):
    # Calendar-only sampling avoids selecting runs based on outcomes.
    runs = []
    for year in range(2020, 2025):
        start = datetime(year, 11, 1, 6, tzinfo=UTC)
        end = datetime(year+1, 4, 30, 6, tzinfo=UTC)
        runs += [(start+i*timedelta(days=1)).replace(hour=h) for i in range(0, (end-start).days+1, stride) for h in cycles]
    # Bridge documented missing 06 UTC origins without conditioning on snowfall.
    if 0 in cycles:
        runs.extend([utc('2022-01-07T00:00:00Z'), utc('2022-04-03T00:00:00Z')])
    return sorted(set(runs))


def _write_archive(folder, paths, columns, filename):
    folder.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=folder, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    count = 0
    try:
        with tmp_path.open('wb') as dest, gzip.GzipFile(filename='', mode='wb', mtime=0, fileobj=dest) as gz, io.TextIOWrapper(gz, encoding='utf-8', newline='') as raw:
            writer = csv.DictWriter(raw, fieldnames=columns)
            writer.writeheader()
            for path in sorted(paths):
                if path.name[:4] > '2025':
                    continue
                data = json.loads(path.read_text())
                for hour in data['hours']:
                    writer.writerow(dict(run_utc=data['run_utc'], available_utc=data['available_utc'], **hour))
                count += 1
        os.replace(tmp_path, folder/filename)
    finally:
        tmp_path.unlink(missing_ok=True)
    return count


def export_history(root):
    folder = Path(root)/CACHE
    return _write_archive(folder, (folder/'runs').glob('*.json'), COLUMNS, 'history.csv.gz')


def history(root, stride, workers, cycles=(6,)):
    runs = history_runs(stride, cycles)
    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_run, root, run): run for run in runs}
        for i, f in enumerate(as_completed(futures), 1):
            run = futures[f]
            try:
                f.result()
            except (OSError, RuntimeError, ValueError) as exc:
                failures[run.isoformat()] = str(exc)
            if i % 40 == 0:
                print(f'{i}/{len(runs)} processed', flush=True)
    count = export_history(root)
    suffix = '-'.join(str(h) for h in cycles)
    atomic_json(Path(root)/CACHE/f'collection-{suffix}.json', dict(complete_or_partial_runs=count,
                failures=failures, schedule=f'Every {stride} calendar days from November 1 to April 30 of winters 2020/21–2024/25, at UTC hours {cycles}.',
                required_fields=list(FIELDS.values()), site=SITE,
                timestamp_convention='start_utc is inclusive, end_utc is exclusive. Forecast hour L is the hour ending at initialization+L hours.'))
    print(f'Exported {count} runs. Acquisition failures {len(failures)}.', flush=True)



PROVIDER_URL = 'https://single-runs-api.open-meteo.com/v1/forecast'


def fetch_provider_run(root, run):
    run = utc(run)
    path = Path(root)/CACHE/'ecmwf_runs'/run.strftime('%Y%m%dT%HZ.json')
    if path.is_file():
        try:
            data = json.loads(path.read_text())
            if data['run_utc'] == run.isoformat() and len(data['hours']) == 72:
                return data
        except (OSError, ValueError, KeyError):
            pass
    params = dict(models='ecmwf_ifs', run=run.strftime('%Y-%m-%dT%H:%M'), latitude=SITE['latitude'],
                  longitude=SITE['longitude'], elevation=SITE['elevation_m'], timezone='GMT',
                  hourly='snowfall_water_equivalent')
    url = PROVIDER_URL+'?'+urlencode(params)
    req = Request(url, headers={'User-Agent':'Snowline noncommercial mountain weather research'})
    with urlopen(req, timeout=25) as response:
        raw = response.read()
    payload = json.loads(raw)
    hourly = payload['hourly']
    if (payload.get('utc_offset_seconds') != 0 or
            payload['hourly_units'].get('snowfall_water_equivalent') != 'mm' or
            hourly['time'][0] != params['run']):
        raise ValueError('ECMWF issue time, time zone or snow water units differ from requested run.')
    if len(hourly['time']) != len(hourly['snowfall_water_equivalent']) or len(hourly['time']) < 73:
        raise ValueError('Incomplete ECMWF time axis')
    hours = []
    for index in range(1, 73):
        end = run + index*HOUR
        if hourly['time'][index] != end.strftime('%Y-%m-%dT%H:%M'):
            raise ValueError('Noncontiguous ECMWF lead sequence.')
        value = hourly['snowfall_water_equivalent'][index]
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError('Missing or invalid ECMWF snowfall water equivalent.')
        hours.append(dict(lead=index, start_utc=(end-HOUR).isoformat(), end_utc=end.isoformat(),
                          snow_water_mm=float(value)))
    data = dict(site=SITE, grid=dict(latitude=payload['latitude'], longitude=payload['longitude'],
                provider_elevation_m=payload['elevation']), run_utc=run.isoformat(),
                available_utc=(run+8*HOUR).isoformat(), retrieved_utc=utc().isoformat(),
                hours=hours, source='Open-Meteo · ECMWF IFS HRES', url=url,
                documentation='https://open-meteo.com/en/docs/single-runs-api', sha256=hashlib.sha256(raw).hexdigest(),
                field='snowfall_water_equivalent', unit='mm', model=params['models'],
                attribution='Open-Meteo and ECMWF (CC BY 4.0)',
                availability_note='Historical benchmark uses an 8-hour IFS publication allowance; first issue receipt times cannot be recovered.')
    atomic_json(path, data)
    return data


def read_provider(root=ROOT, now=None):
    now = utc(now)
    try:
        data = json.loads((Path(root)/CACHE/'ecmwf_current.json').read_text())
        if utc(data['available_utc']) > now:
            raise ValueError('ECMWF forecast is not yet available at requested time.')
        data['status'] = 'ready' if now-utc(data['available_utc']) < 18*HOUR else 'stale'
        return data
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return dict(status='unavailable', site=SITE, hours=[], message=str(exc), run_utc=None, available_utc=None)


def refresh_provider(root=ROOT, now=None):
    now = utc(now)
    permitted = now-8*HOUR
    latest = permitted.replace(hour=permitted.hour//12*12, minute=0, second=0, microsecond=0)
    messages = []
    for offset in (0, 12):
        try:
            data = fetch_provider_run(root, latest-offset*HOUR)
            atomic_json(Path(root)/CACHE/'ecmwf_current.json', data)
            return dict(status='partial' if offset else 'succeeded', run_utc=data['run_utc'],
                        available_utc=data['available_utc'], message='Mammoth ECMWF snowfall water forecast updated.',
                        errors=messages)
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            messages.append(str(exc))
    return dict(status='failed', errors=messages, message='Mammoth ECMWF comparison could not update.')


def export_provider(root):
    folder = Path(root)/CACHE
    return _write_archive(folder, (folder/'ecmwf_runs').glob('*.json'),
                          ['run_utc','available_utc','start_utc','end_utc','lead','snow_water_mm'], 'ecmwf_history.csv.gz')


def provider_history(root, workers):
    # Fixed dates and both global cycles, independent of measured storms.
    periods = [('2024-03-15', '2024-04-30'), ('2024-10-31', '2025-04-30')]
    dates = []
    for start, end in periods:
        a, b = (datetime.fromisoformat(v).replace(tzinfo=UTC) for v in (start,end))
        dates += [a+i*24*HOUR for i in range((b-a).days+1)]
    # Interleave the seasons so interrupted collection still spans the winter.
    jobs = [d+h*HOUR for k in range(5) for d in dates[k::5] for h in (0,12)]
    failures = {}
    def one(run):
        for attempt in range(3):
            try:
                return fetch_provider_run(root, run)
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                wait = exc.headers.get('Retry-After', '')
                time.sleep(min(65, max(2**attempt, int(wait) if wait.isdigit() else 15)))
    with ThreadPoolExecutor(max_workers=min(4,workers)) as pool:
        pending = {pool.submit(one,run):run for run in jobs}
        for i, future in enumerate(as_completed(pending),1):
            try:
                future.result()
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                failures[pending[future].isoformat()] = str(exc)
            if i % 80 == 0:
                print(f'{i}/{len(jobs)} runs processed; {export_provider(root)} exported', flush=True)
    count = export_provider(root)
    atomic_json(Path(root)/CACHE/'ecmwf_collection.json',dict(runs=count, failures=failures,
                schedule=periods, cycles_utc=[0,12], assumed_availability_hours=8,
                source='Open-Meteo single-run ECMWF IFS HRES forecasts, not the stitched best-match product.'))
    print(f'Exported {count} ECMWF forecast cycles, {len(failures)} failures.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', action='store_true')
    p.add_argument('--provider', action='store_true')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--cycles', type=int, nargs='+', choices=(0,6,12,18), default=[6])
    args = p.parse_args()
    if args.provider:
        if args.history: provider_history(ROOT, args.workers)
        else: print(json.dumps(refresh_provider(ROOT), indent=2))
    elif args.history:
        if args.stride < 1 or not (1 <= args.workers <= 16): p.error('Invalid stride or worker count')
        history(ROOT, args.stride, args.workers, tuple(args.cycles))
    else:
        print(json.dumps(refresh_current(ROOT), indent=2))
