"""Read-only model inference and public forecast acquisition for the local dashboard.

Refresh downloads predictors; it never trains on a current forecast or modifies
frozen research artifacts. Hourly provider snowfall and station-calibrated daily
snowfall are separate products with separate target clocks and provenance.
"""
from __future__ import annotations

import copy
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import build_mos_features as mos
import nwp_benchmark as nwp
from snowline_sites import SITES

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/dashboard/current'
MODEL_NAME = '+ wind forecasts'
MODEL_LABEL = 'Current daily model · GFS MOS + calendar + ENSO'
OPEN_METEO = 'https://api.open-meteo.com/v1/forecast'
IEM = 'https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py'
LOCKS = {site: threading.Lock() for site in SITES}
MOS_DOWNLOAD_LOCK = threading.Lock()
LAST_MOS_DOWNLOAD = 0.


def _utc(value=None):
    t = pd.Timestamp.now(tz='UTC') if value is None else pd.Timestamp(value)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def _iso(value):
    return _utc(value).isoformat()


def _number(value):
    if value is None: return None
    x = float(value)
    return x if math.isfinite(x) else None


def list_mountains():
    return [{**copy.deepcopy(x), 'capabilities': _capabilities(key)} for key, x in SITES.items()]


def _capabilities(site):
    return {'max_hours': 48, 'hourly_weather': True, 'daily_snow_model': site != 'mammoth',
            'powder_window_hours': 24 if site != 'mammoth' else None, 'hourly_snow_water_model': site == 'mammoth', 'radar_observation': site == 'mammoth', 'radar_nowcast': site == 'alta',
            'radar_max_hours': 3 if site == 'alta' else 0, 'refresh_online': True,
            'refresh_retrains_model': False, 'hourly_snow_validated': False}


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def _fetch(url, timeout=60):
    request = urllib.request.Request(url, headers={'User-Agent': 'Mountain snow research dashboard / noncommercial public data'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        if len(body) > 8_000_000: raise ValueError('Unexpectedly large forecast response')
        metadata = {'url': url, 'retrieved_at': _iso(None), 'http_status': response.status,
                    'sha256': hashlib.sha256(body).hexdigest()}
    return body, metadata


def _archive(site, kind, body, metadata, extension):
    stamp = _utc(metadata['retrieved_at']).strftime('%Y%m%dT%H%M%S%fZ')
    path = CACHE / 'raw' / f'{site}_{kind}_{stamp}.{extension}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    _atomic_json(path.with_suffix('.meta.json'), metadata)


def _hourly_from_response(response, now):
    """Keep 48 complete future one-hour intervals; sums are endpoint-labelled."""
    hourly = response.get('hourly', {})
    units = response.get('hourly_units', {})
    expected = {'snowfall': 'cm', 'precipitation': 'mm', 'temperature_2m': '°C', 'wind_speed_10m': 'km/h'}
    if any(units.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Unexpected forecast units: {units}')
    times = pd.to_datetime(hourly.get('time', []), utc=True)
    if times.has_duplicates or not times.is_monotonic_increasing: raise ValueError('Hourly timestamps are not unique and ordered')
    if any(len(hourly.get(k, [])) != len(times) for k in expected): raise ValueError('Incomplete hourly variable arrays')
    start = now.ceil('h')
    wanted = pd.date_range(start + pd.Timedelta(hours=1), periods=48, freq='h')
    positions = times.get_indexer(wanted)
    if (positions < 0).any(): raise ValueError('Source does not cover the next 48 full hourly intervals')
    result = []
    for horizon, (time, i) in enumerate(zip(wanted, positions), 1):
        snow = _number(hourly['snowfall'][i])
        result.append({'time': time.isoformat(), 'start': (time-pd.Timedelta(hours=1)).isoformat(),
                       'horizon_hours': horizon, 'snowfall_inches': None if snow is None else snow/2.54,
                       'precipitation_mm': _number(hourly['precipitation'][i]),
                       'temperature_c': _number(hourly['temperature_2m'][i]),
                       'wind_kmh': _number(hourly['wind_speed_10m'][i])})
    if all(row['snowfall_inches'] is None for row in result): raise ValueError('Provider returned no snowfall values')
    return result, start


def _fetch_hourly(site, now):
    target = SITES[site]
    params = {'latitude': target['latitude'], 'longitude': target['longitude'], 'elevation': target['elevation_m'],
              'timezone': 'GMT', 'forecast_days': 4, 'hourly': 'temperature_2m,precipitation,snowfall,wind_speed_10m',
              'temperature_unit': 'celsius', 'precipitation_unit': 'mm', 'wind_speed_unit': 'kmh'}
    url = OPEN_METEO + '?' + urllib.parse.urlencode(params)
    body, metadata = _fetch(url)
    response = json.loads(body)
    if response.get('error'): raise ValueError(str(response.get('reason', 'Provider error')))
    rows, start = _hourly_from_response(response, now)
    _archive(site, 'open_meteo', body, metadata, 'json')
    source = {'name': 'Open-Meteo hourly weather guidance', 'issued_at': None,
              'retrieved_at': metadata['retrieved_at'], 'url': url, 'documentation': 'https://open-meteo.com/en/docs',
              'status': 'refreshed', 'initialization_note': 'This combined-model endpoint does not return a coherent model initialization time.',
              'grid_latitude': response.get('latitude'), 'grid_longitude': response.get('longitude'),
              'elevation_m': response.get('elevation'), 'sha256': metadata['sha256'],
              'snowfall_note': 'Provider snowfall conversion; not our locally trained hourly snow model. No calibrated hourly uncertainty supplied.',
              'license': 'Open-Meteo / underlying weather providers, CC BY 4.0 attribution'}
    return rows, source, start


def _canonical_mos(raw):
    required = {'runtime','ftime','station','model','tmp','dpt','wdr','wsp','p06','q06'}
    if not required.issubset(raw): raise ValueError('MOS response is missing required fields')
    d = raw.copy()
    d['runtime'] = pd.to_datetime(d.runtime, utc=True, errors='raise')
    d['ftime'] = pd.to_datetime(d.ftime, utc=True, errors='raise')
    d = d.sort_values(['runtime','ftime']).drop_duplicates(['runtime','ftime'], keep='last')
    for c in ['tmp','dpt','wdr','wsp','p06','q06']: d[c] = pd.to_numeric(d[c], errors='coerce')
    d['temperature_c'] = (d.tmp.where(d.tmp.ne(999))-32)*5/9
    d['dewpoint_c'] = (d.dpt.where(d.dpt.ne(999))-32)*5/9
    direction = d.wdr.where(d.wdr.between(0,360))
    d['wind_speed_mps'] = d.wsp.where(d.wsp.between(0,98))*.514444
    theta = np.deg2rad(direction)
    d['wind_u_mps'] = -d.wind_speed_mps*np.sin(theta)
    d['wind_v_mps'] = -d.wind_speed_mps*np.cos(theta)
    d['precip_probability_6h'] = d.p06.where(d.p06.between(0,100))/100
    d['qpf_category_6h'] = d.q06.where(d.q06.between(0,6))
    d['lead_hours'] = (d.ftime-d.runtime).dt.total_seconds()/3600
    return d


def _mos_future(raw, site, now):
    """Use only the trained 00UTC cycle and lead-day 1/2 reporting clocks."""
    station, clock = mos.SITES[site]
    d = _canonical_mos(raw)
    if set(d.station) != {station} or set(d.model) != {'GFS'}: raise ValueError('MOS station/model identity mismatch')
    d = d[d.runtime.dt.hour.eq(0) & (d.runtime + pd.Timedelta(hours=6) <= now)]
    for run in sorted(d.runtime.unique(), reverse=True):
        run = _utc(run)
        g = d[d.runtime.eq(run)].sort_values('ftime')
        rows = []
        for lead in [1,2]:
            lo = clock+(lead-1)*24
            start, end = run+pd.Timedelta(hours=lo), run+pd.Timedelta(hours=lo+24)
            if start < now or end > now.ceil('h')+pd.Timedelta(hours=48): continue
            f, reason = mos.daily_features(g, lo, lo+24)
            if reason: continue
            rows.append({'site': site, 'station': station, 'model': 'GFS_MOS', 'archive_type': 'operational_forecast',
                         'forecast_reference_utc': run.isoformat(), 'issue_time_utc': (run+pd.Timedelta(hours=6)).isoformat(),
                         'valid_start_utc': start.isoformat(), 'valid_end_utc': end.isoformat(),
                         'target_date': str(end.date()), 'lead_day': lead, 'target_inches': np.nan, **f})
        if rows: return pd.DataFrame(rows), run
    raise ValueError('No complete, fully future 24-hour MOS reporting window is available within 48 hours')


@lru_cache(maxsize=4)
def _models(site):
    path = ROOT / 'outputs/nwp-weather/mos' / site / 'final_models.joblib'
    # Only repository-owned model artifacts can enter joblib; never user uploads.
    bundle = joblib.load(path)
    return bundle, hashlib.sha256(path.read_bytes()).hexdigest()


def _daily_inference(site, future):
    future = nwp.validate_join(nwp.add_lagged_enso(future))
    bundle, model_hash = _models(site)
    selected = bundle['models'][MODEL_NAME]
    absent = set(selected['columns'])-set(future)
    if absent: raise ValueError(f'Missing trained forecast features: {sorted(absent)}')
    with threadpool_limits(limits=2):
        predicted = selected['model'].predict_distribution(nwp.features(future, selected['columns']))
        historical = nwp.climatology(bundle['climatology_history'], future)
    rows = []
    for i, r in enumerate(future.itertuples()):
        rows.append({'start': r.valid_start_utc.isoformat(), 'end': r.valid_end_utc.isoformat(),
                     'target_date': r.target_date.date().isoformat(), 'mean_inches': float(predicted['mean'][i]),
                     'p_powder': float(predicted['p_powder'][i]), 'q10': float(predicted['q10'][i]), 'q90': float(predicted['q90'][i]),
                     'model': MODEL_LABEL, 'historical_mean_inches': float(historical['mean'][i]),
                     'historical_p_powder': float(historical['p_powder'][i]), 'lead_day': int(r.lead_day),
                     'source_issued_at': r.forecast_reference_utc.isoformat(),
                     'assumed_available_at': r.issue_time_utc.isoformat(), 'powder_threshold_inches': 5})
    return rows, model_hash


def _fetch_mos_body(station, now):
    """Share the VT station and respect the public archive's request rate."""
    global LAST_MOS_DOWNLOAD
    with MOS_DOWNLOAD_LOCK:
        # Reuse a recent original download, keeping its actual retrieval stamp.
        for meta_path in sorted((CACHE/'raw').glob('*_gfs_mos_*.meta.json'), reverse=True):
            try:
                metadata = json.loads(meta_path.read_text())
                requested_station = urllib.parse.parse_qs(urllib.parse.urlsplit(metadata['url']).query).get('station', [])
                age = (now-_utc(metadata['retrieved_at'])).total_seconds()
                data_path = meta_path.with_suffix('').with_suffix('.csv')
                if requested_station == [station] and 0 <= age < 1200 and data_path.exists():
                    metadata['reused_recent_download'] = True
                    return data_path.read_bytes(), metadata
            except (ValueError, KeyError, OSError):
                continue
        params = {'station': station, 'model': 'GFS', 'sts': (now-pd.Timedelta(days=3)).strftime('%Y-%m-%dT00:00Z'),
                  'ets': now.strftime('%Y-%m-%dT%H:%MZ'), 'format': 'csv'}
        url = IEM + '?' + urllib.parse.urlencode(params)
        for attempt in range(2):
            delay = 3 - (time.monotonic()-LAST_MOS_DOWNLOAD)
            if delay > 0: time.sleep(delay)
            LAST_MOS_DOWNLOAD = time.monotonic()
            try:
                body, metadata = _fetch(url)
                _archive('station_'+station, 'gfs_mos', body, metadata, 'csv')
                return body, metadata
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt: raise
                retry_after = exc.headers.get('Retry-After', '6')
                try: delay = max(6., float(retry_after))
                except ValueError:
                    try: delay = max(6., (_utc(parsedate_to_datetime(retry_after))-_utc()).total_seconds())
                    except (TypeError, ValueError): delay = 6.
                if delay > 30: raise  # Do not retry earlier than the provider permits.
                time.sleep(delay)


def _fetch_daily(site, now):
    station = mos.SITES[site][0]
    body, metadata = _fetch_mos_body(station, now)
    url = metadata['url']
    raw = pd.read_csv(io.BytesIO(body))
    future, run = _mos_future(raw, site, now)
    daily, model_hash = _daily_inference(site, future)
    _archive(site, 'gfs_mos', body, metadata, 'csv')
    source = {'name': 'NOAA GFS MOS via Iowa Environmental Mesonet', 'issued_at': run.isoformat(),
              'retrieved_at': metadata['retrieved_at'], 'url': url, 'station': station, 'status': 'refreshed',
              'assumed_available_at': (run+pd.Timedelta(hours=6)).isoformat(), 'sha256': metadata['sha256'],
              'model_sha256': model_hash, 'model_refitted': False,
              'reused_recent_download': metadata.get('reused_recent_download', False),
              'note': 'Regional airport forecast predictors calibrated to the named snow station; first original receipt time is unavailable.'}
    return daily, source


def _saved_snapshot(site, now):
    """Graceful, visibly dated initial display; no implicit network calls."""
    if site == 'mammoth':
        source = {'name': 'Open-Meteo hourly weather guidance', 'url': OPEN_METEO,
                  'issued_at': None, 'retrieved_at': None, 'status': 'unavailable'}
        return _payload(site, [], [], source, None, now, 'unavailable',
                        ['Use Refresh all to download the Main Lodge forecast.'])
    warnings = ['Saved research snapshot. Use Refresh latest data for a new forecast.']
    hourly, daily = [], []
    source = {'name': 'Saved JMA forecast snapshot via Open-Meteo', 'issued_at': '2026-11-04T00:00:00+00:00',
              'retrieved_at': None, 'url': 'https://open-meteo.com/en/docs/jma-api', 'status': 'cached',
              'snowfall_note': 'Provider-derived snowfall; not calibrated hourly snowfall.'}
    path = ROOT/'data/nwp_recent/current_jma_gsm_hourly.csv'
    if path.exists():
        d = pd.read_csv(path); d = d[d.site.eq(site)].copy()
        d['time'] = pd.to_datetime(d.time, utc=True)
        start = now.ceil('h')
        d = d[(d.time > start) & (d.time <= start+pd.Timedelta(hours=48))]
        for r in d.itertuples():
            snowfall = _number(r.snowfall)
            wind = _number(r.wind_speed_10m)
            hourly.append({'time': r.time.isoformat(), 'start': (r.time-pd.Timedelta(hours=1)).isoformat(),
                           'horizon_hours': int((r.time-start).total_seconds()/3600),
                           'snowfall_inches': None if snowfall is None else snowfall/2.54,
                           'precipitation_mm': _number(r.precipitation), 'temperature_c': _number(r.temperature_2m),
                           'wind_kmh': None if wind is None else wind*3.6})
    base = ROOT/'outputs/nwp-weather/mos'/site
    if (base/'current_predictions.csv').exists():
        p = pd.read_csv(base/'current_predictions.csv'); inputs = pd.read_csv(base/'current_input.csv')
        for r in p[p.model.eq(MODEL_NAME)].itertuples():
            x = inputs[inputs.row_id.eq(r.row_id)].iloc[0]
            if _utc(x.valid_start_utc) < now or _utc(x.valid_end_utc) > now.ceil('h')+pd.Timedelta(hours=48): continue
            h = p[p.row_id.eq(r.row_id) & p.model.eq(nwp.NAMES[0])].iloc[0]
            daily.append({'start': _iso(x.valid_start_utc), 'end': _iso(x.valid_end_utc),
                          'target_date': r.target_date, 'mean_inches': r.predicted_inches, 'p_powder': r.p_powder,
                          'q10': r.q10, 'q90': r.q90, 'model': MODEL_LABEL,
                          'historical_mean_inches': h.predicted_inches, 'historical_p_powder': h.p_powder,
                          'source_issued_at': _iso(x.forecast_reference_utc), 'lead_day': int(r.lead_day)})
    if not daily: warnings.append('Saved daily forecast periods have expired. Refresh to obtain future reporting-day predictions.')
    return _payload(site, hourly, daily, source, None, now, 'cached', warnings)


def _evaluation(site):
    path = ROOT/'outputs/nwp-weather/mos'/site/'chronological_metrics.csv'
    if not path.exists(): return None
    d = pd.read_csv(path)
    rows = d[d.model.isin([MODEL_NAME, nwp.NAMES[0]]) & d.leads.astype(str).eq('all')]
    return {'scope': 'Earlier whole-winter chronological benchmark; daily station target, not hourly guidance.',
            'report_url': '/reports/weather/report.html',
            'rows': [{'model': MODEL_LABEL if r.model == MODEL_NAME else 'Historical calendar average',
                      'mae_inches': float(r.mae), 'rmse_inches': float(r.rmse), 'powder_brier': float(r.powder_brier),
                      'cases': int(r.rows), 'winters': int(r.winters)} for r in rows.itertuples()]}


def _payload(site, hourly, daily, source, daily_source, now, status, warnings):
    model_notes = [] if site == 'mammoth' else [
        'Hourly snowfall is weather-provider guidance; daily snow and ≥5-inch powder probabilities come from our separate station model.',
        'Daily forecasts use complete station reporting periods, not arbitrary rolling 24-hour windows. Prediction intervals have not been prospectively calibrated.']
    return {'schema_version': 1, 'site': copy.deepcopy(SITES[site]), 'generated_at': now.isoformat(),
            'source': source, 'daily_source': daily_source, 'status': status,
            'forecast_start': hourly[0]['start'] if hourly else None, 'forecast_end': hourly[-1]['time'] if hourly else None,
            'hourly': hourly, 'daily': daily, 'capabilities': _capabilities(site), 'evaluation': _evaluation(site),
            'warnings': list(dict.fromkeys(warnings + [SITES[site]['target_note'],
                'Hours 1–48 start at the next whole UTC hour; precipitation timestamps mark the end of each one-hour interval.'] + model_notes))}


def _serve(result, now, cached):
    """Age every source and remove expired periods each time a snapshot is served."""
    result = copy.deepcopy(result)
    if isinstance(result.get('site'), dict) and result['site'].get('id') == 'mansfield':
        result['site']['name'] = SITES['mansfield']['name']
    start = now.ceil('h')
    end = start + pd.Timedelta(hours=48)
    original_count = len(result.get('hourly', []))
    rows = [r for r in result.get('hourly', []) if _utc(r['start']) >= start and _utc(r['time']) <= end]
    for row in rows:
        row['horizon_hours'] = int((_utc(row['time'])-start).total_seconds()/3600)
    result['hourly'] = rows
    result['daily'] = [r for r in result.get('daily', []) if _utc(r['start']) >= now and _utc(r['end']) <= end]
    result['forecast_start'] = rows[0]['start'] if rows else None
    result['forecast_end'] = rows[-1]['time'] if rows else None
    result['hours_available'] = len(rows)
    if len(rows) < 48:
        result['warnings'].append(f'Only {len(rows)} complete future hourly periods remain in this snapshot; refresh to request all 48 hours.')
    if original_count != len(rows):
        result['warnings'].append('Expired hourly periods were removed; previous predictions are not relabeled as current data.')
    for key in ['source', 'daily_source']:
        source = result.get(key)
        if source:
            timestamp = source.get('retrieved_at') or source.get('issued_at')
            source['age_hours'] = max(0., (now-_utc(timestamp)).total_seconds()/3600) if timestamp else None
            source['stale'] = source['age_hours'] is None or source['age_hours'] > 6
    age_source = result.get('source') or {}
    result['cache'] = {'served_at': now.isoformat(), 'is_cached': cached,
                       'age_hours': age_source.get('age_hours'),
                       'snapshot_generated_at': result['generated_at']}
    if cached:
        result['status'] = 'cached' if rows or result['daily'] else 'unavailable'
    result['warnings'] = list(dict.fromkeys(result['warnings']))
    return result


def get_forecast(site, refresh=False):
    """Return a JSON-safe forecast. A refresh is acquisition plus inference only."""
    if site not in SITES: raise ValueError(f'Unknown mountain: {site}')
    with LOCKS[site]:
        now = _utc(); path = CACHE/f'{site}.json'; old = None
        if path.exists():
            try: old = json.loads(path.read_text())
            except (OSError, ValueError): pass
        if not refresh:
            result = copy.deepcopy(old) if old else _saved_snapshot(site, now)
            return _serve(result, now, cached=True)
        warnings, errors = [], []
        with ThreadPoolExecutor(max_workers=2) as pool:
            hourly_task = pool.submit(_fetch_hourly, site, now)
            daily_task = pool.submit(_fetch_daily, site, now) if _capabilities(site)['daily_snow_model'] else None
            try:
                hourly, source, _ = hourly_task.result()
            except Exception as exc:
                errors.append('Hourly source refresh failed: '+str(exc))
                fallback = old or _saved_snapshot(site, now)
                hourly = fallback.get('hourly', []); source = copy.deepcopy(fallback['source'])
                source['status'] = 'cached' if hourly else 'unavailable'
                warnings.append('Hourly refresh failed; retained forecast is dated and has not been refreshed.')
            try:
                daily, daily_source = daily_task.result() if daily_task is not None else ([], None)
            except Exception as exc:
                errors.append('Daily model refresh failed: '+str(exc))
                daily = [r for r in (old or {}).get('daily', []) if _utc(r['start']) >= now]
                daily_source = copy.deepcopy((old or {}).get('daily_source'))
                if daily_source: daily_source['status'] = 'cached'
                if daily:
                    warnings.append('Latest daily refresh failed; showing the previously retrieved issued forecast with its original source date.')
                else:
                    warnings.append('Daily model forecast unavailable from the newest data. No powder probability was inferred from hourly snowfall.')
        status = 'ready' if not errors else ('partial' if hourly or daily else 'unavailable')
        result = _payload(site, hourly, daily, source, daily_source, now, status, warnings+errors)
        result['errors'] = errors
        result['last_refresh_attempt_at'] = now.isoformat()
        all_failed = len(errors) == (2 if daily_task is not None else 1)
        if all_failed and old:
            result['generated_at'] = old['generated_at']
        result = _serve(result, now, cached=False)
        if not result['hourly'] and not result['daily']: result['status'] = 'unavailable'
        # Keep the original snapshot when every attempted source failed.
        if not all_failed and result['status'] != 'unavailable': _atomic_json(path, result)
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', choices=list(SITES), default='alta')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    result = get_forecast(args.site, args.refresh)
    print(json.dumps({'site': result['site']['id'], 'status': result['status'], 'hourly_rows': len(result['hourly']),
                      'daily_rows': len(result['daily']), 'warnings': result['warnings'], 'daily': result['daily']}, indent=2))
