"""Issue and serve hourly predictions of snowfall water equivalent at Mammoth CUES."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile
import zlib

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('outputs/mammoth-hourly')
SNAPSHOT = OUTPUT / 'current_forecast.json'
REPORT = '/reports/mammoth/report.html'
HORIZONS = (1, 3, 6, 12, 24, 48)
HOUR = timedelta(hours=1)
TARGET = dict(name='Jeff Dozier Snow Study Site', latitude=37.6431, longitude=-119.0291,
              elevation_m=2936, quantity='snow_water_equivalent', unit='mm',
              note='The target is water in falling snow measured by the disdrometer, about 1.1 km from Mammoth Main Lodge.')
PROVIDER = 'HRRR'


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError('Mammoth forecast times must include the timezone.')
    return value.astimezone(timezone.utc)


def _number(value, *, signed=False):
    if isinstance(value, bool):
        raise ValueError('Boolean forecast amount.')
    value = float(value)
    if not math.isfinite(value) or (not signed and value < 0):
        raise ValueError('A forecast input or prediction is missing or out of range.')
    return value


def _template():
    return dict(site_id='mammoth', model_name='Our model', quantity='snow_water_equivalent',
                unit='mm', provider_name=PROVIDER, target=TARGET.copy(), report_url=REPORT,
                status='unavailable', issued_at=None, generated_at=None, start=None, end=None,
                method_id=None, hourly=[], horizons=[], sources={}, warnings=[])


def _unavailable(message):
    result = _template()
    result['warnings'] = [message]
    result['horizons'] = [dict(hours=h, quantity=result['quantity'], unit='mm', status='unavailable',
                               provider_name=PROVIDER, mean=None, provider_mean=None, lower=None,
                               upper=None, start=None, end=None, issued_at=None, warnings=[message])
                          for h in HORIZONS]
    return result


def _model_file(root):
    for path in (root / OUTPUT / 'model.json', root / 'artifacts/mammoth-hourly/model.json'):
        if path.is_file():
            return path
    raise FileNotFoundError('The fitted Mammoth model is unavailable.')


def _weather_inputs(weather, start, now):
    if weather.get('status') != 'ready':
        raise ValueError('Current CUES-grid HRRR guidance is unavailable or stale.')
    run, available = _utc(weather['run_utc']), _utc(weather['available_utc'])
    if not run <= available <= now or now - run > 12 * HOUR:
        raise ValueError('The HRRR forecast has a future availability time or expired initialization.')
    if weather.get('retrieved_utc') and _utc(weather['retrieved_utc']) > now:
        raise ValueError('The HRRR snapshot was retrieved after the model issue time.')
    site = weather['site']
    if any(abs(float(site[key]) - TARGET[key]) > 0.0001 for key in ('latitude', 'longitude')):
        raise ValueError('The weather input is for a different target than the fitted Mammoth model.')
    by_start = {}
    for row in weather['hours']:
        a, b = _utc(row['start_utc']), _utc(row['end_utc'])
        lead = row['lead']
        if (lead not in range(1, 49) or b != run + HOUR * lead or a != b - HOUR
                or a in by_start):
            raise ValueError('HRRR hours are duplicated or do not match the stated forecast cycle.')
        by_start[a] = row
    rows = []
    for i in range(48):
        a, b = start + HOUR * i, start + HOUR * (i + 1)
        row = dict(start_utc=a.isoformat(), end_utc=b.isoformat(), lead=None,
                   precip_mm=None, snow_water_mm=None, temperature_c=None)
        if b <= run + 48 * HOUR:
            if a not in by_start:
                raise ValueError('The published HRRR forecast has an internal gap.')
            native = by_start[a]
            row.update(lead=native['lead'], precip_mm=_number(native['precip_mm']),
                       snow_water_mm=_number(native['snow_water_mm']),
                       temperature_c=_number(native['temperature_c'], signed=True))
        rows.append(row)
    if rows[0]['snow_water_mm'] is None:
        raise ValueError('The first requested forecast hour has no native HRRR inputs.')
    return rows


def _ecmwf_inputs(root, start, now):
    """Match the optional, independently issued Open-Meteo ECMWF snow-water series."""
    source = {'name': 'Open-Meteo ECMWF IFS', 'status': 'unavailable', 'unit': 'mm'}
    try:
        from mammoth_weather import read_provider
        weather = read_provider(root, now=now)
        source.update({key: weather[key] for key in ('run_utc', 'available_utc', 'retrieved_utc',
                       'source', 'site', 'grid', 'url', 'documentation') if key in weather})
        if weather.get('status') != 'ready':
            raise ValueError(weather.get('message') or 'Current ECMWF forecast is unavailable.')
        run, available = _utc(weather['run_utc']), _utc(weather['available_utc'])
        if not run <= available <= now or (weather.get('retrieved_utc') and _utc(weather['retrieved_utc']) > now):
            raise ValueError('ECMWF initialization or availability is later than issuance.')
        if any(abs(float(weather['site'][key]) - TARGET[key]) > .0001 for key in ('latitude','longitude')):
            raise ValueError('The ECMWF forecast request used different coordinates.')
        amounts = {}
        for row in weather['hours']:
            a, b = _utc(row['start_utc']), _utc(row['end_utc'])
            if a < start or a >= start + 48 * HOUR:
                continue
            if b - a != HOUR or b != run + row['lead'] * HOUR or a in amounts:
                raise ValueError('ECMWF hourly timestamps are duplicated or inconsistent with the forecast cycle.')
            amounts[a] = None if row['snow_water_mm'] is None else _number(row['snow_water_mm'])
        readings = [amounts.get(start + HOUR * i) for i in range(48)]
        matched = sum(v is not None for v in readings)
        if not matched:
            raise ValueError('No ECMWF snow-water hours overlap the model forecast.')
        source.update(status='ready' if matched == 48 else 'partial', matched_hours=matched)
        return readings, source
    except (OSError, ImportError, ValueError, KeyError, TypeError) as exc:
        source.update(status='unavailable', message=str(exc), matched_hours=0)
        return [None] * 48, source


def _radar_input(root, start, now):
    from mammoth_radar import read_current_radar
    from train_mammoth_hourly import radar_features
    cutoff = min(now, start - timedelta(minutes=5))
    scan = read_current_radar(root, cutoff)
    source = {key: scan.get(key) for key in ('source', 'timestamp', 'url', 'quality_url', 'archive_path')}
    source['available'] = False
    try:
        stamp = _utc(scan['timestamp'])
        if stamp > cutoff or start - stamp > timedelta(minutes=90):
            raise ValueError('No radar scan within the model’s training age limit.')
        path = (root / scan['archive_path']).resolve()
        if not path.is_relative_to((root / 'data/mammoth/radar').resolve()):
            raise ValueError('Invalid Mammoth radar crop location.')
        features = radar_features(path)
        if _utc(features['timestamp']) != stamp:
            raise ValueError('Radar feature time differs from the source time.')
        used = any(features.get(name) is not None and math.isfinite(float(features[name]))
                   for name in ('point', 'local', 'regional', 'west'))
        source['available'] = used
        if used:
            source['age_minutes_at_start'] = (start - stamp).total_seconds() / 60
        else:
            source['message'] = 'The radar crop has no quality-screened precipitation features.'
        return features if used else None, source
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile, zlib.error) as exc:
        source['message'] = str(exc)
        return None, source


def _build(root, now):
    from mammoth_weather import read_current
    from train_mammoth_hourly import forecast_interval, predict_hourly
    path = _model_file(root)
    raw = path.read_bytes()
    model = json.loads(raw)
    boundary = now.replace(minute=0, second=0, microsecond=0)
    start = boundary if boundary == now else boundary + HOUR
    weather = read_current(root, now=now)
    inputs = _weather_inputs(weather, start, now)
    ecmwf, ecmwf_source = _ecmwf_inputs(root, start, now)
    radar, radar_source = _radar_input(root, start, now)
    uses_radar = model['uses_radar']
    radar_source['used_as_input'] = bool(uses_radar and radar_source['available'])
    means = predict_hourly(model=model, weather=inputs, radar=radar if uses_radar else None, issue=start)
    if len(means) != 48:
        raise ValueError('The Mammoth fitted model must return 48 hourly predictions.')
    result = _template()
    provider_source = {key: weather[key] for key in ('run_utc', 'available_utc', 'retrieved_utc',
                        'site', 'grid', 'source', 'documentation', 'availability_note') if key in weather}
    native = weather.get('sources', {}).get('snow_water_mm', {})
    provider_source.update(name=PROVIDER, variable='WEASD hourly accumulation', unit='mm',
                           url=native.get('url'), field=native.get('field'))
    sources = dict(provider=provider_source, ecmwf=ecmwf_source, radar=radar_source,
                   model=dict(artifact=str(path.relative_to(root)), sha256=sha256(raw).hexdigest(),
                              family=model['family'], version=model.get('version'), uses_radar=uses_radar))
    result.update(status='ready', issued_at=now.isoformat(), generated_at=now.isoformat(),
                  start=start.isoformat(), end=(start + 48 * HOUR).isoformat(),
                  method_id=model['family'], sources=sources)
    if uses_radar and not radar_source['used_as_input']:
        result['warnings'].append('No usable radar input at issuance. The model uses its weather inputs.')
    if ecmwf_source['status'] == 'unavailable':
        result['warnings'].append('The ECMWF comparison is unavailable. ' + ecmwf_source['message'])
    from mammoth_depth import TARGET as LODGE, predict_depth, read_model
    try:
        depth = read_model(root)
        depths = predict_depth(depth, inputs)
        result['depth'] = dict(status='ready', target=LODGE, quantity='fresh_snow_depth', unit='inches',
                               method=depth['method_id'], ratios_cold_warm=depth['ratios_cold_warm'],
                               temporal_validation='monthly', hourly_validated=False, note=depth['caveat'])
    except (ValueError, OSError, KeyError, TypeError) as exc:
        depths = [None] * 48
        result['depth'] = dict(status='unavailable', target=LODGE, unit='inches', note=str(exc))
    for i, (row, mean) in enumerate(zip(inputs, means)):
        result['hourly'].append(dict(horizon_hours=i + 1, start=row['start_utc'], end=row['end_utc'],
                                     mean_mm=_number(mean), provider_mm=row['snow_water_mm'], ecmwf_mm=ecmwf[i],
                                     depth_inches=depths[i], hrrr_lead=row['lead']))
    for h in HORIZONS:
        rows = result['hourly'][:h]
        projected = math.fsum(r['mean_mm'] for r in rows)
        lower, upper = forecast_interval(model, h, projected)
        lower = _number(lower) if lower is not None else None
        upper = _number(upper) if upper is not None else None
        if (lower is None) != (upper is None) or (lower is not None and lower > upper):
            raise ValueError('Invalid fitted prediction interval.')
        native = [r['provider_mm'] for r in rows if r['provider_mm'] is not None]
        covered = len(native)
        external = [r['ecmwf_mm'] for r in rows if r['ecmwf_mm'] is not None]
        result['horizons'].append(dict(
            hours=h, start=rows[0]['start'], end=rows[-1]['end'], issued_at=now.isoformat(),
            status='ready', quantity='snow_water_equivalent', unit='mm', method_id=model['family'],
            model_name='Our model', mean=projected, provider_name=PROVIDER,
            provider_mean=math.fsum(native) if covered == h else None, provider_hours=covered,
            ecmwf_mean=math.fsum(external) if len(external) == h else None, ecmwf_hours=len(external),
            depth_inches=math.fsum(depths[:h]) if depths[0] is not None else None,
            lower=lower, upper=upper, interval_label='80% validation residual range' if lower is not None else None,
            hourly_validated=True, target=TARGET.copy(), sources=sources,
            warnings=[] if covered == h else [f'Native HRRR covers the first {covered} of {h} hours. '
                                              'The model supplies the remaining forecast hours.'],
        ))
    return result


def _write_atomic(path, value):
    payload = json.dumps(value, allow_nan=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def refresh_mammoth_model(root=None, now=None):
    root = Path(root or ROOT)
    now = _utc(now or datetime.now(timezone.utc))
    try:
        result = _build(root, now)
        _write_atomic(root / SNAPSHOT, result)
    except (OSError, ImportError, ValueError, TypeError, KeyError) as exc:
        message = f'Mammoth hourly model could not update. {exc}'
        return dict(status='failed', message=message, errors=[message])
    return dict(status='succeeded', message='Mammoth hourly snow-water forecast updated.',
                issued_at=result['issued_at'], start=result['start'], end=result['end'],
                model=result['method_id'], native_hours=result['horizons'][-1]['provider_hours'],
                radar_input=result['sources']['radar']['used_as_input'])


def _check_amounts(payload, start):
    """Confirm each displayed total uses exactly its saved hourly series and window."""
    rows = payload['hourly']
    if len(rows) != 48 or [r['hours'] for r in payload['horizons']] != list(HORIZONS):
        raise ValueError('Incomplete model snapshot.')
    for i, row in enumerate(rows, 1):
        if (row['horizon_hours'] != i or _utc(row['start']) != start + (i-1)*HOUR
                or _utc(row['end']) != start + i*HOUR):
            raise ValueError('Noncontiguous forecast hours.')
    for horizon in payload['horizons']:
        h = horizon['hours']
        if (_utc(horizon['start']) != start or _utc(horizon['end']) != start+h*HOUR
                or horizon['quantity'] != 'snow_water_equivalent' or horizon['unit'] != 'mm'):
            raise ValueError('Forecast total uses a different window or quantity.')
        for hourly, total in [('mean_mm', 'mean'), ('provider_mm', 'provider_mean'),
                              ('ecmwf_mm', 'ecmwf_mean'), ('depth_inches', 'depth_inches')]:
            values = [row.get(hourly) for row in rows[:h]]
            value = horizon.get(total)
            if any(x is None for x in values):
                if value is not None or hourly == 'mean_mm':
                    raise ValueError('Amount reported for an incomplete forecast window.')
            elif value is None or not math.isclose(_number(value), math.fsum(map(_number, values)),
                                                   rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError('Forecast total disagrees with its hourly values.')


def read_deployed_mammoth(root=None, now=None):
    """Read the last issuance and age its original windows without making any network calls."""
    now = _utc(now or datetime.now(timezone.utc))
    path = Path(root or ROOT) / SNAPSHOT
    try:
        payload = json.loads(path.read_text())
        if payload['site_id'] != 'mammoth' or payload['quantity'] != 'snow_water_equivalent':
            raise ValueError('Unexpected model snapshot identity.')
        issue, start, end = (_utc(payload[key]) for key in ('issued_at', 'start', 'end'))
        if issue > now or issue > start or end != start + 48 * HOUR:
            raise ValueError('Forecast issue or valid time mismatch.')
        _check_amounts(payload, start)
        if end <= now:
            return _unavailable(f'The last Mammoth forecast ended {end.isoformat()}. Use Refresh all for a new forecast.')
        stale = now - issue > 3 * HOUR
        payload['read_at'] = now.isoformat()
        payload['stale'] = stale
        payload['window_started'] = start < now
        if stale or start < now:
            payload['status'] = 'cached'
            payload['warnings'].append('Showing the saved forecast for its original time window.')
        for row in payload['horizons']:
            row.update(stale=stale, window_started=start < now, expired=_utc(row['end']) <= now)
            if row['window_started'] or stale:
                row['status'] = 'cached'
        return payload
    except FileNotFoundError:
        return _unavailable('Use Refresh all to obtain the Mammoth hourly snow-water forecast.')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _unavailable(f'The Mammoth forecast could not be read. {exc}')
