"""Expose saved, timestamped 24/48-hour neural forecasts to Snowline.

The dashboard never stretches daily targets into hourly snowfall observations.
This adapter reads a local snapshot; refreshing uses the separately trained
model and never starts optimizer updates.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'outputs/nwp-weather/neural48-trained/current_forecast.json'
REPORT = '/reports/weather/neural48-trained/report.html'


def _date(value):
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Forecast timestamps must include a timezone')
    return result.astimezone(timezone.utc)


def read_neural48(site, now=None, path=None):
    now = now or datetime.now(timezone.utc)
    unavailable = {'site_id': site, 'status': 'unavailable', 'hourly_validated': False,
                   'report_url': REPORT, 'warnings': ['Refresh to check the 48-hour neural forecast.']}
    try:
        snapshot = json.loads(Path(path or SNAPSHOT).read_text())
        raw = snapshot.get('sites', {}).get(site)
        if not isinstance(raw, dict):
            return unavailable
        result = deepcopy(raw)
        result['generated_at'] = snapshot.get('generated_at')
        result['report_url'] = REPORT
        result['hourly_validated'] = False
        if result.get('status') == 'unavailable':
            return result
        if result.get('site_id') != site:
            raise ValueError('Forecast mountain identity mismatch')
        start, middle, end = (_date(result[key]) for key in ['start', 'midpoint', 'end'])
        if (middle-start).total_seconds() != 86400 or (end-middle).total_seconds() != 86400:
            raise ValueError('Forecast must contain two exact 24-hour windows')
        totals = result.get('totals', [])
        if [row.get('hours') for row in totals] != [24, 48]:
            raise ValueError('Missing 24/48-hour forecast totals')
        for row, endpoint in zip(totals, [middle, end]):
            if _date(row['end']) != endpoint:
                raise ValueError('Forecast total and observation window disagree')
            for key in ['mean_inches', 'q10_inches', 'q90_inches', 'provider_inches']:
                value = row.get(key)
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                    raise ValueError('Invalid snowfall amount')
            if row.get('mean_inches') is None:
                raise ValueError('Neural mean is missing')
            if row.get('q10_inches') is not None and row.get('q90_inches') is not None and row['q10_inches'] > row['q90_inches']:
                raise ValueError('Prediction range is reversed')
        generated = _date(snapshot['generated_at'])
        issued = _date(result['issued_at'])
        if issued > start:
            raise ValueError('Forecast begins before its issue time')
        result['expired'] = end <= now
        result['window_started'] = start < now
        result['stale'] = (now-issued).total_seconds() > 3*3600 or (now-generated).total_seconds() > 6*3600
        result['fully_future'] = start >= now
        return result
    except FileNotFoundError:
        return unavailable
    except (ValueError, TypeError, KeyError, OSError) as exc:
        unavailable['warnings'] = [f'48-hour neural forecast unavailable: {exc}']
        return unavailable


def refresh_neural48(progress=None):
    from forecast_neural48 import refresh_all
    return refresh_all(progress=progress, force=False)
