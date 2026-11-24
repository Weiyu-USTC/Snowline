"""Current Collins Mid Mountain observations from Alta's public weather report."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import tempfile

import pandas as pd
import requests

from prepare_mountain_hourly import base_frame, finalize


URL = 'https://www.alta.com/weather'
CACHE_NAME = 'alta_collins_resort_current.json'
MAX_BYTES = 4_000_000
ZONE = 'America/Denver'


def _is_collins(location):
    return (isinstance(location, dict)
            and location.get('id') == 2
            and location.get('slug') == 'mid_mountain'
            and location.get('elevation') == 9662)


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _parse_report(body, now):
    """Use the published hourly temp and liquid water columns for Collins only."""
    text = body.decode('utf-8')
    marker = re.search(r'window\.Alta\s*=\s*', text)
    if not marker:
        raise ValueError('Alta did not publish its weather observation table')
    page, _ = json.JSONDecoder().raw_decode(text, marker.end())
    observation = page['weather']['observation']
    if not any(_is_collins(location) for location in observation['locations']):
        raise ValueError('Alta did not identify Collins Mid Mountain at 9,662 feet')
    rows = []
    for reading in observation['weather']:
        if reading.get('location_id') != 2 or not _is_collins(reading.get('location')):
            continue
        # Alta displays local wall time. An ambiguous or nonexistent local hour
        # cannot be mapped to a unique instant using the published row alone.
        stamp = pd.to_datetime(reading.get('dated_at'), format='%Y-%m-%d %H:%M:%S', errors='coerce')
        if pd.isna(stamp):
            continue
        stamp = stamp.tz_localize(ZONE, ambiguous='NaT', nonexistent='NaT')
        if pd.isna(stamp) or not now - pd.Timedelta(hours=26) <= stamp <= now:
            continue

        # Unit pairs appear together in the report. Mask unexpected pairs to
        # guard against changed units or missing raw readings encoded as zero.
        temp_f = _number(reading.get('temperature'))
        temp_c = _number(reading.get('temperature_as_celsius'))
        if temp_f is None or temp_c is None or abs((temp_f - 32) * 5 / 9 - temp_c) > .11:
            temp_c = None
        water_in = _number(reading.get('water'))
        water_cm = _number(reading.get('water_as_cm'))
        water_mm = None
        if water_in is not None and water_cm is not None and abs(water_in * 2.54 - water_cm) <= .011:
            water_mm = water_in * 25.4
        if temp_c is not None or water_mm is not None:
            rows.append((stamp.tz_convert('UTC'), temp_c, water_mm))

    if not rows:
        raise ValueError('Alta has no recent Collins hourly observations')
    values = pd.DataFrame(rows, columns=['valid_time_utc', 'temperature_c', 'precipitation_mm']).drop_duplicates()
    if values.valid_time_utc.duplicated().any():
        raise ValueError('Alta reported conflicting Collins observations for one hour')
    data = base_frame(values.index, 'alta', 'CLN')
    for col in values:
        data[col] = values[col]
    data['source_url'] = URL
    data['source_file'] = 'data/mountain_hourly/raw/' + CACHE_NAME
    data['clock_verified'] = True
    data['source_clock'] = 'Alta local report time, America/Denver; ambiguous transition hours excluded'
    data['quality_notes'] = ('Alta Collins Mid Mountain, 9,662 ft. Reported hourly air temperature and '
                             'liquid water; source provides no per-variable quality flags.')
    data['retrieved_utc'] = now
    result = finalize(data)
    result = result.dropna(subset=['temperature_c', 'precipitation_mm'], how='all')
    temps = result[result.temperature_c.notna()]
    if temps.empty or now - temps.valid_time_utc.max() > pd.Timedelta(hours=3):
        raise ValueError('Alta has no usable Collins temperature in the last three hours')
    return result


def fetch_collins_fallback(raw_dir, timeout=20, now=None):
    """Download a size-limited report and replace the prior cache only on success."""
    with requests.get(URL, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_content(65536):
            if len(body) + len(chunk) > MAX_BYTES:
                raise ValueError('Alta weather response exceeds the size limit')
            body.extend(chunk)
    received = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now).tz_convert('UTC')
    result = _parse_report(body, received)

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=raw_dir, prefix=CACHE_NAME + '.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(result.to_json(orient='records', date_format='iso'))
        temporary.replace(raw_dir / CACHE_NAME)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return result
