"""Read seasonal snowfall observations without inventing missing snowfall.

The GET path is local-only. Refresh downloads bounded public source files into
an independent cache, leaving training data and model selections untouched.
Alta's official reported season totals remain separate from the frozen manual
Collins study measurements; a report date is not an exact observation clock.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import http.cookiejar
import io
import json
import math
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
MAX_DOWNLOAD = 20_000_000
CACHE_HOURS = 1
ALTA_URL = 'https://www.alta.com/weather'
ALTA_HISTORY_URL = 'https://www.alta.com/ajax/snowfall-history'
CAIC_URL = 'https://stations.avalanche.state.co.us/tabular.php'
SITES = {
    'alta': dict(station_id='CLN', target_name='Alta–Collins manual snow-study plot',
        verified_fresh_snow=True, clock_verified=True, timezone='America/Denver',
        note='Historical manual Collins observations; current resort-reported totals are a separate source.'),
    'vail': dict(station_id='CAVMM', target_name='Vail Mid-Mountain Sno24 sensor proxy',
        verified_fresh_snow=False, clock_verified=False, timezone='America/Denver',
        note='Summed 04:00 Sno24 readings are a sensor proxy, not verified accumulated fresh snowfall; reset protocol and source clock are unverified.'),
    'mansfield': dict(station_id='US1VTLM0001', target_name='Stowe 0.2 SW town observer',
        verified_fresh_snow=True, clock_verified=True, timezone='America/New_York',
        note='NOAA/CoCoRaHS town observations at 223 m, not Stowe summit snowfall.'),
    'jay': dict(station_id='US1VTOL0001', target_name='Westfield 0.7 WNW town observer',
        verified_fresh_snow=True, clock_verified=True, timezone='America/New_York',
        note='NOAA/CoCoRaHS nearby town observations at 326 m, not Jay Peak summit snowfall.'),
}


def _number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else None
    except (TypeError, ValueError):
        return None


def _day(value):
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except (TypeError, ValueError):
        return None


def _iso(value):
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt.astimezone(UTC).isoformat() if dt.tzinfo else None
    except (TypeError, ValueError):
        return None


def _cache(root):
    return Path(root) / 'data/dashboard/seasonal/raw'


def _latest(paths):
    existing = [p for p in paths if p.is_file()]
    return max(existing, key=lambda p: (p.stat().st_mtime_ns, str(p))) if existing else None


def _noaa_paths(root, station):
    root = Path(root)
    return [* (root / 'data/neural48_training/current/station_history').glob(f'*/{station}.csv.gz'),
            root / f'data/neural48_training/source_probes/{station}.csv.gz',
            _cache(root) / f'{station}.csv.gz']


def parse_noaa(body, station):
    """SNOW is mm of new snow, not SWE; preserve flags and real report clocks."""
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
        unpacked = stream.read(80_000_001)
    if len(unpacked) > 80_000_000:
        raise ValueError('NOAA station payload exceeds decompressed size limit')
    records = {}
    for fields in csv.reader(io.StringIO(unpacked.decode('utf-8'))):
        if len(fields) < 8 or fields[2] != 'SNOW':
            continue
        if fields[0] != station:
            raise ValueError('NOAA station identifier mismatch')
        day = datetime.strptime(fields[1], '%Y%m%d').date().isoformat()
        if day in records:
            raise ValueError('Duplicate NOAA snowfall observation date')
        raw = _number(fields[3])
        mflag, qflag, clock = fields[4].strip(), fields[5].strip(), fields[7].strip()
        usable = raw is not None and not qflag and mflag not in {'P', 'A'}
        end = None
        if len(clock) == 4 and clock.isdigit():
            hour, minute = int(clock[:2]), int(clock[2:])
            if hour < 24 and minute < 60:
                end = datetime.combine(date.fromisoformat(day), datetime.min.time()).replace(
                    hour=hour, minute=minute, tzinfo=ZoneInfo('America/New_York')).astimezone(UTC).isoformat()
        records[day] = dict(date=day, snowfall_inches=raw / 25.4 if usable else None,
                            end=end, measurement_flag=mflag, quality_flag=qflag,
                            trace=mflag == 'T')
    if not records:
        raise ValueError('NOAA source has no daily SNOW records')
    return list(records.values())


def _alta_object(body):
    """Only parse the publicly embedded JSON, never execute page JavaScript."""
    text = body.decode('utf-8')
    match = re.search(r'window\.Alta\s*=\s*', text)
    if not match:
        raise ValueError('Alta page has no recognizable embedded weather data')
    obj, _ = json.JSONDecoder().raw_decode(text[match.end():])
    return obj


def _alta_rows(history, expected_season=None):
    if not isinstance(history, list) or not history:
        raise ValueError('Alta page has no reported snowfall history')
    if len(history) > 400:
        raise ValueError('Alta season history exceeds expected calendar size')
    records = []
    seen = set()
    for row in history:
        if not isinstance(row, dict):
            raise ValueError('Invalid Alta snowfall record')
        day, total = _day(row.get('reported_at')), _number(row.get('ytd_snow'))
        if not day or total is None:
            continue
        if day in seen:
            raise ValueError('Duplicate Alta report date')
        season = str(row.get('season', ''))
        if expected_season is not None:
            if season != str(expected_season):
                raise ValueError('Alta returned a different season than requested')
            start = int(str(expected_season)[:4])
            if not f'{start}-07-01' <= day < f'{start + 1}-07-01':
                raise ValueError('Alta report date lies outside its declared season')
        seen.add(day)
        records.append(dict(date=day, total_inches=total, snowfall_inches=_number(row.get('new_snow')),
                            end=None, season=season))
    if not records:
        raise ValueError('Alta page has no usable reported season totals')
    return sorted(records, key=lambda r: r['date'])


def parse_alta(body):
    """Official current totals are not derived by summing an incomplete series."""
    return _alta_rows(_alta_object(body).get('weather', {}).get('snowfall', {}).get('history'))


def parse_alta_history(body, expected_season):
    obj = json.loads(body)
    return _alta_rows(obj.get('snowfallHistory'), expected_season)


def _alta_source():
    return dict(name='Alta resort-reported season totals', url=ALTA_URL,
        station_id=None, target_name='Alta official snowfall report',
        unit='inches', units_evidence='Alta weather table labels 24 Hr New Snow and Cumulative Season Snow with inch marks.',
        note='Direct reported cumulative snowfall in inches; separate from the Collins research archive. The report date does not establish an exact observation cutoff.')


def _history_corrections(records):
    """Source revisions are visible, never silently forced to be monotonic."""
    return [b['date'] for a, b in zip(records, records[1:]) if b['total_inches'] < a['total_inches']]


def parse_caic(body, now=None):
    """Keep exact 04:00 source-clock Sno24 endpoints; never use snow depth."""
    from fetch_vail_caic import parse
    # The table parser requires year/month. Enumerate observed source years,
    # bounded to avoid arbitrary work on malformed HTML.
    text = body.decode('utf-8', errors='replace')
    years = sorted({int(y) for y in re.findall(r'\b(20\d{2})\s+[A-Z][a-z]{2}\s+\d', text)})
    if len(years) > 3:
        raise ValueError('CAIC current payload spans too many years')
    records = {}
    for year in years:
        for month in range(1, 13):
            rows, _ = parse(body, 'CAVMM', year, month, CAIC_URL, 'season-current.html')
            for row in rows:
                dt = datetime.fromisoformat(row['datetime_display'])
                if dt.hour != 4 or dt.minute != 0:
                    continue
                day = dt.date().isoformat()
                value = _number(row.get('Sno24'))
                if '2020-03-20' <= day <= '2020-05-31' or '2023-10-01' <= day <= '2023-11-28':
                    value = None
                # Do not promote the historical assumed MST mapping to verified clock evidence.
                record = dict(date=day, snowfall_inches=value, end=None)
                if day in records and records[day]['snowfall_inches'] != value:
                    record['snowfall_inches'] = None
                records[day] = record
    if not records:
        raise ValueError('CAIC returned no 04:00 Sno24 observations')
    return sorted(records.values(), key=lambda r: r['date'])


def _csv_records(root, site):
    path = Path(root) / f'data/neural48_training/targets/{site}_daily.csv'
    if not path.exists():
        path = Path(root) / 'data/neural48_training/targets/daily_targets.csv'
        if not path.exists():
            return []
    records = []
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('site') and row['site'] != site:
                continue
            day = _day(row.get('reporting_date'))
            if not day:
                continue
            valid = str(row.get('valid_snowfall', '')).lower() not in {'false', '0'}
            value = _number(row.get('snowfall_inches')) if valid else None
            if site == 'vail' and ('2020-03-20' <= day <= '2020-05-31' or '2023-10-01' <= day <= '2023-11-28'):
                value = None
            records.append(dict(date=day, snowfall_inches=value,
                end=_iso(row.get('period_end_utc')) if SITES[site]['clock_verified'] else None))
    return records


def read_observations(site, root=None):
    """Read-only, network-free snapshot. Calendar gaps remain the caller's responsibility."""
    if site not in SITES:
        raise ValueError('Unknown mountain')
    root = Path(root or ROOT)
    cfg = SITES[site]
    merged = {r['date']: r for r in _csv_records(root, site)}
    notes = [cfg['note'], 'Missing observations are not zero snowfall; incomplete sums are not complete season totals.']
    sources = []
    reported = []
    reported_history = []
    report_source = None
    if site in {'mansfield', 'jay'}:
        station = cfg['station_id']
        sources.append(dict(name='NOAA GHCN-Daily / CoCoRaHS',
            url=f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station}.csv.gz'))
        paths = sorted((p for p in _noaa_paths(root, station) if p.exists()),
                       key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
        for path in paths:
            try:
                fresh = parse_noaa(path.read_bytes(), station)
                merged.update((r['date'], r) for r in fresh)
                break
            except (OSError, ValueError, EOFError) as exc:
                notes.append(f'A saved NOAA source could not be read: {exc}')
    elif site == 'vail':
        sources.append(dict(name='CAIC / Vail Resort CAVMM', url='https://stations.avalanche.state.co.us/precip.php'))
        paths = [* (root / 'data/neural-live').glob('*/CAVMM.html'), _cache(root) / 'CAVMM.html']
        for path in sorted((p for p in paths if p.exists()), key=lambda p: p.stat().st_mtime_ns):
            try:
                merged.update((r['date'], r) for r in parse_caic(path.read_bytes()))
            except (OSError, ValueError) as exc:
                notes.append(f'A saved Vail source could not be read: {exc}')
    else:
        sources.extend([dict(name='University of Utah Hive / Collins manual study', url='https://hive.utah.edu/records/280qy-4ee96'),
                        dict(name='Alta official snowfall report', url=ALTA_URL)])
        path = _cache(root) / 'alta_weather.html'
        if path.exists():
            try:
                reported = parse_alta(path.read_bytes())
            except (OSError, ValueError) as exc:
                notes.append(f'The saved Alta official source could not be read: {exc}')
        report_source = _alta_source()
        notes.append('Alta official totals are shown separately; no missing early-season daily values are reconstructed from total differences.')
        for history_path in sorted(_cache(root).glob('alta_history_*.json'), reverse=True):
            code = history_path.stem.removeprefix('alta_history_')
            if not re.fullmatch(r'20\d{2}20\d{2}', code):
                continue
            try:
                rows = parse_alta_history(history_path.read_bytes(), code)
                corrections = _history_corrections(rows)
                reported_history.append(dict(season=f'{code[:4]}/{code[-2:]}', reported_totals=rows,
                    source=dict(report_source), correction_dates=corrections))
                if corrections:
                    notes.append(f'Alta {code[:4]}/{code[-2:]} includes downward source revisions on {", ".join(corrections)}; totals are retained as reported.')
            except (OSError, ValueError) as exc:
                notes.append(f'A saved Alta historical season could not be read: {exc}')
        status_path = _cache(root) / 'alta_history_status.json'
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
                if status.get('errors'):
                    notes.append('Some official Alta past seasons could not be downloaded; available saved history is shown.')
            except (OSError, ValueError):
                pass
    records = sorted(merged.values(), key=lambda r: r['date'])
    valid = [r for r in records if r['snowfall_inches'] is not None]
    return dict(site_id=site, records=records, target=dict(cfg), sources=sources, notes=notes,
                latest_observation_date=valid[-1]['date'] if valid else None,
                reported_totals=reported, reported_total_source=report_source,
                reported_total_history=reported_history)


def _atomic(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as tmp:
        temp = Path(tmp.name)
        tmp.write(body)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _download(url):
    request = Request(url, headers={'User-Agent': 'Snowline seasonal observation chart; public data'})
    with urlopen(request, timeout=40) as response:
        body = response.read(MAX_DOWNLOAD + 1)
    if len(body) > MAX_DOWNLOAD:
        raise ValueError('Observation source exceeds bounded download size')
    return body


def _history_timeout(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Optional Alta history refresh reached its time budget')
    return min(10, remaining)


def _response_bytes(response):
    body = response.read(MAX_DOWNLOAD + 1)
    if len(body) > MAX_DOWNLOAD:
        raise ValueError('Alta history exceeds bounded download size')
    return body


def _refresh_alta_history(root, now, deadline):
    """Latest five prior seasons, cached once; anonymous public form workflow."""
    local = now.astimezone(ZoneInfo('America/Denver'))
    current_start = local.year - int(local.month < 10)
    codes = [f'{year}{year + 1}' for year in range(current_start - 1, current_start - 6, -1)]
    rawdir = _cache(root)
    missing, available, errors = [], [], []
    for code in codes:
        path = rawdir / f'alta_history_{code}.json'
        if path.exists():
            try:
                parse_alta_history(path.read_bytes(), code)
                available.append(code)
                continue
            except (OSError, ValueError):
                pass
        missing.append(code)
    if not missing:
        return dict(status='cached', seasons=available, errors=[])

    try:
        # The public form uses a normal anonymous CSRF session. No user browser,
        # saved account credential, or private cookie store is accessed.
        opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
        with opener.open(Request(ALTA_URL, headers={'User-Agent': 'Snowline seasonal observations'}),
                         timeout=_history_timeout(deadline)) as response:
            text = _response_bytes(response).decode('utf-8')
        token = re.search(r'<meta\s+name=["\']csrf-token["\']\s+content=["\']([^"\']+)', text)
        if not token:
            raise ValueError('Alta public history form token is unavailable')
    except Exception as exc:
        errors.append(dict(source='alta_history', message=str(exc)))
    else:
        for code in missing:
            try:
                request = Request(ALTA_HISTORY_URL, data=json.dumps(dict(season=int(code))).encode(),
                    headers={'Content-Type': 'application/json', 'Accept': 'application/json',
                        'X-CSRF-TOKEN': token.group(1), 'Referer': ALTA_URL,
                        'User-Agent': 'Snowline seasonal observations'})
                with opener.open(request, timeout=_history_timeout(deadline)) as response:
                    body = _response_bytes(response)
                rows = parse_alta_history(body, code)
                path = rawdir / f'alta_history_{code}.json'
                _atomic(path, body)
                _atomic(path.with_suffix('.download.json'), json.dumps(dict(url=ALTA_HISTORY_URL,
                    request_season=code, retrieved_at_utc=now.astimezone(UTC).isoformat(),
                    bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), records=len(rows)), indent=2).encode())
                available.append(code)
            except Exception as exc:
                errors.append(dict(source=f'alta_history_{code}', message=str(exc)))
                if time.monotonic() >= deadline:
                    break
    status = dict(status='partial' if available and errors else 'failed' if errors else 'succeeded',
                  seasons=available, errors=errors)
    _atomic(rawdir / 'alta_history_status.json', json.dumps(status, indent=2).encode())
    return status


def refresh_observations(root=None, now=None):
    """Refresh all four sources with at most two concurrent bounded downloads."""
    root = Path(root or ROOT)
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('Refresh time must include timezone')
    rawdir = _cache(root)
    deadline = time.monotonic() + 95

    def one(site):
        cfg = SITES[site]
        if site in {'mansfield', 'jay'}:
            station = cfg['station_id']
            path = rawdir / f'{station}.csv.gz'
            cached = _latest(_noaa_paths(root, station))
            url = f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station}.csv.gz'
            parser = lambda body: parse_noaa(body, station)
        elif site == 'alta':
            path = rawdir / 'alta_weather.html'
            cached, url, parser = path if path.exists() else None, ALTA_URL, parse_alta
        else:
            path = rawdir / 'CAVMM.html'
            cached = path if path.exists() else None
            local = now.astimezone(ZoneInfo('America/Denver'))
            start = date(local.year - int(local.month < 10), 10, 1)
            hours = min(9000, int((local.date() - start).days + 2) * 24)
            url = CAIC_URL + '?' + urlencode(dict(st='CAVMM', date=local.strftime('%Y-%m-%d %H'),
                                                   unit='e', area='caic', range=hours))
            parser = parse_caic
        try:
            if cached and 0 <= now.timestamp() - cached.stat().st_mtime <= CACHE_HOURS * 3600:
                # Validate fresh cached bytes before treating them as a successful refresh.
                try:
                    parser(cached.read_bytes())
                    return site, dict(status='cached', message='Recent observation source reused.', url=url)
                except (OSError, ValueError, EOFError):
                    pass
            body = _download(url)
            parsed = parser(body)
            _atomic(path, body)
            _atomic(path.with_suffix(path.suffix + '.json'), json.dumps(dict(url=url,
                retrieved_at_utc=now.astimezone(UTC).isoformat(), bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(), records=len(parsed)), indent=2).encode())
            return site, dict(status='succeeded', message='Public observation source updated.', url=url)
        except Exception as exc:
            return site, dict(status='failed', message=str(exc), url=url)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='season-observations') as pool:
        sites = dict(pool.map(one, SITES))
    # Optional archive failures do not misreport a successful current source fetch.
    # Both current feeds and optional history share this bounded refresh budget.
    history = None
    if sites['alta']['status'] != 'failed':
        history = _refresh_alta_history(root, now, deadline)
        sites['alta']['history'] = history
    errors = [dict(source=f'season_{site}', message=result['message'])
              for site, result in sites.items() if result['status'] == 'failed']
    status = 'failed' if len(errors) == len(SITES) else 'partial' if errors else 'succeeded'
    return dict(status=status, sites=sites, errors=errors,
                warnings=(history or {}).get('errors', []),
                message='Seasonal observations updated.' if not errors else 'Some seasonal observation sources could not refresh; saved observations remain available.')
