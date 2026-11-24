"""Main Lodge monthly snowfall and independent 15-minute Sesame station readings."""
from __future__ import annotations

import calendar
from contextlib import closing
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from http.client import HTTPException
import json
import math
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

UTC = timezone.utc
PACIFIC = ZoneInfo('America/Los_Angeles')
MAIN_LODGE_URL = 'https://www.mammothmountain.com/on-the-mountain/snowfall-history'
SESAME_URL = 'https://mammothmountain.westernweathergroup.com/709eb85c1d064595a6c64fe3bf9f5d73'
HEADERS = ['Season', 'Total', 'Pre-Oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar',
           'Apr', 'May', 'Jun', 'Jul']
MONTHS = (10, 11, 12, 1, 2, 3, 4, 5, 6, 7)
TARGET = dict(station_id='mammoth-main-lodge', target_name='Mammoth Mountain Main Lodge',
              verified_fresh_snow=True, clock_verified=False, timezone=PACIFIC.key,
              note='Main Lodge reports snowfall by month and season in inches.')
SOURCE = dict(name='Mammoth Mountain Main Lodge snowfall history', url=MAIN_LODGE_URL,
              target_name=TARGET['target_name'], station_id=TARGET['station_id'],
              unit='inches', cadence='monthly',
              note='Month-end points accumulate published monthly snowfall, including pre-October snow. '
                   'Within a month, points show the season total visible on dates Snowline checked. '
                   'The resort does not supply an exact observation time.')
SESAME = dict(name='Sesame snow study site', station_id='MM-SES', elevation_ft=9014,
              url=SESAME_URL, cadence_minutes=15,
              note='HN is a snow-board reading with an unverified clearing time. HS is settled snow depth.')
FIELDS = {
    'snow_board_inches': 'New Snow (HN) (in)', 'snow_depth_inches': 'Snow Depth (HS) (in)',
    'temperature_f': 'Temp (°F)', 'wind_mph': 'Wind Spd (mph)', 'wind_gust_mph': 'Wind Gust (mph)',
    'wind_direction_deg': 'Wind Dir (°)',
}


class _Tables(HTMLParser):
    """Extract table cells without interpreting scripts or styling."""
    def __init__(self, body):
        super().__init__()
        self.tables = []
        self.table = self.row = self.cell = None
        self.feed(body.decode('utf-8'))
        self.close()

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.table = []
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('th', 'td') and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def _number(text):
    if text in ('', '-', '—', 'N/A'):
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError('Non-finite measurement in source table')
    return value


def parse_main_lodge(body):
    if b'Snow totals as reported from Main Lodge' not in body:
        raise ValueError('Main Lodge attribution unavailable on snowfall history page')
    seasons = {}
    for table in _Tables(body).tables:
        if not table or table[0] != HEADERS:
            continue
        for row in table[1:]:
            match = re.fullmatch(r'(\d{4})-(\d{1,2})', row[0] if row else '')
            if not match:
                continue
            year = int(match[1])
            if int(match[2]) != (year + 1) % 100 or len(row) != len(HEADERS):
                raise ValueError('Unrecognized Main Lodge season row')
            values = list(map(_number, row[1:]))
            if any(value is not None and value < 0 for value in values):
                raise ValueError('Negative Main Lodge snowfall')
            season = dict(total=values[0], pre_oct=values[1], months=values[2:])
            key = str(year)
            if key in seasons and seasons[key] != season:
                raise ValueError('Conflicting Main Lodge season rows')
            seasons[key] = season
    if not seasons or not any(s['total'] is not None for s in seasons.values()):
        raise ValueError('Main Lodge snowfall history table unavailable')
    return seasons


def _directory(root):
    return Path(root) / 'data/dashboard/seasonal/raw/mammoth'


def _snapshots(year, season, as_of):
    total = season['pre_oct']
    points = {}
    for month, amount in zip(MONTHS, season['months']):
        y = year + (month < 10)
        last = date(y, month, calendar.monthrange(y, month)[1])
        if (y, month) > (as_of.year, as_of.month):
            break
        if (y, month) == (as_of.year, as_of.month):
            if season['total'] is not None and amount is not None:
                points[as_of] = season['total']
            break
        # A missing month breaks the cumulative sum, including later month-ends.
        total = None if total is None or amount is None else round(total + amount, 4)
        if total is not None:
            points[last] = total
    return [dict(date=day.isoformat(), total_inches=value, end=None) for day, value in points.items()]


def read_mammoth_observations(root):
    directory = _directory(root)
    current, history, notes = [], [], []
    try:
        checks = {}
        archive = directory / 'main_lodge.sqlite3'
        if archive.is_file():
            with closing(sqlite3.connect(archive.resolve().as_uri() + '?mode=ro', uri=True)) as db, db:
                db.execute('BEGIN')
                saved = _saved_lodge(directory, db)
                if saved is not None:
                    for year, day, total in db.execute('SELECT year, day, total FROM checks'):
                        checks.setdefault(str(year), {})[day] = dict(date=day, total_inches=total, end=None)
        else:
            saved = _saved_lodge(directory)
        if saved is None:
            raise FileNotFoundError
        as_of = datetime.fromisoformat(saved['retrieved_at_utc']).astimezone(PACIFIC).date()
        current_year = as_of.year - (as_of.month < 10)
        for key in sorted(set(saved['seasons']) | set(checks)):
            year = int(key)
            dated = checks.get(key, {})
            if key in saved['seasons']:
                # The latest published monthly figures supersede any earlier check on that month-end.
                for row in _snapshots(year, saved['seasons'][key], as_of):
                    dated[row['date']] = row
            points = [dated[day] for day in sorted(dated)]
            if year == current_year:
                current = points
            if points:
                # Include the last downloaded season here too for offline season rollover.
                history.append(dict(season=f'{year}/{str(year + 1)[-2:]}',
                                    reported_totals=points, source=SOURCE.copy()))
    except FileNotFoundError:
        notes.append('Refresh all to download the Main Lodge snowfall record.')
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error):
        current, history = [], []
        notes.append('The saved Main Lodge snowfall record could not be read.')
    return dict(site_id='mammoth', target=TARGET.copy(), sources=[dict(name=SOURCE['name'], url=MAIN_LODGE_URL)],
                records=[], notes=notes, latest_observation_date=current[-1]['date'] if current else None,
                reported_totals=current, reported_total_source=SOURCE.copy(), reported_total_history=history)


def parse_sesame(body):
    if b'Sesame - 9,014ft' not in body or b'America/Los_Angeles' not in body:
        raise ValueError('Sesame station identity or time zone unavailable')
    for table in _Tables(body).tables:
        if not table or not table[0] or table[0][0] != 'Date & Time':
            continue
        headers = table[0]
        if not all(label in headers for label in FIELDS.values()):
            continue
        rows = {}
        for row in table[1:]:
            if not row or not re.fullmatch(r'\d\d/\d\d/\d\d - \d{4}', row[0]):
                continue
            local = datetime.strptime(row[0], '%m/%d/%y - %H%M').replace(tzinfo=PACIFIC)
            # The table omits offsets. Repeated or nonexistent wall times have no unique UTC instant.
            if local.utcoffset() != local.replace(fold=1).utcoffset():
                continue
            stamp = local.astimezone(UTC).isoformat()
            if len(row) != len(headers):
                raise ValueError('Unexpected Sesame observation column count')
            obs = {}
            for name, label in FIELDS.items():
                raw = row[headers.index(label)].split('|')[-1].strip()
                value = _number(raw)
                if name != 'temperature_f' and value is not None and value < 0:
                    value = None
                if name == 'wind_direction_deg' and value is not None and value > 360:
                    value = None
                obs[name] = value
            if any(value is not None for value in obs.values()):
                if stamp in rows and obs != rows[stamp]:
                    raise ValueError('Conflicting Sesame observation time')
                rows[stamp] = obs
        if rows:
            return rows
    raise ValueError('Sesame 15-minute telemetry table unavailable')


def _download(url):
    from season_observations import _download as download
    return download(url)


def _cache_recent(retrieved, now):
    return 0 <= (now - datetime.fromisoformat(retrieved)).total_seconds() < 3600


def _remember_lodge_check(db, saved):
    stamp = saved['retrieved_at_utc']
    local_day = datetime.fromisoformat(stamp).astimezone(PACIFIC).date()
    year = local_day.year - (local_day.month < 10)
    season = saved['seasons'].get(str(year))
    if (season and season['total'] is not None and local_day.month in MONTHS
            and season['months'][MONTHS.index(local_day.month)] is not None):
        # Only the latest lookup on each Pacific calendar date is retained.
        db.execute('INSERT INTO checks VALUES (?,?,?,?) ON CONFLICT(day) DO UPDATE '
                   'SET total=excluded.total, retrieved=excluded.retrieved '
                   'WHERE excluded.retrieved >= checks.retrieved',
                   (year, local_day.isoformat(), season['total'], stamp))


def _saved_lodge(directory, db=None):
    if db is not None:
        row = db.execute('SELECT payload FROM latest WHERE id=1').fetchone()
        if row:
            return json.loads(row[0])
    # Original releases kept the latest checked season in a single JSON file.
    try:
        saved = json.loads((directory / 'main_lodge.json').read_text())
        if isinstance(saved['seasons'], dict) and datetime.fromisoformat(saved['retrieved_at_utc']).tzinfo:
            return saved
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        pass
    return None


def _refresh_lodge(directory, now):
    directory.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(directory / 'main_lodge.sqlite3')) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS latest (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS checks (year INTEGER, day TEXT PRIMARY KEY, '
                   'total REAL NOT NULL, retrieved TEXT NOT NULL)')
        cached = _saved_lodge(directory, db)
        if cached and cached['seasons'] and _cache_recent(cached['retrieved_at_utc'], now):
            if not db.execute('SELECT 1 FROM latest WHERE id=1').fetchone():
                _remember_lodge_check(db, cached)
                db.execute('INSERT OR IGNORE INTO latest VALUES (1, ?)', (json.dumps(cached),))
            return dict(status='cached', retrieved_at_utc=cached['retrieved_at_utc'],
                        seasons=len(cached['seasons']))
        seasons = parse_main_lodge(_download(MAIN_LODGE_URL))
        saved = dict(url=MAIN_LODGE_URL, retrieved_at_utc=now.isoformat(), seasons=seasons)
        db.execute('BEGIN IMMEDIATE')
        cached = _saved_lodge(directory, db)
        if cached:
            _remember_lodge_check(db, cached)
        _remember_lodge_check(db, saved)
        if not cached or datetime.fromisoformat(cached['retrieved_at_utc']) <= now:
            db.execute('INSERT OR REPLACE INTO latest VALUES (1, ?)', (json.dumps(saved),))
    return dict(status='succeeded', retrieved_at_utc=now.isoformat(), seasons=len(seasons))


def _refresh_sesame(directory, now):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'sesame.sqlite3'
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS observations (utc TEXT PRIMARY KEY, readings TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS refresh (id INTEGER PRIMARY KEY, retrieved TEXT NOT NULL)')
        row = db.execute('SELECT retrieved FROM refresh WHERE id=1').fetchone()
        if row and _cache_recent(row[0], now):
            return dict(status='cached', retrieved_at_utc=row[0])
        rows = parse_sesame(_download(SESAME_URL))
        if datetime.fromisoformat(max(rows)) > now:
            raise ValueError('Sesame returned a future observation')
        db.executemany('INSERT OR REPLACE INTO observations VALUES (?, ?)',
                       ((utc, json.dumps(readings)) for utc, readings in rows.items()))
        db.execute('INSERT OR REPLACE INTO refresh VALUES (1, ?)', (now.isoformat(),))
    return dict(status='succeeded', retrieved_at_utc=now.isoformat(), samples_fetched=len(rows))


def read_sesame_observation(root, now=None):
    now = now or datetime.now(UTC)
    path = _directory(root) / 'sesame.sqlite3'
    if path.is_file():
        try:
            with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db, db:
                row = db.execute('SELECT utc, readings FROM observations WHERE utc <= ? '
                                 'ORDER BY utc DESC LIMIT 1', (now.astimezone(UTC).isoformat(),)).fetchone()
            if row:
                age = (now - datetime.fromisoformat(row[0])).total_seconds()
                return dict(SESAME, status='ready' if age < 3 * 3600 else 'stale',
                            observed_at_utc=row[0], **json.loads(row[1]))
        except (sqlite3.Error, ValueError, OSError):
            pass
    return dict(SESAME, status='unavailable', observed_at_utc=None)


def refresh_mammoth_observations(root, now=None):
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('Refresh time must include timezone')
    now = now.astimezone(UTC)
    results = {}
    for key, url, refresh in [('main_lodge', MAIN_LODGE_URL, _refresh_lodge),
                              ('sesame', SESAME_URL, _refresh_sesame)]:
        try:
            results[key] = dict(refresh(_directory(root), now), url=url)
            if key == 'sesame':
                latest = read_sesame_observation(root, now)
                results[key]['observed_at_utc'] = latest['observed_at_utc']
                if latest['status'] == 'stale':
                    results[key].update(status='partial', message='Latest Sesame reading is over three hours old.')
                elif latest['status'] == 'unavailable':
                    results[key].update(status='failed', message='Saved Sesame readings unavailable.')
        except (ValueError, OSError, EOFError, HTTPException, sqlite3.Error) as exc:
            results[key] = dict(status='failed', url=url, message=str(exc))
    states = {r['status'] for r in results.values()}
    issues = [key + ': ' + result['message'] for key, result in results.items() if 'message' in result]
    status = 'failed' if states == {'failed'} else 'partial' if issues else 'succeeded'
    return dict(status=status, sources=results, message='; '.join(issues) if issues
                else 'Main Lodge snowfall and Sesame weather observations available.')
