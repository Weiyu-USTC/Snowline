"""Read-only seasonal snowfall display, with explicit gaps and target provenance.

Season curves use October 1–May 31 reporting dates. An incomplete daily archive
can supply a *reported-only lower bound*, never an observed full-season total.
A forecast continuation requires a known total at the exact forecast start;
neither missing snowfall nor an unobserved clock gap is silently set to zero.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from snowline_sites import SITES

ROOT = Path(__file__).resolve().parents[1]

ZONES = {key: site['timezone'] for key, site in SITES.items()}
UTC = timezone.utc
MAX_HISTORY = 25


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _time(value):
    if not value:
        return None
    try:
        value = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return None
    return value.astimezone(UTC) if value.tzinfo is not None else None


def _date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _dates(start, end):
    for offset in range(max(0, (end - start).days + 1)):
        yield start + timedelta(days=offset)


def _season_year(day):
    return day.year if day.month >= 10 else day.year - 1


def _label(year):
    return f'{year}/{str(year + 1)[-2:]}'


def _observations(site, root):
    if site == 'mammoth':
        from mammoth_observations import read_mammoth_observations
        return read_mammoth_observations(root)
    from season_observations import read_observations
    return read_observations(site, root=root)


def _daily_records(records, today, now):
    """Keep future observations out; ambiguous duplicates become missing."""
    result = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        day = _date(row.get('date'))
        end = _time(row.get('end'))
        if day is None or day > today or (end is not None and end > now):
            continue
        value = _number(row.get('snowfall_inches'))
        record = {'amount': value, 'end': end}
        if day in result and result[day] != record:
            record = {'amount': None, 'end': None}
        result[day] = record
    return result


def _daily_curve(records, start, end):
    total, count, latest, latest_end = 0., 0, None, None
    points = []
    for day in _dates(start, end):
        row = records.get(day, {})
        amount = row.get('amount')
        if amount is None:
            points.append({'date': day.isoformat(), 'cumulative_inches': None})
            continue
        total += amount
        count += 1
        latest, latest_end = day, row.get('end')
        point = {'date': day.isoformat(), 'cumulative_inches': total}
        if latest_end is not None:
            point['time'] = latest_end.isoformat()
        points.append(point)
    expected = max(0, (end - start).days + 1)
    complete = count == expected and expected > 0
    # A complete observed prefix can end before today's report is available.
    # Its measured total is valid *as of that endpoint*, never at today's clock.
    prefix_expected = (latest - start).days + 1 if latest is not None else 0
    prefix_complete = count == prefix_expected and count > 0
    return dict(points=points, complete=complete,
                complete_through_observed=prefix_complete,
                kind='daily_sum', valid_days=count, expected_days=expected,
                recorded_total_inches=total if count else None,
                total_inches=total if prefix_complete else None,
                total_is_lower_bound=not prefix_complete,
                observed_through=latest.isoformat() if latest else None,
                observed_end=latest_end.isoformat() if latest_end else None,
                coverage_label=(f'{count} of {expected} reporting days observed. '
                    + ('Complete record.' if complete else
                       'Recorded snowfall only; gaps are not counted as zero.')))


def _reported_curve(rows, start, end, now, source):
    """Direct dated season totals need not be reconstructed from daily values."""
    snapshots = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        day, amount = _date(row.get('date')), _number(row.get('total_inches'))
        stamp = _time(row.get('end'))
        if day is None or not start <= day <= end or amount is None or (stamp and stamp > now):
            continue
        record = {'amount': amount, 'end': stamp}
        if day in snapshots and snapshots[day] != record:
            record = {'amount': None, 'end': None}
        snapshots[day] = record
    valid = {day: row for day, row in snapshots.items() if row['amount'] is not None}
    if not valid:
        return None
    latest = max(valid)
    points = [{'date': day.isoformat(),
               'cumulative_inches': snapshots.get(day, {}).get('amount')}
              for day in _dates(start, end)]
    for point in points:
        stamp = snapshots.get(_date(point['date']), {}).get('end')
        if point['cumulative_inches'] is not None and stamp is not None:
            point['time'] = stamp.isoformat()
    expected = max(0, (end - start).days + 1)
    stamp = valid[latest]['end']
    return dict(points=points, complete=len(valid) == expected,
                complete_through_observed=True, kind='reported_season_total',
                valid_days=len(valid), expected_days=expected,
                recorded_total_inches=valid[latest]['amount'],
                total_inches=valid[latest]['amount'], total_is_lower_bound=False,
                observed_through=latest.isoformat(),
                observed_end=stamp.isoformat() if stamp else None,
                source=source,
                coverage_label=(f'{len(valid)} dated season-total reports. '
                    'Direct reported totals; dates without reports remain gaps.'))


def _unavailable(note):
    return dict(status='unavailable', label='Our model', points=[],
                note=note, interval_label=None)


def _saved_issued_forecast(site, root):
    """Recover one already-issued selected trajectory, without moving its clock.

    The live comparison adapter intentionally rejects windows that have begun.
    A seasonal plot may still show their unexpired predicted trajectory, provided
    the original start coincides exactly with the observed cumulative baseline.
    """
    try:
        selection = json.loads((root / 'outputs/nwp-weather/forecast-portfolio/deployment_selection.json').read_text())
        choice = selection['sites'][site]['48']
        if choice.get('quantity') != 'snowfall' or choice.get('unit') != 'inches':
            return None
        expected = choice['family']
        snapshot = json.loads((root / 'outputs/nwp-weather/neural48-trained/current_forecast.json').read_text())
        item = snapshot['sites'][site]
        if item.get('site_id') != site or item.get('status') == 'unavailable':
            return None
        selected = item.get('deployed_48')
        if selected is None:
            selected = next(row for row in item.get('totals', []) if row.get('hours') == 48)
        if selected.get('hours') != 48 or selected.get('status') == 'unavailable' or \
                selected.get('family', item.get('family')) != expected:
            return None
        record = dict(hours=48, status='ready', quantity='snowfall', unit='inches',
                      target=item.get('target', {}), method_id='neural48_' + expected,
                      start=selected.get('start', item.get('start')),
                      end=selected.get('end'), issued_at=selected.get('issued_at', item.get('issued_at')),
                      mean=selected.get('mean_inches'), lower=selected.get('q10_inches'),
                      upper=selected.get('q90_inches'), retained_issued_forecast=True,
                      interval_label=selected.get('interval_label', 'Uncalibrated 80% model prediction range'))
        # Recovery is deliberately stricter than general live fallbacks: bounds
        # and the selected family must be present, rather than reconstructed.
        lo, hi = _number(record['lower']), _number(record['upper'])
        if _number(record['mean']) is None or lo is None or hi is None or lo > hi:
            return None
        return dict(site_id=site, target=item.get('target', {}), horizons=[record])
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None


def _continuation(site, current, target, now, season_end, forecast, root):
    if site == 'mammoth':
        return _unavailable('Main Lodge reports accumulated snow depth. The Jeff Dozier hourly model predicts '
                            'water contained in falling snow at a different station. The two curves use different units.')
    baseline = current.get('total_inches')
    if baseline is None:
        return _unavailable('A continuous forecast needs a known season total. '
                            'Missing observations prevent that total from being established.')
    anchor = _time(current.get('observed_end'))
    if anchor is None:
        return _unavailable('The latest reported season total has no verified observation cutoff time; '
                            'it cannot yet be joined to a forecast window.')
    if forecast is None:
        try:
            from dashboard_portfolio import read_deployed
            forecast = read_deployed(site, now=now, root=root)
        except Exception:
            forecast = {}
        ready_48 = next((row for row in forecast.get('horizons', [])
                         if row.get('hours') == 48 and row.get('status') in {'ready', 'fallback'}), None)
        if ready_48 is None:
            forecast = _saved_issued_forecast(site, root) or forecast
    if not isinstance(forecast, dict):
        return _unavailable('The selected forecast is unavailable.')
    forecast = forecast.get('deployed', forecast)
    if forecast.get('site_id') != site:
        return _unavailable('The forecast mountain does not match the observation record.')
    selected = next((row for row in forecast.get('horizons', [])
                     if isinstance(row, dict) and row.get('hours') == 48), {})
    if selected.get('status') not in {'ready', 'fallback'}:
        return _unavailable('A current 48-hour model forecast is unavailable. Refresh all to retry.')
    if selected.get('quantity') != 'snowfall' or selected.get('unit') != 'inches':
        return _unavailable('The forecast quantity does not match snowfall depth.')
    model_target = selected.get('target') or forecast.get('target') or {}
    source_target = current.get('source') if current.get('kind') == 'reported_season_total' else target
    source_target = source_target or {}
    station = source_target.get('station_id')
    if not station or model_target.get('station_id') != station:
        return _unavailable('The current observed total and model use different or unverified observation '
                            'targets. Their totals are not joined.')
    start, end, issued = (_time(selected.get(k)) for k in ('start', 'end', 'issued_at'))
    if not all((start, end, issued)) or end - start != timedelta(hours=48):
        return _unavailable('The model does not supply a valid 48-hour accumulation window.')
    if selected.get('stale') or selected.get('expired') or end <= now or \
            issued > now or issued > start or start - issued > timedelta(hours=3):
        return _unavailable('The saved model window is stale, expired, or was not issued at a valid time. '
                            'Refresh all to retry.')
    if start != anchor:
        return _unavailable('The latest observation ends before or after the model forecast starts. '
                            'That unobserved interval is left open rather than assumed snow-free.')
    if end.astimezone(ZoneInfo(ZONES[site])).date() > season_end:
        return _unavailable('The 48-hour forecast extends beyond this October–May season view.')
    mean = _number(selected.get('mean'))
    lower, upper = _number(selected.get('lower')), _number(selected.get('upper'))
    if mean is None or (lower is not None and upper is not None and lower > upper):
        return _unavailable('The model amount or uncertainty bounds are invalid.')
    has_band = lower is not None and upper is not None
    retained = bool(selected.get('retained_issued_forecast') or start < now)
    points = [dict(time=start.isoformat(), cumulative_inches=baseline,
                   lower_inches=baseline if has_band else None,
                   upper_inches=baseline if has_band else None),
              dict(time=end.isoformat(), cumulative_inches=baseline + mean,
                   lower_inches=baseline + lower if has_band else None,
                   upper_inches=baseline + upper if has_band else None)]
    return dict(status='ready', label='Our model', points=points,
                issued_at=issued.isoformat(), start=start.isoformat(), end=end.isoformat(),
                method_id=selected.get('method_id'), horizon_hours=48,
                retained_issued_forecast=retained,
                interval_label=selected.get('interval_label') if has_band else None,
                note=((f'Issued forecast from {issued.isoformat()}; its elapsed portion remains a prediction, '
                       'not an observation. This is the original issued trajectory, not a newly issued forecast. '
                       if retained else '')
                      + 'The 48-hour amount continues from the measured total at the exact forecast start. '
                      'The dashed connector and band between endpoints are illustrative, not validated '
                      'hourly predictions. The endpoint bounds are translated once, not summed across horizons. '
                      + ('The model range is uncalibrated and excludes uncertainty in the observed baseline.'
                         if has_band else 'This model supplies no uncertainty range.')))


def get_season(site, now=None, root=None, forecast=None):
    """Build the public seasonal chart payload without network or file writes."""
    if site not in ZONES:
        raise ValueError('Unknown mountain')
    now = now or datetime.now(UTC)
    if isinstance(now, str):
        now = _time(now)
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError('Current time must include a timezone')
    now = now.astimezone(UTC)
    today = now.astimezone(ZoneInfo(ZONES[site])).date()
    root = Path(root or ROOT)
    year = _season_year(today)
    start, end = date(year, 10, 1), date(year + 1, 5, 31)
    cutoff = min(today, end)
    failure = None
    try:
        source = _observations(site, root)
    except Exception:
        source = {}
        failure = 'Season observations could not be read. Refresh all to retry.'
    target = source.get('target') or {}
    records = _daily_records(source.get('records', []), today, now)
    current = _daily_curve(records, start, cutoff)
    reported_source = source.get('reported_total_source') or {}
    reported = _reported_curve(source.get('reported_totals', []), start, cutoff, now, reported_source)
    if reported:
        current = reported
    history = []
    years = sorted({_season_year(day) for day, row in records.items()
                    if row['amount'] is not None and (day.month >= 10 or day.month <= 5)
                    and _season_year(day) < year})[-MAX_HISTORY:]
    for previous in years:
        historical = _daily_curve(records, date(previous, 10, 1), date(previous + 1, 5, 31))
        for point in historical['points']:
            point.pop('time', None)  # Past seasons align by reporting month/day.
        historical.update(season=_label(previous))
        history.append(historical)
    official_history = []
    for item in source.get('reported_total_history', []):
        try:
            previous = int(str(item['season']).split('/')[0])
        except (ValueError, KeyError, TypeError):
            continue
        if previous >= year or previous < year - MAX_HISTORY:
            continue
        curve = _reported_curve(item.get('reported_totals', []), date(previous, 10, 1),
                                date(previous + 1, 5, 31), now, item.get('source') or reported_source)
        if curve is not None:
            for point in curve['points']:
                point.pop('time', None)
            curve.update(season=_label(previous))
            official_history.append(curve)
    if official_history:
        history = sorted(official_history, key=lambda row: row['season'])[-MAX_HISTORY:]
    notes = list(source.get('notes') or [])
    if official_history:
        notes = [note for note in notes if not note.startswith('Historical manual Collins observations')]
    if failure:
        notes.append(failure)
    notes.append('Season is October 1–May 31 by reporting date; historical curves align by month and day.')
    if any(row['total_is_lower_bound'] for row in history) or current['total_is_lower_bound']:
        notes.append('Incomplete daily curves sum recorded snowfall only: they are lower bounds, '
                     'not complete season totals. Line gaps mark missing reports; missing values are never zeros.')
    if reported:
        notes.append('The solid current curve uses directly reported season totals; it does not infer '
                     'daily snowfall on dates without reports.')
        if official_history:
            notes.append(f"{SITES[site]['name']} historical and current curves use the same official resort reporting source. "
                         'They retain the source\'s season-total convention rather than resetting totals on October 1.')
        elif history:
            notes.append('Historical study-plot daily sums and current resort season totals use different '
                         'sources and may use different reporting conventions.')
        if reported_source.get('note'):
            notes.append(reported_source['note'])
    continuation = _continuation(site, current, target, now, end, forecast, root)
    has_current = current['valid_days'] > 0
    sources = list(source.get('sources') or [])
    if reported_source.get('name') and reported_source.get('url'):
        link = {key: reported_source[key] for key in ('name', 'url')}
        if link not in sources:
            sources.append(link)
    return dict(site_id=site,
                status='ready' if current['complete'] and continuation['status'] == 'ready' else
                       'partial' if history or has_current else 'unavailable',
                season=_label(year), season_start=start.isoformat(), season_end=end.isoformat(),
                today=today.isoformat(), generated_at=now.isoformat(),
                target_label=(reported_source.get('target_name') if reported and official_history else None)
                             or target.get('target_name') or target.get('name') or site,
                history_target_label=(history[0].get('source', {}).get('target_name') if official_history else None)
                                     or target.get('target_name'),
                history_label='Past official seasons' if official_history else 'Past station seasons',
                current_target_label=reported_source.get('target_name') if reported else target.get('target_name'),
                quantity_label='Accumulated snow sensor proxy' if site == 'vail' else 'Accumulated snowfall',
                observed_through=current['observed_through'], total_inches=current['total_inches'],
                current=current, history=history, forecast=continuation,
                notes=list(dict.fromkeys(notes)), sources=sources)
