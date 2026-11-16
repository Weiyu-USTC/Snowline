"""Prepare snowfall accumulation targets with explicit station and clock evidence.

Fresh snowfall is never inferred from snow-depth changes.  The Vermont targets
are nearby CoCoRaHS observers distributed by NOAA, not resort or summit totals.
Only observed, quality-screened SNOW values are used; absent values stay absent.

Alta has documented 12-hour manual measurements paired into 24-hour intervals.
Vail is retained separately as a provisional Sno24 sensor proxy: its reset
protocol and historical UTC offset are not established by the source.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / 'data/neural48_training/source_probes'
OUT = ROOT / 'data/neural48_training/targets'
UTC = timezone.utc
MST = timezone(timedelta(hours=-7), 'MST')
EASTERN = ZoneInfo('America/New_York')
SITES = {
    'alta': dict(resort_display_name='Alta', station_id='CLN',
        target_name='Alta-Collins manual snow-study plot', latitude=40.57607,
        longitude=-111.63889, elevation_m=2945., nominal_local_clock='0400',
        clock_zone='fixed MST (UTC-07:00)', clock_verified=True,
        verified_fresh_snow=True, target_class='manual_station_fresh_snow',
        qualification='Manual 12-hour fresh snow summed only across complete 24-hour pairs.',
        source_urls=['https://hive.utah.edu/records/280qy-4ee96',
                     'https://hive.utah.edu/records/6hdjr-stw17'],
        attribution='Michael L. Wasserstein and Jim Steenburgh, University of Utah Hive; CC BY-NC 3.0'),
    'vail': dict(resort_display_name='Vail', station_id='CAVMM',
        target_name='Vail Mid-Mountain Sno24 sensor proxy', latitude=39.6176,
        longitude=-106.38, elevation_m=3140., nominal_local_clock='0400',
        clock_zone='ASSUMED fixed MST for exploratory alignment; historical offset unverified',
        clock_verified=False, verified_fresh_snow=False, target_class='unverified_sensor_proxy',
        qualification='Reported 04:00 Sno24 endpoint; reset interval and equivalence to fresh snowfall unverified. Summed endpoints are a proxy score, not certified 48-hour fresh snowfall.',
        source_urls=['https://stations.avalanche.state.co.us/precip.php'],
        attribution='CAIC station display; station provider Vail Resort'),
    'mansfield': dict(resort_display_name='Stowe', station_id='US1VTLM0001',
        target_name='Stowe 0.2 SW (town observer)', latitude=44.464,
        longitude=-72.6907, elevation_m=223.1, nominal_local_clock='0730',
        clock_zone='America/New_York local civil time (EST/EDT)', clock_verified=True,
        verified_fresh_snow=True, target_class='nearby_town_fresh_snow',
        qualification='Town observer approximately 12 km from the former Mount Mansfield target; not Stowe resort or summit snowfall.',
        source_urls=['https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/US1VTLM0001.csv.gz'],
        attribution='NOAA GHCN-Daily / CoCoRaHS observer US1VTLM0001'),
    'jay': dict(resort_display_name='Jay Peak', station_id='US1VTOL0001',
        target_name='Westfield 0.7 WNW (nearby town observer)', latitude=44.8916,
        longitude=-72.4435, elevation_m=325.8, nominal_local_clock='0700',
        clock_zone='America/New_York local civil time (EST/EDT)', clock_verified=True,
        verified_fresh_snow=True, target_class='nearby_town_fresh_snow',
        qualification='Nearby town observer approximately 7 km from the former Jay NOAA target; not Jay Peak resort or summit snowfall.',
        source_urls=['https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/US1VTOL0001.csv.gz'],
        attribution='NOAA GHCN-Daily / CoCoRaHS observer US1VTOL0001'),
}
DOCS = {
    'noaa_format_flags_and_clock': 'https://www.ncei.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf',
    'noaa_daily_format': 'https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt',
    'cocorahs_local_civil_clock': 'https://media.cocorahs.org/docs/TexasSpring21_v7_1.pdf',
    'cocorahs_snowfall_definition': 'https://www.weather.gov/ohx/cocorahsfaq',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_endpoint(day, clock, zone):
    """Parse a documented local clock. Invalid/blank clocks remain unknown."""
    clock = str(clock).strip()
    if len(clock) != 4 or not clock.isdigit():
        return pd.NaT
    hour, minute = int(clock[:2]), int(clock[2:])
    if hour > 23 or minute > 59:
        return pd.NaT
    date = pd.Timestamp(day).date()
    return pd.Timestamp(datetime(date.year, date.month, date.day, hour, minute,
                                 tzinfo=zone).astimezone(UTC))


def fetch_station(station, refresh=False):
    PROBES.mkdir(parents=True, exist_ok=True)
    path = PROBES / f'{station}.csv.gz'
    url = f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station}.csv.gz'
    if refresh or not path.exists():
        with urlopen(Request(url, headers={'User-Agent': 'Snowline research target preparation'}), timeout=90) as response:
            data = response.read(20_000_001)
        if len(data) > 20_000_000:
            raise ValueError('Station download exceeded bounded size')
        gzip.decompress(data)  # Validate before replacing a prior source file.
        path.write_bytes(data)
        path.with_suffix('.download.json').write_text(json.dumps(dict(
            url=url, retrieved_utc=pd.Timestamp.now(tz='UTC').isoformat(),
            sha256=sha256(path), bytes=len(data)), indent=2))
    return path


def noaa_daily(site, refresh=False):
    meta = SITES[site]
    path = fetch_station(meta['station_id'], refresh)
    rows = []
    with gzip.open(path, 'rt', newline='') as f:
        for record in csv.reader(f):
            if len(record) < 8 or record[2] != 'SNOW':
                continue
            station, day, _, value, mflag, qflag, sflag, clock = record[:8]
            if station != meta['station_id']:
                raise ValueError('NOAA station identifier mismatch')
            raw = float(value)
            # P means missing presumed zero; A may represent a multi-day total.
            usable = raw >= 0 and not qflag and mflag not in {'P', 'A'}
            date = pd.to_datetime(day, format='%Y%m%d')
            endpoint = local_endpoint(date, clock, EASTERN)
            nominal = local_endpoint(date, meta['nominal_local_clock'], EASTERN)
            rows.append(dict(reporting_date=date, snowfall_inches=raw / 25.4 if usable else np.nan,
                raw_snowfall_mm=raw if raw != -9999 else np.nan,
                observation_clock_local=clock, measurement_flag=mflag,
                quality_flag=qflag, source_flag=sflag, trace=mflag == 'T',
                valid_snowfall=usable, period_end_utc=endpoint,
                nominal_period_end_utc=nominal, clock_matches_nominal=endpoint == nominal,
                source_path=str(path.relative_to(ROOT))))
    d = pd.DataFrame(rows).sort_values('reporting_date').reset_index(drop=True)
    if d.reporting_date.duplicated().any():
        raise ValueError('Duplicate station/day SNOW rows must be resolved, not silently averaged')
    consecutive = d.reporting_date.diff().eq(pd.Timedelta(days=1))
    # Previous actual observation is the board-cycle start evidence. No missing
    # prior clock is replaced by a fictitious exact clock for strict scoring.
    d['period_start_utc'] = d.period_end_utc.shift().where(consecutive)
    d['period_duration_hours'] = (d.period_end_utc - d.period_start_utc).dt.total_seconds() / 3600
    d['nominal_period_start_utc'] = pd.to_datetime([
        local_endpoint(day - pd.Timedelta(days=1), meta['nominal_local_clock'], EASTERN)
        for day in d.reporting_date], utc=True)
    d['verified_period_24h'] = d.period_duration_hours.eq(24)
    d['period_clock_evidence'] = np.where(d.period_start_utc.notna() & d.period_end_utc.notna(),
        'actual consecutive NOAA observation clocks; CoCoRaHS daily reporting protocol',
        'missing prior or current observation clock; exact interval unknown')
    return finish_daily(d, site), path


def finish_daily(d, site):
    meta = SITES[site]
    for key in ['station_id', 'resort_display_name', 'target_name', 'latitude', 'longitude',
                'elevation_m', 'target_class', 'verified_fresh_snow', 'clock_verified']:
        d[key] = meta[key]
    d['site'] = site
    d['season_end_year'] = d.reporting_date.dt.year + d.reporting_date.dt.month.ge(7).astype(int)
    d['month'] = d.reporting_date.dt.month
    return d


def alta_daily():
    path = ROOT / 'data/east_alta_research/alta_hive/alta_collins_daily_paired.csv'
    raw = pd.read_csv(path)
    end = pd.to_datetime(raw.reporting_endpoint_lst)
    start = pd.to_datetime(raw.period_start_lst)
    valid = raw.complete_24h_pair.astype(str).str.lower().eq('true') & raw.observed_12h_periods.eq(2)
    snowfall = pd.to_numeric(raw.snowfall_24h_inches, errors='coerce')
    valid &= snowfall.ge(0) & end.sub(start).eq(pd.Timedelta(days=1))
    d = pd.DataFrame(dict(reporting_date=end.dt.normalize(),
        snowfall_inches=snowfall.where(valid),
        liquid_precipitation_inches=pd.to_numeric(raw.lpe_24h_inches, errors='coerce').where(valid),
        valid_snowfall=valid, observation_clock_local='0400', measurement_flag='',
        quality_flag='', source_flag='Alta manual 12-hour observations', trace=False,
        period_start_utc=(start + pd.Timedelta(hours=7)).dt.tz_localize('UTC'),
        period_end_utc=(end + pd.Timedelta(hours=7)).dt.tz_localize('UTC'),
        period_duration_hours=24., verified_period_24h=valid,
        clock_matches_nominal=True,
        period_clock_evidence='Documented 04:00 to 04:00 LST; two complete 12-hour observations',
        source_path=str(path.relative_to(ROOT))))
    d['nominal_period_start_utc'] = d.period_start_utc
    d['nominal_period_end_utc'] = d.period_end_utc
    return finish_daily(d, 'alta'), path


def vail_daily():
    path = ROOT / 'outputs/vail-storm-models/daily_target_0400.csv'
    raw = pd.read_csv(path)
    end = pd.to_datetime(raw.reporting_endpoint_source_clock)
    values = pd.to_numeric(raw.reported_sno24_inches, errors='coerce')
    # Repeat the source audit exclusions so a future regeneration of the
    # upstream proxy cannot accidentally restore known artifact windows.
    quarantine = end.between('2020-03-20', '2020-06-01', inclusive='left') | end.between('2023-10-01', '2023-11-29', inclusive='left')
    valid = values.ge(0) & ~quarantine
    assumed_end = (end + pd.Timedelta(hours=7)).dt.tz_localize('UTC')
    d = pd.DataFrame(dict(reporting_date=end.dt.normalize(), snowfall_inches=values.where(valid),
        raw_sno24_inches=values, valid_snowfall=valid, observation_clock_local='0400',
        measurement_flag='', quality_flag=np.where(quarantine, 'known_artifact_window', ''),
        source_flag='CAIC CAVMM Sno24 sensor proxy', trace=False,
        period_start_utc=assumed_end - pd.Timedelta(days=1), period_end_utc=assumed_end,
        nominal_period_start_utc=assumed_end - pd.Timedelta(days=1),
        nominal_period_end_utc=assumed_end, period_duration_hours=24.,
        verified_period_24h=False, clock_matches_nominal=True,
        period_clock_evidence='ASSUMED fixed MST and 24-hour interval; reset protocol and historical UTC offset unverified',
        source_path=str(path.relative_to(ROOT))))
    return finish_daily(d, 'vail'), path


def make_pairs(d):
    """Two consecutive daily observations, preserving true elapsed durations.

    ``strict_48h_eligible`` requires measured fresh snow, verified clock mapping,
    exact 24+24-hour intervals and all observation clocks at the nominal schedule.
    Vail pairs are exported for explicitly qualified proxy analysis only.
    """
    rows = []
    for site, group in d.groupby('site', sort=False):
        group = group.sort_values('reporting_date').reset_index(drop=True)
        for i in range(len(group) - 1):
            a, b = group.iloc[i], group.iloc[i + 1]
            if b.reporting_date - a.reporting_date != pd.Timedelta(days=1):
                continue
            if not a.valid_snowfall or not b.valid_snowfall:
                continue
            start, mid, end = a.period_start_utc, a.period_end_utc, b.period_end_utc
            continuous = pd.notna(mid) and mid == b.period_start_utc
            duration = (end - start).total_seconds() / 3600 if pd.notna(start) and pd.notna(end) else np.nan
            nominal_start = a.nominal_period_start_utc
            nominal_match = a.clock_matches_nominal and b.clock_matches_nominal and start == nominal_start
            exact = bool(continuous and a.verified_period_24h and b.verified_period_24h and duration == 48)
            strict = bool(exact and nominal_match and a.clock_verified and a.verified_fresh_snow)
            origin = a.reporting_date - pd.Timedelta(days=1)
            rows.append(dict(site=site, resort_display_name=a.resort_display_name,
                station_id=a.station_id, target_name=a.target_name, target_class=a.target_class,
                origin_date=origin.date().isoformat(),
                day1_reporting_date=a.reporting_date.date().isoformat(),
                day2_reporting_date=b.reporting_date.date().isoformat(),
                nominal_forecast_start_utc=nominal_start, target_start_utc=start,
                target_midpoint_utc=mid, target_end_utc=end,
                day1_duration_hours=a.period_duration_hours,
                day2_duration_hours=b.period_duration_hours, total_duration_hours=duration,
                snowfall_day1_inches=float(a.snowfall_inches), snowfall_day2_inches=float(b.snowfall_inches),
                snowfall_sum_inches=float(a.snowfall_inches + b.snowfall_inches),
                intervals_continuous=continuous, exact_24h_plus_24h=exact,
                all_clocks_nominal=bool(nominal_match), strict_48h_eligible=strict,
                verified_fresh_snow=bool(a.verified_fresh_snow), clock_verified=bool(a.clock_verified),
                season_end_year=int(origin.year + (origin.month >= 7)),
                target_latitude=a.latitude, target_longitude=a.longitude,
                target_elevation_m=a.elevation_m))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--refresh', action='store_true', help='Refetch public NOAA station files')
    p.add_argument('--output', type=Path, default=OUT)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames, sources = [], []
    for site in SITES:
        frame, source = (alta_daily() if site == 'alta' else vail_daily() if site == 'vail'
                         else noaa_daily(site, args.refresh))
        frames.append(frame)
        sources.append(dict(site=site, path=str(source.relative_to(ROOT)), sha256=sha256(source)))
        frame.to_csv(args.output / f'{site}_daily.csv', index=False)
    daily = pd.concat(frames, ignore_index=True).sort_values(['site', 'reporting_date'])
    daily.to_csv(args.output / 'daily_targets.csv', index=False)
    pairs = make_pairs(daily)
    pairs.to_csv(args.output / 'two_day_targets.csv', index=False)
    pairs[pairs.strict_48h_eligible].to_csv(args.output / 'strict_48h_targets.csv', index=False)
    coverage = []
    for site, g in daily.groupby('site'):
        pair = pairs[pairs.site.eq(site)]
        for season, w in g.groupby('season_end_year'):
            winter = w[w.month.isin([11,12,1,2,3,4])]
            selected = pair[pair.season_end_year.eq(season) & pd.to_datetime(pair.origin_date).dt.month.isin([11,12,1,2,3,4])]
            coverage.append(dict(site=site, season_end_year=int(season),
                valid_daily_nov_apr=int(winter.valid_snowfall.sum()),
                wet_daily_nov_apr=int(winter.snowfall_inches.gt(0).sum()),
                two_day_pairs_nov_apr=len(selected),
                exact_24h_plus_24h_pairs_nov_apr=int(selected.exact_24h_plus_24h.sum()),
                strict_48h_pairs_nov_apr=int(selected.strict_48h_eligible.sum())))
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(args.output / 'coverage.csv', index=False)
    manifest = dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(), sites=SITES,
        source_documents=DOCS, source_files=sources,
        rules=[
            'NOAA SNOW is millimeters of fresh-snow depth; divide by 25.4 for inches. PRCP uses a different unit.',
            'Missing, negative, nonempty-quality-flag, missing-presumed-zero P, and multi-day A values do not become snowfall targets.',
            'Actual NOAA HHMM clocks are retained; CoCoRaHS local civil clocks include EST/EDT. DST changes are not exactly 24-hour days.',
            'Strict 48-hour fresh-snow evaluation requires consecutive actual 24-hour periods and the nominal observation clocks, including the starting observation.',
            'Retrospective clock exceptions may select evaluation cases; they must never be future input features.',
            'Forecast predictors must be published by issue time; future observation clocks and future daily weather are labels/metadata only.',
            'Split by target interval: exclude any training target ending on or beyond a validation/test boundary.',
            'Hourly timing cannot be directly validated with these daily fresh-snow observations.',
            'Vail sensor proxy scores must be shown separately from verified snowfall scores; no strict fresh-snow 48-hour rows are claimed for Vail.',
            'Vermont values are town-station forecasts; do not present them as resort snowfall or apply an unverified elevation scaling.'
        ], daily_rows=len(daily), all_two_day_rows=len(pairs), strict_48h_rows=int(pairs.strict_48h_eligible.sum()),
        historical_hourly_freshsnow_status='Alta current web page has approximately 25 hourly Collins rows; no reproducible multi-year hourly fresh-snow archive verified. SNOTEL SWE/depth and ASOS liquid precipitation are different targets.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(coverage[coverage.season_end_year.between(2021, 2024)].to_string(index=False))
    print(json.dumps(dict(output=str(args.output), daily_rows=len(daily), pairs=len(pairs),
                          strict_48h_rows=int(pairs.strict_48h_eligible.sum()))))


if __name__ == '__main__':
    main()
