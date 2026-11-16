"""Public mountain observations, with explicit clocks and causal availability.

No snow-depth or SWE difference is ever a fresh-snow target. Historical files
may have been revised after the forecast date; the one-hour receipt allowance
is an operational assumption, not an archived proof of historical receipt.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/mountain_hourly'
RAW = OUT / 'raw'
FEATURES = ['temperature_c', 'wet_bulb_c', 'relative_humidity_pct', 'wind_u_ms',
            'wind_v_ms', 'pressure_hpa', 'precipitation_mm', 'snow_depth_cm',
            'snow_water_equivalent_mm', 'cumulative_precipitation_mm']
FEMC_PAGE = ('https://www.uvm.edu/femc/data/archive/project/'
             'mt-mansfield-west-slope-mid-elevation/dataset/mansfield-west-met-site/data')
FEMC_CLOCK = 'https://www.uvm.edu/femc/data/live'
SNOTEL_CLOCK = ('https://www.nrcs.usda.gov/state-offices/montana/montana-snow-survey/'
                'frequently-asked-snow-survey-questions-montana')


def numeric(s):
    return pd.to_numeric(s, errors='coerce')


def base_frame(index, site, station):
    d = pd.DataFrame(index=index)
    d['mountain'] = site
    d['station_id'] = station
    for f in FEATURES:
        d[f] = np.nan
    return d


def finalize(d):
    d['valid_time_utc'] = pd.to_datetime(d.valid_time_utc, utc=True)
    d['assumed_available_time_utc'] = d.valid_time_utc + pd.Timedelta(hours=1)
    d['publication_delay_hours_assumed'] = 1.0
    d['historical_receipt_verified'] = False
    d['hourly_fresh_snow_observed'] = False
    for c, lo, hi in [('temperature_c', -65, 55), ('wet_bulb_c', -65, 55),
                      ('relative_humidity_pct', 0, 100), ('precipitation_mm', 0, 100),
                      ('snow_depth_cm', 0, 1500), ('snow_water_equivalent_mm', 0, 5000)]:
        d[c] = d[c].where(d[c].between(lo, hi))
    return d.sort_values(['station_id', 'valid_time_utc']).reset_index(drop=True)


def parse_snotel(paths=None):
    parts = []
    for p in sorted(paths if paths is not None else RAW.glob('vail_snotel_*.csv')):
        x = pd.read_csv(p, comment='#')
        d = base_frame(x.index, 'vail', '842:CO:SNTL')
        # NRCS fixed PST, never America/Los_Angeles (which would introduce DST).
        d['valid_time_utc'] = pd.to_datetime(x['Date'], utc=True) + pd.Timedelta(hours=8)
        mapping = [('Snow Water Equivalent', 'snow_water_equivalent_mm', 25.4),
                   ('Snow Depth', 'snow_depth_cm', 2.54),
                   ('Precipitation Accumulation', 'cumulative_precipitation_mm', 25.4),
                   ('Air Temperature Observed', 'temperature_c', 1)]
        for label, target, scale in mapping:
            col = next(c for c in x if c.startswith(label+' ('))
            value = numeric(x[col])
            qc = x[label+' QC Flag']
            value = value.where(~qc.isin(['S', 'M']))
            d[target] = (value-32)*5/9 if target == 'temperature_c' else value*scale
            d[target+'_qc_flag'] = qc
            if label+' QA Flag' in x:
                d[target+'_qa_flag'] = x[label+' QA Flag']
        d['source_file'] = str(p.relative_to(ROOT))
        d['source_url'] = 'https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=842&state=CO'
        d['clock_verified'] = True
        d['source_clock'] = 'Fixed PST UTC-08:00; NRCS official documentation'
        d['quality_notes'] = ('Provisional hourly SNOTEL; suspect/missing flags masked; '
            '0.1-inch SWE/precip quantization and sensor noise; cumulative precip kept '
            'as a state, never differenced into hourly snow. Temperature bias can persist.')
        parts.append(d)
    return finalize(pd.concat(parts, ignore_index=True).drop_duplicates(['station_id', 'valid_time_utc'], keep='last'))


def embedded_femc(html):
    start = html.index("'data':") + len("'data':")
    records, _ = json.JSONDecoder().raw_decode(html[start:])
    return pd.DataFrame(records)


def parse_femc(include_archive=True):
    parts = []
    p = RAW/'femc_west_archive.csv.gz'
    if include_archive and p.exists():
        try:
            x = pd.read_csv(p, low_memory=False)
            x = x[pd.to_datetime(x.RECORDTIME, errors='coerce') >= pd.Timestamp('2020-10-29')]
            parts.append(x)
        except pd.errors.EmptyDataError:
            pass
    p = RAW/'femc_data.html'
    if p.exists():
        parts.append(embedded_femc(p.read_text()))
    if not parts:
        return pd.DataFrame()
    x = pd.concat(parts, ignore_index=True).drop_duplicates('RECORDTIME', keep='last')
    d = base_frame(x.index, 'stowe', 'FEMC_MMW')
    d['valid_time_utc'] = pd.to_datetime(x.RECORDTIME, utc=True) + pd.Timedelta(hours=5)
    d['source_temperature_c'] = numeric(x.AIRTEMP)
    d['relative_humidity_pct'] = numeric(x.RELHUMID)
    # The shared probe has documented intermittent signal failures (2020/21).
    # Training records contain -39.5C and exact 0C with missing RH. Require a
    # valid companion RH reading consistently in every year, independently of
    # HRRR/model errors. Preserve the raw temperature and explicit mask.
    d['temperature_shared_probe_healthy'] = d.relative_humidity_pct.between(0,100)
    # Conservatively quarantine the documented freezing-weather logger/probe
    # failure era: CR1000X installed20Aug2020, replaced26Feb2021 because of
    # hardware failures in freezing temperatures (official maintenance log).
    # A valid RH number alone does not guarantee that the faulty probe worked.
    d['temperature_hardware_era_quarantine'] = d.valid_time_utc.between(
        pd.Timestamp('2020-08-20T00:00Z'),pd.Timestamp('2021-02-27T05:00Z'),inclusive='left')
    d['temperature_shared_probe_healthy'] &= ~d.temperature_hardware_era_quarantine
    d['temperature_c'] = d.source_temperature_c.where(d.temperature_shared_probe_healthy)
    d['relative_humidity_pct'] = d.relative_humidity_pct.where(~d.temperature_hardware_era_quarantine)
    rad = np.deg2rad(numeric(x.MEAN_WIND_DIRECTION))
    speed = numeric(x.MEAN_HORIZONTAL_WINDSPEED)
    d['wind_u_ms'], d['wind_v_ms'] = -speed*np.sin(rad), -speed*np.cos(rad)
    # The unheated gauge is covered in winter. Missing rain is not zero snow.
    d['source_rain_gauge_mm'] = numeric(x.PRECIP)
    d['source_file'] = 'data/mountain_hourly/raw/femc_west_archive.csv.gz;femc_data.html'
    d['source_url'] = FEMC_PAGE
    d['clock_verified'] = True
    d['source_clock'] = 'Fixed EST UTC-05:00, no DST; FEMC live data documentation'
    d['quality_notes'] = ('FEMC QA archive/recent public preview; mid-elevation forest '
        '816m, not Stowe summit; unheated/seasonally covered rain gauge excluded from '
        'precip feature; temperature requires valid companion RH due documented '
        'shared probe faults; documented20Aug2020–26Feb2021hardware era quarantined; '
        'recent records provisional; historical final QA revision risk.')
    d['sampling_interval_minutes'] = 15
    return finalize(d)


def parse_alta_file(p):
    p = Path(p).resolve()
    text = gzip.decompress(p.read_bytes()).decode() if p.suffix == '.gz' else p.read_text()
    soup = BeautifulSoup(text, 'html.parser')
    plain = soup.get_text(' ', strip=True)
    match = re.search(r'Tabular Listing of \d+ Observations from (\d\d/\d\d/\d{4} \d\d?:\d\d) GMT to (\d\d/\d\d/\d{4} \d\d?:\d\d) GMT', plain)
    if not match:
        return pd.DataFrame()
    latest = pd.to_datetime(match.group(2), utc=True)
    table = next(t for t in soup.find_all('table') if 'Time (GMT)' in t.get_text(' ', strip=True))
    rows = table.find_all('tr')
    headers = [n.get_text(' ', strip=True) for n in rows[0].find_all(['th', 'td'])]
    records = []
    last = latest + pd.Timedelta(minutes=1)
    for row in rows[1:]:
        values = [n.get_text(' ', strip=True) for n in row.find_all(['th','td'], recursive=False)]
        if not values or not re.fullmatch(r'\d\d?:\d\d', values[0]):
            continue
        hour, minute = map(int, values[0].split(':'))
        valid = last.normalize() + pd.Timedelta(hours=hour, minutes=minute)
        if valid >= last:
            valid -= pd.Timedelta(days=1)
        last = valid
        rec = dict(zip(headers, values))
        rec['valid_time_utc'] = valid
        records.append(rec)
    x = pd.DataFrame(records)
    if x.empty:
        return x
    d = base_frame(x.index, 'alta', 'CLN')
    d['valid_time_utc'] = x.valid_time_utc
    fields = {'Temperature ° F': ('temperature_c', 1),
              'Wet bulb temperature ° F': ('wet_bulb_c', 1),
              'Precipitation 1hr in': ('precipitation_mm',25.4),
              'Snow depth in': ('snow_depth_cm',2.54)}
    for k,(target,scale) in fields.items():
        if k in x:
            v = numeric(x[k])
            d[target] = (v-32)*5/9 if target.endswith('_c') else v*scale
    d['source_file'] = str(p.relative_to(ROOT))
    d['source_url'] = 'https://mesowest.utah.edu/cgi-bin/droman/meso_base.cgi?stn=CLN'
    d['clock_verified'] = True
    d['source_clock'] = 'MesoWest explicit GMT table headers and date range'
    d['quality_notes'] = ('CLN automated observations; no exposed per-variable QC '
        'guarantee; 1h gauge is liquid equivalent. Resetting Snow interval deliberately '
        'excluded; total snow depth is not fresh snowfall. Receipt delay assumed1h.')
    return finalize(d)


def parse_alta():
    files = sorted((RAW/'alta_cln').glob('*.html.gz'))
    current = RAW/'alta_cln_current.html'
    if current.exists():
        files.append(current)
    parts = [parse_alta_file(p) for p in files]
    parts = [d for d in parts if not d.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(['station_id','valid_time_utc'], keep='last')


def parse_asos():
    x = pd.read_csv(ROOT/'data/neural48_training/asos/routine_hourly_observations.csv',low_memory=False)
    d = base_frame(x.index, '', '')
    d['station_id'] = x.station
    d['mountain'] = x.station.map({'SLC':'alta','LXV':'vail','MVL':'stowe_jay_regional'})
    d['valid_time_utc'] = pd.to_datetime(x.observation_time_utc, utc=True)
    d['temperature_c'] = x.temperature_c
    t, dew = x.temperature_c, x.dewpoint_c
    d['relative_humidity_pct'] = 100*np.exp(17.625*dew/(243.04+dew)-17.625*t/(243.04+t))
    rad = np.deg2rad(x.wind_direction_degrees)
    d['wind_u_ms'], d['wind_v_ms'] = -x.wind_speed_mps*np.sin(rad), -x.wind_speed_mps*np.cos(rad)
    d['precipitation_mm'] = x.gauge_precipitation_mm.where(x.eligible_contiguous_hourly_report)
    d['source_file'] = x.raw_file
    d['source_url'] = 'https://mesonet.agron.iastate.edu/request/download.phtml'
    d['clock_verified'] = True
    d['source_clock'] = 'IEM requested tz=UTC; true report minute retained'
    d['quality_notes'] = ('Routine airport report; regional proxy not mountain snow; '
        'liquid precipitation; true hourly reset endpoints independently unverified; '
        'one-hour assumed publication allowance; trace and PNO QC inherited.')
    return finalize(d)


def fetch_alta_history(workers=3):
    dates = pd.to_datetime(pd.read_csv(ROOT/'data/alta_nowcast/hrrr/hourly_point.csv',
        usecols=['run_time_utc']).run_time_utc, utc=True).dt.normalize().drop_duplicates()
    dates = sorted(d for d in dates if 2020 <= d.year <= 2024)
    (RAW/'alta_cln').mkdir(parents=True,exist_ok=True)
    def fetch(day):
        p = RAW/'alta_cln'/f'{day:%Y%m%d}.html.gz'
        if p.exists():
            return True
        url = 'https://mesowest.utah.edu/cgi-bin/droman/meso_table_mesodyn.cgi'
        params = dict(stn='CLN',unit=0,time='GMT',year1=day.year,month1=day.month,
                      day1=day.day,hour1=23,hours=24,past=1,order=1)
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=40)
                r.raise_for_status()
                if 'Tabular Listing' not in r.text:
                    raise ValueError('No observation table')
                p.write_bytes(gzip.compress(r.content))
                p.with_suffix('.json').write_text(json.dumps(dict(url=r.url,
                    retrieved_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    sha256=hashlib.sha256(r.content).hexdigest()),indent=2))
                time.sleep(.3)
                return True
            except Exception:
                if attempt == 2:
                    return False
                time.sleep(1+attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i,ok in enumerate(pool.map(fetch,dates)):
            if i%50 == 0:
                print(f'Alta pages {i+1}/{len(dates)}, latest_ok={ok}',flush=True)
    now = dt.datetime.now(dt.timezone.utc)
    url = 'https://mesowest.utah.edu/cgi-bin/droman/meso_table_mesodyn.cgi'
    r = requests.get(url, params=dict(stn='CLN',unit=0,time='GMT',hours=24,order=1),timeout=40)
    r.raise_for_status()
    (RAW/'alta_cln_current.html').write_text(r.text)


def fetch_current_observations(station_id, timeout=30):
    """Bounded, credential-free current weather fetch; no historical download.

    Callers must still use assumed_available_time_utc <= the forecast issue and
    expose stale data. A fetch failure raises rather than becoming zero weather.
    """
    RAW.mkdir(parents=True,exist_ok=True)
    if station_id=='842:CO:SNTL':
        fields=','.join(f'{v}::{q}' for v in ['WTEQ','SNWD','PREC','TOBS'] for q in ['value','qcFlag','qaFlag'])
        url=f'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/hourly/842:CO:SNTL/-71,0/{fields}'
        p=RAW/'vail_snotel_current.csv'
        r=requests.get(url,timeout=timeout);r.raise_for_status();p.write_bytes(r.content)
        result=parse_snotel([p])
    elif station_id=='FEMC_MMW':
        p=RAW/'femc_data.html'
        r=requests.get(FEMC_PAGE,timeout=timeout);r.raise_for_status();p.write_bytes(r.content)
        result=parse_femc(include_archive=False)
    elif station_id=='CLN':
        p=RAW/'alta_cln_current.html'
        r=requests.get('https://mesowest.utah.edu/cgi-bin/droman/meso_table_mesodyn.cgi',
            params=dict(stn='CLN',unit=0,time='GMT',hours=24,order=1),timeout=timeout)
        r.raise_for_status();p.write_bytes(r.content)
        result=parse_alta_file(p)
    else:
        raise ValueError(f'No verified mountain source for {station_id}')
    result['retrieved_utc']=pd.Timestamp.now(tz='UTC')
    if result.empty:
        raise ValueError(f'Empty current station response for {station_id}')
    return result


def issue_features(observations, issue, station_id, prefix='obs'):
    """Only observations available by issue; values, ages and coverage stay explicit."""
    issue = pd.Timestamp(issue)
    issue = issue.tz_localize('UTC') if issue.tzinfo is None else issue.tz_convert('UTC')
    d = observations[observations.station_id.eq(station_id)].copy()
    d['valid_time_utc'] = pd.to_datetime(d.valid_time_utc, utc=True)
    d['assumed_available_time_utc'] = pd.to_datetime(d.assumed_available_time_utc, utc=True)
    d = d[d.assumed_available_time_utc.le(issue) & d.valid_time_utc.gt(issue-pd.Timedelta(hours=25))]
    d = d.sort_values('valid_time_utc')
    result = {}
    for f in FEATURES:
        good = d[d[f].notna()]
        result[f'{prefix}_{f}_latest'] = float(good[f].iloc[-1]) if len(good) else np.nan
        result[f'{prefix}_{f}_age_hours'] = float((issue-good.valid_time_utc.iloc[-1]).total_seconds()/3600) if len(good) else 25.
        for hours in [3,6,24]:
            recent = good[good.valid_time_utc.gt(issue-pd.Timedelta(hours=hours+1))]
            result[f'{prefix}_{f}_mean_{hours}h'] = float(recent[f].mean()) if len(recent) else np.nan
            result[f'{prefix}_{f}_n_{hours}h'] = len(recent)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch-alta',action='store_true')
    args = parser.parse_args()
    RAW.mkdir(parents=True,exist_ok=True)
    if args.fetch_alta:
        fetch_alta_history()
    frames = []
    coverage = []
    for name,fn in [('vail_snotel',parse_snotel),('stowe_femc',parse_femc),
                    ('alta_cln',parse_alta),('regional_asos',parse_asos)]:
        d = fn()
        if d.empty:
            continue
        d = d.sort_values(['station_id','valid_time_utc']).reset_index(drop=True)
        d.to_csv(OUT/f'{name}.csv.gz',index=False)
        frames.append(d)
        for station,g in d.groupby('station_id'):
            coverage.append(dict(source=name,station_id=station,rows=len(g),
                start=str(g.valid_time_utc.min()),end=str(g.valid_time_utc.max()),
                **{f+'_count':int(g[f].notna().sum()) for f in FEATURES}))
    combined = pd.concat(frames,ignore_index=True)
    combined.to_csv(OUT/'observations.csv.gz',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    manifest = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        feature_contract=FEATURES,coverage=coverage,
        clocks={'842:CO:SNTL':SNOTEL_CLOCK,'FEMC_MMW':FEMC_CLOCK,'CLN':'explicit GMT query/table headers'},
        availability='Conservative assumed one hour after observation, not archived receipt proof; final historical QC can differ from real-time raw data.',
        missing_policy='Missing stays NaN; no forward fill across outages, no implicit zeros. Feature extraction reports age and count.',
        labels='No hourly fresh-snow labels: SWE/snowdepth are state features, CLN and ASOS precip are liquid-equivalent gauges.',
        jay='No public summit hourly fresh-snow archive verified. MVL is regional; never label it Jay Peak mountain observations.',
        femc_license='CC BY-NC-SA4.0, Forest Ecosystem Monitoring Cooperative, University of Vermont; non-commercial research use; retain attribution and share-alike on redistribution.',
        femc_citation='Forest Ecosystem Monitoring Cooperative. 2017-current. Quality-Controlled Mount Mansfield West Meteorological Data. University of Vermont. '+FEMC_PAGE,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/f'{n}.csv.gz' for n in ['vail_snotel','stowe_femc','alta_cln','regional_asos']] if p.exists()})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(coverage)[['source','station_id','rows','start','end']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
