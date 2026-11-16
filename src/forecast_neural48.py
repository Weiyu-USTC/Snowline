"""Live, forecast-conditioned two-day neural snowfall experiment.

The trained targets are reporting-day accumulations. Rolling 24/48-hour live
windows are explicitly experimental; no hourly snowfall supervision is claimed.
Refresh downloads public predictors and runs frozen checkpoints, never trains.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
import threading
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from prepare_neural48_dataset import provider_window
from prepare_neural48_nwp import canonical, sequence
from prepare_neural48_provider import records_to_frames
from prepare_neural48_targets import SITES as TARGETS
from fetch_recent_nwp import VARS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/nwp-weather/neural48-trained'
DATA = ROOT / 'data/neural48_training/current'
SITES = ['alta', 'vail', 'mansfield', 'jay']
STATIONS = {'alta': 'KSLC', 'vail': 'KLXV', 'mansfield': 'KMVL', 'jay': 'KMVL'}
LOCK = threading.Lock()
MODEL_NAMES = {'weather_only': 'Weather TCN', 'weather_radar': 'Weather + radar TCN'}


def utc(value=None):
    t = pd.Timestamp.now(tz='UTC') if value is None else pd.Timestamp(value)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def download(url, path):
    """Bounded public acquisition, immutable content and provenance per cache key."""
    meta = path.with_suffix(path.suffix + '.json')
    if path.exists() and meta.exists():
        return path.read_bytes(), json.loads(meta.read_text())
    response = requests.get(url, timeout=(10, 45), headers={'User-Agent': 'Snowline public research forecast'})
    response.raise_for_status()
    body = response.content
    if len(body) > 80_000_000:
        raise ValueError('Public forecast response exceeded bounded size')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    provenance = {'url': url, 'retrieved_at': utc().isoformat(), 'sha256': hashlib.sha256(body).hexdigest(),
                  'bytes': len(body), 'path': str(path.relative_to(ROOT))}
    atomic_json(meta, provenance)
    return body, provenance


def provider_inputs(issue):
    """Same fixed-offset forecast input used for training; fresh comparison separately."""
    common = {'latitude': ','.join(str(TARGETS[s]['latitude']) for s in SITES),
              'longitude': ','.join(str(TARGETS[s]['longitude']) for s in SITES),
              'elevation': ','.join(str(TARGETS[s]['elevation_m']) for s in SITES),
              'timezone': 'GMT', 'wind_speed_unit': 'ms', 'cell_selection': 'nearest',
              'start_date': str(issue.date()), 'end_date': str((issue + pd.Timedelta(days=4)).date())}
    stamp = issue.strftime('%Y%m%dT%H')
    p = {**common, 'models': 'jma_gsm', 'hourly': ','.join(v + '_previous_day3' for v in VARS)}
    path = DATA / 'raw' / f'jma_{stamp}.json'
    body, jmeta = download('https://previous-runs-api.open-meteo.com/v1/forecast?' + urlencode(p), path)
    records = json.loads(body)
    if isinstance(records, dict) and records.get('error'):
        raise ValueError(records.get('reason', 'JMA forecast response error'))
    inputs = pd.concat(records_to_frames(SITES, records, path), ignore_index=True)
    p = {**common, 'hourly': ','.join(VARS)}
    bpath = DATA / 'raw' / f'best_match_{stamp}.json'
    comparison, bmeta, failure = None, None, None
    try:
        body, bmeta = download('https://api.open-meteo.com/v1/forecast?' + urlencode(p), bpath)
        records = json.loads(body)
        if isinstance(records, dict): records = [records]
        frames = []
        if len(records) != len(SITES): raise ValueError('Provider site count differs')
        for site, record in zip(SITES, records):
            units = record['hourly_units']
            if units.get('snowfall') != 'cm' or units.get('precipitation') != 'mm':
                raise ValueError('Provider accumulation units changed')
            d = pd.DataFrame(record['hourly']).rename(columns={'time': 'valid_time_utc'})
            d['valid_time_utc'] = pd.to_datetime(d.valid_time_utc, utc=True)
            d['site'] = site
            frames.append(d)
        comparison = pd.concat(frames, ignore_index=True)
    except (requests.RequestException, ValueError, KeyError) as exc:
        failure = str(exc)[:300]
    return inputs, comparison, {'jma': jmeta, 'best_match': bmeta, 'best_match_error': failure}


def mos_inputs(issue):
    result, metadata = {}, {}
    for station in sorted(set(STATIONS.values())):
        p = {'station': station, 'model': 'GFS', 'sts': (issue-pd.Timedelta(days=2)).strftime('%Y-%m-%dT00:00Z'),
             'ets': issue.strftime('%Y-%m-%dT%H:%MZ'), 'format': 'csv'}
        path = DATA / 'raw' / f'mos_{station}_{issue:%Y%m%dT%H}.csv'
        # Dashboard weather refresh has usually just fetched these exact airport
        # cycles. Reuse its verified raw response instead of hitting IEM twice.
        body, meta = None, None
        for cached in sorted((ROOT/'data/dashboard/current/raw').glob(f'station_{station}_gfs_mos_*.csv'),reverse=True):
            cache_meta = cached.with_suffix('.meta.json')
            if not cache_meta.exists(): continue
            cm = json.loads(cache_meta.read_text())
            age = (issue-utc(cm['retrieved_at'])).total_seconds()
            if not -600 <= age <= 6*3600: continue
            cb = cached.read_bytes()
            if hashlib.sha256(cb).hexdigest()!=cm['sha256']: continue
            body, meta = cb,{**cm,'path':str(cached.relative_to(ROOT)),'reused_recent_download':True}
            break
        if body is None:
            time.sleep(3)
            body, meta = download('https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?' + urlencode(p), path)
        d = canonical(pd.read_csv(io.BytesIO(body)), cycle_hours=(0,12))
        if len(d) and (set(d.station) != {station} or set(d.model) != {'GFS'}):
            raise ValueError('MOS forecast station identity mismatch')
        # Historical archives lack receipt timestamps; use the same conservative
        # six-hour availability buffer at training and live inference.
        d = d[d.runtime + pd.Timedelta(hours=6) <= issue]
        if d.empty: raise ValueError(f'No issued {station} MOS forecasts available')
        run = d.runtime.max()
        result[station] = d[d.runtime.eq(run)]
        metadata[station] = {**meta, 'run_time_utc': run.isoformat(),
                             'assumed_available_utc': (run + pd.Timedelta(hours=6)).isoformat()}
    return result, metadata


def radar_inputs(issue):
    """Nine causal MRMS frames, decoded once into all three trained domains."""
    import prepare_transfer_radar as transfer
    from prepare_neural48_radar import encode_context
    # Decoder uses fixed trained crop geometry. Vermont local statistics instead
    # use the new town-observer grid cells, exactly as in the training adapter.
    old_regions = transfer.REGIONS
    transfer.REGIONS = {'alta': {'center': [40.583,-111.638], 'stations': {'alta': [40.57607,-111.63889]}},
                       'vail': copy.deepcopy(old_regions['vail']), 'vermont': copy.deepcopy(old_regions['vermont'])}
    base = 'https://noaa-mrms-pds.s3.amazonaws.com/'
    times = pd.date_range(issue-pd.Timedelta(minutes=90), periods=9, freq='10min')
    qtimes = sorted(set(t.floor('h') for t in times))
    jobs = [('PrecipRate', t) for t in times] + [('RadarQualityIndex', t) for t in qtimes]
    sources, decoded = [], {}
    try:
        def acquire(job):
            product, stamp = job
            name = f'MRMS_{product}_00.00_{stamp:%Y%m%d-%H%M%S}.grib2.gz'
            path = DATA / 'radar/raw' / product / f'{stamp:%Y%m%d}' / name
            body, meta = download(base + f'CONUS/{product}_00.00/{stamp:%Y%m%d}/' + name, path)
            return product, stamp, body, meta
        with ThreadPoolExecutor(max_workers=3) as pool:
            acquired = list(pool.map(acquire, jobs))
        for product, stamp, body, meta in acquired:
            fields = transfer.decode_regions(body)
            if any(pd.Timestamp(v['valid_time']) != stamp for v in fields.values()):
                raise ValueError('Radar timestamp differs from requested frame')
            decoded[(product, stamp)] = fields
            sources.append(meta)
        result = {}
        for region, site_names in [('alta',['alta']),('vail',['vail']),('vermont',['mansfield','jay'])]:
            frames = [decoded[('PrecipRate',t)][region] for t in times]
            quals = [decoded[('RadarQualityIndex',t.floor('h'))][region] for t in times]
            lat, lon = frames[0]['latitude'], frames[0]['longitude']
            rate = np.stack([f['values'] for f in frames])
            valid = np.stack([f['valid_fraction'] for f in frames])
            quality = np.stack([f['values'] for f in quals])
            if region == 'vermont':
                indices = [(int(np.argmin(abs(lat-TARGETS[s]['latitude']))), int(np.argmin(abs(lon-TARGETS[s]['longitude'])))) for s in site_names]
                point = np.stack([np.where(valid[:,iy,ix]>=.9,rate[:,iy,ix],np.nan) for iy,ix in indices])
                qp = np.stack([quality[:,iy,ix] for iy,ix in indices])
            else:
                point = np.stack([f['point'] for f in frames], axis=1)
                qp = np.stack([f['point'] for f in quals], axis=1)
            encoded = encode_context({'rate':rate,'valid':valid,'quality':quality,'point_rate':point,'point_quality':qp},
                lat,lon,{s:(TARGETS[s]['latitude'],TARGETS[s]['longitude']) for s in site_names})
            for i, site in enumerate(site_names):
                result[site] = {'radar':encoded['x'],'local':encoded['local_stats'][i],
                    'qualified_fraction':encoded['input_qualified_fraction'],
                    'local_qualified_fraction':float(encoded['local_input_qualified_fraction'][i])}
        return result, {'issue_utc':issue.isoformat(),'first_frame_utc':times[0].isoformat(),
            'last_frame_utc':times[-1].isoformat(),'publication_buffer_minutes':10,'sources':sources}
    finally:
        transfer.REGIONS = old_regions


def current_station_histories(issue):
    """Refresh only modern NOAA observers, preserving frozen training targets."""
    import prepare_neural48_targets as target_module
    previous = target_module.PROBES
    frames, metadata = {}, {}
    try:
        target_module.PROBES = DATA/'station_history'/issue.strftime('%Y%m%dT%H')
        for site in ['mansfield','jay']:
            station = TARGETS[site]['station_id']
            path = target_module.PROBES/f'{station}.csv.gz'
            try:
                _, meta = download(f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station}.csv.gz',path)
                d,_ = target_module.noaa_daily(site)
                frames[site] = d
                metadata[site] = {'raw':meta,'latest_reporting_date':str(d.reporting_date.max().date()),
                    'scope':'Current same-station NOAA records; frozen training target files remain unchanged.'}
            except (requests.RequestException, ValueError, OSError) as exc:
                metadata[site] = {'refresh_error':str(exc)[:300],'fallback':'Last verified cached station records'}
    finally:
        target_module.PROBES = previous
    return frames,metadata


def history_inputs(site, issue, current=None):
    d = pd.read_csv(ROOT / 'data/neural48_training/targets/daily_targets.csv')
    d['period_end_utc'] = pd.to_datetime(d.period_end_utc,utc=True)
    if current is not None:
        # New source rows replace the same station/reporting date only; retain
        # actual observation clocks, quality flags and nonnegative snowfall.
        d['reporting_date'] = pd.to_datetime(d.reporting_date)
        d = pd.concat([d,current],ignore_index=True).drop_duplicates(['site','reporting_date'],keep='last')
    known = d[d.site.eq(site)&d.valid_snowfall.astype(bool)&(d.period_end_utc+pd.Timedelta(hours=24)<=issue)].sort_values('period_end_utc')
    hs = []
    for days in [3,7,14,30,60]:
        recent = known[known.period_end_utc > issue-pd.Timedelta(days=days+1)].snowfall_inches.to_numpy(float)
        hs.extend([float(recent.mean()) if len(recent) else np.nan,float(np.mean(recent>0)) if len(recent) else np.nan,len(recent)/days])
    age = (issue-known.period_end_utc.iloc[-1]).total_seconds()/86400 if len(known) else np.inf
    hs.extend([float(known.snowfall_inches.iloc[-1]) if age<=60 else np.nan,min(60.,age)])
    return np.array(hs,np.float32), {'latest_report_end_utc': known.period_end_utc.iloc[-1].isoformat() if len(known) else None,
        'age_days':float(age) if np.isfinite(age) else None,'recent_history_available':age<=60,
        'availability_policy':'Only reports ending at least24h before issue; no stale last amount older60days.'}


def prepare_current(issue=None, include_radar=True, progress=None):
    now = utc(issue)
    issue = now.floor('10min')
    # Choose a fully future, unambiguous clock-hour accumulation interval.
    start = now.floor('h')+pd.Timedelta(hours=1)
    if progress: progress('Downloading issued weather forecasts for all four 48-hour neural outlooks…')
    mos, mos_meta = mos_inputs(issue)
    provider, best_match, provider_meta = provider_inputs(issue)
    current_history, history_meta = current_station_histories(issue)
    radar, radar_meta, radar_error = {}, None, None
    if include_radar:
        if progress: progress('Reading recent radar near Alta, Vail and Vermont…')
        try: radar, radar_meta = radar_inputs(issue)
        except (requests.RequestException, ValueError, OSError) as exc: radar_error = str(exc)[:400]
    arrays = {key: [] for key in ['nwp','radar','local','static','history','site']}
    rows = []
    for index, site in enumerate(SITES):
        target = TARGETS[site]
        row = {'site_id':site,'display_name':target['resort_display_name'],'issued_at':issue.isoformat(),
               'start':start.isoformat(),'midpoint':(start+pd.Timedelta(hours=24)).isoformat(),
               'end':(start+pd.Timedelta(hours=48)).isoformat(),
               'target':{'name':target['target_name'],'station_id':target['station_id'],'elevation_m':target['elevation_m'],
                   'latitude':target['latitude'],'longitude':target['longitude'],'proxy':site!='alta','note':target['qualification']},
               'status':'experimental','hourly_validated':False,'rolling_window_validated':False,
               'warnings':['Rolling48-hour inference is experimental: training and testing used fixed morning reporting-day windows.',
                           'The neural model predicts two24-hour totals. It does not independently predict observed hourly snowfall.',
                           'The historical provider benchmark is older JMA guidance, while the current comparison uses fresh Open-Meteo guidance.'],
               'sources':{'mos':mos_meta[STATIONS[site]],'provider':provider_meta,'radar':radar_meta}}
        if site!='alta': row['warnings'].append(target['qualification'])
        try:
            group = mos[STATIONS[site]]; run = group.runtime.iloc[0]
            wx = sequence(group,(start-run).total_seconds()/3600,(issue-run).total_seconds()/3600)
            jma, ends = provider_window(provider[provider.site.eq(site)],start)
            latest_source = ends[-1].ceil('h')-pd.Timedelta(days=3)
            if latest_source > issue: raise ValueError('JMA nominal source bound follows forecast issue')
            if not np.isfinite(wx).all() or not np.isfinite(jma).all():
                raise ValueError('Issued weather guidance does not cover all48hours; no extrapolation was used')
            raw = jma[:,4].reshape(2,24).sum(1)
            current = None
            if best_match is not None:
                bx,_ = provider_window(best_match[best_match.site.eq(site)],start)
                if np.isfinite(bx[:,4]).all(): current = bx[:,4].reshape(2,24).sum(1)
            if current is None: row['warnings'].append('Fresh matched-location provider comparison unavailable; neural predictors remain complete.')
            hs,hmeta = history_inputs(site,issue,current_history.get(site))
            if site in history_meta:hmeta['refresh']=history_meta[site]
            if hmeta['age_days'] is None or hmeta['age_days']>3:
                row['warnings'].append('Recent snowfall observations are incomplete; missing-history indicators are used.')
            row['sources']['snow_history'] = hmeta
            phase = 2*np.pi*(start.dayofyear-1)/365.2425
            static = [np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase),target['elevation_m']/3000,
                      target['latitude']/90,target['longitude']/180,(start-issue).total_seconds()/86400]
            has_radar = site in radar and radar[site]['qualified_fraction']>=.1
            row.update(array_row=len(arrays['site']),radar_available=has_radar,
                conservative_provider_daily_inches=raw.tolist(),provider_daily_inches=current.tolist() if current is not None else [None,None])
            if not has_radar: row['warnings'].append('Recent radar unavailable or low quality; the weather-only neural model is used. '+(radar_error or ''))
            arrays['nwp'].append(np.concatenate([wx,jma],axis=-1));arrays['history'].append(hs)
            arrays['radar'].append(radar[site]['radar'] if has_radar else np.zeros((9,5,16,16),np.float32))
            arrays['local'].append(radar[site]['local'] if has_radar else np.full((9,8),np.nan,np.float32))
            arrays['static'].append(static);arrays['site'].append(index)
        except (ValueError, KeyError) as exc:
            row.update(status='unavailable',reason=str(exc))
        rows.append(row)
    prepared = {k:np.asarray(v,dtype='int64' if k=='site' else 'float32') for k,v in arrays.items()}
    folder = DATA / issue.strftime('%Y%m%dT%H%MZ')
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'arrays.npz',**prepared)
    atomic_json(folder/'metadata.json',{'issued_at':issue.isoformat(),'start':start.isoformat(),'sites':rows})
    return prepared, rows, folder


def get_cached_forecasts():
    path = OUT/'current_forecast.json'
    if not path.exists(): return None
    try:
        payload = json.loads(path.read_text())
        now = utc()
        for site in payload.get('sites',{}).values():
            site['stale'] = (now-utc(site['issued_at'])).total_seconds()>3*3600
            site['expired'] = now>=utc(site['end'])
        return payload
    except (OSError,ValueError,KeyError): return None


def reusable_cache(payload,now):
    if not payload or (now-utc(payload['generated_at'])).total_seconds()>=1800:return False
    available=[r for r in payload.get('sites',{}).values() if r.get('status')!='unavailable']
    return bool(available) and all(utc(r['start'])>=now for r in available)


def issue_from_prepared(arrays, rows, folder, family_overrides=None):
    from train_neural48 import load_bundle,predict_neural
    folder=Path(folder).resolve()
    bundle = load_bundle(OUT/'models')
    family_overrides=family_overrides or {}
    if set(family_overrides)-set(SITES) or any(v not in {'weather_only','weather_radar'} for v in family_overrides.values()):
        raise ValueError('Unknown deployed48-hour neural family')
    if not bundle['selection'].get('frozen_utc'):
        raise RuntimeError('Neural model selection is still being finalized; prior forecast retained.')
    baselines = {}
    baseline_path = OUT/'models/baselines.pkl'
    if baseline_path.exists() and len(arrays['site']):
        from neural48_baselines import predict_baselines
        with baseline_path.open('rb') as f:baseline_bundle=pickle.load(f)
        aligned=[r for r in rows if r['status']!='unavailable']
        baselines=predict_baselines(baseline_bundle,arrays['nwp'],arrays['static'],arrays['history'],
            np.array([SITES[int(s)] for s in arrays['site']]),np.arange(len(arrays['site'])),
            np.array([utc(r['start']).dayofyear for r in aligned]))
    result = {}
    previous=get_cached_forecasts()
    # Never feed missing radar into a family selected for complete radar input.
    for row in rows:
        if row['status']=='unavailable':
            old=(previous or {}).get('sites',{}).get(row['site_id'])
            if old and old.get('status')!='unavailable':
                retained=copy.deepcopy(old)
                retained['refresh_error']=row['reason']
                retained.setdefault('warnings',[]).append('Latest refresh unavailable; retaining the previous forecast with its original issue and valid times.')
                result[row['site_id']]=retained
            else:result[row['site_id']]=row
            continue
        i = row['array_row']; one = {k:v[i:i+1] for k,v in arrays.items()}
        family = None if row['radar_available'] else 'weather_only'
        pred = predict_neural(bundle,one,family=family,n_paths=768)
        means = np.asarray(pred['daily_mean'])[0]
        paths = np.asarray(pred['joint_paths'])[0]
        if means.shape!=(2,) or paths.shape!=(768,2) or not np.isfinite(means).all() or not np.isfinite(paths).all():
            raise ValueError('Neural forecast output dimensions or values invalid')
        q = np.quantile(paths,[.1,.9],axis=0)
        daily = []
        for day in range(2):
            daily.append({'start':row['start'] if day==0 else row['midpoint'],
                'end':row['midpoint'] if day==0 else row['end'],'mean_inches':float(means[day]),
                'q10_inches':float(q[0,day]),'q90_inches':float(q[1,day]),
                'provider_inches':row['provider_daily_inches'][day],
                'conservative_provider_inches':row['conservative_provider_daily_inches'][day]})
        totals = []
        for days in [1,2]:
            tq = np.quantile(paths[:,:days].sum(1),[.1,.9])
            pp = row['provider_daily_inches'][:days]
            totals.append({'hours':days*24,'end':row['midpoint'] if days==1 else row['end'],
                'mean_inches':float(means[:days].sum()),'q10_inches':float(tq[0]),'q90_inches':float(tq[1]),
                'provider_inches':float(sum(pp)) if all(p is not None for p in pp) else None,
                'conservative_provider_inches':float(sum(row['conservative_provider_daily_inches'][:days]))})
        method = pred['methods'][0]
        selected_family = family or bundle['selection']['selected_family_by_site'][row['site_id']]
        comparisons=[]
        for name,bpred in baselines.items():
            bmean=np.asarray(bpred['daily_mean'])[i];bpaths=np.asarray(bpred['joint_paths'])[i]
            comparisons.append({'name':name,'daily_mean_inches':bmean.tolist(),'totals':[
                {'hours':days*24,'mean_inches':float(bmean[:days].sum()),
                 'q10_inches':float(np.quantile(bpaths[:,:days].sum(1),.1)),
                 'q90_inches':float(np.quantile(bpaths[:,:days].sum(1),.9))} for days in [1,2]],
                 'target_scope':'Refit on the same station and two-day target as the neural model',
                 'rolling_window_validated':False})
        row.update(model_name=method,family=selected_family,daily=daily,totals=totals,comparators=comparisons,
            interval_label='Uncalibrated80% model prediction interval',
            provider_name='Open-Meteo · matched station location',conservative_provider_name='JMA GSM ·72-hour offset',
            validation_scope='Fixed morning reporting-day24/48-hour totals; held-out2023/24winter; rollinglivewindows unvalidated')
        # Preserve legacy first-day outputs: the newer24h portfolio weights
        # were fitted against these exact families. A horizon-specific48h
        # winner is a separate output, not a replacement of that24h input.
        if row['site_id'] in family_overrides:
            requested=family_overrides[row['site_id']]
            actual=requested if row['radar_available'] or requested=='weather_only' else 'weather_only'
            if actual==selected_family:
                variant_means,variant_paths,variant_name=means,paths,method
            else:
                variant=predict_neural(bundle,one,family=actual,n_paths=768)
                variant_means=np.asarray(variant['daily_mean'])[0]
                variant_paths=np.asarray(variant['joint_paths'])[0]
                variant_name=variant['methods'][0]
            if variant_means.shape!=(2,) or variant_paths.shape!=(768,2) or not np.isfinite(variant_means).all() or not np.isfinite(variant_paths).all():
                raise ValueError('Deployed48-hour variant output invalid')
            quantiles=np.quantile(variant_paths.sum(1),[.1,.9])
            row['deployed_48']={'hours':48,'start':row['start'],'end':row['end'],'issued_at':row['issued_at'],
                'mean_inches':float(variant_means.sum()),'q10_inches':float(quantiles[0]),'q90_inches':float(quantiles[1]),
                'provider_inches':totals[1]['provider_inches'],'family':actual,'requested_family':requested,
                'model_name':variant_name,'status':'fallback' if actual!=requested else 'ready',
                'reason':'Recent radar is unavailable; using the saved weather-only neural fallback.' if actual!=requested else None,
                'interval_label':'Uncalibrated80% model prediction interval',
                'selection_scope':'Per-horizon2022/23validationRMSE; original24h component and frozen training selection retained.',
                'hourly_validated':False,'rolling_window_validated':False}
        row.pop('array_row',None)
        result[row['site_id']] = row
    payload = {'schema_version':1,'generated_at':utc().isoformat(),'sites':result,
        'report_url':'/reports/weather/neural48-trained/report.html','input_snapshot':str(folder.relative_to(ROOT)),
        'deployment':'Experimental selectable forecast; original weather and daily products retained',
        'hourly_validated':False}
    atomic_json(OUT/'current_forecast.json',payload)
    plot_current(payload)
    return payload


def plot_current(payload):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,site in zip(axes.flat,SITES):
        row = payload['sites'][site]
        ax.set_title(row['display_name']+' — '+row['target']['name'],fontsize=10)
        if row['status']=='unavailable':
            ax.text(.5,.5,row['reason'],ha='center',va='center',wrap=True,transform=ax.transAxes);continue
        x = [0,24,48]; t = row['totals']
        mean = [0]+[p['mean_inches'] for p in t]
        lower = [0]+[p['q10_inches'] for p in t]; upper = [0]+[p['q90_inches'] for p in t]
        ax.fill_between(x,lower,upper,color='#dca85b',alpha=.25,label='Uncalibrated 80% model interval')
        ax.plot(x,mean,'o-',color='#b77528',label=row['model_name'])
        if all(p['provider_inches'] is not None for p in t):
            ax.plot(x,[0]+[p['provider_inches'] for p in t],'s--',color='#177a83',label='Open-Meteo same station')
        for comparator in row.get('comparators',[]):
            if comparator['name']=='Current model family · refit':
                ax.plot(x,[0]+[p['mean_inches'] for p in comparator['totals']],'^:',color='#8565a1',label='Current family · refit')
            elif comparator['name']=='Historical calendar distribution':
                ax.plot(x,[0]+[p['mean_inches'] for p in comparator['totals']],'--',color='#72767b',alpha=.5,label='Historical mean (α=0.5)')
        ax.set(xticks=x,xlabel='Hours after forecast start',ylabel='Accumulated snowfall (inches)',ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    windows=[r for r in payload['sites'].values() if r.get('status')!='unavailable']
    dates=''
    if windows:
        first=windows[0]
        dates=f"{utc(first['start']):%b %d, %H:%M} → {utc(first['end']):%b %d, %H:%M UTC}"
    fig.suptitle('Rolling 48-hour neural snowfall outlook — experimental\n'+dates+'\nTwo 24-hour totals; lines connect accumulation checkpoints, not hourly predictions',fontsize=12)
    fig.savefig(OUT/'current_forecast.png',dpi=160);plt.close(fig)


def refresh_all(progress=None,force=False,include_radar=True,family_overrides=None):
    with LOCK:
        cached = get_cached_forecasts()
        requested=family_overrides or {}
        variants_ready=not requested or bool(cached) and all(
            cached.get('sites',{}).get(site,{}).get('deployed_48',{}).get('requested_family')==family
            for site,family in requested.items())
        if not force and variants_ready and reusable_cache(cached,utc()):
            return cached
        if not (OUT/'models/selection.json').exists():
            raise RuntimeError('The48-hour neural checkpoints are not ready yet; saved forecasts were retained.')
        arrays,rows,folder = prepare_current(include_radar=include_radar,progress=progress)
        if progress: progress('Running the selected frozen48-hour neural models…')
        return issue_from_prepared(arrays,rows,folder,family_overrides=family_overrides)


if __name__=='__main__':
    p = argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--no-radar',action='store_true')
    p.add_argument('--prepared',type=Path);a = p.parse_args()
    if a.prepared:
        arrays = dict(np.load(a.prepared/'arrays.npz'));rows = json.loads((a.prepared/'metadata.json').read_text())['sites']
        result = issue_from_prepared(arrays,rows,a.prepared)
    elif a.prepare_only:
        _,rows,path = prepare_current(include_radar=not a.no_radar,progress=lambda s:print(s,flush=True))
        result = {'prepared':str(path),'sites':{r['site_id']:r['status'] for r in rows}}
    else: result = refresh_all(force=True,include_radar=not a.no_radar,progress=lambda s:print(s,flush=True))
    print(json.dumps({k:v for k,v in result.items() if k!='sites'},indent=2),flush=True)
