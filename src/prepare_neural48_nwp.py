"""Forecast-only 48-hour sequences for the new neural experiment.

NOAA GFS MOS is operational forecast guidance archived by Iowa State, not
reanalysis. Native 3/6-hour weather values are bounded-interpolated to hourly
windows. Six-hour precipitation probabilities and categorical QPF retain their
native meaning; they are not hourly probability or snowfall labels.

Frozen previous inputs are read-only. New source responses and all derivatives
are written under data/neural48_training/nwp.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import gzip
import hashlib
import io
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/neural48_training/nwp'
OLD = ROOT / 'data/nwp_archive_probe'
API = 'https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py'
SITES = {'alta': ('KSLC', 11.), 'vail': ('KLXV', 11.),
         'mansfield': ('KMVL', 12.5), 'jay': ('KMVL', 12.)}
NATIVE = ['temperature_c', 'dewpoint_c', 'wind_speed_mps', 'wind_u_mps', 'wind_v_mps']
FEATURES = NATIVE + ['precip_probability_6h', 'qpf_category_6h_ordinal'] + [
    f'qpf_category_{i}_overlap_fraction' for i in range(7)] + ['run_lead_hours', 'issue_lead_hours']


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(station, start, end, out=OUT):
    raw = out / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    name = f'{station}_{start[:10]}_{end[:10]}'
    path, meta = raw / f'{name}.csv.gz', raw / f'{name}.json'
    if path.exists() and meta.exists():
        return path
    url = API + '?' + urlencode(dict(station=station, model='GFS', sts=start, ets=end, format='csv'))
    for attempt in range(3):
        time.sleep(3 if not attempt else 10 * attempt)
        try:
            with urlopen(Request(url, headers={'User-Agent': 'snowfall-research-demo/1.0'}), timeout=90) as r:
                data = r.read(30_000_000)
                headers = dict(r.headers)
            break
        except HTTPError as exc:
            if exc.code not in [429, 502, 503, 504] or attempt == 2:
                raise
            wait = float(exc.headers.get('Retry-After', 10))
            if wait > 60:
                raise
            time.sleep(max(0., wait))
    if len(data) >= 30_000_000:
        raise ValueError('Forecast response exceeded bounded download size')
    df = pd.read_csv(io.BytesIO(data), low_memory=False)
    if not {'runtime', 'ftime', 'station', 'model', 'tmp', 'dpt', 'wdr', 'wsp', 'p06', 'q06'}.issubset(df):
        raise ValueError('Invalid MOS source schema')
    if len(df) and (set(df.station) != {station} or set(df.model) != {'GFS'}):
        raise ValueError('MOS station or forecast model mismatch')
    path.write_bytes(gzip.compress(data, mtime=0))
    meta.write_text(json.dumps(dict(url=url, retrieved_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        raw_sha256=hashlib.sha256(data).hexdigest(), compressed_sha256=hash_file(path),
        rows=len(df), bytes_uncompressed=len(data), headers=headers,
        archive_kind='Issued operational NOAA GFS MOS guidance via IEM; original receipt time unavailable'), indent=2))
    print(json.dumps({'downloaded': name, 'rows': len(df), 'bytes': len(data)}), flush=True)
    return path


def canonical(frame, cycle_hours=(0,)):
    d = frame.copy()
    d['runtime'] = pd.to_datetime(d.runtime, utc=True, format='mixed')
    d['ftime'] = pd.to_datetime(d.ftime, utc=True, format='mixed')
    if cycle_hours is not None:
        d = d[d.runtime.dt.hour.isin(cycle_hours)]
    d = d.sort_values(['runtime', 'ftime']).drop_duplicates(['runtime', 'ftime'], keep='last')
    for c in ['tmp','dpt','wdr','wsp','p06','q06']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['temperature_c'] = (d.tmp.where(d.tmp.ne(999)) - 32) * 5 / 9
    d['dewpoint_c'] = (d.dpt.where(d.dpt.ne(999)) - 32) * 5 / 9
    direction = d.wdr.where(d.wdr.between(0,360))
    d['wind_speed_mps'] = d.wsp.where(d.wsp.between(0,98)) * .514444
    theta = np.deg2rad(direction)
    d['wind_u_mps'] = -d.wind_speed_mps * np.sin(theta)
    d['wind_v_mps'] = -d.wind_speed_mps * np.cos(theta)
    d['precip_probability_6h'] = d.p06.where(d.p06.between(0,100)) / 100
    d['qpf_category_6h'] = d.q06.where(d.q06.between(0,6))
    d['lead_hours'] = (d.ftime - d.runtime).dt.total_seconds() / 3600
    return d


def bounded_mean(hours, values, lo, hi):
    """One-window trapezoid mean; never extrapolate or bridge missing values."""
    before, after = np.flatnonzero(hours <= lo), np.flatnonzero(hours >= hi)
    if not len(before) or not len(after):
        return np.nan
    i, j = before[-1], after[0]
    x, y = hours[i:j+1], values[i:j+1]
    if len(x) < 2 or not np.isfinite(y).all() or np.diff(x).max(initial=0) > 6.01:
        return np.nan
    knots = np.r_[lo, x[(x > lo) & (x < hi)], hi]
    return float(np.trapezoid(np.interp(knots, x, y), knots) / (hi-lo))


def sequence(group, clock, issue_clock=10+50/60):
    """48 hourly-window forecast features at the declared reporting clock."""
    if clock < issue_clock:
        raise ValueError('Target window begins before forecast issue')
    group = group.sort_values('lead_hours')
    h = group.lead_hours.to_numpy(float)
    result = np.full((48, len(FEATURES)), np.nan, dtype='float32')
    p = group.precip_probability_6h.to_numpy(float)
    q = group.qpf_category_6h.to_numpy(float)
    valid = np.isfinite(p) & np.isfinite(q)
    ph, pp, qq = h[valid], p[valid], q[valid]
    if len(ph) > 1 and np.diff(ph).min() < 5.99:
        raise ValueError('Overlapping six-hour precipitation guidance')
    for i in range(48):
        lo, hi = clock+i, clock+i+1
        for j, name in enumerate(NATIVE):
            result[i,j] = bounded_mean(h, group[name].to_numpy(float), lo, hi)
        overlaps = np.maximum(0., np.minimum(ph, hi)-np.maximum(ph-6, lo))
        if np.isclose(overlaps.sum(), 1.):
            result[i,5] = np.sum(overlaps*pp)
            result[i,6] = np.sum(overlaps*qq)
            result[i,7:14] = [np.sum(overlaps*(qq==category)) for category in range(7)]
        result[i,14] = hi
        result[i,15] = hi-issue_clock
    return result


def station_frames(out=OUT):
    records, result = [], {}
    for station in sorted({x[0] for x in SITES.values()}):
        sources = [OLD / f'mos_{station}_00utc.csv.gz', *sorted((out/'raw').glob(f'{station}_*.csv.gz'))]
        frames = []
        for path in sources:
            d = pd.read_csv(path, low_memory=False)
            frames.append(d)
            records.append(dict(station=station, path=str(path.relative_to(ROOT)), sha256=hash_file(path), rows=len(d)))
        d = canonical(pd.concat(frames, ignore_index=True))
        if set(d.station) != {station} or set(d.model) != {'GFS'}:
            raise ValueError('Source identity changed during canonicalization')
        d.to_csv(out/f'mos_{station}_00utc.csv.gz', index=False)
        result[station] = d
    return result, records


def prepare(out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    stations, sources = station_frames(out)
    cohort = pd.read_csv(ROOT/'data/alta_nowcast/radar/cohort_dates.csv', dtype={'case_id':str})
    indices, coverage = [], []
    for site, (station, clock) in SITES.items():
        seqs, rows = [], []
        for run, group in stations[station].groupby('runtime', sort=True):
            start = run+pd.Timedelta(hours=clock); end=start+pd.Timedelta(hours=48)
            issue = run+pd.Timedelta(hours=10,minutes=50)
            if run+pd.Timedelta(hours=6) > issue:
                raise ValueError('Forecast assumed availability follows issue')
            x=sequence(group,clock)
            row=dict(site=site,row=len(rows),case_id=run.strftime('%Y%m%d'),station=station,
                run_time_utc=run.isoformat(),assumed_available_utc=(run+pd.Timedelta(hours=6)).isoformat(),
                forecast_issue_utc=issue.isoformat(),valid_start_utc=start.isoformat(),valid_end_utc=end.isoformat(),
                day1_end_utc=(start+pd.Timedelta(hours=24)).isoformat(),
                target_date_1=str((run+pd.Timedelta(days=1)).date()),target_date_2=str((run+pd.Timedelta(days=2)).date()),
                full_meteorological_coverage=bool(np.isfinite(x[:,:14]).all()),
                valid_meteorological_fraction=float(np.isfinite(x[:,:14]).mean()),
                winter=int(run.year-(run.month<7)),archive_type='operational_forecast')
            seqs.append(x);rows.append(row)
        xs=np.asarray(seqs,dtype='float32')
        frame=pd.DataFrame(rows)
        np.savez_compressed(out/f'broad_{site}.npz',forecast=xs,observed=np.isfinite(xs),
            feature_names=np.asarray(FEATURES),case_ids=frame.case_id.to_numpy(dtype='U8'),
            valid_start_utc=frame.valid_start_utc.to_numpy(dtype='U40'),valid_end_utc=frame.valid_end_utc.to_numpy(dtype='U40'))
        frame.to_csv(out/f'broad_{site}_index.csv',index=False)
        indices.append(frame)
        mapping=frame.set_index('case_id')['row'].to_dict()
        selected=np.full((len(cohort),48,len(FEATURES)),np.nan,dtype='float32')
        found=np.zeros(len(cohort),dtype=bool)
        for i,c in enumerate(cohort.case_id):
            if c in mapping:selected[i]=xs[mapping[c]];found[i]=True
        np.savez_compressed(out/f'radar_cohort_{site}.npz',forecast=selected,observed=np.isfinite(selected),
            source_present=found,feature_names=np.asarray(FEATURES),case_ids=cohort.case_id.to_numpy(dtype='U8'))
        valid=np.isfinite(selected[:,:,:14]).all(axis=(1,2))
        coverage.append(dict(site=site,station=station,broad_runs=len(frame),broad_complete=int(frame.full_meteorological_coverage.sum()),
            first_run=frame.run_time_utc.min(),last_run=frame.run_time_utc.max(),radar_requested=len(cohort),
            radar_source_present=int(found.sum()),radar_complete=int(valid.sum()),
            radar_by_split={s:{'scheduled':int((cohort.split==s).sum()),'complete':int(valid[cohort.split.eq(s)].sum())} for s in cohort.split.unique()}))
    pd.concat(indices,ignore_index=True).to_csv(out/'broad_index.csv',index=False)
    pd.DataFrame([{k:v for k,v in x.items() if k!='radar_by_split'} for x in coverage]).to_csv(out/'coverage.csv',index=False)
    metadata=dict(generated_utc=pd.Timestamp.now(tz='UTC').isoformat(),sources=sources,coverage=coverage,
        feature_names=FEATURES,shape='forecast[N,48,16]; observed is finite-input mask, not meteorological observation label',
        forecast_only=True,source_docs='https://mesonet.agron.iastate.edu/mos/',
        source_native_resolution='3-hour instantaneous forecasts early, 6-hour later; precipitation PoP/QPF categorical 6-hour windows',
        hourly_policy='Instantaneous variables bounded linear/trapezoid mean per hour; six-hour PoP/category features overlap-weighted over each hour; no extrapolation',
        precipitation_warning='Six-hour probability is a predictor, not an hourly precipitation probability. QPF category is ordinal/categorical, not inches. No source snow depth used as truth.',
        issue_policy='00UTC model initialized; assumed available06UTC; common radar issue10:50UTC. Original archive receipt timestamps unavailable.',
        reporting_clocks_utc={s:c for s,(_,c) in SITES.items()},
        reporting_clock_warning='Alta11UTC documented; Vail11UTC source-clock assumption. New Stowe0.2SW proxy07:30EST ->12:30UTC and Westfield0.7WNW Jay proxy07:00EST ->12UTC; target builder must restrict to standard-time dates or match true observation clocks. Site-specific48h daily targets begin at these clocks, not at common issue.',
        missing_policy='NaN remains missing; explicit finite masks; downstream fit-only imputation required',
        code_sha256=hash_file(__file__))
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'prepared':str(out),'coverage':coverage}),flush=True)
    return metadata


def prepare_cached_alta_hrrr(out=OUT):
    """Optional seven-variable prefix; missing last5h remain explicit NaN."""
    source=ROOT/'data/alta_nowcast/hrrr/hourly_mean3x3.csv'
    data=pd.read_csv(source)
    data['run_time_utc']=pd.to_datetime(data.run_time_utc,utc=True)
    data['valid_time_utc']=pd.to_datetime(data.valid_time_utc,utc=True)
    cohort=pd.read_csv(ROOT/'data/alta_nowcast/radar/cohort_dates.csv',dtype={'case_id':str})
    fields=['precip_mm','snow_water_mm','temperature_k','dewpoint_k','wind_u_ms','wind_v_ms','relative_humidity_pct']
    result=np.full((len(cohort),48,len(fields)),np.nan,dtype='float32')
    runs={rt:g.set_index('valid_time_utc') for rt,g in data.groupby('run_time_utc')}
    for i,row in enumerate(cohort.itertuples()):
        day=pd.Timestamp(row.date,tz='UTC');run=day+pd.Timedelta(hours=6)
        if run not in runs:continue
        wanted=pd.date_range(day+pd.Timedelta(hours=12),periods=48,freq='h')
        result[i]=runs[run].reindex(wanted)[fields].to_numpy('float32')
    np.savez_compressed(out/'radar_cohort_alta_hrrr_prefix.npz',forecast=result,observed=np.isfinite(result),
        feature_names=np.asarray(fields),case_ids=cohort.case_id.to_numpy(dtype='U8'))
    metadata=dict(source=str(source.relative_to(ROOT)),source_sha256=hash_file(source),
        variables=fields,units=['mm per preceding hour','mm snow-water per preceding hour','K','K','m/s','m/s','percent'],
        run_utc_hour=6,assumed_publication_buffer_hours=2,forecast_issue_utc_hour=10+50/60,
        forecast_window_start_utc_hour=11,horizon_requested_hours=48,
        scope='3x3 HRRR grid mean near Alta; operational forecasts, not observations',
        target_alignment='Forecast hourly totals end at12UTC through nextsecond11UTC; instantaneous fields use endpoint value',
        missing_policy='HRRR initializerlead48 coversonly43hours after11UTC; last5hours remainNaN. Earlyv3runscoveronly31hours. No padding/extrapolation.',
        finite_fraction_by_lead=np.isfinite(result).mean(axis=(0,2)).tolist(),
        source_manifest=str((ROOT/'data/alta_nowcast/hrrr/manifest.json').relative_to(ROOT)))
    (out/'alta_hrrr_prefix_manifest.json').write_text(json.dumps(metadata,indent=2))
    return metadata


def prepare_dst_and_live_variants(out=OUT):
    """Extra DST clock arrays and recent00/12UTC cycles; basearrays unchanged."""
    cohort=pd.read_csv(ROOT/'data/alta_nowcast/radar/cohort_dates.csv',dtype={'case_id':str})
    native=pd.read_csv(out/'mos_KMVL_00utc.csv.gz',low_memory=False)
    native['runtime']=pd.to_datetime(native.runtime,utc=True)
    groups={rt.strftime('%Y%m%d'):g for rt,g in native.groupby('runtime')}
    variants={}
    for site,clock in [('mansfield',11.5),('jay',11.)]:
        result=np.full((len(cohort),48,len(FEATURES)),np.nan,dtype='float32');rows=[]
        for i,row in enumerate(cohort.itertuples()):
            day=pd.Timestamp(row.date,tz='UTC');start=day+pd.Timedelta(hours=clock)
            if row.case_id in groups:result[i]=sequence(groups[row.case_id],clock)
            rows.append({'case_id':row.case_id,'row':i,'site':site,'valid_start_utc':start.isoformat(),
                'day1_end_utc':(start+pd.Timedelta(hours=24)).isoformat(),'valid_end_utc':(start+pd.Timedelta(hours=48)).isoformat(),
                'clock_utc_hour':clock,'clock_kind':'daylight-time variant; target builder chooses actual local civil clock'})
        np.savez_compressed(out/f'radar_cohort_{site}_dst.npz',forecast=result,observed=np.isfinite(result),
            source_present=np.array([c in groups for c in cohort.case_id]),feature_names=np.asarray(FEATURES),
            case_ids=cohort.case_id.to_numpy(dtype='U8'))
        pd.DataFrame(rows).to_csv(out/f'radar_cohort_{site}_dst_index.csv',index=False)
        variants[site]={'clock_utc_hour':clock,'cases':len(cohort),'complete':int(np.isfinite(result[:,:,:14]).all(axis=(1,2)).sum())}
    live={}
    for station in ['KSLC','KLXV','KMVL']:
        paths=sorted((out/'raw').glob(f'{station}_2026-11-*.csv.gz'))
        current=canonical(pd.concat([pd.read_csv(p,low_memory=False) for p in paths],ignore_index=True),cycle_hours=None)
        wanted=current[current.runtime.dt.hour.isin([0,12])].groupby(current.runtime.dt.hour).runtime.max()
        latest=current[current.runtime.isin(wanted.to_list())].sort_values(['runtime','ftime'])
        latest.to_csv(out/f'latest_{station}_00_12utc.csv.gz',index=False)
        live[station]={'runs':sorted(str(x) for x in latest.runtime.unique()),'rows':len(latest),
            'raw_sources':[{'path':str(p.relative_to(ROOT)),'sha256':hash_file(p)} for p in paths],
            'note':'12UTCcycle has identical variable schema but is not part of the00UTC-only training cohort; use documented6h availability assumption.'}
    m={'dst_variants':variants,'latest_cycles':live,'base_arrays_unchanged':True,
        'code_sha256':hash_file(__file__),'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'target_clock_policy':'Use station localcivil observation times; discard transitions where the paired target is not48h. Select standard/DST arrays against target start UTC.'}
    (out/'variants_manifest.json').write_text(json.dumps(m,indent=2))
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--fetch',action='store_true');p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.fetch:
        for year in range(2020,2024):
            fetch('KMVL',f'{year}-10-01T00:00Z',f'{year+1}-05-01T00:00Z',args.out)
        now=pd.Timestamp.now(tz='UTC')
        for station in ['KSLC','KLXV','KMVL']:
            fetch(station,(now-pd.Timedelta(days=3)).strftime('%Y-%m-%dT00:00Z'),
                  now.strftime('%Y-%m-%dT%H:%MZ'),args.out)
    prepare(args.out)
    prepare_cached_alta_hrrr(args.out)
    prepare_dst_and_live_variants(args.out)


if __name__=='__main__':main()
