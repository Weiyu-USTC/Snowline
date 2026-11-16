"""Isolated live Alta radar inference for the local dashboard.

Only fresh public radar/issued HRRR inputs are fetched. Frozen checkpoints and
benchmark artifacts are read-only. Daily snow probabilities are deliberately
left to the existing daily model; raw radar spread is not snowfall uncertainty.
"""
from __future__ import annotations
import argparse
import copy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/dashboard/alta'
OUT = ROOT / 'outputs/dashboard/alta'
FROZEN = ROOT / 'outputs/alta-nowcast'
LABELS = {'convlstm': 'ConvLSTM (compact)', 'dgmr': 'DGMR adaptation',
          'nowcastnet': 'NowcastNet adaptation', 'hrrr': 'HRRR'}
WARNINGS = [
    'Experimental 1–3-hour liquid precipitation, not fresh-snow depth or powder-day probability.',
    'Raw neural 10th–90th percentile bands are uncalibrated and can be too narrow.',
    'Models were trained on morning windows. Other issue hours have not been independently validated.',
    'Mountain radar coverage is incomplete; this is regional guidance near Alta–Collins.',
]


def utc_now():
    return datetime.now(timezone.utc)


def _parse(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def _age(payload):
    out = copy.deepcopy(payload)
    age = max(0., (utc_now() - _parse(out['issued_at'])).total_seconds()/60)
    out.update(age_minutes=round(age, 1), stale=age > 30,
               expired=utc_now() >= _parse(out['end']))
    if out['stale']:
        out['status'] = 'stale'
    return out


def get_cached_alta_nowcast():
    """Read a saved forecast, retaining its real issue/valid times and age."""
    path = OUT / 'latest.json'
    if path.exists():
        try:
            return _age(json.loads(path.read_text()))
        except (ValueError, KeyError, OSError):
            pass
    # Read-only demonstration snapshot from the completed experiment. It must
    # never masquerade as a new retrieval or get a new issued timestamp.
    mf, cf = FROZEN/'current_manifest.json', FROZEN/'current_short_curves.csv'
    if not mf.exists() or not cf.exists():
        return None
    m = json.loads(mf.read_text())
    start = _parse(m['start_utc'])
    models = []
    with cf.open() as f:
        rows = list(csv.DictReader(f))
    for model, label in LABELS.items():
        points = []
        for row in rows:
            if row['model'] != model:
                continue
            point = {key: float(row[key]) for key in ('hours', 'mean_liquid_mm', 'q10_liquid_mm', 'q90_liquid_mm')}
            point['time'] = (start + timedelta(hours=point['hours'])).isoformat()
            points.append(point)
        if points:
            models.append({'id': model, 'name': label, 'hours': points})
    return _age({'status': 'cached', 'issued_at': m['forecast_issue_utc'],
                 'fetched_at': m['generated_utc'], 'start': start.isoformat(),
                 'end': (start+timedelta(hours=3)).isoformat(), 'horizon_hours': 3,
                 'unit': 'mm liquid equivalent', 'source': 'Saved MRMS + HRRR research snapshot',
                 'models': models, 'warnings': WARNINGS + ['Saved research snapshot; refresh to retrieve new inputs.'],
                 'checkpoint_sha256': m['checkpoint_sha256']})


get_alta_nowcast = get_cached_alta_nowcast


def refresh_alta_nowcast(progress=None):
    """Bounded worker acquisition/inference; raises without destroying cache."""
    if progress:
        progress('Downloading recent Alta radar and issued HRRR precipitation; running three frozen models…')
    interpreter = ROOT/'.venv-hrrr/bin/python'
    if not interpreter.exists():
        interpreter = ROOT/'.venv/bin/python'
    if not interpreter.exists():
        interpreter = Path(sys.executable)
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='alta-radar-job-', dir=OUT) as tmp:
        result = Path(tmp)/'result.json'
        try:
            run = subprocess.run([str(interpreter), str(Path(__file__).resolve()), '--worker', str(result)],
                                 cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, timeout=240)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Radar refresh exceeded four minutes; the previous forecast was retained.') from exc
        if run.returncode or not result.exists():
            reason = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else 'No forecast produced.'
            raise RuntimeError('Radar refresh unavailable: '+reason[:500])
        payload = json.loads(result.read_text())
    _atomic_json(OUT/'latest.json', payload)
    if progress:
        progress('Alta radar inference complete.')
    return _age(payload)


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.', delete=False) as f:
        json.dump(payload, f, allow_nan=False, indent=2)
        temp = Path(f.name)
    temp.replace(path)


def integrate_hrrr(periods, values, run_time, start, hours):
    """Accumulate hourly totals using exact fractional-hour overlap."""
    import numpy as np
    ends = np.asarray([run_time.timestamp()+float(p)*3600 for p in periods])
    begins = ends-3600
    out=[]
    for h in hours:
        seconds=np.maximum(0, np.minimum(ends, start.timestamp()+h*3600)-np.maximum(begins, start.timestamp()))
        if not np.isclose(seconds.sum(), h*3600):
            raise ValueError('HRRR does not cover the forecast window.')
        use=seconds>0
        if not np.isfinite(np.asarray(values)[use]).all():
            raise ValueError('Missing HRRR accumulation; cannot substitute zero.')
        out.append(float(np.sum(np.asarray(values)[use]*seconds[use]/3600)))
    return out


def _worker():
    import contextlib
    import numpy as np
    import pandas as pd
    import requests
    import torch
    import fetch_alta_radar as radar
    import fetch_alta_hrrr as hrrr
    from alta_radar_models import load_bundle, predict_ensemble
    torch.set_num_threads(2)
    radar.OUT = DATA/'radar'
    # A live refresh has strict network bounds and no long automatic retries.
    def bounded_radar_get(url, retries=1):
        response=radar.session().get(url, timeout=(8, 35))
        response.raise_for_status()
        return response.content, dict(response.headers)
    radar.get_bytes=bounded_radar_get
    def bounded_hrrr_get(key, cache, retries=1):
        path=cache/key
        if path.exists():
            data=path.read_bytes()
        else:
            response=requests.get(hrrr.BASE+key, timeout=(8, 25))
            response.raise_for_status()
            data=response.content
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp=path.with_name(path.name+'.tmp')
            tmp.write_bytes(data);tmp.replace(path)
        return data, {'url':hrrr.BASE+key,'cache_path':str(path.relative_to(cache.parent)),
                      'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    hrrr.get=bounded_hrrr_get
    now=utc_now()
    issue=now.replace(minute=now.minute//10*10, second=0, microsecond=0)
    context=None
    radar_error=None
    for candidate in (issue, issue-timedelta(minutes=10)):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                context=radar.fetch_current(candidate, workers=4)
            issue=candidate
            break
        except (requests.RequestException, ValueError, OSError) as exc:
            radar_error=exc
    if context is None:
        raise RuntimeError(f'Recent radar unavailable: {radar_error}')
    start=issue+timedelta(minutes=10)
    run_time=issue.replace(minute=0)-timedelta(hours=2)
    hr=None
    for candidate in (run_time, run_time-timedelta(hours=1), run_time-timedelta(hours=2)):
        try:
            hr=hrrr.field(pd.Timestamp(candidate), 'precip_mm', out=DATA/'hrrr')
            run_time=candidate
            break
        except (requests.RequestException, ValueError, OSError) as exc:
            hrrr_error=exc
    if hr is None:
        raise RuntimeError(f'Recent issued HRRR unavailable: {hrrr_error}')
    with np.load(context) as z:
        rate=z['rate'].copy();quality=z['rqi'].copy();valid=z['valid_fraction']
        bad=(valid<.9)|(~np.isfinite(quality))|(quality<.1)
        rate[bad]=np.nan;quality[bad]=0
        x=np.stack([rate,quality],axis=2)
        iy=int(np.argmin(abs(z['latitude']-40.57607)))
        ix=int(np.argmin(abs(z['longitude']+111.63889)))
    local_valid=float(np.isfinite(rate[:,:,iy-1:iy+2,ix-1:ix+2]).mean())
    regional_valid=float(np.isfinite(rate).mean())
    if regional_valid < .1 or local_valid < .5:
        raise ValueError('Insufficient current radar coverage near Alta; use hourly weather guidance.')
    hours=np.arange(19)/6
    models=[];hashes={}
    for name,label in LABELS.items():
        if name=='hrrr':continue
        weights=FROZEN/'models/selected'/f'{name}.pt'
        hashes[name]=hashlib.sha256(weights.read_bytes()).hexdigest()
        ensemble=predict_ensemble(load_bundle(weights),x,members=8,batch_size=1,seed=20261105)[0]
        local=ensemble[:,1:20,iy-1:iy+2,ix-1:ix+2].mean((2,3))
        cumulative=np.c_[np.zeros(len(local)),np.cumsum((local[:,1:]+local[:,:-1])/12,axis=1)]
        points=[]
        for j,h in enumerate(hours):
            points.append({'time':(start+timedelta(hours=float(h))).isoformat(),'hours':float(h),
                           'mean_liquid_mm':float(cumulative[:,j].mean()),
                           'q10_liquid_mm':float(np.quantile(cumulative[:,j],.1)),
                           'q90_liquid_mm':float(np.quantile(cumulative[:,j],.9))})
        models.append({'id':name,'name':label,'hours':points})
    # HRRR is regridded to the same radar-cell neighborhood, not a different
    # native 3x3 area. Source hourly totals are integrated, never summed twice.
    hvalues=hr['extra']['mrms64'][:,iy-1:iy+2,ix-1:ix+2].mean((1,2))
    accum=integrate_hrrr(hr['periods'], hvalues, run_time,start,hours)
    models.append({'id':'hrrr','name':'HRRR','hours':[
        {'time':(start+timedelta(hours=float(h))).isoformat(),'hours':float(h),
         'mean_liquid_mm':value,'q10_liquid_mm':None,'q90_liquid_mm':None}
        for h,value in zip(hours,accum)]})
    result={'status':'fresh','issued_at':issue.isoformat(),'fetched_at':utc_now().isoformat(),
            'start':start.isoformat(),'end':(start+timedelta(hours=3)).isoformat(),
            'horizon_hours':3,'unit':'mm liquid equivalent','source':'NOAA MRMS + issued HRRR via University of Utah',
            'sources':[{'name':'MRMS','url':'https://www.nssl.noaa.gov/projects/mrms/'},
                       {'name':'HRRR archive','url':'https://mesowest.utah.edu/html/hrrr/zarr_documentation/'}],
            'hrrr_run_utc':run_time.isoformat(),'last_radar_valid_utc':(issue-timedelta(minutes=10)).isoformat(),
            'assumed_radar_latency_minutes':10,'assumed_hrrr_latency_hours':2,
            'quality':{'regional_valid_fraction':regional_valid,'alta_neighborhood_valid_fraction':local_valid},
            'models':models,'warnings':WARNINGS,'checkpoint_sha256':hashes,
            'retrained':False,'context_path':str(context.relative_to(ROOT)),
            'hrrr_provenance':hr['metadata']['provenance']}
    snapshot=OUT/issue.strftime('%Y%m%d-%H%MZ')/'forecast.json'
    _atomic_json(snapshot,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--refresh',action='store_true')
    args=p.parse_args()
    if args.worker:
        _atomic_json(args.worker,_worker())
    elif args.refresh:
        print(json.dumps(refresh_alta_nowcast(),indent=2))
    else:
        print(json.dumps(get_cached_alta_nowcast(),indent=2))
