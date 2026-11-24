"""One bounded, all-mountain refresh using frozen model weights.

This is acquisition and inference only. Each source has its own outcome; a
failed source retains its previous snapshot and original issue timestamps.
Heavy dependencies run in the existing project environments, isolated from
the lightweight local HTTP server. No optimizer is invoked on refresh.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from snowline_sites import SITES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/dashboard/deployed-refresh'
LATEST = OUT/'latest.json'
STAGES = {
    'mammoth_provider': ('default', 120, 'Fetching ECMWF snow-water forecasts for Mammoth…'),
    'mammoth_weather': ('hrrr', 180, 'Fetching HRRR forecasts for the Jeff Dozier Snow Study Site…'),
    'mammoth_model': ('default', 90, 'Running the hourly Mammoth snowfall model…'),
    'mammoth_radar': ('default', 150, 'Updating NOAA radar at Mammoth Mountain…'),
    'mammoth_observations': ('default', 90, 'Updating Main Lodge snowfall reports…'),
    'providers': ('default', 360, 'Updating weather forecasts for all mountains…'),
    'hrrr': ('hrrr', 420, 'Updating hourly HRRR forecasts for the four modeled mountains…'),
    'nexrad_acquisition': ('nexrad', 360, 'Downloading radar near Alta, Vail, Stowe and Jay Peak…'),
    'observations': ('default', 150, 'Reading the latest mountain observations…'),
    'seasonal_observations': ('default', 150, 'Updating seasonal snowfall observations…'),
    'alta_radar': ('hrrr', 280, 'Updating the saved Alta radar models…'),
    'neural48': ('default', 480, 'Running the selected 48-hour models for the four modeled mountains…'),
    'nexrad_models': ('nexrad', 360, 'Combining the new radar and HRRR forecasts…'),
    'temperature': ('default', 120, 'Refining mountain temperatures with the saved models…'),
    'portfolio': ('default', 150, 'Updating the deployed forecasts and comparison report…'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def atomic_frame(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    frame.to_csv(temp, index=False, compression='gzip' if path.suffix == '.gz' else None)
    temp.replace(path)


def interpreter(kind):
    path = (ROOT/f'.venv-{kind}/bin/python') if kind in {'hrrr', 'nexrad'} else Path(sys.executable)
    # Public checkout uses one complete environment; older research checkouts
    # can keep their separate radar/GRIB environments.
    if not path.is_file():
        path = ROOT/'.venv/bin/python'
    if not path.is_file():
        path = Path(sys.executable)
    if not path.is_file():
        raise RuntimeError(f'The saved {kind} forecast environment is unavailable.')
    return path


def run_stage(name, folder, progress=None, deadline=None):
    """Run a known inference stage with a timeout covering all descendants."""
    kind, timeout, message = STAGES[name]
    if progress: progress(message, source=name, source_status='running')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    attempt=1
    while (folder/f'{name}{"" if attempt==1 else "_attempt"+str(attempt)}.json').exists():attempt+=1
    stem=name+('' if attempt==1 else '_attempt'+str(attempt))
    result_path = folder/f'{stem}.json'; log_path = folder/f'{stem}.log'
    started = now(); process = None
    try:
        if deadline is not None:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise RuntimeError('The refresh time limit was reached; saved source timestamps were retained.')
            timeout=min(timeout,remaining)
        command = [str(interpreter(kind)), str(Path(__file__).resolve()), '--stage', name, '--result', str(result_path)]
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=env)
            try: return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill the process group: the older Alta wrapper also launches
                # a child process, which must not keep writing after a timeout.
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
                raise RuntimeError('This source did not finish within the refresh time limit; saved forecasts were retained.')
        if not result_path.exists():
            tail = log_path.read_text(errors='replace').strip().splitlines()
            reason = tail[-1] if tail else 'No forecast result was produced.'
            raise RuntimeError(reason[:450])
        outcome = json.loads(result_path.read_text())
        if not isinstance(outcome,dict) or outcome.get('status') not in {'succeeded', 'partial', 'failed'}:
            raise ValueError('Invalid source status')
        # A stage deliberately exits nonzero after writing a structured failure.
        # Preserve its actual explanation instead of the final "}" in its log.
        if return_code and outcome['status']!='failed':
            raise RuntimeError(f'The {name} source exited unexpectedly (code {return_code}); its reported success was not accepted.')
    except Exception as exc:
        outcome = {'status': 'failed', 'message': str(exc), 'errors': [str(exc)]}
    logged_path=log_path.relative_to(ROOT) if log_path.is_relative_to(ROOT) else log_path
    outcome.update(source=name, started_at=started, completed_at=now(), log_path=str(logged_path))
    atomic_json(result_path, outcome)
    if progress: progress(outcome.get('message', message), source=name, source_status=outcome['status'])
    return outcome


def refresh_all(progress=None):
    """One shared acquisition per source for the whole collection."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    folder = OUT/stamp; folder.mkdir(parents=True, exist_ok=True)
    started = now(); sources = {};deadline=time.monotonic()+15*60
    def group(names):
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix='snow-source') as pool:
            pending = {pool.submit(run_stage, name, folder, progress, deadline): name for name in names}
            for future in as_completed(pending):
                name = pending[future]
                try: sources[name] = future.result()
                except Exception as exc: sources[name] = {'source':name,'status':'failed','message':str(exc),'errors':[str(exc)]}
    # The first group writes disjoint source caches. Model inference follows
    # only after those acquisitions finish, so there is no cross-source race.
    group(['providers', 'hrrr', 'nexrad_acquisition', 'observations', 'seasonal_observations',
           'mammoth_radar', 'mammoth_weather', 'mammoth_provider', 'mammoth_observations'])
    group(['mammoth_model', 'neural48', 'nexrad_models', 'temperature'])
    # Issue the shortest window last so it is still useful when the UI opens.
    sources['alta_radar']=run_stage('alta_radar',folder,progress,deadline)
    # Do not finish a batch with a now-started full-day neural window. This
    # happens only when acquisition crosses an hour boundary. A real new
    # inference, with its own issue and inputs, replaces that window once.
    retry=[]
    for name,path in [('neural48',ROOT/'outputs/nwp-weather/neural48-trained/current_forecast.json'),
                      ('nexrad_models',ROOT/'outputs/nexrad-short/current_forecast.json')]:
        try:
            snapshot=json.loads(path.read_text());clock=datetime.now(timezone.utc)
            entries=[r for r in snapshot['sites'].values() if r.get('start')]
            if entries and any(datetime.fromisoformat(r['start'].replace('Z','+00:00'))<clock for r in entries):retry.append(name)
        except (OSError,ValueError,KeyError):pass
    if retry:group(retry)
    sources['portfolio'] = run_stage('portfolio', folder, progress,deadline)
    errors = [{'source':name, 'message': '; '.join(str(e) for e in item.get('errors',[]) if e) or item.get('message', 'Source could not update.')}
              for name,item in sources.items() if item['status'] in {'failed','partial'}]
    successes = sum(item['status']=='succeeded' for item in sources.values())
    status = 'succeeded' if not errors else 'partial' if successes else 'failed'
    labels={'mammoth_provider':'Mammoth ECMWF','mammoth_model':'Mammoth hourly model','mammoth_weather':'Mammoth HRRR',
            'mammoth_radar':'Mammoth radar','mammoth_observations':'Mammoth snowfall reports',
            'hrrr':'HRRR','nexrad_acquisition':'radar downloads','nexrad_models':'short-range radar',
            'providers':'weather providers','neural48':'48-hour models','alta_radar':'Alta radar',
            'observations':'mountain observations','seasonal_observations':'seasonal snowfall observations',
            'temperature':'temperature corrections','portfolio':'forecast comparison'}
    station_labels = {'CLN': 'Alta Collins', '842:CO:SNTL': 'Vail SNOTEL', 'FEMC_MMW': 'Stowe FEMC'}
    missing_stations = [station_labels.get(station, station)
                        for station, state in sources.get('observations', {}).get('stations', {}).items()
                        if state.get('status') == 'failed']
    if missing_stations:
        labels['observations'] = ' and '.join(missing_stations) + ' observations'
    summary = ('All mountains updated.' if not errors else
               'Some sources could not update: '+', '.join(labels.get(e['source'],e['source']) for e in errors)+'. Available forecasts are shown.')
    payload = {'status':status, 'started_at':started, 'completed_at':now(), 'sites':list(SITES),
               'sources':sources, 'errors':errors, 'retrained':False,
               'message':summary}
    atomic_json(folder/'summary.json', payload); atomic_json(LATEST, payload)
    return payload


def providers_stage():
    from dashboard_forecasts import get_forecast
    results = {}; errors = []
    # Same process shares the existing MOS download lock for Vermont's common
    # airport. Different mountain snapshot files are otherwise independent.
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(get_forecast, site, True):site for site in SITES}
        for future in as_completed(tasks):
            site=tasks[future]
            try:
                data=future.result(); results[site]={'status':data.get('status'),
                    'hourly_source':data.get('source',{}),'daily_source':data.get('daily_source',{}),
                    'last_refresh_attempt_at':data.get('last_refresh_attempt_at')}
                if data.get('status') in {'partial','unavailable','stale'}:
                    errors.append(f"{site}: "+'; '.join(data.get('errors',[]) or data.get('warnings',[]) or ['Some weather sources were retained.']))
            except Exception as exc:
                results[site]={'status':'failed','error':str(exc)};errors.append(f'{site}: {exc}')
    return {'status':'partial' if errors else 'succeeded','message':'Weather providers updated for all mountains.' if not errors else 'Some weather-provider data could not update.', 'sites':results,'errors':errors}


def hrrr_stage():
    import pandas as pd
    import hrrr_portfolio as hrrr
    info=hrrr.current()
    # Alta's trained fixed-window model uses a particular06UTC run; never
    # fetch a future cycle just to make that model appear available.
    issue=pd.Timestamp.now(tz='UTC');starts=pd.date_range(issue.normalize(),periods=4,freq='12h')+pd.Timedelta(hours=11)
    start=next(s for s in starts if s>issue);run=start.normalize()+pd.Timedelta(hours=6)
    alta_error=None
    if run+pd.Timedelta(hours=3)<=issue:
        try: hrrr.fetch_run(run)
        except Exception as exc: alta_error=str(exc)
    return {'status':'partial' if info.get('fallback_run') or alta_error else 'succeeded',
            'message':'HRRR updated for the four modeled mountains.', 'forecast':info,
            'alta12_scheduled_cycle_available':bool(run+pd.Timedelta(hours=3)<=issue),
            'errors':([alta_error] if alta_error else [])+info.get('attempt_errors',[])}


def nexrad_acquisition_stage():
    import nexrad_acquire as acquire
    previous=sys.argv;sys.argv=[str(ROOT/'src/nexrad_acquire.py'),'--current','--workers','5']
    try:acquire.main()
    finally:sys.argv=previous
    rows=json.loads((ROOT/'data/nexrad_short/current_acquisition.json').read_text())
    errors=[f"{r['site']}: {r.get('error','Radar unavailable')}" for r in rows if r['status']!='ok']
    return {'status':'partial' if errors else 'succeeded','message':'Radar downloads checked for the four modeled mountains.',
            'radars':['KMTX','KFTG','KGJX','KCXX'],'acquisitions':rows,'errors':errors}


def observations_stage():
    import pandas as pd
    from prepare_mountain_hourly import fetch_current_observations
    path=ROOT/'data/mountain_hourly/observations.csv.gz';frames=[];details={};errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks={pool.submit(fetch_current_observations,s):s for s in ['CLN','842:CO:SNTL','FEMC_MMW']}
        for future in as_completed(tasks):
            station=tasks[future]
            try:
                d = future.result()
                frames.append(d)
                details[station] = {'status': 'succeeded', 'rows': len(d),
                                    'latest_valid_time_utc': str(d.valid_time_utc.max())}
                if station == 'CLN':
                    details[station].update(feed=d.attrs.get('feed'),
                                            fields=[c for c in ('temperature_c','precipitation_mm','wet_bulb_c','snow_depth_cm') if d[c].notna().any()])
                    if 'primary_error' in d.attrs:
                        details[station]['primary_error'] = d.attrs['primary_error']
            except Exception as exc:details[station]={'status':'failed','error':str(exc)};errors.append(f'{station}: {exc}')
    if frames:
        fresh=pd.concat(frames,ignore_index=True);atomic_frame(ROOT/'data/mountain_hourly/current_observations.csv.gz',fresh)
        previous=pd.read_csv(path,low_memory=False) if path.exists() else pd.DataFrame()
        combined=pd.concat([previous,fresh],ignore_index=True)
        combined['valid_time_utc']=pd.to_datetime(combined.valid_time_utc,utc=True)
        # Keep richer MesoWest rows when the resort reports the same station-hour.
        if details.get('CLN', {}).get('feed') == 'Alta Ski Area':
            primary = combined.station_id.eq('CLN') & combined.source_url.str.startswith('https://mesowest.utah.edu/', na=False)
            redundant = combined.station_id.eq('CLN') & combined.source_url.eq('https://www.alta.com/weather') & combined.valid_time_utc.isin(combined.loc[primary, 'valid_time_utc'])
            combined = combined.loc[~redundant]
        combined=combined.drop_duplicates(['station_id','valid_time_utc'],keep='last').sort_values(['station_id','valid_time_utc'])
        atomic_frame(path,combined)
    return {'status':'partial' if errors and frames else 'failed' if errors else 'succeeded',
            'message':'Mountain observations checked.', 'stations':details,'errors':errors,
            'jay':'No verified summit observation source; regional weather remains explicitly separate.'}


def mammoth_provider_stage():
    from mammoth_weather import refresh_provider
    return refresh_provider(ROOT)


def mammoth_weather_stage():
    from mammoth_weather import refresh_current
    return refresh_current(ROOT)


def mammoth_model_stage():
    from mammoth_forecast import refresh_mammoth_model
    return refresh_mammoth_model(ROOT)


def mammoth_observations_stage():
    from mammoth_observations import refresh_mammoth_observations
    return refresh_mammoth_observations(ROOT)


def mammoth_radar_stage():
    from mammoth_radar import fetch_current_radar
    return fetch_current_radar(ROOT)


def seasonal_observations_stage():
    from season_observations import refresh_observations
    return refresh_observations(root=ROOT)


def alta_radar_stage():
    from dashboard_alta import refresh_alta_nowcast
    result=refresh_alta_nowcast()
    return {'status':'succeeded','message':'Alta radar models updated.','issued_at':result['issued_at'],'start':result['start'],'end':result['end']}


def neural48_stage():
    import pandas as pd
    from forecast_neural48 import refresh_all as refresh
    from dashboard_portfolio import deployment_neural48_families
    result=refresh(force=True,include_radar=True,family_overrides=deployment_neural48_families())
    # A long acquisition can straddle a clock boundary. Rebuild once with a
    # new real issue/start, never move an already-issued prediction in time.
    def started(result):
        return any(pd.Timestamp(r['start'])<pd.Timestamp.now(tz='UTC') for r in result['sites'].values() if r.get('status')!='unavailable')
    if started(result):result=refresh(force=True,include_radar=True,family_overrides=deployment_neural48_families())
    errors=[];details={}
    for site,r in result['sites'].items():
        details[site]={k:r.get(k) for k in ['status','issued_at','start','end','family','refresh_error']}
        if r.get('status')=='unavailable' or r.get('refresh_error') or (r.get('start') and pd.Timestamp(r['start'])<pd.Timestamp.now(tz='UTC')):
            errors.append(f"{site}: "+str(r.get('reason') or r.get('refresh_error') or 'The saved forecast window has started.'))
    return {'status':'partial' if errors else 'succeeded','message':'The four 48-hour model forecasts were updated.','sites':details,'errors':errors}


def merge_radar(previous, fresh):
    """Retain individual last-good windows, including their original issue."""
    retained=[]
    for site,new in fresh['sites'].items():
        old=(previous or {}).get('sites',{}).get(site)
        if not new.get('available') and old and old.get('available'):
            kept=dict(old);kept['refresh_error']=new.get('reason','New radar forecast unavailable.')
            fresh['sites'][site]=kept;retained.append(site)
    return fresh,retained


def nexrad_models_stage():
    import shutil
    import nexrad_train as radar
    import nexrad_portfolio as portfolio
    from datetime import timedelta
    case=json.loads((ROOT/'data/nexrad_short/current_case.json').read_text())
    issue=datetime.fromisoformat(case['forecast_issue_utc'].replace('Z','+00:00'))
    current=datetime.now(timezone.utc)
    if current-issue>timedelta(minutes=20) or issue.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)<=current:
        # Acquisition crossed the target's clock boundary. Fetch a genuinely
        # new scan sequence; do not move the old sequence's forecast times.
        nexrad_acquisition_stage()
    target=ROOT/'outputs/nexrad-short/current_forecast.json'
    previous=json.loads(target.read_text()) if target.exists() else None
    original=radar.OUT
    # Existing entry points read their weights from OUT. Copy only the frozen
    # inference artifacts into a temporary directory so publishing stays atomic.
    with tempfile.TemporaryDirectory(prefix='nexrad-inference-',dir=OUT) as tmp:
        tmp=Path(tmp)
        for name in ['selection.json','calibrators.pkl','portfolio_selection.json']:
            shutil.copyfile(original/name,tmp/name)
        old_radar,old_port=radar.OUT,portfolio.OUT;radar.OUT=portfolio.OUT=tmp
        try:
            radar.current();portfolio.apply_current()
            fresh=json.loads((tmp/'current_forecast.json').read_text())
        finally:radar.OUT,portfolio.OUT=old_radar,old_port
    fresh,retained=merge_radar(previous,fresh);atomic_json(target,fresh)
    unavailable=[s for s,r in fresh['sites'].items() if not r.get('available') and s!='vail']
    errors=[f'{s}: latest radar inference unavailable; original window retained.' for s in retained]
    errors += [f'{s}: '+fresh['sites'][s].get('reason','Radar unavailable.') for s in unavailable]
    return {'status':'partial' if errors else 'succeeded','message':'Short-range radar forecasts updated; Vail uses HRRR.',
            'issued_at':fresh['issued_at'],'retained_sites':retained,'errors':errors,
            'expected_fallbacks':{'vail':'Insufficient quality-qualified radar verification; HRRR is selected.'}}


def temperature_stage():
    import joblib
    import pandas as pd
    from train_mountain_temperature import make_features,predict_temperature
    w=pd.read_csv(ROOT/'data/hrrr_portfolio/current_weather.csv')
    obs=pd.read_csv(ROOT/'data/mountain_hourly/observations.csv.gz',low_memory=False)
    issue=pd.Timestamp.now(tz='UTC');details={};errors=[]
    for site,station in [('vail_snotel','842:CO:SNTL'),('stowe_mountain','FEMC_MMW')]:
        try:
            artifact=joblib.load(ROOT/f'outputs/mountain-temperature/{site}_operational.joblib')
            d=make_features(w,obs,site,station,issue)
            if d.empty:raise ValueError('No future forecast rows available.')
            d['corrected_temperature_c']=predict_temperature(d,artifact)
            atomic_frame(ROOT/f'outputs/mountain-temperature/{site}_current.csv',d)
            details[site]={'status':'succeeded','issued_at':issue.isoformat(),'rows':len(d),'model':artifact['selected_name']}
        except Exception as exc:errors.append(f'{site}: {exc}');details[site]={'status':'failed','error':str(exc)}
    return {'status':'partial' if errors else 'succeeded','message':'Mountain temperature forecasts updated.','sites':details,'errors':errors}


def portfolio_stage():
    from forecast_portfolio import build
    result=build()
    # Reports are a convenience, not a reason to lose a valid live snapshot.
    report_error=None
    try:
        from report_forecast_portfolio import build as report
        report()
    except Exception as exc:report_error=str(exc)
    unavailable=result.get('warnings',[]) if result.get('status')=='partial' else []
    return {'status':'partial' if unavailable else 'succeeded',
            'message':'Some forecast comparisons are unavailable; available predictions were updated.' if unavailable else 'The deployed forecast snapshots are ready.',
            'errors':unavailable,
            'generated_at':result['generated_at'],'start':result['start'],'end':result['end'],
            'alta12':result.get('alta12',{}),'report_error':report_error,'plot_error':result.get('plot_error'),
            'fallbacks':{'alta12':result['alta12'].get('reason')} if not result.get('alta12',{}).get('available') else {}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=STAGES);parser.add_argument('--result',type=Path)
    args=parser.parse_args()
    if args.stage:
        try:
            result=globals()[args.stage+'_stage']()
        except Exception as exc:
            result={'status':'failed','message':str(exc),'errors':[str(exc)]}
        if args.result:atomic_json(args.result,result)
        print(json.dumps(result,indent=2))
        if result['status']=='failed':raise SystemExit(1)
    else:
        result=refresh_all(progress=lambda message,**_:print(message,flush=True));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
