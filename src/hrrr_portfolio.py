"""Issued HRRR forecasts at mountain and verification-station points.

The Utah Zarr mirror supplies forecast weather fields, with missing chunks
retained as missing. Native cumulative ASNOW is read from official NOAA GRIB
and differenced only across adjacent forecast hours. It is forecast snowfall,
not observed fresh snow or a change in the model's snowpack depth.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
from pyproj import Proj
import requests
from fetch_alta_hrrr import PROJ, X0, Y0, DX, FIELDS, EXPECTED_UNITS, decode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/hrrr_portfolio'
OLD = ROOT/'data/alta_nowcast/hrrr/raw'
BASE = 'https://hrrrzarr.s3.amazonaws.com/'
NOAA = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/'
SITES = {
    'alta': (40.57607,-111.63889,2945.),
    'vail': (39.6176,-106.38,3140.),
    'mansfield': (44.464,-72.6907,223.1),
    'jay': (44.8916,-72.4435,325.8),
    'asos_slc': (40.77069,-111.96503,1288.),
    'asos_lxv': (39.2281,-106.3161,3026.),
    'asos_mvl': (44.5346,-72.6140,223.),
    'stowe_mountain': (44.5244,-72.8267,816.),
    'vail_snotel': (39.617,-106.383,3136.),
}
VARIABLES = ['precip_mm','snow_water_mm','temperature_k','dewpoint_k',
             'wind_u_ms','wind_v_ms','relative_humidity_pct']


def location(lat,lon):
    proj=Proj(**PROJ);x,y=proj(lon,lat)
    ix,iy=int(round((x-X0)/DX)),int(round((y-Y0)/DX))
    glon,glat=proj(X0+ix*DX,Y0+iy*DX,inverse=True)
    return {'grid_x':ix,'grid_y':iy,'chunk_x':ix//150,'chunk_y':iy//150,
            'internal_x':ix%150,'internal_y':iy%150,
            'grid_latitude':float(glat),'grid_longitude':float(glon),
            'distance_projected_m':float(np.hypot(x-X0-ix*DX,y-Y0-iy*DX))}


def download(url,path,headers=None):
    if path.exists():return path.read_bytes()
    for attempt in range(3):
        try:
            response=requests.get(url,headers=headers or {},timeout=(10,45))
            if response.status_code==404:raise FileNotFoundError(url)
            response.raise_for_status()
            if headers and 'Range' in headers and response.status_code!=206:
                raise ValueError('Server did not honor bounded GRIB range')
            body=response.content
            if len(body)>32_000_000:raise ValueError('Bounded forecast piece exceeded32MB')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(body)
            path.with_name(path.name+'.source.json').write_text(json.dumps({
                'url':url,'retrieved_utc':pd.Timestamp.now(tz='UTC').isoformat(),
                'request_headers':headers or {},'sha256':hashlib.sha256(body).hexdigest(),
                'bytes':len(body)},indent=2))
            return body
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(2**attempt)


def zget(key):
    old=OLD/key
    if old.exists():return old.read_bytes(),old
    p=OUT/'raw/zarr'/key
    return download(BASE+key,p),p


def field(run,name,sites=SITES):
    spec=FIELDS[name];prefix=run.strftime('sfc/%Y%m%d/%Y%m%d_%Hz_fcst.zarr/')+spec+'/'
    raw,mpath=zget(prefix+'.zmetadata');meta=json.loads(raw)['metadata']
    array=meta[spec+'/.zarray'];attrs=meta[spec+'/.zattrs']
    if len(array['shape'])!=3:raise ValueError('Forecast variable has no time axis')
    units=attrs.get('units')
    if units is None and name=='precip_mm' and attrs.get('GRIB_PARAM')==[2,0,1,8]:units='kg m-2'
    if units not in EXPECTED_UNITS[name]:raise ValueError(f'Unexpected{name}units:{units}')
    pbytes,ppath=zget(prefix+'forecast_period/0')
    periods=decode(pbytes,meta['forecast_period/.zarray'])[:array['shape'][0]]
    rbytes,rpath=zget(prefix+'forecast_reference_time/0')
    reference=float(decode(rbytes,meta['forecast_reference_time/.zarray']))
    if abs(reference-run.timestamp()/3600)>1e-6:raise ValueError('Forecast run differs')
    if meta['forecast_period/.zattrs']['units']!='hours' or not np.array_equal(periods,np.arange(1,len(periods)+1)):
        raise ValueError('Forecast lead sequence is not hourly')
    chunks={};result={};sources=[mpath,ppath,rpath]
    for site,(lat,lon,_) in sites.items():
        loc=location(lat,lon);chunk=f"0.{loc['chunk_y']}.{loc['chunk_x']}"
        if chunk not in chunks:
            raw,path=zget(prefix+spec+'/'+chunk);values=decode(raw,array).astype('float32')[:len(periods)]
            fill=array.get('fill_value')
            if isinstance(fill,(int,float)):values[values==fill]=np.nan
            chunks[chunk]=values;sources.append(path)
        result[site]=chunks[chunk][:,loc['internal_y'],loc['internal_x']]
    return periods,result,{'field':spec,'units':units,'source_paths':[str(p.relative_to(ROOT)) for p in sources],
                           'sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]}


def fetch_run(run,sites=SITES):
    run=pd.Timestamp(run);run=run.tz_localize('UTC') if run.tzinfo is None else run.tz_convert('UTC')
    folder=OUT/'runs'/run.strftime('%Y%m%dT%H00Z');folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'hourly.csv';manifest_path=folder/'manifest.json'
    if cache.exists() and manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if set(old.get('sites',{}))==set(sites) and not old.get('errors'):return pd.read_csv(cache)
    frames=[];metadata={};errors={}
    for name in VARIABLES:
        try:
            periods,points,source=field(run,name,sites);metadata[name]=source
            for site,values in points.items():
                frames.append(pd.DataFrame({'site':site,'forecast_hour':periods,
                                            'variable':name,'value':values}))
        except (requests.RequestException,FileNotFoundError,ValueError,KeyError) as exc:
            errors[name]=str(exc)
    if not frames:raise RuntimeError('No forecast variables available:'+str(errors))
    d=pd.concat(frames).pivot(index=['site','forecast_hour'],columns='variable',values='value').reset_index()
    d['run_time_utc']=run.isoformat();d['assumed_available_time_utc']=(run+pd.Timedelta(hours=3)).isoformat()
    d['valid_time_utc']=(run+pd.to_timedelta(d.forecast_hour,unit='h')).astype(str)
    for name in VARIABLES:
        if name not in d:d[name]=np.nan
    d.to_csv(cache,index=False)
    manifest_path.write_text(json.dumps({'run_time_utc':run.isoformat(),'source':'Operational issued HRRR forecasts via Utah Zarr',
        'availability_buffer_hours':3,'sites':{s:{'latitude':v[0],'longitude':v[1],'target_elevation_m':v[2],**location(*v[:2])} for s,v in sites.items()},
        'fields':metadata,'errors':errors,'note':'Forecasts only; no analysis/reanalysis substituted; missing is never zero.'},indent=2))
    return d


def asnow_grib_hour(run,lead,sites=SITES):
    import eccodes as ec
    stem=run.strftime('hrrr.%Y%m%d/conus/hrrr.t%Hz.wrfsfcf')+f'{lead:02}.grib2'
    idx_path=OUT/'raw/grib'/ (stem+'.idx')
    lines=download(NOAA+stem+'.idx',idx_path).decode().splitlines()
    selected=[i for i,line in enumerate(lines) if ':ASNOW:surface:' in line]
    if len(selected)!=1:raise ValueError('Expected one native snowfall forecast field')
    i=selected[0];begin=int(lines[i].split(':')[1]);end=int(lines[i+1].split(':')[1])-1
    path=OUT/'raw/grib'/(stem+'.asnow')
    body=download(NOAA+stem,path,{'Range':f'bytes={begin}-{end}'})
    handle=ec.codes_new_from_message(body)
    try:
        keys=['units','discipline','parameterCategory','parameterNumber','centre','typeOfLevel','startStep','endStep','stepUnits','stepType','Nx','Ny','iScansNegatively','jScansPositively','jPointsAreConsecutive','dataDate','dataTime','packingType','binaryScaleFactor','decimalScaleFactor','referenceValue']
        info={k:ec.codes_get(handle,k) for k in keys}
        # ecCodes does not recognize this NCEP parameter under table version2.
        # NOAA's parameter table explicitly defines (0,1,29) as Total Snowfall[m].
        # Only accept that exact provider/parameter/surface identity, never infer
        # snowfall units from the magnitude of decoded values.
        identity=(info['centre'],info['discipline'],info['parameterCategory'],info['parameterNumber'],info['typeOfLevel'])
        if identity!=('kwbc',0,1,29,'surface'):
            raise ValueError('Unexpected snowfall GRIB parameter identity:'+str(identity))
        if info['units']=='unknown':
            info['resolved_units']='m'
            info['units_source']='https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfprsf00.grib2.shtml'
        else:info['resolved_units']=info['units']
        if info['resolved_units']!='m' or info['stepUnits']!=1 or info['stepType']!='accum' or info['startStep']!=0 or info['endStep']!=lead:
            raise ValueError('ASNOW field does not represent run-cumulative snowfall metres:'+str(info))
        if (info['Nx'],info['Ny'],info['iScansNegatively'],info['jScansPositively'],info['jPointsAreConsecutive'])!=(1799,1059,0,1,0):
            raise ValueError('Unexpected native HRRR grid orientation')
        if int(info['dataDate'])!=int(run.strftime('%Y%m%d')) or int(info['dataTime'])!=run.hour*100:
            raise ValueError('Native forecast initialization mismatch')
        if info['packingType'] in {'grid_simple','grid_complex','grid_complex_spatial_differencing'}:
            # GRIB2 Y*10^D = R + X*2^E. Rounding gives half a grid step
            # of uncertainty; successive leads can use different precision.
            info['packing_quantum_m']=float(np.ldexp(1.,int(info['binaryScaleFactor']))*10.**(-int(info['decimalScaleFactor'])))
            info['packing_quantum_source']='https://codes.ecmwf.int/grib/format/grib2/regulations/'
        values=ec.codes_get_values(handle).reshape(1059,1799)
        missing=ec.codes_get(handle,'missingValue')
        values[values==missing]=np.nan
        result={}
        for site,(lat,lon,_) in sites.items():
            loc=location(lat,lon);result[site]=float(values[loc['grid_y'],loc['grid_x']])
            if lead==1:
                nearest=ec.codes_grib_find_nearest(handle,lat,lon)[0]
                if int(nearest['index'])!=loc['grid_y']*1799+loc['grid_x']:
                    raise ValueError('Native grid index disagrees with ecCodes nearest point')
                if abs(nearest['value']-result[site])>1e-7:raise ValueError('Native grid extraction disagrees with ecCodes nearest point')
        return result,{'inventory':lines[i],'native_metadata':info,'path':str(path.relative_to(ROOT)),
                       'sha256':hashlib.sha256(body).hexdigest()}
    finally:ec.codes_release(handle)


def reconcile_asnow_packing(cumulative,packing_steps):
    """Remove only nonmonotonicity permitted by each GRIB packing interval.

    Bounded isotonic least squares preserves monotonic values unchanged and
    keeps each adjusted endpoint within half its own quantization step. A
    true reset or accumulated drift with no feasible monotonic sequence is
    rejected. Missing values/packing metadata are never bridged or invented.
    """
    values=np.asarray(cumulative,float);steps=np.asarray(packing_steps,float)
    if values.ndim!=1 or steps.shape!=values.shape:raise ValueError('ASNOW packing arrays must be matching vectors')
    adjusted=values.copy();valid=np.isfinite(values)&np.isfinite(steps)&(steps>0)
    i=0
    while i<len(values):
        if not valid[i]:i+=1;continue
        start=i
        while i<len(values) and valid[i]:i+=1
        blocks=[]
        for j in range(start,i):
            if values[j]<0:raise ValueError('Negative cumulative ASNOW')
            blocks.append({'start':j,'end':j+1,'sum':values[j],'n':1,
                           'lower':max(0.,values[j]-steps[j]/2),'upper':values[j]+steps[j]/2,'value':values[j]})
            while len(blocks)>1 and blocks[-2]['value']>blocks[-1]['value']:
                right=blocks.pop();left=blocks.pop()
                lower=max(left['lower'],right['lower']);upper=min(left['upper'],right['upper'])
                if lower>upper+1e-12:
                    raise ValueError(f"ASNOW decreases beyond GRIB packing bounds at forecast hours {left['start']+1}-{right['end']}")
                if lower>upper:lower=upper=(lower+upper)/2
                total=left['sum']+right['sum'];n=left['n']+right['n']
                blocks.append({'start':left['start'],'end':right['end'],'sum':total,'n':n,
                               'lower':lower,'upper':upper,'value':float(np.clip(total/n,lower,upper))})
        for block in blocks:adjusted[block['start']:block['end']]=block['value']
    changes=[]
    for j in np.flatnonzero(valid & (np.abs(adjusted-values)>1e-12)):
        if abs(adjusted[j]-values[j])>steps[j]/2+1e-12:raise AssertionError('ASNOW correction exceeded packing bounds')
        changes.append({'forecast_hour':int(j+1),'raw_cumulative_m':float(values[j]),
                        'adjusted_cumulative_m':float(adjusted[j]),'packing_quantum_m':float(steps[j]),
                        'adjustment_m':float(adjusted[j]-values[j])})
    return adjusted,changes


def difference_asnow(cumulative):
    values=np.asarray(cumulative,float)
    out=np.diff(np.r_[0.,values])
    out[out < -1e-7]=np.nan
    return np.maximum(out,0.)


def fetch_asnow(run,sites=SITES):
    run=pd.Timestamp(run);parts={};sources={};errors={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs={pool.submit(asnow_grib_hour,run,lead,sites):lead for lead in range(1,49)}
        for job in as_completed(jobs):
            lead=jobs[job]
            try:parts[lead],sources[lead]=job.result()
            except Exception as exc:errors[lead]=str(exc)
    records=[];adjustments=[]
    packing_steps=np.array([sources.get(h,{}).get('native_metadata',{}).get('packing_quantum_m',np.nan) for h in range(1,49)])
    for site in sites:
        cumulative=np.array([parts.get(h,{}).get(site,np.nan) for h in range(1,49)])
        adjusted,changes=reconcile_asnow_packing(cumulative,packing_steps)
        adjustments.extend(dict(site=site,**change) for change in changes)
        hourly=difference_asnow(adjusted)
        for h in range(1,49):records.append({'site':site,'forecast_hour':h,'run_time_utc':run.isoformat(),
            'valid_time_utc':(run+pd.Timedelta(hours=h)).isoformat(),'snow_cumulative_m':cumulative[h-1],
            'snow_cumulative_adjusted_m':adjusted[h-1],'packing_quantum_m':packing_steps[h-1],
            'snowfall_inches':hourly[h-1]/.0254})
    folder=OUT/'runs'/run.strftime('%Y%m%dT%H00Z');folder.mkdir(parents=True,exist_ok=True)
    d=pd.DataFrame(records);d.to_csv(folder/'native_snowfall.csv',index=False)
    (folder/'native_snowfall_manifest.json').write_text(json.dumps({'source':'Official NOAA HRRR cumulative ASNOW GRIB2',
        'run_time_utc':run.isoformat(),'sources':sources,'errors':errors,
        'packing_adjustments':adjustments,
        'packing_policy':'Bounded isotonic cumulative reconciliation; each endpoint stays within half its native GRIB quantum. Larger decreases fail; gaps remain gaps. Raw values retained.',
        'packing_background':'https://confluence.ecmwf.int/plugins/viewsource/viewpagesrc.action?pageId=208501579',
        'conversion':'Difference adjacent packing-reconciled cumulative fields in metres, divide .0254 for inches; gaps remain gaps.'},indent=2))
    return d


def schedule():
    dates=[]
    for winter in range(2020,2024):
        for day in pd.date_range(f'{winter}-11-01',f'{winter+1}-04-30',freq='D',tz='UTC'):
            if day.day in [1,5,9,13,17,21,25,29]:dates.append(day+pd.Timedelta(hours=6))
    return dates


def historical(workers=4):
    OUT.mkdir(parents=True,exist_ok=True);runs=schedule()
    (OUT/'schedule.json').write_text(json.dumps({'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'selection':'Calendar days1,5,9,13,17,21,25,29 November–April, frozen without snowfall outcomes',
        'runs':[r.isoformat() for r in runs]},indent=2))
    done=[];failed=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(fetch_run,r):r for r in runs}
        for job in as_completed(jobs):
            run=jobs[job]
            try:done.append(job.result())
            except Exception as exc:failed.append({'run':run.isoformat(),'error':str(exc)})
            print(json.dumps({'completed':len(done),'failed':len(failed),'total':len(runs),'run':run.isoformat()}),flush=True)
    d=pd.concat(done,ignore_index=True).sort_values(['run_time_utc','site','forecast_hour'])
    d.to_csv(OUT/'historical_hourly.csv.gz',index=False)
    (OUT/'historical_failures.json').write_text(json.dumps(failed,indent=2))
    print('Historical forecast rows:'+str(len(d)),flush=True)


def historical_native(workers=6):
    """Fetch only cumulative endpoints needed for exact 12/24h verification.

    Fractional Vermont reporting hours require adjacent native endpoints;
    their within-hour allocation is explicitly a forecast interpolation.
    """
    leads=[5,6,7,17,29,30,31]
    rows=[];errors=[]
    def one(run,lead):
        vals,meta=asnow_grib_hour(run,lead)
        return [dict(site=s,run_time_utc=run.isoformat(),forecast_hour=lead,
                     valid_time_utc=(run+pd.Timedelta(hours=lead)).isoformat(),
                     snowfall_cumulative_inches=v/.0254) for s,v in vals.items()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(one,r,h):(r,h) for r in schedule() for h in leads}
        done=0
        for job in as_completed(jobs):
            run,lead=jobs[job]
            try:rows.extend(job.result())
            except Exception as exc:errors.append(dict(run=run.isoformat(),lead=lead,error=str(exc)))
            done+=1
            if done%50==0:print(json.dumps({'native_done':done,'total':len(jobs),'errors':len(errors)}),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'historical_native_endpoints.csv.gz',index=False)
    (OUT/'historical_native_errors.json').write_text(json.dumps(errors,indent=2))
    print(json.dumps({'native_complete':len(rows),'errors':len(errors)}),flush=True)


def _atomic_text(path, text):
    temp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temp.write_text(text);temp.replace(path)


def validate_current_pair(weather,snow,run):
    """A partial download must not replace a complete published forecast."""
    required=set(SITES)
    for name,frame,columns in [('weather',weather,VARIABLES),('snowfall',snow,['snowfall_inches'])]:
        if set(frame.site)!=required:raise ValueError(f'{name}: mountain/station set differs')
        if frame.duplicated(['site','forecast_hour']).any():raise ValueError(f'{name}: duplicate hours')
        if set(pd.to_datetime(frame.run_time_utc,utc=True))!={run}:raise ValueError(f'{name}: mixed model runs')
        for site,g in frame.groupby('site'):
            if list(sorted(g.forecast_hour))!=list(range(1,49)):raise ValueError(f'{name}: incomplete native48h for{site}')
        if not np.isfinite(frame[columns].to_numpy(float)).all():raise ValueError(f'{name}: missing forecast values; prior snapshot retained')
        expected=pd.to_datetime(frame.run_time_utc,utc=True)+pd.to_timedelta(frame.forecast_hour,unit='h')
        if not (pd.to_datetime(frame.valid_time_utc,utc=True)==expected).all():raise ValueError(f'{name}: invalid interval endpoints')
    if (snow.snowfall_inches<0).any():raise ValueError('Negative native snowfall')


def current():
    checked=pd.Timestamp.now(tz='UTC');latest=(checked-pd.Timedelta(hours=3)).floor('6h')
    OUT.mkdir(parents=True,exist_ok=True);attempt_errors=[];previous={}
    try:previous=json.loads((OUT/'current.json').read_text())
    except (OSError,ValueError):pass
    for run in [latest,latest-pd.Timedelta(hours=6)]:
        try:
            weather=fetch_run(run);snow=fetch_asnow(run)
            validate_current_pair(weather,snow,run)
            break
        except Exception as exc:attempt_errors.append(f'{run.isoformat()}: {exc}')
    else:raise RuntimeError('No complete HRRR run available; prior current snapshot retained. '+'; '.join(attempt_errors))
    stamp=checked.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    folder=OUT/'current_snapshots'/stamp;folder.mkdir(parents=True,exist_ok=True)
    weather_path=folder/'weather.csv';snow_path=folder/'snowfall.csv'
    weather_text=weather.to_csv(index=False);snow_text=snow.to_csv(index=False)
    weather_path.write_text(weather_text);snow_path.write_text(snow_text)
    # Original retrieval is retained when a refresh checks the same immutable
    # model run. checked_at records the new check without inventing a new issue.
    retrieved=previous.get('retrieved_utc') if previous.get('run_time_utc')==run.isoformat() else checked.isoformat()
    info={'run_time_utc':run.isoformat(),'retrieved_utc':retrieved,'checked_at':checked.isoformat(),
        'assumed_available_utc':(run+pd.Timedelta(hours=3)).isoformat(),'forecast_end_utc':(run+pd.Timedelta(hours=48)).isoformat(),
        'weather_path':str(weather_path.relative_to(ROOT)),'snowfall_path':str(snow_path.relative_to(ROOT)),
        'fallback_run':run!=latest,'attempt_errors':attempt_errors,'sites':list(SITES),
        'source_retrieval_note':'HRRR runs are immutable; individual raw manifests retain original download times. checked_at is a new availability check, not a new model issue.',
        'horizon_caveat':'48hours from initialization; never extend the native forecast beyond this endpoint.'}
    (folder/'manifest.json').write_text(json.dumps(info,indent=2))
    # Compatibility copies for old readers; new readers use the atomic pointer
    # and immutable paired files above. Every consumer must verify run identity.
    _atomic_text(OUT/'current_weather.csv',weather_text);_atomic_text(OUT/'current_native_snowfall.csv',snow_text)
    _atomic_text(OUT/'current.json',json.dumps(info,indent=2))
    print(json.dumps({'current_run':run.isoformat(),'weather_rows':len(weather),'snow_rows':len(snow),'fallback_run':run!=latest}),flush=True)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--historical',action='store_true');p.add_argument('--current',action='store_true');p.add_argument('--native-history',action='store_true');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.current:current()
    if a.historical:historical(a.workers)
    if a.native_history:historical_native(a.workers)
