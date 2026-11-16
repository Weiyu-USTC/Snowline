"""Issue saved neural daily forecasts without refitting or changing frozen data.

The reporting-day amounts may be temporally allocated by a separate application;
this module never presents those allocations as native hourly neural predictions.
"""
from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import torch

import deep_dillon as base
import multi_resort_neural as daily
import vail_storm_neural as events
from fetch_vail_caic import parse as parse_caic

ROOT = Path(__file__).resolve().parents[1]
SITE = {
    'alta': dict(station='USC00420072', old='alta_noaa', clock=11., target='Alta-Collins manual 24-hour new snowfall', clock_note='04:00 fixed Mountain standard time; 11:00 UTC'),
    'mansfield': dict(station='USC00435416', old='mount_mansfield', clock=21., target='NOAA Mount Mansfield station daily SNOW, not resort-wide Stowe snowfall', clock_note='21:00 UTC assumes 16:00 EST; historical observation times are uncertain'),
    'jay': dict(station='USC00434120', old='island_pond', clock=12.5, target='NOAA Jay Peak station daily SNOW at 560.8 m, not summit snowfall', clock_note='12:30 UTC assumes 07:30 EST, documented only in part of station history; weather comes from Island Pond'),
    'vail': dict(clock=11., target='CAVMM 04:00 Sno24 sensor proxy, not verified 24-hour fresh snowfall', clock_note='04:00 source display mapped to 11:00 UTC assuming MST; historical clock/reset ambiguity remains'),
}
STATIONS = ['CAVMM', 'CAVPQ', 'CAVBS', 'CAVCB']
SEED = 20261110


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(t):
    return pd.Timestamp(t).isoformat()


def fetch(url, path):
    """Bounded public download; a saved run is replayable from its raw archive."""
    if path.exists():
        meta = json.loads(path.with_suffix(path.suffix+'.json').read_text())
        return path.read_bytes(), {**meta, 'reused_saved_download': True}
    req = urllib.request.Request(url, headers={'User-Agent':'Mountain snow research / saved neural inference'})
    with urllib.request.urlopen(req, timeout=45) as response:
        body = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    meta = {'url':url, 'retrieved_at_utc':stamp(pd.Timestamp.now(tz='UTC')), 'path':str(path.relative_to(ROOT)), 'sha256':digest(path), 'bytes':len(body)}
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(meta, indent=2))
    return body, meta


def noaa_frame(body, station, cutoff):
    fields = {'TMAX':('tmax_c',10.), 'TMIN':('tmin_c',10.), 'PRCP':('precipitation_inches',254.)}
    rows = {}
    for line in body.decode().splitlines():
        if len(line)<269 or line[:11]!=station or line[17:21] not in fields:
            continue
        year, month = int(line[11:15]), int(line[15:17])
        col, divisor = fields[line[17:21]]
        for day in range(1, calendar.monthrange(year,month)[1]+1):
            date = pd.Timestamp(year,month,day)
            if date>cutoff:
                continue
            block=line[21+8*(day-1):29+8*(day-1)]
            value=int(block[:5]); quality=block[6].strip()
            row=rows.setdefault(date,{})
            if col in row:
                raise ValueError('Duplicate NOAA date/element')
            row[col] = value/divisor if value!=-9999 and not quality else np.nan
    frame=pd.DataFrame.from_dict(rows,orient='index').reindex(columns=['tmax_c','tmin_c','precipitation_inches']).sort_index()
    if frame.empty:
        raise ValueError('NOAA source has no parseable weather observations')
    frame['precipitation_inches']=np.log1p(frame.precipitation_inches.clip(lower=0))
    return frame


def old_noaa(site):
    path=ROOT/'data/multi_resort_daily/current_inputs'/f"{SITE[site]['old']}_daily.csv"
    d=pd.read_csv(path,keep_default_na=False,low_memory=False).set_index('date')
    d.index=pd.to_datetime(d.index)
    frame=pd.DataFrame(index=d.index)
    for col in ['tmax_c','tmin_c','precipitation_inches']:
        frame[col]=pd.to_numeric(d[col],errors='coerce').where(d[col+'_quality_flag'].eq(''))
    frame['precipitation_inches']=np.log1p(frame.precipitation_inches.clip(lower=0))
    return frame,path


def get_noaa(site, rawdir, cutoff):
    cfg=SITE[site]; path=rawdir/f"{cfg['station']}.dly"
    url=f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{cfg['station']}.dly"
    try:
        body,meta=fetch(url,path)
        frame=noaa_frame(body,cfg['station'],cutoff)
        return frame, {'status':'downloaded_or_replayed_public_observations','source':meta}
    except Exception as exc:
        frame,oldpath=old_noaa(site)
        return frame.loc[:cutoff], {'status':'fallback_saved_observations','error':str(exc),'source':{'path':str(oldpath.relative_to(ROOT)),'sha256':digest(oldpath),'original_acquired_date':'2026-11-04'}}


def caic_features(raw):
    """Mirror the frozen Vail preprocessing; preserve missingness and column order."""
    raw=raw.copy();raw['timestamp']=pd.to_datetime(raw.datetime_display)
    for c in ['Temp','DewP','RH','Spd','Gst','Dir','Sno24','Pcp1','MSLP']:
        if c not in raw:raw[c]=np.nan
        raw[c]=pd.to_numeric(raw[c],errors='coerce')
    if raw.duplicated(['station_id','timestamp']).any():
        raise ValueError('Conflicting CAIC clock duplicates were not resolved')
    features={}
    for st in STATIONS:
        part=raw[raw.station_id.eq(st)].set_index('timestamp').sort_index()
        for col,low,high in [('Temp',-70,110),('DewP',-90,100),('RH',0,100),('Spd',0,150),('Gst',0,200)]:
            features[f'{st}_{col}']=part[col].where(part[col].between(low,high))
        direction=part.Dir.where(part.Dir.between(0,360))*np.pi/180
        features[f'{st}_wind_u']=-features[f'{st}_Spd']*np.sin(direction)
        features[f'{st}_wind_v']=-features[f'{st}_Spd']*np.cos(direction)
    mid=raw[raw.station_id.eq('CAVMM')].set_index('timestamp').sort_index()
    snow=mid.Sno24.where(mid.Sno24>=0)
    snow.loc['2020-03-20':'2020-05-31']=np.nan
    snow.loc['2023-10-01':'2023-11-28']=np.nan
    features['CAVMM_Sno24']=np.log1p(snow)
    phq=raw[raw.station_id.eq('CAVPQ')].set_index('timestamp').sort_index()
    for col,lo,hi in [('Pcp1',0,10),('MSLP',850,1100)]:
        features[f'CAVPQ_{col}']=phq[col].where(phq[col].between(lo,hi))
    features['temperature_difference_PQ_MM']=features['CAVPQ_Temp']-features['CAVMM_Temp']
    features['temperature_difference_BS_MM']=features['CAVBS_Temp']-features['CAVMM_Temp']
    return pd.DataFrame(features).sort_index(),snow[snow.index.hour==4].sort_index()


def read_old_caic():
    p=ROOT/'data/vail_hourly/hourly.csv'
    columns=['station_id','datetime_display','Temp','DewP','RH','Spd','Gst','Dir','Sno24','Pcp1','MSLP']
    return pd.read_csv(p,usecols=columns,low_memory=False),p


def get_vail(rawdir, desired):
    # Keep all prior source data for initial event age; current observations take
    # priority only for exact timestamps actually returned by the public endpoint.
    old,oldpath=read_old_caic();metadata=[];fresh=[]
    def one(st):
        params={'st':st,'date':desired.strftime('%Y-%m-%d %H'),'unit':'e','area':'caic','range':24*16}
        url='https://stations.avalanche.state.co.us/tabular.php?'+urllib.parse.urlencode(params)
        try:
            body,meta=fetch(url,rawdir/f'{st}.html')
            rows=[]
            for ym in pd.period_range(desired-pd.Timedelta(days=17),desired,freq='M'):
                rr,_=parse_caic(body,st,ym.year,ym.month,url,f'{st}.html');rows.extend(rr)
            if not rows:raise ValueError('No observations returned for requested source-clock interval')
            return rows,{'station':st,'status':'downloaded_or_replayed_public_observations','source':meta}
        except Exception as exc:return [],{'station':st,'status':'fallback_saved_observations','error':str(exc)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for rows,meta in pool.map(one,STATIONS):fresh.extend(rows);metadata.append(meta)
    current=pd.DataFrame(fresh)
    if len(current):
        current['datetime_display']=pd.to_datetime(current.datetime_display)
        current=current[current.datetime_display<=desired]
        # Numeric ambiguity is missing data, not an arbitrary duplicate winner.
        cols=['Temp','DewP','RH','Spd','Gst','Dir','Sno24','Pcp1','MSLP']
        for c in cols:
            if c not in current:current[c]=np.nan
        for key,part in current.groupby(['station_id','datetime_display']):
            if len(part)>1 and any(part[c].nunique(dropna=False)>1 for c in cols):
                current.loc[part.index,cols]=np.nan
        current=current.drop_duplicates(['station_id','datetime_display'])
        old['datetime_display']=pd.to_datetime(old.datetime_display)
        combined=pd.concat([old,current],ignore_index=True).drop_duplicates(['station_id','datetime_display'],keep='last')
    else:combined=old
    combined['datetime_display']=pd.to_datetime(combined.datetime_display)
    combined=combined[combined.datetime_display<=desired]
    frame,target=caic_features(combined)
    return frame,target,{'current_sources':metadata,'fallback_history':{'path':str(oldpath.relative_to(ROOT)),'sha256':digest(oldpath),'original_acquired_date':'2026-11-04'}}


def future(origin,horizon):
    if not 1<=horizon<=60:raise ValueError('Requested propagation exceeds trained 60-day horizon')
    dates=pd.date_range(origin.normalize()+pd.Timedelta(days=1),periods=horizon)
    roni=base.lagged_roni(origin);lead=np.arange(1,horizon+1)/60.
    known=[roni if np.isfinite(roni) else 0.,float(roni>=1.5),float(roni<=-1.5),float(not np.isfinite(roni))]
    return np.column_stack([base.calendar_features(dates),lead,lead**2,np.tile(known,(horizon,1))]).astype('float32')[None]


def initial_vail(target,origin):
    previous=np.nan;age=0;known=False
    year=origin.year if origin.month>=9 else origin.year-1
    for value in target.reindex(pd.date_range(f'{year}-09-01 04:00',origin)).to_numpy(float):
        if not np.isfinite(value):previous=np.nan;age=0;known=False;continue
        if not np.isfinite(previous):age=0;known=False
        elif (value>=1)!=(previous>=1):age=1;known=True
        elif known:age=min(age+1,14)
        previous=value
    return np.array([[previous,age,float(known),float(np.isfinite(previous))]],dtype='float32')


def windows(site,start,end):
    clock=SITE[site]['clock'];anchor=start.normalize()+pd.Timedelta(hours=clock)
    if anchor>start:anchor-=pd.Timedelta(days=1)
    ends=pd.date_range(anchor+pd.Timedelta(days=1),end+pd.Timedelta(days=1),freq='D')
    ends=ends[(ends>start)&(ends-pd.Timedelta(days=1)<end)]
    return anchor,ends


def infer(model,scaler,x,origin,initial,ends,n_paths):
    leads=np.array([(end.tz_localize(None).normalize()-origin.normalize()).days for end in ends])
    if leads.min()<1:raise ValueError('Forecast includes an already-ended neural target day')
    f=future(origin,int(leads.max()))
    pred=events.forecast_neural(model,scaler,x,f,initial)
    paths=events.simulate_neural(model,scaler,x,f,initial,n_paths=n_paths,seed=SEED)[0]
    take=leads-1
    return pred['mean'][0,take],paths[:,take],leads,f


def replay_check():
    """Restored weights and preprocessing must reproduce the frozen Nov4 means."""
    checks={}
    for site in ['alta','mansfield','jay']:
        frame,_=old_noaa(site);origin=pd.Timestamp('2026-11-04')
        x=frame.reindex(pd.date_range(end=origin-pd.Timedelta(days=2),periods=60)).to_numpy('float32')[None]
        payload=torch.load(ROOT/f'outputs/multi-resort/{site}/final_neural.pt',map_location='cpu',weights_only=True)
        model,scaler=daily.restore_neural(payload)
        p=events.forecast_neural(model,scaler,x,future(origin,3),np.array([[np.nan,0,0,0]],dtype='float32'))['mean'][0]
        old=pd.read_csv(ROOT/f'outputs/multi-resort/{site}/forecast_daily.csv')
        expected=old[old.model.eq('Neural events')].head(3).mean_inches.to_numpy()
        checks[site]=float(np.max(abs(p-expected)));assert np.allclose(p,expected,rtol=1e-6,atol=1e-6),(site,p,expected)
    raw,_=read_old_caic();frame,target=caic_features(raw);origin=pd.Timestamp('2026-11-04 04:00')
    x=frame.reindex(pd.date_range(end=origin,periods=168,freq='h')).to_numpy('float32')[None]
    model,scaler=events.restore_neural(torch.load(ROOT/'outputs/vail-storm-models/neural_event_model.pt',map_location='cpu',weights_only=True))
    p=events.forecast_neural(model,scaler,x,future(origin,3),initial_vail(target,origin))['mean'][0]
    old=pd.read_csv(ROOT/'outputs/vail-storm-models/forecast_daily.csv');expected=old[old.model.eq('Neural events')].head(3).mean_inches.to_numpy()
    checks['vail']=float(np.max(abs(p-expected)));assert np.allclose(p,expected,rtol=1e-6,atol=1e-6),('vail',p,expected)
    return checks


def main(plan_path,out,n_paths):
    plan_path=plan_path.resolve();out=out.resolve()
    plan=json.loads(plan_path.read_text());start=pd.Timestamp(plan['forecast_start_utc']);end=pd.Timestamp(plan['forecast_end_utc'])
    requested=pd.Timestamp(plan['requested_at_utc']);out.mkdir(parents=True,exist_ok=True)
    rawdir=ROOT/'data/neural-live'/out.name;rawdir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    audit={'restoration_replay_max_absolute_error':replay_check()}
    manifest={'requested_at_utc':stamp(requested),'generated_at_utc':stamp(pd.Timestamp.now(tz='UTC')),'forecast_start_utc':stamp(start),'forecast_end_utc':stamp(end),'new_training':False,'path_count':n_paths,'path_seed':SEED,'sites':{},'enso_source':{'path':'data/roni_noaa.csv','sha256':digest(ROOT/'data/roni_noaa.csv'),'note':'Frozen revised lagged RONI convention from original training; not release-vintage index'}}
    rows=[];arrays={}
    for site in plan['sites']:
        anchor,ends=windows(site,start,end)
        if site!='vail':
            desired=anchor.tz_localize(None).normalize()
            cutoff=min(desired-pd.Timedelta(days=2),requested.tz_localize(None).normalize()-pd.Timedelta(days=2))
            frame,provenance=get_noaa(site,rawdir,cutoff)
            frame=frame.loc[:cutoff]
            temperatures=frame[['tmax_c','tmin_c']].notna().all(axis=1)
            if not temperatures.any():raise ValueError(f'{site}: no usable temperature history')
            latest=frame.index[temperatures].max();origin=min(desired,latest+pd.Timedelta(days=2))
            dates=pd.date_range(end=origin-pd.Timedelta(days=2),periods=60)
            x=frame.reindex(dates).to_numpy('float32')[None]
            if np.isfinite(x[0,:,:2]).sum()<60:raise ValueError(f'{site}: context below training eligibility')
            weights=ROOT/f'outputs/multi-resort/{site}/final_neural.pt'
            payload=torch.load(weights,map_location='cpu',weights_only=True);model,scaler=daily.restore_neural(payload)
            initial=np.array([[np.nan,0,0,0]],dtype='float32')
            context_end_utc=pd.Timestamp(dates[-1],tz='UTC')+pd.Timedelta(days=1)
            policy='60 daily weather rows ending two calendar days before neural origin; initial snowfall state unknown as in training'
        else:
            desired=anchor.tz_localize(None).normalize()+pd.Timedelta(hours=4)
            frame,target,provenance=get_vail(rawdir,desired)
            # Use a past 04:00 origin satisfying the original context eligibility.
            origin=None
            for candidate in pd.date_range(desired-pd.Timedelta(days=50),desired,freq='D')[::-1]:
                dates=pd.date_range(end=candidate,periods=168,freq='h')
                xx=frame.reindex(dates).to_numpy('float32')[None]
                exact=frame.reindex([candidate])['CAVMM_Temp'].notna().iloc[0]
                if exact and np.isfinite(xx[0,:,0]).sum()>=84:
                    origin=candidate;x=xx;break
            if origin is None:raise ValueError('Vail lacks a training-eligible recent context')
            weights=ROOT/'outputs/vail-storm-models/neural_event_model.pt'
            payload=torch.load(weights,map_location='cpu',weights_only=True);model,scaler=events.restore_neural(payload)
            initial=initial_vail(target,origin)
            context_end_utc=pd.Timestamp(origin,tz='UTC')+pd.Timedelta(hours=7)
            policy='168 hourly rows through 04:00 source display; initial sensor event state/age from observed history only; UTC mapping assumes MST'
        if context_end_utc>requested:raise ValueError(f'{site}: context extends beyond request issuance')
        before=digest(weights)
        mean,paths,leads,f=infer(model,scaler,x,origin,initial,ends,n_paths)
        assert digest(weights)==before
        assert np.isfinite(paths).all() and (paths>=0).all() and np.isfinite(mean).all() and (mean>=0).all()
        q10,q90=np.quantile(paths,[.1,.9],axis=0)
        arrays[site+'_paths_inches']=paths
        arrays[site+'_window_end_utc']=np.asarray([stamp(t) for t in ends],dtype='U40')
        arrays[site+'_window_start_utc']=np.asarray([stamp(t-pd.Timedelta(days=1)) for t in ends],dtype='U40')
        arrays[site+'_mean_inches']=mean
        arrays[site+'_lead_day']=leads
        contextfile=out/f'{site}_inference_inputs.npz'
        np.savez_compressed(contextfile,context=x,future=f,initial_state=initial,context_times=np.asarray([str(t) for t in dates],dtype='U40'),feature_names=np.asarray(frame.columns,dtype='U80'))
        lastobserved={col:(str(frame.loc[:dates[-1],col].dropna().index.max()) if frame.loc[:dates[-1],col].notna().any() else None) for col in frame}
        entry={'model':'Neural events','target':SITE[site]['target'],'reporting_clock_note':SITE[site]['clock_note'],'reporting_clock_utc_hour':SITE[site]['clock'],'model_origin_date':str(origin.date()),'model_origin_source_clock':str(origin),'latest_requested_daily_anchor_utc':stamp(anchor),'bridge_days':int((anchor.tz_localize(None).normalize()-origin.normalize()).days),'context_policy':policy,'context_end_utc_conservative':stamp(context_end_utc),'context_first':str(dates[0]),'context_last':str(dates[-1]),'context_missing_fraction':float(np.mean(~np.isfinite(x))),'last_observed_by_feature':lastobserved,'initial_state':[float(z) if np.isfinite(z) else None for z in initial[0]],'checkpoint':str(weights.relative_to(ROOT)),'checkpoint_sha256':before,'checkpoint_unchanged':True,'input_file':str(contextfile.relative_to(ROOT)),'input_sha256':digest(contextfile),'feature_names':list(frame.columns),'provenance':provenance,'original_config':payload['config'],'roni_input':float(f[0,0,6]),'roni_after_training_guard':float(scaler.transform_future(f)[0,0,6]),'uncertainty':'Joint simulated daily paths from frozen neural event/amount model; uncalibrated model-based predictive distribution','freshness_note':'Download time is separate from observation date; missing recent observations cause a past-origin free-running bridge, not invented current weather'}
        manifest['sites'][site]=entry
        for i,t in enumerate(ends):
            lo=t-pd.Timedelta(days=1);overlap=max(pd.Timedelta(0),min(end,t)-max(start,lo))
            rows.append({'site':site,'window_start_utc':stamp(lo),'window_end_utc':stamp(t),'target_date':str(t.date()),'mean_inches':float(mean[i]),'q10_inches':float(q10[i]),'q90_inches':float(q90[i]),'model_origin_date':str(origin.date()),'lead_day':int(leads[i]),'overlap_hours':overlap.total_seconds()/3600,'clock_note':SITE[site]['clock_note'],'model':'Neural events'})
        assert np.isclose(sum(r['overlap_hours'] for r in rows if r['site']==site),48.)
        print(json.dumps({'site':site,'origin':str(origin),'leads':leads.tolist(),'daily_means_inches':mean.tolist(),'missing_fraction':entry['context_missing_fraction'],'bridge_days':entry['bridge_days']}),flush=True)
    pd.DataFrame(rows).to_csv(out/'daily_forecasts.csv',index=False)
    np.savez_compressed(out/'joint_paths.npz',**arrays)
    audit.update(status='passed',all_four_sites=True,each_site_covers_48h=True,no_future_observation_context=True,all_checkpoints_unchanged=True,paths_finite_nonnegative=True,mean_is_exact_state_propagation=True,source_file_sha256=digest(__file__))
    (out/'daily_inference_audit.json').write_text(json.dumps(audit,indent=2))
    (out/'daily_inference_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--paths',type=int,default=4096)
    args=parser.parse_args();main(args.plan,args.output,args.paths)
