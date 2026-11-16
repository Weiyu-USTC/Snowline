"""Join only issued predictors with exact two-day station accumulation targets."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/neural48_training'
SITES=['alta','vail','mansfield','jay']

def ns(index):return pd.DatetimeIndex(index).as_unit('ns').asi8

def bounded_instant(times,values,at):
    j=int(np.searchsorted(times,at))
    if j==0 or j>=len(times) or times[j]-times[j-1]>3.6e12:return np.nan
    if not np.isfinite(values[j-1:j+1]).all():return np.nan
    return float(np.interp(at,times[j-1:j+1],values[j-1:j+1]))

def provider_window(frame,start):
    """48 one-hour forecast intervals aligned to the actual observation clock."""
    start=pd.Timestamp(start);ends=pd.date_range(start+pd.Timedelta(hours=1),periods=48,freq='h')
    d=frame[(frame.valid_time_utc>=start.floor('h'))&(frame.valid_time_utc<=ends[-1].ceil('h'))].sort_values('valid_time_utc')
    source_end=pd.DatetimeIndex(d.valid_time_utc);source_start=source_end-pd.Timedelta(hours=1)
    source_values=d[['temperature_2m','precipitation','snowfall','wind_speed_10m','wind_direction_10m','relative_humidity_2m','pressure_msl']].to_numpy(float)
    result=[]
    for end in ends:
        begin=end-pd.Timedelta(hours=1)
        overlap=np.maximum(0,np.minimum(end.value,ns(source_end))-np.maximum(begin.value,ns(source_start)))/3.6e12
        use=overlap>0
        if not np.isclose(overlap.sum(),1):result.append([np.nan]*7);continue
        row=[]
        # Amounts use exact interval overlap. Instantaneous forecast fields use
        # endpoint interpolation at the centre of the target hour; both inherit
        # the provider's coarse native resolution.
        for column in [0,5,6]:
            midpoint=(begin+(end-begin)/2).value
            vv=source_values[:,column]
            row.append(bounded_instant(ns(source_end),vv,midpoint))
        for column in [1,2]:
            vv=source_values[use,column]
            row.append(float(vv@overlap[use]) if np.isfinite(vv).all() else np.nan)
        theta=np.deg2rad(source_values[:,4]);speed=source_values[:,3]
        for vv in [-speed*np.sin(theta),-speed*np.cos(theta)]:
            midpoint=(begin+(end-begin)/2).value
            row.append(bounded_instant(ns(source_end),vv,midpoint))
        result.append(row)
    x=np.asarray(result,dtype='float32')
    x[:,4]/=2.54
    return x,ends

def build():
    targets=pd.read_csv(DATA/'targets/two_day_targets.csv')
    daily=pd.read_csv(DATA/'targets/daily_targets.csv')
    daily['period_end_utc']=pd.to_datetime(daily.period_end_utc,utc=True)
    provider=pd.read_csv(DATA/'provider/historical_hourly.csv.gz')
    provider['valid_time_utc']=pd.to_datetime(provider.valid_time_utc,utc=True)
    provider['implied_source_bound_utc']=pd.to_datetime(provider.implied_source_bound_utc,utc=True)
    out=DATA/'prepared';out.mkdir(exist_ok=True)
    arrays={k:[] for k in ['nwp','radar','local','static','history','y','provider','site']}
    rows=[];reject=[];source_files=[]
    target_names={}
    for site_number,site in enumerate(SITES):
        print(f'Preparing {site}…',flush=True)
        weather_path=DATA/f'nwp/radar_cohort_{site}.npz'
        with np.load(weather_path) as z: weather={k:z[k] for k in z.files}
        ids=weather['case_ids'];lookup={v:i for i,v in enumerate(ids)}
        source_files.append(weather_path)
        dst_path=DATA/f'nwp/radar_cohort_{site}_dst.npz'
        dst=None
        if dst_path.exists():
            with np.load(dst_path) as z: dst={k:z[k] for k in z.files}
        if dst is not None:assert np.array_equal(dst['case_ids'],ids)
        if dst_path.exists():source_files.append(dst_path)
        region=site if site in ['alta','vail'] else 'vermont'
        radar_path=DATA/f'radar/{region}_contexts.npz'
        with np.load(radar_path) as z: radar={k:z[k] for k in z.files}
        assert np.array_equal(radar['case_id'],ids)
        source_files.append(radar_path)
        local_idx=list(radar['site_ids']).index(site)
        local_override=DATA/'radar/vermont_target_local_stats.npz'
        local_extra=None
        if region=='vermont' and local_override.exists():
            with np.load(local_override) as z: local_extra={k:z[k] for k in z.files}
        if local_extra is not None:
            assert np.array_equal(local_extra['case_id'],ids)
            source_files.append(local_override)
        snow=daily[daily.site.eq(site)&daily.valid_snowfall.astype(bool)].sort_values('period_end_utc')
        p=provider[provider.site.eq(site)]
        g=targets[targets.site.eq(site)].copy()
        g['case_id']=pd.to_datetime(g.origin_date).dt.strftime('%Y%m%d')
        g=g[g.case_id.isin(lookup)]
        for r in g.itertuples():
            if site=='vail':
                # Vail is explicitly a sensor proxy: nominal intervals may
                # be used without upgrading its unverified clock/fresh-snow flag.
                eligible=bool(r.intervals_continuous and r.day1_duration_hours==24
                              and r.day2_duration_hours==24 and r.total_duration_hours==48)
            else:eligible=bool(r.strict_48h_eligible)
            if not eligible:
                reject.append({'site':site,'case_id':r.case_id,'reason':'reporting clocks or duration do not define exact nominal24+24h'});continue
            i=lookup[r.case_id];start=pd.Timestamp(r.target_start_utc);end=pd.Timestamp(r.target_end_utc)
            issue=pd.Timestamp(r.origin_date,tz='UTC')+pd.Timedelta(hours=10,minutes=50)
            if start<issue or end-start!=pd.Timedelta(hours=48):
                reject.append({'site':site,'case_id':r.case_id,'reason':'not fully future exact48h'});continue
            standard={'alta':11,'vail':11,'mansfield':12.5,'jay':12}[site]
            start_hour=start.hour+start.minute/60
            if np.isclose(start_hour,standard):wx=weather['forecast'][i]
            elif dst is not None and np.isclose(start_hour,standard-1):wx=dst['forecast'][i]
            else:
                reject.append({'site':site,'case_id':r.case_id,'reason':'NWP alignment unavailable'});continue
            if not np.isfinite(wx).all():
                reject.append({'site':site,'case_id':r.case_id,'reason':'MOS meteorological coverage incomplete'});continue
            jma,ends=provider_window(p,start)
            # End-labelled precipitation includes a possible fractional last
            # source hour. Its nominal72h source bound must precede issue.
            latest=ends[-1].ceil('h')-pd.Timedelta(days=3)
            assert latest<=issue-pd.Timedelta(hours=2)
            raw_days=jma[:,4].reshape(2,24).sum(1)
            if not np.isfinite(jma).all():
                reject.append({'site':site,'case_id':r.case_id,'reason':'matched provider48h coverage incomplete'});continue
            known=snow[snow.period_end_utc+pd.Timedelta(hours=24)<=issue]
            past=known[known.period_end_utc>issue-pd.Timedelta(days=61)]
            hs=[]
            for days in [3,7,14,30,60]:
                recent=past[past.period_end_utc>issue-pd.Timedelta(days=days+1)].snowfall_inches.to_numpy(float)
                hs.extend([float(recent.mean()) if len(recent) else np.nan,
                           float(np.mean(recent>0)) if len(recent) else np.nan,len(recent)/days])
            hs.extend([float(known.snowfall_inches.iloc[-1]) if len(known) else np.nan,
                       min(60.,(issue-known.period_end_utc.iloc[-1]).total_seconds()/86400) if len(known) else 60.])
            phase=2*np.pi*(start.dayofyear-1)/365.2425
            static=[np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase),
                    r.target_elevation_m/3000.,r.target_latitude/90.,r.target_longitude/180.,
                    (start-issue).total_seconds()/86400]
            local=radar['local_stats'][i,local_idx]
            if local_extra is not None:
                # Metadata/schema validated in the separately retained radar audit.
                li=list(local_extra['site_ids']).index(site)
                local=local_extra['local_stats'][i,li]
            arrays['nwp'].append(np.concatenate([wx,jma],axis=-1))
            arrays['radar'].append(radar['x'][i]);arrays['local'].append(local)
            arrays['static'].append(static);arrays['history'].append(hs)
            arrays['y'].append([r.snowfall_day1_inches,r.snowfall_day2_inches]);arrays['provider'].append(raw_days)
            arrays['site'].append(site_number)
            winter=int(r.season_end_year)-1
            rows.append({'site':site,'site_index':site_number,'case_id':r.case_id,'origin_date':r.origin_date,
                'winter':winter,'split':'train' if winter<=2021 else 'validation' if winter==2022 else 'test',
                'issue_utc':issue.isoformat(),'start_utc':start.isoformat(),'midpoint_utc':r.target_midpoint_utc,
                'end_utc':end.isoformat(),'target_name':r.target_name,'target_station':r.station_id,
                'target_class':r.target_class,'target_elevation_m':r.target_elevation_m,
                'clock_verified':bool(r.clock_verified),'fresh_snow_verified':bool(r.verified_fresh_snow),
                'radar_complete':bool(radar['input_complete'][i]),
                'radar_qualified_fraction':float(radar['input_qualified_fraction'][i]),
                'source_bound_latest_utc':latest.isoformat(),
                'past_snow_latest_available_utc':known.period_end_utc.iloc[-1].isoformat() if len(known) else None})
            target_names[site]=r.target_name
    prepared={k:np.asarray(v,dtype='int64' if k=='site' else 'float32') for k,v in arrays.items()}
    meta=pd.DataFrame(rows);meta.insert(0,'row',np.arange(len(meta)))
    assert len(meta)>0 and not meta.duplicated(['site','case_id']).any()
    assert set(meta.winter)=={2020,2021,2022,2023}
    np.savez_compressed(out/'arrays.npz',**prepared)
    meta.to_csv(out/'index.csv',index=False);pd.DataFrame(reject).to_csv(out/'exclusions.csv',index=False)
    counts=meta.groupby(['site','split']).size().unstack(fill_value=0)
    counts.to_csv(out/'coverage.csv')
    source_files += [DATA/'targets/two_day_targets.csv',DATA/'targets/daily_targets.csv',DATA/'provider/historical_hourly.csv.gz']
    audit={'rows':len(meta),'counts':counts.to_dict('index'),'shapes':{k:list(v.shape) for k,v in prepared.items()},
        'input_only_radar':True,'forecast_archive':'00UTC GFS MOS plus conservative JMA72h-offset mosaic',
        'targets':target_names,'horizons':'Two exact successive24h reporting intervals; sum is48h. No independently observed hourly fresh-snow label.',
        'vail_caveat':'04sourceclock mappedMST and Sno24 fresh-snow equivalence remain unverified; proxy cohort reported separately.',
        'provider_comparison':'Same targetlocation/elevation and48hinterval. Fixed72h-offset JMA is not the live Open-Meteo best-match service.',
        'history_lag':'Only daily snow reports ending at least24h before issue; no current-day target state.',
        'source_files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(source_files))],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(counts.to_string());print(json.dumps(audit['shapes']))

if __name__=='__main__':build()
