"""True 12-hour Alta snowfall targets with issued forecasts, no hourly pseudo-labels."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/alta12_training'
HRRR_FIELDS=['precip_mm','snow_water_mm','temperature_k','dewpoint_k','wind_u_ms','wind_v_ms','relative_humidity_pct']

def histories(targets,issue):
    known=targets[targets.end_utc+pd.Timedelta(hours=2)<=issue]
    vals=[]
    for hours in [24,72,168,720]:
        recent=known[known.end_utc>issue-pd.Timedelta(hours=hours)]
        vals += [float(recent.snowfall_inches.mean()) if len(recent) else np.nan,
                 float((recent.snowfall_inches>0).mean()) if len(recent) else np.nan,
                 float(recent.lpe_inches.mean()) if len(recent) else np.nan,len(recent)/(hours/12)]
    age=(issue-known.end_utc.iloc[-1]).total_seconds()/3600 if len(known) else 1440.
    vals += [float(known.snowfall_inches.iloc[-1]) if age<1440 else np.nan,min(1440.,age)]
    return np.array(vals,np.float32)

def static_features(start,issue):
    phase=2*np.pi*(start.dayofyear-1)/365.2425
    return np.array([np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase),
        np.sin(2*np.pi*start.hour/24),np.cos(2*np.pi*start.hour/24),
        (start-issue).total_seconds()/3600],np.float32)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/east_alta_research/alta_hive/alta_collins_12hour_combined.csv'
    targets=pd.read_csv(source)
    for original,converted in [('period_start_lst','start_utc'),('period_end_lst','end_utc')]:
        targets[converted]=(pd.to_datetime(targets[original])+pd.Timedelta(hours=7)).dt.tz_localize('UTC')
    assert ((targets.end_utc-targets.start_utc)==pd.Timedelta(hours=12)).all()
    assert not targets.start_utc.duplicated().any()
    target_lookup=targets.set_index('start_utc')
    nwpbase=ROOT/'data/neural48_training/nwp'
    mos=np.load(nwpbase/'broad_alta.npz')['forecast']
    mosmeta=pd.read_csv(nwpbase/'broad_alta_index.csv')
    hp=ROOT/'data/alta_nowcast/hrrr/hourly_point.csv'
    h=pd.read_csv(hp)
    for k in ['run_time_utc','assumed_available_time_utc','valid_time_utc']:h[k]=pd.to_datetime(h[k],utc=True)
    h=h[h.run_time_utc.dt.hour.eq(6)].drop_duplicates(['run_time_utc','valid_time_utc'])
    hgroups={run:g.set_index('valid_time_utc') for run,g in h.groupby('run_time_utc')}
    radpath=ROOT/'data/neural48_training/radar/alta_contexts.npz'
    with np.load(radpath) as z:rad={k:z[k] for k in z.files}
    ri={c:i for i,c in enumerate(rad['case_id'])}
    rows=[];arr={k:[] for k in ['mos','hrrr','radar','local','static','history','snow','lpe']}
    for r in mosmeta.itertuples():
        if r.winter>2023:continue
        run=pd.Timestamp(r.run_time_utc);hrrr_run=run+pd.Timedelta(hours=6)
        for half in range(2):
            start=run+pd.Timedelta(hours=11+half*12);end=start+pd.Timedelta(hours=12)
            if start not in target_lookup.index:continue
            t=target_lookup.loc[start]
            issue=start-pd.Timedelta(minutes=10)
            x=mos[r.row,half*12:half*12+12].copy();x[:,15]-=half*12
            if not np.isfinite(x).all():continue
            seq=np.full((12,7),np.nan,np.float32)
            hg=hgroups.get(hrrr_run)
            if hg is not None:
                dates=pd.date_range(start+pd.Timedelta(hours=1),end,freq='h')
                picked=hg.reindex(dates)
                assert (picked.assumed_available_time_utc.dropna()<=issue).all()
                seq=picked[HRRR_FIELDS].to_numpy(np.float32)
            case=start.strftime('%Y%m%d');radidx=ri.get(case)
            hasradar=half==0 and radidx is not None
            xx=rad['x'][radidx] if hasradar else np.zeros((9,5,16,16),np.float32)
            local=rad['local_stats'][radidx,0] if hasradar else np.full((9,8),np.nan,np.float32)
            record=dict(row=len(rows),start_utc=start.isoformat(),end_utc=end.isoformat(),issue_utc=issue.isoformat(),
                mos_run_utc=run.isoformat(),mos_assumed_available_utc=(run+pd.Timedelta(hours=6)).isoformat(),
                hrrr_run_utc=hrrr_run.isoformat(),hrrr_complete=bool(np.isfinite(seq).all()),
                radar_available=hasradar,half=half,winter=r.winter,season_end_year=r.winter+1,
                source_file=t.source_file,split='train' if r.winter<=2021 else 'validation' if r.winter==2022 else 'test')
            rows.append(record)
            for key,v in dict(mos=x,hrrr=seq,radar=xx,local=local,static=static_features(start,issue),
                history=histories(targets,issue),snow=t.snowfall_inches,lpe=t.lpe_inches).items():arr[key].append(v)
    data={k:np.asarray(v,dtype=np.float32) for k,v in arr.items()};meta=pd.DataFrame(rows)
    np.savez_compressed(OUT/'arrays.npz',**data);meta.to_csv(OUT/'index.csv',index=False)
    modern=meta.hrrr_complete &meta.winter.between(2020,2023)&pd.to_datetime(meta.start_utc).dt.month.isin([11,12,1,2,3,4])
    meta['modern']=modern;meta.to_csv(OUT/'index.csv',index=False)
    counts=meta.groupby(['winter','half']).size().unstack(fill_value=0)
    counts.to_csv(OUT/'coverage.csv')
    audit={'source_target_rows':len(targets),'matched_MOS_rows':len(meta),'modern_HRRR_rows':int(modern.sum()),
        'time_convention':'Source LST is interval START; fixed MST+7UTC; verified12h endpoints, never daily disaggregation.',
        'targets':'Alta Collins manual fresh snowfall inches and liquid equivalent inches',
        'forecast_policy':'00UTC MOS available06UTC;06UTC HRRR available08UTC. Issue10min before each11/23UTC targetstart. Evening guidance intentionallyolder, no12UTCrun substitution.',
        'history_policy':'Only manual observations whose12hintervalendedatleast2hbeforeissue; publicationlagassumption, receiptsunavailable.',
        'radar_policy':'Morning09:20–10:40UTC context only. Eveningradar explicitlymissing; no carryforward.',
        'hrrr_fields':HRRR_FIELDS,'hrRRpoint':'Nearest3kmgridpointconsistentwithgenericliveHRRR; nativehourlyforecastintervals.',
        'counts':meta.groupby(['split']).size().to_dict(),'modern_counts':meta[modern].groupby('split').size().to_dict(),
        'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [source,nwpbase/'broad_alta.npz',nwpbase/'broad_alta_index.csv',hp,radpath]],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in ['sources']},indent=2))
if __name__=='__main__':main()
