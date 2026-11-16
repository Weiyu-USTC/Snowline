"""Acquire archived JMA GSM forecasts through Open-Meteo, never reanalysis.

The previous-runs service aligns values by fixed lead offset; it does not return
one coherent run. We preserve that distinction. For daily leads1,3,6, offsets
2,4,7 leave a nominal24h source-age buffer at the declared issue time. The
per-hour source-time bound follows service documentation, not release metadata.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,time,urllib.request,urllib.parse,urllib.error
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/nwp_recent'
SITES={'vail':{'latitude':39.6176,'longitude':-106.38,'elevation':3140.},'alta':{'latitude':40.57607,'longitude':-111.63889,'elevation':2945.},'mansfield':{'latitude':44.5247,'longitude':-72.8153,'elevation':1204.},'jay':{'latitude':44.9381,'longitude':-72.5022,'elevation':560.8}}
VARS=['temperature_2m','precipitation','wind_speed_10m','wind_direction_10m','relative_humidity_2m','pressure_msl','snowfall']
OFFSETS=[1,2,3,4,5,6,7];LEADS={2:1,4:3,7:6}
DOCS=['https://open-meteo.com/en/docs/previous-runs-api','https://open-meteo.com/en/docs/single-runs-api','https://open-meteo.com/en/docs/jma-api']


def fetch(url,path):
    meta=path.with_suffix('.meta.json')
    if path.exists() and meta.exists():return json.loads(gzip.decompress(path.read_bytes()))
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Snowfall research demo (public noncommercial data)','Accept-Encoding':'gzip'})
            with urllib.request.urlopen(req,timeout=90) as response:
                data=response.read();headers=dict(response.headers);status=response.status
            if headers.get('Content-Encoding','')=='gzip':data=gzip.decompress(data)
            decoded=json.loads(data)
            if isinstance(decoded,dict) and decoded.get('error'):raise RuntimeError(decoded)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(gzip.compress(data,mtime=0))
            meta.write_text(json.dumps({'url':url,'http_status':status,'response_headers':headers,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'uncompressed_bytes':len(data),'uncompressed_sha256':hashlib.sha256(data).hexdigest()},indent=2))
            return decoded
        except (urllib.error.URLError,TimeoutError) as exc:
            if attempt==3:raise
            time.sleep(2+attempt*3)


def params(sites):
    return {'latitude':','.join(str(SITES[s]['latitude']) for s in sites),'longitude':','.join(str(SITES[s]['longitude']) for s in sites),'elevation':','.join(str(SITES[s]['elevation']) for s in sites),'models':'jma_gsm','timezone':'GMT','wind_speed_unit':'ms','cell_selection':'nearest'}


def historical(first=2018,last=2025):
    OUT.mkdir(exist_ok=True);frames=[];inventories=[]
    for year in range(first,last+1):
        sites=['vail','alta'];p=params(sites);p.update({'start_date':f'{year}-09-29','end_date':f'{year+1}-05-02','hourly':','.join(f'{v}_previous_day{k}' for k in OFFSETS for v in VARS)})
        url='https://previous-runs-api.open-meteo.com/v1/forecast?'+urllib.parse.urlencode(p);path=OUT/'raw'/f'jma_gsm_{year}_{year+1}.json.gz';records=fetch(url,path)
        for site,r in zip(sites,records):
            for offset in OFFSETS:
                d=pd.DataFrame({'valid_time_utc':pd.to_datetime(r['hourly']['time'],utc=True),'site':site,'nominal_offset_days':offset})
                for v in VARS:d[v]=np.asarray(r['hourly'][f'{v}_previous_day{offset}'],dtype=float)
                d['model']='jma_gsm';d['grid_latitude']=r['latitude'];d['grid_longitude']=r['longitude'];d['requested_elevation_m']=r['elevation']
                frames.append(d);inventories.append({'site':site,'winter':year,'nominal_offset_days':offset,'hourly_rows':len(d),'grid_latitude':r['latitude'],'grid_longitude':r['longitude'],'requested_elevation_m':r['elevation'],'valid_counts':{v:int(d[v].notna().sum()) for v in VARS}})
        print(json.dumps({'stage':'downloaded','winter':year,'bytes':path.stat().st_size,'counts':inventories[-1]}),flush=True)
    hourly=pd.concat(frames,ignore_index=True).drop_duplicates(['site','valid_time_utc','nominal_offset_days']).sort_values(['site','valid_time_utc','nominal_offset_days'])
    hourly['implied_source_time_upper_bound_utc']=hourly.valid_time_utc-pd.to_timedelta(hourly.nominal_offset_days,unit='D')
    hourly.to_csv(OUT/'jma_gsm_hourly_fixed_lead.csv.gz',index=False)
    (OUT/'historical_inventory.json').write_text(json.dumps({'source_kind':'Archived operational forecast values at fixed hourly lead offsets; not reanalysis and not a coherent single run','documentation':DOCS,'model':'jma_gsm','periods':inventories,'temperature_elevation_policy':'Explicit fixed target elevation supplied to API; provider terrain adjustment retained, not model-ground temperature','timing_policy':'Per-hour implied source-time upper bound from documented offset; actual initialization/publication timestamps unavailable','lead_policy':'Daily lead1/3/6 use offset2/4/7, leaving at least24h nominal source-age buffer before declared daily issue','forecast_horizon_note':'Availability varies with JMA run horizon/version; missing offset values remain missing','snowfall_field_note':'raw_model_snowfall_inches is an API-derived baseline, not native JMA snow: provider uses a0C phase threshold and0.7cm snow per mm liquid','grid_note':'JMA GSM0.5degree grid; native6hour fields interpolated byprovider tohourly; actualselectedgrids andexplicitrequested elevation retained; targetelevationdownscalingenabled','data_license_url':'https://open-meteo.com/en/terms'},indent=2))
    derive_daily(hourly)


def derive_daily(hourly):
    d=hourly.copy();d['wind_u_10m']=-d.wind_speed_10m*np.sin(np.deg2rad(d.wind_direction_10m));d['wind_v_10m']=-d.wind_speed_10m*np.cos(np.deg2rad(d.wind_direction_10m));d['cold_precipitation_mm']=d.precipitation.where(d.temperature_2m<=1,0).where(d.temperature_2m.notna()&d.precipitation.notna())
    rows=[]
    for endhour in [10,11,12]:
        d['target_endpoint_utc']=(d.valid_time_utc-pd.Timedelta(hours=endhour)).dt.ceil('D')+pd.Timedelta(hours=endhour)
        for (site,offset,endpoint),g in d[d.nominal_offset_days.isin(LEADS)].groupby(['site','nominal_offset_days','target_endpoint_utc']):
            if len(g)!=24:continue
            if endpoint.month not in [10,11,12,1,2,3,4]:continue
            lead=LEADS[offset];issue=endpoint-pd.Timedelta(days=lead)
            row={'site':site,'target_date':str(endpoint.date()),'target_endpoint_utc':endpoint.isoformat(),'target_start_utc':(endpoint-pd.Timedelta(days=1)).isoformat(),'window_end_hour_utc':endhour,'daily_lead_days':lead,'nominal_offset_days':offset,'issue_time_utc':issue.isoformat(),'latest_implied_source_time_utc':g.implied_source_time_upper_bound_utc.max().isoformat(),'source_age_buffer_hours':float((issue-g.implied_source_time_upper_bound_utc.max()).total_seconds()/3600),'hours':len(g),'model':'jma_gsm','temperature_mean_c':g.temperature_2m.mean(),'temperature_min_c':g.temperature_2m.min(),'temperature_max_c':g.temperature_2m.max(),'precipitation_mm':g.precipitation.sum(min_count=24),'cold_precipitation_mm':g.cold_precipitation_mm.sum(min_count=24),'wind_u_mean_ms':g.wind_u_10m.mean(),'wind_v_mean_ms':g.wind_v_10m.mean(),'wind_speed_mean_ms':g.wind_speed_10m.mean(),'wind_speed_max_ms':g.wind_speed_10m.max(),'humidity_mean_pct':g.relative_humidity_2m.mean(),'pressure_mean_hpa':g.pressure_msl.mean(),'pressure_change_hpa':g.pressure_msl.iloc[-1]-g.pressure_msl.iloc[0],'raw_model_snowfall_inches':g.snowfall.sum(min_count=24)/2.54,'complete_core_hours':int(g[['temperature_2m','precipitation','wind_speed_10m','wind_direction_10m']].notna().all(axis=1).sum())}
            assert row['source_age_buffer_hours']>=24
            rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(OUT/'jma_gsm_daily_conservative.csv',index=False)
    print(json.dumps({'stage':'derived','rows':len(result),'complete_core_rows':int((result.complete_core_hours==24).sum())}),flush=True)


def current():
    sites=list(SITES);p=params(sites);p.update({'run':'2026-11-04T00:00','hourly':','.join(VARS),'forecast_days':7})
    url='https://single-runs-api.open-meteo.com/v1/forecast?'+urllib.parse.urlencode(p);records=fetch(url,OUT/'raw'/'current_jma_gsm_2026110400.json.gz')
    frames=[]
    for site,r in zip(sites,records):
        d=pd.DataFrame(r['hourly']);d['site']=site;d['model']='jma_gsm';d['run_requested_utc']='2026-11-04T00:00:00+00:00';d['grid_latitude']=r['latitude'];d['grid_longitude']=r['longitude'];d['requested_elevation_m']=r['elevation'];frames.append(d)
    pd.concat(frames,ignore_index=True).to_csv(OUT/'current_jma_gsm_hourly.csv',index=False)
    print(json.dumps({'stage':'current','sites':sites,'rows':sum(len(d) for d in frames)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['historical','current','all','derive'],default='all');ap.add_argument('--first',type=int,default=2018);ap.add_argument('--last',type=int,default=2025);a=ap.parse_args()
    if a.mode in ['all','historical']:historical(a.first,a.last)
    if a.mode=='derive':
        d=pd.read_csv(OUT/'jma_gsm_hourly_fixed_lead.csv.gz');d['valid_time_utc']=pd.to_datetime(d.valid_time_utc,utc=True);d['implied_source_time_upper_bound_utc']=pd.to_datetime(d.implied_source_time_upper_bound_utc,utc=True);derive_daily(d)
    if a.mode in ['all','current']:current()
