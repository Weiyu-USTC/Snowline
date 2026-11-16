"""Causal conservative provider baseline for the radar-matched 48h experiment.

JMA GSM previous_day3 is a fixed72h-lead mosaic, not a coherent run. All values
in the historical48h target window have implied source bounds before issuance.
This does not reproduce the live best-match forecast or exact publication time.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,urllib.parse
import numpy as np
import pandas as pd
from fetch_recent_nwp import fetch,VARS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/neural48_training/provider'
SITES={'alta':(40.57607,-111.63889,2945.),'vail':(39.6176,-106.38,3140.),
       'mansfield':(44.4640,-72.6907,223.1),'jay':(44.8916,-72.4435,325.8)}

def request(sites,start,end,path):
    p={'latitude':','.join(str(SITES[s][0]) for s in sites),
       'longitude':','.join(str(SITES[s][1]) for s in sites),
       'elevation':','.join(str(SITES[s][2]) for s in sites),'models':'jma_gsm',
       'timezone':'GMT','wind_speed_unit':'ms','cell_selection':'nearest',
       'start_date':start,'end_date':end,
       'hourly':','.join(v+'_previous_day3' for v in VARS)}
    return fetch('https://previous-runs-api.open-meteo.com/v1/forecast?'+urllib.parse.urlencode(p),path)

def records_to_frames(sites,records,source):
    if isinstance(records,dict):records=[records]
    frames=[]
    for site,r in zip(sites,records):
        d=pd.DataFrame({'valid_time_utc':pd.to_datetime(r['hourly']['time'],utc=True)})
        for v in VARS:d[v]=np.asarray(r['hourly'][v+'_previous_day3'],float)
        d['site']=site;d['implied_source_bound_utc']=d.valid_time_utc-pd.Timedelta(days=3)
        d['source_file']=str(source.relative_to(ROOT));d['source_grid_latitude']=r['latitude'];d['source_grid_longitude']=r['longitude']
        frames.append(d)
    return frames

def main(current=False):
    OUT.mkdir(parents=True,exist_ok=True);frames=[];sources=[]
    if current:
        today=pd.Timestamp.now(tz='UTC').date();last=today+pd.Timedelta(days=4)
        path=OUT/'raw'/f'current_{today}.json.gz'
        records=request(list(SITES),str(today),str(last),path)
        frames+=records_to_frames(list(SITES),records,path);sources.append(path)
    else:
        for year in range(2020,2024):
            # Original snapshots preserve actual provider grids and mountain
            # elevations for Alta/Vail. Never overwrite their old response.
            old=ROOT/f'data/nwp_recent/raw/jma_gsm_{year}_{year+1}.json.gz'
            records=json.loads(gzip.decompress(old.read_bytes()))
            frames+=records_to_frames(['vail','alta'],records,old);sources.append(old)
            path=OUT/'raw'/f'vermont_proxy_{year}.json.gz'
            records=request(['mansfield','jay'],f'{year}-10-30',f'{year+1}-05-03',path)
            frames+=records_to_frames(['mansfield','jay'],records,path);sources.append(path)
            print(json.dumps({'winter':year,'provider_raw_bytes':path.stat().st_size}),flush=True)
    d=pd.concat(frames).drop_duplicates(['site','valid_time_utc']).sort_values(['site','valid_time_utc'])
    name='current' if current else 'historical'
    d.to_csv(OUT/f'{name}_hourly.csv.gz',index=False)
    manifest={'source':'Open-Meteo JMA GSM previous_day3 forecasts','sites':SITES,
        'timing':'Each hour has a nominal72h source-age offset; a mosaic, not one coherent run. Original issue and receipt timestamps are unavailable.',
        'baseline':'Provider snow conversion, not native JMA snow depth; do not equate with current best-match Open-Meteo.',
        'resolution':'JMA0.5degree/native6hour guidance interpolated by provider; nominalhourlyvalues are not independent hourlyinformation.',
        'source_urls':['https://open-meteo.com/en/docs/previous-runs-api','https://open-meteo.com/en/docs/jma-api'],
        'raw_files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        'counts':d.groupby('site').size().to_dict(),'generated_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    (OUT/f'{name}_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'status':'complete','counts':manifest['counts']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--current',action='store_true');a=p.parse_args();main(a.current)
