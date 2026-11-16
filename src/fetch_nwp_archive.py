"""Download archived issued GFS MOS point forecasts, with original encodings.

Iowa State's archive contains the NOAA operational station-MOS product, not
reanalysis and not retrospective GEFS forecasts. Sequential modest requests
respect the service's rate limit. Each original CSV is preserved gzip-compressed.
"""
from __future__ import annotations
import argparse,gzip,hashlib,io,json,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/nwp_archive_probe';RAW=OUT/'mos_raw'
BASE='https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py'
DEFAULT={'KSLC':range(2004,2024),'KMVL':range(2004,2017),'KLXV':range(2019,2026)}
FIELDS=['runtime','ftime','station','model','tmp','dpt','wdr','wsp','p06','q06','p12','q12','snw','pos','poz','typ','cld']


def fetch(station,start,end,delay=3.):
    RAW.mkdir(parents=True,exist_ok=True)
    name=f'{station}_{start[:10]}_{end[:10]}';path=RAW/f'{name}.csv.gz';mp=RAW/f'{name}.json'
    url=BASE+'?'+urlencode(dict(station=station,model='GFS',sts=start,ets=end,format='csv'))
    if path.exists() and mp.exists():return path
    time.sleep(delay)
    for attempt in range(4):
        try:
            with urlopen(Request(url,headers={'User-Agent':'snowfall-research-demo/1.0'}),timeout=90) as response:
                body=response.read(30_000_000);headers=dict(response.headers)
                if len(body)>=30_000_000:raise ValueError('30MB per-request budget exceeded')
            break
        except HTTPError as error:
            if error.code not in [429,502,503,504] or attempt==3:raise
            delay=min(45.,max(10.,float(error.headers.get('Retry-After',10)))*(attempt+1));print(json.dumps({'retry':name,'delay':delay,'http':error.code}),flush=True);time.sleep(delay)
    d=pd.read_csv(io.BytesIO(body),low_memory=False)
    if not {'runtime','ftime','station','model'}.issubset(d.columns):raise ValueError(f'Unexpected schema: {body[:200]}')
    if len(d):
        assert set(d.station)=={station};assert set(d.model)=={'GFS'}
        r=pd.to_datetime(d.runtime,utc=True);v=pd.to_datetime(d.ftime,utc=True)
        assert (v>r).all()
    path.write_bytes(gzip.compress(body,mtime=0));mp.write_text(json.dumps({'url':url,'retrieved_utc':pd.Timestamp.now(tz='UTC').isoformat(),'bytes_uncompressed':len(body),'sha256_uncompressed':hashlib.sha256(body).hexdigest(),'rows':len(d),'headers':headers,'archive_kind':'Issued operational NOAA GFS MOS guidance hosted by IEM; model run time is not original receipt time'},indent=2))
    print(json.dumps({'downloaded':name,'rows':len(d),'bytes':len(body)}),flush=True)
    return path


def canonical(station):
    paths=sorted(RAW.glob(f'{station}_*.csv.gz'));frames=[]
    for p in paths:
        d=pd.read_csv(p,low_memory=False);d=d.reindex(columns=FIELDS)
        if len(d):frames.append(d)
    if not frames:return
    d=pd.concat(frames,ignore_index=True)
    d['runtime']=pd.to_datetime(d.runtime,utc=True);d['ftime']=pd.to_datetime(d.ftime,utc=True)
    d=d.sort_values(['runtime','ftime']).drop_duplicates(['station','model','runtime','ftime'],keep='last')
    d=d[d.runtime.dt.hour.eq(0)].copy()
    for col in ['tmp','dpt','wdr','wsp','p06','q06','p12','q12','snw','pos','poz']:
        d[col]=pd.to_numeric(d[col],errors='coerce')
    d['temperature_c']=(d.tmp.where(d.tmp.ne(999))-32)*5/9
    d['dewpoint_c']=(d.dpt.where(d.dpt.ne(999))-32)*5/9
    d['wind_direction_degrees']=d.wdr.where(d.wdr.between(0,360))
    d['wind_speed_mps']=d.wsp.where(d.wsp.between(0,98))*.514444
    theta=np.deg2rad(d.wind_direction_degrees)
    d['wind_u_mps']=-d.wind_speed_mps*np.sin(theta);d['wind_v_mps']=-d.wind_speed_mps*np.cos(theta)
    for hours in [6,12]:
        d[f'precip_probability_{hours}h']=d[f'p{hours:02}'].where(d[f'p{hours:02}'].between(0,100))/100
        d[f'qpf_category_{hours}h']=d[f'q{hours:02}'].where(d[f'q{hours:02}'].between(0,6))
    d['lead_hours']=(d.ftime-d.runtime).dt.total_seconds()/3600
    d.to_csv(OUT/f'mos_{station}_00utc.csv.gz',index=False)
    rows=[]
    for year,g in d.groupby(d.runtime.dt.year):
        rows.append({'station':station,'calendar_year':int(year),'runs':g.runtime.nunique(),'rows':len(g),'first_run':str(g.runtime.min()),'last_run':str(g.runtime.max()),'temp_fraction':float(g.temperature_c.notna().mean()),'wind_fraction':float(g.wind_speed_mps.notna().mean()),'p06_records':int(g.precip_probability_6h.notna().sum()),'q06_records':int(g.qpf_category_6h.notna().sum())})
    pd.DataFrame(rows).to_csv(OUT/f'mos_{station}_coverage.csv',index=False)
    return d


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--station',choices=list(DEFAULT)+['all'],default='all');parser.add_argument('--canonical-only',action='store_true');args=parser.parse_args()
    stations=list(DEFAULT) if args.station=='all' else [args.station]
    for station in stations:
        if not args.canonical_only:
            for y in DEFAULT[station]:fetch(station,f'{y}-10-01T00:00Z',f'{y+1}-05-01T00:00Z')
            fetch(station,'2026-11-03T00:00Z','2026-11-05T00:00Z')
        d=canonical(station);print(json.dumps({'station_complete':station,'rows':len(d),'runs':d.runtime.nunique()}),flush=True)
    metadata={'source':'https://mesonet.agron.iastate.edu/mos/','api_documentation':BASE+'?help','noaa_product':'https://vlab.noaa.gov/web/mdl/mav-card','archive_first_GFS_date':'2003-12-16','selected_model':'GFS','selected_runtime_hour_UTC':0,'horizon_hours':[6,72],'encodings':{'tmp':'whole°F;999 missing','dpt':'whole°F;999 missing','wdr':'IEM already multiplies raw code by10: degrees;990 missing','wsp':'knots;99 missing','p06':'probability of>=.01inch liquidprecip over6h ending ftime, percent;999missing','q06':'ordinal liquidQPFcategory over6h endingftime;0..6;notinches','q06_bounds_inches':{'0':[0,0],'1':[.01,.099999],'2':[.1,.249999],'3':[.25,.499999],'4':[.5,.999999],'5':[1,1.999999],'6':[2,None]}},'time_caution':'runtime is modelinitialization; firstoriginalpublication/receipt timestampnotprovided. A6hour publicationbufferisassumed downstream. Operational model/MOS versions changedoverarchive.'}
    (OUT/'mos_source_metadata.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
