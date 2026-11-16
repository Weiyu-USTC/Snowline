"""Native NOAA NEXRAD Level III DPR acquisition and causal mountain crops.

DPR is a radar-derived liquid precipitation rate, NOT observed fresh snow.
Public archive provenance is retained in each crop. No MRMS values are predictors.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, json, os, tempfile, time, xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import requests
from scipy.ndimage import map_coordinates
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'snowline-mpl'))
from metpy.io import Level3File

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/nexrad_short'
BUCKET='https://unidata-nexrad-level3.s3.amazonaws.com/'
SITES={
 'alta':dict(site='alta', radar='MTX',latitude=40.57607,longitude=-111.63889,elevation_m=2945., target='Alta Collins'),
 'vail_ftg':dict(site='vail',radar='FTG',latitude=39.6176,longitude=-106.38,elevation_m=3140.,target='Vail CAVMM mountain station'),
 'vail_gjx':dict(site='vail',radar='GJX',latitude=39.6176,longitude=-106.38,elevation_m=3140.,target='Vail CAVMM mountain station'),
 'mansfield':dict(site='mansfield',radar='CXX',latitude=44.464,longitude=-72.6907,elevation_m=223.1,target='Stowe town NOAA station; not resort summit'),
 'jay':dict(site='jay',radar='CXX',latitude=44.8916,longitude=-72.4435,elevation_m=325.8,target='Westfield NOAA station near Jay; not resort summit'),
}
GRID_N=129
PIXEL_KM=2.
CONTEXT_LAGS_MINUTES=(70,40,10)
SCAN_MAX_AGE_MINUTES=15
INVENTORY_TTL_SECONDS=300
INVENTORY_SETTLING_HOURS=24

def iso(t): return t.replace(tzinfo=timezone.utc).isoformat() if t.tzinfo is None else t.isoformat()
def utc(t): return datetime.fromisoformat(str(t).replace('Z','+00:00')).astimezone(timezone.utc)
def get(url,params=None,tries=4):
 for k in range(tries):
  try:
   r=requests.get(url,params=params,timeout=(15,70));r.raise_for_status();return r
  except requests.RequestException:
   if k==tries-1:raise
   time.sleep(.5*(k+1))

def inventory(radar,date):
 date=utc(date)
 prefix=f"{radar}_DPR_{date:%Y_%m_%d}_"
 dest=DATA/'inventories'/f'{prefix}.json';dest.parent.mkdir(parents=True,exist_ok=True)
 now=datetime.now(timezone.utc)
 day_end=date.replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=1)
 # A live listing does not become complete merely because UTC midnight passes.
 # Reuse a settled archive only if it was actually listed after the entire day
 # plus an ingestion allowance. Legacy bare lists have no receipt provenance:
 # refresh them once instead of treating their file mtime as completeness.
 if dest.exists():
  try:
   saved=json.loads(dest.read_text())
   if isinstance(saved,dict) and saved.get('schema_version')==2 and saved.get('prefix')==prefix:
    fetched=utc(saved['retrieved_at']);age=(now-fetched).total_seconds()
    settled=fetched>=day_end+timedelta(hours=INVENTORY_SETTLING_HOURS)
    if isinstance(saved.get('entries'),list) and age>=0 and (settled or age<INVENTORY_TTL_SECONDS):
     return saved['entries']
  except (OSError,ValueError,KeyError,TypeError):pass
 items=[];params={'list-type':'2','prefix':prefix,'max-keys':1000}
 while True:
  root=ET.fromstring(get(BUCKET,params).content); ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
  for node in root.findall('s:Contents',ns):
   key=node.findtext('s:Key',namespaces=ns)
   t=datetime.strptime(key[-19:],'%Y_%m_%d_%H_%M_%S').replace(tzinfo=timezone.utc)
   items.append({'key':key,'volume_start_utc':iso(t),'bytes':int(node.findtext('s:Size',namespaces=ns))})
  token=root.findtext('s:NextContinuationToken',namespaces=ns)
  if not token:break
  params['continuation-token']=token
 saved={'schema_version':2,'prefix':prefix,'retrieved_at':iso(datetime.now(timezone.utc)),
        'day_end_utc':iso(day_end),'settling_hours':INVENTORY_SETTLING_HOURS,'entries':items}
 tmp=dest.with_name(dest.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
 tmp.write_text(json.dumps(saved));tmp.replace(dest);return items

def context_inventory_dates(issue):
 """All UTC archive days touched by the causal scan-selection windows."""
 issue=utc(issue)
 first=(issue-timedelta(minutes=max(CONTEXT_LAGS_MINUTES)+SCAN_MAX_AGE_MINUTES)).replace(hour=0,minute=0,second=0,microsecond=0)
 last=(issue-timedelta(minutes=min(CONTEXT_LAGS_MINUTES))).replace(hour=0,minute=0,second=0,microsecond=0)
 days=[]
 while first<=last:
  days.append(first);first+=timedelta(days=1)
 return days

def download(key):
 p=DATA/'raw'/key[:3]/key
 if not p.exists():
  p.parent.mkdir(parents=True,exist_ok=True)
  b=get(BUCKET+key).content
  # Atomic to tolerate shared CXX files requested by two target crops.
  tmp=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp');tmp.write_bytes(b);tmp.replace(p)
 return p

def geometry(radar_lat,radar_lon,lat,lon):
 # Great-circle ground range and forward azimuth from radar, spherical Earth.
 la1=np.deg2rad(radar_lat);la2=np.deg2rad(lat);dl=np.deg2rad(lon-radar_lon)
 a=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin(dl/2)**2
 distance=6371.*2*np.arctan2(np.sqrt(a),np.sqrt(np.maximum(0,1-a)))
 az=(np.degrees(np.arctan2(np.sin(dl)*np.cos(la2),np.cos(la1)*np.sin(la2)-np.sin(la1)*np.cos(la2)*np.cos(dl)))+360)%360
 return distance,az

def decode_crop(path,site):
 f=Level3File(str(path));packet=f.sym_block[0][0];c=packet['components']
 if f.header.code!=176:raise ValueError('Expected genuine LevelIII DPR product176')
 attributes=c.radials[0].attributes
 if 'inches/hour' not in attributes:raise ValueError(f'Unexpected DPR units: {attributes}')
 raw=np.asarray([r.data for r in c.radials]);rate=f.map_data(raw)*25.4
 # A zero DPR code is a valid zero rate, not the missing flag of reflectivity products.
 if not np.all(np.diff([r.azimuth for r in c.radials])>0):raise ValueError('Unordered radial azimuth')
 x=(np.arange(GRID_N)-GRID_N//2)*PIXEL_KM
 lat=site['latitude']+x[:,None]/111.195
 lon=site['longitude']+x[None,:]/(111.195*np.cos(np.deg2rad(site['latitude'])))
 distance,az=geometry(f.lat,f.lon,lat,lon)
 # DPR has one-degree sectors, azimuth gives leading edge; range first_gate is gate center.
 ai=np.searchsorted(np.asarray([r.azimuth for r in c.radials]),az,side='right')-1
 gi=np.rint((distance*1000-c.first_gate)/c.gate_width).astype(int)
 valid=(distance<=f.max_range)&(gi>=0)&(gi<rate.shape[1])
 grid=rate[ai%len(rate),np.clip(gi,0,rate.shape[1]-1)]
 grid=np.where(valid&np.isfinite(grid)&(grid>=0),grid,np.nan).astype(np.float32)
 d,_=geometry(f.lat,f.lon,site['latitude'],site['longitude']);d=float(d)
 h=float(packet['height'])
 # Standard 4/3 Earth,0.5deg beam centre is a geometry diagnostic, not a terrain blockage model.
 re=4/3*6371.;beam=(np.sqrt(d*d+re*re+2*d*re*np.sin(np.deg2rad(.5)))-re)*1000+h
 meta={'key':Path(path).name,'source_url':BUCKET+Path(path).name,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
  'volume_start_utc':iso(f.metadata['vol_time']),'product_generated_utc':iso(f.metadata['prod_time']),
  'radar_latitude':float(f.lat),'radar_longitude':float(f.lon),'radar_elevation_m':h,'distance_km':d,
  'nominal_0_5deg_beam_center_m':beam,'beam_above_target_m':beam-site['elevation_m'],
  'product':'NEXRAD Level III DPR code176','original_units':'inches/hour','units':'mm/hour',
  'valid_grid_fraction':float(np.isfinite(grid).mean()),'max_elevation_used_deg':float(f.metadata.get('max_elev',np.nan)),
  'filled_percent':float(f.metadata.get('percent_filled',np.nan))}
 return grid,meta

def validate_crop(path,case,site_key):
 """Validate reuse against the requested issue; a cache hit is not a new issue."""
 issue=utc(case['forecast_issue_utc']);site=SITES[site_key]
 with np.load(path,allow_pickle=False) as z:
  rates=np.asarray(z['rate_mm_h'])
  saved_case=json.loads(str(z['case_json']));saved_site=json.loads(str(z['site_json']))
  metas=json.loads(str(z['metadata_json']));pixel=float(z['pixel_km'])
 if saved_case.get('case_id')!=case['case_id'] or utc(saved_case.get('forecast_issue_utc'))!=issue:
  raise ValueError('Cached radar crop has a different forecast issue')
 if any(saved_site.get(k)!=site[k] for k in ['site','radar','latitude','longitude','elevation_m']):
  raise ValueError('Cached radar crop has a different target or radar')
 if rates.shape!=(3,GRID_N,GRID_N) or pixel!=PIXEL_KM:
  raise ValueError('Radar crop shape or grid spacing differs')
 if np.isinf(rates).any() or (rates[np.isfinite(rates)]<0).any() or not np.isfinite(rates).any(axis=(1,2)).all():
  raise ValueError('Radar crop rates are invalid or entirely missing')
 if len(metas)!=3 or len({m.get('key') for m in metas})!=3:
  raise ValueError('Radar crop must contain three distinct scans')
 for minutes,m in zip(CONTEXT_LAGS_MINUTES,metas):
  cutoff=issue-timedelta(minutes=minutes)
  volume=utc(m['volume_start_utc']);generated=utc(m['product_generated_utc'])
  if not cutoff-timedelta(minutes=SCAN_MAX_AGE_MINUTES)<=volume<=cutoff or generated>issue-timedelta(minutes=2):
   raise ValueError('Radar crop scan is stale or was unavailable at forecast issue')
  if m.get('units')!='mm/hour' or m.get('original_units')!='inches/hour':
   raise ValueError('Radar crop precipitation units differ')
  if not str(m.get('key','')).startswith(site['radar']+'_DPR_'):
   raise ValueError('Radar crop source product or radar differs')
 return {'forecast_issue_utc':iso(issue),
         'latest_scan_utc':max(m['volume_start_utc'] for m in metas),
         'latest_product_generated_utc':max(m['product_generated_utc'] for m in metas)}

def acquire_case(case,site_key):
 site=SITES[site_key];issue=utc(case['forecast_issue_utc']);dest=DATA/'crops'/site_key/(case['case_id']+'.npz')
 cache_error=None
 if dest.exists():
  try:
   info=validate_crop(dest,case,site_key)
   return {'case_id':case['case_id'],'site':site_key,'status':'ok','cached':True,
           'checked_at':iso(datetime.now(timezone.utc)),**info}
  except Exception as exc:cache_error=str(exc)
 try:
  entries=[]
  for day in context_inventory_dates(issue):entries.extend(inventory(site['radar'],day))
  grids=[];metas=[]
  for minutes in CONTEXT_LAGS_MINUTES:
   cutoff=issue-timedelta(minutes=minutes)
   candidates=[e for e in entries if cutoff-timedelta(minutes=SCAN_MAX_AGE_MINUTES)<=utc(e['volume_start_utc'])<=cutoff]
   chosen=None
   for entry in sorted(candidates,key=lambda e:e['volume_start_utc'],reverse=True):
    g,m=decode_crop(download(entry['key']),site)
    if utc(m['product_generated_utc'])<=issue-timedelta(minutes=2):chosen=(g,m);break
   if chosen is None:raise ValueError(f'No available scan before {iso(cutoff)}')
   grids.append(chosen[0]);metas.append(chosen[1])
  if len({m['key'] for m in metas})!=3:raise ValueError('Need three distinct scans')
  dest.parent.mkdir(parents=True,exist_ok=True)
  tmp=dest.with_name(dest.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
  try:
   with tmp.open('wb') as output:
    np.savez_compressed(output,rate_mm_h=np.stack(grids),metadata_json=np.array(json.dumps(metas)),case_json=np.array(json.dumps(case)),site_json=np.array(json.dumps(site)),pixel_km=np.array(PIXEL_KM))
   info=validate_crop(tmp,case,site_key);tmp.replace(dest)
  finally:
   if tmp.exists():tmp.unlink()
  return {'case_id':case['case_id'],'site':site_key,'status':'ok','cached':False,
          'checked_at':iso(datetime.now(timezone.utc)),**info}
 except Exception as e:
  error=str(e)+(f' (existing cache rejected: {cache_error})' if cache_error else '')
  return {'case_id':case['case_id'],'site':site_key,'status':'error','error':error}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--current',action='store_true');ap.add_argument('--workers',type=int,default=10);ap.add_argument('--limit',type=int);a=ap.parse_args()
 DATA.mkdir(parents=True,exist_ok=True)
 if a.current:
  now=datetime.now(timezone.utc);issue=now.replace(second=0,microsecond=0,minute=(now.minute//10)*10)
  cases=[dict(case_id='current_'+issue.strftime('%Y%m%dT%H%M'),forecast_issue_utc=iso(issue),split='live',winter=issue.year)]
  (DATA/'current_case.json').write_text(json.dumps(cases[0],indent=2))
 else:cases=json.loads((DATA/'frozen_schedule.json').read_text())['cases']
 if a.limit:cases=cases[:a.limit]
 tasks=[(c,s) for c in cases for s in SITES];results=[]
 with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
  futs=[ex.submit(acquire_case,c,s) for c,s in tasks]
  for i,f in enumerate(cf.as_completed(futs),1):
   r=f.result();results.append(r)
   if r['status']=='error' or i%25==0:print(i,'/',len(tasks),json.dumps(r),flush=True)
 path=DATA/('current_acquisition.json' if a.current else 'acquisition.json');path.write_text(json.dumps(results,indent=2))
 print('finished',len(results),'errors',sum(r['status']=='error' for r in results),flush=True)
if __name__=='__main__':main()
