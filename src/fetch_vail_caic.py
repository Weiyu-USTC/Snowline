"""Acquire public CAIC Vail station tables, preserving source fields and clocks."""
from __future__ import annotations
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/vail_hourly'
RAW=OUT/'raw'
STATIONS={
 'CAVMM':{'name':'Vail SA - Mid-Mtn','elevation_m':3140,'elevation_ft':10303,'provider':'VailResort','role':'candidate daily Sno24 target and hourly inputs','latitude':39.6176,'longitude':-106.38},
 'CAVPQ':{'name':'Vail SA - PHQ','elevation_m':3428,'elevation_ft':11248,'provider':'VailResort','role':'hourly inputs; target suitability not presumed','latitude':39.6055,'longitude':-106.357},
 'CAVBS':{'name':'Vail SA - Blue Sky','elevation_m':3386,'elevation_ft':11109,'provider':'VailResort','role':'hourly wind/temperature inputs','latitude':39.5711,'longitude':-106.332},
 'CAVCB':{'name':'Vail SA - China Bowl','elevation_m':3362,'elevation_ft':11030,'provider':'VailResort','role':'hourly wind/temperature inputs','latitude':39.5977,'longitude':-106.338}}

def parse(data,station,year,month,url,name):
 text=data.decode('utf-8',errors='replace')
 head=re.search(r'<font[^>]*>(.*?)</font>',text,re.S)
 columns=re.sub('<[^>]+>','',head.group(1)).split()[2:] if head else []
 rows=[]
 for line in re.findall(r'<span[^>]*>(.*?)</span>',text,re.S):
  clean=html.unescape(re.sub('<[^>]+>','',line)).strip()
  parts=clean.split()
  if len(parts)<5: continue
  try: dt=datetime.strptime(' '.join(parts[:5]),'%Y %b %d %I:%M %p')
  except ValueError:continue
  if dt.year!=year or dt.month!=month:continue
  vals=parts[5:]
  if len(vals)!=len(columns):raise ValueError((name,columns,clean))
  row={'station_id':station,'datetime_display':dt.isoformat(),'date':dt.date().isoformat(),'original_timestamp':' '.join(parts[:5]),'source_file':'raw/'+name,'source_url':url,'source_unit_parameter':'e','raw_values_json':json.dumps(dict(zip(columns,vals)),separators=(',',':')),'qc_notes':'station_protocol_unverified;timestamp_offset_unverified;provider_qc_flags_not_exposed'}
  for col,v in zip(columns,vals):
   try: row[col]=float(v)
   except ValueError:row[col]=None
  if row.get('Sno24') is not None and (row['Sno24']<0 or row['Sno24']>=24):row['qc_notes']+=';Sno24_extreme_review'
  rows.append(row)
 return rows,columns

def fetch(task):
 station,year,month=task
 # Freeze acquisition at 04:00 displayed station time on November 4, 2026.
 day,hour=(4,4) if (year,month)==(2026,11) else (calendar.monthrange(year,month)[1],23)
 hours=(day-1)*24+hour+1
 params={'st':station,'date':f'{year}-{month:02d}-{day:02d} {hour:02d}','unit':'e','area':'caic','range':hours}
 url='https://stations.avalanche.state.co.us/tabular.php?'+urllib.parse.urlencode(params)
 name=f'{station}_{year}{month:02d}.html';path=RAW/name
 fetched=False
 try:
  if path.exists(): data=path.read_bytes()
  else:
   with urllib.request.urlopen(url,timeout=45) as response:data=response.read()
   path.write_bytes(data);fetched=True
   time.sleep(.3)
  rows,columns=parse(data,station,year,month,url,name)
  meta={'station_id':station,'year':year,'month':month,'url':url,'file':'raw/'+name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'records':len(rows),'columns':columns,'http_status':200,'retrieved_or_checked_utc':datetime.now(timezone.utc).isoformat(),'downloaded_this_run':fetched}
  return rows,meta
 except Exception as exc:
  return [],{'station_id':station,'year':year,'month':month,'url':url,'file':'raw/'+name,'error':str(exc),'records':0}

def write_results(results,prefix):
 import csv
 rows=[r for block,meta in results for r in block]
 dedup={};conflicts={}
 for r in rows:
  key=(r['station_id'],r['datetime_display'])
  if key in dedup:
   if dedup[key]['raw_values_json']!=r['raw_values_json']:
    conflicts.setdefault(key,[dedup[key]]).append(r)
   continue
  dedup[key]=r
 for key,originals in conflicts.items():
  r=dedup[key].copy()
  for field in list(r):
   if field not in {'station_id','datetime_display','date','original_timestamp','source_file','source_url','source_unit_parameter','raw_values_json','qc_notes'}:r[field]=None
  r['qc_notes']+=';ambiguous_duplicate_source_clock;numeric_values_masked'
  r['duplicate_raw_values_json']=json.dumps([json.loads(v['raw_values_json']) for v in originals],separators=(',',':'))
  dedup[key]=r
 rows=[dedup[k] for k in sorted(dedup)]
 leading=['station_id','datetime_display','date','original_timestamp']
 trailing=['source_file','source_url','source_unit_parameter','raw_values_json','qc_notes']
 cols=leading+sorted(set().union(*(r.keys() for r in rows))-set(leading+trailing))+trailing if rows else leading+trailing
 with (OUT/f'{prefix}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 manifest={'retrieval_date':'2026-11-04','source_clock_cutoff':'2026-11-04T04:00:00','stations':STATIONS,'timezone_policy':'Naive source display times preserved. Current CAIC page labels MST; historical DST/UTC-offset convention and publication lag unverified. Do not assume UTC.', 'snowfall_policy':'Reported Sno24 retained, never inferred from depth. Instrument/reset protocol and equivalence to official manual resort report remain unverified.', 'source_inventory':'data/resort_daily_research/abasin_caic_summary_20261104.json from public CAIC dashboard','source_definition_url':'https://stations.avalanche.state.co.us/precip.php','downloads':[meta for block,meta in results],'hourly_rows':len(rows),'file':f'{prefix}.csv'}
 (OUT/f'{prefix}_manifest.json').write_text(json.dumps(manifest,indent=2))
 (OUT/f'{prefix}_duplicate_clock_review.json').write_text(json.dumps([{'station_id':k[0],'datetime_display':k[1],'reports':v} for k,v in conflicts.items()],indent=2))
 return rows

def audit():
 """Describe observed coverage and questionable values; never repair labels."""
 import pandas as pd
 df=pd.read_csv(OUT/'hourly.csv',low_memory=False)
 df['timestamp']=pd.to_datetime(df.datetime_display)
 df['season_start']=df.timestamp.dt.year-(df.timestamp.dt.month<9).astype(int)
 df['winter']=df.season_start.astype(str)+'/'+(df.season_start+1).astype(str)
 numeric=[c for c in ['Temp','RH','Spd','Dir','Gst','Sno24','SnoHt','Pcp1','Pcp24','PcpAc'] if c in df]
 coverage=df.groupby(['station_id','winter']).agg(hourly_rows=('timestamp','size'),first_clock=('datetime_display','min'),last_clock=('datetime_display','max'))
 for c in numeric:coverage[c+'_valid']=df.groupby(['station_id','winter'])[c].count()
 coverage.to_csv(OUT/'coverage_by_winter.csv')
 ranges=[]
 for station,part in df.groupby('station_id'):
  for field in numeric:
   x=part[field].dropna()
   ranges.append({'station_id':station,'field':field,'valid':len(x),'missing':len(part)-len(x),'minimum':x.min(),'q01':x.quantile(.01),'median':x.median(),'q99':x.quantile(.99),'maximum':x.max()})
 pd.DataFrame(ranges).to_csv(OUT/'field_ranges.csv',index=False)
 endpoints=[]
 for hour in [4,6]:
  d=df[(df.station_id=='CAVMM')&(df.timestamp.dt.hour==hour)&(df.timestamp.dt.minute==0)].copy()
  d['Sno24_negative_review']=d.Sno24.lt(0)
  d['Sno24_ge24_review']=d.Sno24.ge(24)
  d['suspected_unreset_2020_spring']=d.timestamp.between('2020-03-20','2020-05-31 23:59:59')
  d['suspected_unreset_2023_autumn']=d.timestamp.between('2023-10-01','2023-11-28 23:59:59')
  d['small_positive_at_warm_endpoint_review']=d.Sno24.gt(0)&d.Sno24.le(1)&d.Temp.gt(35)
  d['target_definition']='reported Sno24 at fixed source-clock hour; board reset protocol unverified; not certified 24-hour fresh snowfall'
  d.drop(columns='timestamp').to_csv(OUT/f'daily_{hour:02d}00.csv',index=False)
  d['endpoint_hour']=hour
  d['month']=d.timestamp.dt.to_period('M').astype(str)
  endpoints.append(d)
 daily=pd.concat(endpoints,ignore_index=True)
 monthly=daily.groupby(['endpoint_hour','winter','month']).agg(clock_records=('timestamp','size'),Sno24_present=('Sno24','count'),Sno24_sum_raw=('Sno24',lambda x:x.sum(min_count=1)),Sno24_min=('Sno24','min'),Sno24_max=('Sno24','max'),zero_days=('Sno24',lambda x:int(x.eq(0).sum())),small_positive_days=('Sno24',lambda x:int((x.gt(0)&x.le(1)).sum())),negative_review=('Sno24_negative_review','sum'),ge24_review=('Sno24_ge24_review','sum'),warm_small_positive_review=('small_positive_at_warm_endpoint_review','sum'))
 monthly.to_csv(OUT/'target_monthly_qa.csv')
 seasonal=daily.groupby(['endpoint_hour','winter']).agg(clock_records=('timestamp','size'),Sno24_present=('Sno24','count'),Sno24_sum_raw=('Sno24',lambda x:x.sum(min_count=1)),Sno24_min=('Sno24','min'),Sno24_max=('Sno24','max'),zero_days=('Sno24',lambda x:int(x.eq(0).sum())),negative_review=('Sno24_negative_review','sum'),ge24_review=('Sno24_ge24_review','sum'))
 seasonal.to_csv(OUT/'target_winter_qa.csv')
 # Sharp changes towards zero identify candidate board resets, not certified resets.
 events=[]
 for st in ['CAVMM','CAVPQ']:
  p=df[df.station_id==st].sort_values('timestamp').copy()
  p['previous_Sno24']=p.Sno24.shift();p['Sno24_change']=p.Sno24.diff();p['SnoHt_change']=p.SnoHt.diff()
  sel=p.timestamp.diff().eq(pd.Timedelta(hours=1))&p.previous_Sno24.ge(2)&p.Sno24.le(.5)&p.Sno24_change.le(-1.5)&p.SnoHt_change.gt(-2.5)
  q=p.loc[sel,['station_id','datetime_display','winter','previous_Sno24','Sno24','Sno24_change','SnoHt_change','source_file']].copy()
  q['hour']=pd.to_datetime(q.datetime_display).dt.hour
  events.append(q)
 reset=pd.concat(events,ignore_index=True)
 reset.to_csv(OUT/'candidate_reset_events.csv',index=False)
 reset.groupby(['station_id','winter','hour']).size().rename('candidate_events').to_csv(OUT/'candidate_reset_hour_counts.csv')
 artifact_windows=[
  {'station_id':'CAVMM','field':'Sno24','first_clock':'2020-03-20T00:00:00','last_clock':'2020-05-31T23:59:59','status':'suspected accumulation/reset/sensor artifact; raw values retained','evidence':'April06 raw sum314in; many consecutive7-20in values with settlingdepth; lateMay20in with0-3indepth','recommended_use':'quarantine input and target; cause unverified'},
  {'station_id':'CAVMM','field':'Sno24','first_clock':'2023-10-01T00:00:00','last_clock':'2023-11-28T23:59:59','status':'suspected accumulation/reset artifact; broad conservative review interval','evidence':'Nov1-5daily04=12.0,10.7,9.7,8.2,6.6in with settlingdepth; Nov20-28continuouspositive thenNov29reset','recommended_use':'quarantine input and target; exact operator reset chronology unverified'}]
 (OUT/'artifact_windows.json').write_text(json.dumps(artifact_windows,indent=2))
 # Inspect positive consecutive daily labels without assuming all snow days are false.
 runs=[]
 for hour in [4,6]:
  p=daily[daily.endpoint_hour==hour].sort_values('timestamp').copy()
  run_id=(~p.timestamp.diff().eq(pd.Timedelta(days=1))|p.Sno24.ne(p.Sno24.shift())).cumsum()
  for _,g in p.groupby(run_id):
   if len(g)>=4 and pd.notna(g.Sno24.iloc[0]) and g.Sno24.iloc[0]>0:
    runs.append({'endpoint_hour':hour,'first_clock':g.datetime_display.iloc[0],'last_clock':g.datetime_display.iloc[-1],'consecutive_days':len(g),'Sno24':g.Sno24.iloc[0]})
 pd.DataFrame(runs,columns=['endpoint_hour','first_clock','last_clock','consecutive_days','Sno24']).to_csv(OUT/'constant_positive_daily_runs.csv',index=False)
 meta=json.loads((OUT/'hourly_manifest.json').read_text())
 report=[
  'Vail CAIC hourly public archive: acquisition and measurement audit',
  'Acquired 2026-11-04; source clock frozen at 2026-11-04 04:00.',
  'All downloaded HTML is preserved, with request URLs and SHA256 in hourly_manifest.json.',
  f"Parsed {len(df):,} timestamped records across {df.station_id.nunique()} Vail Resort stations; {int(df.timestamp.dt.minute.ne(0).sum()):,} are subhourly wind-station records in October2021. Preserve raw; filter exact top-of-hour or explicitly aggregate for hourly modeling.",
  'Station metadata: '+json.dumps(STATIONS),
  'Data domains: September-May winters 2019/20 through 2025/26; September-November 4 current2026. Earlier January2015-2019 probes were empty for these IDs, not proof that no other archive exists.',
  'Missing fields remain missing. Raw strings retained; provider QC codes are not exposed by the HTML table. No snowfall was calculated from changes in snow depth.',
  'Native English unit query unit=e. Snowfall/depth inches, temperature Fahrenheit; wind units require source metadata confirmation before publication. Values are not converted.',
  'Source times are naive display times. Current page says MST but historical fall-DST duplicated01h has different observations. Conflicting duplicate source hours have numeric fields masked in hourly.csv; raw reports are retained in hourly_duplicate_clock_review.json. Do not assume source times are UTC or fixed-offset.',
  'Target caveat: Sno24 is a reported snow sensor variable, not verified manual resort new snowfall. A fixed04 reading is not established as rolling24h snowfall. Example: Jan18,2020 Sno24=5.0at06,0.4at07 while SnoHt=48.0,47.8; this strongly suggests a board/reset process. Candidate reset events are descriptive and do not establish the full protocol.',
  'Spring2020 clear artifact: daily06 March20 onward persists7-20in as snowdepth settles. April06 raw sum314.0in. May27/28/30 reports~20in when SnowHt~0-3in. Suspected failure to reset/sensor error, cause unverified. Entire March20-May31 segment is marked for review; raw values unchanged.',
  'Autumn2023 second accumulation artifact: daily04 Nov1-5Sno24=12.0,10.7,9.7,8.2,6.6 whileSnoHt=19.6,18.1,16.9,15.6,15.5. Nov20-28 remainspositive thenNov29resets. Conservative October1-November28 review/quarantinewindow is marked, not proof every enclosedrecord is wrong. Operator/resetmetadata is needed to narrow it.',
  '2024 October small positive Sno24 values persist during warm conditions, indicating possible sensor offset/baseline. They are preserved, not silently zeroed. Warm endpoint temperature alone cannot disprove overnight snowfall.',
  'CAVMM January2025 has a suspicious SnowHt=95in plateau; November2023 has repeated114in values interspersedwith14-22in readings. CAVPQ Pcp1 reaches140.1in, a clearprecipitationoutlier. CAVBS gust225mph requires review. Preserve source and prevent these values from silently defining snowfall.',
  '2025/26has lowraw04sum73in andmanyzeros. January2026depth20-37in isalso low; this alone doesnot establishbadSno24, so no automaticquarantine basedonseasonaltotal. Missingstormreadings mayundercount. Independentmanualreports are needed.',
  '04:00 and06:00 endpoint files retain all available target observations and explicit QA booleans. Raw sums in QA tables are diagnostics, not certified seasonal snowfall totals.',
  'Candidate reset events can contain actual physical changes or instrument behavior. Constant positive runs can represent repeated real snowfall; review is not automatic rejection.',
  'A forecast trained on these data must be labeled a provisional forecast of the CAIC CAVMM reported Sno24 proxy. Independent manual snow measurements and operator reset documentation are needed for verified daily resort snowfall.',
  '\nDAILY ENDPOINT QA\n'+seasonal.to_string(),
  '\nHOURLY COVERAGE\n'+coverage.to_string(),
  '\nDOWNLOAD ERRORS\n'+json.dumps([m for m in meta['downloads'] if 'error' in m],indent=2),
  '\nSOURCE LINKS\nhttps://stations.avalanche.state.co.us/precip.php\nhttps://stations.avalanche.state.co.us/tabular.php?st=CAVMM&date=2020-01-31+23&unit=e&area=caic&range=744',
 ]
 (ROOT/'research/vail_data_audit.txt').write_text('\n\n'.join(report))
 print(seasonal.to_string())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['probe','bulk','audit'],default='probe');parser.add_argument('--start-year',type=int,default=2019);args=parser.parse_args()
 RAW.mkdir(parents=True,exist_ok=True)
 if args.mode=='audit':audit();return
 if args.mode=='probe':
  tasks=[(st,y,1) for st in STATIONS for y in [2015,2020,2023,2025]]+[(st,2026,11) for st in STATIONS]
 else:
  tasks=[]
  for st in STATIONS:
   for winter in range(args.start_year,2026):
    tasks += [(st,winter,m) for m in [9,10,11,12]]+[(st,winter+1,m) for m in [1,2,3,4,5]]
   tasks += [(st,2026,m) for m in [9,10,11]]
 with ThreadPoolExecutor(max_workers=3) as pool:
  results=[]
  for rows,meta in pool.map(fetch,tasks):
   results.append((rows,meta))
   print(meta['station_id'],meta['year'],meta['month'],meta['records'],meta.get('error',''),flush=True)
 rows=write_results(results,'probe' if args.mode=='probe' else 'hourly')
 print('TOTAL',len(rows),flush=True)
 if args.mode=='bulk':audit()

if __name__=='__main__':main()
