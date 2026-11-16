"""Inventory NOAA predictor snapshots for Alta, Mansfield and Jay experiments.

Regional weather stations remain distinct from snowfall targets. This exports
raw converted values and their source flags without imputing missing weather.
"""
from pathlib import Path
import calendar
from datetime import datetime,timezone
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/multi_resort_daily/current_inputs'
SITES={'alta_noaa':'USC00420072','mount_mansfield':'USC00435416',
       'island_pond':'USC00434120','newport':'USC00435542',
       'enosburg_falls':'USC00432769','enosburg_falls_2':'USC00432773'}
ELEMENTS={'SNOW':('snowfall_inches',25.4),'SNWD':('snow_depth_inches',25.4),
          'PRCP':('precipitation_inches',254.),'TMAX':('tmax_c',10.),
          'TMIN':('tmin_c',10.),'TOBS':('tobs_c',10.)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for old,new,sid in [('alta','alta_noaa','USC00420072'),('mount_mansfield','mount_mansfield','USC00435416')]:
        shutil.copyfile(ROOT/f'data/east_alta_research/{old}_{sid}.dly',OUT/f'{new}_{sid}.dly')
    metadata={}
    for line in (ROOT/'data/ghcnd-stations.txt').read_text().splitlines():
        if line[:11] in SITES.values():
            metadata[line[:11]]=dict(name=line[41:71].strip(),latitude=float(line[12:20]),
                                    longitude=float(line[21:30]),elevation_m=float(line[31:37]))
    inventory=[]
    for key,sid in SITES.items():
        path=OUT/f'{key}_{sid}.dly';rows={};seen=set()
        for line in path.read_text().splitlines():
            assert line[:11]==sid and len(line)>=269
            element=line[17:21]
            if element not in ELEMENTS:continue
            name,divisor=ELEMENTS[element]
            year,month=int(line[11:15]),int(line[15:17])
            for day in range(1,calendar.monthrange(year,month)[1]+1):
                date=pd.Timestamp(year,month,day)
                if date>pd.Timestamp('2026-11-04'):continue
                assert (date,element) not in seen;seen.add((date,element))
                b=line[21+8*(day-1):29+8*(day-1)]
                value=int(b[:5]);m,q,s=[x.strip() for x in b[5:8]]
                r=rows.setdefault(date,{})
                r[name]=value/divisor if value!=-9999 else np.nan
                r[name+'_measurement_flag']=m;r[name+'_quality_flag']=q;r[name+'_source_flag']=s
                r[name+'_quality_ok']=int(value!=-9999 and not q)
                r[name+'_observed_not_presumed_zero']=int(value!=-9999 and not q and m!='P')
        frame=pd.DataFrame.from_dict(rows,orient='index').sort_index()
        fields=[v[0] for v in ELEMENTS.values() if v[0] in frame]
        populated=frame[fields].notna().any(axis=1)
        last=frame.index[populated].max();first=frame.index[populated].min()
        frame=frame.reindex(pd.date_range(first,pd.Timestamp('2026-11-04')))
        frame.index.name='date';frame.to_csv(OUT/f'{key}_daily.csv')
        variables={}
        for element,(name,divisor) in ELEMENTS.items():
            if name not in frame:continue
            valid=frame[name].notna();quality=frame[name+'_quality_flag'].fillna('');m=frame[name+'_measurement_flag'].fillna('')
            good=valid & quality.eq('')
            recent=frame.loc['2026-09-06':'2026-11-04']
            recent_good=recent[name].notna() & recent[name+'_quality_flag'].fillna('').eq('')
            variables[element]=dict(first_nonmissing=str(frame.index[valid].min().date()),last_nonmissing=str(frame.index[valid].max().date()),
                nonmissing=int(valid.sum()),quality_flagged=int((valid&quality.ne('')).sum()),
                presumed_zero=int((valid&m.eq('P')).sum()),
                presumed_zero_fraction_of_quality_ok=float((good&m.eq('P')).sum()/good.sum()) if good.any() else None,
                last_quality_ok=str(frame.index[good].max().date()),
                last60_dates=60,last60_nonmissing=int(recent[name].notna().sum()),last60_quality_ok=int(recent_good.sum()),
                last60_presumed_zero=int((recent[name].notna()&recent[name+'_measurement_flag'].fillna('').eq('P')).sum()))
        inventory.append(dict(key=key,station_id=sid,**metadata[sid],source_url=f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{sid}.dly',
                              raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
                              first_any_observation=str(first.date()),last_any_observation=str(last.date()),variables=variables))
    manifest=dict(retrieved_date='2026-11-04',prepared_utc=datetime.now(timezone.utc).isoformat(),stations=inventory,
        recommendation={'alta':'Alta NOAA COOP as weather predictor for separate Alta-Collins manual-snow target',
                        'mount_mansfield':'Same NOAA station weather remains current but SNOW target stops2017',
                        'jay_peak':'Island Pond as primary regional weather predictor; keep Jay Peak NOAA snowfall target separate'},
        current_boundary='Chosen station temperature observations extend to Nov1,2026. Alta and Island Pond precipitation also reach Nov1, while Mansfield precipitation reaches Oct31 and has substantial gaps. Nov2–4 are unavailable in these snapshots, not zero or observed current weather.',
        values='Raw unit-converted values; retain quality and measurement flags. Use quality_ok to screen Q flags; observed_not_presumed_zero also excludes P.',
        missing='Calendar rows reindexed throughNov4; gaps and future/unreported dates remain NaN. No imputation or weather extrapolation.',
        station_identity='Never substitute Island Pond SNOW for Jay Peak target, or Alta NOAA SNOW for Alta-Collins target.',
        time='Dates are NOAA observation dates. Exact observation windows and historical publication latency require station-specific review before operational claims.')
    (OUT/'current_input_inventory.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({r['key']:{k:(v['first_nonmissing'],v['last_nonmissing'],v['last60_quality_ok']) for k,v in r['variables'].items()} for r in inventory},indent=2))


if __name__=='__main__':main()
