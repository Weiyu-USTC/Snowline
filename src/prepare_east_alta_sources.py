"""Export downloaded NOAA source files with explicit units, flags and coverage.

This is an acquisition inventory, not a claim that every station measures its
associated resort or that the record is complete between its first/last dates.
"""
from pathlib import Path
import calendar
import hashlib
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/east_alta_research'
OUT=DATA/'derived_noaa'
SITES={
    'mount_mansfield':('USC00435416','Stowe / Mount Mansfield','Mountain station; not the resort reporting series'),
    'jay_peak':('USC00434189','Jay Peak','Lower-elevation station, distinct from resort summit reporting'),
    'rutland':('USC00436995','Killington regional covariate','Low-elevation Rutland station; not Killington snowfall'),
    'alta':('USC00420072','Alta NOAA COOP','2655m station; distinct from the 2945m Collins study plot'),
}
ELEMENTS={'SNOW':('snowfall_inches',25.4),'SNWD':('snow_depth_inches',25.4),
          'PRCP':('precipitation_inches',254.),'TMAX':('tmax_c',10.),
          'TMIN':('tmin_c',10.),'TOBS':('tobs_c',10.)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metadata={}
    for line in (ROOT/'data/ghcnd-stations.txt').read_text().splitlines():
        if line[:11] in [v[0] for v in SITES.values()]:
            metadata[line[:11]]=dict(station_name=line[41:71].strip(),latitude=float(line[12:20]),
                                    longitude=float(line[21:30]),elevation_m=float(line[31:37]))
    inventory=[];coverage=[]
    for name,(station,association,qualification) in SITES.items():
        path=DATA/f'{name}_{station}.dly';records={};seen=set()
        for line in path.read_text().splitlines():
            assert len(line)>=269 and line[:11]==station
            element=line[17:21]
            if element not in ELEMENTS:continue
            variable,divisor=ELEMENTS[element];year,month=int(line[11:15]),int(line[15:17])
            for day in range(1,calendar.monthrange(year,month)[1]+1):
                date=pd.Timestamp(year,month,day)
                if date>pd.Timestamp('2026-11-04'):continue
                assert (date,element) not in seen
                seen.add((date,element));block=line[21+8*(day-1):29+8*(day-1)]
                value=int(block[:5]);m,q,s=[x.strip() for x in block[5:8]]
                row=records.setdefault(date,{})
                row[variable]=value/divisor if value!=-9999 else np.nan
                row[variable+'_measurement_flag']=m;row[variable+'_quality_flag']=q;row[variable+'_source_flag']=s
                row[variable+'_usable_excluding_presumed_zero']=int(value!=-9999 and not q and m!='P')
        frame=pd.DataFrame.from_dict(records,orient='index').sort_index()
        value_cols=[v[0] for v in ELEMENTS.values() if v[0] in frame]
        populated=frame[value_cols].notna().any(axis=1)
        start,end=frame.index[populated][[0,-1]]
        frame=frame.reindex(pd.date_range(start,end));frame.index.name='date'
        # Missing dates and values remain missing; flags are never stripped.
        frame.to_csv(OUT/f'{name}_daily.csv')
        site=dict(key=name,station_id=station,association=association,qualification=qualification,
                  **metadata[station],source_url=f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station}.dly',
                  retrieved_date='2026-11-04',raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  calendar_rows=len(frame),first_any_observation=str(start.date()),last_any_observation=str(end.date()),variables={})
        for element,(variable,divisor) in ELEMENTS.items():
            if variable not in frame:continue
            observed=frame[variable].notna();quality=frame[variable+'_quality_flag'].fillna('')
            measurement=frame[variable+'_measurement_flag'].fillna('')
            valid=observed & quality.eq('');strict=valid & measurement.ne('P')
            dates=frame.index[observed]
            info=dict(first_nonmissing=str(dates.min().date()),last_nonmissing=str(dates.max().date()),
                      nonmissing=int(observed.sum()),quality_flagged=int((observed & quality.ne('')).sum()),
                      presumed_zero=int((observed & measurement.eq('P')).sum()),
                      usable_excluding_presumed_zero=int(strict.sum()),unit='degrees Celsius' if variable.endswith('_c') else 'inches')
            site['variables'][element]=info
            coverage.append(dict(station_id=station,station_name=metadata[station]['station_name'],element=element,**info))
        inventory.append(site)
        print(name,site['variables']['SNOW'])
    (DATA/'noaa_source_inventory.json').write_text(json.dumps(dict(stations=inventory,
        missingness='Blank values are unavailable, never dry days. First/last dates do not imply complete coverage.',
        flags='Original GHCN measurement/quality/source flags retained. P means missing presumed zero; T is trace. Usable flag is conservative and excludes P and any quality flag.',
        date_semantics='Source observation dates; observation window/timezone/publication lag require station-specific review.',
        units='SNOW/SNWD source mm divided by25.4; PRCP source0.1mm divided by254; temperatures0.1C divided by10.',
        docs='https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt'),indent=2))
    pd.DataFrame(coverage).to_csv(DATA/'noaa_variable_coverage.csv',index=False)


if __name__=='__main__':main()
