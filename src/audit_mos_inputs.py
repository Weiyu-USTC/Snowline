"""Read-only checks of final features against independent raw MOS calculations."""
from pathlib import Path
import gzip,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/nwp_archive_probe'

def main():
    raw={};downloads=[]
    for p in sorted((D/'mos_raw').glob('*.csv.gz')):
        meta=json.loads(p.with_suffix('').with_suffix('.json').read_text());body=gzip.decompress(p.read_bytes());assert hashlib.sha256(body).hexdigest()==meta['sha256_uncompressed']
        station=p.name[:4];r=pd.read_csv(p,low_memory=False);r['runtime']=pd.to_datetime(r.runtime,utc=True);r['ftime']=pd.to_datetime(r.ftime,utc=True);raw.setdefault(station,[]).append(r)
        downloads.append({'file':str(p.relative_to(ROOT)),'rows':len(r),'bytes':len(body),'verified_sha256':True})
    raw={s:pd.concat(frames,ignore_index=True).drop_duplicates(['runtime','ftime']) for s,frames in raw.items()}
    results={};rng=np.random.default_rng(731)
    for label,suffix in [('main',''),('minus','_clock_minus'),('plus','_clock_plus')]:
        d=pd.read_csv(D/f'mos_daily_features{suffix}.csv')
        times={c:pd.to_datetime(d[c],utc=True) for c in ['forecast_reference_utc','issue_time_utc','valid_start_utc','valid_end_utc']}
        ref,issue,start,end=[times[c] for c in times]
        assert (ref.dt.hour==0).all();assert (issue-ref).eq(pd.Timedelta(hours=6)).all();assert (start>=issue).all();assert (end-start).eq(pd.Timedelta(hours=24)).all();assert (end<=ref+pd.Timedelta(hours=72)).all()
        assert not d.duplicated(['site','issue_time_utc','target_date','lead_day']).any()
        featurecols=[c for c in d if c.startswith('forecast_') and pd.api.types.is_numeric_dtype(d[c])]
        assert np.isfinite(d[featurecols].to_numpy()).all()
        selection=[]
        for site,g in d.groupby('site'):
            selection.extend(g.sample(n=min(6,len(g)),random_state=31).index)
            selection.extend(g.nlargest(6,'forecast_qpf_ordinal_weighted_mean').index)
        checks=[]
        for i in sorted(set(selection)):
            row=d.loc[i];runtime=pd.Timestamp(row.forecast_reference_utc);rr=raw[row.station];r=rr[rr.runtime.eq(runtime)].sort_values('ftime');assert len(r)
            h=(r.ftime-runtime).dt.total_seconds().to_numpy()/3600
            a=(pd.Timestamp(row.valid_start_utc)-runtime).total_seconds()/3600;b=a+24
            # Independent1-minute integration exactly resolves all linear segments,
            # including both integer and half-hour report clocks in this experiment.
            grid=np.linspace(a,b,24*60+1)
            tmp=(pd.to_numeric(r.tmp).to_numpy(float)-32)*5/9;dew=(pd.to_numeric(r.dpt).to_numpy(float)-32)*5/9
            speed=pd.to_numeric(r.wsp).to_numpy(float)*.514444;direction=np.deg2rad(pd.to_numeric(r.wdr).to_numpy(float))
            variable={'temp':tmp,'dewpoint':dew,'wind_speed':speed,'wind_u':-speed*np.sin(direction),'wind_v':-speed*np.cos(direction)}
            for prefix,v in variable.items():
                unit='c' if prefix in ['temp','dewpoint'] else 'ms';z=np.interp(grid,h,v)
                expected={'mean':np.trapezoid(z,grid)/24,'min':z.min(),'max':z.max(),'change':z[-1]-z[0]}
                for statistic,value in expected.items():assert np.isclose(value,row[f'forecast_{prefix}_{statistic}_{unit}'],atol=1e-9)
            pp=pd.to_numeric(r.p06).to_numpy(float);qq=pd.to_numeric(r.q06).to_numpy(float)
            good=np.isfinite(pp)&np.isfinite(qq)&(pp<=100)&(qq<=6)
            endpoint=h[good];overlap=np.maximum(0,np.minimum(endpoint,b)-np.maximum(endpoint-6,a));assert np.isclose(overlap.sum(),24)
            pp=pp[good]/100;qq=qq[good]
            assert np.isclose(np.sum(overlap*pp)/24,row.forecast_pop_6h_weighted_mean)
            assert np.isclose(pp[overlap>0].max(),row.forecast_pop_6h_max)
            assert np.isclose(np.sum(overlap*qq)/24,row.forecast_qpf_ordinal_weighted_mean)
            assert np.isclose(qq[overlap>0].max(),row.forecast_qpf_ordinal_max)
            for category in range(7):assert np.isclose(np.sum(overlap*(qq==category))/24,row[f'forecast_qpf_category_{category}_fraction'])
            checks.append({'site':row.site,'runtime':row.forecast_reference_utc,'target_date':row.target_date,'lead':int(row.lead_day),'qpf_mean_ordinal':row.forecast_qpf_ordinal_weighted_mean,'passed':True})
        results[label]={'rows':len(d),'all_timestamps_and_feature_finiteness_checked':True,'independent_random_and_largest_QPF_raw_cases':checks,'all_windows_start_after_assumed_8h_publication':bool((start>=ref+pd.Timedelta(hours=8)).all()),'minimum_window_start_lead_hours':float(((start-ref).dt.total_seconds()/3600).min())}
    result={'status':'passed','raw_files':len(downloads),'raw_bytes':sum(x['bytes'] for x in downloads),'source_hashes':downloads,'feature_checks':results,'publication_note':'For mainclock, an8hbufferstillprecedesALLtargetwindows; changingassumedpublication6h→8hdoesnotchangeinputforecastsorverificationwindows,onlyreportedlead.'}
    (D/'mos_independent_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':'passed','raw_files':len(downloads),'raw_bytes':result['raw_bytes'],'independent_cases':{k:len(v['independent_random_and_largest_QPF_raw_cases']) for k,v in results.items()},'all_main_windows_after_8h':results['main']['all_windows_start_after_assumed_8h_publication']}))

if __name__=='__main__':main()
