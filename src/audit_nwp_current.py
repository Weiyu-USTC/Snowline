"""Independent source-target and saved-current-model replay audit."""
from pathlib import Path
import calendar,json,sys
import numpy as np
import pandas as pd
import joblib
import nwp_benchmark as nwp
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/nwp-weather'


def independent_target(site):
    if site=='alta':
        d=pd.read_csv(ROOT/'data/east_alta_research/alta_hive/alta_collins_daily_paired.csv')
        assert d.complete_24h_pair.all() and (d.observed_12h_periods==2).all()
        return pd.Series(d.snowfall_24h_inches.to_numpy(),index=pd.to_datetime(d.reporting_endpoint_lst).dt.normalize()),'Alta-Collins manual paired24h'
    if site=='vail':
        d=pd.read_csv(ROOT/'data/vail_hourly/hourly.csv',usecols=['station_id','datetime_display','Sno24']);d=d[d.station_id=='CAVMM'].copy();d['time']=pd.to_datetime(d.datetime_display)
        d=d[d.time.dt.hour.eq(4)&d.time.dt.minute.eq(0)].set_index('time');s=d.Sno24.where(d.Sno24>=0)
        s.loc['2020-03-20':'2020-05-31']=np.nan;s.loc['2023-10-01':'2023-11-28']=np.nan;s.index=s.index.normalize()
        assert s.index.is_unique
        return s,'CAVMM04:00 source Sno24, documented QC masks; not verified fresh snow'
    key='mount_mansfield_USC00435416' if site=='mansfield' else 'jay_peak_USC00434189';values={}
    for line in (ROOT/f'data/east_alta_research/{key}.dly').read_text().splitlines():
        if line[17:21]!='SNOW':continue
        year,month=int(line[11:15]),int(line[15:17])
        for day in range(1,calendar.monthrange(year,month)[1]+1):
            cell=line[21+(day-1)*8:21+day*8];v=int(cell[:5]);quality=cell[6]
            values[pd.Timestamp(year,month,day)]=v/25.4 if v!=-9999 and quality==' ' else np.nan
    return pd.Series(values).sort_index(),f'Original NOAA{key} SNOW, blank quality flags, presumed zeros retained'


def main():
    results={}
    for site in ['alta','vail','mansfield','jay']:
        folder=OUT/'mos'/site;train=pd.read_csv(folder/'matched_input.csv');future=pd.read_csv(folder/'current_input.csv');pred=pd.read_csv(folder/'current_predictions.csv');bundle=joblib.load(folder/'final_models.joblib')
        train['target_date']=pd.to_datetime(train.target_date);future['target_date']=pd.to_datetime(future.target_date)
        assert len(future)==2 and set(future.target_date)=={pd.Timestamp('2026-11-05'),pd.Timestamp('2026-11-06')}
        assert future.target_inches.isna().all();assert train.target_inches.notna().all()
        assert train.target_date.max()<pd.Timestamp('2026-11-04');assert (train.winter<2026).all()
        assert (pd.to_datetime(train.forecast_reference_utc,utc=True)<pd.Timestamp('2026-11-04',tz='UTC')).all()
        source,definition=independent_target(site);expected=source.reindex(train.target_date).to_numpy(float)
        assert np.allclose(expected,train.target_inches,rtol=1e-10,atol=1e-10,equal_nan=False),site
        history=bundle['climatology_history'];assert set(history.winter).issubset(set(train.winter));assert pd.to_datetime(history.target_date).max()<pd.Timestamp('2026-11-04')
        expected_history=source.reindex(pd.to_datetime(history.target_date)).to_numpy(float);assert np.allclose(expected_history,history.target_inches,equal_nan=True,rtol=1e-10,atol=1e-10)
        replay={nwp.NAMES[0]:nwp.climatology(history,future)}
        for name,data in bundle['models'].items():
            model,columns=data['model'],data['columns'];assert np.array_equal(model.y,train.target_inches.to_numpy())
            replay[name]=model.predict_distribution(nwp.features(future,columns))
        replay_errors={}
        for name,distribution in replay.items():
            saved=pred[pred.model==name].set_index('row_id').reindex(future.row_id)
            for key in ['mean','p_powder','p_snow','q10','q90']:
                column='predicted_inches' if key=='mean' else key
                assert np.allclose(distribution[key],saved[column],rtol=1e-11,atol=1e-11), (site,name,key)
            replay_errors[name]=max(float(np.max(abs(distribution[key]-saved['predicted_inches' if key=='mean' else key].to_numpy()))) for key in ['mean','p_powder','p_snow','q10','q90'])
        plan=json.loads((folder/'benchmark_plan.json').read_text());counts={}
        for split in ['crossvalidation','chronological']:
            p=pd.read_csv(folder/f'{split}_predictions.csv');sets=[set(g.row_id) for _,g in p.groupby('model')];assert all(s==sets[0] for s in sets)
            assert not p.duplicated(['row_id','model']).any()
            actual=p.target_inches.to_numpy(float);truth=source.reindex(pd.to_datetime(p.target_date)).to_numpy(float);assert np.allclose(actual,truth,rtol=1e-10,atol=1e-10)
            assert set(p.winter).issubset(set(train.winter))
            for spec in [s for s in plan['splits'] if s['split']==split]:
                assert not set(spec['train_winters'])&set(spec['test_winters'])
                part=p[p['split']==f'{split}_{spec["fold"]}'];assert part.row_id.nunique()==spec['test_rows'];assert set(part.winter)==set(spec['test_winters'])
            if split=='crossvalidation':assert sets[0]==set(train.row_id)
            counts[split]={'pairs_per_model':len(sets[0]),'models':p.model.nunique(),'winters':p.winter.nunique()}
        results[site]={'status':'passed','target_source':definition,'historical_join_rows':len(train),'training_winters':sorted(train.winter.unique().tolist()),'last_training_target_date':str(train.target_date.max().date()),'current_rows_per_model':2,'all_current_targets_missing':True,'future_2026_winter_excluded':True,'stored_model_training_outcomes_exactly_match_join':True,'baseline_donors_match_original_sources':True,'replay_max_abs_error':replay_errors,'evaluation_cohorts':counts}
    (OUT/'current_replay_audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()
