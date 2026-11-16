"""Restore final event models and verify published curves and matched cohorts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch

import vail_hourly_model as old
import vail_storm_neural as neural
from vail_baselines import tabular_summary
from run_vail_storm_models import initial_history, ages, SEED, NAMES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/vail-storm-models'
PREVIOUS=ROOT/'outputs/vail-hourly'


def main():
    required=['forecast_manifest.json','forecast_daily.csv','forecast_paths.npz',
              'statistical_event_models.joblib','neural_event_model.pt',
              'crossvalidation_daily_predictions.csv','crossvalidation_onset_predictions.csv']
    missing=[name for name in required if not (OUT/name).exists()]
    if missing:raise SystemExit('Benchmark not complete; missing: '+', '.join(missing))
    old.seed_everything(SEED)
    manifest=json.loads((OUT/'forecast_manifest.json').read_text())
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    digest=hashlib.sha256((ROOT/'data/vail_hourly/hourly.csv').read_bytes()).hexdigest()
    assert digest==manifest['source_sha256']==plan['source_sha256']
    # load_data writes derived coverage files; keep validation side effects out
    # of both frozen benchmark directories.
    prior_out=old.OUT
    with tempfile.TemporaryDirectory() as temporary:
        old.OUT=Path(temporary)
        try:_,frame,target,years=old.load_data()
        finally:old.OUT=prior_out
    origin=pd.Timestamp(manifest['issue_source_clock'])
    xx=frame.reindex(pd.date_range(end=origin,periods=168,freq='h')).to_numpy('float32')[None]
    ff=old.future_features(origin)[None]
    initial=initial_history(target,[origin]);summary=tabular_summary(xx)
    np.testing.assert_array_equal(initial,np.asarray(manifest['initial'],dtype='float32'))
    dates=pd.date_range(origin+pd.Timedelta(days=1),periods=60)
    keep=np.flatnonzero(dates<=pd.Timestamp('2026-12-31 04:00'))
    published=pd.read_csv(OUT/'forecast_daily.csv')
    saved_paths=np.load(OUT/'forecast_paths.npz')
    restored={};replication={}
    statistical=joblib.load(OUT/'statistical_event_models.joblib')
    for name,model in statistical.items():
        restored[name]=model.forecast(summary,ff[:,:,:4],initial[:,0],ages(initial),n_paths=manifest['paths_per_model'],seed=SEED)
    checkpoint=torch.load(OUT/'neural_event_model.pt',weights_only=True)
    model,scaler=neural.restore_neural(checkpoint)
    restored['Neural events']=neural.forecast_neural(model,scaler,xx,ff,initial)
    restored['Neural events']['paths']=neural.simulate_neural(model,scaler,xx,ff,initial,
                                                           n_paths=manifest['paths_per_model'],seed=SEED)
    for name,prediction in restored.items():
        rows=published[published.model==name].sort_values('lead')
        assert len(rows)==57
        assert list(rows.date)==list(dates[keep].strftime('%Y-%m-%d'))
        differences={}
        for output,column in [('mean','mean_inches'),('p_active','p_active'),('p_onset','p_onset')]:
            predicted=prediction[output][0,keep]
            difference=float(np.max(np.abs(predicted-rows[column].to_numpy())))
            assert difference<2e-6,(name,column,difference)
            differences[column]=difference
        paths=np.take(prediction['paths'][0],keep,axis=1)
        assert paths.shape==(1024,57)
        assert np.isfinite(paths).all() and (paths>=0).all()
        path_difference=float(np.max(np.abs(paths-saved_paths[name])))
        assert path_difference<2e-6,(name,path_difference)
        q10,q90=np.quantile(paths,[.1,.9],axis=0)
        np.testing.assert_allclose(q10,rows.q10_inches,atol=2e-6,rtol=0)
        np.testing.assert_allclose(q90,rows.q90_inches,atol=2e-6,rtol=0)
        assert ((rows.p_active>=0)&(rows.p_active<=1)).all()
        assert ((rows.p_onset>=0)&(rows.p_onset<=rows.p_active+1e-7)).all()
        differences['paths_max_difference']=path_difference
        replication[name]=differences
    cumulative=restored['Neural events']['p_onset_within']
    assert np.all(np.diff(cumulative,axis=1)>=-1e-7)
    assert (cumulative>=-1e-7).all() and (cumulative<=1+1e-7).all()

    prior=pd.read_csv(PREVIOUS/'crossvalidation_predictions.csv')
    previous=prior[prior.model=='Hourly TCN']
    key=['winter','origin','lead']
    expected=previous.set_index(key).sort_index()
    rows=pd.read_csv(OUT/'crossvalidation_daily_predictions.csv')
    assert set(rows.model)==set(NAMES)
    assert len(expected)==19632 and expected.date.nunique()==969
    for name,part in rows.groupby('model'):
        actual=part.set_index(key).sort_index()
        assert actual.index.equals(expected.index),name
        np.testing.assert_allclose(actual.actual_inches,expected.actual_inches,atol=1e-7,rtol=0)
        assert np.isfinite(actual[['predicted_inches','p_active','q10_inches','q90_inches','crps']]).all().all()
        assert (actual.q10_inches<=actual.q90_inches).all()
        assert (actual.crps>=-1e-7).all()
        if name=='Previous hourly TCN':
            np.testing.assert_allclose(actual.predicted_inches,expected.predicted_inches,atol=2e-5,rtol=0)
    assert not pd.to_datetime(rows.date).between('2020-03-20','2020-05-31').any()
    assert not pd.to_datetime(rows.date).between('2023-10-01','2023-11-28').any()
    onset=pd.read_csv(OUT/'crossvalidation_onset_predictions.csv')
    onset_key=['winter','origin','window_days']
    onset_sets=[set(map(tuple,p[onset_key].to_numpy())) for _,p in onset.groupby('model')]
    assert all(s==onset_sets[0] for s in onset_sets)
    assert onset.actual_onset.isin([0,1]).all()
    assert onset.p_onset.between(-1e-7,1+1e-7).all()

    # Functional probes of every saved held-out checkpoint. Restoring the first
    # and last issue in each split exercises both forecast extremes without
    # repeating fitting or changing the evaluation results.
    arrays=np.load(PREVIOUS/'training_arrays.npz')
    meta=pd.read_csv(PREVIOUS/'origins.csv')
    outer_replication={}
    for tag in ['fold0','fold1','fold2','chronological']:
        heldout=pd.read_csv(OUT/f'{tag}_daily_predictions.csv')
        issues=sorted(heldout.origin.unique())
        ix=np.flatnonzero(meta.origin.isin([issues[0],issues[-1]]))
        bx,bf=arrays['x'][ix],arrays['f'][ix]
        bi=initial_history(target,meta.iloc[ix].origin)
        bs=tabular_summary(bx)
        checkpoint=torch.load(OUT/f'{tag}_neural_checkpoint.pt',weights_only=True)
        assert set(checkpoint['training_metadata']['training_winters']).isdisjoint(set(heldout.winter))
        model,scaler=neural.restore_neural(checkpoint)
        predictions={'Neural events':neural.forecast_neural(model,scaler,bx,bf,bi)}
        for name,model in joblib.load(OUT/f'{tag}_statistical_models.joblib').items():
            predictions[name]=model.forecast(bs,bf[:,:,:4],bi[:,0],ages(bi),n_paths=1,seed=SEED)
        errors={}
        for name,prediction in predictions.items():
            reference=heldout[heldout.model==name].set_index(['origin','lead'])
            differences=[]
            for j,index in enumerate(ix):
                for lead in np.flatnonzero(np.isfinite(arrays['y'][index])):
                    row=reference.loc[(meta.iloc[index].origin,int(lead)+1)]
                    for key,column in [('mean','predicted_inches'),('p_active','p_active'),('p_onset','p_onset')]:
                        differences.append(abs(prediction[key][j,lead]-row[column]))
            errors[name]=float(max(differences))
            assert errors[name]<2e-6,(tag,name,errors[name])
        outer_replication[tag]=errors
    result={'source_hash_matches':True,'forecast_endpoints':57,
            'forecast_models':list(restored),'restoration_errors':replication,
            'heldout_checkpoint_restoration_errors':outer_replication,
            'matched_pairs_per_model':len(expected),'unique_target_dates':969,
            'benchmark_models':len(NAMES),'onset_cohort_matches':True,
            'previous_point_forecasts_reproduced':True,'quarantine_targets_excluded':True,
            'initial_history_reproduced':True,'no_monthly_fresh_snow_claim':True,
            'target':'Daily CAVMM reported04:00 sensor endpoint, not verified fresh snowfall'}
    (OUT/'validation_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
