"""Matched whole-winter benchmark of observed-state Vail event models.

Hourly weather inputs; daily >=1-inch CAVMM reported-endpoint states. This does
not turn CAIC board readings into verified hourly fresh snowfall or storm truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.model_selection import KFold
import torch

import vail_hourly_model as old
from vail_baselines import tabular_summary
from vail_storm_statistical import EventModel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/vail-storm-models'
PREVIOUS=ROOT/'outputs/vail-hourly'
SEED=20261105
HORIZON=60
STAT_CONFIGS=[dict(regularization=.05,weather_decay_days=3.),
              dict(regularization=.5,weather_decay_days=7.)]
NEURAL_CONFIGS=[dict(width=16,dropout=.25,lr=.001,weight_decay=.02,weather_decay_days=3.),
                dict(width=24,dropout=.35,lr=.0006,weight_decay=.04,weather_decay_days=7.)]
NAMES=['Climatology','Previous hourly TCN','Seasonal Markov','Seasonal duration','Weather duration','Neural events']


def initial_history(target, origins):
    """Only observed history, with unknown age until a documented state change."""
    result=[]
    for origin in pd.to_datetime(origins):
        winter=origin.year if origin.month>=9 else origin.year-1
        dates=pd.date_range(f'{winter}-09-01 04:00',origin)
        values=target.reindex(dates).to_numpy(float)
        previous=np.nan;age=0;age_known=False
        for value in values:
            if not np.isfinite(value):
                previous=np.nan;age=0;age_known=False;continue
            state=int(value>=1)
            if not np.isfinite(previous): age=0;age_known=False
            elif state!=int(previous>=1): age=1;age_known=True
            elif age_known: age=min(age+1,14)
            previous=value
        result.append([previous,age,float(age_known),float(np.isfinite(previous))])
    return np.asarray(result,dtype='float32')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    previous_out=old.OUT;old.OUT=OUT
    try:_,frame,target,years=old.load_data()
    finally:old.OUT=previous_out
    meta=pd.read_csv(PREVIOUS/'origins.csv')
    data=np.load(PREVIOUS/'training_arrays.npz')
    x,f,y=data['x'],data['f'],data['y']
    assert len(meta)==len(x)==342
    assert hashlib.sha256((ROOT/'data/vail_hourly/hourly.csv').read_bytes()).hexdigest()==json.loads((PREVIOUS/'forecast_manifest.json').read_text())['source_sha256']
    initial=initial_history(target,meta.origin)
    summaries=tabular_summary(x)
    # One-step training includes all eligible dates in training winters only.
    one_meta=[];one_x=[];one_y=[]
    for year in years:
        for origin in pd.date_range(f'{year}-10-08 04:00',f'{year+1}-04-29 04:00'):
            yy=target.reindex([origin,origin+pd.Timedelta(days=1)]).to_numpy(float)
            if not np.isfinite(yy).all():continue
            xx=frame.reindex(pd.date_range(end=origin,periods=168,freq='h')).to_numpy('float32')
            if np.isfinite(xx[:,0]).sum()<84:continue
            one_meta.append(dict(winter=year,origin=str(origin)));one_x.append(xx);one_y.append(yy[1])
    one_meta=pd.DataFrame(one_meta)
    one_summary=tabular_summary(np.array(one_x))
    one_initial=initial_history(target,one_meta.origin)
    one_calendar=old.calendar_features(pd.to_datetime(one_meta.origin)+pd.Timedelta(days=1))
    one_y=np.asarray(one_y)
    meta.to_csv(OUT/'benchmark_origins.csv',index=False)
    one_meta.to_csv(OUT/'one_step_training_inventory.csv',index=False)
    plan=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),seed=SEED,
              target='Daily04:00 CAVMM Sno24 sensor proxy, NOT verified fresh snowfall',
              state='reported endpoint>=1 inch; below-threshold emissions retain zeros AND positive values below1',
              source_sha256=hashlib.sha256((ROOT/'data/vail_hourly/hourly.csv').read_bytes()).hexdigest(),
              years=years,origins=len(meta),valid_forecast_pairs=int(np.isfinite(y).sum()),
              one_step_training_examples=len(one_meta),missing_issue_readings=int((initial[:,3]==0).sum()),
              unknown_issue_ages=int((initial[:,2]==0).sum()),age_cap_days=14,
              validation='Same three shuffled whole-winter outer folds as previous hourly experiment (seed20261104). One inner validation winter for choices. Additional latest-two-winter chronological check.',
              statistical_selection='Two weather-hazard C/decay combinations selected by mean Brier score for>=1 readings at leads1–7 on inner validation winter, freely rolled out',
              neural_selection='Two configurations and early stopping selected by inner masked conditional teacher-forced likelihood; all reported forecast errors use free-running exact state recursion; missing-state intervals are not fully marginalized during fitting',
              stat_configs=STAT_CONFIGS,neural_configs=NEURAL_CONFIGS,
              means='Exact forward probability recursion over two states and15 age values; no Monte Carlo noise in point means',
              trajectories='256 IID paths per heldout issue for fair CRPS/quantiles;1024 paths per final issue. PreviousTCN CRPS analytic; climatology CRPS exact for its empirical distribution.',
              scope='Observed-state daily Markov/semi-Markov models with state-age emissions; not a fitted continuous-time Hawkes or normalized single-total pulse model',
              no_future_weather='Frozen issue-time hourly input. Weather effect attenuates with lead; no future observed atmospheric inputs.',
              current_forecast='FrozenNovember4,2026 04:00 source-clock issue, reporting endpointsNov5–Dec31')
    (OUT/'frozen_plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(stage='prepared',origins=len(meta),one_step=len(one_meta),missing_initial=int((initial[:,3]==0).sum()))),flush=True)
    return dict(frame=frame,target=target,years=years,meta=meta,x=x,f=f,y=y,initial=initial,summary=summaries,
                one_meta=one_meta,one_summary=one_summary,one_initial=one_initial,one_calendar=one_calendar,one_y=one_y)


def ages(initial):
    return np.where(initial[:,2]>0,initial[:,1],np.nan)


def fit_stat(d,years,kind='semi_markov',use_weather=True,config=None):
    ix=np.flatnonzero(d['one_meta'].winter.isin(years))
    config=config or dict(regularization=.1,weather_decay_days=3.)
    model=EventModel(kind=kind,use_weather=use_weather,seed=SEED,**config)
    model.fit(d['one_summary'][ix],d['one_calendar'][ix],d['one_initial'][ix,0],ages(d['one_initial'][ix]),d['one_y'][ix])
    return model


def predict_stat(model,d,ix,n_paths=256,seed=SEED):
    return model.forecast(d['summary'][ix],d['f'][ix,:,:4],d['initial'][ix,0],ages(d['initial'][ix]),n_paths=n_paths,seed=seed)


def train_years(meta,ix):return sorted(map(int,meta.iloc[ix].winter.unique()))


def select_stat(d,fit_years,validation_years,tag):
    vi=np.flatnonzero(d['meta'].winter.isin(validation_years))
    trials=[]
    for config in STAT_CONFIGS:
        model=fit_stat(d,fit_years,config=config)
        pred=predict_stat(model,d,vi,n_paths=1)
        mask=np.isfinite(d['y'][vi,:7]);truth=(d['y'][vi,:7]>=1)
        score=float(np.mean((pred['p_active'][:,:7][mask]-truth[mask])**2))
        trials.append(dict(config=config,validation_brier_first7=score))
    chosen=min(trials,key=lambda t:t['validation_brier_first7'])['config']
    (OUT/f'{tag}_statistical_selection.json').write_text(json.dumps(dict(training_winters=fit_years,validation_winters=validation_years,trials=trials,chosen=chosen),indent=2))
    return chosen


def crps_sample(paths,actual,fair=False):
    # Fair correction estimates the parent distribution CRPS from IID paths.
    p=paths.shape[1]
    sorted_paths=np.sort(paths.astype('float64'),axis=1)
    weights=(2*np.arange(1,p+1)-p-1)[None,:,None]
    denominator=p*(p-1) if fair else p*p
    if denominator==0:raise ValueError('Fair CRPS needs at least two samples')
    return np.mean(np.abs(paths-actual[:,None,:]),axis=1)-np.sum(weights*sorted_paths,axis=1)/denominator


def hurdle_lognormal_crps(params,actual):
    p,mu,sigma=params[...,0],params[...,1],params[...,2]
    y=np.asarray(actual,dtype=float)
    z=np.where(y>0,(np.log(np.maximum(y,1e-300))-mu)/sigma,-np.inf)
    m=np.exp(mu+.5*sigma*sigma)
    return y*(1-2*p+2*p*ndtr(z))+2*p*m*(p*ndtr(-sigma/np.sqrt(2))-ndtr(z-sigma))


def window_onset_labels(y,initial,window):
    series=np.column_stack([initial[:,0],y[:,:window]])
    known=np.isfinite(series).all(axis=1)
    state=series>=1
    event=np.any(~state[:,:-1]&state[:,1:],axis=1)
    return event,known


def rows_for_prediction(d,ix,name,pred,tag):
    y=d['y'][ix];mask=np.isfinite(y)
    paths=pred.get('paths')
    if paths is not None and paths.shape[1]:
        q10,q90=np.quantile(paths,[.1,.9],axis=1)
        crps=crps_sample(paths,np.nan_to_num(y,nan=0.),fair=True)
    else:q10,q90,crps=pred['q10'],pred['q90'],pred['crps']
    records=[]
    for j,index in enumerate(ix):
        for h in np.flatnonzero(mask[j]):
            records.append(dict(model=name,split=tag,winter=int(d['meta'].iloc[index].winter),origin=d['meta'].iloc[index].origin,
                                date=str((pd.Timestamp(d['meta'].iloc[index].origin)+pd.Timedelta(days=int(h)+1)).date()),lead=int(h)+1,
                                actual_inches=float(y[j,h]),predicted_inches=float(pred['mean'][j,h]),
                                p_active=float(pred['p_active'][j,h]),q10_inches=float(q10[j,h]),q90_inches=float(q90[j,h]),
                                crps=float(crps[j,h]),p_onset=float(pred.get('p_onset',np.full_like(y,np.nan))[j,h])))
    onset_rows=[]
    for window in [1,3,7]:
        truth,known=window_onset_labels(y,d['initial'][ix],window)
        if 'p_onset_within' in pred:
            value=pred['p_onset_within']
            probability=value[:,window-1] if not isinstance(value,dict) else value[window]
        elif 'onset_paths' in pred and pred['onset_paths'].shape[1]:
            probability=pred['onset_paths'][:,:,:window].any(axis=2).mean(axis=1)
        else:continue
        for j in np.flatnonzero(known):
            index=ix[j]
            onset_rows.append(dict(model=name,split=tag,winter=int(d['meta'].iloc[index].winter),origin=d['meta'].iloc[index].origin,
                                   window_days=window,actual_onset=int(truth[j]),p_onset=float(np.clip(probability[j],0.,1.))))
    return pd.DataFrame(records),pd.DataFrame(onset_rows)


def climate_prediction(target,years,origins):
    values=pd.concat([target.loc[f'{yr}-10-01':f'{yr+1}-04-30'] for yr in years]).dropna()
    doy=np.array([pd.Timestamp(2000,d.month,d.day).dayofyear for d in values.index])
    lookup={}
    for day in range(1,367):
        delta=np.abs(doy-day);sample=values.to_numpy()[np.minimum(delta,366-delta)<=15]
        if len(sample)==0:sample=values.to_numpy()
        sorted_sample=np.sort(sample);n=len(sample)
        half_pair=float(np.sum((2*np.arange(1,n+1)-n-1)*sorted_sample)/(n*n))
        lookup[day]=(sample,half_pair)
    shape=(len(origins),60)
    pred={key:np.empty(shape) for key in ['mean','p_active','q10','q90','crps']}
    for j,origin in enumerate(pd.to_datetime(origins)):
        for h,date in enumerate(pd.date_range(origin+pd.Timedelta(days=1),periods=60)):
            sample,half_pair=lookup[pd.Timestamp(2000,date.month,date.day).dayofyear]
            pred['mean'][j,h]=sample.mean();pred['p_active'][j,h]=(sample>=1).mean()
            pred['q10'][j,h],pred['q90'][j,h]=np.quantile(sample,[.1,.9])
            actual=target.get(date,np.nan)
            pred['crps'][j,h]=np.mean(abs(sample-actual))-half_pair if np.isfinite(actual) else np.nan
    return pred


def previous_prediction(d,train_idx,test_idx,tag):
    record=json.loads((PREVIOUS/f'{tag}_training.json').read_text())
    model,scaler,_,_,_=old.train_model(d['x'],d['f'],d['y'],train_idx,None,record['selected_config'],epochs=record['selected_epochs'],seed=old.SEED)
    params=old.predict(model,scaler,d['x'][test_idx],d['f'][test_idx])
    mean=old.distribution_mean(params)
    # Regenerate distribution from frozen selected model only, never retune it.
    reference=pd.read_csv(PREVIOUS/f'{tag}_predictions.csv')
    reference=reference[reference.model=='Hourly TCN']
    ref=reference.set_index(['origin','lead']).predicted_inches
    difference=[]
    for j,index in enumerate(test_idx):
        for h in np.flatnonzero(np.isfinite(d['y'][index])):
            difference.append(abs(mean[j,h]-ref.loc[(d['meta'].iloc[index].origin,int(h)+1)]))
    assert max(difference)<2e-5,max(difference)
    # Analytic marginal CRPS avoids truncation error in a long lognormal tail.
    return dict(mean=mean,p_active=params[...,0]*(1-ndtr(-params[...,1]/params[...,2])),
                q10=old.distribution_quantile(params,.1),q90=old.distribution_quantile(params,.9),
                crps=hurdle_lognormal_crps(params,np.nan_to_num(d['y'][test_idx],nan=0.))),float(max(difference))


def select_neural(d,train_idx,validation_years,tag):
    import vail_storm_neural as neural
    val=train_idx[d['meta'].iloc[train_idx].winter.isin(validation_years).to_numpy()]
    fit=train_idx[~d['meta'].iloc[train_idx].winter.isin(validation_years).to_numpy()]
    trials=[]
    for c,config in enumerate(NEURAL_CONFIGS):
        model,scaler,logs,epoch,score=neural.train_neural(d['x'],d['f'],d['y'],d['initial'],fit,val,config=config,epochs=35,seed=SEED+c)
        trials.append(dict(config=config,best_epoch=epoch,validation_nll=score,logs=logs))
        print(json.dumps(dict(stage=tag,family='neural',config=c,best_epoch=epoch,val_nll=score)),flush=True)
    chosen=min(trials,key=lambda q:q['validation_nll'])
    model,scaler,logs,_,_=neural.train_neural(d['x'],d['f'],d['y'],d['initial'],train_idx,None,config=chosen['config'],epochs=chosen['best_epoch'],seed=SEED)
    metadata=dict(tag=tag,training_winters=train_years(d['meta'],train_idx),validation_winters=validation_years,trials=trials,
                  selected=chosen['config'],epochs=chosen['best_epoch'],refit_logs=logs,parameters=sum(p.numel() for p in model.parameters()))
    (OUT/f'{tag}_neural_training.json').write_text(json.dumps(metadata,indent=2))
    return model,scaler,metadata


def evaluate(d,train_idx,test_idx,tag,replay=False):
    import vail_storm_neural as neural
    years=train_years(d['meta'],train_idx)
    # Reuse previous inner validation membership for matched selection boundaries.
    record=json.loads((PREVIOUS/f'{tag}_training.json').read_text())
    val_years=record['inner_validation_winters'];fit_years=[y for y in years if y not in val_years]
    config=json.loads((OUT/f'{tag}_statistical_selection.json').read_text())['chosen'] if replay else select_stat(d,fit_years,val_years,tag)
    rows=[];onsets=[];models={}
    for name,kind,weather,cfg in [('Seasonal Markov','markov',False,None),('Seasonal duration','semi_markov',False,None),('Weather duration','semi_markov',True,config)]:
        model=fit_stat(d,years,kind,weather,cfg);models[name]=model
        pred=predict_stat(model,d,test_idx)
        r,o=rows_for_prediction(d,test_idx,name,pred,tag);rows.append(r);onsets.append(o)
        print(json.dumps(dict(stage=tag,model=name,rmse=float(np.sqrt(np.mean((r.predicted_inches-r.actual_inches)**2))))),flush=True)
    if replay:
        metadata=json.loads((OUT/f'{tag}_neural_training.json').read_text())
        model,scaler,_,_,_=neural.train_neural(d['x'],d['f'],d['y'],d['initial'],train_idx,None,config=metadata['selected'],epochs=metadata['epochs'],seed=SEED)
    else:model,scaler,metadata=select_neural(d,train_idx,val_years,tag)
    payload=neural.checkpoint_payload(model,scaler,metadata['selected']);payload['training_metadata']=metadata
    torch.save(payload,OUT/f'{tag}_neural_checkpoint.pt')
    joblib.dump(models,OUT/f'{tag}_statistical_models.joblib')
    pred=neural.forecast_neural(model,scaler,d['x'][test_idx],d['f'][test_idx],d['initial'][test_idx])
    pred['paths']=neural.simulate_neural(model,scaler,d['x'][test_idx],d['f'][test_idx],d['initial'][test_idx],n_paths=256,seed=SEED)
    r,o=rows_for_prediction(d,test_idx,'Neural events',pred,tag);rows.append(r);onsets.append(o)
    pred=climate_prediction(d['target'],years,d['meta'].iloc[test_idx].origin)
    r,o=rows_for_prediction(d,test_idx,'Climatology',pred,tag);rows.append(r)
    pred,diff=previous_prediction(d,train_idx,test_idx,tag)
    r,o=rows_for_prediction(d,test_idx,'Previous hourly TCN',pred,tag);rows.append(r)
    print(json.dumps(dict(stage=tag,previous_reproduction_max_difference=diff)),flush=True)
    result=pd.concat(rows,ignore_index=True);events=pd.concat(onsets,ignore_index=True)
    result.to_csv(OUT/f'{tag}_daily_predictions.csv',index=False)
    events.to_csv(OUT/f'{tag}_onset_predictions.csv',index=False)
    return result,events


def summarize(rows,onsets,prefix):
    results=[]
    bands=[(1,60,'1–60'),(1,1,'1'),(1,3,'1–3'),(4,7,'4–7'),(8,14,'8–14'),(15,30,'15–30'),(31,60,'31–60')]
    for name,data in rows.groupby('model'):
        for lo,hi,label in bands:
            r=data[data.lead.between(lo,hi)];error=r.predicted_inches-r.actual_inches
            results.append(dict(model=name,leads=label,pairs=len(r),unique_days=r.date.nunique(),winters=r.winter.nunique(),
                                mse=float(np.mean(error**2)),rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(abs(error))),bias=float(error.mean()),
                                crps=float(r.crps.mean()),active_brier=float(np.mean((r.p_active-(r.actual_inches>=1))**2)),
                                coverage80=float(((r.actual_inches>=r.q10_inches)&(r.actual_inches<=r.q90_inches)).mean()),
                                width80=float((r.q90_inches-r.q10_inches).mean())))
    pd.DataFrame(results).to_csv(OUT/f'{prefix}_metrics.csv',index=False)
    event_scores=[]
    for (name,window),r in onsets.groupby(['model','window_days']):
        event_scores.append(dict(model=name,window_days=int(window),origins=len(r),actual_onsets=int(r.actual_onset.sum()),
                                 brier=float(np.mean((r.p_onset-r.actual_onset)**2)),predicted_rate=float(r.p_onset.mean()),observed_rate=float(r.actual_onset.mean())))
    pd.DataFrame(event_scores).to_csv(OUT/f'{prefix}_onset_metrics.csv',index=False)
    # Matched, cluster-resampled differences; simulation paths are not extra data.
    key=['winter','origin','lead'];reference=rows[rows.model=='Climatology'].set_index(key)
    rng=np.random.default_rng(SEED);comparisons=[]
    for name,r in rows.groupby('model'):
        if name=='Climatology':continue
        r=r.set_index(key).reindex(reference.index)
        assert r.actual_inches.notna().all() and np.allclose(r.actual_inches,reference.actual_inches)
        for score in ['mse','crps','active_brier']:
            if score=='mse':delta=(r.predicted_inches-r.actual_inches)**2-(reference.predicted_inches-reference.actual_inches)**2
            elif score=='crps':delta=r.crps-reference.crps
            else:delta=(r.p_active-(r.actual_inches>=1))**2-(reference.p_active-(reference.actual_inches>=1))**2
            agg=delta.groupby(level='winter').agg(['sum','count'])
            choices=rng.integers(0,len(agg),(4000,len(agg)))
            draws=agg['sum'].to_numpy()[choices].sum(1)/agg['count'].to_numpy()[choices].sum(1)
            comparisons.append(dict(model=name,reference='Climatology',metric=score,difference=float(delta.mean()),ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975)),winters=len(agg)))
    pd.DataFrame(comparisons).to_csv(OUT/f'{prefix}_paired_differences.csv',index=False)


def summarize_saved():
    """Refresh scores from frozen predictions without fitting or selecting models."""
    daily=[];events=[]
    for tag in ['fold0','fold1','fold2','chronological']:
        r=pd.read_csv(OUT/f'{tag}_daily_predictions.csv')
        o=pd.read_csv(OUT/f'{tag}_onset_predictions.csv')
        # Float32 forward-state mass can produce ~3e-8 negative cumulative risk.
        assert o.p_onset.between(-1e-6,1+1e-6).all()
        o['p_onset']=o.p_onset.clip(0,1)
        o.to_csv(OUT/f'{tag}_onset_predictions.csv',index=False)
        if tag=='chronological':summarize(r,o,tag)
        else:daily.append(r);events.append(o)
    r=pd.concat(daily,ignore_index=True);o=pd.concat(events,ignore_index=True)
    r.to_csv(OUT/'crossvalidation_daily_predictions.csv',index=False)
    o.to_csv(OUT/'crossvalidation_onset_predictions.csv',index=False)
    summarize(r,o,'crossvalidation')
    summarize(r[pd.to_datetime(r.date).dt.month.isin([11,12])],
              o[pd.to_datetime(o.origin).dt.month.isin([11,12])],'nov_dec')
    # Additional prespecified lead slices qualify short-range claims. Six winters
    # remain the resampling units; overlapping origins are not independent data.
    rows=[];key=['winter','origin','lead'];rng=np.random.default_rng(SEED)
    for lo,hi,label in [(1,60,'1–60'),(1,1,'1'),(1,3,'1–3'),(4,7,'4–7')]:
        subset=r[r.lead.between(lo,hi)]
        for ref_name in ['Climatology','Previous hourly TCN']:
            reference=subset[subset.model==ref_name].set_index(key)
            for name,g in subset.groupby('model'):
                if name==ref_name:continue
                g=g.set_index(key).reindex(reference.index)
                assert np.allclose(g.actual_inches,reference.actual_inches)
                delta=(g.predicted_inches-g.actual_inches)**2-(reference.predicted_inches-reference.actual_inches)**2
                agg=delta.groupby(level='winter').agg(['sum','count'])
                choices=rng.integers(0,len(agg),(4000,len(agg)))
                draws=agg['sum'].to_numpy()[choices].sum(1)/agg['count'].to_numpy()[choices].sum(1)
                rows.append(dict(model=name,reference=ref_name,leads=label,metric='mse',difference=float(delta.mean()),
                                 ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975)),winters=len(agg)))
    pd.DataFrame(rows).to_csv(OUT/'lead_paired_differences.csv',index=False)


def issue(d):
    import vail_storm_neural as neural
    origin=pd.Timestamp('2026-11-04 04:00');dates=pd.date_range(origin+pd.Timedelta(days=1),periods=60)
    xx=d['frame'].reindex(pd.date_range(end=origin,periods=168,freq='h')).to_numpy('float32')[None]
    ff=old.future_features(origin)[None];initial=initial_history(d['target'],[origin]);summary=tabular_summary(xx)
    val_years=json.loads((PREVIOUS/'final_training.json').read_text())['inner_validation_winters']
    cfg=select_stat(d,[y for y in d['years'] if y not in val_years],val_years,'final')
    models={};predictions={}
    for name,kind,weather,config in [('Seasonal Markov','markov',False,None),('Seasonal duration','semi_markov',False,None),('Weather duration','semi_markov',True,cfg)]:
        model=fit_stat(d,d['years'],kind,weather,config);models[name]=model
        predictions[name]=model.forecast(summary,ff[:,:,:4],initial[:,0],ages(initial),n_paths=1024,seed=SEED)
    joblib.dump(models,OUT/'statistical_event_models.joblib')
    model,scaler,metadata=select_neural(d,np.arange(len(d['meta'])),val_years,'final')
    predictions['Neural events']=neural.forecast_neural(model,scaler,xx,ff,initial)
    predictions['Neural events']['paths']=neural.simulate_neural(model,scaler,xx,ff,initial,n_paths=1024,seed=SEED)
    # Keep full objects for exact restoration under the included source version.
    payload=neural.checkpoint_payload(model,scaler,metadata['selected']);payload['training_metadata']=metadata
    torch.save(payload,OUT/'neural_event_model.pt')
    keep=dates<=pd.Timestamp('2026-12-31 04:00');rows=[];path_arrays={}
    for name,pred in predictions.items():
        q10,q90=np.quantile(pred['paths'][0],[.1,.9],axis=0)
        # np.take avoids advanced-index axis movement ambiguity.
        path_arrays[name]=np.take(pred['paths'][0],np.flatnonzero(keep),axis=1)
        for h in np.flatnonzero(keep):
            rows.append(dict(model=name,date=str(dates[h].date()),lead=int(h)+1,mean_inches=float(pred['mean'][0,h]),
                             q10_inches=float(q10[h]),q90_inches=float(q90[h]),p_active=float(pred['p_active'][0,h]),p_onset=float(pred['p_onset'][0,h])))
    pd.DataFrame(rows).to_csv(OUT/'forecast_daily.csv',index=False)
    np.savez_compressed(OUT/'forecast_paths.npz',**path_arrays)
    climate=climate_prediction(d['target'],d['years'],[origin])
    pd.DataFrame(dict(date=dates[keep].date,mean_inches=climate['mean'][0,keep],q10_inches=climate['q10'][0,keep],q90_inches=climate['q90'][0,keep])).to_csv(OUT/'forecast_climatology.csv',index=False)
    (OUT/'forecast_manifest.json').write_text(json.dumps(dict(issue_source_clock=str(origin),first_reporting_endpoint=str(dates[0]),last_reporting_endpoint=str(dates[keep][-1]),
         target='CAVMM reported04:00 sensor reading; >=1inch activity state; NOT verified fresh snowfall',
         models=list(predictions),source_sha256=hashlib.sha256((ROOT/'data/vail_hourly/hourly.csv').read_bytes()).hexdigest(),
         training_winters=d['years'],neural_parameters=metadata['parameters'],initial=initial.tolist(),
         paths_per_model=1024,interval='Pointwise empirical10–90% from model paths, not calibrated confidence intervals; no monthlyfreshsnowtotal',
         model_scope='Discrete daily state-duration processes with age-dependent amount emissions, hourly observed covariates; not continuous-time Hawkes or a fitted smooth-pulse storm-total model'),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','run','forecast','rescore','summarize'],default='run');args=parser.parse_args()
    if args.mode=='summarize':
        summarize_saved();return
    d=prepare()
    if args.mode=='prepare':return
    if args.mode in ('run','rescore'):
        daily=[];onsets=[]
        for fold,(tr,te) in enumerate(KFold(n_splits=3,shuffle=True,random_state=old.SEED).split(d['years'])):
            train_idx=np.flatnonzero(d['meta'].winter.isin(np.array(d['years'])[tr]));test_idx=np.flatnonzero(d['meta'].winter.isin(np.array(d['years'])[te]))
            r,o=evaluate(d,train_idx,test_idx,f'fold{fold}',replay=args.mode=='rescore');daily.append(r);onsets.append(o)
        rows=pd.concat(daily,ignore_index=True);events=pd.concat(onsets,ignore_index=True)
        rows.to_csv(OUT/'crossvalidation_daily_predictions.csv',index=False);events.to_csv(OUT/'crossvalidation_onset_predictions.csv',index=False)
        summarize(rows,events,'crossvalidation')
        seasonal=rows[pd.to_datetime(rows.date).dt.month.isin([11,12])]
        summarize(seasonal,events[pd.to_datetime(events.origin).dt.month.isin([11,12])],'nov_dec')
        tr=np.flatnonzero(d['meta'].winter.isin(d['years'][:-2]));te=np.flatnonzero(d['meta'].winter.isin(d['years'][-2:]))
        r,o=evaluate(d,tr,te,'chronological',replay=args.mode=='rescore');summarize(r,o,'chronological')
    if args.mode!='rescore':issue(d)
    print('DONE',flush=True)


if __name__=='__main__':main()
