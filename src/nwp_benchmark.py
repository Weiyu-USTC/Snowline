"""Matched, whole-winter snowfall benchmark using archived forecast covariates.

Input is a daily forecast-to-observation join, prepared separately with explicit
issue/valid times. Every numeric forecast_* column is a forecast, not an observed
future value. Categorical QPF codes may be numeric features but are never silently
converted to precipitation amounts. Each site is fitted independently.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import hashlib
import warnings
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

SEED=20261104
CONFIG=dict(n_estimators=128,max_depth=7,min_samples_leaf=15,max_features=.8,bootstrap=True,random_state=SEED,n_jobs=2)
NAMES=['Historical calendar mean','Calendar + ENSO','+ precipitation forecasts','+ temperature forecasts','+ wind forecasts']


def add_lagged_enso(frame):
    """Same conservative revised-index convention as the existing experiment."""
    from deep_dillon import lagged_roni
    d=frame.copy()
    if 'roni' not in d:
        d['roni']=[lagged_roni(t.tz_localize(None)) for t in pd.to_datetime(d.issue_time_utc,utc=True)]
    return d


def read_targets(root,site):
    """Preserve existing target identities and missingness; join on report date."""
    root=Path(root)
    if site=='vail':
        p=root/'outputs/vail-storm-models/daily_target_0400.csv'
        if not p.exists():p=root/'outputs/vail-hourly/daily_target_0400.csv'
        d=pd.read_csv(p);dates=pd.to_datetime(d.iloc[:,0]).dt.normalize();values=pd.to_numeric(d.iloc[:,1],errors='coerce')
        target='CAVMM04:00 reported Sno24 proxy, not verified fresh snowfall'
        presumed=np.zeros(len(d),dtype=bool)
    elif site in ['alta','mansfield','jay']:
        d=pd.read_csv(root/f'outputs/multi-resort/{site}/target_daily.csv')
        dates=pd.to_datetime(d.date);values=pd.to_numeric(d.snowfall_inches,errors='coerce');presumed=d.noaa_presumed_zero.astype(bool).to_numpy()
        target={'alta':'Alta-Collins24h manual new snowfall ending04:00LST','mansfield':'NOAA USC00435416 daily SNOW','jay':'NOAA USC00434189 daily SNOW at560.8m'}[site]
    else:raise ValueError(site)
    return pd.DataFrame({'target_date':dates,'target_inches':values,'presumed_zero':presumed,'target_definition':target})


def validate_join(frame):
    """Reject impossible timing and analysis/reanalysis covariates before fitting."""
    d=frame.copy()
    required=['site','issue_time_utc','valid_start_utc','valid_end_utc','target_date','target_inches','archive_type']
    absent=set(required)-set(d)
    if absent:raise ValueError(f'Missing required columns: {sorted(absent)}')
    for col in ['issue_time_utc','valid_start_utc','valid_end_utc']:
        d[col]=pd.to_datetime(d[col],utc=True,errors='raise')
        if d[col].isna().any():raise ValueError(f'{col} has missing timestamps')
    d['target_date']=pd.to_datetime(d.target_date,errors='raise').dt.normalize()
    if ((d.valid_start_utc<d.issue_time_utc)|(d.valid_end_utc<=d.valid_start_utc)).any():
        raise ValueError('Forecast valid interval begins before issue or has nonpositive duration')
    duration=(d.valid_end_utc-d.valid_start_utc).dt.total_seconds()/3600
    if (~duration.between(22,26)).any():raise ValueError('Daily target intervals must be22–26hours')
    if 'forecast_reference_utc' in d:
        d['forecast_reference_utc']=pd.to_datetime(d.forecast_reference_utc,utc=True)
        if (d.forecast_reference_utc>d.issue_time_utc).any():raise ValueError('Forecast run starts after issue time')
    allowed={'operational_forecast','retrospective_reforecast'}
    if not set(d.archive_type).issubset(allowed):raise ValueError('Unsupported archive_type: observed future/reanalysis cannot be forecast covariates')
    if 'history_end_utc' in d:
        h=pd.to_datetime(d.history_end_utc,utc=True)
        if (h>d.issue_time_utc).any():raise ValueError('Observed history extends after issuance')
    elif any(c.startswith('observed_') for c in d):
        raise ValueError('Observed history covariates require history_end_utc')
    if 'lead_day' not in d:
        d['lead_day']=np.ceil((d.valid_end_utc-d.issue_time_utc).dt.total_seconds()/86400).astype(int)
    if (d.lead_day<1).any():raise ValueError('lead_day must be positive')
    if d.duplicated(['site','issue_time_utc','target_date','lead_day']).any():raise ValueError('Duplicate site/issue/target/lead rows')
    if ((d.target_inches<0)&d.target_inches.notna()).any():raise ValueError('Negative snowfall target')
    # Winter grouping is assigned by target date. All issue/valid intervals remain
    # within the same July–June group in the autumn/winter forecast benchmark.
    d['winter']=d.target_date.dt.year-(d.target_date.dt.month<7).astype(int)
    issue_winter=d.issue_time_utc.dt.year-(d.issue_time_utc.dt.month<7).astype(int)
    if (issue_winter!=d.winter).any():raise ValueError('Forecast crosses winter-group boundary')
    forecast=[c for c in d if c.startswith('forecast_') and c not in ['forecast_reference_utc'] and pd.api.types.is_numeric_dtype(d[c])]
    if not forecast:raise ValueError('No numeric forecast_* predictors')
    d['row_id']=[hashlib.sha256(f'{r.site}|{r.issue_time_utc}|{r.target_date}|{r.lead_day}'.encode()).hexdigest()[:20] for r in d.itertuples()]
    return d


def feature_groups(d):
    all_forecasts=[c for c in d if c.startswith('forecast_') and c!='forecast_reference_utc' and pd.api.types.is_numeric_dtype(d[c])]
    # Interactions enter only once every variable they use has been added.
    # In particular QPF×coldness must not leak temperature into precipitation-only.
    wind=[c for c in all_forecasts if any(t in c.lower() for t in ['wind','gust','u_ms','v_ms'])]
    temp=[c for c in all_forecasts if any(t in c.lower() for t in ['temp','dew','freez','cold','humidity','rh_']) and c not in wind]
    precip=[c for c in all_forecasts if any(t in c.lower() for t in ['precip','qpf','pop','snow','rain']) and c not in temp+wind]
    other=[c for c in all_forecasts if c not in precip+temp+wind]
    # Other forecast fields are included only in the final all-forecast step.
    base=[c for c in d if (c.startswith('observed_') or c in ['roni','enso']) and pd.api.types.is_numeric_dtype(d[c])]
    return {NAMES[1]:base,NAMES[2]:base+precip,NAMES[3]:base+precip+temp,NAMES[4]:base+precip+temp+wind+other}


def features(d,columns):
    phase=2*np.pi*(d.target_date.dt.dayofyear.to_numpy()-1)/365.2425
    calendar=np.column_stack([np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase),d.lead_day.to_numpy()/14.])
    return np.column_stack([calendar,d[columns].to_numpy(float)]) if columns else calendar


class EmpiricalForest:
    """Training-only leaf outcome distributions, with coherent means and tails."""
    def __init__(self,config=None):self.config={**CONFIG,**(config or {})}
    def fit(self,x,y):
        x=np.asarray(x,float);self.y=np.asarray(y,float)
        if not np.isfinite(self.y).all():raise ValueError('Training targets must be observed')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);self.median=np.nanmedian(x,axis=0)
        self.median=np.nan_to_num(self.median);xx=self.transform(x)
        self.forest=RandomForestRegressor(**self.config).fit(xx,self.y)
        train_leaves=self.forest.apply(xx)
        self.donors=[]
        for t in range(train_leaves.shape[1]):
            self.donors.append({int(leaf):np.flatnonzero(train_leaves[:,t]==leaf) for leaf in np.unique(train_leaves[:,t])})
        return self
    def transform(self,x):
        x=np.asarray(x,float);missing=~np.isfinite(x)
        return np.column_stack([np.where(missing,self.median,x),missing.astype(float)])
    def predict_distribution(self,x,y=None):
        leaves=self.forest.apply(self.transform(x));parts=[]
        for start in range(0,len(leaves),256):
            z=leaves[start:start+256];w=np.zeros((len(z),len(self.y)),dtype=float)
            for t,lookup in enumerate(self.donors):
                for leaf in np.unique(z[:,t]):
                    rows=np.flatnonzero(z[:,t]==leaf);donors=lookup[int(leaf)]
                    w[np.ix_(rows,donors)]+=1/(len(self.donors)*len(donors))
            parts.append(weighted_distribution(self.y,w,None if y is None else np.asarray(y)[start:start+len(z)]))
        return {k:np.concatenate([r[k] for r in parts]) for k in parts[0]}


def weighted_distribution(values,weights,actual=None):
    values=np.asarray(values,float);w=np.asarray(weights,float)
    if not np.allclose(w.sum(1),1.,atol=1e-8):raise ValueError('Probability weights do not sum to1')
    order=np.argsort(values);v=values[order];w=w[:,order];cdf=np.cumsum(w,axis=1)
    result={'mean':w@v,'p_powder':w@(v>=5),'p_snow':w@(v>0),'q10':v[(cdf>=.1).argmax(1)],'q90':v[(cdf>=.9).argmax(1)]}
    if actual is not None:
        pair_half=np.sum(w*v*(2*cdf-w-1),axis=1)
        result['crps']=np.sum(w*abs(v[None,:]-np.asarray(actual)[:,None]),axis=1)-pair_half
    return result


def climatology(train_history,test):
    history=train_history.dropna(subset=['target_inches']).drop_duplicates('target_date')
    values=history.target_inches.to_numpy(float)
    if not len(values):raise ValueError('No climatology donors')
    dates=pd.to_datetime(history.target_date)
    doy=np.array([pd.Timestamp(2000,t.month,t.day).dayofyear for t in dates]);day=np.array([pd.Timestamp(2000,t.month,t.day).dayofyear for t in test.target_date])
    distance=abs(day[:,None]-doy[None,:]);mask=np.minimum(distance,366-distance)<=15
    mask[~mask.any(1)]=True
    w=mask/mask.sum(1,keepdims=True)
    return weighted_distribution(values,w,test.target_inches.to_numpy())


def fit_predict(train,test,target_history=None,config=None,model_names=None):
    history=train if target_history is None else target_history
    results={NAMES[0]:climatology(history,test)}
    groups=feature_groups(train)
    fitted={}
    for name,columns in groups.items():
        if model_names is not None and name not in model_names:continue
        key=tuple(columns)
        if key not in fitted:
            estimator=EmpiricalForest(config).fit(features(train,columns),train.target_inches.to_numpy())
            fitted[key]=estimator.predict_distribution(features(test,columns),test.target_inches.to_numpy())
        results[name]=fitted[key]
    if 'raw_model_snowfall_inches' in test and (model_names is None or 'Raw model snow conversion' in model_names):
        raw=test.raw_model_snowfall_inches.to_numpy(float)
        if np.isfinite(raw).all():
            results['Raw model snow conversion']={'mean':raw,'p_powder':(raw>=5).astype(float),'p_snow':(raw>0).astype(float),'q10':raw,'q90':raw,'crps':abs(raw-test.target_inches.to_numpy())}
    return results,groups


def score_predictions(predictions):
    rows=[]
    maxlead=int(predictions.lead_day.max())
    lead_sets=[('all',1,maxlead),('1–3',1,3),('4–7',4,7),('8–14',8,14)]+[(str(h),h,h) for h in sorted(predictions.lead_day.unique())]
    for model,data in predictions.groupby('model'):
        for label,lo,hi in lead_sets:
            r=data[data.lead_day.between(lo,hi)]
            if not len(r):continue
            error=r.predicted_inches-r.target_inches;event=(r.target_inches>=5).to_numpy(float);p=np.clip(r.p_powder.to_numpy(),1e-6,1-1e-6)
            rows.append(dict(model=model,leads=label,rows=len(r),winters=r.winter.nunique(),rmse=float(np.sqrt(np.mean(error**2))),mae=float(abs(error).mean()),bias=float(error.mean()),crps=float(r.crps.mean()),powder_brier=float(np.mean((r.p_powder-event)**2)),powder_logloss=float(np.mean(-event*np.log(p)-(1-event)*np.log(1-p))),powder_observed_rate=float(event.mean()),powder_predicted_rate=float(r.p_powder.mean()),coverage80=float(np.mean((r.target_inches>=r.q10)&(r.target_inches<=r.q90)))))
    return pd.DataFrame(rows)


def paired_comparisons(predictions,draws=4000,reference=NAMES[0],models=None):
    ref=predictions[predictions.model==reference].set_index('row_id');rng=np.random.default_rng(SEED);rows=[]
    for model,data in predictions.groupby('model'):
        if model==reference or (models is not None and model not in models):continue
        g=data.set_index('row_id').reindex(ref.index)
        if not np.allclose(g.target_inches,ref.target_inches):raise ValueError('Models scored on different target cohorts')
        for metric in ['mse','crps','powder_brier']:
            if metric=='mse':delta=(g.predicted_inches-g.target_inches)**2-(ref.predicted_inches-ref.target_inches)**2
            elif metric=='crps':delta=g.crps-ref.crps
            else:delta=(g.p_powder-(g.target_inches>=5))**2-(ref.p_powder-(ref.target_inches>=5))**2
            a=pd.DataFrame({'winter':g.winter,'delta':delta}).groupby('winter').delta.agg(['sum','count']);ix=rng.integers(0,len(a),(draws,len(a)))
            boot=a['sum'].to_numpy()[ix].sum(1)/a['count'].to_numpy()[ix].sum(1)
            rows.append(dict(model=model,reference=reference,metric=metric,difference=float(delta.mean()),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),winters=len(a)))
    return pd.DataFrame(rows)


def evaluate(frame,out,target_history=None,config=None,model_names=None):
    d=validate_join(frame);d=d[d.target_inches.notna()].copy().reset_index(drop=True)
    if d.site.nunique()!=1:raise ValueError('Fit one site at a time')
    years=np.array(sorted(d.winter.unique()));out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if len(years)<3:raise ValueError('At least3winter groups required for held-out evaluation')
    if target_history is not None:
        target_history=target_history.copy();target_history['target_date']=pd.to_datetime(target_history.target_date)
        target_history['winter']=target_history.target_date.dt.year-(target_history.target_date.dt.month<7).astype(int)
    d.to_csv(out/'matched_input.csv',index=False)
    allframes=[];plans=[]
    for name,splits in [('crossvalidation',KFold(n_splits=min(3,len(years)),shuffle=True,random_state=SEED).split(years)),('chronological',[(np.arange(max(1,len(years)-max(1,len(years)//4))),np.arange(max(1,len(years)-max(1,len(years)//4)),len(years)))])]:
        frames=[]
        for fold,(train_year,test_year) in enumerate(splits):
            train=d[d.winter.isin(years[train_year])];test=d[d.winter.isin(years[test_year])]
            if set(train.winter)&set(test.winter):raise AssertionError('Winter leakage')
            history=None if target_history is None else target_history[target_history.winter.isin(years[train_year])]
            outputs,groups=fit_predict(train,test,history,config,model_names)
            for model,p in outputs.items():
                columns=['row_id','site','issue_time_utc','target_date','lead_day','winter','target_inches']+[c for c in ['presumed_zero','valid_start_utc','valid_end_utc','forecast_reference_utc','archive_type','station','model_id','alignment'] if c in test]
                r=test[columns].copy();r['model']=model;r['split']=f'{name}_{fold}'
                for k,v in p.items():r['predicted_inches' if k=='mean' else k]=v
                frames.append(r)
            plans.append(dict(split=name,fold=fold,train_winters=years[train_year].tolist(),test_winters=years[test_year].tolist(),train_rows=len(train),test_rows=len(test),features=groups))
        p=pd.concat(frames,ignore_index=True);p.to_csv(out/f'{name}_predictions.csv',index=False)
        score_predictions(p).to_csv(out/f'{name}_metrics.csv',index=False);paired_comparisons(p).to_csv(out/f'{name}_paired.csv',index=False)
        incremental=[paired_comparisons(p,reference=ref,models=[model]) for ref,model in zip(NAMES[1:-1],NAMES[2:]) if ref in set(p.model) and model in set(p.model)]
        if incremental:pd.concat(incremental,ignore_index=True).to_csv(out/f'{name}_incremental.csv',index=False)
        allframes.append(p)
    (out/'benchmark_plan.json').write_text(json.dumps(dict(site=d.site.iloc[0],config={**CONFIG,**(config or {})},archive_types=sorted(d.archive_type.unique()),forecast_features=feature_groups(d),splits=plans,seed=SEED,selection='Fixed random-forest hyperparameters; no tuning using outer test. Primary contrast: all forecast features vs climatology. Other ablations, metrics and lead strata exploratory, without multiplicity adjustment. Interpret paired intervals with small-winter limitations.',probabilities='Empirical distribution of training outcomes in forest leaves; no calibration on test labels.',baseline_donors='Unique observed target dates from training winters;±15calendar days.',enso='Conservatively lagged revised NOAA RONI, not vintagecorrect historical releases.'),indent=2))
    return allframes


def forecast_current(training_frame,future_frame,out,target_history=None,config=None):
    """Refit each predeclared model; predict current covariates without labels."""
    train=validate_join(add_lagged_enso(training_frame));train=train[train.target_inches.notna()].copy()
    future=future_frame.copy();future['target_inches']=np.nan
    future=validate_join(add_lagged_enso(future));out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if train.site.nunique()!=1 or set(future.site)!=set(train.site):raise ValueError('One matching site per fit')
    history=train if target_history is None else target_history.copy()
    history['target_date']=pd.to_datetime(history.target_date)
    history['winter']=history.target_date.dt.year-(history.target_date.dt.month<7).astype(int)
    history=history[history.winter.isin(train.winter.unique())]
    groups=feature_groups(train);estimators={};predictions={NAMES[0]:climatology(history,future)}
    for name,columns in groups.items():
        if not set(columns).issubset(future):raise ValueError(f'Current features missing {set(columns)-set(future)}')
        model=EmpiricalForest(config).fit(features(train,columns),train.target_inches.to_numpy())
        predictions[name]=model.predict_distribution(features(future,columns));estimators[name]={'model':model,'columns':columns}
    joblib.dump({'models':estimators,'climatology_history':history,'configuration':{**CONFIG,**(config or {})}},out/'final_models.joblib')
    rows=[]
    for name,p in predictions.items():
        r=future[['row_id','site','issue_time_utc','target_date','lead_day']].copy();r['model']=name
        for k,v in p.items():
            if k!='crps':r['predicted_inches' if k=='mean' else k]=v
        rows.append(r)
    result=pd.concat(rows,ignore_index=True);result.to_csv(out/'current_predictions.csv',index=False)
    future.to_csv(out/'current_input.csv',index=False)
    return result


def attach_targets_for_training(frame,root):
    """Shared declared historical eligibility used by every alignment variant."""
    frame=add_lagged_enso(frame);parts=[];histories={}
    for site,data in frame.groupby('site'):
        history=read_targets(root,site)
        history['winter']=history.target_date.dt.year-(history.target_date.dt.month<7).astype(int)
        history=history[history.winter<=2025]
        if site=='vail':eligible=[2019,2020,2022,2023,2024,2025]
        else:
            cool=history[(history.target_date.dt.month>=10)|(history.target_date.dt.month<=4)].dropna(subset=['target_inches'])
            counts=cool.groupby('winter').target_date.nunique();eligible=counts[counts>=100].index.to_list()
        history=history[history.winter.isin(eligible)]
        if 'target_inches' not in data:
            data=data.assign(target_date=pd.to_datetime(data.target_date)).merge(history,on='target_date',how='left',validate='many_to_one')
        else:
            date=pd.to_datetime(data.target_date);winter=date.dt.year-(date.dt.month<7).astype(int)
            data=data[winter.isin(eligible)]
        parts.append(data);histories[site]=history
    return pd.concat(parts,ignore_index=True),histories


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);parser.add_argument('--site',default=None);parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));args=parser.parse_args()
    frame=pd.read_csv(args.input)
    if args.site:frame=frame[frame.site==args.site]
    frame,histories=attach_targets_for_training(frame,args.root)
    with threadpool_limits(limits=2):
        for site,data in frame.groupby('site'):
            evaluate(data,Path(args.output)/site,histories[site])

if __name__=='__main__':main()
