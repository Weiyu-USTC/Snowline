"""Independent daily snowfall event experiments for Alta, Mansfield and Jay.

Whole-winter evaluation, unknown issue-time local snowfall state, lagged NOAA
weather only. Resort website values never enter training or model selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
import torch

from deep_dillon import calendar_features,lagged_roni,seed_everything
from vail_storm_statistical import EventModel
import multi_resort_neural as neural
from run_vail_storm_models import crps_sample

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/multi-resort'
INPUT=ROOT/'data/multi_resort_daily/current_inputs'
SEED=20261104;CONTEXT=60;HORIZON=60;WEATHER_LAG=2
NAMES=['Climatology','Markov + ENSO','Duration + ENSO','Weather duration + ENSO','Neural events']
SITES={
 'alta':dict(title='Alta-Collins',weather='alta_noaa_daily.csv',first=1999,last=2023,target='Manual Alta-Collins 24-hour new snowfall, ending04:00 LST'),
 'mansfield':dict(title='Mount Mansfield NOAA',weather='mount_mansfield_daily.csv',first=1980,last=2016,target='NOAA USC00435416 daily SNOW; mountain station, not Stowe resort board'),
 'jay':dict(title='Jay Peak NOAA',weather='island_pond_daily.csv',first=1990,last=2012,target='NOAA USC00434189 daily SNOW at560.8m; not summit resort snowfall'),
}
STAT_CONFIGS=[dict(regularization=.05,weather_decay_days=3.),dict(regularization=.5,weather_decay_days=7.)]
NN_CONFIGS=[dict(width=16,dropout=.25,lr=.001,weight_decay=.02,weather_decay_days=3.,batch_size=64),
            dict(width=24,dropout=.35,lr=.0006,weight_decay=.04,weather_decay_days=7.,batch_size=64)]


def future(origin):
    dates=pd.date_range(origin+pd.Timedelta(days=1),periods=HORIZON)
    roni=lagged_roni(origin);lead=np.arange(1,HORIZON+1)/HORIZON
    known=[roni if np.isfinite(roni) else 0.,float(roni>=1.5),float(roni<=-1.5),float(not np.isfinite(roni))]
    return np.column_stack([calendar_features(dates),lead,lead**2,np.tile(known,(HORIZON,1))]).astype('float32')


def event_covariates(f):return np.concatenate([f[...,:4],f[...,6:]],axis=-1)


def weather_summary(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        values=[x[:,-1]]+[np.nanmean(x[:,-w:],axis=1) for w in [3,7,14,30,60]]+[np.nanstd(x[:,-7:],axis=1)]
    values.extend([np.mean(~np.isfinite(x[:,-7:]),axis=1),np.mean(~np.isfinite(x),axis=1)])
    valid=np.isfinite(x);age=np.where(valid.any(1),np.argmax(valid[:,::-1],axis=1),x.shape[1])
    return np.concatenate(values+[age],axis=1)


def observed_history(target,origins):
    result=[]
    for origin in origins:
        winter=origin.year if origin.month>=9 else origin.year-1
        values=target.reindex(pd.date_range(f'{winter}-09-01',origin)).to_numpy(float)
        previous=np.nan;age=0;known=False
        for value in values:
            if not np.isfinite(value):previous=np.nan;age=0;known=False;continue
            if not np.isfinite(previous):age=0;known=False
            elif (value>=1)!=(previous>=1):age=1;known=True
            elif known:age=min(age+1,14)
            previous=value
        result.append([previous,age if known else np.nan])
    return np.asarray(result)


def read_weather(path):
    d=pd.read_csv(path,keep_default_na=False,low_memory=False).set_index('date');d.index=pd.to_datetime(d.index)
    columns=['tmax_c','tmin_c','precipitation_inches'];frame=pd.DataFrame(index=d.index)
    for col in columns:
        values=pd.to_numeric(d[col],errors='coerce')
        quality=d[col+'_quality_flag'].eq('')
        frame[col]=values.where(quality)
    frame['precipitation_inches']=np.log1p(frame.precipitation_inches.clip(lower=0))
    return frame


def prepare(site):
    cfg=SITES[site];dest=OUT/site;dest.mkdir(parents=True,exist_ok=True)
    source_files=[INPUT/cfg['weather'],ROOT/'data/roni_noaa.csv']
    if site=='alta':
        path=ROOT/'data/east_alta_research/alta_hive/alta_collins_daily_paired.csv'
        d=pd.read_csv(path);target=pd.Series(d.snowfall_24h_inches.to_numpy(),index=pd.to_datetime(d.reporting_endpoint_lst).dt.normalize())
        presumed=pd.Series(False,index=target.index)
    else:
        key='mount_mansfield' if site=='mansfield' else 'jay_peak'
        path=ROOT/f'data/east_alta_research/derived_noaa/{key}_daily.csv'
        d=pd.read_csv(path,keep_default_na=False,low_memory=False);d.index=pd.to_datetime(d.date)
        target=pd.to_numeric(d.snowfall_inches,errors='coerce').where(d.snowfall_inches_quality_flag.eq(''))
        presumed=d.snowfall_inches_measurement_flag.eq('P')
    source_files.append(path)
    target=target.sort_index();assert target.index.is_unique
    weather=read_weather(INPUT/cfg['weather'])
    coverage=[]
    for year in range(cfg['first'],cfg['last']+1):
        dates=pd.date_range(f'{year}-11-01',f'{year+1}-04-30');y=target.reindex(dates)
        coverage.append(dict(winter=year,observed_nov_apr=int(y.notna().sum()),expected=len(dates),
                             presumed_zero_nov_apr=int(presumed.reindex(dates,fill_value=False).sum())))
    coverage=pd.DataFrame(coverage);years=coverage[coverage.observed_nov_apr>=150].winter.to_list()[-30:]
    assert len(years)>=12
    coverage['selected']=coverage.winter.isin(years);coverage.to_csv(dest/'winter_coverage.csv',index=False)
    weather=weather.reindex(pd.date_range(f'{min(years)}-07-01','2026-11-04'))
    wx=weather.to_numpy('float32');first=weather.index[0]
    windows=np.lib.stride_tricks.sliding_window_view(wx,(CONTEXT,3))[:,0]
    def contexts(origins):
        ends=pd.DatetimeIndex(origins)-pd.Timedelta(days=WEATHER_LAG)
        positions=(ends-first).days.to_numpy()-CONTEXT+1
        assert positions.min()>=0 and positions.max()<len(windows)
        return np.array(windows[positions],dtype='float32')
    origins=[];ys=[];fs=[]
    for year in years:
        for origin in pd.date_range(f'{year}-10-15',f'{year+1}-02-28',freq='7D'):
            y=target.reindex(pd.date_range(origin+pd.Timedelta(days=1),periods=HORIZON)).to_numpy(float)
            x=contexts([origin])[0]
            if np.isfinite(y).sum()<45 or np.isfinite(x[:,:2]).sum()<60:continue
            origins.append(dict(winter=year,origin=str(origin.date())));ys.append(y);fs.append(future(origin))
    meta=pd.DataFrame(origins);x=contexts(pd.to_datetime(meta.origin));f=np.asarray(fs);y=np.asarray(ys,dtype='float32')
    initial=np.tile([np.nan,0.,0.,0.],(len(meta),1)).astype('float32')
    one_dates=[];one_years=[]
    for year in years:
        for origin in pd.date_range(f'{year}-10-08',f'{year+1}-04-29'):
            if np.isfinite(target.get(origin+pd.Timedelta(days=1),np.nan)):
                one_dates.append(origin);one_years.append(year)
    one_x=contexts(one_dates);one_history=observed_history(target,one_dates)
    one_y=target.reindex(pd.DatetimeIndex(one_dates)+pd.Timedelta(days=1)).to_numpy(float)
    one_f=np.stack([event_covariates(future(o))[0] for o in one_dates])
    meta.to_csv(dest/'origins.csv',index=False)
    pd.DataFrame({'date':target.index,'snowfall_inches':target.to_numpy(),'noaa_presumed_zero':presumed.reindex(target.index,fill_value=False).to_numpy()}).to_csv(dest/'target_daily.csv',index=False)
    np.savez_compressed(dest/'training_arrays.npz',x=x,f=f,y=y,initial=initial)
    plan=dict(site=site,**cfg,years=years,origins=len(meta),pairs=int(np.isfinite(y).sum()),one_step_rows=len(one_y),
              context_days=CONTEXT,horizon=HORIZON,weather_lag_days=WEATHER_LAG,features=['tmax_c','tmin_c','log1p_precip_inches'],
              winter_rule='Last30eligible winters within declared date span with>=150observed November-April target days; weekly origins need>=45/60target labels and>=50%temperature context',
              target_policy='NOAA quality-flagged values excluded; NOAA P-presumed-zero values included as supplied and separately counted; no other missing values filled',
              initial_policy='Current target unavailable: initial snow state/age withheld at EVERYbenchmarkorigin and currentissue; no target-lag weather channels',
              validation='3shuffled whole-winter outerfolds; innerlast2trainwinters forconfig/stopping; separate chronological last3winters test; no test tuning',
              selection='Primary currentfamily selectedbyshuffledCVdailyRMSE; weatherhyperparams innerfree-runningBrier>=1leads1–7; neuralinnerconditionalNLL',
              historical_vintage='Revised NOAA/RONI observations, not archived issue-date vintages; conservative2dayweatherlag butpublicationtimingunverified',
              unknown_current_enso='RONImagnitude clipped to trainrange; constantfuturefeatures heldatfitvalue',
              nn_configs=NN_CONFIGS,stat_configs=STAT_CONFIGS,seed=SEED,
              source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    (dest/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(site=site,stage='prepared',years=years,origins=len(meta),pairs=int(np.isfinite(y).sum()))),flush=True)
    return dict(site=site,dest=dest,years=years,meta=meta,x=x,f=f,y=y,initial=initial,target=target,weather=weather,
                summary=weather_summary(x),one_years=np.array(one_years),one_f=one_f,one_y=one_y,
                one_history=one_history,one_summary=weather_summary(one_x),contexts=contexts,plan=plan)


def guarded(f,model):
    f=event_covariates(f).copy();lo=model.future_min;hi=model.future_max
    constant=hi-lo<1e-7;f[...,constant]=lo[constant];f[...,4]=np.clip(f[...,4],lo[4],hi[4])
    return f


def fit_stat(d,years,kind='semi_markov',weather=False,config=None):
    ix=np.flatnonzero(np.isin(d['one_years'],years));cfg=config or STAT_CONFIGS[0]
    model=EventModel(kind=kind,use_weather=weather,seed=SEED,**cfg)
    model.fit(d['one_summary'][ix],d['one_f'][ix],d['one_history'][ix,0],d['one_history'][ix,1],d['one_y'][ix])
    model.future_min=d['one_f'][ix].min(0);model.future_max=d['one_f'][ix].max(0)
    return model


def predict_stat(model,d,ix,n_paths=128):
    n=len(ix)
    return model.forecast(d['summary'][ix],guarded(d['f'][ix],model),np.full(n,np.nan),np.full(n,np.nan),
                          n_paths=n_paths,seed=SEED,exceedance_threshold=5.)


def climate(d,years,origins,actual=None):
    values=pd.concat([d['target'].loc[f'{year}-10-01':f'{year+1}-04-30'] for year in years]).dropna()
    doy=np.array([pd.Timestamp(2000,v.month,v.day).dayofyear for v in values.index]);lookup={}
    for day in range(1,367):
        diff=np.abs(doy-day);sample=values.to_numpy()[np.minimum(diff,366-diff)<=15]
        if not len(sample):sample=values.to_numpy()
        ss=np.sort(sample);n=len(ss);half=(np.sum((2*np.arange(1,n+1)-n-1)*ss)/(n*n))
        lookup[day]=(sample,half)
    result={k:np.zeros((len(origins),HORIZON)) for k in ['mean','p_active','p_powder','q10','q90','crps']}
    for j,origin in enumerate(pd.to_datetime(origins)):
        for h,date in enumerate(pd.date_range(origin+pd.Timedelta(days=1),periods=HORIZON)):
            sample,half=lookup[pd.Timestamp(2000,date.month,date.day).dayofyear]
            result['mean'][j,h]=sample.mean();result['p_active'][j,h]=np.mean(sample>=1);result['p_powder'][j,h]=np.mean(sample>=5)
            result['q10'][j,h],result['q90'][j,h]=np.quantile(sample,[.1,.9])
            if actual is not None:result['crps'][j,h]=np.mean(abs(sample-actual[j,h]))-half if np.isfinite(actual[j,h]) else np.nan
    return result


def prediction_rows(d,ix,name,pred,tag):
    y=d['y'][ix]
    if 'paths' in pred:
        low,high=np.quantile(pred['paths'],[.1,.9],axis=1);crps=crps_sample(pred['paths'],np.nan_to_num(y),fair=True)
    else:low,high,crps=pred['q10'],pred['q90'],pred['crps']
    records=[]
    for j,index in enumerate(ix):
        origin=pd.Timestamp(d['meta'].iloc[index].origin)
        for h in np.flatnonzero(np.isfinite(y[j])):
            records.append(dict(model=name,split=tag,winter=int(d['meta'].iloc[index].winter),origin=str(origin.date()),
                date=str((origin+pd.Timedelta(days=int(h)+1)).date()),lead=int(h)+1,actual_inches=float(y[j,h]),predicted_inches=float(pred['mean'][j,h]),
                p_active=float(pred['p_active'][j,h]),p_powder=float(pred.get('p_powder',pred.get('p_exceedance'))[j,h]),
                q10_inches=float(low[j,h]),q90_inches=float(high[j,h]),crps=float(crps[j,h])))
    return pd.DataFrame(records)


def select_and_fit(d,train_idx,tag):
    years=sorted(d['meta'].iloc[train_idx].winter.unique().tolist());valid_years=years[-2:];fit_years=years[:-2]
    valid=train_idx[d['meta'].iloc[train_idx].winter.isin(valid_years).to_numpy()]
    fit=train_idx[~d['meta'].iloc[train_idx].winter.isin(valid_years).to_numpy()]
    trials=[]
    for cfg in STAT_CONFIGS:
        model=fit_stat(d,fit_years,weather=True,config=cfg);pred=predict_stat(model,d,valid,n_paths=1)
        yy=d['y'][valid,:7];mask=np.isfinite(yy)
        score=float(np.mean((pred['p_active'][:,:7][mask]-(yy[mask]>=1))**2))
        trials.append(dict(config=cfg,validation_brier=score))
    stat_chosen=min(trials,key=lambda r:r['validation_brier'])['config']
    models={}
    for name,kind,weather in [('Markov + ENSO','markov',False),('Duration + ENSO','semi_markov',False),('Weather duration + ENSO','semi_markov',True)]:
        models[name]=fit_stat(d,years,kind,weather,stat_chosen if weather else None)
    neural_trials=[]
    for c,cfg in enumerate(NN_CONFIGS):
        model,scaler,logs,epoch,score=neural.train_neural(d['x'],d['f'],d['y'],d['initial'],fit,valid,config=cfg,epochs=30,seed=SEED+c)
        neural_trials.append(dict(config=cfg,best_epoch=epoch,validation_nll=score,logs=logs))
        print(json.dumps(dict(site=d['site'],stage=tag,model='Neural',config=c,best_epoch=epoch,val_nll=score)),flush=True)
    chosen=min(neural_trials,key=lambda r:r['validation_nll'])
    model,scaler,logs,_,_=neural.train_neural(d['x'],d['f'],d['y'],d['initial'],train_idx,None,config=chosen['config'],epochs=chosen['best_epoch'],seed=SEED)
    record=dict(training_years=years,validation_years=valid_years,statistical_trials=trials,statistical_selected=stat_chosen,
                neural_trials=neural_trials,neural_selected=chosen['config'],neural_epochs=chosen['best_epoch'],
                parameters=sum(p.numel() for p in model.parameters()),refit_logs=logs)
    (d['dest']/f'{tag}_training.json').write_text(json.dumps(record,indent=2))
    joblib.dump(models,d['dest']/f'{tag}_statistical.joblib')
    torch.save(neural.checkpoint_payload(model,scaler,chosen['config']),d['dest']/f'{tag}_neural.pt')
    return models,model,scaler


def evaluate(d,train_idx,test_idx,tag):
    models,model,scaler=select_and_fit(d,train_idx,tag);frames=[]
    for name,estimator in models.items():
        prediction=predict_stat(estimator,d,test_idx)
        frames.append(prediction_rows(d,test_idx,name,prediction,tag))
    prediction=neural.forecast_neural(model,scaler,d['x'][test_idx],d['f'][test_idx],d['initial'][test_idx],return_components=True)
    prediction['p_powder']=neural.exceedance_probability(prediction,5.)
    prediction['paths']=neural.simulate_neural(model,scaler,d['x'][test_idx],d['f'][test_idx],d['initial'][test_idx],n_paths=128,seed=SEED)
    frames.append(prediction_rows(d,test_idx,'Neural events',prediction,tag))
    years=sorted(d['meta'].iloc[train_idx].winter.unique())
    prediction=climate(d,years,d['meta'].iloc[test_idx].origin,d['y'][test_idx])
    frames.append(prediction_rows(d,test_idx,'Climatology',prediction,tag))
    rows=pd.concat(frames,ignore_index=True);rows.to_csv(d['dest']/f'{tag}_predictions.csv',index=False)
    print(json.dumps(dict(site=d['site'],stage=tag,rows=len(rows))),flush=True)
    return rows


def metrics(rows,dest,prefix):
    scores=[]
    for name,data in rows.groupby('model'):
        for lo,hi,label in [(1,60,'1–60'),(1,1,'1'),(1,3,'1–3'),(4,7,'4–7'),(8,14,'8–14'),(15,30,'15–30'),(31,60,'31–60')]:
            r=data[data.lead.between(lo,hi)];error=r.predicted_inches-r.actual_inches
            scores.append(dict(model=name,leads=label,pairs=len(r),winters=r.winter.nunique(),unique_dates=r.date.nunique(),
                mse=float(np.mean(error**2)),rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(abs(error))),bias=float(error.mean()),
                crps=float(r.crps.mean()),active_brier=float(np.mean((r.p_active-(r.actual_inches>=1))**2)),
                powder_brier=float(np.mean((r.p_powder-(r.actual_inches>=5))**2)),
                coverage80=float(np.mean((r.actual_inches>=r.q10_inches)&(r.actual_inches<=r.q90_inches))),width80=float(np.mean(r.q90_inches-r.q10_inches))))
    scores=pd.DataFrame(scores);scores.to_csv(dest/f'{prefix}_metrics.csv',index=False)
    ref=rows[rows.model=='Climatology'].set_index(['winter','origin','lead']);comparisons=[];rng=np.random.default_rng(SEED)
    for name,g in rows.groupby('model'):
        if name=='Climatology':continue
        g=g.set_index(['winter','origin','lead']).reindex(ref.index)
        assert np.allclose(g.actual_inches,ref.actual_inches)
        for score in ['mse','crps','powder_brier']:
            if score=='mse':delta=(g.predicted_inches-g.actual_inches)**2-(ref.predicted_inches-ref.actual_inches)**2
            elif score=='crps':delta=g.crps-ref.crps
            else:delta=(g.p_powder-(g.actual_inches>=5))**2-(ref.p_powder-(ref.actual_inches>=5))**2
            agg=delta.groupby(level='winter').agg(['sum','count']);draw=rng.integers(0,len(agg),(4000,len(agg)))
            boot=agg['sum'].to_numpy()[draw].sum(1)/agg['count'].to_numpy()[draw].sum(1)
            comparisons.append(dict(model=name,metric=score,difference=float(delta.mean()),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),winters=len(agg)))
    pd.DataFrame(comparisons).to_csv(dest/f'{prefix}_paired.csv',index=False)
    return scores


def issue(d):
    models,model,scaler=select_and_fit(d,np.arange(len(d['meta'])),'final')
    origin=pd.Timestamp('2026-11-04');xx=d['contexts']([origin]);ff=future(origin)[None]
    initial=np.array([[np.nan,0,0,0]],dtype='float32');summary=weather_summary(xx)
    predictions={}
    for name,estimator in models.items():
        predictions[name]=estimator.forecast(summary,guarded(ff,estimator),np.array([np.nan]),np.array([np.nan]),n_paths=2048,seed=SEED,exceedance_threshold=5.)
    p=neural.forecast_neural(model,scaler,xx,ff,initial,return_components=True)
    p['p_powder']=neural.exceedance_probability(p,5.);p['paths']=neural.simulate_neural(model,scaler,xx,ff,initial,n_paths=2048,seed=SEED)
    predictions['Neural events']=p
    predictions['Climatology']=climate(d,d['years'],[origin])
    dates=pd.date_range(origin+pd.Timedelta(days=1),periods=60);keep=dates<=pd.Timestamp('2026-12-31')
    rows=[];paths={};totals=[]
    for name,p in predictions.items():
        if 'paths' in p:
            paths[name]=p['paths'][0][:,keep]
            low,high=np.quantile(paths[name],[.1,.9],axis=0)
        else:low,high=p['q10'][0,keep],p['q90'][0,keep]
        mean=p['mean'][0,keep];powder=p.get('p_powder',p.get('p_exceedance'))[0,keep]
        for i,date in enumerate(dates[keep]):
            rows.append(dict(model=name,date=str(date.date()),lead=i+1,mean_inches=float(mean[i]),q10_inches=float(low[i]),q90_inches=float(high[i]),
                             p_active=float(p['p_active'][0,i]),p_powder=float(powder[i]),cumulative_mean_inches=float(mean[:i+1].sum())))
        for period,mask in [('Nov5–Dec31',np.ones(sum(keep),dtype=bool)),('Nov5–30',dates[keep].month==11),('December',dates[keep].month==12)]:
            row=dict(model=name,period=period,expected_snowfall_inches=float(mean[mask].sum()),expected_powder_days=float(powder[mask].sum()))
            if name in paths:
                total=paths[name][:,mask].sum(1);counts=(paths[name][:,mask]>=5).sum(1)
                row.update(q10_inches=float(np.quantile(total,.1)),q90_inches=float(np.quantile(total,.9)),powder_count_q10=float(np.quantile(counts,.1,method='inverted_cdf')),powder_count_q90=float(np.quantile(counts,.9,method='inverted_cdf')))
            totals.append(row)
    pd.DataFrame(rows).to_csv(d['dest']/'forecast_daily.csv',index=False)
    pd.DataFrame(totals).to_csv(d['dest']/'forecast_totals.csv',index=False)
    np.savez_compressed(d['dest']/'forecast_paths.npz',**paths)
    (d['dest']/'forecast_manifest.json').write_text(json.dumps(dict(issue_date=str(origin.date()),target=SITES[d['site']]['target'],
        first_date='2026-11-05',last_date='2026-12-31',initial_state='Unknown; train-onlycalendarprior',
        weather_nominal_last_date='2026-11-02',weather_last_observed={c:str(d['weather'][c].dropna().index.max().date()) for c in d['weather']},
        context_missing_fraction=float(np.mean(~np.isfinite(xx))),current_lagged_roni=float(ff[0,0,6]),
        neural_roni_train_range=[float(scaler.future_min[6]),float(scaler.future_max[6])],paths=2048,
        intervals='Uncalibrated model-based10–90%predictive ranges; cumulativequantiles fromjointpaths',
        interpretation='Frozen observed-weather/ENSO outlook; no futureatmosphericforecastinputs; targethistoricalstationdefinition remains unchanged'),indent=2))


def run(site,mode):
    d=prepare(site)
    if mode=='prepare':return
    if mode=='run':
        frames=[]
        for fold,(tr,te) in enumerate(KFold(n_splits=3,shuffle=True,random_state=SEED).split(d['years'])):
            train=np.flatnonzero(d['meta'].winter.isin(np.array(d['years'])[tr]));test=np.flatnonzero(d['meta'].winter.isin(np.array(d['years'])[te]))
            frames.append(evaluate(d,train,test,f'fold{fold}'))
        rows=pd.concat(frames,ignore_index=True);rows.to_csv(d['dest']/'crossvalidation_predictions.csv',index=False)
        metrics(rows,d['dest'],'crossvalidation')
        metrics(rows[pd.to_datetime(rows.date).dt.month.isin([11,12])],d['dest'],'nov_dec')
        train=np.flatnonzero(d['meta'].winter.isin(d['years'][:-3]));test=np.flatnonzero(d['meta'].winter.isin(d['years'][-3:]))
        chrono=evaluate(d,train,test,'chronological');metrics(chrono,d['dest'],'chronological')
    issue(d)
    print(json.dumps(dict(site=site,stage='DONE')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',choices=list(SITES)+['all'],default='all');parser.add_argument('--mode',choices=['prepare','run','forecast'],default='run')
    args=parser.parse_args();seed_everything(SEED)
    with threadpool_limits(limits=2):
        for site in SITES if args.site=='all' else [args.site]:run(site,args.mode)
