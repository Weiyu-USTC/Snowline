"""Matched short-range radar and independently measured daily snow evaluation."""
from pathlib import Path
import argparse, json, hashlib, warnings
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, shift
from sklearn.linear_model import LogisticRegression
import joblib
import nwp_benchmark as nwp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/alta-nowcast'
HRRR=ROOT/'data/alta_nowcast/hrrr/runs'
LABELS={'convlstm':'ConvLSTM + HRRR','dgmr':'DGMR adaptation + HRRR','nowcastnet':'NowcastNet adaptation + HRRR'}
RADAR_LABELS={'convlstm':'ConvLSTM','dgmr':'DGMR adaptation','nowcastnet':'NowcastNet adaptation'}
FOREST_CONFIG=dict(n_estimators=128,max_depth=5,min_samples_leaf=10,max_features=.8,n_jobs=2,random_state=20261105)
SEED=20261105


def integral(rate,hours):
    """Forecast positions1..19 span11:00..14:00UTC, rate inmm/hour."""
    return np.trapezoid(rate[...,1:6*hours+2,:,:],dx=1/6,axis=-3)


def ensemble_crps(ensemble,actual,axis=0):
    e=np.moveaxis(np.asarray(ensemble,float),axis,0)
    n=len(e);s=np.sort(e,axis=0)
    c=(2*np.arange(1,n+1)-n-1).reshape((n,)+(1,)*(e.ndim-1))
    return np.mean(abs(e-actual),axis=0)-np.sum(c*s,axis=0)/(n*n)


def phase_flow(x,leads=20):
    """Traditional global phase-correlation advection; not the pySTEPS package."""
    rate=x[:,0] if x.ndim==4 else x
    a=np.nan_to_num(rate[-4:]);motions=[]
    taper=np.outer(np.hanning(a.shape[-2]),np.hanning(a.shape[-1]))
    for before,after in zip(a[:-1],a[1:]):
        if max(before.max(),after.max())<.05:
            motions.append((0.,0.));continue
        f=np.fft.fft2(np.log1p(after)*taper)*np.conj(np.fft.fft2(np.log1p(before)*taper))
        f/=np.maximum(abs(f),1e-9)
        corr=np.fft.ifft2(f).real
        yy,xx=np.unravel_index(np.argmax(corr),corr.shape)
        yy=yy if yy<=a.shape[-2]//2 else yy-a.shape[-2]
        xx=xx if xx<=a.shape[-1]//2 else xx-a.shape[-1]
        motions.append((np.clip(yy,-4,4),np.clip(xx,-4,4)))
    dy,dx=np.median(motions,axis=0)
    return np.stack([shift(a[-1],((i+1)*dy,(i+1)*dx),order=1,mode='nearest',prefilter=False) for i in range(leads)])


def load_hrrr_run(case_id):
    folder=HRRR/(str(case_id)+'T0600Z')
    names=['precip_mm','snow_water_mm','temperature_k','dewpoint_k','wind_u_ms','wind_v_ms','relative_humidity_pct']
    d={}
    for name in names:
        file=folder/(name+'.npz')
        if not file.exists():return None
        with np.load(file) as z:
            periods=z['periods'].astype(int)
            if not np.all(np.isin(np.arange(6,30),periods)):return None
            d[name]={k:z[k].copy() for k in ['periods','point','mean3x3']}
            if name=='precip_mm':
                if 'mrms64' not in z:return None
                d[name]['mrms64']=z['mrms64'].copy()
    if any(not np.isfinite(v['mean3x3'][np.isin(v['periods'],np.arange(6,30))]).all() for v in d.values()):return None
    return d


def weather_features(d,date,start_period=6):
    """Same forecast hours and variables for every daily calibration model."""
    result={}
    for name,v in d.items():
        p=v['periods'];a=v['mean3x3']
        for h in [3,12,24]:
            selected=a[(p>=start_period)&(p<start_period+h)]
            if len(selected)!=h:raise ValueError('Incomplete HRRR forecast interval')
            if name in ['precip_mm','snow_water_mm']:
                result[f'{name}_sum_{h}h']=float(selected.sum())
            else:
                result[f'{name}_mean_{h}h']=float(selected.mean())
                if h==24:
                    result[f'{name}_min_24h']=float(selected.min())
                    result[f'{name}_max_24h']=float(selected.max())
    phase=2*np.pi*(pd.Timestamp(date).dayofyear-1)/365.2425
    result['calendar_sin']=np.sin(phase);result['calendar_cos']=np.cos(phase)
    return result


def local_radar_features(ensemble,iy,ix):
    """No observed future snowfall is used in forecast features."""
    e=np.asarray(ensemble,float)
    if e.ndim==3:e=e[None]
    result={}
    for h in [1,2,3]:
        a=integral(e,h)
        local=a[:,max(0,iy-1):iy+2,max(0,ix-1):ix+2].mean((1,2))
        result[f'radar_liquid_{h}h_mean']=float(local.mean())
        if h==3:
            result['radar_liquid_3h_std']=float(local.std())
            result['radar_liquid_3h_q90']=float(np.quantile(local,.9))
            result['radar_liquid_3h_region_mean']=float(a[:,iy-8:iy+9,ix-8:ix+9].mean())
            result['radar_liquid_3h_region_max']=float(a[:,iy-8:iy+9,ix-8:ix+9].mean(0).max())
    return result


def history_features(x,iy,ix):
    rate=x[:,0];quality=x[:,1]
    local=rate[:,iy-1:iy+2,ix-1:ix+2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        vals=np.nanmean(local,(1,2))
        result={'radar_current_rate':float(vals[-1]),'radar_history_mean':float(np.nanmean(vals)),
                'radar_history_change':float(vals[-1]-vals[0]),'radar_input_quality':float(quality[:,iy-1:iy+2,ix-1:ix+2].mean())}
    return result


def load_all():
    # Materialize once; NPZ otherwise decompresses the full array for every
    # per-case index, which would dominate the evaluation and figure runtime.
    with np.load(OUT/'learning_arrays.npz') as archive:
        z={k:archive[k] for k in archive.files}
    cases=pd.read_csv(OUT/'case_inventory.csv',dtype={'case_id':str})
    ids=z['case_id'].astype(str);assert np.array_equal(ids,cases.case_id)
    pred={};lookup={}
    for name in LABELS:
        p=OUT/'radar_predictions'/f'{name}.npy'
        pred[name]=np.load(p,mmap_mode='r')
        idx=np.load(p.with_name(name+'_index.npz'))
        lookup[name]={str(v):i for i,v in enumerate(idx['case_id'])}
    return z,cases,pred,lookup


def field_scores(ens,truth,mask):
    e=np.asarray(ens,float);mean=e.mean(0)
    good=mask&np.isfinite(mean)&np.isfinite(truth)&np.isfinite(e).all(0)
    if good.sum()==0:return None
    error=(mean-truth)[good];event=truth>=1.;warning=mean>=1.
    prob=(e>=1.).mean(0)
    tp=int((event&warning&good).sum());fp=int((~event&warning&good).sum());fn=int((event&~warning&good).sum())
    count=uniform_filter(good.astype(float),size=5,mode='constant')
    observed_fraction=uniform_filter((event&good).astype(float),size=5,mode='constant')/np.maximum(count,1e-8)
    forecast_fraction=uniform_filter((warning&good).astype(float),size=5,mode='constant')/np.maximum(count,1e-8)
    neighborhood=count>=.7
    return dict(mae=float(abs(error).mean()),mse=float((error**2).mean()),bias=float(error.mean()),
                crps=float(ensemble_crps(e,truth)[good].mean()),brier=float(((prob-event)**2)[good].mean()),
                tp=tp,fp=fp,fn=fn,pixels=int(good.sum()),
                fss_numerator=float(((forecast_fraction-observed_fraction)**2)[neighborhood].sum()),
                fss_denominator=float((forecast_fraction**2+observed_fraction**2)[neighborhood].sum()))


def short_evaluate(z,cases,pred,lookup,hrrr):
    rows=[];iy=np.argmin(abs(z['latitude']-40.57607));ix=np.argmin(abs(z['longitude']+111.63889))
    test=cases[(cases.split=='test')&cases.radar_eligible&cases.case_id.isin(hrrr)]
    for i,r in test.iterrows():
        cid=r.case_id
        if any(cid not in lookup[n] for n in LABELS):continue
        y=z['y'][i];x=z['x'][i]
        persistence=np.broadcast_to(np.nan_to_num(x[-1,0]),(1,20,64,64))
        flow=phase_flow(x)[None]
        ensembles={'Persistence':persistence,'Motion extrapolation':flow,
                   **{RADAR_LABELS[n]:np.asarray(pred[n][lookup[n][cid]],float) for n in LABELS}}
        a=hrrr[cid]['precip_mm'];period=a['periods']
        for h in [1,2,3]:
            truth=integral(y,h);mask=np.isfinite(y[1:6*h+2]).all(0)
            forecasts={name:integral(e,h) for name,e in ensembles.items()}
            forecasts['HRRR']=a['mrms64'][(period>=6)&(period<6+h)].sum(0)[None]
            common=mask.copy()
            for e in forecasts.values():common&=np.isfinite(e).all(0)
            for name,e in forecasts.items():
                scores=field_scores(e,truth,common)
                if scores:
                    row={'case_id':cid,'date':r.date,'model':name,'hours':h,**scores}
                    row['alta_truth_mm']=float(truth[iy,ix]) if common[iy,ix] else np.nan
                    row['alta_predicted_mm']=float(e[:,iy,ix].mean())
                    row['alta_mae_mm']=abs(row['alta_truth_mm']-row['alta_predicted_mm'])
                    row['alta_mse']=float((row['alta_truth_mm']-row['alta_predicted_mm'])**2)
                    rows.append(row)
    per=pd.DataFrame(rows);per.to_csv(OUT/'short_case_metrics.csv',index=False)
    summary=[]
    for (name,h),d in per.groupby(['model','hours']):
        tp,fp,fn=d[['tp','fp','fn']].sum();den=d.fss_denominator.sum()
        summary.append(dict(model=name,hours=int(h),cases=len(d),mae_mm=d.mae.mean(),rmse_mm=np.sqrt(d.mse.mean()),
                            crps_mm=d.crps.mean(),bias_mm=d.bias.mean(),raw_event_brier=d.brier.mean(),
                            precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan,
                            csi=tp/(tp+fp+fn) if tp+fp+fn else np.nan,fss=1-d.fss_numerator.sum()/den if den else np.nan,
                            alta_cases=int(d.alta_truth_mm.notna().sum()),alta_mae_mm=d.alta_mae_mm.mean(),alta_rmse_mm=np.sqrt(d.alta_mse.mean())))
    pd.DataFrame(summary).to_csv(OUT/'short_summary.csv',index=False)
    comparisons=[]
    for h in [1,2,3]:
        ss=per[per.hours==h];base=ss[ss.model=='HRRR'].set_index('case_id')
        for name,g in ss.groupby('model'):
            if name=='HRRR':continue
            g=g.set_index('case_id').reindex(base.index)
            for metric in ['mae','mse','crps']:
                ci=block_difference(g[metric].to_numpy()-base[metric].to_numpy(),base.date)
                comparisons.append(dict(model=name,reference='HRRR',hours=h,metric=metric,**ci))
    pd.DataFrame(comparisons).to_csv(OUT/'short_paired_intervals.csv',index=False)
    return per


def block_difference(delta,dates,draws=2500):
    delta=np.asarray(delta,float);dates=pd.to_datetime(np.asarray(dates));good=np.isfinite(delta)
    delta=delta[good];dates=dates[good]
    blocks=((dates-dates.min()).days//7).to_numpy()
    frame=pd.DataFrame({'block':blocks,'delta':delta}).groupby('block').delta.agg(['sum','count'])
    rng=np.random.default_rng(SEED);ii=rng.integers(0,len(frame),(draws,len(frame)))
    boot=frame['sum'].to_numpy()[ii].sum(1)/frame['count'].to_numpy()[ii].sum(1)
    return {'difference':float(delta.mean()),'ci95_low':float(np.quantile(boot,.025)),
            'ci95_high':float(np.quantile(boot,.975)),'blocks':len(frame),'winters':1}


def daily_evaluate(z,cases,pred,lookup,hrrr):
    iy=np.argmin(abs(z['latitude']-40.57607));ix=np.argmin(abs(z['longitude']+111.63889))
    rows=[]
    for i,r in cases[(cases.split.isin(['validation','development','val','test']))&cases.radar_eligible&cases.case_id.isin(hrrr)].iterrows():
        cid=r.case_id
        if any(cid not in lookup[n] for n in LABELS):continue
        if not np.isfinite(r.snowfall_24h_inches):continue
        common={'case_id':cid,'date':r.date,'target_date':r.target_date,'split':r.split,'winter':r.winter,
                'actual':r.snowfall_24h_inches,**weather_features(hrrr[cid],r.date),**history_features(z['x'][i],iy,ix)}
        for name in LABELS:
            feats=local_radar_features(np.asarray(pred[name][lookup[name][cid]],float),iy,ix)
            common.update({f'{name}__{k}':v for k,v in feats.items()})
        rows.append(common)
    d=pd.DataFrame(rows)
    if len(d)==0:raise ValueError('No matched development/test daily data')
    baseline=pd.read_csv(ROOT/'outputs/nwp-weather/mos/alta/matched_input.csv')
    baseline['target_date']=pd.to_datetime(baseline.target_date).dt.strftime('%Y-%m-%d')
    baseline_training=baseline.copy()
    # Same physical day and freshest archived00UTC MOS run; olderlead2 duplicate excluded.
    baseline=baseline[baseline.lead_day==1].copy()
    d['target_date']=pd.to_datetime(d.target_date).dt.strftime('%Y-%m-%d')
    d=d[d.target_date.isin(baseline.target_date)].reset_index(drop=True)
    d.to_csv(OUT/'daily_matched_features.csv',index=False)
    test=d[d.split=='test'].copy();dev=d[d.split!='test'].copy()
    assert set(dev.winter)=={2022} and set(test.winter)=={2023}
    metadata=['case_id','date','target_date','split','winter','actual']
    weather=[c for c in d if c not in metadata and '__' not in c and not c.startswith('radar_')]
    history=[c for c in d if c.startswith('radar_')]
    groups={'HRRR calibrated (matched)':weather,'HRRR + observed radar':weather+history}
    groups.update({label:weather+history+[c for c in d if c.startswith(name+'__')] for name,label in LABELS.items()})
    forecasts=[];fitted={}
    for name,columns in groups.items():
        fit=nwp.EmpiricalForest(FOREST_CONFIG).fit(dev[columns],dev.actual)
        p=fit.predict_distribution(test[columns],test.actual)
        rr=test[metadata].copy();rr['model']=name
        for k,v in p.items():rr['predicted_inches' if k=='mean' else k]=v
        forecasts.append(rr);fitted[name]={'estimator':fit,'columns':columns}
    # Preserve the existing model's training on both daily leads, but evaluate
    # only its freshest lead1 forecast for the same future physical24h window.
    old_train=baseline_training[baseline_training.winter<=2022].copy()
    old_test=baseline.set_index('target_date').loc[test.target_date].reset_index()
    assert np.allclose(old_test.target_inches,test.actual)
    old_train['target_date']=pd.to_datetime(old_train.target_date)
    old_test['target_date']=pd.to_datetime(old_test.target_date)
    cols=nwp.feature_groups(old_train)[nwp.NAMES[-1]]
    old=nwp.EmpiricalForest().fit(nwp.features(old_train,cols),old_train.target_inches)
    p=old.predict_distribution(nwp.features(old_test,cols),old_test.target_inches)
    rr=test[metadata].copy();rr['model']='Current MOS model (refitted)'
    for k,v in p.items():rr['predicted_inches' if k=='mean' else k]=v
    forecasts.append(rr)
    # Calendar baseline uses every historical manual outcome beforetheheldoutwinter.
    history=nwp.read_targets(ROOT,'alta');history=history[pd.to_datetime(history.target_date)<pd.Timestamp('2023-07-01')]
    query=pd.DataFrame({'target_date':pd.to_datetime(test.target_date),'target_inches':test.actual})
    p=nwp.climatology(history,query)
    rr=test[metadata].copy();rr['model']='Historical mean'
    for k,v in p.items():rr['predicted_inches' if k=='mean' else k]=v
    forecasts.append(rr)
    result=pd.concat(forecasts,ignore_index=True);result.to_csv(OUT/'daily_predictions.csv',index=False)
    joblib.dump({'new_models':fitted,'old_model':old,'old_columns':cols,'old_training_cases':len(old_train),
                 'development_cases':len(dev),'test_cases':len(test),'config':FOREST_CONFIG},OUT/'daily_calibrators.joblib')
    summaries=[]
    for name,g in result.groupby('model'):
        e=g.predicted_inches-g.actual;event=(g.actual>=5).astype(float)
        tp=((g.p_powder>=.5)&(event==1)).sum();fp=((g.p_powder>=.5)&(event==0)).sum();fn=((g.p_powder<.5)&(event==1)).sum()
        summaries.append(dict(model=name,cases=len(g),powder_days=int(event.sum()),mae_inches=abs(e).mean(),
                              rmse_inches=np.sqrt((e*e).mean()),bias_inches=e.mean(),crps_inches=g.crps.mean(),
                              powder_brier=((g.p_powder-event)**2).mean(),coverage80=((g.actual>=g.q10)&(g.actual<=g.q90)).mean(),
                              predicted_powder_rate=g.p_powder.mean(),actual_powder_rate=event.mean(),
                              powder_precision=tp/(tp+fp) if tp+fp else np.nan,powder_recall=tp/(tp+fn) if tp+fn else np.nan))
    s=pd.DataFrame(summaries)
    ref=s[s.model=='Current MOS model (refitted)'].iloc[0]
    s['mse_skill_vs_current']=1-s.rmse_inches**2/ref.rmse_inches**2
    s['brier_skill_vs_current']=1-s.powder_brier/ref.powder_brier
    s.to_csv(OUT/'daily_summary.csv',index=False)
    comparisons=[]
    for reference in ['Current MOS model (refitted)','HRRR calibrated (matched)','HRRR + observed radar']:
        b=result[result.model==reference].set_index('case_id')
        for name,g in result.groupby('model'):
            if name==reference:continue
            g=g.set_index('case_id').reindex(b.index)
            deltas={'mse':(g.predicted_inches-g.actual)**2-(b.predicted_inches-b.actual)**2,
                    'mae':abs(g.predicted_inches-g.actual)-abs(b.predicted_inches-b.actual),
                    'crps':g.crps-b.crps,
                    'powder_brier':(g.p_powder-(g.actual>=5))**2-(b.p_powder-(b.actual>=5))**2}
            for metric,delta in deltas.items():comparisons.append(dict(model=name,reference=reference,metric=metric,**block_difference(delta.to_numpy(),b.date)))
    pd.DataFrame(comparisons).to_csv(OUT/'daily_paired_intervals.csv',index=False)
    reliability=[]
    for name,g in result.groupby('model'):
        bins=np.minimum((g.p_powder.to_numpy()*5).astype(int),4)
        for b in range(5):
            z=g[bins==b]
            if len(z):reliability.append({'model':name,'bin':b,'cases':len(z),'predicted':z.p_powder.mean(),'observed':(z.actual>=5).mean()})
    pd.DataFrame(reliability).to_csv(OUT/'daily_reliability.csv',index=False)
    audit={'matched_development_cases':len(dev),'matched_test_cases':len(test),'test_winter':'2023/24 only',
           'prior_model_training_cases':len(old_train),'prior_model_latest_training_winter':int(old_train.winter.max()),
           'prior_model_refitted_not_loaded_full_data_weights':True,'powder_threshold_inches':5,
           'new_calibrators_training_winter':2022,'new_calibrators_use_neural_training_examples':False,
           'all_daily_models_same_test_case_ids':True,'manual24h_window':'11UTC–next11UTC',
           'forecast_models':'Issued06UTC HRRR; issued00UTC GFS MOS; causalradarending10:40UTC',
           'limitation':'Neural checkpoint selection and daily calibration both use development winter2022/23; no2023/24 outcomes enter either.'}
    (OUT/'evaluation_audit.json').write_text(json.dumps(audit,indent=2))
    return result


def main():
    z,cases,pred,lookup=load_all()
    hrrr={}
    for cid in cases[cases.split!='train'].case_id:
        d=load_hrrr_run(cid)
        if d is not None:hrrr[cid]=d
    print(f'Complete development/test HRRR runs: {len(hrrr)}',flush=True)
    short_evaluate(z,cases,pred,lookup,hrrr)
    daily_evaluate(z,cases,pred,lookup,hrrr)
    print(pd.read_csv(OUT/'daily_summary.csv').to_string(index=False),flush=True)


if __name__=='__main__':main()
