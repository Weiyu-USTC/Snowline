"""Independent final checks; execution requires explicit test-unblinding consent.

Run only after root has frozen model selection and released the test results:
  MPLCONFIGDIR=.mplconfig .venv-hrrr/bin/python src/audit_alta_nowcast_results.py --allow-test-unblinding
No training, tuning, data correction or result-file mutation occurs here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/alta-nowcast'
DATA=ROOT/'data/alta_nowcast'
KEYS=('convlstm','dgmr','nowcastnet')
LABELS={'convlstm':'ConvLSTM','dgmr':'DGMR adaptation','nowcastnet':'NowcastNet adaptation'}
CHECKS=[]


def check(name,passed,detail=None):
    CHECKS.append(dict(check=name,passed=bool(passed),detail=detail))
    if not passed:
        raise AssertionError(f'{name}: {detail}')


def close(name,got,expected,atol=2e-6,rtol=2e-6):
    ok=np.allclose(np.asarray(got,float),np.asarray(expected,float),atol=atol,rtol=rtol,equal_nan=True)
    check(name,ok)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def integrate(a,h):
    """Explicit trapezoidal weights, independently of evaluation helper."""
    a=np.asarray(a,float)[...,1:6*h+2,:,:]
    return (a[...,0,:,:]/2+a[...,-1,:,:]/2+a[...,1:-1,:,:].sum(axis=-3))/6


def discrete_distribution(values,weights,actual=None):
    order=np.argsort(values);v=np.asarray(values,float)[order];w=np.asarray(weights,float)[order]
    check('donor probability mass',np.isclose(w.sum(),1,atol=1e-9))
    cdf=w.cumsum()
    result={'mean':float(w@v),'p_powder':float(w@(v>=5)),'p_snow':float(w@(v>0)),
            'q10':float(v[np.searchsorted(cdf,.1)]),'q90':float(v[np.searchsorted(cdf,.9)])}
    if actual is not None:
        # Independent O(n log n) identity for the weighted pair-distance term.
        prior_weight=np.r_[0,cdf[:-1]]
        prior_moment=np.r_[0,np.cumsum(w*v)[:-1]]
        half_pair=np.sum(w*(v*prior_weight-prior_moment))
        result['crps']=float(w@abs(v-actual)-half_pair)
    return result


def replay_forest(estimator,row,actual=None):
    raw=np.asarray(row,float).reshape(1,-1)
    missing=~np.isfinite(raw)
    xx=np.column_stack([np.where(missing,estimator.median,raw),missing.astype(float)])
    leaves=estimator.forest.apply(xx)[0]
    weights=np.zeros(len(estimator.y))
    for t,leaf in enumerate(leaves):
        donors=estimator.donors[t][int(leaf)]
        weights[donors]+=1/(len(leaves)*len(donors))
    return discrete_distribution(estimator.y,weights,actual)


def probability_checks(frame,mean_column,prefix):
    cols=[mean_column,'p_powder','p_snow','q10','q90']
    check(prefix+' finite probabilities/distribution',np.isfinite(frame[cols].to_numpy()).all())
    check(prefix+' probability bounds',frame[['p_powder','p_snow']].ge(0).all().all()
          and frame[['p_powder','p_snow']].le(1+1e-10).all().all())
    check(prefix+' powder implies snow',(frame.p_powder<=frame.p_snow+1e-10).all())
    check(prefix+' nonnegative mean/quantiles',(frame[[mean_column,'q10','q90']]>=0).all().all())
    check(prefix+' ordered quantiles',(frame.q10<=frame.q90).all())
    # A heavy-tailed mean need not lie inside a central80% interval. Exact
    # distribution replay below, rather than that invalid constraint, checks it.


def audit_daily():
    result=pd.read_csv(OUT/'daily_predictions.csv',dtype={'case_id':str})
    summary=pd.read_csv(OUT/'daily_summary.csv').set_index('model')
    features=pd.read_csv(OUT/'daily_matched_features.csv',dtype={'case_id':str})
    cases=pd.read_csv(OUT/'case_inventory.csv',dtype={'case_id':str}).set_index('case_id')
    check('daily duplicate model/case rows',not result.duplicated(['model','case_id']).any())
    ids=[set(g.case_id) for _,g in result.groupby('model')]
    check('all daily models identical cases',all(v==ids[0] for v in ids),len(ids[0]))
    check('daily test winter only',(result.winter==2023).all() and (result['split']=='test').all())
    check('same actual across methods',result.groupby('case_id').actual.nunique().eq(1).all())
    probability_checks(result,'predicted_inches','daily')
    raw=pd.read_csv(ROOT/'data/east_alta_research/alta_hive/alta_collins_daily_paired.csv')
    raw=raw[raw.complete_24h_pair.astype(bool)].copy()
    raw['target_date']=pd.to_datetime(raw.reporting_endpoint_lst).dt.strftime('%Y-%m-%d')
    joined=result.merge(raw[['target_date','period_start_lst','reporting_endpoint_lst','snowfall_24h_inches']],on='target_date',validate='many_to_one')
    check('all daily rows have raw manual target',len(joined)==len(result))
    close('raw manual amounts',joined.actual,joined.snowfall_24h_inches)
    start=pd.to_datetime(joined.period_start_lst)+pd.Timedelta(hours=7)
    end=pd.to_datetime(joined.reporting_endpoint_lst)+pd.Timedelta(hours=7)
    date=pd.to_datetime(joined.date)
    check('raw manual interval physical clock',((start==date+pd.Timedelta(hours=11))&(end==date+pd.Timedelta(hours=35))).all())
    recalculated={}
    for name,g in result.groupby('model'):
        e=g.predicted_inches.to_numpy()-g.actual.to_numpy();event=g.actual.to_numpy()>=5
        values=dict(cases=len(g),powder_days=int(event.sum()),mae_inches=float(abs(e).mean()),
                    rmse_inches=float(np.sqrt(np.mean(e*e))),bias_inches=float(e.mean()),
                    powder_brier=float(np.mean((g.p_powder.to_numpy()-event)**2)),
                    crps_inches=float(g.crps.mean()),coverage80=float(((g.actual>=g.q10)&(g.actual<=g.q90)).mean()))
        for k,v in values.items():close(f'daily summary {name} {k}',summary.loc[name,k],v,atol=1e-9,rtol=1e-9)
        recalculated[name]=values
    base=recalculated['Current MOS model (refitted)']
    for name,v in recalculated.items():
        close(f'daily MSE skill {name}',summary.loc[name,'mse_skill_vs_current'],1-v['rmse_inches']**2/base['rmse_inches']**2)
        close(f'daily Brier skill {name}',summary.loc[name,'brier_skill_vs_current'],1-v['powder_brier']/base['powder_brier'])
    fits=joblib.load(OUT/'daily_calibrators.joblib')
    dev=features[features['split']!='test'].copy()
    check('daily calibration excludes test winter',set(dev.winter)=={2022})
    check('stored development count',fits['development_cases']==len(dev))
    sample_ids=sorted(ids[0])[::max(1,len(ids[0])//20)][:20]
    indexed=features.set_index('case_id')
    predicted=result.set_index(['model','case_id'])
    for name,b in fits['new_models'].items():
        estimator=b['estimator'];cols=b['columns']
        close(f'new forest donors {name}',estimator.y,dev.actual,atol=0,rtol=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            expected_median=np.nan_to_num(np.nanmedian(dev[cols].to_numpy(float),axis=0))
        close(f'training-only imputation {name}',estimator.median,expected_median)
        for cid in sample_ids:
            actual=float(predicted.loc[(name,cid),'actual'])
            p=replay_forest(estimator,indexed.loc[cid,cols],actual)
            for key,value in p.items():
                saved='predicted_inches' if key=='mean' else key
                close(f'daily distribution replay {name} {cid} {key}',predicted.loc[(name,cid),saved],value)
    mos=pd.read_csv(ROOT/'outputs/nwp-weather/mos/alta/matched_input.csv')
    train=mos[mos.winter<=2022].copy()
    check('old MOS training excludes test winter',train.winter.max()==2022)
    check('old MOS retains both leads',set(train.lead_day)=={1,2})
    close('old MOS donor outcomes',fits['old_model'].y,train.target_inches,atol=0,rtol=0)
    check('old MOS training count',fits['old_training_cases']==len(train))
    mos['target_date']=pd.to_datetime(mos.target_date).dt.strftime('%Y-%m-%d')
    latest=mos[mos.lead_day==1].set_index('target_date')
    for cid in sample_ids:
        r=predicted.loc[('Current MOS model (refitted)',cid)];src=latest.loc[r.target_date]
        phase=2*np.pi*(pd.Timestamp(r.target_date).dayofyear-1)/365.2425
        x=np.r_[np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase),float(src.lead_day)/14,src[fits['old_columns']].to_numpy(float)]
        p=replay_forest(fits['old_model'],x,float(r.actual))
        for key,value in p.items():close(f'old MOS replay {cid} {key}',r['predicted_inches' if key=='mean' else key],value)
    history=pd.read_csv(ROOT/'outputs/multi-resort/alta/target_daily.csv')
    history['date']=pd.to_datetime(history.date)
    history=history[(history.date<pd.Timestamp('2023-07-01'))&history.snowfall_inches.notna()].drop_duplicates('date')
    doy=np.asarray([pd.Timestamp(2000,t.month,t.day).dayofyear for t in history.date])
    for cid in sample_ids:
        r=predicted.loc[('Historical mean',cid)];date=pd.Timestamp(r.target_date)
        day=pd.Timestamp(2000,date.month,date.day).dayofyear
        distance=abs(doy-day);near=np.minimum(distance,366-distance)<=15
        if not near.any():near[:]=True
        p=discrete_distribution(history.snowfall_inches,near/near.sum(),float(r.actual))
        for key,value in p.items():close(f'climatology replay {cid} {key}',r['predicted_inches' if key=='mean' else key],value)
    metadata=json.loads((OUT/'evaluation_audit.json').read_text())
    check('evaluation metadata excludes test fitting',metadata['prior_model_latest_training_winter']==2022
          and metadata['new_calibrators_training_winter']==2022 and metadata['prior_model_refitted_not_loaded_full_data_weights'])
    return dict(models=len(ids),test_cases=len(ids[0]),distribution_replay_cases_per_model=len(sample_ids))


def audit_model_identity():
    cases=pd.read_csv(OUT/'case_inventory.csv',dtype={'case_id':str}).set_index('case_id')
    selection=json.loads((OUT/'models/selected/recipe_selection.json').read_text())
    records={r['model']:r for r in selection};result={}
    arrays_hash=sha(OUT/'learning_arrays.npz')
    for name in KEYS:
        record=records[name];selected=OUT/'models/selected'/f'{name}.pt'
        digest=sha(selected)
        check(f'{name} recipe selected by development',record['selected']['development_mse']==min(c['development_mse'] for c in record['candidates']))
        check(f'{name} recipe no test score',record['test_score_used'] is False)
        check(f'{name} selected checkpoint identity',digest==record['selected']['checkpoint_sha256'])
        meta=json.loads(Path(str(selected)+'.json').read_text())
        check(f'{name} adaptation provenance',meta['manifest']['author_checkpoint'] is False and meta['config']['name']==name)
        recipe=record['selected']['recipe']
        expected_recipe='wet_mse' if recipe=='wet_mse_linear_head' else recipe
        check(f'{name} selected recipe flags',meta['config'].get('recipe','uniform_l1')==expected_recipe
              and bool(meta['config'].get('linear_rate_head',False))==(recipe=='wet_mse_linear_head'))
        check(f'{name} checkpoint training finished',not meta.get('training_in_progress',False))
        pm=json.loads((OUT/'radar_predictions'/f'{name}_manifest.json').read_text())
        check(f'{name} final prediction checkpoint',pm['checkpoint_sha256']==digest and pm['status']=='complete')
        check(f'{name} immutable input arrays',pm['learning_arrays_sha256']==arrays_hash)
        check(f'{name} test excluded from neural fitting',pm['test_used_for_fitting_or_selection'] is False)
        check(f'{name} training winter membership',set(cases.loc[pm['training_cases'],'winter'])<={2020,2021})
        check(f'{name} selection winter membership',set(cases.loc[pm['checkpoint_selection_cases'],'winter'])=={2022})
        check(f'{name} fitting/selection disjoint',not set(pm['training_cases'])&set(pm['checkpoint_selection_cases']))
        check(f'{name} no test fitting dates',not set(pm['training_cases']+pm['checkpoint_selection_cases'])&set(cases[cases.winter==2023].index))
        check(f'{name} prediction file hash',sha(OUT/'radar_predictions'/f'{name}.npy')==pm['prediction_sha256'])
        idx=np.load(OUT/'radar_predictions'/f'{name}_index.npz')
        check(f'{name} prediction IDs agree',idx['case_id'].astype(str).tolist()==pm['prediction_cases'])
        result[name]=digest
    return result


def raw_hrrr_map(case_id,lat,lon,hours):
    """Independent Zarr decode and scipy interpolation from original chunks."""
    import numcodecs
    from pyproj import Proj
    from scipy.interpolate import RegularGridInterpolator
    spec='surface/APCP_1hr_acc_fcst'
    prefix=DATA/'hrrr/raw/sfc'/case_id/(case_id+'_06z_fcst.zarr')/spec
    meta=json.loads((prefix/'.zmetadata').read_text())['metadata']
    def dec(path,m):
        raw=path.read_bytes();raw=numcodecs.get_codec(m['compressor']).decode(raw) if m.get('compressor') else raw
        return np.frombuffer(raw,dtype=m['dtype']).reshape(m['chunks']).astype(float)
    periods=dec(prefix/'forecast_period/0',meta['forecast_period/.zarray'])
    m=meta[spec+'/.zarray'];parts=[]
    for chunk in ['0.3.3','0.4.3']:
        a=dec(prefix/spec/chunk,m);a[a==m['fill_value']]=np.nan
        parts.append(a[(periods>=6)&(periods<6+hours)].sum(0))
    native=np.concatenate(parts,axis=0)
    proj=Proj(a=6371229,b=6371229,proj='lcc',lon_0=262.5,lat_0=38.5,lat_1=38.5,lat_2=38.5)
    lo,la=np.meshgrid(lon,lat);x,y=proj(lo,la)
    xx=(x+2697520.1425219304)/3000;yy=(y+1587306.1525566636)/3000
    interp=RegularGridInterpolator((np.arange(450,750),np.arange(450,600)),native,bounds_error=True)
    return interp(np.c_[yy.ravel(),xx.ravel()]).reshape(64,64)


def audit_radar():
    per=pd.read_csv(OUT/'short_case_metrics.csv',dtype={'case_id':str})
    summary=pd.read_csv(OUT/'short_summary.csv').set_index(['model','hours'])
    check('short duplicate rows',not per.duplicated(['case_id','model','hours']).any())
    for h,sub in per.groupby('hours'):
        sets=[set(g.case_id) for _,g in sub.groupby('model')]
        check(f'short identical cases {h}h',all(s==sets[0] for s in sets))
    for (model,h),g in per.groupby(['model','hours']):
        for key,value in {'cases':len(g),'mae_mm':g.mae.mean(),'rmse_mm':np.sqrt(g.mse.mean()),'crps_mm':g.crps.mean(),
                          'raw_event_brier':g.brier.mean(),'bias_mm':g.bias.mean()}.items():
            close(f'short summary {model} {h}h {key}',summary.loc[(model,h),key],value)
    raw=np.load(DATA/'radar/radar_sequences.npz')
    ids=raw['case_id'].astype(str);lookup={str(cid):i for i,cid in enumerate(ids)}
    rate=raw['rate'];valid=raw['valid_fraction'];rqi=raw['rqi_hourly']
    times=raw['radar_time'].astype('datetime64[s]');qtime=raw['quality_time'].astype('datetime64[s]')
    lat,lon=raw['latitude'],raw['longitude'];iy=np.argmin(abs(lat-40.57607));ix=np.argmin(abs(lon+111.63889))
    neural={};indices={}
    for name in KEYS:
        neural[name]=np.load(OUT/'radar_predictions'/f'{name}.npy',mmap_mode='r')
        index=np.load(OUT/'radar_predictions'/f'{name}_index.npz')['case_id'].astype(str)
        indices[name]={str(cid):i for i,cid in enumerate(index)}
    base=per[per.model=='HRRR'][['case_id','hours']].sort_values(['case_id','hours']).reset_index(drop=True)
    chosen=np.random.default_rng(120626).choice(len(base),size=min(20,len(base)),replace=False)
    table=per.set_index(['case_id','model','hours']);checked=[]
    for position in chosen:
        cid,h=base.iloc[position];cid=str(cid);h=int(h);j=lookup[cid]
        source=rate[j,9:].astype(float)
        qt=np.searchsorted(qtime[j],times[j,9:],side='right')-1
        mask=(valid[j,9:]>=.9)&np.isfinite(rqi[j,qt])&(rqi[j,qt]>=.5)
        mask[:,:4,:]=False;mask[:,-4:,:]=False;mask[:,:,:4]=False;mask[:,:,-4:]=False
        good=mask[1:6*h+2].all(0)&np.isfinite(source[1:6*h+2]).all(0)
        truth=integrate(source,h)
        with np.load(DATA/'hrrr/runs'/(cid+'T0600Z')/'precip_mm.npz') as hz:
            hp=hz['periods'];stored=hz['mrms64'][(hp>=6)&(hp<6+h)].sum(0)
        independent=raw_hrrr_map(cid,lat,lon,h)
        close(f'raw HRRR map reconstruction {cid} {h}h',stored,independent,atol=2e-5,rtol=2e-5)
        forecasts={'HRRR':stored[None]}
        forecasts.update({LABELS[n]:integrate(neural[n][indices[n][cid]],h) for n in KEYS})
        for e in forecasts.values():good&=np.isfinite(e).all(0)
        for model,e in forecasts.items():
            mean=e.mean(0);err=(mean-truth)[good]
            row=table.loc[(cid,model,h)]
            for key,value in {'mae':abs(err).mean(),'mse':np.mean(err**2),'bias':err.mean(),'pixels':good.sum()}.items():
                close(f'raw radar score {cid} {model} {h}h {key}',row[key],value)
            close(f'Alta raw truth {cid} {h}h',row.alta_truth_mm,truth[iy,ix] if good[iy,ix] else np.nan)
            close(f'Alta raw prediction {cid} {model} {h}h',row.alta_predicted_mm,mean[iy,ix])
        checked.append(dict(case_id=cid,hours=h,models=list(forecasts)))
    return dict(sampled_case_horizon_pairs=checked)


def audit_current(hashes):
    check('current forecast exists',(OUT/'current_manifest.json').exists())
    m=json.loads((OUT/'current_manifest.json').read_text())
    issue=pd.Timestamp(m['forecast_issue_utc']);start=pd.Timestamp(m['start_utc']);end=pd.Timestamp(m['end_utc']);run=pd.Timestamp(m['hrrr_run_utc'])
    check('live issue is not future',issue<=pd.Timestamp.now(tz='UTC'))
    check('live HRRR available',run+pd.Timedelta(hours=2)<=issue)
    check('live physical forecast window',start==issue+pd.Timedelta(minutes=10) and end-start==pd.Timedelta(hours=24))
    check('live uses benchmark checkpoint hashes',m.get('checkpoint_sha256')==hashes)
    context=Path(m['radar_context']);context=context if context.is_absolute() else ROOT/context
    z=np.load(context);t=pd.to_datetime(z['radar_time'],utc=True)
    check('live radar is causal',t.max()==issue-pd.Timedelta(minutes=10) and t.min()==issue-pd.Timedelta(minutes=90))
    cmeta=json.loads((context.parent/'manifest.json').read_text())
    check('live source manifest has no future observations',cmeta['future_observations_used'] is False)
    frame=pd.read_csv(OUT/'current_24h_predictions.csv');probability_checks(frame,'mean','current')
    arrays=np.load(OUT/'current_radar_forecasts.npz');curves=pd.read_csv(OUT/'current_short_curves.csv')
    lat,lon=z['latitude'],z['longitude'];iy=np.argmin(abs(lat-40.57607));ix=np.argmin(abs(lon+111.63889))
    input_rate=z['rate'].copy();input_quality=z['rqi'].copy()
    bad=(z['valid_fraction']<.9)|(~np.isfinite(input_quality))|(input_quality<.1)
    input_rate[bad]=np.nan;input_quality[bad]=0
    close('live source input rate',arrays['input_rate'],input_rate)
    close('live source input quality',arrays['input_quality'],input_quality)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        local=np.nanmean(input_rate[0,:,iy-1:iy+2,ix-1:ix+2],axis=(1,2))
        common={'radar_current_rate':float(local[-1]),'radar_history_mean':float(np.nanmean(local)),
                'radar_history_change':float(local[-1]-local[0]),
                'radar_input_quality':float(input_quality[0,:,iy-1:iy+2,ix-1:ix+2].mean())}
    phase=2*np.pi*(start.dayofyear-1)/365.2425
    common.update(calendar_sin=np.sin(phase),calendar_cos=np.cos(phase))
    for name in KEYS:
        e=arrays[name][:,1:20,iy-1:iy+2,ix-1:ix+2].astype(float).mean((2,3))
        cumulative=np.c_[np.zeros(len(e)),np.cumsum((e[:,:-1]+e[:,1:])/12,axis=1)]
        rows=curves[curves.model==name].sort_values('hours')
        close(f'live curve mean {name}',rows.mean_liquid_mm,cumulative.mean(0))
        close(f'live curve q10 {name}',rows.q10_liquid_mm,np.quantile(cumulative,.1,axis=0))
        close(f'live curve q90 {name}',rows.q90_liquid_mm,np.quantile(cumulative,.9,axis=0))
        for hours in [1,2,3]:
            acc=integrate(arrays[name],hours)
            station=acc[:,iy-1:iy+2,ix-1:ix+2].mean((1,2))
            common[f'{name}__radar_liquid_{hours}h_mean']=float(station.mean())
            if hours==3:
                common[f'{name}__radar_liquid_3h_std']=float(station.std())
                common[f'{name}__radar_liquid_3h_q90']=float(np.quantile(station,.9))
                region=acc[:,iy-8:iy+9,ix-8:ix+9]
                common[f'{name}__radar_liquid_3h_region_mean']=float(region.mean())
                common[f'{name}__radar_liquid_3h_region_max']=float(region.mean(0).max())
    variables=['precip_mm','snow_water_mm','temperature_k','dewpoint_k','wind_u_ms','wind_v_ms','relative_humidity_pct']
    for variable in variables:
        path=DATA/'hrrr/runs'/run.strftime('%Y%m%dT%H%MZ')/(variable+'.npz')
        h=np.load(path);ends=run+pd.to_timedelta(h['periods'],unit='h');begins=ends-pd.Timedelta(hours=1)
        weights=np.asarray([max(0.,(min(b,end)-max(a,start)).total_seconds())/3600 for a,b in zip(begins,ends)])
        check(f'live HRRR full24h {variable}',np.isclose(weights.sum(),24))
        selected=weights>0;check(f'live HRRR finite selected values {variable}',np.isfinite(h['mean3x3'][selected]).all())
        if variable in variables[:2]:
            key='hrrr_24h_liquid_mm' if variable=='precip_mm' else 'hrrr_24h_snow_water_mm'
            close(f'live source amount {variable}',m[key],np.dot(weights[selected],h['mean3x3'][selected]))
        hourly=[]
        for j in range(24):
            lo=start+pd.Timedelta(hours=j);hi=lo+pd.Timedelta(hours=1)
            w=np.asarray([max(0.,(min(b,hi)-max(a,lo)).total_seconds())/3600 for a,b in zip(begins,ends)])
            hourly.append(float(np.dot(w[w>0],h['mean3x3'][w>0])))
        for hours in [3,12,24]:
            vals=np.asarray(hourly[:hours])
            if variable in variables[:2]:common[f'{variable}_sum_{hours}h']=float(vals.sum())
            else:
                common[f'{variable}_mean_{hours}h']=float(vals.mean())
                if hours==24:
                    common[f'{variable}_min_24h']=float(vals.min())
                    common[f'{variable}_max_24h']=float(vals.max())
    fits=joblib.load(OUT/'daily_calibrators.joblib')['new_models']
    for model,bundle in fits.items():
        row=frame[frame.model==model].iloc[0]
        x=[common[c] for c in bundle['columns']]
        replay=replay_forest(bundle['estimator'],x)
        for key,value in replay.items():close(f'current distribution replay {model} {key}',row[key],value)
    return dict(available=True,issue_utc=issue.isoformat(),hrrr_run_utc=run.isoformat())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-test-unblinding',action='store_true')
    args=parser.parse_args()
    if not args.allow_test_unblinding:
        parser.error('Root must explicitly release the test results before --allow-test-unblinding is used.')
    details={};error=None
    try:
        hashes=audit_model_identity()
        details['daily']=audit_daily()
        details['radar']=audit_radar()
        details['current']=audit_current(hashes)
    except Exception as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
    payload={'generated_utc':pd.Timestamp.now(tz='UTC').isoformat(),'passed':error is None and all(c['passed'] for c in CHECKS),
             'checks':CHECKS,'details':details,'error':error,'script_sha256':sha(__file__)}
    (OUT/'final_independent_audit.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps({'passed':payload['passed'],'checks':len(CHECKS),'error':error,'details':details},indent=2))
    if not payload['passed']:raise SystemExit(1)


if __name__=='__main__':main()
