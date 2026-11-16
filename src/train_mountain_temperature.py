"""Honest hourly temperature refinement, distinct from fresh-snow verification.

Whole winters are held apart. Selection uses validation RMSE only; paired test
uncertainty resamples complete forecast issue dates, not correlated hours.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/mountain-temperature'
WEATHER = ['temperature_k','dewpoint_k','relative_humidity_pct','wind_u_ms','wind_v_ms','precip_mm']
FEATURES = WEATHER + ['forecast_hour','hours_after_issue','recent_bias_c','recent_bias_age_h',
    'bias_decay_3h','bias_decay_12h','bias_decay_48h','recent_temperature_mean_6h',
    'recent_temperature_trend_c','temperature_observation_count_6h','sin_year','cos_year','sin_hour','cos_hour']


def make_features(weather, observations, site='vail_snotel', station='842:CO:SNTL', issue=None):
    w = weather[weather.site.eq(site)].copy()
    for c in ['run_time_utc','valid_time_utc','assumed_available_time_utc']:
        w[c] = pd.to_datetime(w[c],utc=True)
    obs = observations[observations.station_id.eq(station)].copy()
    for c in ['valid_time_utc','assumed_available_time_utc']:
        obs[c] = pd.to_datetime(obs[c],utc=True)
    obs = obs.sort_values('valid_time_utc').drop_duplicates('valid_time_utc',keep='last')
    truth = obs.set_index('valid_time_utc').temperature_c
    parts = []
    for run,g in w.groupby('run_time_utc'):
        issued = pd.Timestamp(issue) if issue is not None else run+pd.Timedelta(hours=4,minutes=50)
        issued = issued.tz_localize('UTC') if issued.tzinfo is None else issued.tz_convert('UTC')
        if g.assumed_available_time_utc.max() > issued:
            continue
        known = obs[(obs.assumed_available_time_utc<=issued) &
                    (obs.valid_time_utc>issued-pd.Timedelta(hours=25)) & obs.temperature_c.notna()]
        joined = known.merge(g[['valid_time_utc','temperature_k']],on='valid_time_utc')
        if len(joined):
            last = joined.sort_values('valid_time_utc').iloc[-1]
            bias = float(last.temperature_c-(last.temperature_k-273.15))
            age = (issued-last.valid_time_utc).total_seconds()/3600
        else:
            bias,age = np.nan,25.
        recent = known[known.valid_time_utc>issued-pd.Timedelta(hours=7)]
        g = g[g.valid_time_utc>issued].copy()
        g['issue_utc'] = issued
        g['hours_after_issue'] = (g.valid_time_utc-issued).dt.total_seconds()/3600
        g['recent_bias_c'],g['recent_bias_age_h'] = bias,age
        for tau in [3,12,48]:
            g[f'bias_decay_{tau}h'] = bias*np.exp(-g.hours_after_issue/tau)
        g['recent_temperature_mean_6h'] = recent.temperature_c.mean()
        g['recent_temperature_trend_c'] = (recent.temperature_c.iloc[-1]-recent.temperature_c.iloc[0]) if len(recent)>1 else np.nan
        g['temperature_observation_count_6h'] = len(recent)
        phase = 2*np.pi*(g.valid_time_utc.dt.dayofyear-1)/365.2425
        hour = 2*np.pi*g.valid_time_utc.dt.hour/24
        g['sin_year'],g['cos_year'] = np.sin(phase),np.cos(phase)
        g['sin_hour'],g['cos_hour'] = np.sin(hour),np.cos(hour)
        g['observed_temperature_c'] = g.valid_time_utc.map(truth)
        g['raw_hrrr_temperature_c'] = g.temperature_k-273.15
        g['winter'] = np.where(run.month>=7,run.year,run.year-1)
        parts.append(g)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def bias_prediction(d,scale,tau):
    return d.raw_hrrr_temperature_c.to_numpy()+scale*d.recent_bias_c.fillna(0).to_numpy()*np.exp(-d.hours_after_issue.to_numpy()/tau)


def score(y,p):
    return dict(rmse_c=float(np.sqrt(np.mean((y-p)**2))),mae_c=float(np.mean(abs(y-p))),bias_c=float(np.mean(p-y)))


def predict_temperature(d,artifact):
    chosen = artifact['selected_name']
    if chosen == 'Raw HRRR':
        return d.raw_hrrr_temperature_c.to_numpy()
    if chosen == 'Recent station bias':
        return bias_prediction(d,**artifact['bias_params'])
    return d.raw_hrrr_temperature_c.to_numpy()+artifact['model'].predict(d[artifact['features']])


def paired_bootstrap(d,pred,seed=712,iterations=2000):
    z = pd.DataFrame({'issue':d.issue_utc,'raw_sq':(d.raw_hrrr_temperature_c-d.observed_temperature_c)**2,
                      'model_sq':(pred-d.observed_temperature_c)**2}).groupby('issue').agg(['sum','count'])
    a=z['raw_sq']['sum'].to_numpy();b=z['model_sq']['sum'].to_numpy();n=z['raw_sq']['count'].to_numpy()
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(z),(iterations,len(z)))
    delta=np.sqrt(a[indices].sum(1)/n[indices].sum(1))-np.sqrt(b[indices].sum(1)/n[indices].sum(1))
    return dict(raw_minus_model_rmse_c=float(np.sqrt(a.sum()/n.sum())-np.sqrt(b.sum()/n.sum())),
                confidence_95_low=float(np.quantile(delta,.025)),confidence_95_high=float(np.quantile(delta,.975)),
                bootstrap_unit='whole forecast issue date',issue_dates=len(z))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--site',default='vail_snotel');ap.add_argument('--station',default='842:CO:SNTL');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    hist=ROOT/'data/hrrr_portfolio/historical_hourly.csv.gz'
    if hist.exists():
        weather=pd.read_csv(hist)
    else:
        files=sorted((ROOT/'data/hrrr_portfolio/runs').glob('*/hourly.csv'))
        weather=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    weather=weather.drop_duplicates(['site','run_time_utc','valid_time_utc'])
    obs=pd.read_csv(ROOT/'data/mountain_hourly/observations.csv.gz',low_memory=False)
    d=make_features(weather,obs,args.site,args.station)
    d=d[d.observed_temperature_c.notna() & d.raw_hrrr_temperature_c.notna() & d.winter.between(2020,2023)].copy()
    d['split']=np.where(d.winter<=2021,'train',np.where(d.winter==2022,'validation','test'))
    tr,va,te=[d[d.split.eq(s)].copy() for s in ['train','validation','test']]
    if min(len(tr),len(va),len(te))==0:
        raise RuntimeError('All whole-winter train/validation/test partitions must exist before selection')
    train_y=tr.observed_temperature_c.to_numpy();val_y=va.observed_temperature_c.to_numpy()
    residual=train_y-tr.raw_hrrr_temperature_c.to_numpy()
    candidates={'Raw HRRR':dict(model=None,pred=lambda f:f.raw_hrrr_temperature_c.to_numpy())}
    records=[]
    best=None
    for scale in [.25,.5,.75,1.,1.25]:
        for tau in [3.,6.,12.,24.,48.,1e12]:
            rmse=score(val_y,bias_prediction(va,scale,tau))['rmse_c']
            records.append(dict(family='Recent station bias',scale=scale,tau=tau,validation_rmse_c=rmse))
            if best is None or rmse<best[0]:best=(rmse,dict(scale=scale,tau=tau))
    params=best[1]
    candidates['Recent station bias']=dict(model=None,pred=lambda f:bias_prediction(f,**params))
    for family in ['Weather-only ridge','Ridge correction','Weather-only boosted trees','Boosted tree correction']:
        bestfit=None
        ridge = 'ridge' in family.lower()
        cols = WEATHER+['forecast_hour','hours_after_issue','sin_year','cos_year','sin_hour','cos_hour'] if family.startswith('Weather-only') else FEATURES
        settings=([1.,10.,100.,1000.] if ridge else [2,3,5])
        for setting in settings:
            model=(make_pipeline(SimpleImputer(add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=setting))
                if ridge else HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,max_depth=setting,min_samples_leaf=80,l2_regularization=10,early_stopping=False,random_state=712))
            model.fit(tr[cols],residual)
            pred=va.raw_hrrr_temperature_c.to_numpy()+model.predict(va[cols])
            rmse=score(val_y,pred)['rmse_c']
            records.append(dict(family=family,setting=setting,validation_rmse_c=rmse))
            if bestfit is None or rmse<bestfit[0]:bestfit=(rmse,model,setting)
        model=bestfit[1]
        candidates[family]=dict(model=model,setting=bestfit[2],features=cols,pred=lambda f,m=model,cc=cols:f.raw_hrrr_temperature_c.to_numpy()+m.predict(f[cc]))
    selected=min(candidates,key=lambda name:score(val_y,candidates[name]['pred'](va))['rmse_c'])
    metrics=[];prediction_rows=[]
    for name,c in candidates.items():
        for split,f in [('train',tr),('validation',va),('test',te)]:
            pred=c['pred'](f);y=f.observed_temperature_c.to_numpy()
            metrics.append(dict(model=name,split=split,rows=len(f),issue_dates=f.issue_utc.nunique(),selected=name==selected,**score(y,pred)))
            z=f[['issue_utc','valid_time_utc','forecast_hour','hours_after_issue','observed_temperature_c','raw_hrrr_temperature_c','recent_bias_c']].copy()
            z['split'],z['model'],z['predicted_temperature_c']=split,name,pred
            prediction_rows.append(z)
    allpred=pd.concat(prediction_rows,ignore_index=True)
    pd.DataFrame(metrics).to_csv(OUT/f'{args.site}_metrics.csv',index=False)
    pd.DataFrame(records).to_csv(OUT/f'{args.site}_validation_search.csv',index=False)
    allpred.to_csv(OUT/f'{args.site}_predictions.csv.gz',index=False)
    tested_bootstrap=paired_bootstrap(te,candidates[selected]['pred'](te))
    artifact=dict(selected_name=selected,model=candidates[selected]['model'],bias_params=params,features=candidates[selected].get('features',FEATURES),
        station_id=args.station,site=args.site,scope='hourly station temperature Celsius; not validated snowfall correction',fit_years=[2020,2021])
    # Preserve the exact test-scored model; operational refit is separate and labelled.
    joblib.dump(artifact,OUT/f'{args.site}_tested.joblib')
    if artifact['model'] is not None:
        fit=pd.concat([tr,va]);artifact['model'].fit(fit[artifact['features']],fit.observed_temperature_c-fit.raw_hrrr_temperature_c)
        artifact['fit_years']=[2020,2021,2022]
    joblib.dump(artifact,OUT/f'{args.site}_operational.joblib')
    current=pd.read_csv(ROOT/'data/hrrr_portfolio/current_weather.csv')
    # Use actual current clock, never all observations after a historical issue.
    issue=pd.Timestamp.now(tz='UTC')
    live=make_features(current,obs,args.site,args.station,issue)
    if len(live):
        live['corrected_temperature_c']=predict_temperature(live,artifact)
        live.to_csv(OUT/f'{args.site}_current.csv',index=False)
    audit=dict(selected_by='lowest2022/23validationRMSE; no2023/24test-basedmodelselection',selected=selected,
        train_winters=[2020,2021],validation_winter=2022,test_winter=2023,
        metrics=metrics,paired_test_rmse=tested_bootstrap,
        feature_receipts='Only actual observation timestamps + assumed1h latency <=issue. Final historical QC is not an as-issued snapshot.',
        evaluation_clock='06UTC HRRR run, available09UTC assumed; issued10:50UTC. Only valid endpoints afterissue, f5..48 where available; early HRRR runs endf36.',
        target='Actual station hourly temperature, not snowfall, not resort-wide weather',
        scope='Any snowfall phase improvement from this correction remains unvalidated; station and resort gridpoint elevations differ.',
        station=args.station,site=args.site,bias_params=params,current_issue_utc=str(issue))
    audit['test_reuse_warning']='2023/24 is held out from this fit and hyperparameter choice but reused in prior project experiments; prospective validation is still needed.'
    if args.station=='FEMC_MMW':
        audit['quality_policy']='FEMC temperature requires valid companionRH. Entire documented faulty CR1000X era20Aug2020–26Feb2021 quarantined until26Feb replacement; no model-error threshold used.'
        audit['quality_source']='https://www.uvm.edu/femc/data/archive/project/mt-mansfield-west-slope-mid-elevation/dataset/mansfield-west-met-site/metadata'
    (OUT/f'{args.site}_audit.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(metrics).to_string(index=False));print(json.dumps(audit['paired_test_rmse'],indent=2))


if __name__=='__main__':main()
