"""Test richer observed Dillon weather for Copper monthly snowfall predictions.

Same eligible seasons for coarse and daily-derived inputs at each origin. No
observations from future months determine eligibility or enter query features.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from monthly import ROOT,MONTHS,SEED,load_training

SUFFIXES=['snow_total_inches','snow_days','max_daily_snow_inches','mean_tmax_celsius',
          'mean_tmin_celsius','precipitation_total_mm','freezing_days','late7_snow_inches']


def features(origin,mode):
    suffixes=SUFFIXES if mode=='daily_context' else ['snow_total_inches']
    return [f'{m}_{s}' for m in MONTHS[:origin+1] for s in suffixes]


def xy(frame,origin,mode):
    X=frame[features(origin,mode)].to_numpy(float)
    enso=frame[f'enso_{origin}'].to_numpy(float)
    return np.column_stack([X,enso,(enso>=1.5).astype(float),(enso<=-1.5).astype(float)])


def configurations():
    choices=[{'family':'climatology','inputs':'none'}]
    for mode in ['coarse_monthly','daily_context']:
        choices += [{'family':'ridge','inputs':mode,'alpha':a} for a in [10.,100.,1000.]]
        choices += [{'family':'analog','inputs':mode,'bandwidth':h} for h in [1.,3.]]
        choices += [{'family':'extra_trees','inputs':mode,'min_leaf':n} for n in [3,5]]
    return choices


def fit_predict(train,test,origin,config):
    Y=train[MONTHS[origin+1:]].to_numpy(float)
    if config['family']=='climatology':return np.tile(Y.mean(0),(len(test),1))
    X=xy(train,origin,config['inputs']);query=xy(test,origin,config['inputs'])
    mean=X.mean(0);scale=X.std(0);scale[scale<1e-8]=1.
    X=(X-mean)/scale;query=(query-mean)/scale
    if not np.isfinite(X).all() or not np.isfinite(query).all():raise ValueError('Incomplete observed context')
    if config['family']=='analog':
        distance=np.mean((query[:,None,:]-X[None,:,:])**2,axis=2)
        logs=-distance/(2*config['bandwidth']**2)
        weights=np.exp(logs-logs.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
        return (.8*weights+.2/len(train))@Y
    ym=Y.mean(0);ys=Y.std(0);ys[ys<1e-8]=1.
    Z=(Y-ym)/ys
    model=Ridge(alpha=config['alpha']) if config['family']=='ridge' else ExtraTreesRegressor(
        n_estimators=32,min_samples_leaf=config['min_leaf'],random_state=SEED,n_jobs=1)
    model.fit(X,Z.ravel() if Y.shape[1]==1 else Z)
    p=np.asarray(model.predict(query)).reshape(len(test),Y.shape[1])
    return np.maximum(0.,p*ys+ym)


def tune(train,origin):
    splits=list(KFold(n_splits=4,shuffle=True,random_state=SEED).split(train))
    scored=[]
    for config in configurations():
        errors=[]
        for a,b in splits:
            p=fit_predict(train.iloc[a],train.iloc[b],origin,config)
            errors.extend(((p-train.iloc[b][MONTHS[origin+1:]].to_numpy(float))**2).ravel())
        scored.append({'config':config,'inner_rmse':float(np.sqrt(np.mean(errors)))})
    result={'selected':min(scored,key=lambda v:v['inner_rmse']),
            'climatology':scored[0]}
    for mode in ['coarse_monthly','daily_context']:
        result[mode]=min([v for v in scored if v['config']['inputs']==mode],key=lambda v:v['inner_rmse'])
    return result


def outer(args):
    frame,origin,held=args
    train=frame.drop(index=held).reset_index(drop=True);test=frame.iloc[held:held+1]
    with threadpool_limits(limits=1):
        choices=tune(train,origin);rows=[]
        for name,choice in choices.items():
            p=fit_predict(train,test,origin,choice['config'])[0]
            for j,month in enumerate(MONTHS[origin+1:]):
                rows.append({'start_year':int(test.start_year.iloc[0]),'season':test.season.iloc[0],
                    'origin':origin,'target_month':month,'horizon':j+1,'comparison':name,
                    'actual':float(test[month].iloc[0]),'prediction':float(p[j]),
                    'chosen_config':json.dumps(choice['config'],sort_keys=True),'n_train':len(train)})
    return rows


def summarize(predictions):
    rows=[]
    for scope,part in [('initial_six_months',predictions.loc[predictions.origin==0]),
                       ('updating_next_month',predictions.loc[predictions.horizon==1])]:
        baseline=part.loc[part.comparison=='climatology'].set_index(['start_year','origin','target_month'])
        for name,g in part.groupby('comparison'):
            g=g.set_index(['start_year','origin','target_month']).sort_index()
            error=g.prediction-g.actual
            base=baseline.loc[g.index].prediction-baseline.loc[g.index].actual
            gains=abs(base)-abs(error)
            season_blocks=gains.groupby(level='start_year').agg(['sum','count']).to_numpy()
            rng=np.random.default_rng(SEED)
            draws=season_blocks[rng.integers(len(season_blocks),size=(10000,len(season_blocks)))].sum(1)
            boot=draws[:,0]/draws[:,1]
            rows.append({'scope':scope,'comparison':name,'n_seasons':len(season_blocks),'n_predictions':len(g),
                'mae_inches':float(abs(error).mean()),'rmse_inches':float(np.sqrt(np.mean(error**2))),
                'mean_mae_improvement_vs_matched_baseline':float(gains.mean()),
                'season_bootstrap_improvement_ci95':np.quantile(boot,[.025,.975]).tolist()})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    source=ROOT/'data/dillon_season_daily_features.csv'
    daily=pd.read_csv(source)
    frame=load_training('copper').merge(daily.drop(columns=['season'],errors='ignore'),on='start_year',validate='one_to_one')
    cohorts={i:frame.dropna(subset=features(i,'daily_context')).reset_index(drop=True) for i in range(6)}
    for i,d in cohorts.items():
        if len(d)<8:raise ValueError(f'Too few complete observed contexts at origin{i}')
        print('Observed through',MONTHS[i],'matched seasons',len(d),flush=True)
    tasks=[(d,i,j) for i,d in cohorts.items() for j in range(len(d))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=[row for chunk in pool.map(outer,tasks) for row in chunk]
    predictions=pd.DataFrame(rows);predictions.to_csv(ROOT/'outputs/monthly_daily_context_predictions.csv',index=False)
    final={}
    with threadpool_limits(limits=1):
        for i,d in cohorts.items():final[str(i)]=tune(d,i)
    current=daily.loc[daily.start_year==2026].copy();current['enso_0']=1.7
    forecasts={}
    for name,choice in final['0'].items():
        p=fit_predict(cohorts[0],current,0,choice['config'])[0]
        forecasts[name]={'config':choice['config'],'monthly_inches':dict(zip(MONTHS[1:],p.tolist())),
                         'nov_apr_total_inches':float(p.sum())}
    result={'as_of':'2026-11-03','target_site':'Copper resort','observed_feature_site':'Dillon1E',
        'hypothesis':'Do daily-derived observations add predictive value beyond observed regional monthly snowfall and same ENSO information?',
        'evaluation':'Whole-season nestedLOSO; inner4-fold; matched complete observed-prefix cohorts; optimize all remaining-month RMSE. Seasons exchangeable.',
        'cohorts':{str(i):{'observed_through':MONTHS[i],'n':len(d),'years':d.start_year.astype(int).tolist()} for i,d in cohorts.items()},
        'metrics':summarize(predictions),'final_choices':final,'current_copper_forecasts_from_dillon_inputs':forecasts,
        'current_observation':{'Dillon_october_snow_inches':float(current.oct_snow_total_inches.iloc[0]),'Copper_october_snow_inches':None},
        'limits':['No measured A-Basin/Keystone resort target labels; no resort transfer skill claim.',
                  'Many correlated daily-derived features but only roughly20 seasonal samples.',
                  'No future observed weather or future-month completeness used in features/cohort selection.',
                  'Current laggedRONI1.7 exceeds primaryCopperhistorical range; no calibrated current intervals added.'],
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (ROOT/'outputs/monthly_daily_context_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result['metrics'],indent=2),flush=True)


if __name__=='__main__':main()
