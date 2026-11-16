"""Season-conditioned monthly snowfall, with nested whole-season validation.

Seasons are exchangeable, as requested. Each origin observes a prefix and predicts
all remaining months directly. Never substitute held-out future observations into
an earlier-origin forecast. Current unknown Copper October has its own experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
MONTHS = ['oct','nov','dec','jan','feb','mar','apr']
LAG_PERIODS = ['JAS','ASO','SON','OND','NDJ','DJF']
SEED = 20261103


def load_training(dataset='copper'):
    if dataset == 'copper':
        frame = pd.read_csv(ROOT/'data/copper_monthly_user.csv')
        frame = frame.loc[frame.start_year != 2025].copy()
    else:
        frame = pd.read_csv(ROOT/'data/ghcn_monthly_wide.csv')
        ids = {'dillon':'USC00052281','climax':'USC00051660'}
        frame = frame.loc[(frame.station_id == ids[dataset]) &
                          (frame.start_year <= 2025) & (frame.complete_oct_apr == 1)].copy()
    frame = frame.sort_values('start_year').reset_index(drop=True)
    if frame.start_year.duplicated().any() or frame[MONTHS].isna().any().any():
        raise ValueError('Need unique seasons and complete primary monthly labels')
    roni = pd.read_csv(ROOT/'data/roni_noaa.csv').set_index('year')
    for origin, period in enumerate(LAG_PERIODS):
        years = frame.start_year + (1 if origin == 5 else 0)
        frame[f'enso_{origin}'] = years.map(roni[period])
    if frame[[f'enso_{i}' for i in range(6)]].isna().any().any():
        raise ValueError('Missing issue-time ENSO feature')
    return frame


def make_features(frame, origin, include_enso, snow_context=True):
    names = MONTHS[:origin+1] if snow_context else []
    X = frame[names].to_numpy(float)
    if include_enso:
        e = frame[f'enso_{origin}'].to_numpy(float)
        X = np.column_stack([X,e,(e>=1.5).astype(float),(e<=-1.5).astype(float)])
        names = names + ['lagged_roni','lagged_strong_warm','lagged_strong_cold']
    if not np.isfinite(X).all():
        raise ValueError('Missing/nonfinite observed feature, do not fill unknown months with zero')
    return X,names


def candidates(snow_context=True):
    result = [{'family':'climatology','enso':False}]
    enso_options = [False,True] if snow_context else [True]
    for enso in enso_options:
        result += [{'family':'ridge','enso':enso,'alpha':a} for a in [1.,10.,100.]]
        result += [{'family':'analog','enso':enso,'bandwidth':h,'uniform_mix':.2} for h in [.5,1.5,4.]]
        result += [{'family':'extra_trees','enso':enso,'min_samples_leaf':k,'n_estimators':32} for k in [3,6]]
        result += [{'family':'neural_mlp','enso':enso,'hidden':4,'alpha':a} for a in [1.,20.]]
    return result


def fit_predict(train, test, origin, config, snow_context=True):
    """All preprocessing is fitted on train only. Return nonnegative mean estimates."""
    Y = train[MONTHS[origin+1:]].to_numpy(float)
    n_outputs = Y.shape[1]
    if config['family'] == 'climatology':
        return np.tile(Y.mean(axis=0),(len(test),1))
    X,_ = make_features(train,origin,config['enso'],snow_context)
    Xtest,_ = make_features(test,origin,config['enso'],snow_context)
    mean,scale = X.mean(axis=0),X.std(axis=0)
    scale = np.where(scale > 1e-8,scale,1.)
    X, Xtest = (X-mean)/scale, (Xtest-mean)/scale
    ymean,yscale = Y.mean(axis=0),Y.std(axis=0)
    yscale = np.where(yscale > 1e-8,yscale,1.)
    Z = (Y-ymean)/yscale
    family = config['family']
    if family == 'analog':
        distance = np.mean((Xtest[:,None,:]-X[None,:,:])**2,axis=2)
        log_weights = -distance/(2*config['bandwidth']**2)
        weights = np.exp(log_weights-log_weights.max(axis=1,keepdims=True))
        weights /= weights.sum(axis=1,keepdims=True)
        mix = config['uniform_mix']
        weights = (1-mix)*weights + mix/len(train)
        return weights @ Y
    if family == 'ridge':
        estimator = Ridge(alpha=config['alpha'])
    elif family == 'extra_trees':
        estimator = ExtraTreesRegressor(n_estimators=config['n_estimators'],
            min_samples_leaf=config['min_samples_leaf'],random_state=SEED,n_jobs=1)
    elif family == 'neural_mlp':
        estimator = MLPRegressor(hidden_layer_sizes=(config['hidden'],),activation='tanh',
            solver='lbfgs',alpha=config['alpha'],max_iter=250,max_fun=5000,random_state=SEED,tol=1e-5)
    else:
        raise ValueError(f'Unknown family: {family}')
    # scikit-learn expects a vector for one output; all interfaces return a matrix.
    fit_y = Z.ravel() if n_outputs == 1 else Z
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        estimator.fit(X,fit_y)
    predicted = np.asarray(estimator.predict(Xtest)).reshape(len(test),n_outputs)
    return np.maximum(0.,predicted*yscale+ymean)


def tune(train, origin, snow_context=True):
    """Inner 4-fold CV contains only the outer training seasons."""
    splits = list(KFold(n_splits=4,shuffle=True,random_state=SEED).split(train))
    scored=[]
    for config in candidates(snow_context):
        squared=[]
        for a,b in splits:
            pred=fit_predict(train.iloc[a],train.iloc[b],origin,config,snow_context)
            y=train.iloc[b][MONTHS[origin+1:]].to_numpy(float)
            squared.extend(((pred-y)**2).ravel())
        scored.append({'config':config,'inner_rmse':float(np.sqrt(np.mean(squared)))})
    # Deterministic tie preference follows candidate order, with climatology first.
    selected=min(scored,key=lambda row:row['inner_rmse'])
    family_best={}
    for row in scored:
        key=row['config']['family']
        if key not in family_best or row['inner_rmse'] < family_best[key]['inner_rmse']:
            family_best[key]=row
    return selected,family_best


def outer_fold(args):
    frame,origin,holdout,snow_context,live_rows=args
    with threadpool_limits(limits=1):
        train=frame.drop(index=holdout).reset_index(drop=True)
        test=frame.iloc[holdout:holdout+1]
        selected,family_best=tune(train,origin,snow_context)
        rows=[]
        for name,choice in {'selected':selected,**family_best}.items():
            p=fit_predict(train,test,origin,choice['config'],snow_context)[0]
            for j,month in enumerate(MONTHS[origin+1:]):
                rows.append({'season':str(test.season.iloc[0]),'start_year':int(test.start_year.iloc[0]),
                    'origin':origin,'observed_through':MONTHS[origin],'target_month':month,
                    'horizon':j+1,'family':name,'chosen_family':choice['config']['family'],
                    'enso_included':choice['config']['enso'],'actual':float(test[month].iloc[0]),
                    'prediction':float(p[j]),'inner_rmse':choice['inner_rmse'],
                    'config':json.dumps(choice['config'],sort_keys=True)})
        live=[]
        if live_rows is not None:
            p=fit_predict(train,live_rows,origin,selected['config'],snow_context)
            held=fit_predict(train,test,origin,selected['config'],snow_context)[0]
            y=test[MONTHS[origin+1:]].to_numpy(float)[0]
            for k in range(len(live_rows)):
                live.append({'scenario':str(live_rows.scenario.iloc[k]),'holdout':holdout,
                             'prediction':p[k].tolist(),'held_prediction':held.tolist(),
                             'held_actual':y.tolist()})
        return {'predictions':rows,'live':live,'selection':{'origin':origin,'held_out_season':str(test.season.iloc[0]),
                'training_years':train.start_year.astype(int).tolist(),**selected}}


def metrics(predictions):
    table=[]
    for scope,subset in [('all_origins_all_horizons',predictions),
                         ('sequential_next_month',predictions.loc[predictions.horizon==1]),
                         ('october_only_all_six',predictions.loc[predictions.origin==0])]:
        base=subset.loc[subset.family=='climatology'].sort_values(['start_year','origin','horizon'])
        for name,g in subset.groupby('family',sort=False):
            g=g.sort_values(['start_year','origin','horizon'])
            errors=g.prediction-g.actual
            baseline_error=base.prediction.to_numpy()-base.actual.to_numpy()
            improvement=abs(baseline_error)-abs(errors.to_numpy())
            # Resample independent seasons, retaining all monthly errors within each.
            by_year=pd.DataFrame({'year':g.start_year.to_numpy(),'improvement':improvement}).groupby('year').improvement.mean().to_numpy()
            rng=np.random.default_rng(SEED)
            boot=by_year[rng.integers(len(by_year),size=(10000,len(by_year)))].mean(axis=1)
            table.append({'scope':scope,'family':name,'n_seasons':len(by_year),'n_forecasts':len(g),
                          'mae_inches':float(np.mean(abs(errors))),
                          'rmse_inches':float(np.sqrt(np.mean(errors**2))),
                          'mean_mae_improvement_inches':float(np.mean(improvement)),
                          'mae_improvement_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist()})
    return table


def jackknife_bounds(rows, target_months):
    """Finite-rank jackknife+ bands from nested omitted-season models, clipped at 0.

    Reported as exploratory under exchangeable seasons, not climate-robust or
    simultaneous six-month coverage. Total uses season-total residuals directly.
    """
    pred=np.array([r['prediction'] for r in rows])
    err=np.array([r['held_actual'] for r in rows])-np.array([r['held_prediction'] for r in rows])
    n=len(rows)
    if n < 9:
        raise ValueError('At least nine omitted-season fits are required for these finite alpha=.10 bounds')
    lo_index=max(0,int(math.floor((n+1)*.1))-1)
    hi_index=min(n-1,int(math.ceil((n+1)*.9))-1)
    lower=np.maximum(0,np.sort(pred-abs(err),axis=0)[lo_index])
    upper=np.maximum(0,np.sort(pred+abs(err),axis=0)[hi_index])
    total_pred=pred.sum(axis=1)
    total_error=abs(err.sum(axis=1))
    total=[max(0.,float(np.sort(total_pred-total_error)[lo_index])),
           max(0.,float(np.sort(total_pred+total_error)[hi_index]))]
    return {'monthly_lower':dict(zip(target_months,lower.tolist())),
            'monthly_upper':dict(zip(target_months,upper.tolist())),
            'remaining_total_interval':total,'calibration_seasons':n,
            'jackknife_alpha':.10,'nominal_coverage':.90,
            'method':'Nested omitted-season jackknife+ finite-rank range, alpha=.10 (nominal90%; theoretical lower bound80% only under required exchangeability/symmetry assumptions). Exploratory marginal ranges, not simultaneous or empirically calibrated coverage.'}


def run_experiment(frame,label,origins,snow_context,live_rows,workers):
    tasks=[(frame,origin,i,snow_context,live_rows if origin==0 else None) for origin in origins for i in range(len(frame))]
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(outer_fold,tasks)):
            results.append(result)
            if (i+1)%len(frame)==0:
                print(f'{label}: completed origin {(i+1)//len(frame)}/{len(origins)}',flush=True)
    predictions=pd.DataFrame([v for r in results for v in r['predictions']])
    predictions.to_csv(OUT/f'{label}_predictions.csv',index=False)
    final_models={}
    for origin in origins:
        with threadpool_limits(limits=1):
            selection,families=tune(frame,origin,snow_context)
        final_models[str(origin)]={'selection':selection,'family_choices':families}
    forecasts={}
    if live_rows is not None:
        selected=final_models['0']['selection']
        with threadpool_limits(limits=1):
            p=fit_predict(frame,live_rows,0,selected['config'],snow_context)
        all_live=[v for r in results for v in r['live']]
        for j,row in live_rows.iterrows():
            folded=[r for r in all_live if r['scenario']==row.scenario]
            forecasts[row.scenario]={'point_inches':dict(zip(MONTHS[1:],p[j].tolist())),
                'nov_apr_total_inches':float(p[j].sum()),'selected_config':selected['config'],
                **jackknife_bounds(folded,MONTHS[1:])}
    result={'label':label,'n_seasons':len(frame),'years':frame.start_year.astype(int).tolist(),
            'snow_context_used':snow_context,'origin_indices':origins,
            'evaluation':'Nested leave-one-season-out; inner shuffled 4-fold season CV; outer whole-season exclusion; years assumed exchangeable.',
            'tuning_objective':'RMSE across all remaining months at each origin; target scaling fitted inside training folds.',
            'metrics':metrics(predictions),'final_models':final_models,'forecasts':forecasts,
            'selections':[r['selection'] for r in results]}
    (OUT/f'{label}_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def live_frame(october_values,enso):
    return pd.DataFrame([{'scenario':name,'start_year':2026,'season':'2026/2027','oct':value,
                          **{f'enso_{i}':enso for i in range(6)}} for name,value in october_values.items()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--experiment',choices=['copper','copper_missing_october','copper_regional_october','dillon','all'],default='all')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    roni=pd.read_csv(ROOT/'data/roni_noaa.csv').set_index('year')
    current_enso=float(roni.loc[2026,'JAS'])
    manifest={'as_of':'2026-11-03','season':'2026/2027','enso_feature':'JAS2026 RONI',
              'enso_value':current_enso,'enso_limitation':'Published revised histories, not original issue-date vintages. Future winter strength reports are context, not realized predictors.',
              'copper_october_status':'unverified; no resort October total substituted or filled with zero',
              'assumption':'Seasons independent/exchangeable as requested; no chronological generalization claim.',
              'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/copper_monthly_user.csv',ROOT/'data/roni_noaa.csv',ROOT/'data/ghcn_monthly_wide.csv']}}
    for experiment in (['copper','copper_missing_october','copper_regional_october','dillon'] if args.experiment=='all' else [args.experiment]):
        if experiment=='copper':
            frame=load_training('copper')
            scenarios=live_frame({f'October scenario {v} inches':v for v in [0,5,10,20,30,40]},current_enso)
            run_experiment(frame,'monthly_copper',list(range(6)),True,scenarios,args.workers)
        elif experiment=='copper_missing_october':
            frame=load_training('copper')
            scenarios=live_frame({'October unverified; ENSO information only':np.nan},current_enso)
            run_experiment(frame,'monthly_copper_missing_october',[0],False,scenarios,args.workers)
        elif experiment=='copper_regional_october':
            # A real regional observation can be an explicitly named predictor.
            # It is never relabeled as observed Copper snowfall in saved data.
            frame=load_training('copper').rename(columns={'oct':'copper_oct'})
            station=pd.read_csv(ROOT/'data/ghcn_monthly_wide.csv')
            dillon=station.loc[station.station_id=='USC00052281',['start_year','oct']].rename(columns={'oct':'dillon_oct'})
            frame=frame.merge(dillon,on='start_year',validate='one_to_one').dropna(subset=['dillon_oct']).reset_index(drop=True)
            frame['oct']=frame.dillon_oct  # Internal feature alias for origin 0 only.
            frame.to_csv(OUT/'monthly_copper_regional_training.csv',index=False)
            value=float(dillon.loc[dillon.start_year==2026,'dillon_oct'].iloc[0])
            scenarios=live_frame({'Copper target; observed Dillon October predictor':value},current_enso)
            regional=run_experiment(frame,'monthly_copper_regional_october',[0],True,scenarios,args.workers)
            regional['feature_source']='The origin-0 oct feature is observed Dillon October, not Copper October. Labels are Copper Nov-Apr; 19 complete matched seasons.'
            (OUT/'monthly_copper_regional_october_results.json').write_text(json.dumps(regional,indent=2,allow_nan=False))
        elif experiment=='dillon':
            frame=load_training('dillon')
            station=pd.read_csv(ROOT/'data/ghcn_monthly_wide.csv')
            current=station.loc[(station.station_id=='USC00052281')&(station.start_year==2026)].iloc[0]
            value=float(current.oct)
            if not np.isfinite(value): raise ValueError('Dillon October incomplete')
            scenarios=live_frame({'Dillon measured October':value},current_enso)
            run_experiment(frame,'monthly_dillon',[0],True,scenarios,args.workers)
            manifest['dillon_october_inches']=value
    manifest_path=OUT/'monthly_run_manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text())
        existing.update(manifest)
        manifest=existing
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
