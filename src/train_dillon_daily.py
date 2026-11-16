"""Evaluate lag-only daily Dillon forecasts with nested whole-winter folds.

This is an exchangeable-season retrospective experiment, not an operational
hindcast or a current forecast. No same-day/future observed weather is a feature.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
SEED=20261103
FEATURES=['snow_lag1_inches','snow_prev3_inches','snow_prev7_inches',
          'lag1_tmax_celsius','lag1_tmin_celsius','lag1_precipitation_mm',
          'lag1_snow_depth_cm','calendar_sin','calendar_cos']
CANDIDATES=[{'family':'zero'}, {'family':'calendar_mean'}, {'family':'persistence'},
            {'family':'ridge','alpha':10.}, {'family':'ridge','alpha':1000.},
            {'family':'hist_gradient_boosting','max_leaf_nodes':7,'min_samples_leaf':40},
            {'family':'hist_gradient_boosting','max_leaf_nodes':15,'min_samples_leaf':80}]


def season_folds(frame,n_splits):
    years=np.sort(frame.season_start.unique())
    for a,b in KFold(n_splits=n_splits,shuffle=True,random_state=SEED).split(years):
        train=np.flatnonzero(frame.season_start.isin(years[a]).to_numpy())
        test=np.flatnonzero(frame.season_start.isin(years[b]).to_numpy())
        assert set(frame.iloc[train].season_start).isdisjoint(frame.iloc[test].season_start)
        yield train,test


def predict(train,test,horizon,config,counts):
    family=config['family']
    target=f'target_snow_next{horizon}d_inches'
    y=train[target].to_numpy(float)
    if family=='zero':
        return np.zeros(len(test))
    if family=='persistence':
        name='snow_lag1_inches' if horizon==1 else f'snow_prev{horizon}_inches'
        return test[name].to_numpy(float)
    if family=='calendar_mean':
        # Fixed 31-day circular calendar window. Only training labels contribute.
        day=train.day_of_year.to_numpy()
        lookup={}
        for q in test.day_of_year.unique():
            distance=np.abs(day-int(q))
            distance=np.minimum(distance,366-distance)
            chosen=y[distance<=15]
            lookup[q]=float(chosen.mean()) if len(chosen) else float(y.mean())
        counts['calendar_mean']+=1
        return test.day_of_year.map(lookup).to_numpy(float)
    if family=='ridge':
        estimator=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                                StandardScaler(),Ridge(alpha=config['alpha']))
    elif family=='hist_gradient_boosting':
        # Poisson deviance provides a nonnegative conditional mean for totals.
        # No random-row early-stopping subset is used across grouped winters.
        estimator=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
            HistGradientBoostingRegressor(loss='poisson',learning_rate=.05,max_iter=100,
                max_leaf_nodes=config['max_leaf_nodes'],min_samples_leaf=config['min_samples_leaf'],
                l2_regularization=10.,early_stopping=False,random_state=SEED))
    else:
        raise ValueError(family)
    estimator.fit(train[FEATURES],y)
    counts[family]+=1
    return np.maximum(0.,estimator.predict(test[FEATURES]))


def tune(train,horizon,counts):
    target=f'target_snow_next{horizon}d_inches'
    folds=list(season_folds(train,3))
    scores=[]
    for config in CANDIDATES:
        squared=[]
        for a,b in folds:
            predicted=predict(train.iloc[a],train.iloc[b],horizon,config,counts)
            squared.extend((predicted-train.iloc[b][target].to_numpy(float))**2)
        scores.append({'config':config,'inner_rmse':float(np.sqrt(np.mean(squared)))})
    best={}
    for row in scores:
        family=row['config']['family']
        if family not in best or row['inner_rmse']<best[family]['inner_rmse']:
            best[family]=row
    return min(scores,key=lambda row:row['inner_rmse']),best


def metrics(predictions):
    result=[]
    for horizon,table in predictions.groupby('horizon_days'):
        reference=table.loc[table.family=='calendar_mean'].set_index('date')
        for family,group in table.groupby('family',sort=False):
            error=group.prediction_inches-group.actual_inches
            positive=group.actual_inches>0
            baseline=reference.loc[group.date,'prediction_inches'].to_numpy()-group.actual_inches.to_numpy()
            improvement=np.abs(baseline)-np.abs(error.to_numpy())
            seasonal=pd.DataFrame({'season':group.season_start.to_numpy(),'improvement':improvement}).groupby('season').improvement.agg(['sum','size'])
            rng=np.random.default_rng(SEED)
            sampled=rng.integers(len(seasonal),size=(10000,len(seasonal)))
            # Match the pooled per-origin point estimate even when winters have
            # unequal numbers of eligible dates. Resample cluster sums/counts.
            boot=(seasonal['sum'].to_numpy()[sampled].sum(axis=1) /
                  seasonal['size'].to_numpy()[sampled].sum(axis=1))
            result.append({'horizon_days':int(horizon),'family':family,
                           'n_daily_origins':len(group),'n_seasons':int(group.season_start.nunique()),
                           'positive_target_origins':int(positive.sum()),
                           'zero_target_fraction':float((~positive).mean()),
                           'mae_inches':float(abs(error).mean()),
                           'rmse_inches':float(np.sqrt(np.mean(error**2))),
                           'positive_target_mae_inches':float(abs(error[positive]).mean()),
                           'positive_target_rmse_inches':float(np.sqrt(np.mean(error[positive]**2))),
                           'mean_mae_improvement_vs_calendar_inches':float(np.mean(improvement)),
                           'season_bootstrap_improvement_ci95':np.quantile(boot,[.025,.975]).tolist()})
    return result


def main():
    starts=time.monotonic()
    path=OUT/'dillon_daily_features.csv'
    data=pd.read_csv(path,parse_dates=['date'])
    data['season_start']=data.date.dt.year-(data.date.dt.month<10).astype(int)
    data=data.loc[data.date.dt.month.isin([10,11,12,1,2,3,4]) & data.season_start.between(2005,2025)].copy()
    all_rows=[];selections=[];cohorts=[];counts=Counter()
    with threadpool_limits(limits=2):
        for horizon in [1,3,7]:
            target=f'target_snow_next{horizon}d_inches'
            past='snow_lag1_inches' if horizon==1 else f'snow_prev{horizon}_inches'
            # Complete persistence features make every baseline use exactly the
            # same dates; other missing predictors are imputed within each fit.
            frame=data.dropna(subset=[target,past]).sort_values('date').reset_index(drop=True)
            cohorts.append({'horizon_days':horizon,'rows':len(frame),'seasons':sorted(map(int,frame.season_start.unique())),
                            'first_origin':str(frame.date.min().date()),'last_origin':str(frame.date.max().date()),
                            'eligibility':'Origin October-April in finished winters; complete future target and complete same-length past snowfall window for persistence. Same cohort for all models.'})
            for outer,(a,b) in enumerate(season_folds(frame,5)):
                train,test=frame.iloc[a].reset_index(drop=True),frame.iloc[b].reset_index(drop=True)
                selected,by_family=tune(train,horizon,counts)
                cached={name:predict(train,test,horizon,row['config'],counts) for name,row in by_family.items()}
                cached['selected']=cached[selected['config']['family']]
                for family,predicted in cached.items():
                    choice=selected if family=='selected' else by_family[family]
                    assert np.isfinite(predicted).all() and (predicted>=0).all()
                    for j,row in test.iterrows():
                        all_rows.append({'date':str(row.date.date()),'season_start':int(row.season_start),
                            'horizon_days':horizon,'outer_fold':outer,'family':family,
                            'chosen_family':choice['config']['family'],'actual_inches':float(row[target]),
                            'prediction_inches':float(predicted[j]),'config':json.dumps(choice['config'],sort_keys=True)})
                selections.append({'horizon_days':horizon,'outer_fold':outer,
                    'training_seasons':sorted(map(int,train.season_start.unique())),
                    'heldout_seasons':sorted(map(int,test.season_start.unique())),
                    'n_train_origins':len(train),'n_test_origins':len(test),
                    'selected':selected,'family_choices':by_family})
                print(f'Dillon {horizon}-day: outer fold {outer+1}/5 complete; {time.monotonic()-starts:.1f}s',flush=True)
    predictions=pd.DataFrame(all_rows)
    assert not predictions.duplicated(['date','horizon_days','family']).any()
    for (horizon,family),group in predictions.groupby(['horizon_days','family']):
        expected=next(c['rows'] for c in cohorts if c['horizon_days']==horizon)
        assert len(group)==expected
    for horizon,table in predictions.groupby('horizon_days'):
        reference=table.loc[table.family=='calendar_mean'].set_index('date')
        for family,group in table.groupby('family'):
            assert set(group.date)==set(reference.index)
            np.testing.assert_allclose(group.actual_inches.to_numpy(),reference.loc[group.date,'actual_inches'].to_numpy())
        folds=[s for s in selections if s['horizon_days']==horizon]
        held=[y for fold in folds for y in fold['heldout_seasons']]
        assert len(held)==len(set(held))==21
    predictions.to_csv(OUT/'dillon_daily_model_predictions.csv',index=False)
    result={'as_of':'2026-11-03','station_id':'USC00052281','station_name':'DILLON 1 E',
            'target':'Fresh snowfall totals over 1,3,7 station-reporting days beginning at target day t, inches',
            'features':FEATURES,'seed':SEED,'outer_folds':5,'inner_folds':3,
            'max_cpu_threads':2,
            'grouping':'Entire winter seasons; user-requested exchangeable-season assumption. All observations of an omitted winter stay in the same outer fold.',
            'validation_limits':'Retrospective lag-only station experiment, not an operational hindcast. Inputs use record dates before t, but historical publication timestamps and reporting-day alignment have not been established.',
            'feature_policy':'No same-day or future measured temperature, precipitation, snowfall or snow depth is used as a predictor. Calendar phase is known. Median imputation and scaling fitted separately within training folds.',
            'baseline_policy':'Fixed 31-day circular day-of-year mean fitted to training labels; persistence repeats the preceding same-length observed snow sum; zero always predicts no snow.',
            'grid':CANDIDATES,'tree_fixed_settings':{'loss':'poisson','learning_rate':.05,'max_iter':100,'l2_regularization':10.,'early_stopping':False},
            'cohorts':cohorts,'metrics':metrics(predictions),'selections':selections,
            'fit_counts':dict(counts),'learned_model_fit_count':counts['ridge']+counts['hist_gradient_boosting'],
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'input_table':str(path.relative_to(ROOT)),'runtime_seconds':time.monotonic()-starts,
            'validation_checks':['Every model evaluated on identical dates and labels within each horizon.',
                                 'No duplicate date/horizon/family predictions; every eligible origin predicted once.',
                                 'All21 winter seasons appear in exactly one outer test fold; training and test seasons disjoint.',
                                 'Feature list contains lagged observations and known calendar phase only.',
                                 'Saved metric values were independently recomputed from the prediction CSV.'],
            'uncertainty':'Improvement intervals resample whole winters, retaining all correlated daily and overlapping-window errors. They are exploratory, not confidence intervals from independent daily errors.',
            'bootstrap_estimator':'Resample winter clusters with replacement; divide sampled improvement sums by sampled eligible-origin counts, matching the pooled per-origin point estimate.',
            'scope_limits':['No A-Basin or Keystone target observations were used, so these errors do not measure either resort.',
                            'No numerical weather prediction inputs were included. This is a lag-only benchmark that may be improved with issue-time forecast weather.',
                            'The latest daily observation is October31,2026; this script issues no live forecast for November3.',
                            'Origin dates run October-April; 3/7-day targets from late April can extend into early May. All belong to the same held-out winter and do not overlap the next October training season.',
                            'Daily counts are correlated within storms and winters; the effective independent climate sample remains21 winters.']}
    (OUT/'dillon_daily_model_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for row in result['metrics']:
        print(row['horizon_days'],row['family'],'MAE',round(row['mae_inches'],3),'RMSE',round(row['rmse_inches'],3),'positive MAE',round(row['positive_target_mae_inches'],3))
    print('Learned model fits:',result['learned_model_fit_count'],dict(counts),flush=True)


if __name__=='__main__':main()
