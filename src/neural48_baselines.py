"""Training-only joint two-day benchmarks for the 48-hour neural experiment.

The current widget's empirical-distribution forest family is refitted to the
new matched targets; its old fitted weights cannot be reused after target or
forecast-window changes. Trees partition paired two-day outcomes. Sampling
whole donor pairs preserves their daily dependence when calculating 48h totals.
"""
from __future__ import annotations

import hashlib
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

SEED = 20261110
FOREST_CONFIG = dict(n_estimators=128, max_depth=7, min_samples_leaf=15,
                     max_features=.8, bootstrap=True, random_state=SEED, n_jobs=1)
CURRENT = 'Current model family · refit'
ALL_WEATHER = 'Weather forest · MOS + JMA'
CLIMATE = 'Historical calendar distribution'


def _indices(values, n):
    idx = np.asarray(values, dtype=int)
    if idx.ndim != 1 or not len(idx) or (idx < 0).any() or (idx >= n).any():
        raise ValueError('Indices must select at least one valid row')
    if len(np.unique(idx)) != len(idx):
        raise ValueError('Repeated indices would duplicate training or test outcomes')
    return idx


def _arrays(nwp, static, history, site, day_of_year):
    nwp, static = np.asarray(nwp, float), np.asarray(static, float)
    history, site = np.asarray(history), np.asarray(site).astype(str)
    doy = np.asarray(day_of_year, float)
    n = len(nwp)
    if nwp.ndim != 3 or nwp.shape[1] != 48 or nwp.shape[2] < 16:
        raise ValueError('NWP input must be [N,48,F], with the first16 MOS features')
    if static.ndim != 2 or len(static) != n or len(history) != n or site.shape != (n,) or doy.shape != (n,):
        raise ValueError('Baseline arrays do not share their case axis')
    if not np.isfinite(doy).all() or (doy < 1).any() or (doy > 366).any():
        raise ValueError('day_of_year must lie in1–366')
    return nwp, static, history, site, doy


def _forecast_summary(nwp, static, feature_count):
    """Known forecast summaries; no target/history values enter these columns."""
    x = np.asarray(nwp[:, :, :feature_count], float).copy()
    x[~np.isfinite(x)] = np.nan
    parts = [np.asarray(static, float)]
    for lo, hi in [(0,24),(24,48)]:
        block = x[:,lo:hi]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            parts.extend([np.nanmean(block,axis=1),np.nanmin(block,axis=1),
                np.nanmax(block,axis=1),np.nanstd(block,axis=1),
                block[:,-1]-block[:,0],np.mean(~np.isfinite(block),axis=1)])
    return np.concatenate(parts,axis=1)


class JointEmpiricalForest:
    """Multioutput forest leaves define a distribution over training-only pairs."""
    def __init__(self, seed=SEED):
        self.config = {**FOREST_CONFIG, 'random_state':int(seed)}

    def transform(self, x):
        x=np.asarray(x,float);missing=~np.isfinite(x)
        if x.ndim!=2 or x.shape[1]!=len(self.median):
            raise ValueError('Predictor columns differ from fitted forest')
        return np.concatenate([np.where(missing,self.median,x),missing.astype(float)],axis=1)

    def fit(self, x, y):
        x,y=np.asarray(x,float),np.asarray(y,float)
        if y.ndim!=2 or y.shape!=(len(x),2) or not np.isfinite(y).all() or (y<0).any():
            raise ValueError('Forest donors must be observed nonnegative paired daily snowfall')
        self.y=y.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            self.median=np.nanmedian(np.where(np.isfinite(x),x,np.nan),axis=0)
        self.median=np.nan_to_num(self.median,nan=0.,posinf=0.,neginf=0.)
        xx=self.transform(x)
        with threadpool_limits(limits=1):
            self.forest=RandomForestRegressor(**self.config).fit(xx,self.y)
            leaves=self.forest.apply(xx)
        self.donors=[{int(leaf):np.flatnonzero(leaves[:,tree]==leaf)
            for leaf in np.unique(leaves[:,tree])} for tree in range(leaves.shape[1])]
        return self

    def weights(self, x):
        with threadpool_limits(limits=1):
            leaves=self.forest.apply(self.transform(x))
        weights=np.zeros((len(leaves),len(self.y)),float)
        for tree,lookup in enumerate(self.donors):
            for leaf in np.unique(leaves[:,tree]):
                rows=np.flatnonzero(leaves[:,tree]==leaf);donors=lookup[int(leaf)]
                weights[np.ix_(rows,donors)]+=1/(len(self.donors)*len(donors))
        if not np.allclose(weights.sum(1),1.,atol=1e-10):
            raise ValueError('Forest donor weights do not sum to one')
        return weights


def _draw_pairs(values, weights, uniforms):
    """One donor index chooses both days, never two independent day draws."""
    cumulative=np.cumsum(weights,axis=1)
    cumulative[:,-1]=1.
    selected=np.array([np.searchsorted(cumulative[i],uniforms[i],side='right')
                       for i in range(len(weights))])
    return np.asarray(values,float)[selected]


def fit_baselines(nwp, static, history, site, y, train_idx, day_of_year, *, n_paths=512, seed=SEED):
    """Fit only train_idx; no validation/test targets, scalers or donors used."""
    nwp,static,history,site,doy=_arrays(nwp,static,history,site,day_of_year)
    idx=_indices(train_idx,len(nwp));y=np.asarray(y,float)
    if y.shape!=(len(nwp),2):raise ValueError('y must contain two daily targets per case')
    if not np.isfinite(y[idx]).all() or (y[idx]<0).any():
        raise ValueError('All fitting target pairs must be observed and nonnegative')
    if int(n_paths)<256:raise ValueError('At least256 joint paths are required')
    methods={CURRENT:16}
    if nwp.shape[2]>16:methods[ALL_WEATHER]=nwp.shape[2]
    summaries={name:_forecast_summary(nwp,static,count) for name,count in methods.items()}
    bundle={'schema_version':1,'n_paths':int(n_paths),'seed':int(seed),
        'nwp_feature_count':nwp.shape[2],'static_feature_count':static.shape[1],
        'forest_feature_counts':methods,'sites':{},'train_idx':idx.copy(),
        'train_target_sha256':hashlib.sha256(np.ascontiguousarray(y[idx]).tobytes()).hexdigest(),
        'metadata':{'targets':'Paired first24h and second24h snowfall at the matched target station',
            'current_family_scope':'Same empirical forest family and fixed hyperparameters as widget; refit on new matched two-day targets, not reused old weights',
            'forest_parameters':{**FOREST_CONFIG,'random_state':int(seed)},
            'feature_scope':'Static known covariates plus per-day forecast summaries; supplied observed history is not used by this current-family comparator',
            'history_used':False,'calibration':'Training-only empirical leaf distributions, not prospectively coverage-calibrated',
            'joint_policy':'A sampled training donor provides both daily outcomes; all48h quantiles must use sums of these joint paths'}}
    for site_id in sorted(np.unique(site[idx])):
        selected=idx[site[idx]==site_id]
        record={'train_idx':selected.copy(),'y':y[selected].copy(),'day_of_year':doy[selected].copy(),
                'forests':{}}
        for name,summary in summaries.items():
            record['forests'][name]=JointEmpiricalForest(seed).fit(summary[selected],y[selected])
        bundle['sites'][site_id]=record
    return bundle


def predict_baselines(bundle,nwp,static,history,site,test_idx,day_of_year):
    """Return test-order exact daily means and reproducible joint donor paths."""
    nwp,static,history,site,doy=_arrays(nwp,static,history,site,day_of_year)
    idx=_indices(test_idx,len(nwp))
    if nwp.shape[2]!=bundle['nwp_feature_count'] or static.shape[1]!=bundle['static_feature_count']:
        raise ValueError('Current predictor schema differs from fitting')
    absent=set(site[idx])-set(bundle['sites'])
    if absent:raise ValueError(f'No training donors for sites: {sorted(absent)}')
    methods=list(bundle['forest_feature_counts'])+[CLIMATE]
    n_paths=bundle['n_paths']
    result={name:{'daily_mean':np.full((len(idx),2),np.nan),
                  'joint_paths':np.full((len(idx),n_paths,2),np.nan),
                  'metadata':{'method':name,'paths':n_paths,'joint_days':True,
                    'train_only':True,'forecast_input_scope':'MOS + static' if name==CURRENT else
                       'MOS + JMA + static' if name==ALL_WEATHER else 'Site and calendar ±15 days',
                    'prediction_row_indices':idx.tolist()}}
            for name in methods}
    summaries={name:_forecast_summary(nwp[idx],static[idx],count)
               for name,count in bundle['forest_feature_counts'].items()}
    # A stable case index/site seed gives the same samples even when an
    # evaluation is split into batches or its order changes.
    uniforms=np.empty((len(idx),n_paths))
    for j,row in enumerate(idx):
        site_seed=int(hashlib.sha256(str(site[row]).encode()).hexdigest()[:8],16)
        uniforms[j]=np.random.default_rng(np.random.SeedSequence([bundle['seed'],int(row),site_seed])).random(n_paths)
    for site_id in sorted(np.unique(site[idx])):
        positions=np.flatnonzero(site[idx]==site_id);rows=idx[positions]
        saved=bundle['sites'][site_id];values=saved['y']
        for name in methods:
            if name==CLIMATE:
                distance=abs(doy[rows,None]-saved['day_of_year'][None,:])
                weights=(np.minimum(distance,366-distance)<=15).astype(float)
                weights[weights.sum(1)==0]=1.
                weights/=weights.sum(1,keepdims=True)
            else:
                weights=saved['forests'][name].weights(summaries[name][positions])
            result[name]['daily_mean'][positions]=weights@values
            result[name]['joint_paths'][positions]=_draw_pairs(values,weights,uniforms[positions])
    for r in result.values():
        if not np.isfinite(r['daily_mean']).all() or not np.isfinite(r['joint_paths']).all():
            raise ValueError('Incomplete baseline prediction')
    return result
