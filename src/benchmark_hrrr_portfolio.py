"""Matched 24-hour HRRR/previous-neural comparison on frozen calendar dates.

Native ASNOW forecast endpoints are compared with actual reporting intervals.
Only forecast amounts are linearly interpolated at fractional clock endpoints.
The two-day network is evaluated on its first24h output; native HRRR cannot
cover its full second day when forecasts are issued several hours after run.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/nwp-weather/forecast-portfolio'
OLD=ROOT/'outputs/nwp-weather/neural48-trained'

def cumulative_at(d,hour):
    d=d.sort_values('forecast_hour')
    x=d.forecast_hour.to_numpy(float);y=d.snowfall_cumulative_inches.to_numpy(float)
    exact=np.flatnonzero(np.isclose(x,hour))
    if len(exact):return float(y[exact[0]])
    before=np.flatnonzero(x<hour);after=np.flatnonzero(x>hour)
    if not len(before) or not len(after):return np.nan
    a,b=before[-1],after[0]
    if x[b]-x[a]>1.00001:return np.nan
    return float(np.interp(hour,x[[a,b]],y[[a,b]]))

def score(y,p):
    return dict(n=len(y),mae_inches=float(np.mean(abs(y-p))),rmse_inches=float(np.sqrt(np.mean((y-p)**2))),bias_inches=float(np.mean(p-y)))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(ROOT/'data/neural48_training/prepared/index.csv')
    a=np.load(ROOT/'data/neural48_training/prepared/arrays.npz')
    native=pd.read_csv(ROOT/'data/hrrr_portfolio/historical_native_endpoints.csv.gz')
    native['run_time_utc']=pd.to_datetime(native.run_time_utc,utc=True)
    groups={(s,r):g for (s,r),g in native.groupby(['site','run_time_utc'])}
    prior={}
    for split in ['validation','test']:
        z=np.load(OLD/f'{split}_predictions.npz')
        for i,r in enumerate(z['row_indices']):prior[int(r)]=float(z['5_mean'][i,0])
    rows=[]
    for r in meta.itertuples():
        start=pd.Timestamp(r.start_utc);run=pd.Timestamp(r.origin_date,tz='UTC')+pd.Timedelta(hours=6)
        if run+pd.Timedelta(hours=3)>pd.Timestamp(r.issue_utc):continue
        if r.split in ['validation','test'] and not np.isfinite(prior.get(r.row,np.nan)):continue
        g=groups.get((r.site,run))
        if g is None:continue
        h=(start-run).total_seconds()/3600
        amount=cumulative_at(g,h+24)-cumulative_at(g,h)
        if not np.isfinite(amount) or amount<0:continue
        rows.append(dict(row=r.row,site=r.site,split=r.split,case_id=r.case_id,start=start.isoformat(),
            truth=float(a['y'][r.row,0]),hrrr_native=amount,prior_neural=prior.get(r.row,np.nan),
            target=r.target_name,clock_verified=r.clock_verified,fresh_snow_verified=r.fresh_snow_verified))
    d=pd.DataFrame(rows);choices={};metrics=[]
    for site,g in d.groupby('site'):
        tr=g[g.split.eq('train')];v=g[g.split.eq('validation')];t=g[g.split.eq('test')]
        if len(tr)<15 or len(v)<10 or len(t)<10:continue
        # Train-only one-parameter multiplicative calibration, no wet-only cherry-pick.
        x=tr.hrrr_native.to_numpy();y=tr.truth.to_numpy()
        scale=float(np.clip(x@y/max(x@x,1e-8),.1,4.))
        for ix in g.index:d.loc[ix,'hrrr_calibrated']=scale*d.loc[ix,'hrrr_native']
        v=d.loc[v.index];t=d.loc[t.index]
        candidates={k:float(np.mean((v[k]-v.truth)**2)) for k in ['hrrr_native','hrrr_calibrated','prior_neural']}
        # Freeze a nonnegative convex mixture using validation only.
        weights=np.linspace(0,1,21)
        hrrr_kind=min(['hrrr_native','hrrr_calibrated'],key=candidates.get)
        losses=[float(np.mean((w*v[hrrr_kind]+(1-w)*v.prior_neural-v.truth)**2)) for w in weights]
        w=float(weights[int(np.argmin(losses))]);candidates['portfolio']=min(losses)
        selected=min(candidates,key=candidates.get)
        d.loc[g.index,'portfolio']=w*d.loc[g.index,hrrr_kind]+(1-w)*d.loc[g.index,'prior_neural']
        choices[site]=dict(scale_fit_on_train=scale,native_hrrr_component=hrrr_kind,
            hrrr_weight=w,prior_neural_weight=1-w,selected=selected,validation_mse=candidates,
            training_n=len(tr),validation_n=len(v),test_n=len(t),
            note='Weights fit only to fixed-reporting24h windows; hourly and second-day improvement unverified.')
        tq=d.loc[t.index];vq=d.loc[v.index]
        choices[site]['validation_residual_p80_inches']=float(np.quantile(abs(vq.portfolio-vq.truth),.8))
        # Resample complete calendar weeks, keeping serially related storms together.
        weeks=pd.to_datetime(tq.start,utc=True).dt.strftime('%G-%V').to_numpy()
        keys=np.unique(weeks);rng=np.random.default_rng(428)
        comparisons={}
        for baseline in ['hrrr_native','prior_neural']:
            diffs=[]
            for _ in range(1600):
                ix=np.concatenate([np.flatnonzero(weeks==k) for k in rng.choice(keys,len(keys),replace=True)])
                sub=tq.iloc[ix]
                diffs.append(float(np.sqrt(np.mean((sub.portfolio-sub.truth)**2))-np.sqrt(np.mean((sub[baseline]-sub.truth)**2))))
            comparisons[baseline]=dict(portfolio_minus_baseline_rmse_95ci=np.quantile(diffs,[.025,.975]).tolist(),bootstrap_unit='whole calendar week',blocks=len(keys))
        choices[site]['test_uncertainty']=comparisons
        for split in ['validation','test']:
            q=d[d.site.eq(site)&d.split.eq(split)]
            for method in candidates:metrics.append(dict(site=site,split=split,method=method,**score(q.truth.to_numpy(),q[method].to_numpy())))
    d.to_csv(OUT/'hrrr24_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'hrrr24_metrics.csv',index=False)
    (OUT/'hrrr24_selection.json').write_text(json.dumps(choices,indent=2))
    print(pd.DataFrame(metrics).query('split == "test"').to_string(index=False))

if __name__=='__main__':main()
