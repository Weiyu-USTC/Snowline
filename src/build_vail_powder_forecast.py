"""Exact daily >=5-inch probabilities from the frozen Vail event models.

The threshold applies to CAVMM's reported sensor endpoint, not verified fresh
snowfall. This derived analysis restores existing models; it does not refit them.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit,log_ndtr
import torch

import vail_hourly_model as old
import vail_storm_neural as neural
from vail_baselines import tabular_summary
from run_vail_storm_models import initial_history,ages,SEED
from build_vail_storm_report import COLORS,plot_setup,save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/vail-storm-models'
THRESHOLD=5.
NAMES=['Seasonal Markov','Seasonal duration','Weather duration','Neural events']


def neural_exceedance(prediction,threshold=THRESHOLD):
    """Exact tail of the daily state mixture; >=1 uses truncated lognormal."""
    if not np.isfinite(threshold) or threshold<1:
        raise ValueError('This tail helper supports finite thresholds >=1 inch')
    params=np.asarray(prediction['component_params'],dtype=np.float64)
    weights=np.asarray(prediction['component_weights'],dtype=np.float64)
    mu,sigma=params[...,4],params[...,5]
    tail=np.exp(log_ndtr((mu-np.log(threshold))/sigma)-log_ndtr(mu/sigma))
    return np.clip((weights*expit(params[...,0])*tail).sum(axis=-1),0.,1.)


def historical_probability(target,years,dates,threshold=THRESHOLD):
    """Same pooled +/-15-calendar-day climatology as the existing mean line."""
    values=pd.concat([target.loc[f'{year}-10-01':f'{year+1}-04-30'] for year in years]).dropna()
    doy=np.array([pd.Timestamp(2000,d.month,d.day).dayofyear for d in values.index])
    result=[]
    for date in dates:
        day=pd.Timestamp(2000,date.month,date.day).dayofyear
        delta=np.abs(doy-day);sample=values.to_numpy()[np.minimum(delta,366-delta)<=15]
        if not len(sample):sample=values.to_numpy()
        result.append((float(np.mean(sample>=threshold)),len(sample),int(np.sum(sample>=threshold))))
    return np.asarray(result)


def plot_probability(rows,summary):
    plot_setup()
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[1.35,1]})
    for name in NAMES+['Historical climatology']:
        r=rows[rows.model==name].sort_values('lead');dates=pd.to_datetime(r.date)
        historical=name=='Historical climatology'
        color=COLORS['Climatology'] if historical else COLORS[name]
        style=dict(color=color,lw=2.1,ls='--' if historical else '-',alpha=.5 if historical else 1.)
        axes[0].plot(dates,100*r.probability,label='Historical rate (α = 0.5)' if historical else name,**style)
        anchor=dates.iloc[0]-pd.Timedelta(days=1)
        axes[1].plot(pd.DatetimeIndex([anchor]).append(pd.DatetimeIndex(dates)),np.r_[0.,r.probability.cumsum()],**style)
        axes[1].scatter(dates.iloc[-1],r.probability.sum(),s=22,color=color,alpha=style['alpha'],zorder=4)
    axes[0].set_ylim(0,max(18,float(rows.probability.max()*115)))
    axes[0].set_ylabel('Daily probability (%)')
    axes[0].set_title('Chance of a ≥5-inch reading on each date',loc='left',fontsize=12,weight='bold',pad=10)
    axes[0].legend(frameon=False,ncol=3,loc='upper center',fontsize=9,bbox_to_anchor=(.54,1.02))
    axes[1].set_ylim(0,summary[summary.period=='Nov 5–Dec 31'].expected_days.max()*1.16)
    axes[1].set_ylabel('Expected powder-day count')
    axes[1].set_title('Expected number of ≥5-inch days accumulated from November 5',loc='left',fontsize=12,weight='bold',pad=10)
    for ax in axes:ax.grid(axis='y',alpha=.8);ax.set_axisbelow(True)
    axes[1].set_xlim(pd.Timestamp('2026-11-04'),pd.Timestamp('2027-01-02'))
    axes[1].set_xticks(pd.to_datetime(['2026-11-05','2026-11-12','2026-11-19','2026-11-26','2026-12-03','2026-12-10','2026-12-17','2026-12-24','2026-12-31']))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[1].set_xlabel('Reporting date, 2026 · 04:00 CAIC display-clock endpoint')
    fig.suptitle('Vail powder-day probabilities through December 31',x=.085,y=.98,ha='left',fontsize=19,weight='bold')
    fig.text(.085,.925,'Powder-day threshold: ≥5 inches · frozen November 4 forecast · daily probabilities from fitted distributions',fontsize=10,color='#58707c')
    fig.text(.085,.032,'Provisional CAVMM sensor-reading proxy; fresh snowfall and ≥5-inch probability calibration remain unverified.\n'
             'Long-range curves mainly reflect seasonal risk, not identified storm dates. Earlier-season observations are excluded.',fontsize=9,color='#795440')
    fig.subplots_adjust(left=.085,right=.98,top=.82,bottom=.15,hspace=.38)
    save(fig,'powder_day_probability')


def main():
    manifest=json.loads((OUT/'forecast_manifest.json').read_text())
    old.seed_everything(SEED)
    prior_out=old.OUT
    with tempfile.TemporaryDirectory() as temporary:
        old.OUT=Path(temporary)
        try:_,frame,target,years=old.load_data()
        finally:old.OUT=prior_out
    assert years==manifest['training_winters']
    source_hash=hashlib.sha256((ROOT/'data/vail_hourly/hourly.csv').read_bytes()).hexdigest()
    assert source_hash==manifest['source_sha256']
    origin=pd.Timestamp(manifest['issue_source_clock'])
    xx=frame.reindex(pd.date_range(end=origin,periods=168,freq='h')).to_numpy('float32')[None]
    ff=old.future_features(origin)[None];initial=initial_history(target,[origin]);summary=tabular_summary(xx)
    published=pd.read_csv(OUT/'forecast_daily.csv')
    dates=pd.DatetimeIndex(pd.to_datetime(published[published.model=='Neural events'].sort_values('lead').date))
    n=len(dates);probabilities={};checks={}
    for name,model in joblib.load(OUT/'statistical_event_models.joblib').items():
        prediction=model.forecast(summary,ff[:,:,:4],initial[:,0],ages(initial),n_paths=1,seed=SEED,
                                  exceedance_threshold=THRESHOLD)
        probabilities[name]=prediction['p_exceedance'][0,:n]
        reference=published[published.model==name].sort_values('lead')
        np.testing.assert_allclose(prediction['mean'][0,:n],reference.mean_inches,atol=2e-6,rtol=0)
        checks[name]=float(np.max(abs(prediction['mean'][0,:n]-reference.mean_inches.to_numpy())))
    checkpoint=torch.load(OUT/'neural_event_model.pt',weights_only=True)
    model,scaler=neural.restore_neural(checkpoint)
    prediction=neural.forecast_neural(model,scaler,xx,ff,initial,return_components=True)
    probabilities['Neural events']=neural_exceedance(prediction)[0,:n]
    np.testing.assert_allclose(neural_exceedance(prediction,1.),prediction['p_active'],atol=1e-7,rtol=0)
    reference=published[published.model=='Neural events'].sort_values('lead')
    np.testing.assert_allclose(prediction['mean'][0,:n],reference.mean_inches,atol=2e-6,rtol=0)
    checks['Neural events']=float(np.max(abs(prediction['mean'][0,:n]-reference.mean_inches.to_numpy())))
    historical=historical_probability(target,years,dates)
    probabilities['Historical climatology']=historical[:,0]
    paths=np.load(OUT/'forecast_paths.npz');records=[];summaries=[];sampling={}
    for name,p in probabilities.items():
        assert p.shape==(n,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        for i,date in enumerate(dates):
            record=dict(model=name,date=str(date.date()),lead=i+1,threshold_inches=THRESHOLD,
                        probability=float(p[i]),cumulative_expected_days=float(p[:i+1].sum()))
            if name=='Historical climatology':
                record.update(historical_donor_count=int(historical[i,1]),historical_powder_readings=int(historical[i,2]))
            records.append(record)
        if name in paths:
            events=paths[name]>=THRESHOLD
            sampling[name]=dict(paths=int(len(events)),max_daily_absolute_difference=float(np.max(abs(events.mean(0)-p))))
        for label,mask in [('Nov 5–Dec 31',np.ones(n,dtype=bool)),('Nov 5–30',dates.month==11),('December',dates.month==12)]:
            record=dict(model=name,period=label,days=int(mask.sum()),expected_days=float(p[mask].sum()),
                        average_daily_probability=float(p[mask].mean()))
            if name in paths:
                counts=events[:,mask].sum(axis=1);lo,hi=np.quantile(counts,[.1,.9],method='inverted_cdf')
                record.update(count_q10=int(lo),count_q90=int(hi),probability_at_least_one_sampled=float(np.mean(counts>=1)))
            summaries.append(record)
    rows=pd.DataFrame(records);summary=pd.DataFrame(summaries)
    rows.to_csv(OUT/'powder_day_forecast.csv',index=False)
    summary.to_csv(OUT/'powder_day_summary.csv',index=False)
    metadata=dict(issue_source_clock=str(origin),first_date=str(dates[0].date()),last_date=str(dates[-1].date()),
                  threshold_inches=THRESHOLD,inclusive_threshold=True,target='CAVMM reported04:00 sensor reading, not verified fresh snowfall',
                  probability_method='Exact state-mixture tail: empirical donor fractions for statistical models; analytic truncated-lognormal tail for neural model',
                  historical_method='Same pooled +/-15-calendar-day observed endpoint climatology across six partial training winters',
                  expected_count='Sum of exact daily probabilities; linearity of expectation does not require independent dates',
                  count_range='Pointwise period-count10–90% range from1024 saved coherent event paths; uncalibrated',
                  at_least_one='Estimated from joint saved paths; not one-minus-product of daily marginals',
                  no_retraining=True,source_sha256=source_hash,mean_restoration_errors=checks,sampling_checks=sampling,
                  checkpoints={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['statistical_event_models.joblib','neural_event_model.pt']})
    (OUT/'powder_day_manifest.json').write_text(json.dumps(metadata,indent=2))
    plot_probability(rows,summary)
    print(summary[summary.period=='Nov 5–Dec 31'].to_string(index=False))


if __name__=='__main__':main()
