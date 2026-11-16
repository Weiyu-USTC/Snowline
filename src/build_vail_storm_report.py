"""Render the fitted Vail event-model benchmark and daily scenario forecasts."""
from __future__ import annotations
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/vail-storm-models'
COLORS={'Climatology':'#6e7880','Previous hourly TCN':'#b27542',
        'Seasonal Markov':'#629177','Seasonal duration':'#a37d32',
        'Weather duration':'#006d8c','Neural events':'#75529b'}
LEADS=['1–3','4–7','8–14','15–30','31–60']
ORDER=['Climatology','Previous hourly TCN','Seasonal Markov','Seasonal duration','Weather duration','Neural events']


def read_json(path):return json.loads(path.read_text())


def table(frame,digits=3):
    return '<div class="table-wrap">'+frame.to_html(index=False,border=0,escape=True,
           float_format=lambda x:f'{x:,.{digits}f}',na_rep='—')+'</div>'


def plot_setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'text.color':'#203c49','axes.labelcolor':'#203c49',
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#bac8d0','xtick.color':'#4e6570','ytick.color':'#4e6570',
                         'grid.color':'#e0e8ed','figure.facecolor':'white','axes.facecolor':'white',
                         'savefig.facecolor':'white','svg.fonttype':'none'})


def save(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def forecast_plot(forecast,climate,manifest):
    plot_setup()
    dates=pd.to_datetime(climate.date)
    fig,axes=plt.subplots(3,1,figsize=(12,8.5),sharex=True,
                         gridspec_kw={'height_ratios':[2,2,1.05]})
    for ax,name in zip(axes[:2],['Weather duration','Neural events']):
        values=forecast[forecast.model==name].sort_values('lead')
        ax.fill_between(dates,climate.q10_inches,climate.q90_inches,color=COLORS['Climatology'],alpha=.11,step='mid',label='Historical 10–90% range')
        ax.plot(dates,climate.mean_inches,color=COLORS['Climatology'],alpha=.5,lw=2,ls='--',label='Historical mean (α = 0.5)')
        ax.fill_between(dates,values.q10_inches,values.q90_inches,color=COLORS[name],alpha=.22,step='mid',label='Event model 10–90% range')
        ax.plot(dates,values.mean_inches,color=COLORS[name],lw=2,marker='o',ms=2.5,label='Event model expected reading')
        ax.set_ylabel('Station reading (in)');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.8)
        ax.set_title(name,loc='left',fontsize=11,weight='bold',pad=8)
    axes[0].legend(frameon=False,ncol=2,fontsize=8.5,loc='upper left')
    for name in ['Weather duration','Neural events']:
        values=forecast[forecast.model==name].sort_values('lead')
        axes[2].plot(dates,values.p_active*100,color=COLORS[name],lw=2,label=name)
    axes[2].set_ylabel('P(reading ≥1 in)\n(%)');axes[2].set_ylim(0,100);axes[2].set_yticks([0,50,100])
    axes[2].grid(axis='y',alpha=.8);axes[2].legend(frameon=False,ncol=2,fontsize=9,loc='upper right')
    axes[2].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO));axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[2].set_xlabel('Reporting date, 2026 · 04:00 CAIC display-clock endpoint')
    fig.suptitle('Vail event-model outlook through December 31',x=.09,y=.98,ha='left',fontsize=19,weight='bold')
    fig.text(.09,.93,'Input through November 4, 2026 04:00 · 168 hourly observations from four resort stations',fontsize=10,color='#58707c')
    fig.text(.09,.022,'Provisional CAVMM sensor proxy; these are not verified 24-hour fresh-snow totals.\n'
             'Shading is an uncalibrated pointwise predictive range. A smooth mean averages over uncertain event dates.',fontsize=9,color='#795440')
    fig.subplots_adjust(left=.09,right=.98,top=.86,bottom=.14,hspace=.35)
    save(fig,'forecast_comparison')


def error_plot(metrics):
    plot_setup()
    overall=metrics[metrics.leads=='1–60'].set_index('model').reindex(ORDER)
    fig,axes=plt.subplots(2,2,figsize=(12,8.6))
    for ax,metric,title in [(axes[0,0],'rmse','Daily amount RMSE'),(axes[0,1],'crps','Daily distribution CRPS')]:
        sorted_values=overall.sort_values(metric)
        yy=np.arange(len(sorted_values))
        ax.barh(yy,sorted_values[metric],color=[COLORS[m] for m in sorted_values.index],height=.62)
        ax.set_yticks(yy,sorted_values.index,fontsize=9);ax.invert_yaxis()
        ax.set_xlim(0,sorted_values[metric].max()*1.19)
        for i,value in enumerate(sorted_values[metric]):ax.text(value+.014*sorted_values[metric].max(),i,f'{value:.3f}',va='center',fontsize=9)
        ax.set_title(title,loc='left',fontsize=12,weight='bold');ax.set_xlabel('Inches of reported station reading')
        ax.grid(axis='x',alpha=.6);ax.set_axisbelow(True)
    for ax,metric,title in [(axes[1,0],'rmse','Amount error by lead'),(axes[1,1],'active_brier','Probability error by lead')]:
        for name in ORDER:
            values=metrics[(metrics.model==name)&metrics.leads.isin(LEADS)].set_index('leads').reindex(LEADS)
            ax.plot(np.arange(len(LEADS)),values[metric],marker='o',ms=3,lw=2.2 if name in ['Weather duration','Neural events'] else 1.3,
                    color=COLORS[name],alpha=.9,label=name)
        ax.set_xticks(np.arange(len(LEADS)),LEADS);ax.set_xlabel('Days ahead');ax.grid(axis='y',alpha=.7)
        ax.set_title(title,loc='left',fontsize=12,weight='bold')
        ax.set_ylabel('RMSE (in)' if metric=='rmse' else 'Brier score for reading ≥1 in')
    axes[1,1].legend(frameon=False,ncol=2,fontsize=7.5,loc='best')
    fig.suptitle('Vail: held-out event-model performance',x=.075,y=.98,ha='left',fontsize=19,weight='bold')
    fig.text(.075,.93,'Same 342 issue dates and observed targets across six partial winters · lower scores are better',fontsize=10,color='#58707c')
    fig.text(.075,.025,'Three shuffled whole-winter folds. Point means use exact state propagation; paths estimate fair event-model CRPS.\n'
             'Results concern the CAIC endpoint proxy, not official Vail fresh snowfall or a comparison with issued NWP forecasts.',fontsize=9,color='#58707c')
    fig.subplots_adjust(left=.16,right=.98,top=.85,bottom=.13,wspace=.48,hspace=.5)
    save(fig,'error_comparison')


def sample_plot(forecast,paths):
    plot_setup()
    names=['Weather duration','Neural events']
    fig,axes=plt.subplots(3,2,figsize=(12,8.3),sharex=True,sharey='col')
    for col,name in enumerate(names):
        data=forecast[forecast.model==name].sort_values('lead');dates=pd.to_datetime(data.date)
        for row,index in enumerate([0,1,2]):
            ax=axes[row,col]
            ax.bar(dates,paths[name][index],width=.82,color=COLORS[name],alpha=.75,label='One sampled sequence')
            ax.plot(dates,data.mean_inches,lw=1.5,ls='--',color='#697982',alpha=.65,label='Model mean')
            ax.axhline(1.,color='#c2cdd3',ls=':',lw=1)
            ax.set_title(f'{name} · sample {index+1}',loc='left',fontsize=10,pad=7)
            ax.grid(axis='y',alpha=.65);ax.set_axisbelow(True);ax.set_ylim(bottom=0)
            if col==0:ax.set_ylabel('Reading (in)')
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO,interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[0,0].legend(frameon=False,fontsize=8,ncol=2,loc='upper right')
    for ax in axes[-1]:ax.set_xlabel('Reporting date, 2026')
    fig.suptitle('Possible event sequences, not predicted storm dates',x=.08,y=.98,ha='left',fontsize=19,weight='bold')
    fig.text(.08,.925,'First three paths from each fixed-seed fitted-model ensemble; no selection of visually dramatic examples.',fontsize=10,color='#58707c')
    fig.text(.08,.025,'Active state: reported CAVMM endpoint ≥1 inch. Below-threshold readings preserve zero and small positive values.\n'
             'These are daily sensor-proxy scenarios, not verified fresh-snow totals or a fitted continuous-time pulse process.',fontsize=9,color='#795440')
    fig.subplots_adjust(left=.08,right=.98,top=.85,bottom=.135,wspace=.2,hspace=.42)
    save(fig,'event_sample_paths')


def cumulative_plot(forecast,climate,paths,manifest):
    """Sum coherent sequences, preserving the sensor-proxy measurement scope."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plot_setup()
    names=['Weather duration','Neural events','Seasonal Markov','Seasonal duration']
    climate=climate.sort_values('date').reset_index(drop=True)
    dates=pd.to_datetime(climate.date)
    origin=pd.Timestamp(manifest['issue_source_clock']).normalize()
    assert dates.iloc[0]==origin+pd.Timedelta(days=1)
    plot_dates=pd.DatetimeIndex([origin]).append(pd.DatetimeIndex(dates))
    historical=climate.mean_inches.to_numpy(float).cumsum()
    records=[];curves={}
    for name in names:
        data=forecast[forecast.model==name].sort_values('lead').reset_index(drop=True)
        assert data.date.tolist()==climate.date.tolist()
        raw=np.asarray(paths[name],dtype=np.float64)
        assert raw.shape==(manifest['paths_per_model'],len(dates))
        assert np.isfinite(raw).all() and (raw>=0).all()
        cumulative=raw.cumsum(axis=1)
        low,high=np.quantile(cumulative,[.1,.9],axis=0)
        mean=data.mean_inches.to_numpy(float).cumsum()
        assert np.all(np.diff(mean)>=0) and np.all(low<=high)
        curves[name]=(mean,low,high)
        for i,date in enumerate(climate.date):
            records.append(dict(model=name,date=date,lead=int(data.lead.iloc[i]),
                                cumulative_mean_sensor_inches=float(mean[i]),
                                cumulative_q10_sensor_inches=float(low[i]),
                                cumulative_q90_sensor_inches=float(high[i]),
                                cumulative_historical_mean_sensor_inches=float(historical[i])))
    pd.DataFrame(records).to_csv(OUT/'forecast_cumulative.csv',index=False)
    description=dict(first_date=str(dates.iloc[0].date()),last_date=str(dates.iloc[-1].date()),
                     zero_anchor=str(origin.date()),paths_per_model=manifest['paths_per_model'],
                     target='Cumulative sum of daily CAVMM 04:00 sensor readings; not verified fresh snowfall',
                     mean='Cumulative sum of exact forecast daily means',
                     interval='Pointwise 10th/90th percentiles of cumulative coherent model paths; uncalibrated',
                     historical='Cumulative sum of pooled calendar-climatology means, +/-15 calendar days across six partial winters; not complete-season mean totals',
                     earlier_season_observations_included=False)
    (OUT/'forecast_cumulative_manifest.json').write_text(json.dumps(description,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(12,8.3),sharey=True)
    ymax=max(v[2].max() for v in curves.values())*1.14
    for ax,name in zip(axes.flat,names):
        mean,low,high=curves[name]
        ax.fill_between(plot_dates,np.r_[0.,low],np.r_[0.,high],color=COLORS[name],alpha=.20)
        ax.plot(plot_dates,np.r_[0.,historical],color=COLORS['Climatology'],alpha=.5,lw=2.1,ls='--')
        ax.plot(plot_dates,np.r_[0.,mean],color=COLORS[name],lw=2.3)
        ax.scatter(dates.iloc[-1],mean[-1],color=COLORS[name],s=22,zorder=4)
        ax.set_title(name,loc='left',fontsize=12,weight='bold',pad=11)
        ax.text(.02,.96,f'Dec 31 mean {mean[-1]:.1f}  |  10–90% range {low[-1]:.1f}–{high[-1]:.1f}',
                transform=ax.transAxes,va='top',fontsize=9,color=COLORS[name])
        ax.set_ylim(0,ymax);ax.set_xlim(origin,dates.iloc[-1]+pd.Timedelta(days=2))
        ax.set_xticks(pd.to_datetime(['2026-11-05','2026-11-19','2026-12-03','2026-12-17','2026-12-31']))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.grid(axis='y',alpha=.8);ax.set_axisbelow(True)
    for ax in axes[:,0]:ax.set_ylabel('Cumulative station readings (in)')
    for ax in axes[-1]:ax.set_xlabel('Reporting date, 2026')
    handles=[Line2D([0],[0],color='#203c49',lw=2.3,label='Model mean'),
             Patch(facecolor='#6e7880',alpha=.20,label='Model 10–90% predictive range'),
             Line2D([0],[0],color=COLORS['Climatology'],alpha=.5,lw=2.1,ls='--',label='Historical calendar mean (α = 0.5)')]
    fig.suptitle('Vail cumulative outlook versus historical mean',x=.08,y=.98,ha='left',fontsize=19,weight='bold')
    fig.text(.08,.932,'Forecast-only sums: November 5–December 31, 2026 · zero on November 4 · 1,024 sequences per model',fontsize=10,color='#58707c')
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.075,.91),frameon=False,ncol=3,fontsize=9)
    fig.text(.08,.032,'Cumulative sensor proxy; fresh snowfall totals remain unverified. Earlier-season readings are excluded.\n'
             'Bands come from summing each simulated sequence; they are uncalibrated. Historical endpoint: '+f'{historical[-1]:.1f}.',
             fontsize=9,color='#795440')
    fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.15,wspace=.13,hspace=.31)
    save(fig,'forecast_cumulative')


def metric_table(metrics,columns=None):
    columns=columns or ['model','rmse','mae','crps','active_brier','coverage80']
    d=metrics[metrics.leads=='1–60'][columns].sort_values('rmse').copy()
    if 'coverage80' in d:d.coverage80*=100
    return table(d.rename(columns={'model':'Method','rmse':'RMSE (in)','mae':'MAE (in)','crps':'CRPS (in)',
                                   'active_brier':'≥1 in Brier','coverage80':'80% range coverage (%)'}))


def main():
    required=['crossvalidation_metrics.csv','crossvalidation_onset_metrics.csv','crossvalidation_paired_differences.csv','lead_paired_differences.csv',
              'nov_dec_metrics.csv','chronological_metrics.csv','forecast_daily.csv','forecast_climatology.csv',
              'forecast_paths.npz','forecast_manifest.json','frozen_plan.json','final_neural_training.json']
    missing=[name for name in required if not (OUT/name).exists()]
    if missing:raise SystemExit('Waiting for completed model outputs: '+', '.join(missing))
    metrics=pd.read_csv(OUT/'crossvalidation_metrics.csv',dtype={'leads':str})
    novdec=pd.read_csv(OUT/'nov_dec_metrics.csv',dtype={'leads':str})
    chrono=pd.read_csv(OUT/'chronological_metrics.csv',dtype={'leads':str})
    onsets=pd.read_csv(OUT/'crossvalidation_onset_metrics.csv')
    differences=pd.read_csv(OUT/'crossvalidation_paired_differences.csv')
    lead_differences=pd.read_csv(OUT/'lead_paired_differences.csv',dtype={'leads':str})
    forecast=pd.read_csv(OUT/'forecast_daily.csv');climate=pd.read_csv(OUT/'forecast_climatology.csv')
    paths=np.load(OUT/'forecast_paths.npz')
    plan=read_json(OUT/'frozen_plan.json');manifest=read_json(OUT/'forecast_manifest.json')
    neural_training=read_json(OUT/'final_neural_training.json')
    forecast_plot(forecast,climate,manifest);error_plot(metrics);sample_plot(forecast,paths)
    cumulative_plot(forecast,climate,paths,manifest)
    overall=metrics[metrics.leads=='1–60'].set_index('model')
    lead_metrics=metrics.set_index(['model','leads'])
    weather_day1=lead_metrics.loc[('Weather duration','1')]
    climate_day1=lead_metrics.loc[('Climatology','1')]
    seasonal_day3=lead_metrics.loc[('Seasonal duration','1–3')]
    climate_day3=lead_metrics.loc[('Climatology','1–3')]
    weather_day1_ci=lead_differences[(lead_differences.model=='Weather duration')&(lead_differences.reference=='Climatology')&(lead_differences.leads=='1')].iloc[0]
    nn=overall.loc['Neural events'];base=overall.loc['Climatology'];prior=overall.loc['Previous hourly TCN']
    delta=differences[(differences.model=='Neural events')&(differences.metric=='mse')].iloc[0]
    difference_conclusion=('The winter-bootstrap MSE interval includes zero, so the amount-error improvement over climatology remains uncertain.'
                           if delta.ci95_low<=0<=delta.ci95_high else
                           'The winter-bootstrap MSE interval is entirely below zero, favoring the event model on this dataset.' if delta.ci95_high<0 else
                           'The winter-bootstrap MSE interval favors climatology on this dataset.')
    point_improvement=100*(1-nn.rmse/base.rmse)
    prior_improvement=100*(1-nn.rmse/prior.rmse)
    best_rmse=overall.rmse.idxmin();best_crps=overall.crps.idxmin();best_brier=overall.active_brier.idxmin()
    onset_table=onsets.pivot(index='model',columns='window_days',values='brier').reset_index()
    onset_table.columns=['Model','Onset within 1 day Brier','Within 3 days Brier','Within 7 days Brier']
    onset1=onsets[(onsets.model=='Neural events')&(onsets.window_days==1)].iloc[0]
    onset_weather=onsets[(onsets.model=='Weather duration')&(onsets.window_days==1)].iloc[0]
    onset_seasonal=onsets[(onsets.model=='Seasonal duration')&(onsets.window_days==1)].iloc[0]
    chrono_overall=chrono[chrono.leads=='1–60'].set_index('model')
    chrono_nn=chrono_overall.loc['Neural events'];chrono_base=chrono_overall.loc['Climatology']
    onset_winners=[f'{int(w)} day'+('s' if int(w)>1 else '')+': '+r.loc[r.brier.idxmin(),'model']
                   for w,r in onsets.groupby('window_days')]
    selected=[]
    for name in ['fold0','fold1','fold2','chronological','final']:
        sp=read_json(OUT/f'{name}_statistical_selection.json');np_record=read_json(OUT/f'{name}_neural_training.json')
        selected.append({'Split':name,'Weather C':sp['chosen']['regularization'],'Weather decay (days)':sp['chosen']['weather_decay_days'],
                         'Neural width':np_record['selected']['width'],'Neural decay (days)':np_record['selected']['weather_decay_days'],
                         'Neural epochs':np_record['epochs']})
    powder_section=''
    if (OUT/'powder_day_probability.png').exists() and (OUT/'powder_day_summary.csv').exists():
        powder=pd.read_csv(OUT/'powder_day_summary.csv')
        powder=powder[powder.period=='Nov 5–Dec 31'][['model','expected_days','count_q10','count_q90']]
        powder=powder.rename(columns={'model':'Model','expected_days':'Expected ≥5-inch days','count_q10':'Count 10th percentile','count_q90':'Count 90th percentile'})
        powder_section='''<h3>Powder-day probabilities: reading ≥5 inches</h3>
<p>The requested powder-day threshold is inclusive: a reported reading of exactly 5 inches qualifies. Daily probabilities are calculated directly from the restored event-model distributions, using empirical emission tails for the statistical models and a truncated-lognormal mixture tail for the neural model. No model was retrained for this threshold. These are probabilities of the same station-reading proxy; verified fresh snowfall and calibration at this threshold remain unresolved.</p>
<div class="figure"><img src="powder_day_probability.png" alt="Daily probability of a Vail sensor reading at least five inches and expected remaining powder-day count"></div>
<p>Expected counts sum daily probabilities without assuming independent dates. Count ranges come from 1,024 coherent simulated sequences and remain uncalibrated. Historical rates use the same pooled ±15-calendar-day baseline across six partial winters. All counts cover November 5–December 31, excluding earlier observations.</p>'''+table(powder,2)+'''
<p class="caption"><a href="powder_day_forecast.csv">Daily probabilities</a> · <a href="powder_day_summary.csv">Monthly and remaining-period count summaries</a> · <a href="powder_day_probability.svg">Vector figure</a> · <a href="powder_day_manifest.json">Calculation details</a>.</p>'''
    report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vail storm-event model results</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf3f6;color:#203b49;font:16px/1.65 system-ui,-apple-system,sans-serif}}
main{{max-width:1180px;margin:30px auto;padding:46px 54px;background:white;border-radius:16px}}h1{{font-size:39px;line-height:1.15;letter-spacing:-.03em;margin:14px 0 20px}}h2{{font-size:25px;line-height:1.25;margin-top:43px;border-top:1px solid #dce6eb;padding-top:27px}}h3{{font-size:18px;margin-top:27px}}a{{color:#006d8c;text-underline-offset:3px}}.eyebrow{{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:#00768c;font-weight:700}}.lead{{font-size:19px;color:#526b79}}.warning{{background:#fff5e8;padding:20px 24px;border-left:4px solid #bc7a36;border-radius:5px}}.callout{{background:#eef6f8;padding:20px 24px;border-radius:8px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin:24px 0}}.card{{background:#f0f6f8;padding:18px;border-radius:9px}}.number{{display:block;font-size:27px;font-weight:750;line-height:1.3}}.label,.small,.caption{{font-size:13px;color:#607583}}.figure{{margin:23px -10px 12px}}img{{width:100%;height:auto;display:block}}.table-wrap{{overflow:auto;margin:18px 0}}table{{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid #dce6eb}}th{{background:#eef5f7}}li{{margin:8px 0}}details{{background:#f0f6f8;padding:15px 20px;border-radius:8px;margin:18px 0}}summary{{font-weight:650;cursor:pointer}}code{{font-family:ui-monospace,monospace;font-size:.9em;background:#edf3f6;padding:2px 4px}}.downloads{{display:flex;flex-wrap:wrap;gap:10px}}.downloads a{{border:1px solid #bdced6;border-radius:6px;padding:8px 12px;text-decoration:none;font-size:14px}}footer{{font-size:12px;color:#607583;border-top:1px solid #dce6eb;margin-top:40px;padding-top:20px}}@media(max-width:760px){{main{{margin:0;padding:25px 20px}}h1{{font-size:30px}}.cards{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="eyebrow">Fitted event models · November 4, 2026</div>
<h1>Storm sequences, daily probabilities,<br>and the Vail error comparison</h1>
<p class="lead">The event models now generate active episodes and quiet intervals. Their held-out results show how much that structure improves amount forecasts, probability forecasts and event onset.</p>
<div class="warning"><strong>Target qualification:</strong> all values below predict the CAIC CAVMM <code>Sno24</code> reading at 04:00. Reset behavior and sensor offsets remain unresolved. An “active event” means a reading ≥1 inch; a lower reading can still be positive. These are provisional sensor episodes, not verified daily fresh-snow totals or official resort storm reports.</div>
<div class="cards"><div class="card"><span class="number">6 partial winters</span><span class="label">{plan['origins']:,} matched forecast origins</span></div><div class="card"><span class="number">{plan['valid_forecast_pairs']:,}</span><span class="label">forecast-target pairs; {int(nn.unique_days):,} distinct target dates</span></div><div class="card"><span class="number">{manifest['neural_parameters']:,}</span><span class="label">parameters in the neural event model</span></div></div>

<h2>What improved</h2>
<p>The clearest gains are at short leads. The weather-duration model's next-day RMSE is <strong>{weather_day1.rmse:.3f} inches</strong>, versus <strong>{climate_day1.rmse:.3f}</strong> for climatology, a <strong>{100*(1-weather_day1.rmse/climate_day1.rmse):.1f}%</strong> reduction. Across days 1–3, the seasonal duration model has RMSE <strong>{seasonal_day3.rmse:.3f}</strong>, versus <strong>{climate_day3.rmse:.3f}</strong> for climatology. These gains dilute over the full 60-day outlook.</p>
<p>The neural event model has daily RMSE <strong>{nn.rmse:.3f} inches</strong>, compared with <strong>{base.rmse:.3f}</strong> for climatology and <strong>{prior.rmse:.3f}</strong> for the previous hourly TCN. That is a {point_improvement:.1f}% RMSE change in the favorable direction versus climatology and {prior_improvement:.1f}% versus the previous network. {escape(difference_conclusion)}</p>
<p>The lowest point estimates belong to <strong>{escape(best_rmse)}</strong> for RMSE, <strong>{escape(best_crps)}</strong> for CRPS and <strong>{escape(best_brier)}</strong> for the ≥1-inch probability Brier score. Lower is better for all three. These are matched October–April targets at leads 1–60; repeated forecasts of the same day are not independent observations.</p>
<p>Adding complexity is not a consistent advantage. The simple seasonal Markov model has MAE <strong>{overall.loc['Seasonal Markov'].mae:.3f}</strong>, compared with <strong>{nn.mae:.3f}</strong> for neural events. Adding issue-time weather to the duration model changes overall RMSE from <strong>{overall.loc['Seasonal duration'].rmse:.3f}</strong> to <strong>{overall.loc['Weather duration'].rmse:.3f}</strong>, without an improvement. The strongest evidence for that weather correction appears in the immediate onset test below.</p>
{metric_table(metrics)}
<p class="small">CRPS evaluates the entire daily amount distribution. Brier evaluates the probability that a reading reaches 1 inch. The nominal 80% range is not recalibrated using held-out targets; coverage above 80% alone does not prove better calibration because zero mass, width and conditional behavior also matter.</p>
<div class="figure"><img src="error_comparison.png" alt="Matched model RMSE and CRPS with lead-dependent amount and probability scores"></div>
<details><summary>Paired uncertainty relative to climatology</summary>
<p>Differences below are model loss minus climatology loss. Negative values favor the model. Intervals resample whole winter clusters 4,000 times; only six winters are available. They are approximate measures of uncertainty, and several models/scores were inspected. Small probabilistic gains should be treated as tentative rather than a definitive winner.</p>
{table(differences[['model','metric','difference','ci95_low','ci95_high']].rename(columns={'model':'Model','metric':'Score','difference':'Difference','ci95_low':'95% lower','ci95_high':'95% upper'}),digits=5)}
<p>Event-model CRPS uses 256 independent sampled paths per issue with the fair finite-ensemble correction; quantiles use those same paths. Climatology CRPS is exact for its empirical training distribution, and prior-TCN CRPS uses its analytic zero-plus-lognormal distribution. The correction removes finite-ensemble bias in expectation; Monte Carlo error remains a practical limit on very small differences.</p></details>
<details><summary>Short-lead paired uncertainty</summary><p>For weather-duration at day 1, the MSE difference versus climatology is {weather_day1_ci.difference:.3f} in², with a whole-winter bootstrap 95% interval [{weather_day1_ci.ci95_low:.3f}, {weather_day1_ci.ci95_high:.3f}]. The table applies the same paired calculation to short-lead scores. These comparisons are exploratory, use only six winter clusters, and are not adjusted for the number of models, lead ranges or scores examined.</p>
{table(lead_differences[(lead_differences.reference=='Climatology')&(lead_differences.leads!='1–60')][['model','leads','difference','ci95_low','ci95_high']].rename(columns={'model':'Model','leads':'Lead days','difference':'MSE difference (in²)','ci95_low':'95% lower','ci95_high':'95% upper'}),digits=4)}</details>

<h2>Event onset is a separate test</h2>
<p>An onset is a transition from a below-1-inch reading to a reading ≥1 inch. “An active reading tomorrow” can describe continuation of an existing episode; it is not the same event. The onset test counts any such transition within 1, 3 or 7 days and requires the issue reading and all verifying days in that window to be observed.</p>
{table(onset_table,digits=4)}
<p>Best point estimates by window: <strong>{escape('; '.join(onset_winners))}</strong>. The neural model forecasts a one-day onset rate of <strong>{100*onset1.predicted_rate:.1f}%</strong> against an observed <strong>{100*onset1.observed_rate:.1f}%</strong> on the matched onset cohort. Amount and daily-distribution gains therefore do not imply that the neural model has solved event timing.</p>
<p>For onset within one day, weather conditioning reduces Brier from <strong>{onset_seasonal.brier:.4f}</strong> to <strong>{onset_weather.brier:.4f}</strong> relative to the seasonal duration model, a <strong>{100*(1-onset_weather.brier/onset_seasonal.brier):.1f}%</strong> improvement in this sample. Its advantage does not extend consistently across the longer onset windows. This is a predictive comparison of local covariates, not a causal test of frontal dynamics. All event-model onset probabilities use exact state propagation.</p>
<p class="small">There are {int(onset1.origins)} eligible one-day origins; longer windows lose cases with missing readings. Marginal climatology and the previous TCN do not define a directly comparable learned onset process here, so no onset score is invented for them. These onset scores have a different cohort from the all-lead amount table.</p>

<h2>November–December and chronological checks</h2>
<p>The calendar-specific check filters daily predictions by November–December <em>target date</em>. It uses the same saved out-of-fold forecasts, with no retraining or choice of a favorable subset.</p>
{metric_table(novdec)}
<p>The chronological experiment trains on earlier winters and tests 2024/25 and 2025/26. It provides a different deployment-style stress test, not additional independent winters.</p>
{metric_table(chrono)}
<p>In the chronological test, neural-event and climatology RMSE are nearly identical: <strong>{chrono_nn.rmse:.4f}</strong> and <strong>{chrono_base.rmse:.4f}</strong> inches. The lowest CRPS belongs to <strong>{escape(chrono_overall.crps.idxmin())}</strong>. The neural model's nominal 80% range covers <strong>{100*chrono_nn.coverage80:.1f}%</strong> of observations here, showing that uncertainty calibration still needs work. Model rankings depend on the held-out period and score.</p>

<h2>Outlook for November 5–December 31, 2026</h2>
<p>The issue uses observations through <strong>{escape(manifest['issue_source_clock'])}</strong> in CAIC's display clock. Each daily output predicts the next 04:00 endpoint. The gray historical mean has alpha 0.5. The event-model bands come from {manifest['paths_per_model']:,} sampled trajectories.</p>
<div class="figure"><img src="forecast_comparison.png" alt="Daily weather-duration and neural-event model means and predictive ranges through December 31"></div>
<h3>Cumulative station-reading outlook</h3>
<p>The next figure accumulates the forecast readings from November 5 through December 31, with a zero anchor on November 4. Each band is calculated by summing each of the 1,024 coherent simulated sequences and then taking the 10th and 90th percentiles at each date. Daily interval endpoints are not added. The dashed line accumulates the historical calendar mean (alpha 0.5), which pools available observations within ±15 calendar days across six partial winters. It is not the average of complete historical season totals.</p>
<div class="figure"><img src="forecast_cumulative.png" alt="Four Vail event models: cumulative sensor-proxy mean and predictive range versus historical calendar mean"></div>
<p class="caption">These sums describe a cumulative sensor proxy. They are not verified fresh-snowfall totals, and readings before November 5 are excluded. Cumulative predictive ranges remain uncalibrated. <a href="forecast_cumulative.csv">Download cumulative values</a> · <a href="forecast_cumulative.svg">Export vector figure</a> · <a href="forecast_cumulative_manifest.json">Aggregation definitions</a>.</p>
{powder_section}
<p class="caption">The expected line averages over uncertain episode dates. It need not be spiky even when individual simulated trajectories have distinct episodes. Bands are pointwise 10–90% predictive ranges, not confidence intervals for the mean and not simultaneous whole-season guarantees.</p>
<div class="figure"><img src="event_sample_paths.png" alt="Three fixed-index sampled event sequences from each of the weather-duration and neural models"></div>
<p class="caption">These are the first three fixed-seed sample paths from each model, not handpicked storms. The images show possible sequences; they do not assert that the visible peaks will occur on those calendar dates. No monthly fresh-snow total is published because endpoint readings are not established accumulation increments.</p>

<h2>What was actually trained</h2>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Arrival/persistence structure</th><th>Amount distribution</th></tr></thead><tbody>
<tr><td>Climatology</td><td>Training-only ±15 calendar-day neighborhood.</td><td>Empirical daily readings; no learned episode state.</td></tr>
<tr><td>Previous hourly TCN</td><td>Frozen earlier selected architecture and epochs, regenerated and checked.</td><td>Zero-plus-lognormal daily marginals.</td></tr>
<tr><td>Seasonal Markov</td><td>Current threshold state plus seasonal calendar.</td><td>Training-only state/calendar empirical donors.</td></tr>
<tr><td>Seasonal duration</td><td>Current state plus time in state and season.</td><td>Empirical donors conditioned on state, season and episode age where supported.</td></tr>
<tr><td>Weather duration</td><td>Seasonal duration hazard plus issue-time weather correction, decaying with lead.</td><td>The same age/state/calendar empirical donor construction.</td></tr>
<tr><td>Neural events</td><td>168-hour TCN encoder plus a state/age/lead decoder; weather representation attenuates with lead.</td><td>Below 1 inch: zero plus Beta; at least 1 inch: lower-truncated lognormal.</td></tr>
</tbody></table></div>
<p>The three statistical event models use regularized logistic transition probabilities. Duration-aware models track two states and ages 0–14: age 0 means unknown/censored duration, while age 14 is a continuing 14+ tail bin. It does not force an event to end after 14 days. Statistical amount donors use a ±45-day calendar neighborhood, with active-age groups 1 day, 2 days, 3+ days or unknown; pools with fewer than 20 donors fall back to broader state/season groups.</p>
<p>The neural event model has six residual blocks with 12 causal convolutions. Its final width is {neural_training['selected']['width']}, selected training length {neural_training['epochs']} epochs and parameter count {manifest['neural_parameters']:,}. Unlike the previous independent daily decoder, its forecast evolves a distribution over threshold state and duration. Means and one-day probabilities use exact finite-state propagation, so apparent RMSE gains cannot come from a fortunate Monte Carlo average.</p>
<p>These are <strong>discrete daily observed-state event models</strong>. They are not a fitted continuous-time Hawkes process, a hidden-state model, or a model assigning one normalized total to each smooth storm pulse. The daily proxy cannot justify hour-precise storm labels. This experiment tests the first implementable duration/occurrence component of the proposed event approach.</p>
<p>The hourly inputs include temperature, dew point, humidity, wind, available pressure/precipitation and inter-station temperature differences. All are known by issuance. Four sites at different elevations are not a vertically collocated sounding. No issued upstream weather-model fields were added, so this experiment does not isolate the value of detecting an approaching front from a regional atmospheric forecast.</p>

<h2>Validation, missing data and limits</h2>
<p>Three shuffled whole-winter outer folds exactly reuse the earlier Vail partitions. Each outer training set has a separate inner validation winter. The weather-duration model chooses between two predeclared penalty/decay settings using freely simulated first-seven-day activity Brier score; the neural model chooses between two configurations and early stopping using a masked conditional sequence likelihood. All published forecast scores use free-running predictions.</p>
{table(pd.DataFrame(selected),digits=3)}
<p>The statistical transitions use {plan['one_step_training_examples']:,} eligible one-step examples across all six winters, with fit folds receiving only their own winter rows. The neural model learns from the same {plan['origins']} origin sequences used in the prior experiment; overlapping target appearances do not create new independent storms. There are {plan['missing_issue_readings']} origins with a missing issue reading and {plan['unknown_issue_ages']} with unknown duration. Missing initial state is integrated over a training-only seasonal prior; unknown age stays unknown until a state transition establishes a start.</p>
<p>Missing target days remain missing and do not become zero. Neural training uses observed previous states within each training sequence; across missing gaps it is a masked conditional score rather than a fully marginalized missing-state likelihood. This approximation is disclosed, and teacher-forced errors are not reported as forecasts. The source quarantines remain March 20–May 31, 2020 and October 1–November 28, 2023, applied to inputs and targets.</p>
<p>The neural future inputs retain known calendar/lead and lagged ENSO covariates with training-only safeguards for unsupported values. Historical RONI values come from a revised record, not archived release vintages, so this is not a fully issue-time-realistic ENSO backtest. The current strong positive ENSO indicator lacks examples in these six winters, so the experiment cannot establish that effect. Standardization, emission donors, missing-state priors, stopping and hyperparameter choices are confined to the appropriate training partitions.</p>
<p>The tests cover statistical baselines and the previous local neural model. They do not establish superiority to operational NWP or human snowfall forecasts. Source-clock UTC offset, historical publication latency and the mapping to manual resort snowfall remain unresolved. More realistic event sequences alone are not sufficient evidence of better advance forecasting.</p>

<h2>The next useful experiment</h2>
<p>First reconcile the sensor with verified resort snowfall and its reset schedule. Then add archived, as-issued regional weather forecasts: upstream moisture, temperature profiles, winds and frontal position can contain information before a front reaches these four stations. Keep the same complete-winter validation, and test whether those inputs improve onset within 1–3 days before increasing network size.</p>
<p>For the event model itself, test onset calibration and a joint episode-level duration/amount distribution against the present age-conditioned daily emissions. The neural onset bias and chronological interval undercoverage identify concrete targets. Any new correction must be fitted on training/validation winters only; the current results should remain the frozen comparison.</p>

<h2>Files and scientific context</h2>
<p>The observations originate in the <a href="https://stations.avalanche.state.co.us/precip.php">CAIC station archive</a>, which identifies Vail Resort as the provider. The model design was informed by <a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2009WR008602">duration-based precipitation models</a> and <a href="https://arxiv.org/abs/1909.12127">conditional event-time modeling</a>. The implemented daily episode model is a tailored experiment, not a reproduction of those papers. See the <a href="../vail-storm-review/storm_model_review.html">literature review and proposed redesign</a> and <a href="../vail-hourly/vail_hourly_report.html">previous Vail benchmark</a>.</p>
<div class="downloads"><a href="forecast_daily.csv">Daily event-model predictions</a><a href="forecast_comparison.png">Forecast figure</a><a href="event_sample_paths.png">Event scenarios</a><a href="crossvalidation_metrics.csv">Daily error table</a><a href="crossvalidation_onset_metrics.csv">Onset errors</a><a href="crossvalidation_paired_differences.csv">Paired error uncertainty</a><a href="nov_dec_metrics.csv">November–December errors</a><a href="chronological_metrics.csv">Chronological errors</a><a href="forecast_manifest.json">Forecast provenance</a><a href="frozen_plan.json">Experiment plan</a><a href="neural_event_model.pt">Neural checkpoint</a><a href="statistical_event_models.joblib">Statistical models</a></div>
<footer>Experimental predictions of a reported sensor endpoint. Source SHA-256: <code>{escape(manifest['source_sha256'])}</code><br>No official resort snowfall claim; no verified monthly fresh-snow totals.</footer>
</main></body></html>'''
    (OUT/'storm_model_report.html').write_text(report)
    print(json.dumps(dict(report=str(OUT/'storm_model_report.html'),neural_rmse=float(nn.rmse),climatology_rmse=float(base.rmse),previous_rmse=float(prior.rmse),forecast_days=len(climate))))


if __name__=='__main__':main()
