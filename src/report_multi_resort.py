"""Render audited multi-site training, forecast and source-comparison results."""
from pathlib import Path
import json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/multi-resort'
SITES={'alta':'Alta–Collins','mansfield':'Mount Mansfield NOAA','jay':'Jay Peak NOAA (1,840 ft)'}
COLORS={'Climatology':'#6b7280','Markov + ENSO':'#059669','Duration + ENSO':'#2563eb','Weather duration + ENSO':'#8b5cf6','Neural events':'#d97706'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'figure.dpi':140,'savefig.dpi':180})

def table(frame):return frame.to_html(index=False,border=0,float_format=lambda v:f'{v:.3f}',escape=True)
def source(url,label):return f'<a href="{url}">{label}</a>'

def accumulation_metrics():
    rows=[]
    for site in SITES:
        for split in ['crossvalidation','chronological']:
            p=pd.read_csv(OUT/site/f'{split}_predictions.csv')
            for horizon in [30,60]:
                r=p[p.lead<=horizon].groupby(['model','winter','origin']).agg(n=('actual_inches','count'),actual=('actual_inches','sum'),predicted=('predicted_inches','sum'))
                r=r[r.n==horizon].reset_index()
                for model,g in r.groupby('model'):
                    e=g.predicted-g.actual
                    rows.append(dict(site=site,split=split,horizon_days=horizon,model=model,complete_origins=len(g),winters=g.winter.nunique(),rmse_total_inches=np.sqrt(np.mean(e**2)),mae_total_inches=np.mean(abs(e)),bias_total_inches=e.mean()))
    result=pd.DataFrame(rows);result.to_csv(OUT/'accumulation_metrics.csv',index=False)
    return result

def main():
    accumulation=accumulation_metrics()
    bundles={};summary=[];forecast_summary=[];cohorts=[];scores=[];flag_rows=[]
    for site,title in SITES.items():
        dest=OUT/site
        b={n:pd.read_csv(dest/f'{n}.csv') for n in ['crossvalidation_metrics','chronological_metrics','crossvalidation_paired','forecast_daily','forecast_totals']}
        b['plan']=json.loads((dest/'plan.json').read_text());b['manifest']=json.loads((dest/'forecast_manifest.json').read_text());b['training']=json.loads((dest/'final_training.json').read_text());b['paths']=np.load(dest/'forecast_paths.npz')
        overall=b['crossvalidation_metrics'].query("leads == '1–60'").sort_values('rmse');best=overall.iloc[0];event=overall[overall.model!='Climatology'].iloc[0];b['best']=best.model;b['event']=event.model
        baseline=overall[overall.model=='Climatology'].iloc[0];chrono=b['chronological_metrics'].query("leads == '1–60'").sort_values('rmse');ci=b['crossvalidation_paired'].query("model == @event.model and metric == 'mse'").iloc[0]
        summary.append({'site':title,'lowest_cv_rmse':best.model,'cv_rmse_inches':best.rmse,'climatology_rmse_inches':baseline.rmse,'best_event_model':event.model,'event_rmse_inches':event.rmse,'event_mse_difference':ci.difference,'event_mse_difference_ci95_low':ci.ci95_low,'event_mse_difference_ci95_high':ci.ci95_high,'chronological_lowest_rmse':chrono.iloc[0].model,'chronological_rmse_inches':chrono.iloc[0].rmse})
        for model in dict.fromkeys(['Climatology',event.model,'Neural events']):
            z=b['forecast_totals'].query("model == @model").copy();z.insert(0,'site',title);forecast_summary.append(z)
        coverage=pd.read_csv(dest/'winter_coverage.csv').query('selected')
        cohorts.append({'site':title,'winters':len(b['plan']['years']),'first_winter':min(b['plan']['years']),'last_winter':max(b['plan']['years']),'forecast_origins':b['plan']['origins'],'scored_origin_day_pairs':b['plan']['pairs'],'unique_observed_Nov_Apr_days':int(coverage.observed_nov_apr.sum()),'presumed_zero_days':int(coverage.presumed_zero_nov_apr.sum()),'neural_parameters':b['training']['parameters']})
        z=overall.copy();z.insert(0,'site',title);scores.append(z)
        p=pd.read_csv(dest/'crossvalidation_predictions.csv',parse_dates=['date']);t=pd.read_csv(dest/'target_daily.csv',parse_dates=['date']).set_index('date');p['presumed_zero']=p.date.map(t.noaa_presumed_zero).fillna(False).astype(bool)
        for name,g in p.groupby('model'):
            for policy,r in [('main: include NOAA presumed zeros',g),('sensitivity: exclude P targets only',g[~g.presumed_zero])]:
                flag_rows.append({'site':title,'model':name,'evaluation_policy':policy,'pairs':len(r),'rmse':float(np.sqrt(np.mean((r.predicted_inches-r.actual_inches)**2))),'powder_brier':float(np.mean((r.p_powder-(r.actual_inches>=5))**2))})
        b['daily']=b['forecast_daily'];b['daily']['date']=pd.to_datetime(b['daily'].date);bundles[site]=b
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'model_selection_summary.csv',index=False)
    forecasts=pd.concat(forecast_summary,ignore_index=True);forecasts.to_csv(OUT/'forecast_summary.csv',index=False)
    cohorts=pd.DataFrame(cohorts);cohorts.to_csv(OUT/'training_cohorts.csv',index=False)
    scores=pd.concat(scores,ignore_index=True);scores.to_csv(OUT/'model_comparison.csv',index=False)
    pd.DataFrame(flag_rows).to_csv(OUT/'noaa_presumed_zero_sensitivity.csv',index=False)

    for kind in ['cumulative','daily','powder']:
        fig,axes=plt.subplots(1,3,figsize=(15.5,5.1),sharex=True)
        for ax,(site,title) in zip(axes,SITES.items()):
            b=bundles[site];d=b['daily'];chosen=b['event'];q=d[d.model==chosen];dates=q.date;paths=b['paths'][chosen]
            if kind=='cumulative':low,high=np.quantile(paths.cumsum(1),[.1,.9],axis=0);column='cumulative_mean_inches';ylabel='Snowfall since November 5 (inches)'
            elif kind=='daily':low,high=q.q10_inches,q.q90_inches;column='mean_inches';ylabel='Daily new snowfall (inches)'
            else:column='p_powder';ylabel='Probability of ≥5 inches (%)'
            if kind!='powder':ax.fill_between(dates,low,high,color='#2563eb',alpha=.13,label='Event model 10–90% range')
            for name,color,alpha,label,lw in [('Climatology','#374151',.5,'Historical calendar mean',2.6),(chosen,'#2563eb',1.,f'Best event: {chosen}',2.)]+([] if chosen=='Neural events' else [('Neural events','#d97706',.95,'Neural event model',1.7)]):
                r=d[d.model==name];values=r[column]*(100 if kind=='powder' else 1)
                ax.plot(r.date,values,color=color,alpha=alpha,lw=lw,label=label,linestyle='--' if name=='Climatology' else '-')
            ax.set_title(title,fontweight='bold');ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO,interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));ax.tick_params(axis='x',rotation=20);ax.legend(loc='upper left',fontsize=7.5)
            if b['best']=='Climatology':ax.text(.98,.02,'Historical baseline has lowest CV RMSE',transform=ax.transAxes,ha='right',fontsize=8,color='#374151')
        heading={'cumulative':'Cumulative snowfall outlook · November 5–December 31, 2026','daily':'Daily snowfall outlook · November 5–December 31, 2026','powder':'Daily powder probability · ≥5 inches in the reporting day'}[kind]
        fig.suptitle(heading,fontsize=16,fontweight='bold',y=.99)
        fig.text(.5,.02,'Issued November 4 · NOAA weather through November 1 or earlier · Unknown current local snowfall state · No future weather-model input',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.065,1,.93]);fig.savefig(OUT/f'{kind}_outlook.png');fig.savefig(OUT/f'{kind}_outlook.svg');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14.5,4.9))
    for ax,(site,title) in zip(axes,SITES.items()):
        b=bundles[site];a=b['crossvalidation_metrics'].query("leads == '1–60'").set_index('model');c=b['chronological_metrics'].query("leads == '1–60'").set_index('model');names=list(COLORS);positions=np.arange(len(names))
        ax.barh(positions-.18,a.loc[names,'rmse'],height=.35,color='#2563eb',label='Shuffled whole-winter CV')
        ax.barh(positions+.18,c.loc[names,'rmse'],height=.35,color='#9ca3af',label='Last three winters holdout')
        ax.set_yticks(positions,names,fontsize=8);ax.invert_yaxis();ax.set_xlabel('Daily RMSE (inches; lower is better)');ax.set_title(title,fontweight='bold');ax.legend(fontsize=7,loc='lower right');ax.set_xlim(left=0)
    fig.suptitle('Five model families tested on unseen winters',fontsize=16,fontweight='bold');fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(OUT/'model_comparison.png');plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for ax,(site,title) in zip(axes,SITES.items()):
        b=bundles[site];q=b['daily'].query("model == 'Neural events'");paths=b['paths']['Neural events'];ax.plot(q.date,paths[0],color='#a3bffa',alpha=.75,lw=1,label='One simulated event sequence');ax.plot(q.date,q.mean_inches,color='#d97706',lw=2,label='Neural expected daily snowfall');ax.set_title(title,loc='left');ax.set_ylabel('Inches');ax.legend(fontsize=8);ax.set_ylim(bottom=0)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));fig.suptitle('Spiky snowfall scenarios, smooth probability-weighted means',fontsize=15,fontweight='bold');fig.text(.5,.015,'The simulated sequence illustrates the event model; its storm dates are not an additional forecast.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(OUT/'event_scenarios.png');plt.close(fig)

    findings=[]
    for r in summary.to_dict('records'):
        sig='The event model improves MSE with a winter-bootstrap interval below zero.' if r['event_mse_difference_ci95_high']<0 else 'The event-model difference is not securely below the historical baseline.'
        findings.append(f"<li><b>{r['site']}:</b> lowest shuffled-CV RMSE: {r['lowest_cv_rmse']} ({r['cv_rmse_inches']:.3f} inches); climatology {r['climatology_rmse_inches']:.3f}. {sig} The chronological winner is {r['chronological_lowest_rmse']}.</li>")
    sections=[]
    for site,title in SITES.items():
        b=bundles[site];z=b['forecast_totals'].query("period == 'Nov5–Dec31'")[['model','expected_snowfall_inches','q10_inches','q90_inches','expected_powder_days']]
        sections.append(f'<h3>{title}</h3>'+table(z)+f'<p>Best event candidate: <b>{b["event"]}</b>. Neural model: {b["training"]["parameters"]:,} parameters, selected configuration {html.escape(str(b["training"]["neural_selected"]))}, refitted for {b["training"]["neural_epochs"]} epochs. Lagged RONI supplied: {b["manifest"]["current_lagged_roni"]:.2f}; neural training range: {b["manifest"]["neural_roni_train_range"]}. Current weather-context missing fraction: {b["manifest"]["context_missing_fraction"]:.1%}.</p><p><a href="{site}/forecast_daily.csv">Daily forecasts</a> · <a href="{site}/forecast_totals.csv">Monthly and period totals</a> · <a href="{site}/crossvalidation_metrics.csv">Metrics by lead</a> · <a href="{site}/crossvalidation_paired.csv">Paired winter-bootstrap comparisons</a> · <a href="{site}/nov_dec_metrics.csv">November–December target-date scores</a> · <a href="{site}/plan.json">Data and evaluation plan</a></p>')
    report='''<!doctype html><html><head><meta charset="utf-8"><title>Alta and Vermont snowfall experiments · November 2026</title><style>body{max-width:1200px;margin:40px auto;padding:0 25px;color:#183044;background:#fcfcfd;font:16px/1.6 system-ui}h1,h2,h3{line-height:1.2}h2{margin-top:45px}p,li{max-width:1050px}img{width:100%;border:1px solid #e5e7eb;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}th,td{padding:8px 10px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}a{color:#1d4ed8}.note{padding:16px;background:#eff6ff;border-left:4px solid #2563eb}code{font-size:13px}</style></head><body><h1>Alta and Vermont snowfall experiments</h1><p>Issued November 4, 2026 · Forecast period November 5–December 31 · Amounts in inches of daily new snowfall.</p>'''
    report+='<div class="note"><b>Interpretation:</b> These are experimental seasonal outlooks, not forecasts of exact storm dates. The target snowfall archives stop in 2024 for Alta–Collins, 2017 for Mansfield and 2014 for Jay. All historical tests withhold current local snowfall state to match that limitation. Current lagged NOAA weather is available, but future numerical weather forecasts are absent.</div><h2>Which model worked best?</h2><ul>'+''.join(findings)+'</ul><img src="model_comparison.png" alt="Model error comparison">'+table(scores[['site','model','rmse','mae','crps','powder_brier','coverage80']])
    report+='<p>RMSE ranks the expected daily amount. CRPS scores the full daily distribution; Brier scores powder-day probabilities. These can rank models differently. The winner is the lowest point estimate among five candidates, not a proven universal winner. Selection itself introduces optimism; the separate chronological comparison is essential. The 95% paired intervals resample whole winters (4,000 draws), so overlapping forecast dates are not treated as independent years.</p>'
    report+='<h3>Does the ranking hold for two-month totals?</h3><p>We also sum each complete 60-day forecast and compare it with the observed sum. Missing target days disqualify an origin from this check; they are never counted as zero. Alta’s Markov model again has the lowest RMSE. Mansfield’s historical baseline wins this metric. Jay’s duration model looks better, but only27 complete origins from6 winters remain, so that result has limited support and a different observation cohort. These are supplementary metrics; they did not retune or change the daily-RMSE model selection used in the plots.</p>'+table(accumulation.query("split == 'crossvalidation' and horizon_days == 60")[['site','model','complete_origins','winters','rmse_total_inches']])+'<p><a href="accumulation_metrics.csv">30/60-day accumulation scores, including chronological holdout</a></p>'
    report+='<h2>Rest-of-2026 outlook</h2><p>Blue shows the lowest-CV-RMSE <em>event</em> candidate, orange the neural event model, and gray the historical calendar mean with alpha 0.5. The baseline remains visible even when it outperforms every trained event candidate. Shading is a model-based 10–90% predictive range from 2,048 joint sequences, not a confidence interval for the mean and not calibrated to 80% coverage. Cumulative bands are quantiles of cumulative paths; daily quantiles are never summed.</p><img src="cumulative_outlook.png" alt="Cumulative snowfall"><img src="daily_outlook.png" alt="Daily snowfall"><img src="powder_outlook.png" alt="Powder probability">'+''.join(sections)
    report+='<p>Climatology is a ±15-calendar-day empirical baseline pooled over eligible training winters. It has daily probability distributions but no fitted joint temporal distribution; cumulative range cells are therefore blank. The expected number of powder days is the sum of daily probabilities and need not be an integer.</p><h2>How the event models work</h2><p>Markov models retain a wet/dry threshold state; duration models also track how long the current state has persisted. The threshold defining the active state is ≥1 inch; powder probability uses ≥5 inches. Zero and sub-inch amounts remain part of the model. The weather-duration model adds summaries of lagged temperature and precipitation, with their influence decaying over the forecast horizon. All trained models include calendar and lagged ENSO inputs.</p><p>The neural model has six residual temporal blocks (12 causal convolutions) over a 60-day, three-variable NOAA weather history plus missingness and recency channels. An autoregressive duration-dependent decoder predicts event transitions, zero/sub-inch amounts, and a truncated lognormal snowfall tail. It is trained by conditional likelihood; forecast means and probabilities integrate over unknown event states, and future paths use model-generated events. A single simulated sequence is spiky even when the expected curve is smooth:</p><img src="event_scenarios.png" alt="Event scenarios">'
    report+='<h2>Training and validation</h2>'+table(cohorts)+'<ul><li>Separate model at each site. Alta target: paired 12-hour manual Collins readings ending 04:00 local standard time. Mansfield/Jay targets: NOAA GHCN daily SNOW at their named stations. Jay weather predictor: NOAA Island Pond; the target remains the lower-elevation Jay cooperative station.</li><li>Whole winters are shuffled into three outer folds. The last two available training winters form an inner validation split for two neural configurations and two weather-model configurations. Final models are refitted using all eligible winters. A second test trains on earlier winters and holds out the latest three eligible winters.</li><li>Origins are weekly, October–February, with 60 future daily targets. The weather history ends two days before issuance. No target snowfall or snow depth enters the input. No observed future weather is used. Missing future labels are masked; origins need at least 45 observed targets. Weather records have a two-day date lag, but true historical publication timestamps and revision vintages are unavailable.</li><li>NOAA quality-flagged values are excluded. NOAA measurement flag P is a presumed zero, and these values are included as supplied. Other missing values are never filled with zero. Excluding P from scoring changes the target sample and often removes dry days; the sensitivity table is descriptive and is not a separately retrained unbiased benchmark.</li><li>ENSO uses revised historical RONI with a conservative calendar lag. Current November uses JAS; it is frozen across this forecast. Magnitudes are clipped to training support and unseen constant indicator features are held at their training value. This is not a causal estimate of ENSO influence.</li><li>Neural fits run up to 30 epochs with inner-validation selection. Several fits reach the cap; the two-configuration search is a reproducible demo, not exhaustive architecture optimization. Statistical emissions use training-only historical amount distributions.</li></ul><p><a href="noaa_presumed_zero_sensitivity.csv">Presumed-zero evaluation sensitivity</a> · <a href="training_cohorts.csv">Training cohort table</a> · <a href="model_selection_summary.csv">Model-selection summary</a> · <a href="forecast_summary.csv">Forecast summary</a></p>'
    report+='<h2>Do resorts inflate their numbers?</h2><p>The available comparison does <b>not</b> identify a defensible inflation rate. Jay has no complete October–May NOAA seasons overlapping its official annual archive. Three seasons with at least 95% coverage give official total / observed NOAA sum ratios of <b>1.61–1.99</b>; missing snow makes these incomplete-denominator comparisons, and measurement location/window differ. NOAA’s Jay cooperative record was itself collected at the resort, so NOAA hosting does not guarantee an independent observer.</p><p>No official Stowe annual archive was verified. Five third-party backup seasons show ratios of <b>1.18–1.43</b> and descriptive correlation <b>r=0.954</b>, each with 1–5 missing NOAA days. This is suggestive seasonal co-variation, not a fitted or validated correction. Stowe’s new-snow board is around 3,000 ft, while the Mansfield station is near 3,900 ft. We apply no multiplier to these NOAA forecasts and do not label them summit-resort predictions.</p><p><a href="reporting-ratios/report.html">Full reporting comparison, coverage and source audit</a>. Resort figures never entered model fitting, tuning or model selection.</p>'
    report+='<h2>Sources and reproducibility</h2><ul>'+''.join('<li>'+source(url,label)+'</li>' for url,label in [('https://hive.utah.edu/records/280qy-4ee96','University of Utah: Alta–Collins manual observations 2000–2023'),('https://hive.utah.edu/records/6hdjr-stw17','University of Utah: Alta–Collins 2023/24 extension'),('https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt','NOAA GHCN-Daily documentation and flags'),('https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00435416.dly','NOAA Mount Mansfield daily record'),('https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00434189.dly','NOAA Jay Peak daily record'),('https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00434120.dly','NOAA Island Pond regional weather record'),('https://www.uvm.edu/water/news/talking-snow-stowe-mountain-resort','UVM: Stowe and Mansfield measurement locations')])+'</ul><p>Raw snapshots, flags, source hashes, split membership, selected hyperparameters, checkpoints, exact daily forecast means, and joint forecast paths are retained with the project. Run <code>python3 src/train_multi_resort.py --site all --mode run</code>, then <code>python3 src/report_multi_resort.py</code>. This reuses the checked-in data snapshots. Alta’s research dataset carries a noncommercial license; retain its license and attribution in any later distribution.</p></body></html>'
    (OUT/'report.html').write_text(report)
    print(summary.to_string(index=False));print(forecasts.query("period == 'Nov5–Dec31'").to_string(index=False))

if __name__=='__main__':main()
