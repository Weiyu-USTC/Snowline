"""Source-audited hourly mountain temperature comparisons and forecast curves."""
from pathlib import Path
import html
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/mountain-temperature'
SITES={'vail_snotel':'Vail Mountain','stowe_mountain':'Stowe · west-slope forest station'}
COLORS={'Raw HRRR':'#74828c','Recent station bias':'#cb9845','Weather-only ridge':'#7968a7','Ridge correction':'#3b8b8c','Weather-only boosted trees':'#c25b61','Boosted tree correction':'#075e67'}


def paired_gain(predictions,baseline,model,split='test'):
    d=predictions[predictions.split.eq(split)]
    a=d[d.model.eq(baseline)].set_index(['issue_utc','valid_time_utc'])
    b=d[d.model.eq(model)].set_index(['issue_utc','valid_time_utc'])
    z=pd.DataFrame({'baseline_sq':(a.predicted_temperature_c-a.observed_temperature_c)**2,
                    'model_sq':(b.predicted_temperature_c-b.observed_temperature_c)**2}).dropna().groupby('issue_utc').agg(['sum','count'])
    a=z.baseline_sq['sum'].to_numpy();b=z.model_sq['sum'].to_numpy();n=z.model_sq['count'].to_numpy()
    ix=np.random.default_rng(712).integers(0,len(z),(3000,len(z)))
    difference=np.sqrt(a[ix].sum(1)/n[ix].sum(1))-np.sqrt(b[ix].sum(1)/n[ix].sum(1))
    return dict(baseline=baseline,model=model,rmse_reduction_c=float(np.sqrt(a.sum()/n.sum())-np.sqrt(b.sum()/n.sum())),
        confidence95_low_c=float(np.quantile(difference,.025)),confidence95_high_c=float(np.quantile(difference,.975)),
        issue_dates=len(z),bootstrap_unit='complete forecast issue dates',iterations=3000)


def make_figures(site,title,pred,audit):
    selected=audit['selected'];test=pred[pred.split.eq('test')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                        'axes.edgecolor':'#aebec1','axes.labelcolor':'#294950','text.color':'#183b43'})
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for model,g in test.groupby('model',sort=False):
        error=g.predicted_temperature_c-g.observed_temperature_c
        curve=g.assign(sq=error**2).groupby('forecast_hour').sq.mean().pow(.5)
        axes[0].plot(curve.index,curve,label=model,color=COLORS[model],linewidth=2 if model==selected else 1.2,
                     alpha=1 if model in ['Raw HRRR',selected] else .65)
    axes[0].set(xlabel='Native HRRR forecast lead (hours)',ylabel='Held-out temperature RMSE (°C)',title='Error across the held-out winter')
    # Select by observed temperature swing, never by the chosen model's score.
    raw=test[test.model.eq('Raw HRRR')]
    date=raw.groupby('issue_utc').observed_temperature_c.agg(lambda s:s.max()-s.min()).idxmax()
    case=test[test.issue_utc.eq(date)]
    actual=case[case.model.eq('Raw HRRR')].sort_values('valid_time_utc')
    axes[1].plot(pd.to_datetime(actual.valid_time_utc,utc=True),actual.observed_temperature_c,color='#142c34',linewidth=2,label='Observed station temperature')
    for model in ['Raw HRRR',selected]:
        g=case[case.model.eq(model)].sort_values('valid_time_utc')
        axes[1].plot(pd.to_datetime(g.valid_time_utc,utc=True),g.predicted_temperature_c,color=COLORS[model],linewidth=1.8,
                     linestyle='--' if model=='Raw HRRR' else '-',label=model)
    axes[1].set(title='Largest observed temperature swing in test set',xlabel='Valid time (UTC)',ylabel='Temperature (°C)')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M',tz='UTC'))
    for ax in axes:ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    fig.suptitle(title+' · hourly temperature verification',fontsize=16,fontweight='bold')
    fig.savefig(OUT/f'{site}_test_curves.png',dpi=160);plt.close(fig)
    current=pd.read_csv(OUT/f'{site}_current.csv')
    fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained')
    t=pd.to_datetime(current.valid_time_utc,utc=True)
    ax.plot(t,current.raw_hrrr_temperature_c,color=COLORS['Raw HRRR'],linestyle='--',linewidth=1.7,label='Raw HRRR')
    ax.plot(t,current.corrected_temperature_c,color=COLORS[selected],linewidth=2.2,label='Selected temperature correction')
    ax.axhline(0,color='#93a9ad',linewidth=.8)
    ax.set(title=title+' · current hourly temperature guidance',xlabel='Valid time (UTC)',ylabel='Temperature (°C)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M',tz='UTC'));ax.grid(alpha=.15);ax.legend(frameon=False)
    issue=str(current.issue_utc.iloc[0]);fig.suptitle('Issued '+issue+' · station location, not resort-wide snowfall',fontsize=9)
    fig.savefig(OUT/f'{site}_current_curve.png',dpi=160);plt.close(fig)


def main():
    parts=[];summary=[]
    for site,title in SITES.items():
        audit=json.loads((OUT/f'{site}_audit.json').read_text())
        pred=pd.read_csv(OUT/f'{site}_predictions.csv.gz')
        # The dashboard serves CSV downloads, but deliberately does not expose
        # compressed archive extensions. Keep the training archive compressed.
        public=ROOT/'outputs/nwp-weather/forecast-portfolio/mountain-temperature'
        public.mkdir(parents=True,exist_ok=True)
        plain_predictions=gzip.decompress((OUT/f'{site}_predictions.csv.gz').read_bytes())
        (public/f'{site}_predictions.csv').write_bytes(plain_predictions)
        (OUT/f'{site}_predictions.csv').write_bytes(plain_predictions)
        metrics=pd.read_csv(OUT/f'{site}_metrics.csv')
        overall=paired_gain(pred,'Raw HRRR',audit['selected'])
        isolated=paired_gain(pred,'Weather-only ridge','Ridge correction')
        isolated_tree=paired_gain(pred,'Weather-only boosted trees','Boosted tree correction')
        (OUT/f'{site}_observation_gain.json').write_text(json.dumps(dict(overall=overall,matched_ridge_observation_gain=isolated,matched_tree_observation_gain=isolated_tree),indent=2))
        make_figures(site,title,pred,audit)
        table=metrics.pivot(index='model',columns='split',values='rmse_c').reindex(metrics.model.drop_duplicates())[['train','validation','test']]
        table.columns=['Training RMSE °C','Validation RMSE °C','Test RMSE °C']
        table=table.round(3)
        chosen=audit['selected']
        parts.append(f'<section><h2>{html.escape(title)}</h2><p>Validation selected <strong>{html.escape(chosen)}</strong>. '
            f'Test: {overall["issue_dates"]} complete forecast dates. Overall RMSE reduction versus raw HRRR: '
            f'<strong>{overall["rmse_reduction_c"]:.3f}°C</strong> '
            f'(95% date-block bootstrap interval {overall["confidence95_low_c"]:.3f} to {overall["confidence95_high_c"]:.3f}°C).</p>'
            f'<p>The matched ridge comparison isolates the added station observations from ordinary weather calibration: '
            f'RMSE reduction <strong>{isolated["rmse_reduction_c"]:.3f}°C</strong> '
            f'(95% interval {isolated["confidence95_low_c"]:.3f} to {isolated["confidence95_high_c"]:.3f}°C). '
            f'The matched boosted-tree comparison gives {isolated_tree["rmse_reduction_c"]:.3f}°C '
            f'(95% interval {isolated_tree["confidence95_low_c"]:.3f} to {isolated_tree["confidence95_high_c"]:.3f}°C). '
            'An interval spanning zero does not establish a repeatable benefit. Positive numbers favor adding observations.</p>'+table.to_html(border=0)+
            f'<img src="{site}_test_curves.png" alt="Held-out temperature errors and a temperature event">'
            f'<img src="{site}_current_curve.png" alt="Current HRRR and corrected station temperature">'
            f'<p><a href="{site}_metrics.csv">Metrics</a> · <a href="{site}_predictions.csv">All test/training predictions</a> · '
            f'<a href="{site}_audit.json">Evaluation audit</a> · <a href="{site}_observation_gain.json">Paired uncertainty</a></p></section>')
        summary.append(dict(site=site,selected=chosen,overall_gain=overall,isolated_observation_gain=isolated,isolated_tree_observation_gain=isolated_tree))
    body=''.join(parts)
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Mountain observations: hourly temperature verification</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{font:16px/1.55 system-ui;color:#183b43;background:#f2f5f1;margin:0}}main{{max-width:1130px;margin:auto;padding:36px 22px}}h1{{font-size:34px;line-height:1.1}}h2{{font-size:26px}}section{{background:white;padding:25px;margin:24px 0;border-radius:15px}}p{{max-width:1000px}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:8px;text-align:right;border-bottom:1px solid #e2e8e5}}th:first-child{{text-align:left}}a{{color:#126d78}}.note{{background:#e2eeea;padding:18px;border-radius:12px}}</style>
<main><p>SNOWLINE · RESEARCH VALIDATION</p><h1>Do mountain observations improve hourly forecasts?</h1>
<p class="note">These experiments verify <strong>air temperature</strong> against actual mountain stations. They establish neither hourly fresh-snow accuracy nor a snowfall benefit from temperature adjustment. Snowpack depth and SWE are not hourly snowfall labels.</p>
<p>Fit winters2020/21–2021/22; choose model family and settings on2022/23; report2023/24 separately after selection. These2023/24 records are held out from this fit and selection, but have been used in other project experiments: they are a reused retrospective benchmark, not a never-seen project-wide test. A future winter is needed for prospective confirmation. Forecast issue dates are frozen independently of storms. Within each date, all models use identical observed hours. Uncertainty resamples entire issue dates to preserve correlation among forecast hours.</p>
{body}
<section><h2>How the comparison is kept causal</h2><p>The HRRR run starts06:00UTC and is assumed available09:00UTC. Forecasts are issued10:50UTC; only valid endpoints after issue are scored. The native lead range is5–48hours (early2020 HRRR only reached36hours), so this experiment does not pretend that an already-aged run covers48new hours. Past station observations are eligible only after an assumed one-hour reporting allowance. Actual historical receipt timestamps are unavailable, and historical quality control may differ from the original live data.</p>
<p>Raw HRRR is compared with a recent-bias correction whose amplitude and decay are validation-selected; a weather-only ridge calibration; the same ridge calibration with lagged station readings; and a boosted-tree correction with the same observation features. Current curves use models refitted on training plus validation winters. Test tables and confidence intervals retain the exact model fitted on training winters only.</p>
<h2>Station and source audit</h2><ul>
<li><strong>Vail:</strong> <a href="https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=842&state=CO">NRCS Vail Mountain SNOTEL842</a>,10290ft. Fixed PST is converted toUTC+8hours; suspect/missing flags are excluded. Temperature sensor bias and provisional hourly values remain limitations. SWE/depth/cumulative precipitation are retained as states, not differenced into fresh snow. <a href="https://www.nrcs.usda.gov/state-offices/montana/montana-snow-survey/frequently-asked-snow-survey-questions-montana">Official clock/reporting documentation</a>.</li>
<li><strong>Stowe:</strong> <a href="https://www.uvm.edu/femc/data/archive/project/mt-mansfield-west-slope-mid-elevation/dataset/mansfield-west-met-site/metadata">FEMC Mount Mansfield West</a>, a mid-elevation forest station around816m, not the summit. Genuine15minute averages are matched at hourly endpoints. <a href="https://www.uvm.edu/femc/data/live">FEMC documents fixed EST without daylight saving</a>. The unheated rain gauge is covered in winter and is excluded from precipitation features. Source maintenance notes document shared temperature/RH probe and freezing-weather logger failures. Temperature requires valid companionRH, and the suspect logger era20August2020–26February2021 is conservatively quarantined through its documented replacement. This source-based rule applies to features and targets; it was fixed from training-data inspection and maintenance notes, not optimized against test scores. The archive also has battery-related outages.</li>
<li><strong>Alta:</strong> public <a href="https://mesowest.utah.edu/cgi-bin/droman/meso_base.cgi?stn=CLN">CollinsCLN</a> historical hourly pages provide temperature, wet-bulb temperature, liquid precipitation and total depth. ExplicitGMT headers establish the clock. Its resetting “Snow interval” is excluded as an hourly snowfall label. These inputs support a separate Alta12hour snowfall ablation.</li>
<li><strong>Jay Peak:</strong> no independently verified public summit hourly snowfall archive was obtained. MVL airport weather remains a regional proxy and is never relabelled as a Jay Peak mountain observation.</li></ul>
<p>FEMC attribution: Forest Ecosystem Monitoring Cooperative, University of Vermont, Quality-Controlled Mount Mansfield West Meteorological Data,2017–current. Data licenseCCBY-NC-SA4.0; this is noncommercial research use. Preserve attribution and applicable share-alike conditions if redistributing data.</p>
<p>Reproducible data, quality notes and hashes: <a href="../../data/mountain_hourly/manifest.json">source manifest</a> · <a href="../../data/mountain_hourly/coverage.csv">coverage</a>.</p></section></main></html>'''
    (OUT/'index.html').write_text(report)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
