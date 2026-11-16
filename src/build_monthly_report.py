#!/usr/bin/env python3
"""Build a static, portable monthly snowfall research report from actual results."""
from __future__ import annotations

import base64
import html
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
MONTHS = ['nov', 'dec', 'jan', 'feb', 'mar', 'apr']
LABELS = {'climatology':'Monthly historical mean', 'ridge':'Ridge', 'analog':'Shrunk analogs',
          'extra_trees':'ExtraTrees', 'neural_mlp':'Small neural network', 'selected':'Inner-selected procedure'}

CSS = """
:root{--ink:#142e40;--muted:#526b7a;--teal:#176b87;--line:#dae5e9;--pale:#edf5f7;--orange:#a65a2d}
*{box-sizing:border-box}body{margin:0;background:#eef2f4;color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1140px;margin:30px auto;background:white;box-shadow:0 8px 30px #142e4010}header{padding:48px 60px;background:var(--ink);color:white;border-top:7px solid #7fbac8}.eyebrow{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:#b0dce6}
h1{font-size:clamp(32px,4.4vw,49px);line-height:1.1;letter-spacing:-.03em;margin:16px 0 20px}.dek{font-size:20px;line-height:1.5;color:#e6f0f4;max-width:900px}.date{font-size:13px;color:#b6cbd5;margin-top:24px}.body{padding:30px 60px 50px}
h2{font-size:26px;line-height:1.25;margin:36px 0 15px;letter-spacing:-.02em}h3{font-size:19px;margin:23px 0 12px}p{margin:0 0 15px}a{color:var(--teal);text-underline-offset:3px}.small{font-size:13px;color:var(--muted)}.callout{border-left:4px solid var(--orange);background:#fcf5ee;padding:18px 22px;margin:20px 0}.callout p:last-child{margin:0}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:25px}.card{padding:18px;background:var(--pale);border-top:3px solid var(--teal)}.card b{display:block;font-size:29px;line-height:1.2;margin:7px 0}.card span{font-size:12px;text-transform:uppercase;letter-spacing:.04em}.card small{color:var(--muted);font-size:13px;display:block}
.table-wrap{overflow-x:auto;margin:19px 0}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:10px 11px;border-bottom:1px solid var(--line);vertical-align:top}thead th{background:var(--pale);font-size:12px;color:var(--muted)}figure{margin:26px -25px;padding:12px;background:#f6f9fc;border:1px solid var(--line)}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:var(--muted);padding:10px 5px 0}ul,ol{padding-left:23px}li{margin:8px 0}code{font-size:.9em;background:#edf3f5;padding:2px 4px}.links{border-top:1px solid var(--line);padding-top:18px;margin-top:30px;font-size:14px}.links a{display:inline-block;margin-right:18px;margin-bottom:8px}footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px;margin-top:25px}
@media(max-width:760px){main{margin:0}header{padding:32px 24px}.body{padding:24px}.cards{gap:8px}.card{padding:12px}.card b{font-size:23px}.card span{font-size:10px}figure{margin:20px -16px}table{font-size:12px}th,td{padding:8px}.dek{font-size:18px}}
@media(max-width:460px){.cards{grid-template-columns:1fr}.card b{font-size:25px}}
@media print{@page{size:A4;margin:14mm}body{background:white;font-size:10pt;line-height:1.4}main{margin:0;box-shadow:none}header{background:#edf5f7;color:var(--ink);padding:20px 25px}.dek,.eyebrow,.date{color:var(--muted)}h1{font-size:28pt}.dek{font-size:12pt}.body{padding:10px 0}h2{font-size:17pt;break-after:avoid}figure{margin:18px 0}figure,.callout,.cards,tr{break-inside:avoid}table{font-size:9pt}thead{display:table-header-group}.small,figcaption{font-size:8.5pt}.links{display:none}}
"""


def read(name):
    return json.loads((OUT/name).read_text())


def number(value, digits=1):
    return f'{float(value):.{digits}f}'


def table(headers, rows):
    head=''.join(f'<th scope="col">{html.escape(str(v))}</th>' for v in headers)
    body=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def metric(data, scope, family):
    return next(v for v in data['metrics'] if v['scope']==scope and v['family']==family)


def chart(forecast, baseline, copper):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/copper-monthly-report-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'#f6f9fc','axes.facecolor':'#f6f9fc'})
    fig, axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    x=np.arange(6)
    points=np.array([forecast['point_inches'][m] for m in MONTHS])
    lower=np.array([forecast['monthly_lower'][m] for m in MONTHS])
    upper=np.array([forecast['monthly_upper'][m] for m in MONTHS])
    axes[0].vlines(x,lower,upper,color='#b4ccd5',linewidth=6,alpha=.8,label='Exploratory jackknife+ range (α = 0.10)')
    axes[0].scatter(x,points,color='#176b87',s=45,zorder=4,label='Experimental ENSO-only model')
    axes[0].plot(x,[baseline[m] for m in MONTHS],color='#aa5b2e',marker='s',linestyle='--',label='Monthly historical mean')
    axes[0].set(xticks=x,xticklabels=[m.title() for m in MONTHS],ylabel='Monthly new snowfall (inches)',
                ylim=(0,max(upper)*1.15),title='Copper 2026/27: October remains unverified')
    axes[0].legend(loc='upper center',fontsize=8,frameon=False,ncol=1)
    families=['climatology','ridge','analog','extra_trees','neural_mlp','selected']
    y=np.arange(len(families))
    a=[metric(copper,'october_only_all_six',f)['rmse_inches'] for f in families]
    b=[metric(copper,'sequential_next_month',f)['rmse_inches'] for f in families]
    axes[1].scatter(a,y-.12,s=45,color='#176b87',label='All six months issued after October')
    axes[1].scatter(b,y+.12,s=45,color='#aa5b2e',marker='s',label='Next month using actual growing prefix')
    axes[1].set(yticks=y,yticklabels=[LABELS[f] for f in families],xlabel='Held-out monthly RMSE (inches; lower is better)',
                title='Additional context has not established skill',xlim=(19.7,20.5))
    axes[1].invert_yaxis()
    axes[1].set_ylim(6.6,-.5)
    axes[1].legend(loc='lower right',fontsize=8,frameon=False)
    fig.suptitle('Monthly snowfall: forecast uncertainty and separate validation tasks',fontsize=15,weight='bold')
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=160);plt.close(fig)
    encoded=base64.b64encode(buffer.getvalue()).decode('ascii')
    return '<figure><img alt="Monthly forecasts with wide uncertainty and model errors for two separate prediction tasks." src="data:image/png;base64,'+encoded+'"><figcaption>Ranges apply to the experimental ENSO-only predictions, not simultaneous six-month coverage. Validation uses 20 independent seasons; 120 monthly errors are not 120 independent winters.</figcaption></figure>'


def tabpfn_section():
    path=OUT/'monthly_tabpfn_results.json'
    if not path.exists():
        return '<p>The separate TabPFN benchmark has not yet produced a completed result file. No TabPFN performance claim is made in this report version.</p>',None
    result=json.loads(path.read_text())
    rows=[]
    for row in result.get('metrics',[]):
        mode={'observed_prefix':'Observed monthly prefix','october_unverified_enso_only':'ENSO only; October missing'}.get(row['mode'],row['mode'])
        scope={'october_issued_all_six':'Initial six-month outlook','sequential_next_month':'Growing prefix → next month'}.get(row['scope'],row['scope'])
        family={'tabpfn_v2':'TabPFN-v2','climatology':'Monthly mean'}.get(row['family'],row['family'])
        rows.append((html.escape(str(mode)),html.escape(str(scope)),
                     html.escape(str(family)),row.get('n_seasons','—'),
                     number(row['mae_inches'],2),number(row['rmse_inches'],2)))
    content='<p><strong>TabPFN-v2 was run locally as a genuine in-context challenger.</strong> Its synthetic-pretrained neural weights remain frozen. Each omitted-season fit supplies labeled historical rows as context; there is no snowfall-specific gradient training. The fixed two-member CPU configuration is evaluated separately and is not part of the nested family selector above.</p>'
    content+=table(['Inputs','Prediction task','Model','Seasons','MAE · inches','RMSE · inches'],rows)
    content+='<p>The frozen in-context model also fails to improve the baseline on these tests. Its completed experiment is useful evidence about this dataset; it does not invalidate in-context learning on richer or better-matched tasks.</p>'
    content+='<p class="small">Built with PriorLabs-TabPFN. The explicit legacy v2 checkpoint was used, not the newer default. Package/model versions and checkpoint hashes are preserved in the result file. Model weights use the <a href="../licenses/TabPFN-v2.txt">Prior Labs 1.1 license</a>, which differs from newer weights. This is a retrospective independent-season test, not evidence the model existed at those historical issue dates.</p>'
    content+='<p><a href="monthly_tabpfn_results.json">Full TabPFN results and forecasts</a> · <a href="monthly_tabpfn_predictions.csv">Every held-out TabPFN prediction</a></p>'
    return content,result


def regional_section():
    path=OUT/'monthly_copper_regional_october_results.json'
    if not path.exists():
        return ''
    result=json.loads(path.read_text())
    forecast=next(iter(result['forecasts'].values()))
    b=metric(result,'october_only_all_six','climatology')
    s=metric(result,'october_only_all_six','selected')
    content=f'<h3>Using observed regional October snow to predict Copper</h3><p>A separate model pairs <strong>actual Dillon October snowfall as a regional predictor</strong> with Copper November–April targets. This uses the new online observation without calling it Copper snowfall. It has {result["n_seasons"]} matched historical seasons, one fewer than the main experiment. Its selected-procedure RMSE is {number(s["rmse_inches"],2)} inches versus {number(b["rmse_inches"],2)} for the mean on that same cohort, so it remains an experimental sensitivity.</p>'
    content+=table(['Copper target month','Regional-input estimate · inches','Exploratory jackknife+ range'],[
        (month.title(),number(forecast['point_inches'][month]),number(forecast['monthly_lower'][month],0)+'–'+number(forecast['monthly_upper'][month],0)) for month in MONTHS])
    config=forecast['selected_config']
    content+=f'<p class="small">The full-data choice is {html.escape(config["family"])} with penalty α = {number(config["alpha"],0)} and ENSO {"included" if config["enso"] else "omitted"} by the inner validation. This is a different training cohort and predictor definition; it is not a corrected October observation. <a href="monthly_copper_regional_october_results.json">Regional experiment and source definition</a>.</p>'
    return content


def main():
    copper=read('monthly_copper_results.json')
    missing=read('monthly_copper_missing_october_results.json')
    dillon=read('monthly_dillon_results.json')
    current=json.loads((ROOT/'data/current_2026_inputs.json').read_text())
    forecast=next(iter(missing['forecasts'].values()))
    history=pd.read_csv(ROOT/'data/copper_monthly_user.csv')
    history=history.loc[history.start_year.isin(copper['years'])]
    baseline=history[MONTHS].mean().to_dict()
    forecast_table=table(['Month','Historical mean · inches','Experimental ENSO-only · inches','Experimental jackknife+ range'],[
        (month.title(),number(baseline[month]),number(forecast['point_inches'][month]),
         number(forecast['monthly_lower'][month],0)+'–'+number(forecast['monthly_upper'][month],0)) for month in MONTHS])
    comparison=[]
    for family in LABELS:
        a=metric(copper,'october_only_all_six',family)
        b=metric(copper,'sequential_next_month',family)
        comparison.append((LABELS[family],number(a['mae_inches'],2)+' / '+number(a['rmse_inches'],2),
                           number(b['mae_inches'],2)+' / '+number(b['rmse_inches'],2)))
    model_table=table(['Model family','October → all six · MAE / RMSE','Growing prefix → next month · MAE / RMSE'],comparison)
    method_table=table(['Trained locally','Hyperparameters / controls tested'],[
        ('Monthly historical mean','No fitted predictor effects; one mean per target month.'),
        ('Ridge regression','Penalty α ∈ {1, 10, 100}; ENSO features included or omitted.'),
        ('Shrunk analogs','Gaussian distance bandwidth ∈ {0.5, 1.5, 4}; fixed 20% uniform-weight mixture; ENSO included or omitted.'),
        ('ExtraTrees','32 trees; minimum leaf size ∈ {3, 6}; ENSO included or omitted.'),
        ('Neural MLP','One hidden layer of 4 tanh units; penalty α ∈ {1, 20}; L-BFGS, at most 250 iterations; ENSO included or omitted.')])
    tab_section,tab_result=tabpfn_section()
    regional=regional_section()
    mb=metric(missing,'october_only_all_six','climatology')
    ms=metric(missing,'october_only_all_six','selected')
    db=metric(dillon,'october_only_all_six','climatology')
    ds=metric(dillon,'october_only_all_six','selected')
    total=forecast['remaining_total_interval']
    current_dillon=next(iter(dillon['forecasts'].values()))
    figure=chart(forecast,baseline,copper)
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Copper Mountain · Monthly snowfall research</title><style>{CSS}</style></head><body><main>
<header><div class="eyebrow">2026/27 monthly outlook · Research demo</div><h1>Forecast each month as the season unfolds</h1><p class="dek">The expanding-prefix system is trained and tested. New context can update the prediction, but these small-sample tests have not established a reliable improvement over monthly historical averages.</p><div class="date">Issued for research November 3, 2026 · Copper Mountain, Colorado · November–April snowfall</div></header>
<div class="body"><div class="cards"><div class="card"><span>Historical Copper sample</span><b>{copper['n_seasons']} seasons</b><small>Whole seasons held out together</small></div><div class="card"><span>October 2026 at Copper</span><b>Unverified</b><small>No stale zero or nearby station substituted</small></div><div class="card"><span>Current climate context</span><b>Strong El Niño</b><small>JAS RONI +{number(current['enso']['observed_indices_2026']['roni']['JAS'])}°C; outside Copper training range</small></div></div>

<h2>What can be forecast today?</h2>
<p>The <a href="https://www.coppercolorado.com/the-mountain/mountain-safety-stats/winter-mountain-stats/">official monthly table</a> has not published October 2026. The live <a href="https://www.coppercolorado.com/the-mountain/conditions-weather/snow-report/">snow report</a> displays zeros beside a May 3, 2026 timestamp and the prior season’s closing message. Those are stale/reset fields, not October observations. A forecast conditioned on measured Copper October therefore cannot yet be claimed.</p>
<p>The table gives two transparent estimates: historical monthly means as the conservative reference, and a separately trained ENSO-only forecast that explicitly omits unknown October. The latter is experimental, with no demonstrated forecast advantage. All values measure new snowfall in inches; they are not snow depth or water equivalent.</p>
{forecast_table}
<p>The historical means sum to <strong>{number(sum(baseline.values()),0)} inches</strong>; the experimental forecast sums to <strong>{number(forecast['nov_apr_total_inches'],0)} inches</strong>. Its exploratory seasonal-total range is {number(total[0],0)}–{number(total[1],0)} inches. Monthly interval endpoints should not be added: the total range is calculated from seasonal residuals. May is excluded because only three usable historical May observations are available in the main cohort.</p>
<div class="callout"><p><strong>Uncertainty remains large.</strong> These omitted-season jackknife+ ranges are nominal 90% marginal bands (α = 0.10) under exchangeable-season assumptions, not empirically established conditional calibration or simultaneous coverage for all six months. Revised indices, unusual current ENSO, small sample size and measurement uncertainty limit their interpretation.</p></div>
{figure}

<h2>Your October → November → December idea is implemented</h2>
<p>At the first origin, October and the available climate index predict November. After actual November snowfall is reported, October–November predict December. The prefix expands through March to predict April. Each origin can also predict all remaining months directly. Actual future months never enter an earlier forecast, and predicted snowfall is never relabeled as a new observation.</p>
<p>Two evaluations stay separate: <strong>October → all six months</strong> evaluates a season outlook issued with only October known; <strong>growing prefix → next month</strong> evaluates monthly updates after each new observation arrives. The second procedure cannot provide its later actual inputs today. Hypothetical October inputs from 0 to 40 inches are saved as conditional scenarios, with no scenario asserted to be observed.</p>
<p>This is sequential conditioning. A pretrained in-context learner goes further: it receives historical labeled seasons as prediction-time context while its neural weights remain fixed. A small network gradient-trained on Copper’s rows is a different experiment; both are distinguished below.</p>

<h2>Training, tuning and the independent-season test</h2>
<p>Following your assumption, seasons are treated as independent and exchangeable. Every outer evaluation omits one entire season. Hyperparameters and the family selector are chosen using shuffled four-fold validation only inside the remaining seasons. The tuning objective is RMSE across all remaining months at that origin; next-month accuracy is also reported separately. Feature and target scaling remain inside training folds. All target labels from a held-out winter stay outside the training data; its already observed prefix is permitted as query input. This evaluates interpolation across comparable seasons, not chronological deployment.</p>
{method_table}
<p class="small">When included, ENSO features are the lagged continuous RONI and indicators at ≥+1.5°C and ≤−1.5°C. The lag advances from JAS through DJF as the observed prefix grows. These thresholds are modeling features, not complete official ENSO event definitions. MLP fits were attempted under the stated iteration limit; universal optimizer convergence is not asserted.</p>
{model_table}
<p class="small">Entries are MAE / RMSE in inches; lower is better. Each column uses 120 monthly predictions across 20 independent seasons, not 120 independent winters. “Inner-selected” evaluates the entire model-selection procedure. Comparing many outer results and then declaring the smallest one a confirmed winner would be optimistic.</p>
<p>The nonlinear analog, tree and neural candidates do not demonstrate a dependable gain. Strong regularization makes many neural fits nearly identical to monthly means; differences at numerical precision are not predictive discoveries. For the actual missing-October task, the selected procedure has RMSE {number(ms['rmse_inches'],2)} versus {number(mb['rmse_inches'],2)} for monthly means. Its all-data final choice is ENSO-only ridge with α = {number(forecast['selected_config']['alpha'],0)}, producing the experimental outlook above.</p>

<h2>A genuine pretrained in-context challenger</h2>
{tab_section}
<p>The <a href="https://www.nature.com/articles/s41586-024-08328-6">TabPFN research</a> and <a href="https://arxiv.org/abs/2501.02945">TabPFN-TS study</a> motivate this small-table approach. The current general-purpose software default is newer than the explicitly pinned v2 experiment. A published benchmark on other datasets is not evidence of Copper snowfall skill.</p>

<h2>How El Niño enters the forecast</h2>
<p>The <a href="https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml">October 8 NOAA discussion</a> reports strengthening El Niño. Its <a href="https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/strengths/">strength forecast</a> assigns 97% probability to strong or very strong conditions in December–February. This is an ENSO probability, not a probability of heavy snowfall at Copper.</p>
<p>The comparable issue-time predictor is published JAS RONI +1.7°C. It exceeds the Copper training maximum of +1.5°C, so the current climate context involves extrapolation. ASO is now published, but silently replacing a JAS-trained input with ASO would change the feature definition. Future winter indices are not yet observed. To learn from NOAA’s future strength probabilities directly, the next dataset must include archived forecasts from matching historical issue dates, rather than substitute final winter labels. <a href="https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/">NOAA RONI</a>.</p>

<h2>Second data source and the next experiments</h2>
<p>Dillon 1 E has a verified October 2026 observation of <strong>{number(current['dillon_october']['value_inches'],2)} inches</strong>, with 31/31 usable daily observations. Its separate 41-season experiment gives selected-procedure RMSE {number(ds['rmse_inches'],2)} versus {number(db['rmse_inches'],2)} for its monthly means. The separate current Dillon estimate totals {number(current_dillon['nov_apr_total_inches'],0)} November–April inches. Dillon’s lower elevation and different snowfall are not a replacement for Copper’s missing measurement. <a href="https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00052281.dly">NOAA daily source</a>.</p>
{regional}
<p>Useful further challengers include <a href="https://github.com/amazon-science/chronos-forecasting">Chronos-2</a>, <a href="https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/">TimesFM 3</a> and the lightweight <a href="https://mistral.ai/news/timestral/">Timestral</a>. These are research candidates, not models claimed to have run here. Their multivariate and covariate support is relevant, but one observed month remains little context. Do not create an artificial April-to-October monthly sequence or fill unobserved summer months with zero.</p>
<p>A stronger next dataset would combine consistent daily resort observations, archived climate forecasts and multiple mountains. A shared model can borrow information across locations while holding out entire years across all sites, then test on a future season. Before adopting a model, require improvement over the monthly baseline, reliable uncertainty, and an explanation of which information actually contributes.</p>

<div class="links"><strong>Reproduce and inspect</strong><br><a href="forecast_2026_27.csv">2026/27 forecast table</a><a href="forecast_2026_27.json">Forecasts and uncertainty</a><a href="monthly_copper_results.json">Copper results and scenarios</a><a href="monthly_copper_missing_october_results.json">Current ENSO-only forecast</a><a href="monthly_copper_predictions.csv">Held-out predictions</a><a href="../data/current_2026_inputs.json">Current source audit</a><a href="../src/monthly.py">Training code</a><a href="../research/monthly_methods.txt">Methods and papers</a></div>
<footer>Static research report generated from saved model outputs. No external scripts, fonts or images are required. Data-source corrections and missingness policies from the first demo remain in force, including exclusion of the unresolved 2025/26 Copper row. Latest model weights are not implied by a legacy-model result. Research date: November 3, 2026.</footer></div></main></body></html>'''
    path=OUT/'monthly_report.html';path.write_text(text,encoding='utf-8')
    print(f'Wrote {path} ({len(text.encode()):,} bytes); TabPFN included: {tab_result is not None}')


if __name__=='__main__':
    main()
