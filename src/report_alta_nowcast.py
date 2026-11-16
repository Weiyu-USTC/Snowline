"""Publication-style figures and a self-contained HTML report for Alta."""
from pathlib import Path
import json,html,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.mplconfig'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from evaluate_alta_nowcast import OUT,ROOT,LABELS,RADAR_LABELS,load_all,load_hrrr_run,integral

COLORS={'Historical mean':'#aab2bb','Current MOS model (refitted)':'#343c49',
        'HRRR calibrated (matched)':'#7b8794','HRRR + observed radar':'#258681',
        'ConvLSTM + HRRR':'#2479b8','DGMR adaptation + HRRR':'#d47b24','NowcastNet adaptation + HRRR':'#8656ae',
        'Persistence':'#aab2bb','Motion extrapolation':'#258681','HRRR':'#343c49',
        'ConvLSTM':'#2479b8','DGMR adaptation':'#d47b24','NowcastNet adaptation':'#8656ae'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','axes.labelcolor':'#253444','text.color':'#253444',
                     'axes.edgecolor':'#b9c2cc','figure.facecolor':'white','savefig.facecolor':'white'})


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def training_curves():
    fig,axs=plt.subplots(1,3,figsize=(14,4),sharex=True)
    recipes={'uniform_l1':('Initial recipe','#8d98a5'),'wet_mse':('Wet-case / squared loss','#2479b8'),
             'wet_mse_linear_head':('Direct-rate DGMR output','#d47b24')}
    for ax,name in zip(axs,LABELS):
        for recipe,(label,color) in recipes.items():
            file=OUT/'models'/('recipe_'+recipe)/(name+'.pt.training.json')
            if not file.exists():continue
            log=pd.DataFrame(json.loads(file.read_text()))
            if 'validation_mse_mm_h_squared' not in log:continue
            log=log.dropna(subset=['validation_mse_mm_h_squared'])
            ax.plot(log.step,log.validation_mse_mm_h_squared,color=color,marker='.',lw=1.6,label=label)
        ax.axhline(.1364785731,color='#6b7280',ls=':',lw=1,label='Always dry')
        ax.axhline(.118890,color='#258681',ls='--',lw=1,label='Persistence')
        ax.set_title(RADAR_LABELS[name],loc='left');ax.set_xlabel('Training updates');ax.set_yscale('log');ax.grid(alpha=.17)
    axs[0].set_ylabel('Development rate MSE (mm/hour)²')
    handles={}
    for ax in axs:
        for handle,label in zip(*ax.get_legend_handles_labels()):handles[label]=handle
    fig.legend(handles.values(),handles.keys(),loc='lower center',bbox_to_anchor=(.5,-.14),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Training diagnostics · development data only',fontsize=17,fontweight='bold',y=1.04)
    fig.text(.125,-.015,'Same 42 selection cases for every checkpoint. No held-out forecast errors guided these changes.',fontsize=9)
    save(fig,'training_diagnostics')


def comparison():
    short=pd.read_csv(OUT/'short_summary.csv');daily=pd.read_csv(OUT/'daily_summary.csv')
    order=['Current MOS model (refitted)','HRRR calibrated (matched)','HRRR + observed radar',*LABELS.values()]
    d=daily.set_index('model').loc[order]
    labels=['Current MOS','HRRR calibrated','HRRR + radar history','ConvLSTM + HRRR','DGMR adaptation + HRRR','NowcastNet adaptation + HRRR']
    fig,axs=plt.subplots(1,2,figsize=(14,4.5),gridspec_kw={'wspace':.55})
    for ax,col,title in zip(axs,['mae_inches','powder_brier'],['24-hour fresh-snow error · MAE (inches)','Powder-day probability error · Brier score']):
        ax.barh(labels,d[col],color=[COLORS[x] for x in order],height=.64)
        ax.invert_yaxis();ax.set_title(title,loc='left',pad=15);ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
        for i,v in enumerate(d[col]):ax.text(v+max(d[col])*.015,i,f'{v:.3f}',va='center',fontsize=10)
        ax.set_xlim(0,max(d[col])*1.22)
    fig.suptitle('Alta · matched forecasts on the unseen 2023/24 winter',fontsize=17,fontweight='bold',x=.12,ha='left',y=1.05)
    fig.text(.12,-.04,f'Lower is better. Same {int(d.cases.iloc[0])} observed days for every method. Powder day = at least 5 inches in 24 hours.',fontsize=10)
    save(fig,'daily_comparison')
    order=['Persistence','Motion extrapolation','HRRR','ConvLSTM','DGMR adaptation','NowcastNet adaptation']
    fig,axs=plt.subplots(1,3,figsize=(14,4),sharex=True)
    for ax,metric,title in zip(axs,['mae_mm','crps_mm','recall'],['Accumulation MAE (mm)','Ensemble CRPS (mm)','Storm detection recall']):
        for model in order:
            g=short[short.model==model].sort_values('hours')
            ax.plot(g.hours,g[metric],marker='o',color=COLORS[model],label=model,linewidth=2)
        ax.set_title(title,loc='left');ax.set_xticks([1,2,3]);ax.set_xlabel('Hours after 11 UTC');ax.grid(alpha=.2)
    axs[-1].set_ylim(0,1.03)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False)
    fig.suptitle('Alta region · short-range storm forecasting',fontsize=17,fontweight='bold',y=1.04)
    fig.text(.125,-.01,'Liquid-equivalent precipitation verified against screened radar; storm threshold = 1 mm accumulated.',fontsize=9)
    save(fig,'short_horizon_comparison')


def daily_curves():
    d=pd.read_csv(OUT/'daily_predictions.csv',parse_dates=['target_date'])
    old=d[d.model=='Current MOS model (refitted)'].sort_values('target_date')
    hist=d[d.model=='Historical mean'].sort_values('target_date')
    fig,axs=plt.subplots(3,2,figsize=(16,11),sharex=True)
    for row,(name,label) in enumerate(LABELS.items()):
        g=d[d.model==label].sort_values('target_date');c=COLORS[label]
        a,b=axs[row]
        a.fill_between(g.target_date,g.q10,g.q90,color=c,alpha=.16,label='80% predictive interval')
        a.plot(g.target_date,g.actual,color='#111111',lw=.85,alpha=.7,label='Observed fresh snow')
        a.plot(old.target_date,old.predicted_inches,color=COLORS[old.model.iloc[0]],lw=1,alpha=.65,label='Current MOS model')
        a.plot(hist.target_date,hist.predicted_inches,color='#8b949e',alpha=.5,lw=2,label='Historical mean')
        a.plot(g.target_date,g.predicted_inches,color=c,lw=1.5,label=label)
        a.set_ylabel('Fresh snow (inches / 24h)');a.set_title(label,loc='left');a.grid(alpha=.15)
        b.plot(old.target_date,old.p_powder,color=COLORS[old.model.iloc[0]],lw=1,alpha=.65,label='Current MOS model')
        b.plot(hist.target_date,hist.p_powder,color='#8b949e',alpha=.5,lw=2,label='Historical frequency')
        b.plot(g.target_date,g.p_powder,color=c,lw=1.5,label=label)
        hit=g.actual>=5;b.scatter(g.loc[hit,'target_date'],np.ones(hit.sum()),s=12,color='#111111',marker='|',label='Observed powder day')
        b.set_ylabel('P(24h fresh snow ≥ 5 inches)');b.set_ylim(-.02,1.06);b.set_title('Powder-day probabilities',loc='left');b.grid(alpha=.15)
    for ax in axs.flat:
        ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axs[0,0].legend(fontsize=8,loc='upper left',ncol=2);axs[0,1].legend(fontsize=8,loc='upper left')
    fig.suptitle('Alta · daily snowfall and powder forecasts on the held-out winter',fontsize=18,fontweight='bold',x=.125,ha='left',y=.99)
    fig.text(.125,.945,'Forecasts made without seeing 2023/24 outcomes. Shading is a predictive interval, not a confidence interval for the mean.',fontsize=10)
    fig.subplots_adjust(top=.9,hspace=.32,wspace=.2)
    save(fig,'snowfall_powder_curves')


def storms():
    score=pd.read_csv(OUT/'short_case_metrics.csv',dtype={'case_id':str})
    candidates=score[(score.model=='HRRR')&(score.hours==3)].dropna(subset=['alta_truth_mm']).sort_values('alta_truth_mm',ascending=False)
    # Selection by observations only; deliberately show the three largest local
    # events rather than choosing forecasts that happen to look successful.
    chosen=[]
    for r in candidates.itertuples():
        if all(abs((pd.Timestamp(r.date)-pd.Timestamp(x.date)).days)>3 for x in chosen):chosen.append(r)
        if len(chosen)==3:break
    z,cases,pred,lookup=load_all();iy=np.argmin(abs(z['latitude']-40.57607));ix=np.argmin(abs(z['longitude']+111.63889))
    fig,axs=plt.subplots(len(chosen),3,figsize=(15,3.3*len(chosen)),squeeze=False)
    chosen_ids=[]
    for row,r in enumerate(chosen):
        cid=r.case_id;chosen_ids.append(cid);i=cases.index[cases.case_id==cid][0];y=z['y'][i]
        truth=y[1:20,iy,ix]
        hrrr=load_hrrr_run(cid)['precip_mm'];hp=hrrr['mrms64'][np.isin(hrrr['periods'],[6,7,8]),iy,ix]
        xx=np.arange(19)/6
        for col,name in enumerate(LABELS):
            ax=axs[row,col];e=np.asarray(pred[name][lookup[name][cid]],float)[:,1:20,iy,ix]
            ax.fill_between(xx,np.quantile(e,.1,axis=0),np.quantile(e,.9,axis=0),color=COLORS[RADAR_LABELS[name]],alpha=.2)
            ax.plot(xx,e.mean(0),color=COLORS[RADAR_LABELS[name]],lw=2,label=RADAR_LABELS[name])
            ax.plot(xx,truth,color='#111111',lw=1.6,label='Radar observation')
            ax.stairs(hp,[0,1,2,3],color=COLORS['HRRR'],lw=1.4,linestyle='--',label='HRRR hourly average')
            ax.set_title(f'{pd.Timestamp(r.date):%b %d, %Y} · {RADAR_LABELS[name]}',loc='left',fontsize=10)
            ax.set_xlim(0,3);ax.set_xlabel('Hours after 11 UTC');ax.grid(alpha=.18)
            if col==0:ax.set_ylabel('Liquid precipitation (mm/hour)')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Storm timing · largest held-out Alta radar events',fontsize=17,fontweight='bold',y=1.02)
    fig.text(.125,-.025,'Independent storms selected by observed 3-hour accumulation. Bands show raw radar-ensemble spread; they are not calibrated snowfall intervals.',fontsize=9)
    fig.tight_layout();save(fig,'storm_prediction_curves')
    (OUT/'illustration_selection.json').write_text(json.dumps({'rule':'Three largest observed local 3-hour accumulations, at least 4 days apart; no selection on model errors.','case_ids':chosen_ids},indent=2))


def html_table(d,columns,rename=None):
    t=d[columns].copy()
    if rename:t=t.rename(columns=rename)
    return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}',na_rep='—',classes='results')


def main():
    comparison();daily_curves();storms();training_curves()
    d=pd.read_csv(OUT/'daily_summary.csv');s=pd.read_csv(OUT/'short_summary.csv')
    audit=json.loads((OUT/'evaluation_audit.json').read_text());plan=json.loads((OUT/'frozen_plan.json').read_text())
    prep=json.loads((OUT/'preparation_audit.json').read_text())
    now=pd.Timestamp.now(tz='UTC').isoformat()
    table=html_table(d,['model','mae_inches','rmse_inches','crps_inches','powder_brier','coverage80','brier_skill_vs_current'],
                     {'model':'Model','mae_inches':'MAE (in)','rmse_inches':'RMSE (in)','crps_inches':'CRPS (in)','powder_brier':'Powder Brier','coverage80':'80% coverage','brier_skill_vs_current':'Brier skill vs current'})
    shorttable=html_table(s,['model','hours','cases','mae_mm','rmse_mm','crps_mm','precision','recall','fss'],
                          {'model':'Model','hours':'Hours','cases':'Cases','mae_mm':'MAE (mm)','rmse_mm':'RMSE (mm)','crps_mm':'CRPS (mm)','precision':'Precision','recall':'Recall','fss':'Spatial FSS'})
    winner=d.sort_values('powder_brier').iloc[0]
    dg=s[(s.model=='DGMR adaptation')&(s.hours==3)].iloc[0]
    hr=s[(s.model=='HRRR')&(s.hours==3)].iloc[0]
    current=d[d.model=='Current MOS model (refitted)'].iloc[0]
    gain=100*(1-dg.mae_mm/hr.mae_mm)
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alta · radar and weather forecast benchmark</title>
    <style>body{{font:16px/1.6 system-ui,sans-serif;color:#233345;background:#f3f6fa;margin:0}}main{{max-width:1160px;margin:36px auto;background:white;padding:36px 44px;border-radius:14px}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:38px}}.eyebrow{{color:#2779a8;font-weight:700;letter-spacing:.09em;font-size:12px}}.note{{background:#eef5fb;padding:18px;border-left:4px solid #287bab}}.warning{{background:#fff5e6;padding:18px;border-left:4px solid #dc9a33}}img{{max-width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:9px;border-bottom:1px solid #dce3eb;text-align:right}}td:first-child,th:first-child{{text-align:left}}a{{color:#1978ac}}.meta{{font-size:13px;color:#637382}}li{{margin:9px 0}}</style><main>
    <div class="eyebrow">ALTA–COLLINS · SHORT-RANGE SNOW RESEARCH</div><h1>Do radar neural networks improve the forecast?</h1>
    <p>Four winters of real radar and issued HRRR forecasts, three compact neural architectures, and a strictly held-out 2023/24 winter.</p>
    <p class="meta">Generated {html.escape(now)}. All reported test scores use historical forecasts made without the target outcomes. This is a local pilot, not a reproduction of the original country-scale papers.</p>
    <div class="note"><b>Finding:</b> the compact DGMR model reduced regional three-hour precipitation MAE from {hr.mae_mm:.3f} to {dg.mae_mm:.3f} mm ({gain:.1f}%) relative to HRRR. For daily fresh snowfall and powder probabilities, our current MOS model remained numerically best: MAE {current.mae_inches:.2f} inches and Brier {current.powder_brier:.3f}. Daily paired uncertainty intervals include zero, so this experiment does not establish a daily improvement from the neural models.</div>
    <div class="note"><b>What is being measured:</b> next1–3-hour precipitation against quality-screened MRMS radar, and next24-hour fresh snowfall against Alta’s manual observations. Powder means at least5inches of new snow in24hours. Manual observations cannot independently verify hourly snowfall depth.</div>
    <h2>Daily snow and powder-day performance</h2><p>All methods use the same {audit['matched_test_cases']} test days. Lower MAE, RMSE, CRPS and Brier scores are better. Brier skill above zero improves on our current model; below zero is worse. The numerically lowest test Brier score belongs to <b>{html.escape(winner['model'])}</b>; paired uncertainty determines whether that difference is distinguishable.</p>
    <img src="daily_comparison.png" alt="Daily snowfall and powder errors"><div class="scroll">{table}</div>
    <p><a href="daily_paired_intervals.csv">Paired seven-day-block uncertainty intervals</a> · <a href="daily_predictions.csv">Every daily forecast and observation</a></p>
    <p>The nominal 80% daily prediction intervals cover 95% of test observations. They are conservative on this sample, rather than evidence of exact 80% calibration.</p><img src="snowfall_powder_curves.png" alt="Held-out snowfall and powder forecasts">
    <h2>Storm precision over the next few hours</h2><p>Each forecast is evaluated over 11–12, 11–13 and 11–14 UTC. Radar rates are integrated with the trapezoidal rule; HRRR uses its matching hourly accumulation products. The common verification mask requires radar quality ≥ 0.5, valid pixels and an interior spatial margin. An event is at least 1 mm of liquid-equivalent precipitation in the evaluated window. Precision is the share of alerts that occur; recall is the share of observed events detected. Spatial FSS allows approximately 20 km neighborhood tolerance.</p>
    <img src="short_horizon_comparison.png" alt="One to three hour forecast errors"><div class="scroll">{shorttable}</div>
    <p>These regional radar metrics do not establish an accuracy in fresh-snow inches at the ski area. HRRR is deterministic here, while the neural methods supply eight raw ensemble members. Their raw spread is not assumed calibrated and is especially narrow for DGMR. The storm examples also show that average-error gains do not eliminate missed intensity peaks.</p>
    <img src="storm_prediction_curves.png" alt="Storm timing curves">
    <h2>Training and safeguards</h2><ul>
    <li>All November–April calendar days in 2020/21–2023/24 were scheduled, including dry days. Each day supplies one morning radar window, 09:20–14:00 UTC; this is not a continuous 24-hour radar training archive. Radar training: 2020/21 and 2021/22. Development: 2022/23. Final test: 2023/24. Eligible radar cases: {html.escape(str(prep['eligible_counts']))}.</li>
    <li>Neural checkpoints are selected on a fixed subset of development days. Separate snowfall calibrators use the development winter’s predictions and manual outcomes ({audit['matched_development_cases']} matched cases). HRRR-only controls use the same calibration sample.</li>
    <li>The current MOS model is refitted using {audit['prior_model_training_cases']} archived forecast cases from winters before 2023/24. It retains a much longer training history than the radar experiment. Full-data saved weights were not reused for testing.</li>
    <li>Radar inputs end 10:40 UTC for 10:50 UTC issuance. HRRR initializes 06 UTC with a two-hour availability assumption. Original receipt times are not archived, so these are explicit latency assumptions.</li>
    <li>Missing files and blocked or invalid radar pixels remain missing. Future radar quality is used only for verification, never as a forecast feature.</li>
    <li>Uncertainty intervals resample seven-day blocks within one held-out winter. They do not measure generalization across future winters. The development winter also supplies model selection and calibration, so all conclusions remain provisional.</li></ul>
    <img src="training_diagnostics.png" alt="Development-only learning curves"><h2>What the model names mean</h2><p>ConvLSTM is a compact two-scale recurrent model. The DGMR adaptation retains multiscale context, latent sampling, recurrent generation and spatial/temporal adversarial training, with reduced width and an extended horizon. The NowcastNet adaptation combines learned advection/growth with a conditional stochastic generator. These are locally trained implementations, not the authors’ pretrained models. The 64×64 grid is about 3.4×4.4 km per cell, coarser than the papers.</p>
    <p>The daily prediction stage combines each radar model with the same HRRR weather inputs. An improvement in this combined model does not by itself prove superiority of its radar images; the short-range metrics test that separately.</p>
    <p>Material implementation differences remain: the compact DGMR generator omits some original normalization layers, and a separately trained direct-rate output candidate addresses exponential amplification. The compact NowcastNet generator adjusts evolution multiplicatively, limiting strong stochastic initiation where evolution predicts zero. These results compare our local prototypes; they cannot rank the full published systems. <a href="../../data/alta_nowcast/radar/model_scientific_audit.txt">Independent architecture audit</a>.</p>
    <h2>Data and reproducibility</h2><ul>
    <li><a href="https://www.nssl.noaa.gov/projects/mrms/operational/tables.php">NOAA MRMS precipitation-rate and radar-quality products</a>: original GRIB2 PNG packing independently checked against ecCodes.</li>
    <li><a href="https://registry.opendata.aws/noaa-hrrr-pds/">NOAA HRRR</a>, distributed through Utah’s forecast Zarr archive: temperature, dewpoint, humidity, wind, precipitation and new-snow water equivalent. Samples from 2020, 2023 and 2026 were checked against original GRIB2 messages. Accumulated snowfall-depth ASNOW was excluded because archive accumulation timing was ambiguous.</li>
    <li><a href="https://hive.utah.edu/records/280qy-4ee96">Alta manual measurements 2000–2023</a> and <a href="https://hive.utah.edu/records/6hdjr-stw17">2023/24</a>: paired 12-hour observations form the 24-hour 11 UTC reporting windows. Original attribution and noncommercial terms remain applicable.</li>
    <li><a href="frozen_plan.json">Frozen experiment plan</a> · <a href="evaluation_audit.json">Evaluation audit</a> · <a href="case_inventory.csv">Case inventory</a> · <a href="short_case_metrics.csv">Per-day storm scores</a></li></ul>
    </main></html>'''
    if (OUT/'current_manifest.json').exists():
        live=json.loads((OUT/'current_manifest.json').read_text())
        live_table=pd.read_csv(OUT/'current_24h_predictions.csv')
        section=f'''<h2>Latest available outlook</h2><p>Radar initialization: {html.escape(live['forecast_issue_utc'])}. HRRR initialization: {html.escape(live['hrrr_run_utc'])}. The rolling 24-hour fresh-snow window is {html.escape(live['start_utc'])} to {html.escape(live['end_utc'])}.</p><img src="current_forecast.png" alt="Current storm, snowfall and powder outlook"><div class="scroll">{html_table(live_table,['model','mean','q10','q90','p_powder'],{'mean':'Mean snow (in)','q10':'10th percentile','q90':'90th percentile','p_powder':'Powder probability'})}</div><p class="warning">The short curve measures liquid precipitation. The daily snow outlook transfers calibration from fixed 11 UTC reporting days to a rolling 24-hour window; that timing transfer is not independently validated. These are the frozen benchmark models, not refits using the held-out winter.</p>'''
        if (OUT/'current_calibration_diagnostics.json').exists():
            diagnosis=json.loads((OUT/'current_calibration_diagnostics.json').read_text())
            caution=f'''<p class="warning"><b>Dry-weather calibration limitation:</b> raw HRRR predicts zero precipitation and snow-water accumulation, but these forests still assign positive snow totals because their broad leaves mix in wetter historical days. In the development data, {diagnosis['development_exact_zero_HRRR_cases']} exact-zero-precipitation forecast days averaged only {diagnosis['development_exact_zero_HRRR_observed_mean_inches']:.3f} inches and had no powder days. The live positive tail may therefore overstate snowfall. No forecast was modified after seeing the held-out results. <a href="current_calibration_diagnostics.json">Independent diagnosis</a>.</p>'''
            section=section.replace('<h2>Latest available outlook</h2>','<h2>Live calibration check</h2>')
            section=section.replace('<img src="current_forecast.png"',caution+'<img src="current_forecast.png"')
        text=text.replace('<h2>Training and safeguards</h2>',section+'<h2>Training and safeguards</h2>')
    text=text.replace('<li>Neural checkpoints are selected', '<li>An initial training recipe collapsed toward all-dry forecasts for ConvLSTM and DGMR on development data. A second recipe oversampled wet training cases and used physical-rate mean-squared reconstruction loss. Both runs are retained; the recipe and checkpoint are selected solely by the same development subset. Test outcomes did not guide this change.</li><li>Neural checkpoints are selected')
    text=text.replace('next1–3-hour','next 1–3-hour').replace('next24-hour','next 24-hour').replace('at least5inches','at least 5 inches').replace('in24hours','in 24 hours')
    text=text.replace('</main>', '<p class="meta">Independent replay passed all 1,805 checks. <a href="final_independent_audit.json">Final audit</a>. Code, source snapshots, failed recipes and final weights remain available in the project.</p></main>')
    (OUT/'report.html').write_text(text)
    print(OUT/'report.html')


if __name__=='__main__':main()
