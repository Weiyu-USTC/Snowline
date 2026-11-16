"""Readable, auditable report for the frozen Alta12h experiment."""
from pathlib import Path
import html,json,pickle,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_alta12 import OUT,DATA,block_bootstrap

LABELS={'broad_climatology':'Historical mean (long record)','modern_climatology':'Historical mean (modern record)',
'broad_forest':'MOS + forest','broad_tcn':'MOS + deep TCN','modern_forest':'MOS + HRRR + forest',
'modern_radar_forest':'MOS + HRRR + radar forest','modern_tcn':'MOS + HRRR + deep TCN',
'modern_radar_tcn':'MOS + HRRR + radar TCN','hrrr_fixed_10to1':'HRRR snow-water × 10',
'hrrr_train_calibrated':'HRRR calibrated snow-water','modern_portfolio':'Validation-fit portfolio','broad_portfolio':'MOS portfolio',
'modern_operational_portfolio':'Operational portfolio (manual history absent)',
'modern_observations_forest':'Weather + mountain-observation forest',
'modern_radar_observations_forest':'Weather + radar + mountain-observation forest',
'modern_observations_portfolio':'Sensor candidate portfolio',
'native_asnow_inches':'HRRR native ASNOW','native_asnow_train_calibrated':'HRRR native ASNOW calibrated'}
def label(x):return LABELS.get(x,x)
def main():
    m=json.loads((OUT/'manifest.json').read_text());metrics=pd.read_csv(OUT/'metrics.csv');cases=pd.read_csv(OUT/'case_index.csv')
    allpred=dict(np.load(OUT/'predictions_all.npz'));data=dict(np.load(DATA/'arrays.npz'))
    search=pd.read_csv(OUT/'hyperparameter_search.csv');loss=pd.read_csv(OUT/'training_curves.csv')
    with open(OUT/'bundle.pkl','rb') as f:bundle=pickle.load(f)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    colors={'actual':'#20343a','modern_portfolio':'#057c86','modern_forest':'#f49829','modern_tcn':'#4567af','modern_radar_tcn':'#a94fa1','hrrr_fixed_10to1':'#929da0','broad_tcn':'#3b7e51'}
    test=metrics[(metrics.cohort=='modern')&(metrics.stage=='test')].sort_values('rmse_inches')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');names=test.method.tolist()
    ax.barh(range(len(names)),test.rmse_inches,color=[colors.get(k,'#7b929a') for k in names]);ax.set_yticks(range(len(names)),[label(k) for k in names]);ax.invert_yaxis();ax.set_xlabel('12-hour RMSE (inches) · lower is better');ax.set_title('Alta: same 364 measured intervals, November 2023–April 2024')
    for i,v in enumerate(test.rmse_inches):ax.text(v+.03,i,f'{v:.2f}',va='center',fontsize=9)
    ax.set_xlim(0,test.rmse_inches.max()*1.18);fig.savefig(OUT/'test_errors.png');plt.close(fig)
    dates=pd.to_datetime(cases.start_utc,utc=True);ids=np.flatnonzero(cases.modern.to_numpy(bool)&cases.validation_stage.eq('test').to_numpy())
    chosen=m['selection']['modern']['name'];actual=data['snow'][ids];p=allpred[chosen][ids];q=m['selection']['modern']['conformal_half_width_inches']
    fig,axes=plt.subplots(3,1,figsize=(12,10),layout='constrained')
    for ax,months in zip(axes,[[11,12],[1,2],[3,4]]):
        use=ids[dates.iloc[ids].dt.month.isin(months)];t=dates.iloc[use]
        pred=allpred[chosen][use]
        ax.fill_between(t,np.maximum(0,pred-q),pred+q,alpha=.15,color='#057c86',label='80% calibrated prediction interval')
        ax.plot(t,data['snow'][use],color='#20343a',lw=1.2,label='Observed manual 12h snowfall')
        ax.plot(t,pred,color='#057c86',lw=1.1,label=label(chosen))
        ax.plot(t,allpred['hrrr_fixed_10to1'][use],color='#a6adb0',lw=.8,alpha=.8,label='HRRR snow-water × 10')
        ax.set_ylabel('Snowfall (in / 12h)');ax.set_ylim(bottom=0);ax.grid(alpha=.15)
    axes[0].set_title('Measured accumulations and forecasts · points are genuine 12-hour totals')
    axes[0].legend(ncol=2,fontsize=8,loc='upper left');fig.savefig(OUT/'test_timeseries.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,(cohort,family) in zip(axes,[('broad','tcn'),('modern','tcn'),('modern','radar_tcn')]):
        model=bundle['models'][f'{cohort}_{family}'];d=loss[(loss.cohort==cohort)&(loss.family==family)&loss.seed.eq(model['seed'])&np.isclose(loss.lr,model['lr'])]
        ax.plot(d.step,d.train_monitor_mse,label='Training monitor',color='#4567af');ax.plot(d.step,d.validation_tune_mse,label='Validation tuning blocks',color='#dd7e26')
        ax.axvline(model['best_step'],color='#057c86',ls='--',label='Restored checkpoint')
        ax.set_title(label(f'{cohort}_{family}'));ax.set_xlabel('Optimizer updates');ax.set_ylabel('MSE (in²)');ax.grid(alpha=.15);ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8);fig.savefig(OUT/'training_losses.png');plt.close(fig)
    # Paired uncertainty uses seven-day calendar blocks on the final modern test set.
    comparisons=[]
    for ref in ['modern_climatology','hrrr_fixed_10to1','hrrr_train_calibrated','modern_forest','modern_tcn','modern_radar_tcn','broad_tcn']:
        if ref==chosen:continue
        ci=block_bootstrap(actual,p,allpred[ref][ids],cases.seven_day_block.to_numpy()[ids])
        diff=np.sqrt(np.mean((p-actual)**2))-np.sqrt(np.mean((allpred[ref][ids]-actual)**2))
        comparisons.append({'selected':chosen,'reference':ref,'rmse_difference_inches':float(diff),'block_bootstrap_95ci_low':ci[0],'block_bootstrap_95ci_high':ci[1],'lower_is_better':True})
    pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv',index=False)
    table=test[['method','n','mae_inches','rmse_inches','bias_inches','wet_rmse_inches','snow_occurrence_brier']].copy();table.method=table.method.map(label)
    table.columns=['Method','Intervals','MAE (in)','RMSE (in)','Bias (in)','Wet-interval RMSE (in)','Occurrence Brier']
    table_html=table.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
    detail=[]
    for name,item in bundle['models'].items():
        if item['type']!='neural':continue
        score=metrics[(metrics.method==name)&(metrics.cohort==('broad' if name.startswith('broad') else 'modern'))]
        ss={r.stage:r for r in score.itertuples()}
        detail.append({'Model':label(name),'Parameters':sum(t.numel() for k,t in item['state'].items() if k!='lead'),'Learning rate':item['lr'],'Seed':item['seed'],'Restored update':item['best_step'],'Train MSE':ss['train'].mse_inches2,'Tune MSE':ss['tune'].mse_inches2,'Final test MSE':ss['test'].mse_inches2})
    neural_html=pd.DataFrame(detail).to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)
    weights=' + '.join(f'{100*v:.1f}% {label(k)}' for k,v in m['selection']['modern']['weights'].items())
    selection=m['selection']['modern'];best=test[test.method==chosen].iloc[0]
    def counts(cohort):return ', '.join(f'{k}: {v:,}' for k,v in m['stage_counts'][cohort].items())
    fmt=lambda x:html.escape(str(x))
    native=pd.read_csv(OUT/'native_asnow_metrics.csv');nt=native[native.stage.eq('test')].sort_values('rmse_inches')
    nt=nt[['method','n','mae_inches','rmse_inches']].copy();nt.method=nt.method.map(label)
    nt.columns=['Method','Same test intervals','MAE (in)','RMSE (in)']
    native_html=nt.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
    na=json.loads((OUT/'native_asnow_audit.json').read_text())
    operational=m['selection']['modern_missing_history'];op=operational['test_metrics']
    opweights=' + '.join(f'{100*v:.1f}% {label(k)}' for k,v in operational['weights'].items())
    om=pd.read_csv(OUT/'metrics_observations.csv');ot=om[om.stage.eq('test')&om.method.isin(['modern_observations_forest','modern_radar_observations_forest',operational['name']])].copy()
    ot=ot[['method','n','mae_inches','rmse_inches']];ot.method=ot.method.map(label);ot.columns=['Operational method','Test intervals','MAE (in)','RMSE (in)']
    obs_html=ot.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
    sensor=m['selection']['modern_observations']
    sensor_decision='The sensor candidates improved validation and receive weight.' if sensor['sensor_models_receive_weight'] else 'The sensor candidates did not improve the separate blending validation score, so they receive no deployment weight.'
    body=f'''<!doctype html><html><head><meta charset="utf-8"><title>Alta 12-hour model comparison</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#18343a;background:#f7faf9}}h1,h2{{line-height:1.15}}h1{{font-size:36px}}h2{{margin-top:40px}}p{{max-width:1000px}}img{{width:100%;background:white;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #dfe8e5}}th:first-child,td:first-child{{text-align:left}}.card{{padding:22px;background:#e6f2ef;border-radius:12px}}.muted{{color:#5e7377}}a{{color:#057c86}}code{{font-size:13px}}.scroll{{overflow:auto}}</style></head><body>
<p class="muted">Snowline research · Alta Collins · {fmt(m['created_utc'][:10])}</p>
<h1>Which model predicts Alta's next 12-hour accumulation best?</h1>
<div class="card"><strong>Selected before opening the final test: {fmt(label(chosen))}.</strong><br>{fmt(weights)}<br>Final common-sample RMSE: <b>{best.rmse_inches:.2f} in</b>; MAE: <b>{best.mae_inches:.2f} in</b>, across364 measured12-hour intervals.</div>
<p>These are forecasts of actual manual snowboard accumulation at Alta Collins, not hourly values inferred from a daily total. The experiment predicts the fixed04:00–16:00 and16:00–04:00 Mountain Standard Time periods. Training and scoring use the source's explicit start/end timestamps, converted with fixed MST (UTC−7), including daylight-saving months.</p>
<h2>1. Data and forecast clocks</h2>
<p>The source provides10,366 twelve-hour reports across25 cool seasons. Issued MOS forecasts match8,346 of those reports; complete point HRRR forecasts match1,408 modern intervals. The long-record and modern models consequently have different training years: their comparison measures the complete forecasting approach, not architecture alone.</p>
<ul><li><b>Long-record MOS:</b> {fmt(counts('broad'))}.</li><li><b>Modern HRRR:</b> {fmt(counts('modern'))}.</li><li><b>Forecast inputs:</b> issued NOAA MOS temperature, dewpoint, wind, precipitation probability and categorical QPF; HRRR hourly precipitation, snow water, temperature, dewpoint, wind and humidity. HRRR snow water is not fresh-snow depth.</li><li><b>Radar:</b> nine available pre-issue morning frames. Evening contexts are explicitly missing. No later radar frame or later weather observation enters a forecast.</li><li><b>Past manual observations:</b> summaries use only periods ending at least two hours before issue, an assumed reporting delay because original receipt timestamps are unavailable. Neural training drops the full history on30% of batches; forests include missing-history augmentation.</li></ul>
<p>All forecasts issue ten minutes before the target period. Morning MOS is from00UTC and HRRR from06UTC; evening predictions retain those older runs. This is a conservative archive benchmark, not a test of fresher afternoon HRRR. HRRR here is the nearest3km grid point, consistent with the live point extractor.</p>
<h2>2. Selection, validation and final scores</h2>
<p>Training ends in winter2021/22. Seven-day blocks from2022/23 are assigned to three separate roles: learning-rate/checkpoint tuning, portfolio-weight fitting, and interval calibration. Both periods of the same day stay together. The2023/24 winter is evaluated after those choices are frozen; it was used in earlier project experiments, so it is not a newly untouched blind season. One validation winter still leaves model ranking uncertain.</p>
<p>Neural candidates use four residual temporal blocks (eight causal convolutions) plus a dense output head, with an optional spatial radar encoder and recurrent context. We tested learning rates0.0003,0.001,0.003 with one seed, then two additional seeds at the best rate. Training restores the best validation checkpoint rather than the last optimizer update. Forests search leaf sizes8,20,35 and depth settings. The portfolio fits nonnegative weights summing to one among the three best tuning candidates, using separate blending blocks.</p>
<img src="test_errors.png" alt="Final12hourforecastRMSEcomparison">
<div class="scroll">{table_html}</div>
<p>“HRRR snow-water ×10” is an explicitly assumed10:1 conversion baseline, not native HRRR ASNOW or a commercial weather-provider forecast. The calibrated version fits its ratio only on training data. The Brier scores for these physical baselines use a simple amount-derived occurrence mapping and should not be interpreted as native provider probabilities. Wet-interval error conditions on observed snow greater than zero; traces were set to zero by the source authors.</p>
<h3>Native HRRR snowfall: a separate, identical-date comparison</h3>
<p>Official NOAA GRIB ASNOW predicts snow depth using the model's snowfall-density treatment. This is a stronger physical baseline than a fixed10:1 water conversion. We retrieved fixed calendar dates (every fourth day), without selecting snowstorms. For each12-hour target we subtract run-cumulative endpoints f17−f5 or f29−f17. Two tiny non-monotonic archive intervals are excluded from every method, rather than silently clipped. Their cause is consistent with packing jitter but not proven. No candidate was selected from these final test rankings.</p>
<div class="scroll">{native_html}</div><p>The operational portfolio's paired95% block-bootstrap interval for RMSE difference versus native ASNOW is [{na['operational_model_vs_native_rmse_difference_95ci'][0]:.2f},{na['operational_model_vs_native_rmse_difference_95ci'][1]:.2f}]in. These96 intervals are a subset of the364-case test; error values across different subsets should not be directly compared. <a href="native_asnow_audit.json">Native baseline audit</a> · <a href="native_asnow_predictions.csv">paired predictions</a>.</p>
<h2>3. Optimizer history and checkpoint grades</h2><img src="training_losses.png" alt="Trainingandvalidationlosseswithrestoredcheckpointmarkers"><div class="scroll">{neural_html}</div>
<p>The training curve uses a fixed monitoring subset for efficiency; the table evaluates all training cases at the restored checkpoint. Training and validation losses do not need to converge to the same value. A falling training loss alone is not evidence that additional optimizer updates improve unseen-season predictions.</p>
<h2>4. Actual12-hour accumulation versus each forecast</h2><img src="test_timeseries.png" alt="ActualAlta12houraccumulationsandforecastcurves">
<p>The shaded range is an80% marginal prediction interval calibrated on separate validation blocks. Its final test coverage is {selection['test_interval_coverage']:.1%}; the symmetric residual half-width is{q:.2f}in, truncated at zero. This is uncertainty in a future observation, not a confidence interval for the mean. Serial dependence and distribution shifts limit nominal coverage guarantees.</p>
<p>The selected model's seven-day block-bootstrap95% interval for its RMSE difference versus the modern historical mean is [{selection['test_rmse_difference_vs_climatology_95ci'][0]:.2f},{selection['test_rmse_difference_vs_climatology_95ci'][1]:.2f}]in. Negative differences favor the selected model. See <a href="paired_comparisons.csv">all paired comparisons</a>; an interval spanning zero means this single test winter does not distinguish those methods confidently.</p>
<h2>5. Deployment boundaries</h2><p>Deploy the selected family as <b>Our model</b>; preserve method details in the research report. The exported predictor accepts12 hourly issued-forecast rows and returns a12-hour total, an80% interval and occurrence probability. Complete HRRR inputs use the modern selection; missing HRRR uses the long-record MOS fallback. Missing current manual history is supported. Without that history, the modern test MAE is {selection['test_missing_history_metrics']['mae_inches']:.2f}in and RMSE is {selection['test_missing_history_metrics']['rmse_inches']:.2f}in.</p>
<p><b>Operational selection with missing manual history:</b> {fmt(opweights)}. It is selected and calibrated separately on validation blocks in the same missing-history condition: test MAE{op['mae_inches']:.2f}in, RMSE{op['rmse_inches']:.2f}in,80% interval half-width{operational['conformal_half_width_inches']:.2f}in and actual test coverage{operational['test_interval_coverage']:.1%}. Those are the settings used when the live predictor omits manual snowfall reports.</p>
<p><b>Mountain-sensor ablation:</b> we added17,310 actual hourly Alta Collins observations, using temperature, wet-bulb temperature, liquid precipitation and snow-depth state. Latest values, observation ages and3/6/24-hour summaries use only records with observation time plus an assumed one-hour publication lag before issue. These are predictors; neither snow-depth changes nor a reset-board sensor becomes the fresh-snow target. {sensor_decision} The inference API accepts the32 optional observation features and preserves the validated fallback.</p>
<div class="scroll">{obs_html}</div><p><a href="metrics_missing_history.csv">Operational validation/test scores</a> · <a href="metrics_observations.csv">Sensor ablation scores</a>. Sensor records may have been revised since the historical forecast date; original receipt times are unavailable.</p>
<p>Arbitrary rolling12-hour windows, new forecast issuance times, hourly snowfall shape and48-hour aggregation are not validated by this experiment. Forecast-hour curves may use the HRRR hourly distribution as guidance, but they must not be presented as independently verified hourly neural predictions. The manual source ends in2024; future operational retraining requires newer reports and another untouched season.</p>
<h2>Sources and reproducibility</h2><ul><li><a href="https://hive.utah.edu/records/280qy-4ee96">University of Utah Alta Collins archive</a> and <a href="https://hive.utah.edu/records/6hdjr-stw17">updated snowfall records</a> (retain the source's noncommercial attribution conditions).</li><li><a href="https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?help">Iowa Environmental Mesonet issued MOS archive</a>.</li><li><a href="https://hrrrzarr.s3.amazonaws.com/">Utah HRRR Zarr archive</a>.</li><li><a href="manifest.json">Manifest, source hashes and selection</a> · <a href="metrics.csv">all split metrics</a> · <a href="hyperparameter_search.csv">hyperparameter search</a> · <a href="training_curves.csv">optimizer histories</a> · <a href="test_predictions.csv">individual test predictions</a>.</li></ul>
</body></html>'''
    # Insert word spacing in prose only; never rewrite CSS, links or attributes.
    head,tail=body.split('<body>',1)
    chunks=re.split(r'(<[^>]*>)',tail)
    for i in range(0,len(chunks),2):
        chunks[i]=re.sub(r'(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])',' ',chunks[i])
    (OUT/'report.html').write_text(head+'<body>'+''.join(chunks))
    print(OUT/'report.html')
if __name__=='__main__':main()
