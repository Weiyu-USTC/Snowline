"""Build the unified, locally served forecast experiment report."""
from pathlib import Path
import html,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/nwp-weather/forecast-portfolio'
NAMES={'alta':'Alta','vail':'Vail','mansfield':'Stowe','jay':'Jay Peak'}

def table(d):return d.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}',na_rep='—',escape=True)
def read(path):return json.loads(path.read_text())

def build():
    forecast=read(OUT/'current_forecast.json');choice=read(OUT/'hrrr24_selection.json')
    alta=read(ROOT/'outputs/nwp-weather/alta12-trained/manifest.json')
    a=pd.read_csv(ROOT/'outputs/nwp-weather/alta12-trained/metrics.csv')
    am=a[(a.cohort=='modern')&a.stage.eq('test')][['method','n','mae_inches','rmse_inches','bias_inches']]
    metric=pd.read_csv(OUT/'hrrr24_metrics.csv');m=metric[metric.split.eq('test')].pivot(index='site',columns='method',values='rmse_inches')
    m=m[['hrrr_native','prior_neural','portfolio']].rename(columns={'hrrr_native':'HRRR native','prior_neural':'Previous neural','portfolio':'Selected portfolio'})
    m.insert(0,'Test windows',[choice[s]['test_n'] for s in m.index]);m.index=m.index.map(NAMES);m=m.reset_index(names='Location')
    radar=pd.read_csv(ROOT/'outputs/nexrad-short/paired_metrics.csv')
    radar=radar[radar.split.eq('test')][['site','model','n','RMSE_1h_mm','RMSE_3h_mm']]
    radar['site']=radar.site.map(NAMES)
    temp=[]
    for site in ['vail_snotel','stowe_mountain']:
        p=ROOT/f'outputs/mountain-temperature/{site}_metrics.csv'
        if p.exists():
            d=pd.read_csv(p);d=d[d.split.eq('test')][['model','rows','issue_dates','rmse_c','mae_c','selected']];d.insert(0,'Station',site);temp.append(d)
    temp_table=table(pd.concat(temp,ignore_index=True)) if temp else '<p>Temperature comparison unavailable.</p>'
    sums=[]
    for site,s in forecast['sites'].items():
        h=s['hourly'];v=[r['display_snow_inches'] for r in h];provider=[r['provider_snow_inches'] for r in h]
        sums.append({'Location':NAMES[site],'HRRR + provider tail (in)':sum(v) if all(x is not None for x in v) else np.nan,
            'Provider48h (in)':sum(provider) if all(x is not None for x in provider) else np.nan,'Previous neural48h (in)':s['existing_neural'][1]['mean_inches'],
            'Portfolio24h (in)':s['portfolio24']['mean_inches'],'HRRR future hours':s['native_future_hours'],'Provider tail hours':s['provider_tail_hours']})
    p12=forecast['alta12'];selected=alta['selection']['modern']
    if p12.get('available'):
        text12=f"Next fixed12h window: {html.escape(p12['start'])} → {html.escape(p12['end'])}. Forecast <strong>{p12['mean_inches']:.2f}in</strong>; nominal80% model range {p12['q10_inches']:.2f}–{p12['q90_inches']:.2f}in. The live issue gap differs from the historical10-minute gap; this is an operational extrapolation."
    else:text12='Alta12h forecast unavailable: '+html.escape(p12.get('reason','Missing predictors.'))
    image12='<img src="alta12_current.png" alt="Alta next fixed12-hour snowfall comparison and model range">' if p12.get('available') else ''
    notices=forecast.get('warnings',[])
    source_notice='<p class="status"><strong>Some forecast components are unavailable.</strong> '+html.escape(' '.join(notices))+' Missing amounts remain unavailable; other saved-model forecasts are retained.</p>' if notices else ''
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Snowline · Forecast portfolio</title>
<style>body{{margin:0;background:#edf2ef;color:#17343d;font:16px/1.6 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}}main{{max-width:1150px;margin:36px auto;padding:0 24px 70px}}h1{{font-size:38px;line-height:1.1}}h2{{margin:0 0 16px;font-size:25px}}section{{background:white;border:1px solid #d8e4df;border-radius:18px;padding:26px;margin:22px 0}}p{{max-width:950px}}.eyebrow{{text-transform:uppercase;letter-spacing:2px;font-size:12px;color:#16747d}}.note{{font-size:14px;color:#536b72}}.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e2e9e6}}th{{background:#f1f6f3}}img{{display:block;width:100%;height:auto;margin-top:20px}}a{{color:#0a737c}}.status{{background:#e8f1ec;padding:18px;border-radius:12px}}code{{font-size:12px}}nav a{{margin-right:16px}}li{{margin-bottom:8px}}</style>
<main><p class="eyebrow">Snowline research · {html.escape(forecast['generated_at'])}</p><h1>Radar, weather and mountain observations</h1>
<p>A tested forecast portfolio for Alta, Vail, Stowe and Jay Peak. The components have different verified targets: true12-hour snowfall at Alta, hourly liquid precipitation for radar, and hourly temperature at mountain stations.</p>
<nav><a href="#alta">Alta12h</a><a href="#radar">Radar1–3h</a><a href="#hrrr">HRRR and24h blending</a><a href="#temperature">Mountain observations</a><a href="#forecast">Current curves</a><a href="dry-wet/comparison.html">Dry / wet: model vs HRRR &amp; provider</a></nav>
<section><h2>What improved—and what did not</h2><ul>
<li>The validation-selected24h portfolio combines HRRR snowfall and the previous neural forecast. Alta test RMSE is1.81in versus2.16in for the previous network and3.11in for native HRRR on48matched reporting windows.</li>
<li>Stowe is a counterexample: its selected blend loses to raw HRRR on the test winter. Keeping the frozen selection makes that generalization failure visible.</li>
<li>NEXRAD is usable in the tested Alta and Vermont domains. Vail lacks quality-qualified radar verification, so it retains HRRR. Native NEXRAD did not consistently beat Alta's existing MRMS neural nowcasts.</li>
<li>Mountain observations refine temperature. That improvement does not by itself establish an improvement in fresh snowfall or hourly storm timing.</li></ul>
<p class="note">2023/24 is held out from the new training and weight fitting, but was inspected in earlier project work. This is not a new blind test. Model choices use2022/23 validation; train-only preprocessing and explicit publication delays prevent future observations entering predictors.</p></section>
<section id="alta"><h2>Alta: genuine12-hour snowfall</h2><p>There are1,408 complete HRRR-matched manual12h measurements:682train,362validation and364test. The longer MOS experiment contains8,346 windows. The neural candidates have four residual temporal blocks (eight convolutions), with an optional radar CNN/GRU; forests, climatology and HRRR snow-water conversions are also tested.</p>
<p>The selected observation-informed method is <strong>{html.escape(selected['name'])}</strong>, with weights <code>{html.escape(json.dumps(selected['weights']))}</code>. Validation is divided into separate week blocks for tuning, blending and interval calibration. A separate missing-history selection is used live because recent manual reports are unavailable.</p>
<div class="table">{table(am)}</div><p><a href="../alta12-trained/report.html">Full Alta12h training curves, actual accumulations and model comparisons ↗</a></p>
<p class="status">{text12}</p>{image12}<p class="note">The radar TCN has57,882parameters; validation selected learning rate0.0003 and restored optimizer update240. Its advantage over the forest alone is not statistically clear. Recent station-sensor candidates did not improve blending validation and receive zero weight.</p></section>
<section id="radar"><h2>Native NEXRAD:1–3-hour liquid precipitation</h2><p>189calendar-selected dates across four winters, with three pre-issue LevelIII DPR scans. Candidate methods include persistence, advection, ridge regression, extra trees and a regularized MLP. Validation chooses the radar method and nonnegative radar/HRRR weights. Radar predictions are tested against quality-screened MRMS—not an independent fresh-snow gauge.</p>
<div class="table">{table(radar)}</div><p class="note">Errors here are millimetres of liquid-equivalent precipitation, not inches of fresh snow. Vail has no qualified verification windows at the fixed radar-quality threshold. Stowe and Jay targets are the nearby town observers. Calm hours dominate this small pilot; wet-event counts and biases are in the detailed report.</p><p><a href="nexrad/report.html">Radar coverage, previous neural comparisons and short-range curves ↗</a></p></section>
<section id="hrrr"><h2>HRRR hourly forecasts and24-hour portfolios</h2><p>Official NOAA forecasts provide temperature, humidity, winds, liquid precipitation, snow water and native variable-density snowfall. ASNOW is cumulative from initialization; adjacent native endpoints are differenced to obtain hourly new snowfall. Missing values remain missing. The calendar cohort contains186complete meteorological runs and189native snowfall runs.</p><div class="table">{table(m)}</div>
<p>Table entries are24h snowfall RMSE in inches, on identical cases. Vail is a Sno24 sensor endpoint proxy whose reset/reporting interval and equivalence to fresh snowfall remain unverified. The previous neural model was originally selected for48h and is scored here on its first24h output. Calibration uses training data; portfolio weights use validation data. Vermont fractional reporting-hour boundaries interpolate forecast accumulation only.</p>
<p class="note">Alta's portfolio weights are30%HRRR/70%neural; Vail45/55; Stowe85%calibratedHRRR/15%neural; Jay60/40. These are24h weights—not evidence of hourly or second-day skill. Week-block bootstrap uncertainty and the frozen choices are in <a href="hrrr24_selection.json">the selection audit</a>; current small-cohort gains need a new independent winter.</p></section>
<section id="temperature"><h2>Mountain observations refine hourly temperature</h2><div class="table">{temp_table}</div>
<p>Vail's correction uses prior NRCS842hourly temperatures; the archive also includes SWE and depth, which are not used in this temperature model. Stowe uses FEMC15-minute mountain weather. Validation selected a weather-only correction at Stowe: adding lagged mountain observations did not improve its score. Only observations available before issue enter Vail's correction. Snowpack depth/SWE changes are not treated as fresh snowfall, and the unheated Stowe winter precipitation gauge is excluded.</p><p><a href="mountain-temperature/report.html">Temperature curves, quality controls and observation ablations ↗</a></p></section>
<section id="forecast"><h2>Current48-hour outlook</h2><p>{html.escape(forecast['start'])} → {html.escape(forecast['end'])}. Native HRRR run {html.escape(forecast['native_hrrr']['run_time_utc'])} ends {html.escape(forecast['native_hrrr']['forecast_end_utc'])}. A48-hour weather run already has several elapsed hours when received: the final future hours use a separately identified Open-Meteo extension.</p>
{source_notice}<div class="table">{table(pd.DataFrame(sums))}</div><img src="hourly_forecasts.png" alt="Hourly snowfall forecast for all four station targets"><img src="cumulative_forecasts.png" alt="Cumulative snowfall and neural forecast checkpoints">
<p class="note">The hourly bars are native weather guidance. Learned snowfall is shown at12/24/48h accumulation checkpoints; no artificial hourly observations were created.24h residual ranges are descriptive and are not calibrated hourly bands. Alta/Vail are mountain stations; Stowe/Westfield training targets are lower-elevation town observers, so their numbers must not be presented as resort summit snowfall.</p><p><a href="current_forecast.json">Download forecast, sources and component metadata</a></p></section>
<section><h2>Snowline deployment and reproducibility</h2><p>The widget and dashboard now display the validation-selected method for each mountain and horizon as <strong>Our model</strong>. <a href="deployment_selection.json">Frozen deployment choices</a> record the selection evidence. Refresh all updates radar and HRRR inputs for all four mountains, checks available mountain observations, and runs the saved models without optimizer updates. Original source times are retained; missing or stale inputs trigger an identified HRRR fallback where supported. The 1–3h comparisons measure liquid precipitation; longer horizons measure snowfall. These are different targets, and hourly fresh-snow timing remains unvalidated.</p><p>Saved inputs, model weights, forecast timestamps, data clocks, training losses and held-out predictions are retained locally. Jay Peak's 48h radar model is selected separately from the weather-only component used by its 24h portfolio. Alta's trained12h method applies to fixed reporting windows; other12h windows use HRRR.</p>
<p>The immediate next improvement is more independent storm seasons and verified hourly fresh-snow measurements at resort elevation. A positive radar or temperature score alone is insufficient to justify calling the resulting snowfall curve validated.</p>
<p>Sources: <a href="https://rapidrefresh.noaa.gov/hrrr/">NOAA HRRR</a> · <a href="https://www.nco.ncep.noaa.gov/pmb/products/hrrr/">NOAA forecast-field inventory</a> · <a href="https://www.ncei.noaa.gov/products/radar/next-generation-weather-radar">NOAA NEXRAD</a> · <a href="https://hive.utah.edu/records/280qy-4ee96">University of Utah Alta observations</a> · <a href="https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=842&state=CO">NRCS Vail station</a> · <a href="https://www.uvm.edu/femc/data/archive/project/mt-mansfield-west-slope-mid-elevation/dataset/mansfield-west-met-site/metadata">FEMC mountain station</a>.</p></section></main></html>'''
    (OUT/'report.html').write_text(body)

if __name__=='__main__':build()
