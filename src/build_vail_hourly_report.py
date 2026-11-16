"""Render the Vail hourly experiment without relabeling its sensor proxy.

Run after ``python3 src/vail_hourly_model.py --mode run``. The report deliberately
does not sum endpoint readings or publish them as fresh-snowfall totals.
"""
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


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/vail-hourly'
COLORS = {'Hourly TCN': '#006c88', 'Climatology': '#68737b',
          'Ridge': '#a65b2a', 'HGB': '#7b5b9b', 'Persistence': '#a5a9ad'}
NAMES = {'Hourly TCN': 'Hourly TCN', 'Climatology': 'Calendar climatology',
         'Ridge': 'Ridge regression', 'HGB': 'Boosted trees', 'Persistence': 'Persistence'}
LEADS = ['1', '2–3', '4–7', '8–14', '15–30', '31–60']


def read_json(path):
    return json.loads(path.read_text())


def table(frame, *, digits=3):
    return frame.to_html(index=False, border=0, classes='data-table', escape=True,
                         float_format=lambda x:f'{x:,.{digits}f}', na_rep='—')


def setup_plots():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.edgecolor':'#bcc6cd', 'axes.labelcolor':'#243d4a',
                         'text.color':'#243d4a', 'xtick.color':'#435965',
                         'ytick.color':'#435965', 'grid.color':'#e1e7eb',
                         'figure.facecolor':'white', 'axes.facecolor':'white',
                         'savefig.facecolor':'white', 'svg.fonttype':'none'})


def forecast_figure(forecast, manifest):
    setup_plots()
    dates = pd.to_datetime(forecast.date)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.3), sharex=True,
                             gridspec_kw={'height_ratios':[3.2, 1.15]})
    ax = axes[0]
    ax.fill_between(dates, forecast.historical_q10_inches,
                    forecast.historical_q90_inches, color=COLORS['Climatology'],
                    alpha=.12, step='mid', label='Historical 10–90% range')
    ax.plot(dates, forecast.historical_mean_inches, color=COLORS['Climatology'],
            alpha=.5, lw=2, ls='--', label='Historical mean (α = 0.5)')
    ax.fill_between(dates, forecast.q10_inches, forecast.q90_inches,
                    color=COLORS['Hourly TCN'], alpha=.22, step='mid',
                    label='TCN model-based 10–90% range')
    ax.plot(dates, forecast.mean_inches, color=COLORS['Hourly TCN'], lw=2,
            marker='o', ms=3.1, label='TCN expected endpoint reading')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('04:00 station reading (inches)')
    ax.grid(axis='y', alpha=.8)
    ax.legend(frameon=False, ncol=2, loc='upper left', fontsize=9)
    ax.set_title('Daily values refer to the CAIC CAVMM Sno24 sensor endpoint',
                 loc='left', fontsize=11, pad=11)
    prob = axes[1]
    prob.plot(dates, forecast.probability_at_least_one_inch*100,
              color=COLORS['Hourly TCN'], lw=1.8, marker='o', ms=2.7)
    prob.set_ylim(0, 100)
    prob.set_yticks([0, 25, 50, 75, 100])
    prob.set_ylabel('P(reading ≥ 1 in)\n(%)')
    prob.grid(axis='y', alpha=.8)
    prob.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    prob.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    prob.set_xlabel('Reporting date, 2026 · CAIC display clock; UTC offset unverified')
    fig.suptitle('Vail hourly-input outlook through December 31', x=.085, y=.98,
                  ha='left', fontsize=19, weight='bold')
    origin = pd.Timestamp(manifest['issue_endpoint_source_clock']).strftime('%b %d, %Y %H:%M')
    fig.text(.085, .925, f'Input through {origin} · four mountain stations · 168 observed hours',
             fontsize=10, color='#5a6d77')
    fig.text(.085, .027,
             'Provisional sensor proxy: daily reset timing is unverified. These are not validated 24-hour fresh-snow totals.\n'
             'Bands are pointwise predictive ranges, not confidence intervals for the mean; distant dates do not identify specific storms.',
             fontsize=9, color='#6f5144', va='bottom')
    fig.subplots_adjust(left=.085, right=.98, top=.845, bottom=.145, hspace=.2)
    for ext in ('png', 'svg'):
        fig.savefig(OUT/f'forecast_figure.{ext}', dpi=180)
    plt.close(fig)


def error_figure(metrics):
    setup_plots()
    overall = metrics[metrics.leads=='1–60'].sort_values('rmse')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={'width_ratios':[1, 1.45]})
    ax = axes[0]
    y = np.arange(len(overall))
    ax.barh(y, overall.rmse, color=[COLORS[x] for x in overall.model], height=.62)
    ax.set_yticks(y, [NAMES[x] for x in overall.model])
    ax.invert_yaxis()
    ax.set_xlim(0, max(overall.rmse)*1.19)
    for i, value in enumerate(overall.rmse):
        ax.text(value+.025, i, f'{value:.3f}', va='center', fontsize=10)
    ax.set_xlabel('RMSE (inches of endpoint reading)')
    ax.set_title('All leads: 1–60 days', loc='left', pad=12)
    ax.grid(axis='x', alpha=.7)
    ax.set_axisbelow(True)
    ax = axes[1]
    for model in ['Climatology','Ridge','HGB','Persistence','Hourly TCN']:
        vals = metrics[(metrics.model==model)&metrics.leads.isin(LEADS)].set_index('leads').reindex(LEADS)
        ax.plot(np.arange(len(LEADS)), vals.rmse, marker='o', ms=4,
                lw=2.2 if model=='Hourly TCN' else 1.5,
                color=COLORS[model], alpha=.55 if model=='Persistence' else 1,
                label=NAMES[model])
    ax.set_xticks(np.arange(len(LEADS)), LEADS)
    ax.set_xlabel('Days ahead at issue time')
    ax.set_ylabel('RMSE (inches of endpoint reading)')
    ax.set_title('Error by forecast lead', loc='left', pad=12)
    ax.grid(axis='y', alpha=.7)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc='best')
    fig.suptitle('Vail: held-out winter error', x=.08, y=.98, ha='left', fontsize=19, weight='bold')
    fig.text(.08, .9, 'Same targets and forecast origins for every method · lower is better', fontsize=10, color='#5a6d77')
    fig.text(.08, .035, 'Statistical baselines only. This is not a comparison with archived operational weather forecasts.\n'
             'Repeated forecasts of the same day are dependent; uncertainty comparisons resample whole winters.', fontsize=9, color='#5a6d77')
    fig.subplots_adjust(left=.15, right=.98, top=.8, bottom=.2, wspace=.38)
    for ext in ('png','svg'):
        fig.savefig(OUT/f'errorcomparison.{ext}', dpi=180)
    plt.close(fig)


def main():
    required = ['forecast_daily.csv', 'forecast_manifest.json', 'crossvalidation_metrics.csv',
                'chronological_metrics.csv', 'paired_error_bootstrap.json', 'frozen_plan.json',
                'target_coverage.json', 'final_training.json']
    missing = [name for name in required if not (OUT/name).exists()]
    if missing:
        raise SystemExit('Run model training first; missing: '+', '.join(missing))
    forecast = pd.read_csv(OUT/'forecast_daily.csv')
    manifest = read_json(OUT/'forecast_manifest.json')
    cv = pd.read_csv(OUT/'crossvalidation_metrics.csv', dtype={'leads':str})
    chrono = pd.read_csv(OUT/'chronological_metrics.csv', dtype={'leads':str})
    bootstrap = pd.DataFrame(read_json(OUT/'paired_error_bootstrap.json'))
    plan = read_json(OUT/'frozen_plan.json')
    coverage = pd.DataFrame(read_json(OUT/'target_coverage.json'))
    source = read_json(ROOT/'data/vail_hourly/hourly_manifest.json')
    final_training = read_json(OUT/'final_training.json')
    official_path = ROOT/'data/vail_hourly/vail_official_report_20261104.json'
    official = read_json(official_path) if official_path.exists() else None

    forecast_figure(forecast, manifest)
    error_figure(cv)

    overall = cv[cv.leads=='1–60'].sort_values('rmse')
    neural = overall[overall.model=='Hourly TCN'].iloc[0]
    climate = overall[overall.model=='Climatology'].iloc[0]
    best = overall.iloc[0]
    difference_percent = 100*(neural.rmse/climate.rmse-1)
    versus = 'lower' if difference_percent < 0 else 'higher'
    bcl = bootstrap[bootstrap.comparison=='TCN minus Climatology'].iloc[0]
    if bcl.ci95_high < 0:
        evidence = 'The winter-block bootstrap favors the TCN over climatology on this station-proxy dataset.'
    elif bcl.ci95_low > 0:
        evidence = 'The winter-block bootstrap favors climatology over the TCN on this station-proxy dataset.'
    else:
        evidence = 'The winter-block bootstrap does not establish a difference from climatology.'
    comparison_cols = ['model','mse','rmse','mae','bias']
    overview = overall[comparison_cols].copy()
    overview.model = overview.model.map(NAMES)
    overview.columns = ['Method','MSE (in²)','RMSE (in)','MAE (in)','Bias (in)']
    lead_table = cv[cv.leads.isin(LEADS)].pivot(index='leads',columns='model',values='rmse').reindex(LEADS)
    lead_table = lead_table[['Climatology','Persistence','Ridge','HGB','Hourly TCN']].rename(columns=NAMES).reset_index()
    lead_table = lead_table.rename(columns={'leads':'Days ahead'})
    chrono_table = chrono[chrono.leads=='1–60'][comparison_cols].sort_values('rmse').copy()
    chrono_table.model = chrono_table.model.map(NAMES)
    chrono_table.columns = overview.columns
    uncertainty = cv[(cv.leads=='1–60') & cv.model.isin(['Hourly TCN','Climatology'])][['model','coverage80','width80','interval_score80']].copy()
    uncertainty.model = uncertainty.model.map(NAMES)
    uncertainty.coverage80 *= 100
    uncertainty.columns = ['Method','80% range coverage (%)','Mean width (in)','Interval score (in)']
    boot_table = bootstrap[['comparison','mse_difference','ci95_low','ci95_high']].copy()
    boot_table.columns = ['Comparison','ΔMSE (in²)','95% lower','95% upper']
    cov_table = coverage[['winter','reported_days','positive_days','one_inch_days','max']].copy()
    cov_table.winter = cov_table.winter.map(lambda x:f'{x}/{str(x+1)[-2:]}')
    cov_table.columns = ['Winter','Usable Oct–Apr endpoints','Reading > 0','Reading ≥ 1 in','Maximum (in)']
    stations = []
    for code, item in source['stations'].items():
        stations.append({'Station':code, 'Location':item['name'], 'Elevation (ft)':item['elevation_ft'],
                         'Role':'Target + hourly inputs' if code=='CAVMM' else 'Hourly inputs'})
    station_table = table(pd.DataFrame(stations),digits=0)
    start = pd.Timestamp(forecast.date.iloc[0]).strftime('%B %d')
    end = pd.Timestamp(forecast.date.iloc[-1]).strftime('%B %d, %Y')
    years = ', '.join(f'{y}/{str(y+1)[-2:]}' for y in plan['years'])
    n_channels = len(plan['channels'])
    hourly_rows = source.get('hourly_rows',0)
    valid_days = int(coverage.reported_days.sum())
    official_note = ''
    if official:
        report = official['snowReportData']
        target = pd.read_csv(OUT/'daily_target_0400.csv')
        stamp_col = target.columns[0]
        check = target[pd.to_datetime(target[stamp_col])==pd.Timestamp('2026-11-03 04:00')]
        station_value = f'{float(check.iloc[0,1]):.1f}' if len(check) else 'unavailable'
        official_note = f'''<p>The saved official report, <strong>{escape(report['LastUpdatedText'])}</strong>,
          lists <strong>{escape(report['TwentyFourHourSnowfall']['Inches'])} inches</strong> for 24 hours.
          The CAIC mid-mountain sensor at 04:00 on that reporting date reads <strong>{station_value} inches</strong>.
          These differ in observation time and may differ in measurement and reporting procedure. One comparison
          does not estimate a correction factor. <a href="https://www.vail.com/the-mountain/mountain-conditions/snow-and-weather-report.aspx">Official Vail report</a> ·
          <a href="../../data/vail_hourly/vail_official_report_20261104.json">Saved source snapshot</a>.</p>'''
    training_rows = []
    for path in sorted(OUT.glob('fold*_training.json')):
        record = read_json(path)
        training_rows.append({'Fold':record['tag'], 'Training winters':', '.join(map(str,record['training_winters'])),
                              'Inner validation winters':', '.join(map(str,record['inner_validation_winters'])),
                              'Width':record['selected_config']['width'], 'Epochs':record['selected_epochs'],
                              'Parameters':record['parameter_count']})
    configs = table(pd.DataFrame(plan['configs']).rename(columns={'width':'Width','dropout':'Dropout','lr':'Learning rate','weight_decay':'Weight decay'}),digits=4)
    roni_used = manifest.get('roni_used_by_model', manifest['roni_at_origin'])
    roni_range = manifest.get('roni_training_range')
    roni_guard = ''
    if roni_range is not None:
        roni_guard = f'''<p><strong>ENSO extrapolation is restricted.</strong> The lagged observed RONI at issue is
          {manifest['roni_at_origin']:.2f}, while the final model's training range is [{roni_range[0]:.2f}, {roni_range[1]:.2f}].
          The neural model uses {roni_used:.2f}, clipped to that range. A future covariate that was constant in a training split
          is held at its training value, so an unseen strong-El-Niño flag cannot activate an untrained weight.
          This guard is fitted independently inside every training split and also used for inner validation and the final outlook.
          <strong>The strong El Niño effect cannot be estimated from these six winters.</strong></p>'''
    seasonal_sensitivity = ''
    seasonal_path = OUT/'nov_dec_target_metrics.csv'
    if seasonal_path.exists():
        seasonal = pd.read_csv(seasonal_path, dtype={'leads':str})
        seasonal_overall = seasonal[seasonal.leads=='1–60'].sort_values('rmse')
        seasonal_display = seasonal_overall[comparison_cols].copy()
        seasonal_display.model = seasonal_display.model.map(NAMES)
        seasonal_display.columns = overview.columns
        sample = seasonal_overall.iloc[0]
        season_bootstrap = read_json(OUT/'nov_dec_paired_error_bootstrap.json')
        season_ci = next(item for item in season_bootstrap if item['comparison']=='TCN minus Climatology')
        season_neural = seasonal_overall[seasonal_overall.model=='Hourly TCN'].iloc[0]
        season_climate = seasonal_overall[seasonal_overall.model=='Climatology'].iloc[0]
        season_comparison = 'lower' if season_neural.rmse < season_climate.rmse else 'higher'
        season_crosses = season_ci['ci95_low'] <= 0 <= season_ci['ci95_high']
        season_interval = 'spanning zero' if season_crosses else 'excluding zero'
        seasonal_sensitivity = f'''<h3>Matching the outlook season: November–December targets</h3>
          <p>Restricting the existing held-out forecasts to November–December target dates gives
          {int(sample.pairs):,} matched forecast-target pairs across {int(sample.unique_days):,} distinct reporting dates.
          This is a calendar sensitivity analysis, not an independent new test. Seasonal gaps remain:
          2022 contributes only December 6–31, and 2023 contributes November 29–December 31 with one missing day.</p>
          <div class="table-wrap">{table(seasonal_display)}</div>
          <p>The TCN's seasonal RMSE is {season_comparison} than calendar climatology's.
          TCN minus climatology ΔMSE is {season_ci['mse_difference']:.3f} in²;
          its paired winter-bootstrap 95% interval is [{season_ci['ci95_low']:.3f}, {season_ci['ci95_high']:.3f}] in², {season_interval}.
          <a href="nov_dec_target_metrics.csv">Season-specific error table</a> ·
          <a href="nov_dec_evaluation_summary.json">Calendar coverage details</a>.</p>'''
    first_last = f'{start}–{end}'
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vail hourly-input model · daily sensor outlook</title>
<style>
:root{{--ink:#193744;--muted:#5d727e;--teal:#006c88;--line:#dbe5ea;--soft:#f2f7f9}}
*{{box-sizing:border-box}} body{{margin:0;background:#edf3f6;color:var(--ink);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.65}}
main{{max-width:1160px;margin:32px auto;padding:48px 52px;background:white;border-radius:18px;box-shadow:0 8px 32px #1937440d}}
.eyebrow{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--teal);font-weight:750}}h1{{font-size:39px;line-height:1.15;letter-spacing:-.03em;margin:12px 0 18px;max-width:850px}}
.lead{{font-size:19px;color:#49616d;max-width:920px}}h2{{font-size:25px;line-height:1.25;margin-top:46px;padding-top:28px;border-top:1px solid var(--line)}}h3{{font-size:18px;margin-top:26px}}p{{margin:12px 0}}a{{color:var(--teal);text-underline-offset:3px}}.warning{{padding:22px 26px;background:#fff5e9;border-left:5px solid #bb772b;border-radius:5px;margin:28px 0}}.warning p:first-child{{margin-top:0}}.warning p:last-child{{margin-bottom:0}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:25px 0}}.card{{padding:20px;background:var(--soft);border-radius:10px}}.card .number{{display:block;font-size:27px;font-weight:750;line-height:1.3}}.card .label{{font-size:13px;color:var(--muted)}}.figure{{margin:22px -12px 12px}}.figure img{{width:100%;height:auto;display:block}}.caption,.small{{font-size:13px;color:var(--muted)}}.table-wrap{{overflow:auto;margin:18px 0}}table{{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}}th{{background:#eef4f7;color:#34515f;text-align:left;font-weight:650}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line)}}td:first-child{{font-weight:550}}tr:last-child td{{border-bottom:0}}code{{font-family:ui-monospace,SFMono-Regular,monospace;font-size:.9em;background:#edf3f6;padding:2px 5px;border-radius:3px}}ul,ol{{padding-left:22px}}li{{margin:7px 0}}.downloads{{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px}}.downloads a{{padding:9px 13px;border:1px solid #bfcfd7;border-radius:7px;text-decoration:none;font-size:14px}}details{{margin:18px 0;background:var(--soft);padding:14px 20px;border-radius:8px}}summary{{cursor:pointer;font-weight:650}}footer{{margin-top:44px;padding-top:20px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}}
@media(max-width:760px){{main{{margin:0;padding:26px 20px;border-radius:0}}h1{{font-size:31px}}.cards{{grid-template-columns:1fr}}.figure{{margin-left:-12px;margin-right:-12px}}}}
@media print{{body{{background:white}}main{{box-shadow:none;margin:0;padding:0}}.table-wrap{{overflow:visible}}h2{{break-after:avoid}}.figure,.warning,.cards{{break-inside:avoid}}}}
</style></head><body><main>
<div class="eyebrow">Independent Vail experiment · November 4, 2026</div>
<h1>Hourly mountain observations,<br>daily outlook through year-end</h1>
<p class="lead">A deeper temporal neural network trained on four Vail mountain stations, with held-out winter comparisons against calendar climatology, persistence, ridge regression and boosted trees.</p>
<div class="warning"><p><strong>The target needs a measurement qualification.</strong> This experiment predicts the CAIC CAVMM <code>Sno24</code> reading at 04:00 each day. Source behavior indicates a resetting snow-board measurement, and its reset time is not verified. A fixed-time reading is therefore a <strong>provisional sensor proxy, not validated 24-hour fresh snowfall or an official resort snow report</strong>.</p>
<p>The models and comparisons below are valid for that recorded endpoint. Their error cannot yet be interpreted as error in Vail’s daily fresh-snow totals. We do not publish sums of these readings as monthly snowfall.</p></div>
<div class="cards"><div class="card"><span class="number">{hourly_rows:,}</span><span class="label">timestamped station records downloaded</span></div><div class="card"><span class="number">{len(plan['years'])} partial winters</span><span class="label">{plan['origins']:,} eligible forecast origins</span></div><div class="card"><span class="number">{manifest['parameters']:,}</span><span class="label">parameters in the final 12-convolution TCN</span></div></div>

<h2>What the held-out errors show</h2>
<p>The TCN’s daily RMSE is <strong>{neural.rmse:.3f} inches</strong>, {abs(difference_percent):.1f}% {versus} than climatology’s {climate.rmse:.3f} inches. The lowest overall RMSE belongs to <strong>{NAMES[best.model]}</strong> ({best.rmse:.3f} inches). {evidence}</p>
<p>The main benchmark uses <strong>October–April target dates</strong> and lead times 1–60 on the same {int(neural.pairs):,} observed forecast-target pairs, representing <strong>{int(neural.unique_days):,} distinct reporting dates</strong>. Several origins forecast the same day, so these are not {int(neural.pairs):,} independent weather outcomes. The independent test units are whole winters.</p>
<div class="table-wrap">{table(overview)}</div>
<div class="figure"><img src="errorcomparison.png" alt="Held-out winter RMSE for five methods, overall and by forecast lead"></div>
<p class="caption">All error units refer to the station endpoint proxy. “Traditional methods” here means statistical baselines; historical issued NWP and human forecasts were not acquired and aligned for a like-for-like comparison in this experiment.</p>
<h3>RMSE by days ahead</h3><div class="table-wrap">{table(lead_table)}</div>
{seasonal_sensitivity}
<h3>Chronological check: train on earlier winters, test the latest two</h3>
<p>The separate chronological experiment tests {plan['years'][-2]}/{str(plan['years'][-2]+1)[-2:]} and {plan['years'][-1]}/{str(plan['years'][-1]+1)[-2:]}. It is a sensitivity check, not an extra source of independent winters or a second model-selection set.</p>
<div class="table-wrap">{table(chrono_table)}</div>
<details><summary>Uncertainty in the error comparison and predictive ranges</summary>
<p>Paired differences subtract each baseline’s squared error from the TCN’s on matched origin/lead/target pairs. Negative ΔMSE favors the TCN. The 95% interval resamples whole winter clusters 4,000 times; with only {len(plan['years'])} winters, it is a rough measure of uncertainty and cannot resolve instrument changes.</p>
<div class="table-wrap">{table(boot_table)}</div>
<p>Nominal 80% predictive ranges were not recalibrated using held-out labels. Coverage measures the fraction of observations between the lower and upper endpoints; interval score penalizes width and missed observations. A well-calibrated model should be assessed across both measures.</p>
<div class="table-wrap">{table(uncertainty)}</div></details>

<h2>The daily outlook: {first_last}</h2>
<p>Input ends at <strong>{escape(manifest['issue_endpoint_source_clock'])}</strong> in the CAIC display clock. Every future day is predicted directly from the same observed 168-hour context; no future observed weather enters the model. The gray historical mean uses alpha 0.5, with its historical 10–90% range behind it.</p>
<div class="figure"><img src="forecast_figure.png" alt="Daily Vail station endpoint proxy forecast and predictive band through December 31, with historical mean and range"></div>
<p class="caption">The shaded TCN range is the model’s pointwise 10th–90th percentile, not a confidence band around its mean. It is not guaranteed to widen monotonically with lead. Daily climatology, zero-probability and storm variability can change its width. Beyond short weather-prediction horizons, the curve is a conditional outlook, not a claim about the dates of individual storms.</p>
<p>The lower panel gives the model probability that the <em>recorded endpoint</em> is at least one inch. Because small positive sensor offsets occur, “greater than zero” would be an unreliable definition of a fresh-snow day.</p>

<h2>Data and measurement audit</h2>
<p>The four stations span approximately 10,303–11,248 feet. CAVMM is the target site; observations at the other elevations contribute temperature, moisture and wind context. CAIC identifies the resort as provider. The source contains 163,218 exact-hour records and 2,271 off-hour records; the latter are preserved in the download but excluded by the hourly model grid. <a href="https://stations.avalanche.state.co.us/precip.php">CAIC station table</a>.</p>
<div class="table-wrap">{station_table}</div>
<p>The October–April endpoint audit retains {valid_days:,} usable readings across the years below. Historical gaps remain missing. Each training origin needs at least 45 observed targets among its next 60 days and at least half of the mid-mountain temperature readings in its previous week. Winter 2021/22 has no usable target in this retrieval; the selected winters are {years}.</p>
<div class="table-wrap">{table(cov_table,digits=1)}</div>
<p>Snow depth is excluded from the model because the source contains suspicious plateaus. The CAVMM accumulation regimes from March 20 through May 31, 2020 and October 1 through November 28, 2023 are excluded from both context and targets after source review. Nonnegative endpoint values are otherwise retained without an invented trace correction. Positive offsets and shifts in occurrence frequency remain limitations.</p>
{official_note}
<p>Vail’s 2013 announcement describes snow stakes that clear automatically on a daily schedule, but it does not map their settings to the CAIC field. A firsthand 2012 account describes patrol clearing around 06:30–07:00 and an adjustment needed to turn the earlier board reading into a 24-hour resort report. That historical procedure is evidence of the distinction, not proof of today’s reset schedule. <a href="https://news.vailresorts.com/2013-11-06-VAIL-RESORTS-ANNOUNCES-NEW-LIVE-LOOK-ON-SNOW-REPORTING">Official 2013 announcement</a> · <a href="https://opensnow.com/news/post/how-does-vail-actually-measure-snow-behind-the-scenes">Firsthand measurement account</a>.</p>
<p>CAIC display timestamps are preserved. Historical UTC offset, daylight-saving convention and publication latency have not been verified, so the tests are date-indexed retrospective experiments rather than a fully reconstructed operational hindcast.</p>

<h2>Model and validation design</h2>
<p>The hourly encoder has <strong>six residual blocks and 12 causal convolutions</strong>, with dilations 1, 2, 4, 8, 16 and 32. Its theoretical receptive field is 253 hours, covering all 168 available input hours. The last encoded state and the pooled history feed a shared daily decoder. This architecture follows the residual, dilated-convolution approach described by <a href="https://arxiv.org/abs/1803.01271">Bai, Kolter and Koltun</a>; this experiment does not establish state-of-the-art weather skill.</p>
<p>The {n_channels} physical and derived input channels include temperature, dew point, relative humidity, wind speed/gusts, wind components, inter-station temperature differences, available pressure and precipitation, and the past CAVMM endpoint stream. Each has an explicit missingness channel, plus two hour-of-day features: {2*n_channels+2} encoder channels altogether. Imputation and scaling are fitted inside each training split. Future inputs contain only calendar, lead time and lagged ENSO features.</p>
<p>The output is a zero-plus-lognormal probability distribution, predicting both an occurrence probability and a positive-amount distribution. The final width is {final_training['selected_config']['width']}, selected training length {final_training['selected_epochs']} epochs, and parameter count {manifest['parameters']:,}. The forecast context has {100*manifest['forecast_input_missing_fraction']:.1f}% missing physical/derived values, represented explicitly to the model.</p>
<p>The future covariates include lagged RONI and strong positive/negative indicators, held fixed for every forecast lead. The index comes from revised historical data, not archived release vintages, so this experiment does not prove ENSO causality or a real-time ENSO advantage.</p>
{roni_guard}
<p>Validation uses <strong>{plan['outer_folds']} shuffled whole-winter outer folds</strong>, preserving every overlapping hourly context and future target inside its winter group. Each outer training set has one inner whole-winter split to choose between the two configurations below and choose training length. The chosen model is then refitted on all outer-training winters. No held-out winter is used to select hyperparameters.</p>
<div class="table-wrap">{configs}</div>
<div class="table-wrap">{table(pd.DataFrame(training_rows),digits=0)}</div>
<p>The baselines use the same eligible targets. Calendar climatology uses training-only calendar neighborhoods; persistence repeats the issue-time reading; ridge uses a fixed penalty of 100; boosted trees use 120 iterations, at most seven leaves, minimum leaf size 40 and L2 penalty 10. Tabular methods receive summaries of the preceding 6, 24, 72 and 168 hours, recent variability and missingness, plus known future features. Their settings were fixed before test scoring.</p>

<h2>What would make this a resort snowfall forecast</h2>
<ol><li>Verify the CAVMM measurement and reset protocol, or obtain dated manual daily snowfall from Vail; validate the mapping on overlapping storms.</li><li>Add archived weather forecasts available at each historical issue time, especially precipitation, temperature and wind by elevation. This addresses incoming storms that local past observations alone cannot identify.</li><li>Retrain and test the snowfall target on untouched future winters; calibrate probabilities on separate validation winters and compare against the corresponding archived NWP forecasts.</li></ol>
<p>The current deliverable is an independently trained Vail station experiment and an honest error comparison. Additional neural depth cannot repair an ambiguous training label or create long-range storm timing information.</p>

<h2>Files</h2>
<div class="downloads"><a href="forecast_daily.csv">Daily predictions CSV</a><a href="forecast_figure.png">Outlook figure PNG</a><a href="forecast_figure.svg">Outlook figure SVG</a><a href="crossvalidation_metrics.csv">Held-out errors CSV</a><a href="chronological_metrics.csv">Chronological errors CSV</a><a href="crossvalidation_predictions.csv">All held-out predictions CSV</a><a href="forecast_manifest.json">Forecast provenance</a><a href="frozen_plan.json">Frozen model plan</a><a href="../../data/vail_hourly/hourly_manifest.json">Source download manifest</a><a href="vail_hourly_tcn.pt">Trained model</a></div>
<p class="small">Monthly and seasonal sums are not published because these endpoint readings are not established fresh-snow totals. Automated checks cover time-window separation, label masking, probability outputs and the treatment of future covariates outside the training range. Reproducibility metadata and the source checksum are preserved with the model.</p>
<footer>Generated from saved model outputs. Source data SHA-256: <code>{escape(manifest['source_sha256'])}</code><br>Independent experimental model; not an official Vail or CAIC forecast.</footer>
</main></body></html>'''
    (OUT/'vail_hourly_report.html').write_text(html)
    print(json.dumps({'report':str(OUT/'vail_hourly_report.html'), 'forecast_days':len(forecast),
                      'tcn_rmse':float(neural.rmse), 'climatology_rmse':float(climate.rmse),
                      'best_model':str(best.model), 'target':'04:00 CAVMM endpoint proxy'}))


if __name__=='__main__':
    main()
