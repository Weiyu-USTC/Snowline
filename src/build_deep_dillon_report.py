"""Render the saved daily neural experiment; never retrain or select a model."""
from pathlib import Path
import base64
import html
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = ROOT / 'outputs'
BLUE, GRAY, INK = '#146b94', '#747a80', '#182d3c'


def number(x, digits=2):
    return f'{x:,.{digits}f}'


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'


def embed(path, alt):
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


def main():
    result = json.loads((OUT / 'deep_dillon_results.json').read_text())
    direct = json.loads((OUT / 'deep_dillon_direct_monthly_baseline.json').read_text())
    status = json.loads((ROOT / 'data/dillon_current_observation_status.json').read_text())
    forecast = pd.read_csv(OUT / 'deep_dillon_forecast_daily.csv', parse_dates=['date'])
    months = pd.read_csv(OUT / 'deep_dillon_forecast_monthly.csv')
    test = pd.read_csv(OUT / 'deep_dillon_test_daily.csv')
    monthly_test = pd.read_csv(OUT / 'deep_dillon_test_monthly.csv')
    ensemble = np.load(OUT / 'deep_dillon_forecast_paths.npz')
    paths = ensemble['paths_inches']
    assert np.array_equal(ensemble['dates'], forecast.date.dt.strftime('%Y-%m-%d').to_numpy())
    keep = forecast.date.between('2026-11-04', '2026-12-31')
    display = forecast.loc[keep].reset_index(drop=True)
    display_paths = paths[:, keep.to_numpy()]
    cumulative = np.cumsum(display_paths, axis=1)
    cumlo, cumhi = np.quantile(cumulative, [.1, .9], axis=0)
    display['display_cumulative_mean_inches'] = display.predicted_inches.cumsum()
    display['display_cumulative_q10_inches'] = cumlo
    display['display_cumulative_q90_inches'] = cumhi
    display.to_csv(OUT / 'deep_dillon_display_daily.csv', index=False, float_format='%.8f')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': INK, 'text.color': INK,
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.5), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.05]})
    dates = display.date
    axes[0].fill_between(dates, display.q10_inches, display.q90_inches, step='mid',
                         color=BLUE, alpha=.18, label='Model-based 80% daily prediction range')
    axes[0].step(dates, display.predicted_inches, where='mid', color=BLUE, lw=2.1,
                 label='Neural model: expected daily snowfall')
    axes[0].step(dates, display.historical_daily_mean_inches, where='mid', color=GRAY,
                 alpha=.5, lw=2, linestyle='--', label='Historical calendar mean (alpha = 0.5)')
    axes[0].set_ylabel('Daily new snow (inches)')
    axes[0].set_title('Daily amounts: individual storms are not dated by this outlook', loc='left', fontsize=12)
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    axes[0].set_ylim(bottom=0)
    axes[1].fill_between(dates, cumlo, cumhi, step='mid', color=BLUE, alpha=.18,
                         label='80% cumulative range from correlated daily paths')
    axes[1].step(dates, display.display_cumulative_mean_inches, where='mid', color=BLUE, lw=2.1,
                 label='Expected cumulative snowfall')
    axes[1].step(dates, display.historical_daily_mean_inches.cumsum(), where='mid', color=GRAY,
                 alpha=.5, lw=2, linestyle='--', label='Cumulative historical calendar mean')
    axes[1].set_ylabel('New snow accumulated\nsince November 4 (inches)')
    axes[1].set_title('Accumulation: sum each complete daily scenario before taking its range', loc='left', fontsize=12)
    axes[1].legend(loc='upper left', frameon=False, fontsize=9)
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.grid(axis='y', alpha=.17)
        ax.axvline(pd.Timestamp('2026-12-01'), color=GRAY, linestyle=':', lw=1)
        ax.margins(x=.006)
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WE, interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.suptitle('Dillon daily snowfall outlook · November 4–December 31, 2026', x=.075,
                 ha='left', fontsize=17, fontweight='bold', y=.985)
    fig.text(.075, .945, '60-day weather context · two-layer GRU · observations available through November 1', fontsize=11)
    fig.text(.075, .034, 'Bands are pointwise model-based prediction ranges, not confidence intervals for the mean or guarantees.\n'
             'November 2–3 are unobserved bridge days, excluded from this display. No future weather observations or numerical weather forecasts enter the model.',
             fontsize=9, color='#4d5962', linespacing=1.5)
    fig.subplots_adjust(left=.085, right=.975, top=.89, bottom=.13, hspace=.25)
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'deep_dillon_daily_outlook.{ext}', dpi=180)
    plt.close(fig)

    trials = result['hyperparameter_trials']
    fig, axes = plt.subplots(1, len(trials), figsize=(11.5, 3.5), squeeze=False)
    for ax, trial in zip(axes[0], trials):
        log = pd.DataFrame(trial['logs'])
        ax.plot(log.epoch, log.train_nll, label='Training', color=GRAY)
        ax.plot(log.epoch, log.validation_nll, label='Separate-winter validation', color=BLUE)
        ax.axvline(trial['best_epoch'], color=BLUE, lw=1, linestyle=':')
        ax.set_title(f"GRU width {trial['config']['hidden']} · {trial['parameter_count']:,} parameters", fontsize=11)
        ax.set_xlabel('Training epoch')
        ax.set_ylabel('Negative log likelihood')
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'deep_dillon_training_curves.png', dpi=160)
    plt.close(fig)

    daily_scores = result['daily_test_scores']
    nn = daily_scores['multivariate_gru']
    cal = daily_scores['daily_calendar_climatology']
    monthly_scores = result['fixed_nov1_monthly_test_scores']
    mn = monthly_scores['multivariate_gru']
    mc = monthly_scores['direct_monthly_climatology']
    dr = direct['heldout']['direct_monthly_ridge']['metrics']
    # Verify the independently implemented baseline uses the same ten labels.
    a = monthly_test.loc[(monthly_test.model == 'direct_monthly_climatology') & monthly_test.fixed_nov1]
    b = pd.DataFrame(direct['heldout']['direct_monthly_climatology']['predictions'])
    a = a.assign(month_number=pd.to_datetime(a.month).dt.month)
    paired = a.merge(b, left_on=['season_start', 'month_number'], right_on=['season_start_year', 'month'], suffixes=('_neural', '_direct'))
    assert len(paired) == 10
    assert np.allclose(paired.actual_inches_neural, paired.actual_inches_direct)
    assert abs(mc['rmse_inches'] - direct['heldout']['direct_monthly_climatology']['metrics']['rmse_inches']) < 1e-6

    daily_table = table(['Held-out daily forecast', 'RMSE (in)', 'MAE (in)', 'Snow-day Brier score', '80% range coverage'], [
        ['Historical calendar distribution', number(cal['rmse_inches'], 3), number(cal['mae_inches'], 3), number(cal['snow_occurrence_brier'], 3), f"{cal['central80_coverage']:.1%}"],
        ['Multivariate GRU', number(nn['rmse_inches'], 3), number(nn['mae_inches'], 3), number(nn['snow_occurrence_brier'], 3), f"{nn['central80_coverage']:.1%}"]])
    month_table = table(['Fixed October-cutoff approach', 'Monthly RMSE (in)', 'Monthly MAE (in)'], [
        ['Direct historical monthly mean', number(mc['rmse_inches']), number(mc['mae_inches'])],
        ['Direct monthly ridge on October weather aggregates', number(dr['rmse_inches']), number(dr['mae_inches'])],
        ['Sum of daily neural means', number(mn['rmse_inches']), number(mn['mae_inches'])]])
    forecast_rows = []
    for row in months.to_dict('records'):
        forecast_rows.append([row['month'], number(row['predicted_full_month_inches'], 1),
                              f"{row['q10_full_month_inches']:.1f}–{row['q90_full_month_inches']:.1f}",
                              'Nov 1 observed + Nov 2–30 modeled' if row['month'] == '2026-11' else 'Dec 1–31 modeled'])
    forecast_table = table(['Month', 'Expected snowfall (in)', 'Model-based 80% range (in)', 'Period included'], forecast_rows)
    trial_table = table(['GRU width', 'Parameters', 'Chosen epoch', 'Validation negative log likelihood'], [
        [t['config']['hidden'], f"{t['parameter_count']:,}", t['best_epoch'], number(t['validation_nll'], 4)] for t in trials])
    lead_table = table(['Lead after last input day', 'Calendar RMSE (in)', 'Neural RMSE (in)', 'Neural 80% coverage'], [
        [group + ' days', number(result['lead_group_test_scores']['daily_calendar_climatology'][group]['rmse_inches'], 3),
         number(v['rmse_inches'], 3), f"{v['central80_coverage']:.1%}"]
        for group, v in result['lead_group_test_scores']['multivariate_gru'].items()])
    daily_change = (1-nn['mse_square_inches']/cal['mse_square_inches'])*100
    monthly_change = (1-mn['rmse_inches']**2/mc['rmse_inches']**2)*100
    verdict = ('The neural model slightly improved monthly RMSE in this small test, but a reliable accuracy gain is not established.' if monthly_change > 0 else
               'The neural model did not improve held-out monthly RMSE over the historical monthly mean.')
    width = display.q90_inches-display.q10_inches
    summary = {'display_start': str(dates.iloc[0].date()), 'display_end': str(dates.iloc[-1].date()),
               'display_days': len(display), 'monthly_neural_vs_direct_climatology_mse_improvement_percent': monthly_change,
               'daily_neural_vs_climatology_mse_improvement_percent': daily_change,
               'display_remaining_november_mean_inches': float(display.loc[display.date.dt.month == 11, 'predicted_inches'].sum()),
               'display_daily_band_width_first7_mean_inches': float(width.iloc[:7].mean()),
               'display_daily_band_width_last7_mean_inches': float(width.iloc[-7:].mean()),
               'display_cumulative_band_width_first_day_inches': float(cumhi[0]-cumlo[0]),
               'display_cumulative_band_width_final_day_inches': float(cumhi[-1]-cumlo[-1]),
               'daily_band_monotonically_nondecreasing': bool((np.diff(width) >= -1e-7).all()),
               'display_total_expected_inches': float(display.predicted_inches.sum()),
               'display_total_q10_q90_inches': [float(cumlo[-1]), float(cumhi[-1])],
               'monthly_comparator_target_match_verified': True}
    (OUT / 'deep_dillon_report_summary.json').write_text(json.dumps(summary, indent=2))

    enso = ('The network also receives conservatively lagged RONI as ENSO context, including strong warm/cold indicators. The origin-known value stays fixed for every future lead: it never receives the realized future winter ENSO. These are revised historical values, so the backtest is pseudo-real-time rather than a reconstruction of the original published vintages. Current JAS 2026 RONI is 1.7. No separate ENSO ablation was run, so its importance is not established by this model.'
            if result.get('enso_predictor') else
            'ENSO was not included in this daily neural experiment; prior ENSO experiments in the project remain separate. This run therefore does not isolate ENSO’s contribution.')
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dillon daily neural snowfall experiment</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#182d3c;background:#f3f6f8;margin:0}}main{{max-width:1050px;margin:0 auto;background:white;padding:40px 44px 60px}}h1{{font-size:34px;line-height:1.2}}h2{{margin-top:36px;font-size:23px}}h3{{font-size:18px}}.meta{{color:#647682}}.finding{{border-left:4px solid #146b94;padding:12px 20px;background:#eef6fa}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #dbe3e8}}th{{background:#edf3f6}}a{{color:#146b94}}code{{font-size:13px}}li{{margin-bottom:8px}}.small{{font-size:13px;color:#53616c}}@media(max-width:700px){{main{{padding:22px 16px}}table{{font-size:12px}}th,td{{padding:7px 5px}}h1{{font-size:28px}}}}</style></head><body><main>
<p class="meta">Research experiment · November 3, 2026 · NOAA Dillon 1 E, Colorado</p>
<h1>Daily weather history → daily snowfall → monthly totals</h1>
<p class="finding"><strong>{verdict}</strong> The model was actually trained on multivariate daily observations, with architecture and stopping chosen on separate winters. Summing daily predictions is mathematically valid; it does not by itself establish greater forecast accuracy. Five untouched winters provide the final comparison.</p>
<h2>November–December 2026 outlook</h2>
<p>The figure displays the remaining future period, November 4–December 31. Blue steps are expected daily amounts; the shaded region is the central 80% of the fitted predictive distribution. The gray historical mean uses alpha = 0.5. A smooth or gently varying expected curve is compatible with actual weather consisting of dry days and occasional snowstorms.</p>
{embed(OUT / 'deep_dillon_daily_outlook.png', 'Daily expected snowfall and pointwise 80 percent prediction range, followed by cumulative snowfall and its path-based range')}
{forecast_table}
<p>Latest available observations end November 1, with zero new snow that day. November 2–3 are unobserved bridge days, not observed zeros. The full-November table includes their modeled amounts; the displayed November 4–30 expected total is <strong>{summary['display_remaining_november_mean_inches']:.1f} inches</strong>. December contains only forecast days. These amounts are for the Dillon station and cannot be treated as resort totals.</p>
<h2>Does uncertainty widen toward year end?</h2>
<p>It was not forced to widen. In this run it does widen gently: the mean daily band width is {summary['display_daily_band_width_first7_mean_inches']:.2f} inches during the first seven displayed days and {summary['display_daily_band_width_last7_mean_inches']:.2f} during the last seven. This is consistent with the conjecture for this forecast, but does not establish a universal lead-time effect: seasonal snowfall frequency also changes from November to December. Other forecasts can flatten or shrink. The cumulative band grows from {summary['display_cumulative_band_width_first_day_inches']:.2f} inches on the first displayed day to {summary['display_cumulative_band_width_final_day_inches']:.1f} by December 31. That is uncertainty about accumulated snow, a different quantity from uncertainty about one future day.</p>
<p>The band is a <strong>prediction range for realized snowfall</strong>, not a confidence interval for the mean. The 80% label is model-based and pointwise; it does not mean an entire two-month trajectory has an 80% chance of remaining inside it. Rare heavy storms can lie above its upper edge. The model does not identify reliable storm dates two months ahead.</p>
<h2>What was trained</h2>
<p>The dataset contains {status['table_rows']:,} calendar rows and {status['usable_days']['SNOW']:,} usable snowfall observations from October 2005 through November 1, 2026. Each forecast reads 60 preceding days of snowfall, precipitation, standing snow depth, and minimum/maximum temperature, with explicit missingness indicators and calendar information. Future observed weather is never supplied. {enso}</p>
<p>A two-layer gated recurrent network (GRU) encodes the daily sequence. A decoder uses the encoded history, known future calendar dates and forecast lead to predict 61 distributions. Each distribution has a probability of no snow and a lognormal positive-snow amount. The selected network has <strong>{result['parameter_count']:,} trainable parameters</strong>. It is a locally trained recurrent model, not a pretrained weather foundation model or a claimed reproduction of DeepAR.</p>
{trial_table}
{embed(OUT / 'deep_dillon_training_curves.png', 'Training and separate-winter validation negative log likelihood for the two prespecified recurrent-network widths')}
<p>Training winters: 2005/06–2016/17. Validation winters: 2017/18–2020/21. Untouched test winters: 2021/22–2025/26. Validation selected architecture and stopping by negative log likelihood; test results did not select hyperparameters. The chosen width-32 network reached its best validation loss at epoch 2; training continued through epoch 15 to check improvement, then restored that best point. The two widths had nearly tied validation losses. The selected configuration was refit for the fixed selected number of epochs on 2005/06–2020/21 before testing. A separate forecast checkpoint was then fit to all 21 completed winters, using the already selected settings. Training and validation curves are from the selection stage.</p>
<p>Overlapping daily windows create more learning examples but remain correlated. They do not turn 21 winters into thousands of independent seasonal climates. Each context and target stays within one winter group, with non-overlapping dates across training, validation and test partitions.</p>
<h2>Held-out daily accuracy</h2>
{daily_table}
<p>The daily test has {nn['origin_lead_pairs']:,} valid forecast-date/lead pairs across {nn['unique_target_days']:,} unique target dates. It uses weekly October–February origins plus month starts, with 61 days predicted at each origin. Repeated target days at different leads are different forecast tasks, not independent observations. Neural daily MSE changes by {daily_change:.1f}% relative to the calendar baseline, where a positive value means improvement. Neural fitted-training RMSE is {result['training_rmse_inches']:.3f} inches; held-out RMSE is {nn['rmse_inches']:.3f}. Different winter climates also affect that training/test comparison.</p>
{lead_table}
<p>The whole-winter bootstrap interval for neural minus calendar daily MSE is {result['daily_mse_winter_bootstrap']['whole_winter_bootstrap95'][0]:.4f} to {result['daily_mse_winter_bootstrap']['whole_winter_bootstrap95'][1]:.4f} square inches. Negative favors the network. With only five test winters, this interval is descriptive and sensitive to which winters were held out.</p>
<p>The neural q10–q90 band covers {nn['central80_coverage']:.1%} of all held-out daily values and {nn['central80_snow_day_coverage']:.1%} of snow days. Snowfall has a large probability mass at zero, so q10 is usually zero: a closed q10–q90 interval can naturally cover about 90% overall, even under a correctly specified distribution. Neither coverage number alone demonstrates miscalibration. The interval score, which penalizes both excessive width and misses, is {nn['central80_interval_score_inches']:.3f} inches for the neural model versus {cal['central80_interval_score_inches']:.3f} for the calendar baseline; lower is better.</p>
<h2>Does summing daily forecasts improve monthly accuracy?</h2>
{month_table}
<p>This comparison contains exactly ten monthly totals: November and December in each of five test winters. Each two-month forecast uses observations only through October 31. December predictions do not receive actual November temperatures or snowfall. The network’s monthly means are the sums of its original daily means, rather than sums of repeatedly updated one-day forecasts.</p>
<p>The daily neural approach changes monthly MSE by {monthly_change:.1f}% relative to the direct historical monthly mean. Monthly neural 80% range coverage is {mn['central80_coverage']:.0%}, or {round(mn['central80_coverage']*mn['month_predictions'])}/{mn['month_predictions']} totals. Ten totals cannot establish robust calibration. The direct ridge comparator uses October weather aggregates and validation-selected regularization. Because the GRU uses a longer, richer sequence and a different model class, this evaluates the overall approaches; it does not isolate a causal effect of temporal resolution alone.</p>
<p>The whole-winter bootstrap 95% interval for neural minus direct-climatology monthly MSE is {result['fixed_nov1_monthly_mse_winter_bootstrap']['direct_monthly_climatology']['whole_winter_bootstrap95'][0]:.1f} to {result['fixed_nov1_monthly_mse_winter_bootstrap']['direct_monthly_climatology']['whole_winter_bootstrap95'][1]:.1f} square inches, spanning substantial improvement and deterioration. Monthly MAE is slightly worse for the neural model. Across the broader month-start test, covering {result['all_monthstart_monthly_test_scores']['multivariate_gru']['month_predictions']} forecasts of {result['all_monthstart_monthly_test_scores']['multivariate_gru']['unique_target_months']} unique months, neural monthly RMSE is {result['all_monthstart_monthly_test_scores']['multivariate_gru']['rmse_inches']:.2f} inches versus {result['all_monthstart_monthly_test_scores']['direct_monthly_climatology']['rmse_inches']:.2f} for direct climatology. The November–December gain is therefore not consistent across origins, and greater accuracy from daily aggregation has not been established.</p>
<h2>How daily and monthly uncertainty are connected</h2>
<p>Expected monthly snow equals the sum of expected daily snow. Monthly uncertainty also depends on correlation between days: persistent storms and dry spells matter. We produce 4,096 coherent future paths by imposing ranks from complete, seasonally aligned historical snowfall blocks on the network’s daily distributions, a Schaake-style shuffle. Held-out evaluation uses only pre-test templates. Monthly and cumulative ranges are quantiles of path sums; daily interval endpoints are never added together.</p>
<p>This assumes historical temporal dependence remains useful. It does not include uncertainty over all possible fitted networks or atmospheric forecast uncertainty, and the bands have not been calibrated to guarantee their stated coverage. Snow occurrence is highly intermittent, so unconditional daily coverage can look high while coverage on snow days is substantially lower. The machine-readable results retain coverage, width and forecast errors.</p>
<h2>What this experiment establishes—and what it still needs</h2>
<p>The requested daily neural pipeline, monthly aggregation, held-out evaluation and two-month daily outlook are implemented. Accuracy is judged against matched baselines rather than training error. A forecast that locates upcoming storms needs information about the approaching atmosphere; this experiment contains historical station observations and seasonal context, but no numerical weather forecast inputs. Archived forecast ensembles and additional nearby stations would be the next informative additions, followed by the same held-out evaluation.</p>
<p>NOAA reporting dates are available, but original publication times and revised ENSO vintages are not fully archived. This is a date-indexed research backtest, not a verified reconstruction of real-time operations. Snow depth is used as an input, not substituted for fresh snowfall. Missing or flagged measurements remain missing.</p>
<h2>Files and primary sources</h2>
<ul><li><a href="deep_dillon_forecast_daily.csv">Daily forecast, including unobserved bridge</a> · <a href="deep_dillon_display_daily.csv">Displayed daily and cumulative values</a> · <a href="deep_dillon_forecast_monthly.csv">Monthly sums and ranges</a></li>
<li><a href="deep_dillon_results.json">Training, validation and test results</a> · <a href="deep_dillon_test_daily.csv">Held-out daily predictions</a> · <a href="deep_dillon_test_monthly.csv">Held-out monthly predictions</a></li>
<li><a href="deep_dillon_forecast_paths.npz">Correlated forecast paths</a> · <a href="deep_models/dillon_gru_forecast.pt">Trained forecast checkpoint</a> · <a href="../src/deep_dillon.py">Reproducible training source</a></li>
<li><a href="https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00052281.dly">NOAA Dillon daily observations</a> and <a href="https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt">GHCN-Daily format and quality flags</a>; local snapshot provenance: <a href="../data/dillon_current_observation_status.json">source dates and hashes</a>.</li>
<li><a href="https://arxiv.org/abs/1704.04110">Salinas et al., DeepAR</a>: probabilistic recurrent forecasting; methodological background, not this model’s validation.</li>
<li><a href="https://pubs.usgs.gov/publication/70027101">Clark et al., The Schaake Shuffle</a>: historical rank reordering to reconstruct dependence.</li>
<li><a href="https://www.nature.com/articles/s41586-024-08252-9">Price et al., GenCast</a>: atmospheric ensemble forecasting to 15 days; its results do not demonstrate two-month local daily skill for this model.</li></ul>
<p class="small">All snowfall amounts are inches of fresh snow. Forecast ranges describe this fitted experimental model. Prepared from the saved experiment without selecting or tuning against test outcomes.</p>
</main></body></html>'''
    (OUT / 'deep_dillon_report.html').write_text(body)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
