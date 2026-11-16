"""Combine shuffled-winter validation, neural forecast curves and data research."""
from pathlib import Path
import base64
import html
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
NAMES = {'gru': 'GRU', 'tcn': 'Residual TCN', 'patch_transformer': 'Patch Transformer',
         'equal_family_mean': 'Equal-family ensemble', 'validation_selected_network': 'Validation-selected network'}


def table(headers, rows):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    result = json.loads((OUT / 'shuffled_snow_results.json').read_text())
    if 'production_models' not in result:
        raise RuntimeError('Shuffled training and final fitting have not finished')
    forecast = json.loads((OUT / 'shuffled_forecast_manifest.json').read_text())
    daily = pd.read_csv(OUT / 'shuffled_forecast_daily.csv', parse_dates=['date'])
    months = pd.read_csv(OUT / 'shuffled_forecast_monthly.csv')
    assignment = pd.read_csv(OUT / 'shuffled_snow_fold_assignment.csv')
    balance = result['fold_balance']
    metrics = result['metrics']
    families = ['gru', 'tcn', 'patch_transformer']
    score_rows = []
    for name in ['baseline'] + families + ['equal_family_mean', 'validation_selected_network']:
        d = metrics['daily']['daily_calendar_climatology' if name == 'baseline' else name]
        m = metrics['fixed_nov1_monthly']['direct_monthly_climatology' if name == 'baseline' else name]
        score_rows.append(['Historical baselines' if name == 'baseline' else NAMES[name],
                           f"{d['rmse_inches']:.3f}", f"{d['mae_inches']:.3f}", f"{m['rmse_inches']:.2f}", f"{m['mae_inches']:.2f}"])
    scores = table(['Approach', 'Daily RMSE (in)', 'Daily MAE (in)', 'Monthly RMSE (in)', 'Monthly MAE (in)'], score_rows)
    ranges = table(['Model', 'Monthly MSE difference from historical mean (in²)', 'Whole-winter bootstrap 95% interval'], [
        [NAMES[f], f"{result['paired_winter_bootstrap'][f]['fixed_nov1_monthly']['model_minus_baseline_mse_square_inches']:.2f}",
         ' to '.join(f'{x:.2f}' for x in result['paired_winter_bootstrap'][f]['fixed_nov1_monthly']['whole_winter_bootstrap95'])]
        for f in families])
    fold_table = table(['Fold', 'Held-out winter start years', 'Warm / neutral / cold JAS', 'Complete Nov–Dec total: mean [min–max]'], [
        [b['outer_fold']+1, ', '.join(str(y) for y in b['season_start_years']),
         f"{b['warm_jas_seasons']} / {b['neutral_jas_seasons']} / {b['cold_jas_seasons']}",
         f"{b['nov_dec_total_inches_mean']:.1f} [{b['nov_dec_total_inches_min']:.1f}–{b['nov_dec_total_inches_max']:.1f}] in; n={b['nov_dec_total_inches_n_complete']}"]
        for b in balance])
    monthly_rows = []
    for f in families:
        for row in months.loc[months.model.eq(f)].itertuples():
            monthly_rows.append([NAMES[f], row.month, f'{row.predicted_full_month_inches:.1f}',
                                 f'{row.q10_full_month_inches:.1f}–{row.q90_full_month_inches:.1f}'])
    monthly_table = table(['Model', 'Month', 'Expected snowfall (in)', 'Model-based q10–q90 range (in)'], monthly_rows)
    settings = table(['Model', 'Parameters', 'Chosen fit epochs', 'Configuration'], [
        [NAMES[f], f"{forecast['models'][f]['architecture']['trainable_parameters']:,}", forecast['models'][f]['epochs'],
         ', '.join(f'{k}={v}' for k, v in forecast['models'][f]['config'].items() if k != 'family')]
        for f in families])
    image = base64.b64encode((OUT / 'shuffled_forecast_curves.png').read_bytes()).decode()
    snow_widths = {}
    for f in families:
        part = daily.loc[daily.model.eq(f) & daily.date.between('2026-11-04', '2026-12-31')]
        width = part.q90_inches - part.q10_inches
        snow_widths[f] = {'daily_width_first7_inches': float(width.iloc[:7].mean()),
                          'daily_width_last7_inches': float(width.iloc[-7:].mean()),
                          'display_total_mean_inches': float(part.predicted_inches.sum())}
    proof_path = ROOT / 'data/forecast_input_proof/decoded_gefs_sample.json'
    proof = json.loads(proof_path.read_text())
    sample_rows = []
    for record in proof['records']:
        meta, point = record['metadata'], record['nearest_grid_point']
        value = point['value']
        shown = f"{value:.3f} {meta['units']}"
        if meta['units'] == 'K':
            shown += f" = {value-273.15:.2f} °C"
        sample_rows.append([meta['name'], f"{meta['dataDate']} {meta['dataTime']:04d} UTC",
                            f"{meta['validityDate']} {meta['validityTime']:04d} UTC", shown])
    proof_table = table(['Decoded forecast variable', 'GRIB reference time', 'Valid time', 'Nearest-grid forecast value'], sample_rows)
    summary = {'prepared_date': '2026-11-04', 'observation_cutoff': forecast['observation_cutoff'],
               'folds': 5, 'inner_folds': 2, 'total_neural_fits': result['total_fit_count'],
               'production_selected_family': result['production_selected_family'], 'monthly_forecast': months.to_dict('records'),
               'display_band_widths': snow_widths, 'warm_jas_seasons': int(assignment.jas_enso_phase.eq('warm').sum()),
               'strong_warm_jas_seasons': int(assignment.strong_warm_jas.sum()),
               'daily_scores': metrics['daily'], 'fixed_nov1_monthly_scores': metrics['fixed_nov1_monthly']}
    (OUT / 'shuffled_snow_report_summary.json').write_text(json.dumps(summary, indent=2))

    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shuffled-winter validation and advanced snowfall forecasts</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f3f6f8;color:#182d3c;margin:0}}main{{max-width:1120px;margin:auto;padding:40px;background:white}}h1{{font-size:34px;line-height:1.2}}h2{{margin-top:35px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{text-align:left;border-bottom:1px solid #dae3e8;padding:10px}}th{{background:#edf3f6}}img{{width:100%;height:auto}}a{{color:#146b94}}.note{{border-left:4px solid #146b94;padding:12px 20px;background:#eef6fa}}.small{{font-size:13px;color:#53616c}}li{{margin-bottom:8px}}@media(max-width:700px){{main{{padding:22px 15px}}table{{font-size:12px}}th,td{{padding:6px}}}}</style></head><body><main>
<p class="small">Dillon 1 E · November 4, 2026 · nested cross-validation, forecast comparison and engineering research</p>
<h1>Shuffle winters, test the models, then show their forecasts</h1>
<p class="note">The requested shuffled cross-validation is now implemented and trained. Entire winters remain intact. The final GRU, residual TCN and patch Transformer were each refit after cross-validation, and their daily and cumulative November–December curves are shown below. Atmospheric forecast data access has also been demonstrated with actual decoded public records; those inputs have not yet been added to the neural models.</p>
<h2>Why shuffle winters rather than days?</h2>
<p>Your proposed independent-year assumption supports a shuffled-winter experiment. Each season can be wet or dry and have a different climate context. Randomly shuffling individual daily rows would let neighboring days from the same storm, and overlapping 60-day contexts or 61-day targets, appear in both fitting and evaluation. That would overstate the information available when forecasting a new winter.</p>
<p>The new run uses five shuffled outer folds across 21 winter groups, with two shuffled inner folds inside each outer training set. The inner folds select architecture settings and stopping; the held-out outer winters score those decisions. All three neural families use the same inputs, probability model and forecast targets, with two candidate settings per family. The full experiment performed {result['total_fit_count']} neural fits, including the final all-history models. The seed and rules were frozen before examining results; no favorable seed was searched for.</p>
<p>The earlier chronological test remains useful: it asks how a model trained on earlier winters performs on later winters. Shuffled validation instead assumes winters are exchangeable and can train on later years when evaluating an earlier one. It addresses sensitivity to chronological partitioning but does not establish that climate drift is absent. Both are retrospective research because some years were examined in earlier experiments.</p>
<h2>ENSO and snowfall balance</h2>{fold_table}
<p>Warm/cold here means the <strong>pre-November JAS RONI reading</strong> is at least +0.5°C or at most −0.5°C. It is not a classification based on the future winter’s realized ENSO. Only two of the 21 years meet the warm threshold at this issuance point, and only one meets +1.5°C. Consequently, five folds cannot each contain a warm or strong-warm case. The two warm cases fall in different folds in this frozen shuffle, while one fold contains only cold cases. Randomization does not guarantee perfect climate balance.</p>
<p>Snowfall totals in this table are a post-assignment diagnostic, not a feature used to forecast the same winter or a reason to redraw the folds. Incomplete totals remain missing. ENSO is an input to all three neural models; these comparisons do not isolate its incremental value. That would require a with/without-ENSO comparison on the same folds. Predictive evaluation alone would still not demonstrate causality.</p>
<h2>Cross-validated accuracy</h2>{scores}
<p>The models remain close. The TCN’s fixed-origin monthly RMSE is {metrics['fixed_nov1_monthly']['tcn']['rmse_inches']:.2f} inches versus {metrics['fixed_nov1_monthly']['direct_monthly_climatology']['rmse_inches']:.2f} for the historical monthly mean, a small point difference. The Transformer has lower monthly MAE but higher RMSE than that baseline. Shuffling has not revealed a consistently superior neural family; the paired uncertainty below is important when interpreting these rankings.</p>
<p>There are {result['coverage']['daily_unique_test_winters']} evaluated winter groups, {metrics['daily']['gru']['n']:,} valid daily origin/lead pairs and {result['daily_probabilistic_metrics']['gru']['unique_target_days']:,} unique observed target dates with scores. Four additional requested target dates lack usable observations and are not scored. The fixed-November-origin comparison contains {metrics['fixed_nov1_monthly']['gru']['n']} monthly totals across {len(result['coverage']['fixed_nov1_test_winters'])} winters. The 2005 November origin lacks the full preceding context; models use the same eligible targets. Daily windows within a winter are dependent.</p>
<p>Monthly totals are obtained by summing daily predictions issued once after October 31. December does not receive actual November weather in that fixed-origin evaluation. The validation-selected row evaluates choosing a family by its inner validation likelihood; the equal-family ensemble averages the three means with weights fixed in advance. Settings are not selected using outer error rankings.</p>
{ranges}
<p>Negative differences favor the neural model. These descriptive intervals resample whole winters. They do not erase the exploratory nature of repeated work on a small historical record. The new 21-winter scores and earlier nine-winter chronological scores cover different cohorts and training sample sizes, so a numerical change between reports is not proof that shuffling improved forecasting skill. More flexible networks and randomization can change the estimate without adding useful predictive information.</p>
<h2>Daily forecasts from the cross-validation-selected fits</h2>
<p>For a fair model comparison, all curves use the same observations through <strong>November 1, 2026</strong>. This is a frozen-input comparison prepared November 4, not a new weather-data refresh. November 2–3 remain unobserved bridge predictions. The displayed period is November 4–December 31; November 4 is explicitly forecast, not observed.</p>
<img src="data:image/png;base64,{image}" alt="Three rows comparing GRU, residual convolution and Transformer daily snowfall predictions and cumulative snowfall ranges">
{monthly_table}
<p>November’s full-month total combines observed November 1 with modeled November 2–30. December is entirely predicted. The daily curves are conditional expected amounts, not a claim that snow falls every day at that amount. Actual realizations contain dry days and storm spikes. Gray dashed lines show the historical calendar mean at alpha = 0.5; shared vertical scales make the models comparable.</p>
<p>The shaded q10–q90 ranges are model-based prediction ranges, not confidence intervals for the mean or calibrated guarantees. The daily lower bound is often zero because of the probability of no snow. Monthly and cumulative ranges use sums of 4,096 correlated daily paths per model, rather than sums of daily range endpoints. Daily band widening is not imposed; seasonal variation and model dependence on lead time can change its shape. The paths retain historical dependence assumptions and do not fully represent uncertainty about the fitted parameters.</p>
{settings}
<p>After the outer evaluation was frozen, a separate prespecified two-fold shuffled procedure over all 21 completed winters selected each family’s final configuration and epoch count. Each was then refit on all 21 winters. The production family selected by that internal likelihood criterion is <strong>{NAMES[result['production_selected_family']]}</strong>; all three are displayed for transparency. This selection is not a claim that its outer scores establish superiority. These refits differ from the prior report’s checkpoints, so forecast differences should not be interpreted as new weather evidence.</p>
<h2>Concrete progress toward atmospheric forecast inputs</h2>
<p>I verified a public NOAA GEFSv12 reforecast archive and downloaded two individual GRIB records using byte ranges, rather than retrieving full global files. They were decoded with ECMWF ecCodes. The sample is the control-member forecast dated November 1, 2019, at the nearest 0.25° grid point to Dillon: 39.5°N, 106.0°W, about 14.13 km away.</p>
{proof_table}
<p>These are historical model forecast values, not observed weather or today’s forecast. The successful download and decode establishes data-access feasibility; it does not establish snowfall model accuracy. The precipitation record covers hours 0–3. Other archive records overlap accumulation windows, so precipitation must be de-accumulated from the actual GRIB intervals rather than blindly summed.</p>
<p>The archive has daily five-member reforecasts to 16 days and weekly 11-member extensions to 35 days for 2000–2019. Those standardized retrospective runs differ from the original forecasts issued in those historical years. We must also verify the operational model version used for later/live predictions. The sample’s GRIB reference time is 00 UTC; a publication describes a different phase-specific initialization convention. That unresolved convention is retained in the source audit. Inconsistent ensemble-count header fields also mean member availability must be checked from actual archive objects.</p>
<p>Public access does not make full extraction small: a byte range selects a compressed global field, not one geographic point. At the sample field sizes, two variables, 56 three-hour leads, five members and 20 years of daily initializations would transfer roughly 2.44 TB. This is an illustrative estimate, not a measured full-archive size. Start with a few dates and leads, then a bounded pilot; extract all regional stations from each downloaded field and cache the compact local features.</p>
<h2>Recommended next implementation</h2>
<ol><li><strong>Build a regional forecast/observation table:</strong> extract a small neighborhood around Dillon, Climax, Vail and eventual resort stations. Preserve forecast reference time, valid time, lead, ensemble member and grid elevation. Align accumulated precipitation with the station’s actual observation window; do not silently assume midnight UTC.</li>
<li><strong>Train a forecast correction model:</strong> combine the observed-history encoder with ensemble precipitation, temperature/humidity profiles, wind and freezing-level forecasts. Learn snow occurrence and positive amount locally. Do not replace fresh snowfall with SWE or standing depth, and do not assume a fixed snow-to-liquid ratio.</li>
<li><strong>Separate useful horizons:</strong> evaluate daily storm forecasts over days 1–7, then extend to days 8–16 where supported. Use the longer reforecasts for weekly/monthly probabilities; a 35-day archive alone cannot supply atmospheric inputs for every day of a 61-day outlook.</li>
<li><strong>Retain both validation views:</strong> shuffled whole-winter nested validation under the independent-year assumption, plus chronological and eventually prospective tests. Compare against raw forecast ensembles and climatology, and evaluate uncertainty as well as daily/monthly point errors.</li></ol>
<p>Recent <a href="https://www.nature.com/articles/s41586-026-11126-x">subseasonal probabilistic forecast-correction research</a> supports this engineering direction. The research note also assesses HRRR, ECMWF/S2S, current GEFS access and reanalysis limitations. The present GRU/TCN/Transformer forecasts still use station history and ENSO only; the atmospheric sample has not been included in their training.</p>
<h2>Files and sources</h2><ul>
<li><a href="shuffled_snow_results.json">Full cross-validation results and training histories</a> · <a href="shuffled_snow_frozen_plan.json">Frozen plan</a> · <a href="shuffled_snow_fold_assignment.csv">Winter assignments and ENSO/snow diagnostics</a></li>
<li><a href="shuffled_forecast_daily.csv">Daily model predictions</a> · <a href="shuffled_forecast_monthly.csv">Monthly sums</a> · <a href="shuffled_forecast_manifest.json">Forecast provenance and assumptions</a> · <a href="shuffled_forecast_curves.svg">Vector figure</a></li>
<li><a href="../research/forecast_input_feasibility.txt">Atmospheric-input feasibility research</a> · <a href="../data/forecast_input_sources.json">Source and access audit</a> · <a href="../data/forecast_input_proof/decoded_gefs_sample.json">Decoded public forecast proof</a></li>
<li><a href="https://psl.noaa.gov/forecasts/reforecast2/">NOAA reforecast archive guidance</a> · <a href="https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf">GEFSv12 archive documentation</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html">Grouped cross-validation documentation</a></li>
<li><a href="../src/crossvalidate_advanced_snow.py">Cross-validation implementation</a> · <a href="../src/issue_advanced_snow_forecast.py">Forecast generation</a> · <a href="advanced_snow_report.html">Earlier chronological comparison</a>.</li></ul>
<p class="small">All snowfall values are inches of new snow at Dillon 1 E. Research forecasts are not interchangeable with Copper Mountain, A-Basin or Keystone resort snowfall.</p>
</main></body></html>'''
    (OUT / 'shuffled_snow_report.html').write_text(body)
    print(json.dumps({'report': str(OUT / 'shuffled_snow_report.html'), 'monthly_forecasts': summary['monthly_forecast'],
                      'daily_scores': metrics['daily'], 'monthly_scores': metrics['fixed_nov1_monthly']}, indent=2))


if __name__ == '__main__':
    main()
