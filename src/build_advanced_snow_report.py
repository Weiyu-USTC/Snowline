"""Render the completed architecture comparison without selecting on test scores."""
from pathlib import Path
import base64
import html
import json
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from advanced_snow_models import architecture_metadata

OUT = ROOT/'outputs'
NAMES = {'daily_calendar_climatology': 'Calendar baseline', 'direct_monthly_climatology': 'Monthly baseline',
         'gru': 'Original GRU', 'tcn': 'Residual TCN', 'patch_transformer': 'Patch Transformer',
         'equal_family_mean': 'Equal-family ensemble', 'validation_selected_network': 'Validation-selected network'}


def table(headers, rows):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    result = json.loads((OUT/'advanced_snow_results.json').read_text())
    if 'final_models' not in result:
        raise RuntimeError('Training/refitting has not finished')
    metrics = result['metrics']
    names = ['gru', 'tcn', 'patch_transformer', 'equal_family_mean', 'validation_selected_network']
    configs = result['plan']['configs']
    architectures = [architecture_metadata(c) for c in configs]
    (OUT/'advanced_snow_architectures.json').write_text(json.dumps(architectures, indent=2))
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.2), gridspec_kw={'height_ratios': [1, 1]})
    colors = ['#89959c', '#2f6388', '#2a8b83', '#d98c31', '#756499', '#658b48']
    for col, (task, baseline, title) in enumerate([
        ('daily', 'daily_calendar_climatology', 'Daily RMSE · 1–61-day leads'),
        ('fixed_nov1_monthly', 'direct_monthly_climatology', 'November–December monthly RMSE')]):
        order = [baseline]+names
        scores = [metrics[task][name]['rmse_inches'] for name in order]
        ax = axes[0, col]
        ax.barh(np.arange(len(order)), scores, color=colors, height=.65)
        ax.set_yticks(np.arange(len(order)), [NAMES[x] for x in order], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(scores)*1.22)
        for y, val in enumerate(scores):
            ax.text(val+max(scores)*.025, y, f'{val:.3f}' if col == 0 else f'{val:.2f}', va='center', fontsize=10)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        ax.set_xlabel('RMSE (inches); lower is better')
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
        ax = axes[1, col]
        for i, name in enumerate(names):
            b = result['paired_winter_bootstrap'][name][task]
            point, interval = b['model_minus_baseline_mse_square_inches'], b['whole_winter_bootstrap95']
            ax.plot(interval, [i, i], color=colors[i+1], lw=2)
            ax.scatter([point], [i], color=colors[i+1], s=40, zorder=3)
        ax.axvline(0, color='#4a555d', linestyle='--', lw=1)
        ax.set_yticks(range(len(names)), [NAMES[x] for x in names], fontsize=9)
        ax.invert_yaxis()
        ax.set_title('MSE difference · 95% interval', loc='left', fontsize=11)
        ax.set_xlabel('MSE difference (inches²)\nNegative favors neural model')
        ax.grid(axis='x', alpha=.15)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Do deeper snowfall models improve unseen-winter forecasts?', fontsize=17, fontweight='bold', x=.04, ha='left')
    figure.text(.04, .93, 'Nine historical winters · three chronological outer blocks · settings selected using earlier winters', fontsize=11)
    figure.text(.04, .025, 'Retrospective exploratory comparison: some winters were examined in prior experiments.\n'
                'Intervals resample whole winters; overlapping daily forecasts are not independent examples.', fontsize=9, color='#53616c')
    figure.subplots_adjust(left=.20, right=.98, top=.86, bottom=.12, hspace=.44, wspace=.86)
    for ext in ['png', 'svg']:
        figure.savefig(OUT/f'advanced_snow_comparison.{ext}', dpi=180, facecolor='white')
    plt.close(figure)

    rows = []
    for name in ['baseline']+names:
        d = metrics['daily']['daily_calendar_climatology' if name == 'baseline' else name]
        m = metrics['fixed_nov1_monthly']['direct_monthly_climatology' if name == 'baseline' else name]
        rows.append(['Historical baselines' if name == 'baseline' else NAMES[name], f"{d['rmse_inches']:.3f}",
                     f"{d['mae_inches']:.3f}", f"{m['rmse_inches']:.2f}", f"{m['mae_inches']:.2f}"])
    scores_table = table(['Approach', 'Daily RMSE', 'Daily MAE', 'Monthly RMSE', 'Monthly MAE'], rows)
    arch_table = table(['Family', 'Implemented structure', 'Trainable parameters tested'], [
        ['GRU reference', 'Two recurrent layers + three linear decoder transformations', '24,899 / 58,371'],
        ['Residual TCN', 'Five residual blocks, two dilated causal convolutions each; 10 convolution layers', '50,275 / 93,523'],
        ['Custom patch Transformer', '12 five-day history tokens; three attention encoder blocks; forecast-query cross-attention', '99,315 / 173,379']])
    split_table = table(['Evaluation winters (start years)', 'Fit winters', 'Inner validation', 'Family selected inside training'], [
        [f"{f['test_years'][0]}–{f['test_years'][-1]}", f"{f['fit_years'][0]}–{f['fit_years'][-1]}",
         f"{f['inner_validation_years'][0]}–{f['inner_validation_years'][-1]}", NAMES[f['selected_family']]] for f in result['folds']])
    choices_table = table(['Outer block', 'Family', 'Chosen parameters', 'Chosen epoch', 'Inner validation NLL', 'Fitted training RMSE'], [
        [f"{f['test_years'][0]}–{f['test_years'][-1]}", NAMES[name], f"{c['parameter_count']:,}", c['best_epoch'],
         f"{c['validation_nll']:.4f}", f"{f['refits'][name]['training_rmse_inches']:.3f}"]
        for f in result['folds'] for name, c in f['choices'].items()])
    prob_table = table(['Family', 'Daily NLL', 'Snow-occurrence Brier', 'q10–q90 interval score'], [
        [NAMES[name], f"{r['mean_negative_log_likelihood']:.4f}" if 'mean_negative_log_likelihood' in r else '—',
         f"{r['snow_occurrence_brier']:.4f}", f"{r['central80_interval_score_inches']:.3f}"]
        for name, r in result['daily_probabilistic_metrics'].items()])
    broad_table = table(['Approach', 'All month-start RMSE (in)', 'All month-start MAE (in)'], [
        [NAMES[name], f"{metrics['all_monthstart_monthly'][name]['rmse_inches']:.2f}",
         f"{metrics['all_monthstart_monthly'][name]['mae_inches']:.2f}"]
        for name in ['direct_monthly_climatology']+names])
    image = base64.b64encode((OUT/'advanced_snow_comparison.png').read_bytes()).decode()
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dillon: stronger neural model designs and benchmark</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f3f6f8;color:#182d3c;margin:0}}main{{max-width:1100px;margin:auto;padding:42px;background:white}}h1{{font-size:34px;line-height:1.2}}h2{{margin-top:34px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{text-align:left;border-bottom:1px solid #dae3e8;padding:10px}}th{{background:#edf3f6}}img{{width:100%}}a{{color:#146b94}}.note{{border-left:4px solid #146b94;padding:12px 20px;background:#eef6fa}}.small{{font-size:13px;color:#53616c}}.flow{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}.box{{background:#edf3f6;border:1px solid #d6e1e8;padding:14px;flex:1;min-width:150px}}@media(max-width:700px){{main{{padding:22px 15px}}table{{font-size:12px}}th,td{{padding:6px}}}}</style></head><body><main>
<p class="small">Dillon snowfall research · November 4, 2026 · frozen observation snapshot through November 1</p>
<h1>Beyond the baseline GRU: stronger temporal architectures</h1>
<p class="note">Two layers do not make a neural model merely theoretical. The original GRU performs recurrent computation across 60 days and has an additional three-stage decoder. Its size was appropriate as a small-data baseline. The new experiment implements and trains deeper convolutional and attention alternatives, then tests whether their added capacity helps.</p>
<p><strong>Result: added depth did not deliver consistently better forecasts in this pilot.</strong> Fixed-origin monthly RMSE is {metrics['fixed_nov1_monthly']['gru']['rmse_inches']:.2f} inches for the GRU, {metrics['fixed_nov1_monthly']['tcn']['rmse_inches']:.2f} for the TCN, and {metrics['fixed_nov1_monthly']['patch_transformer']['rmse_inches']:.2f} for the Transformer, versus {metrics['fixed_nov1_monthly']['direct_monthly_climatology']['rmse_inches']:.2f} for historical monthly means. The TCN has better monthly MAE ({metrics['fixed_nov1_monthly']['tcn']['mae_inches']:.2f} versus {metrics['fixed_nov1_monthly']['direct_monthly_climatology']['mae_inches']:.2f}), so there is a metric tradeoff. The family-level paired MSE intervals span both improvement and deterioration. These results do not show that deeper architectures can never help; they show no robust gain from these particular configurations on this dataset.</p>
<h2>What is now implemented</h2>{arch_table}
<div class="flow"><div class="box">60 observed days<br>Weather + missingness</div><span>→</span><div class="box">GRU / residual TCN / patch attention</div><span>→</span><div class="box">Known calendar + lagged ENSO<br>Probabilistic snowfall decoder</div><span>→</span><div class="box">61 daily distributions<br>Sum means for monthly forecasts</div></div>
<p>The TCN learns patterns at several temporal scales through dilations 1, 2, 4, 8 and 16, with residual connections. Its theoretical receptive field is 125 days, but only the same 60 observed days are supplied; padding is not additional data. The Transformer makes five-day multivariate patches and lets each forecast lead attend to the observed history. It is a custom patch Transformer inspired by patch-based forecasting, not a faithful PatchTST implementation.</p>
<p>All three families retain the same observed inputs, conservatively lagged ENSO, 60-day context, 61-day horizon, and zero-inflated-lognormal snowfall output distribution. They train with the same likelihood objective, optimizer, batch size and stopping rule. Architecture, parameter count and dropout differ. An equal-weight ensemble averages the three conditional means with weights fixed before evaluation.</p>
<h2>Matched historical results</h2>{scores_table}
<p>All errors are inches of fresh snowfall; lower is better. Daily scores include {metrics['daily']['gru']['n']:,} valid origin/lead predictions across nine winters. The fixed-origin monthly comparison contains {metrics['fixed_nov1_monthly']['gru']['n']} complete November and December totals. Both months are forecast after October 31, without inserting observed November weather into December inputs. Monthly predictions are sums of the original daily predictions.</p>
<img src="data:image/png;base64,{image}" alt="Daily and monthly errors for all neural families, with whole-winter paired uncertainty intervals">
<p>These new scores must not be compared directly with the earlier five-winter scores: the evaluation winters, training partitions and selected configurations have changed. The validation-selected row scores the entire family-selection procedure, using only inner-validation losses. The ensemble row is an arithmetic mean fixed in advance, not a blend weighted using outer errors.</p>
<h2>Why the evaluation changed</h2>{split_table}
<p>Each outer model fits winters strictly before its evaluation block. The last three fit winters form an inner validation set for choosing width and stopping; preprocessing is fitted only on the earlier inner-training winters. The selected model then refits all outer-fit winters for a fixed number of epochs. Observations in an outer evaluation block do not choose its model, scaling or historical uncertainty templates. A past block may legitimately enter training for a later block.</p>
<p><strong>This is retrospective exploratory evidence.</strong> Several historical winters were inspected in earlier experiments, so isolating the new code’s folds does not make those years fresh confirmatory data. Candidate architectures and selection rules were frozen before this run. The paired intervals resample whole winters, rather than treating thousands of correlated daily windows as independent climates. Future prospective seasons would provide stronger confirmation.</p>
{choices_table}
<p>The training cap was 50 epochs, with at least 12 epochs checked and patience of eight epochs without validation improvement; the best earlier checkpoint was retained. Two prespecified configurations per family were tested, with one fixed random initialization per fold. This is a controlled pilot rather than an exhaustive search for the best possible version of each architecture. Selection minimizes probability-distribution loss, which need not also minimize daily or monthly RMSE.</p>
<h2>Other scores and broader monthly checks</h2>{prob_table}
<p>NLL evaluates the fitted probability distribution; Brier score evaluates snow occurrence; interval score penalizes wide ranges and misses. Lower is better. Empirical climatology does not have the same continuous density model, so its NLL is omitted. The equal-family ensemble is evaluated for point forecasts only here. Large probability mass at zero means a q10–q90 closed interval can cover more than 80% even when the distribution is correctly specified.</p>
{broad_table}
<p>The broader monthly check includes every complete month spanned by a month-start forecast, including multiple forecasts of some target months. It is a separate task from the fixed October-issued November–December comparison; it does not add independent winters.</p>
<h2>The stronger engineering design I recommend next</h2>
<p>Depth is only one part of the design. A useful local snowfall system should combine a regional observation encoder with an atmospheric-forecast encoder, then learn a calibrated local snowfall distribution. Adding information about approaching precipitation, wind, humidity and temperature profiles is a more substantive change than asking a larger network to reconstruct missing atmospheric information from one station.</p>
<ol><li><strong>Regional observations:</strong> train a shared encoder across nearby stations, using elevation/location and measurement-quality masks. Use actual resort snowfall labels to calibrate resort outputs; do not relabel Dillon as A-Basin or Keystone.</li>
<li><strong>Forecast inputs available at issuance:</strong> use archived numerical-weather ensembles, including precipitation and temperature/humidity profiles. Training must use what was forecast at the time, rather than future observations or later analyses.</li>
<li><strong>Different horizon responsibilities:</strong> model daily storm probabilities at short leads and weekly/monthly distributions at long leads. A sampled 60-day sequence is a plausible scenario, not a claim to know individual storm dates.</li>
<li><strong>Probabilistic outputs and coherent aggregation:</strong> model snow occurrence separately from positive amount. Preserve storm and dry-spell dependence when summing scenarios. Calibrate on training/validation winters, then assess performance on held-out winters.</li>
<li><strong>Operational controls:</strong> version source snapshots, publication times, model checkpoints and forecast issuance; monitor baseline-relative error and calibration after deployment. The winning encoder can remain small if it performs better.</li></ol>
<p>These additional forecast and multi-station inputs are proposed next work, not features silently included in the current comparison. Recent <a href="https://www.nature.com/articles/s41586-026-11126-x">October 2026 subseasonal research</a> learns corrections to ensemble forecast distributions from archived forecasts and observations. It supports that design direction, but does not establish snowfall skill at Dillon.</p>
<h2>Artifacts and sources</h2><ul>
<li><a href="advanced_snow_results.json">Full results, model selections and training histories</a> · <a href="advanced_snow_frozen_plan.json">Frozen benchmark plan</a> · <a href="advanced_snow_code_snapshot.json">Training-source hashes</a></li>
<li><a href="advanced_snow_daily_predictions.csv">Daily out-of-sample predictions</a> · <a href="advanced_snow_monthly_predictions.csv">Monthly out-of-sample sums</a> · <a href="advanced_snow_comparison.svg">Vector comparison figure</a></li>
<li><a href="../src/advanced_snow_models.py">Implemented architectures</a> · <a href="../src/benchmark_advanced_snow.py">Training and evaluation source</a> · <a href="../tests/test_advanced_snow.py">Independent model and information-boundary tests</a></li>
<li>Full-history trained checkpoints: <a href="advanced_models/gru_all_completed_winters.pt">GRU</a>, <a href="advanced_models/tcn_all_completed_winters.pt">TCN</a>, <a href="advanced_models/patch_transformer_all_completed_winters.pt">patch Transformer</a>. These final fits occur after evaluation and do not replace the fold-specific checkpoints used for scores.</li>
<li><a href="https://arxiv.org/abs/1803.01271">Bai et al., temporal convolutional networks</a>; <a href="https://arxiv.org/abs/2211.14730">Nie et al., PatchTST</a>; <a href="https://arxiv.org/abs/1912.09363">Lim et al., Temporal Fusion Transformers</a>.</li>
<li><a href="../research/advanced_weather_methods.txt">Primary-source research and production design</a> · <a href="../research/advanced_daily_design.txt">Evaluation audit</a>.</li></ul>
<p class="small">No new current forecast was issued in this comparison. The November 3 forecast artifacts remain preserved. Historical observation-publication times and unrevised ENSO vintages remain limitations.</p>
</main></body></html>'''
    (OUT/'advanced_snow_report.html').write_text(body)
    print(json.dumps({'report': str(OUT/'advanced_snow_report.html'), 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
