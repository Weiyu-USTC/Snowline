"""Build an auditable learning-history report from frozen logs and diagnostics.

No training, checkpoint selection, or model-file mutation takes place here.
"""
from pathlib import Path
import base64
import hashlib
import html
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/alta-nowcast'
OUT = BASE / 'learning-review'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from report_alta_extensions import tuning_section, accumulation_section, transfer_section

NAMES = ['convlstm', 'dgmr', 'nowcastnet']
LABELS = {'convlstm': 'ConvLSTM', 'dgmr': 'DGMR adaptation', 'nowcastnet': 'NowcastNet adaptation'}
COLORS = {'convlstm': '#257db0', 'dgmr': '#c17a32', 'nowcastnet': '#8055a6'}
RECIPES = {'uniform_l1': 'Initial recipe', 'wet_mse': 'Wet-case sampling + squared loss',
           'wet_mse_linear_head': 'Wet-case recipe + direct-rate head'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
    'axes.labelcolor': '#18383e', 'text.color': '#18383e', 'axes.edgecolor': '#a8b9bd',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=175, bbox_inches='tight')
    plt.close(fig)


def clean(ax):
    ax.grid(alpha=.16)
    ax.set_axisbelow(True)


def image(name, alt):
    raw = (OUT / f'{name}.png').read_bytes()
    return f'<img src="data:image/png;base64,{base64.b64encode(raw).decode()}" alt="{html.escape(alt)}">'


def table(frame, formats=None):
    return frame.to_html(index=False, border=0, classes='data-table', escape=True,
                         formatters=formats or {}, na_rep='Unavailable')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    new_training = tuning_section()
    observed_accumulation = accumulation_section()
    other_mountains = transfer_section()
    selected = {x['model']: x for x in json.loads((BASE / 'models/selected/recipe_selection.json').read_text())}
    bundles = {n: json.loads((BASE / f'models/selected/{n}.pt.json').read_text()) for n in NAMES}
    logs = {n: pd.DataFrame(bundles[n]['training_log']) for n in NAMES}
    case = pd.read_csv(BASE / 'short_case_metrics.csv', parse_dates=['date'])
    summary = pd.read_csv(BASE / 'short_summary.csv')
    # Mean per-case MSE, matching the frozen evaluator; not mean daily RMSE.
    test_mse = case.groupby(['model', 'hours'], as_index=False)['mse'].mean()
    merged = summary.merge(test_mse, on=['model', 'hours'], validate='one_to_one')
    assert np.allclose(merged.rmse_mm ** 2, merged.mse, rtol=1e-10, atol=1e-12)
    common = pd.read_csv(OUT / 'checkpoint_losses.csv')
    audit = json.loads((OUT / 'checkpoint_loss_audit.json').read_text())
    # The helper's column schema is explicitly normalized at the input boundary.
    common = common.rename(columns={'rate_mse_mm_h_squared': 'mse', 'rate_mae_mm_h': 'mae'})
    if 'mse' not in common:
        raise ValueError(f'Unexpected diagnostic columns: {list(common)}')

    all_logs, candidates, rows = [], [], []
    for name in NAMES:
        for candidate in selected[name]['candidates']:
            recipe = candidate['recipe']
            file = BASE / f'models/recipe_{recipe}/{name}.pt.training.json'
            for row in json.loads(file.read_text()):
                all_logs.append({'model': name, 'recipe': recipe, **row,
                    'selected_recipe': recipe == selected[name]['selected']['recipe']})
            candidates.append({'Model': LABELS[name], 'Recipe': RECIPES[recipe],
                'Completed steps': candidate['completed_logged_steps'], 'Best step': candidate['best_step'],
                'Best validation rate MSE': candidate['development_mse'],
                'Used for final test': recipe == selected[name]['selected']['recipe']})
        chosen = selected[name]['selected']
        test = merged[merged.model == LABELS[name]].set_index('hours')
        best = chosen['development_mse']
        last = logs[name].dropna(subset=['validation_mse_mm_h_squared']).iloc[-1]
        rows.append({'Model': LABELS[name], 'Selected step': chosen['best_step'],
            'Parameters': bundles[name]['manifest']['parameters'],
            'Best validation rate MSE': best, 'Final-step validation MSE': last.validation_mse_mm_h_squared,
            'Later deterioration (%)': 100 * (last.validation_mse_mm_h_squared / best - 1),
            '1h test MSE': test.loc[1, 'mse'], '3h test MSE': test.loc[3, 'mse'],
            '1h test MAE': test.loc[1, 'mae_mm'], '3h test MAE': test.loc[3, 'mae_mm']})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'model_comparison.csv', index=False)
    pd.DataFrame(all_logs).to_csv(OUT / 'training_histories.csv', index=False)
    pd.DataFrame(candidates).to_csv(OUT / 'recipe_comparison.csv', index=False)
    assessments = {'convlstm':'Best score at cutoff; plateau not established',
        'dgmr':'Retain update 1350; final validation worsened',
        'nowcastnet':'Retain update 750; later validation deteriorated'}
    end_table = result[['Model','Selected step','Best validation rate MSE','Final-step validation MSE','Later deterioration (%)']].copy()
    end_table['Assessment'] = [assessments[n] for n in NAMES]
    end_table = end_table.rename(columns={'Selected step':'Best update','Final-step validation MSE':'Final score at update 1500'})
    end_table.to_csv(OUT / 'end_of_training_scores.csv', index=False)
    batch = pd.read_csv(OUT / 'batch_difficulty.csv')
    batch_tail = batch[(batch.model_id=='convlstm') & batch.step.isin([1350,1500])].copy()
    batch_table = batch_tail[['step','case_id_1','case_id_2','target_mean_rate_mm_h',
        'target_dry_baseline_mse_mm_h_squared','target_wet_fraction_above_0_1_mm_h']].rename(columns={
            'step':'Update','case_id_1':'First case','case_id_2':'Second case',
            'target_mean_rate_mm_h':'Mean target rate (mm/h)',
            'target_dry_baseline_mse_mm_h_squared':'Zero-forecast batch MSE (mm/h)²',
            'target_wet_fraction_above_0_1_mm_h':'Wet pixel fraction'})
    b1350,b1500 = batch_tail.set_index('step').loc[1350],batch_tail.set_index('step').loc[1500]
    difficulty_ratio = b1500.target_dry_baseline_mse_mm_h_squared/b1350.target_dry_baseline_mse_mm_h_squared
    fig, ax = plt.subplots(figsize=(11, 3.7), layout='constrained')
    for i,n in enumerate(NAMES):
        best = result.iloc[i]['Best validation rate MSE']
        last = result.iloc[i]['Final-step validation MSE']
        ax.plot([best,last],[i,i], color=COLORS[n], lw=3, alpha=.5)
        ax.scatter(last,i,s=80,marker='s',facecolor='white',edgecolor=COLORS[n],linewidth=2,zorder=4,
                   label='Final update 1500' if i==1 else None)
        ax.scatter(best,i,s=190,marker='*',color=COLORS[n],zorder=5,
                   label='Best validation checkpoint' if i==1 else None)
        ax.annotate(f"Best: {best:.5f} (update {int(result.iloc[i]['Selected step'])})",(best,i),
                    xytext=(0,14),textcoords='offset points',ha='center',fontsize=9)
        ax.annotate(f"Final: {last:.5f} · +{result.iloc[i]['Later deterioration (%)']:.1f}%",(last,i),
                    xytext=(0,-22),textcoords='offset points',ha='center',fontsize=9)
    ax.set_yticks(range(3),[LABELS[n] for n in NAMES]); ax.invert_yaxis()
    ax.set(xlabel='Fixed-set validation rate MSE · (mm/hour)² · lower is better',
           xlim=(.066,.092),ylim=(2.55,-.55),title='End-of-training score: best checkpoint versus final update')
    clean(ax); ax.legend(loc='lower center',bbox_to_anchor=(.5,-.37),ncol=2,frameon=False,fontsize=9)
    save(fig,'end_of_training')

    # 1. Main learning curve: the same validation metric, cases and seeds.
    fig, ax = plt.subplots(figsize=(11.5, 5.5), layout='constrained')
    for n in NAMES:
        d = logs[n].dropna(subset=['validation_mse_mm_h_squared'])
        ax.plot(d.step, d.validation_mse_mm_h_squared, marker='o', lw=2, color=COLORS[n], label=LABELS[n])
        s = selected[n]['selected']
        ax.scatter([s['best_step']], [s['development_mse']], marker='*', s=220, color=COLORS[n], edgecolors='white', zorder=5)
        ax.annotate(f"Selected: {s['best_step']}", (s['best_step'], s['development_mse']),
                    xytext=((8, 17) if n == 'nowcastnet' else (-65, -22 if n != 'dgmr' else 17)),
                    textcoords='offset points', color=COLORS[n], fontsize=9,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':1})
    baselines = pd.read_csv(BASE / 'development_baselines.csv').drop_duplicates('model').set_index('model')
    for name, color, style in [('Always dry', '#859094', ':'), ('Persistence', '#306963', '--')]:
        ax.axhline(baselines.loc[name, 'selection_rate_mse_mm_h_squared'], color=color, ls=style, lw=1.3, label=name)
    ax.set(xlabel='Optimizer update (not epoch)', ylabel='Validation rate MSE · (mm/hour)²',
           title='Same validation error measure for every architecture', xlim=(100, 1550))
    ax.set_yscale('log'); ax.set_yticks([.07,.08,.1,.15,.2,.3], ['0.07','0.08','0.10','0.15','0.20','0.30']); ax.minorticks_off(); clean(ax)
    ax.legend(ncol=3, frameon=False, loc='upper right', fontsize=9)
    fig.get_layout_engine().set(rect=(0, .08, 1, .92))
    fig.text(.08, .025, '42 fixed development cases · two-member ensemble mean · stars mark retained checkpoints\nNowcastNet has no validation scores before update 600; missing history is not filled in.', fontsize=9)
    save(fig, 'validation_history')

    # 2. Training objectives: independent y-scales and honest stage boundaries.
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
    for ax, n in zip(axs, NAMES):
        d = logs[n]
        for phase in d.phase.unique():
            part = d[d.phase == phase]
            ax.plot(part.step, part.loss, 'o-', color=COLORS[n], alpha=1 if phase == 'generator' else .5,
                    lw=1.5, label='Main objective' if n != 'nowcastnet' else phase.title() + ' objective')
        if n != 'convlstm':
            d = d[d.phase == 'generator']
            ax.plot(d.step, d.discriminator_loss, '.--', color='#879398', label='Discriminator objective')
        if n == 'nowcastnet':
            ax.axvline(500, color='#73858a', ls=':', lw=1)
            ax.text(.04, .9, 'Evolution → generator\nafter update 500', transform=ax.transAxes, fontsize=8)
        ax.axvline(selected[n]['selected']['best_step'], color=COLORS[n], alpha=.35, ls=':')
        ax.set(title=LABELS[n], xlabel='Optimizer update', ylabel='Logged minibatch objective · model-specific')
        clean(ax); ax.legend(fontsize=8, frameon=False, loc='upper right')
    fig.suptitle('Training history: different objectives, separate scales', fontsize=16, fontweight='bold')
    fig.get_layout_engine().set(rect=(0, .1, 1, .90))
    fig.text(.04, .02, 'Each point is one sampled minibatch of two cases, not an epoch average. No smoothing or missing points added.\nLoss magnitude cannot rank architectures: adversarial terms, reconstruction weights and NowcastNet stages differ.', fontsize=9)
    save(fig, 'training_objectives')

    # 3. Comparable split losses at selected weights, recomputed without fitting.
    split_order = ['train', 'selection', 'validation', 'test']
    split_aliases = {'training':'train', 'dev_selection':'selection', 'validation_selection':'selection',
        'selection_dev':'selection', 'development_selection':'selection', 'development':'validation',
        'full_validation':'validation', 'all_validation':'validation', 'dev_full':'validation',
        'full_development':'validation', 'development_full':'validation', 'development_all':'validation'}
    common['split'] = common['split'].replace(split_aliases)
    assert set(split_order) <= set(common.split), list(common.split.unique())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for ax, metric, ylabel in zip(axs, ['mse', 'mae'], ['Rate MSE · (mm/hour)²', 'Rate MAE · mm/hour']):
        for j, n in enumerate(NAMES):
            d = common[common.model_id == n].set_index('split').loc[split_order]
            ax.bar(np.arange(4)+(j-1)*.23, d[metric], width=.21, color=COLORS[n], label=LABELS[n])
        ax.set_xticks(range(4), ['Training\n352 cases', 'Selection dev\n42 cases', 'All dev\n167 cases', 'Held-out test\n180 cases'])
        ax.set_ylabel(ylabel); clean(ax)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Selected checkpoints: same physical loss on each split', fontsize=16, fontweight='bold')
    fig.get_layout_engine().set(rect=(0, .1, 1, .90))
    fig.text(.055, .025, 'New inference-only diagnostic: natural case frequency, same quality mask, two ensemble members and fixed seed.\nThe 42 selection cases are a subset of all development cases. These bars are checkpoints, not learning curves.', fontsize=9)
    save(fig, 'checkpoint_split_losses')

    # 4. Truly held-out historical performance, alongside traditional methods.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    order = [('Persistence','#a8b4ba'), ('Motion extrapolation','#398c82'), ('HRRR','#354a50')]
    order += [(LABELS[n], COLORS[n]) for n in NAMES]
    for ax, metric, ylabel in zip(axs, ['mse', 'mae_mm'], ['Accumulation MSE · mm²', 'Accumulation MAE · mm']):
        for label, color in order:
            d = merged[merged.model == label].sort_values('hours')
            ax.plot(d.hours, d[metric], marker='o', label=label, color=color, lw=2 if label in LABELS.values() else 1.3,
                    ls='-' if label in LABELS.values() else '--')
        ax.set(xlabel='Forecast accumulation horizon · hours', ylabel=ylabel, xticks=[1,2,3]); clean(ax)
    fig.suptitle('Final test errors: 180 unseen dates in winter 2023/24', fontsize=16, fontweight='bold')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.get_layout_engine().set(rect=(0, .16, 1, .84))
    save(fig, 'heldout_errors')

    # 5. Calendar-time test performance is not training-step test performance.
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, layout='constrained')
    for ax, horizon in zip(axs, [1, 3]):
        for n in NAMES:
            d = case[(case.model == LABELS[n]) & (case.hours == horizon)].set_index('date').sort_index()
            ax.plot(d.index, d.mse.rolling('7D', min_periods=3).mean(), label=LABELS[n], color=COLORS[n], lw=1.6)
        ax.set(ylabel=f'{horizon}h accumulation MSE · mm²'); clean(ax)
    axs[0].legend(frameon=False, ncol=3, fontsize=9)
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator()); axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axs[-1].set_xlabel('Forecast date · held-out winter')
    fig.suptitle('When did the models struggle? Test errors through the winter', fontsize=16, fontweight='bold')
    fig.get_layout_engine().set(rect=(0, .07, 1, .93))
    fig.text(.055, .018, 'Trailing 7-calendar-day mean of per-case regional MSE; at least three available days.\nWeights stay fixed throughout. This is a test-period timeline, not a test learning curve.', fontsize=9)
    save(fig, 'test_error_history')

    # 6. Retained development-only trials, including failed / stopped candidates.
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    styles = [('uniform_l1','#a0a9ae'), ('wet_mse','#3a89ab'), ('wet_mse_linear_head','#bf792f')]
    for ax, n in zip(axs, NAMES):
        for recipe, color in styles:
            path = BASE / f'models/recipe_{recipe}/{n}.pt.training.json'
            if not path.exists(): continue
            d = pd.DataFrame(json.loads(path.read_text())).dropna(subset=['validation_mse_mm_h_squared'])
            ax.plot(d.step, d.validation_mse_mm_h_squared, 'o-', color=color, lw=1.4, label=RECIPES[recipe])
        ax.axhline(baselines.loc['Always dry','selection_rate_mse_mm_h_squared'], color='#74878a', ls=':', label='Always dry')
        ax.set(title=LABELS[n], xlabel='Optimizer update', ylabel='Validation rate MSE · (mm/hour)²'); ax.set_yscale('log'); clean(ax)
    legend = {}
    for ax in axs:
        for h,l in zip(*ax.get_legend_handles_labels()): legend[l] = h
    fig.legend(legend.values(), legend.keys(), loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig.suptitle('All saved candidate histories, including unsuccessful attempts', fontsize=16, fontweight='bold')
    fig.get_layout_engine().set(rect=(0, .17, 1, .83))
    save(fig, 'recipe_histories')

    # Existing paired uncertainty, not error bars fabricated from one training run.
    paired = pd.read_csv(BASE / 'short_paired_intervals.csv')
    paired = paired[(paired.model.isin(LABELS.values())) & paired.hours.isin([1,3]) & (paired.metric=='mae')]
    ptable = paired[['model','hours','difference','ci95_low','ci95_high']].rename(columns={
        'model':'Model','hours':'Horizon (h)','difference':'MAE minus HRRR (mm)','ci95_low':'95% lower','ci95_high':'95% upper'})
    ptable.to_csv(OUT / 'paired_test_uncertainty.csv', index=False)
    common.to_csv(OUT / 'checkpoint_losses_display.csv', index=False)
    common_table = common[['model','split','cases','mse','mae']].rename(columns={
        'model':'Model','split':'Dataset','cases':'Cases','mse':'Rate MSE (mm/h)²','mae':'Rate MAE (mm/h)'})
    common_table['Dataset'] = common_table.Dataset.replace({'train':'Training','selection':'Selection development',
        'validation':'All development','test':'Held-out test'})

    timestamp = datetime.now(ZoneInfo('America/New_York')).strftime('%B %d, %Y · %I:%M %p %Z')
    overview = result[['Model','Selected step','Best validation rate MSE','1h test MSE','3h test MSE','1h test MAE','3h test MAE']]
    methodology = '''<ol><li><strong>Time-separated cases.</strong> Training: winters 2020/21–2021/22 (352 eligible dates). Development: 2022/23 (167 dates); 42 fixed every-fourth dates choose checkpoints and recipes. Held-out test: 2023/24 (180 dates). No random mixing of adjacent windows across splits.</li>
    <li><strong>Small engineering prototypes.</strong> Nine input radar frames at 10-minute spacing; 20 predicted rate frames on a pooled 64×64 regional grid. Each selected candidate ran 1,500 optimizer updates, minibatch size 2, learning rate 0.0003, base channels 8, one seed. The training data comprise roughly 80 minutes of context per case, not full continuous all-hour multi-year training.</li>
    <li><strong>Different training objectives.</strong> ConvLSTM optimizes a physical/log reconstruction objective; DGMR combines weighted reconstruction and an adversarial objective; NowcastNet trains evolution first and then a stochastic generator with reconstruction, pooling and adversarial terms. Wet cases are oversampled during training; evaluation preserves natural case frequency.</li>
    <li><strong>Checkpoint diagnostic.</strong> The selected weights were evaluated again without training, using two ensemble members, fixed seed 98123, the original target-quality mask, and pixel-weighted errors on all 20 rate frames. This reproduces the saved selection losses. Training versus test differences can reflect changing weather distributions as well as model generalization; different winters are not identical draws.</li>
    <li><strong>Two evaluation targets.</strong> Rate MSE uses (mm/hour)². Frozen short-horizon testing uses integrated precipitation, regional interior masks, eight ensemble members, and equal averaging of case errors; accumulation MSE uses mm². They answer different questions and must not share a numeric ranking axis.</li>
    <li><strong>What is missing.</strong> Intermediate checkpoint weights and test losses by optimizer update were not retained. There is no recoverable per-step test curve and no multi-seed confidence band for learning curves. Missing NowcastNet validation history before update 600 is intentional.</li>
    <li><strong>Limits of the result.</strong> One held-out winter and morning issue times cannot establish across-winter, all-hour or other-mountain performance. Seven-day block-bootstrap intervals quantify within-winter uncertainty only. These are compact local adaptations, not reproductions of the full published systems. No historical Open-Meteo benchmark is included.</li></ol>'''
    findings = '''<div class="findings"><article><span>Validation winner</span><h3>NowcastNet adaptation</h3><p>Best rate MSE: <strong>0.06990</strong> at update 750. Its final-step validation loss is 23.4% higher; the earlier checkpoint was retained.</p></article>
    <article><span>Test MAE winner</span><h3>DGMR adaptation</h3><p>Regional accumulation MAE: <strong>0.06396 mm at 1h</strong> and <strong>0.18804 mm at 3h</strong>. This later test-based display choice differs from checkpoint selection.</p></article>
    <article><span>Interpretation</span><h3>Training loss alone cannot rank models</h3><p>Objectives differ and each saved training point is one minibatch. Compare common validation loss, selected-checkpoint split losses, and independent test errors.</p></article></div>'''
    files = ['training_histories.csv','end_of_training_scores.csv','batch_difficulty.csv','batch_difficulty_audit.json',
        'model_comparison.csv','checkpoint_losses.csv','checkpoint_loss_audit.json','recipe_comparison.csv','paired_test_uncertainty.csv']
    downloads = ' · '.join(f'<a href="{f}">{f.replace("_"," ")}</a>' for f in files)
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alta · Training and testing report</title>
    <style>:root{{--ink:#18383e;--muted:#587276;--teal:#14656a;--bg:#f2f5f1}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}}main{{max-width:1180px;margin:auto;padding:48px 28px 80px}}a{{color:var(--teal);text-underline-offset:3px}}.eyebrow{{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:var(--teal);font-weight:700}}h1{{font-size:clamp(32px,5vw,52px);line-height:1.1;margin:14px 0}}h2{{font-size:27px;line-height:1.25}}h3{{font-size:18px;margin:8px 0}}.lede{{font-size:19px;max-width:900px}}.meta,figcaption,.small{{font-size:13px;color:var(--muted)}}nav{{display:flex;gap:18px;flex-wrap:wrap;margin:25px 0}}.notice{{background:#fff4df;border-left:4px solid #bd843c;padding:15px 20px;border-radius:8px}}.findings{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}}.findings article,section{{background:white;border:1px solid #dae4df;border-radius:15px;padding:24px}}.findings span{{font-size:11px;text-transform:uppercase;letter-spacing:1.3px;color:var(--muted)}}.findings p{{font-size:14px}}section{{margin:22px 0;scroll-margin-top:20px}}figure{{margin:22px 0}}img{{width:100%;height:auto}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:12px;text-align:right;border-bottom:1px solid #dfe7e3}}th:first-child,td:first-child{{text-align:left}}th{{background:#f0f5f2;color:#36575b}}li{{margin:12px 0}}details{{padding:14px 0}}summary{{cursor:pointer;font-weight:650}}.split{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:22px 0}}.split div{{background:#eaf2ee;border-radius:9px;padding:14px;font-size:13px}}.split strong{{display:block;font-size:18px}}.downloads{{font-size:13px;overflow-wrap:anywhere}}@media(max-width:720px){{main{{padding:26px 14px}}.findings,.split{{grid-template-columns:1fr}}section{{padding:18px}}}}@media print{{body{{background:white}}main{{padding:0}}nav{{display:none}}section{{break-inside:avoid}}details{{display:block}}}}</style></head><body><main>
    <div class="eyebrow">Snowline research · Alta</div><h1>Training progress.<br>Testing performance.</h1><p class="lede">A comparison of ConvLSTM, DGMR and NowcastNet adaptations, with controlled learning-rate experiments, observed accumulation curves, and the original pilot audit.</p>
    <p class="meta">Generated {timestamp} · Research checkpoints · Liquid precipitation, not fresh-snow depth</p>
    <nav><a href="#tuning">New training experiment</a><a href="#validation">Original validation curves</a><a href="#end-review">Original end-of-run score</a><a href="#training">Training loss</a><a href="#checkpoints">Train / validation / test</a><a href="#heldout">Original test errors</a><a href="#history">Observed accumulation</a><a href="#radar-transfer">Other mountains</a><a href="#methods">Methods & files</a></nav>
    {new_training}<div class="notice"><strong>Original pilot record below.</strong> Sections 1–4 and 6 preserve the initial 1,500-update experiments. Validation selected checkpoints during training; the test winter was evaluated afterward. The new experiment above reuses that historical benchmark only after its choices are frozen.</div>{findings}
    <div class="split"><div><strong>352 training dates</strong>Winters 2020/21 and 2021/22</div><div><strong>167 development dates</strong>Winter 2022/23</div><div><strong>42 selection dates</strong>Fixed subset of development</div><div><strong>180 test dates</strong>Winter 2023/24</div></div>
    <section id="validation"><div class="eyebrow">01 · During training</div><h2>Compare the same validation loss</h2><p>Every curve uses the same 42 cases and physical rate MSE. Stars identify the weights retained for final testing. No test observations were used to choose these checkpoints.</p><figure>{image('validation_history','Validation MSE by optimizer update, with selected checkpoints and baseline scores')}<figcaption>The logarithmic y-axis exposes both early instability and late improvements. Logged values are connected only as a visual guide.</figcaption></figure></section>
    <section id="end-review"><div class="eyebrow">End-of-run assessment</div><h2>Are 1,500 updates sufficient?</h2><p><strong>Sufficient for an initial pilot; not evidence of convergence or an optimal training budget.</strong> These are 1,500 optimizer updates, not epochs. With batches of two and 352 training cases, that is 3,000 case draws, or about <strong>8.52 nominal dataset passes</strong>. Weighted sampling visited 340 unique cases; 12 were never drawn. Of the 3,000 draws, 2,242 came from the 56 wet cases.</p>
    <p><strong>The final score below is validation MSE at update 1,500</strong>, evaluated on the same fixed 42 cases. It is a consistent measure of forecasting error, unlike the last minibatch objective. Lower is better.</p>{image('end_of_training','Best and final validation scores for each model, with the selected update and percentage deterioration')}
    <div class="table-wrap">{table(end_table,{'Best validation rate MSE':lambda x:f'{x:.5f}','Final score at update 1500':lambda x:f'{x:.5f}','Later deterioration (%)':lambda x:f'{x:+.1f}%'})}</div>
    <p class="small">ConvLSTM reaches its best logged validation score at the stopping boundary, so a longer controlled run could still help. DGMR and NowcastNet finish worse than their selected earlier checkpoints. More updates at the same learning rate are not automatically an improvement. Final-update weights were not retained separately for the latter two; their published test scores use the selected checkpoints, not update 1,500.</p>
    <h3>Why do all three training losses spike at the end?</h3><p>Reconstructing the saved random sampler against the archived training code shows that all three models saw <strong>the same last minibatch</strong>. Its squared target precipitation magnitude is <strong>{difficulty_ratio:.2f}×</strong> that of the batch at update 1,350. That much wetter batch is a strong shared explanation for synchronized raw-loss spikes; it does not isolate the contribution of changing weights or adversarial states.</p>
    <div class="table-wrap">{table(batch_table,{'Mean target rate (mm/h)':lambda x:f'{x:.5f}','Zero-forecast batch MSE (mm/h)²':lambda x:f'{x:.5f}','Wet pixel fraction':lambda x:f'{100*x:.2f}%'})}</div>
    <p class="small">Zero-forecast batch MSE is mean squared observed rate over valid target pixels, not the trained model's loss. The dates and sampler reconstruction are in the downloadable audit. Adam's learning rate stayed at 0.0003; there was no special learning-rate change at the final update. From update 1,350 to 1,500, validation MSE improves 1.88% for ConvLSTM and 2.58% for NowcastNet, while DGMR worsens 12.85%.</p>
    <h3>A better stopping decision</h3><p>Use a fixed, naturally sampled training probe and the fixed validation set every 50–100 updates, plus interval-average minibatch losses. Reduce the learning rate when validation stalls (for example, with <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html">PyTorch’s validation-driven scheduler</a>), retain the best checkpoint, and stop after sustained non-improvement. This was the original recommendation. The new experiment at the top of this report now tests a 3,000-update ceiling, three initial learning rates and repeated seeds with adaptive scheduling and early stopping. Use a new untouched winter to confirm improvements after further development.</p></section>
    <section id="training"><div class="eyebrow">02 · Optimization behavior</div><h2>Training loss is useful within each model</h2><p>These are sparse, unsmoothed minibatch losses. A spike can come from a wetter sampled batch or the adversarial objective; it is not automatically evidence of worse forecasting.</p>{image('training_objectives','Separate raw training-objective histories for each model, with NowcastNet phase boundary')}<p class="small">Do not compare the heights between panels. The plotted loss functions and their scales differ, and the NowcastNet objective changes after update 500.</p></section>
    <section id="checkpoints"><div class="eyebrow">03 · Direct split comparison</div><h2>Evaluate selected weights with one loss definition</h2><p>To make a fair training-versus-testing comparison, the retained weights were reevaluated without fitting. Both rate MSE and rate MAE use the same mask, units, ensemble size and random seed across splits.</p>{image('checkpoint_split_losses','Comparable selected-checkpoint rate MSE and MAE on training, selection development, all development, and held-out test')}<details><summary>View the checkpoint-loss values</summary><div class="table-wrap">{table(common_table,{'Rate MSE (mm/h)²':lambda x:f'{x:.6f}','Rate MAE (mm/h)':lambda x:f'{x:.6f}'})}</div></details><p class="small">A training–test gap is descriptive, not a causal diagnosis of overfitting: the splits contain different winters and storm distributions. The selection subset was already used for model choice.</p></section>
    <section id="heldout"><div class="eyebrow">04 · Forecasts on an unseen winter</div><h2>Test errors at 1, 2 and 3 hours</h2><p>The frozen benchmark integrates predicted precipitation and compares the same 180 dates. Traditional persistence, motion extrapolation and HRRR are included. Lower values indicate smaller errors.</p>{image('heldout_errors','Final held-out accumulation MSE and MAE by horizon for neural models and traditional baselines')}<div class="table-wrap">{table(overview,{c:lambda x:f'{x:.5f}' for c in overview.select_dtypes('number').columns if c!='Selected step'})}</div><p class="small">Validation rate MSE: (mm/hour)². Test accumulation MSE: mm². Test accumulation MAE: mm. These columns have different targets and cannot be compared directly.</p><details><summary>How uncertain are improvements over HRRR?</summary><p>Paired seven-day block-bootstrap 95% intervals within the single test winter. A negative difference favors the neural prototype. These intervals do not quantify uncertainty across winters or across retraining seeds.</p><div class="table-wrap">{table(ptable,{c:lambda x:f'{x:.5f}' for c in ['MAE minus HRRR (mm)','95% lower','95% upper']})}</div></details></section>
    <section id="history"><div class="eyebrow">05 · Observations and historical forecasts</div><h2>Actual accumulation and forecast comparison</h2>{observed_accumulation}<details><summary>Original pilot: how errors varied across the winter</summary><p>All forecasts use the original fixed selected weights. The horizontal axis is calendar date, not training progress.</p>{image('test_error_history','Trailing seven-day test MSE through the held-out winter for 1h and 3h predictions')}</details></section>
    <section id="recipes"><div class="eyebrow">06 · Complete experiment record</div><h2>Keep unsuccessful candidates visible</h2><p>Initial ConvLSTM and DGMR candidates approached the always-dry development score. Wet-case sampling and a revised reconstruction objective were tested on development data. A later DGMR head correction was also selected there. Stopped runs end where their logs end.</p>{image('recipe_histories','Development histories of all retained training recipes, including stopped candidates')}<details><summary>View all candidate recipes and selected steps</summary><div class="table-wrap">{table(pd.DataFrame(candidates),{'Best validation rate MSE':lambda x:f'{x:.6f}'})}</div></details></section>
    {other_mountains}<section id="methods"><div class="eyebrow">07 · Interpretation and reproducibility</div><h2>Original pilot methods and limitations</h2>{methodology}<p>The new experiment at the top of this report records fixed training-probe and validation loss, averages minibatch objectives, retains best and final optimizer states, and repeats seeds. Its separate frozen plan specifies selection and stopping rules. An additional untouched winter remains necessary for a fresh confirmation after this development cycle.</p>
    <h3>Download the underlying numbers</h3><p class="downloads">{downloads}</p><p class="small">Plots are embedded in this HTML so the report itself opens offline. CSV/JSON files are adjacent downloads. Source logs and frozen evaluation results are preserved unchanged.</p><p><a href="../report.html">Original Alta evaluation</a> · <a href="/">Return to Snowline</a></p></section>
    </main></body></html>'''
    temporary_report = OUT / 'report.html.tmp'
    temporary_report.write_text(body)
    temporary_report.replace(OUT / 'report.html')
    sources = [BASE/'models/selected/recipe_selection.json',BASE/'short_summary.csv',BASE/'short_case_metrics.csv',OUT/'batch_difficulty.csv',OUT/'batch_difficulty_audit.json',
        BASE/'short_paired_intervals.csv',OUT/'checkpoint_losses.csv',OUT/'checkpoint_loss_audit.json']
    sources += [BASE/f'models/selected/{n}.pt.json' for n in NAMES]
    sources += [p for p in [ROOT/'src/report_alta_learning_review.py',ROOT/'src/report_alta_extensions.py',
        BASE/'tuning-20261107/plan.json',BASE/'tuning-20261107/final_selection.json',
        BASE/'tuning-20261107/evaluation_audit.json',BASE/'tuning-20261107/integrity_audit.json',
        ROOT/'outputs/radar-transfer/prepared/verification_audit.json'] if p.exists()]
    manifest = {'generated_at':datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'includes_new_training':bool(new_training), 'new_diagnostic':'original pilot audit plus isolated controlled tuning and observed accumulation comparisons',
        'tuning_selection_frozen':(BASE/'tuning-20261107/final_selection.json').exists(),
        'tuned_test_evaluation_complete':(BASE/'tuning-20261107/evaluation_audit.json').exists(),
        'history_is_test_per_step':False, 'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'figures':[p.name for p in OUT.glob('*.png')]}
    (OUT / 'report_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(OUT / 'report.html')
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
