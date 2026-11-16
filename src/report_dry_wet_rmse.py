"""Partition frozen held-out snowfall errors by observed zero/positive totals.

Reads saved predictions only: no fitting, selection changes, or downloads.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/nwp-weather/forecast-portfolio/dry-wet'
NAMES = {'alta': 'Alta', 'vail': 'Vail', 'mansfield': 'Stowe', 'jay': 'Jay Peak'}
INPUTS = set()


def source(relative):
    path = ROOT / relative
    INPUTS.add(path)
    return path


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads(source('outputs/nwp-weather/forecast-portfolio/deployment_selection.json').read_text())
    rows = []

    def add(site, hours, method, identity, dates, truth, prediction):
        y, p = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
        assert y.shape == p.shape and len(y) == len(dates)
        assert np.isfinite(y).all() and np.isfinite(p).all()
        assert (y >= 0).all() and (p >= 0).all()
        for date, actual, estimate in zip(dates, y, p):
            rows.append(dict(site=site, mountain=NAMES[site], horizon_hours=hours,
                method=method, frozen_method=identity, window_start=str(date),
                observed_inches=actual, predicted_inches=estimate,
                observed_cm=actual*2.54, predicted_cm=estimate*2.54,
                category='dry' if actual == 0 else 'wet'))

    prefix = 'outputs/nwp-weather/forecast-portfolio/'
    frame = pd.read_csv(source(prefix+'hrrr24_predictions.csv'))
    reference24 = pd.read_csv(source(prefix+'hrrr24_metrics.csv'))
    for site in NAMES:
        d = frame[frame.site.eq(site) & frame.split.eq('test')]
        for col, label in [('portfolio', 'Our model'), ('hrrr_native', 'Native HRRR'), ('prior_neural', 'Previous neural')]:
            expected = reference24.query('site == @site and split == "test" and method == @col').iloc[0]
            assert len(d) == expected['n']
            assert np.isclose(np.sqrt(np.mean((d[col]-d.truth)**2)), expected.rmse_inches, rtol=1e-6)
            add(site, 24, label, col, d.start, d.truth, d[col])

    prefix = 'outputs/nwp-weather/neural48-trained/'
    z = np.load(source(prefix+'test_predictions.npz'))
    methods = json.loads(source(prefix+'test_methods.json').read_text())
    meta = pd.read_csv(source('data/neural48_training/prepared/index.csv')).iloc[z['row_indices']].reset_index(drop=True)
    assert meta.split.eq('test').all()
    assert np.array_equal(meta.row.to_numpy(), z['row_indices'])
    reference48 = pd.read_csv(source(prefix+'metrics.csv'))
    for site in NAMES:
        mask = meta.site.eq(site).to_numpy()
        family = selection['sites'][site]['48']['family']
        chosen = {'weather_only': 'Weather TCN', 'weather_radar': 'Radar + weather TCN'}[family]
        for identity, label in [(chosen, 'Our model'),
                                ('JMA provider · fixed older guidance', 'Archived JMA provider'),
                                ('Current model family · refit', 'Previous forest'),
                                ('Historical calendar distribution', 'Climatology')]:
            truth = z['truth'][mask].astype(float).sum(axis=1)
            prediction = z[f'{methods.index(identity)}_mean'][mask].astype(float).sum(axis=1)
            expected = reference48.query('site == @site and split == "test" and horizon_hours == 48 and method == @identity').iloc[0]
            assert len(truth) == expected['n']
            assert np.isclose(np.sqrt(np.mean((prediction-truth)**2)), expected.rmse_inches, rtol=1e-5)
            add(site, 48, label, identity, meta.loc[mask, 'start_utc'], truth, prediction)

    prefix = 'outputs/nwp-weather/alta12-trained/'
    manifest = json.loads(source(prefix+'manifest.json').read_text())
    identity = manifest['selection']['modern_missing_history']['name']
    meta = pd.read_csv(source(prefix+'case_index.csv'))
    ids = meta.loc[meta.modern.astype(bool) & meta.validation_stage.eq('test'), 'row'].to_numpy(int)
    assert np.array_equal(meta.row, np.arange(len(meta)))
    truth = np.load(source('data/alta12_training/arrays.npz'))['snow'][ids]
    prediction = np.load(source(prefix+'predictions_missing_history.npz'))[identity][ids]
    expected = manifest['selection']['modern_missing_history']['test_metrics']
    assert len(ids) == expected['n']
    assert np.isclose(np.sqrt(np.mean((prediction-truth)**2)), expected['rmse_inches'], rtol=1e-5)
    add('alta', 12, 'Our model', identity, meta.iloc[ids].start_utc, truth, prediction)
    climate = np.load(source(prefix+'predictions_all.npz'))['modern_climatology'][ids]
    add('alta', 12, 'Climatology', 'modern_climatology', meta.iloc[ids].start_utc, truth, climate)

    cases = pd.DataFrame(rows)
    assert not cases.duplicated(['site', 'horizon_hours', 'method', 'window_start']).any()
    cases.to_csv(OUT/'heldout_cases.csv', index=False)
    scores = []
    for (site, hours, method), group in cases.groupby(['site', 'horizon_hours', 'method'], sort=False):
        parts = {}
        for category in ['all', 'dry', 'wet']:
            d = group if category == 'all' else group[group.category.eq(category)]
            error = (d.predicted_cm-d.observed_cm).to_numpy()
            parts[category] = (len(d), float(np.sum(error**2)))
            scores.append(dict(site=site, mountain=NAMES[site], horizon_hours=hours, method=method,
                category=category, n=len(d), rmse_cm=float(np.sqrt(np.mean(error**2))) if len(d) else np.nan,
                mae_cm=float(np.mean(np.abs(error))) if len(d) else np.nan,
                bias_cm=float(np.mean(error)) if len(d) else np.nan,
                observed_mean_cm=float(d.observed_cm.mean()), predicted_mean_cm=float(d.predicted_cm.mean())))
        assert parts['dry'][0] + parts['wet'][0] == parts['all'][0]
        assert np.isclose(parts['dry'][1] + parts['wet'][1], parts['all'][1])
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT/'metrics.csv', index=False)
    selected = scores[scores.method.eq('Our model')]
    summary = selected.pivot(index=['site', 'horizon_hours'], columns='category', values=['rmse_cm', 'n'])
    summary.columns = [f'{category}_{metric}' for metric, category in summary.columns]
    summary = summary.reset_index().sort_values(['horizon_hours', 'site'], key=lambda s: s.map({k:i for i,k in enumerate(NAMES)}) if s.name == 'site' else s)
    summary['mountain'] = summary.site.map(NAMES)
    summary.to_csv(OUT/'selected_summary.csv', index=False)

    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(11.5, 8.2), layout='constrained')
    y = np.arange(len(summary))
    for category, offset, color in [('dry', -.18, '#157c87'), ('wet', .18, '#c17a35')]:
        bars = ax.barh(y+offset, summary[f'{category}_rmse_cm'], height=.32, color=color, label=category.capitalize())
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in summary[f'{category}_rmse_cm']], padding=4, fontsize=10)
    ax.set_yticks(y, [f"{r.mountain} · {r.horizon_hours}h\n{int(r.dry_n)} dry / {int(r.wet_n)} wet" for r in summary.itertuples()])
    ax.invert_yaxis(); ax.set_xlabel('RMSE (cm)'); ax.set_xlim(0, 16)
    ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    for boundary in [.5, 4.5]: ax.axhline(boundary, color='#c9d1d0', linewidth=.7)
    ax.legend(loc='lower right', frameon=False)
    fig.suptitle('Our model: errors on dry versus snowy periods', fontsize=18, weight='bold')
    fig.supxlabel('Dry = observed accumulation exactly 0; wet = observed accumulation > 0. Frozen 2023/24 test predictions.\n48-hour targets overlap. Vail is a sensor proxy; Stowe and Jay Peak use nearby town observations.', fontsize=10)
    fig.savefig(OUT/'dry_wet_rmse.png', dpi=170); plt.close(fig)
    display = summary[['mountain','horizon_hours','all_rmse_cm','dry_rmse_cm','wet_rmse_cm','dry_n','wet_n']].copy()
    display.columns = ['Mountain','Hours','All RMSE (cm)','Dry RMSE (cm)','Wet RMSE (cm)','Dry n','Wet n']
    display[['Dry n','Wet n']] = display[['Dry n','Wet n']].astype(int)
    audit = dict(generated_at=datetime.now(timezone.utc).isoformat(), split='2023/24 test winter',
        source_units='inches of snowfall; every prediction and observation multiplied by 2.54 before scoring',
        dry_definition='Observed window total == 0 exactly; recorded zeros may include trace/rounding.',
        wet_definition='Observed window total > 0; no threshold based on the forecast.',
        formula='RMSE = sqrt(mean((prediction - observation)^2)) within each observed category',
        changes_to_models=False, matched_comparators=True,
        checks=['Reproduced saved overall RMSE and counts','Finite nonnegative predictions and observations','Unique method/site/window rows','Dry/wet counts and squared errors partition overall score'],
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)})
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2))
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Snowline · Dry and wet errors</title>
<style>body{{background:#f2f5f2;color:#193c43;font:16px/1.6 system-ui;margin:0}}main{{max-width:1150px;padding:30px;margin:auto}}section{{background:white;padding:24px;border-radius:15px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;text-align:right;border-bottom:1px solid #ddd}}th:first-child,td:first-child{{text-align:left}}img{{width:100%}}a{{color:#087a84}}.scroll{{overflow:auto}}</style><main><a href="../report.html">← Forecast comparison</a><h1>Dry periods versus snowy periods</h1>
<p>Saved held-out predictions from the deployed selections, with errors reported in centimetres. No models were retrained or selected using this breakdown.</p><p><a href="comparison.html">Compare with HRRR and the archived weather provider on matching dry and snowy periods →</a></p>
<section><h2>Our model</h2><p>Dry means the observed total is exactly zero; wet means it is greater than zero. A 48-hour window is dry only when both daily observations are zero. Recorded zeros can include trace or rounded snowfall.</p><div class="scroll">{display.to_html(index=False,border=0,float_format=lambda x:f'{x:.2f}')}</div><img src="dry_wet_rmse.png" alt="Dry and wet RMSE for current snowfall models"></section>
<section><h2>How to interpret the split</h2><p>Dry-period RMSE measures false snowfall amounts. Wet-period RMSE measures errors on periods when snow was observed. Snowy periods have larger errors at every listed target. The overall RMSE is the square root of the sample-weighted dry and wet squared RMSEs, not their simple average.</p>
<p>Previous scorecards use inches: Alta12h 1.47in is 3.73cm. Jay Peak's current48h radar model differs from the weather-only model shown in the earlier neural report; this table follows the deployed selection.</p>
<p>These are fixed reporting-window test scores, not a retrospective verification of the rolling widget forecasts. Adjacent48h windows overlap and samples within storms are dependent. The24h benchmark uses a smaller matched HRRR calendar subset. Differences across horizons are not controlled experiments on the same dates.</p>
<p>Vail's Sno24 sensor/reset interval remains unverified. Stowe and Jay Peak use lower-elevation town observers, not summit snowfall. Verified6h and other mountains'12h fresh-snow test labels are unavailable. The1–3h radar models predict liquid precipitation and are excluded from this snowfall table.</p></section>
<section><h2>Matched comparisons and evidence</h2><p>The detailed scorecard includes HRRR/previous neural for24h, the archived JMA provider/previous forest/climatology for48h, and climatology forAlta12h, each on the same dates as its selected model. The archived JMA benchmark is older issued guidance, not the live Open-Meteo forecast.</p><div class="scroll">{scores.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')}</div><p><a href="selected_summary.csv">Selected-model table</a> · <a href="metrics.csv">All metrics</a> · <a href="heldout_cases.csv">Per-case predictions</a> · <a href="audit.json">Source audit</a></p></section></main></html>'''
    (OUT/'report.html').write_text(body)
    print(display.to_string(index=False,float_format=lambda x:f'{x:.3f}'))


if __name__ == '__main__':
    build()
