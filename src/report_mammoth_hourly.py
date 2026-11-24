"""Build the Mammoth held-out forecast report from saved predictions."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 3, 6, 12, 24, 48)


def paired_scores(rows, baseline):
    """Compare forecasts only where every hourly target and both predictions exist."""
    def rmse(errors):
        return np.sqrt(np.mean(np.square(errors))) if len(errors) else np.nan

    results = []
    for h in HORIZONS:
        block = rows[rows.hour <= h].groupby('issue_utc')
        totals = block[['observed_mm', 'model_mm', baseline]].sum(min_count=h).dropna()
        wet = totals.observed_mm > 0
        model = totals.model_mm - totals.observed_mm
        other = totals[baseline] - totals.observed_mm
        results.append([f'{h}h', len(totals), int(wet.sum()), rmse(model), rmse(other),
                        rmse(model[wet]), rmse(other[wet]), rmse(model[~wet]), rmse(other[~wet])])
    return pd.DataFrame(results, columns=['Horizon','Windows','Wet','Our model','Reference',
                                         'Our wet','Ref wet','Our dry','Ref dry'])


def table(frame):
    return frame.to_html(index=False, border=0, na_rep='—',
                         float_format=lambda v: f'{v:,.2f}')


def minute_check(rows, root, output):
    ground = pd.read_csv(root/'data/mammoth/research/cues_disdrometer_hourly.csv.gz')
    partial = root/'data/mammoth/research/cues_partial_hour_subtotals.csv.gz'
    if not partial.is_file():
        return ''
    ground = ground.merge(pd.read_csv(partial),on='start_utc',how='left',validate='one_to_one')
    ground['observed_minutes_swe_mm'] = ground.snow_swe_mm.combine_first(ground.observed_minutes_swe_mm)
    ground['start_utc'] = pd.to_datetime(ground.start_utc, utc=True)
    subset = rows.copy()
    subset['start_utc'] = pd.to_datetime(subset.start_utc, utc=True)
    subset = subset.merge(ground[['start_utc','snow_valid_minutes','snow_rate_at_cap_minutes',
                                  'observed_minutes_swe_mm']],on='start_utc',validate='many_to_one')
    results = []
    for h in (24,48):
        grouped = subset[subset.hour <= h].groupby('issue_utc')
        totals = grouped[['snow_valid_minutes', 'snow_rate_at_cap_minutes', 'observed_minutes_swe_mm']].sum()
        # At most one missing minute in any included hour. No extrapolated station totals.
        eligible = totals[(grouped.snow_valid_minutes.min() >= 59) &
                          (grouped.size()==h) & (totals.snow_rate_at_cap_minutes==0)]
        w = subset[(subset.hour <= h) & subset.issue_utc.isin(eligible.index)].copy()
        for ref,name in [('hrrr_mm','Native HRRR'),('ecmwf_mm','ECMWF'),('hrrr_tail_baseline_mm','HRRR + tail')]:
            if ref not in w:
                continue
            # An hourly forecast is constant on its native interval for this minute-mask comparison.
            w['forecast'] = w.model_mm * w.snow_valid_minutes / 60.
            w['baseline'] = w[ref] * w.snow_valid_minutes / 60.
            amounts = w.groupby('issue_utc')[['forecast','baseline','observed_minutes_swe_mm']].sum(min_count=h).dropna()
            if amounts.empty:
                continue
            observed = amounts.observed_minutes_swe_mm
            wet = observed > 0
            our, other = amounts.forecast-observed,amounts.baseline-observed
            def rmse(x):
                return float(np.sqrt(np.mean(x**2))) if len(x) else np.nan
            results.append([f'{h}h',name,len(amounts),int(wet.sum()),rmse(our),rmse(other),rmse(our[wet]),rmse(other[wet])])
    scored = pd.DataFrame(results, columns=['Horizon','Reference','Windows','Wet','Our RMSE','Ref RMSE','Our wet','Ref wet'])
    scored.to_csv(output/'minute_coverage.csv',index=False)
    return ('<h2>Long windows with a missing sensor minute</h2><p>During 2024/25 the sensor regularly missed one '
            'minute at 12 UTC. Here every hour has at least 59 measured minutes. We sum only measured minutes, '
            'and weight each issued hourly prediction by its observed duration. This assumes a uniform forecast '
            'rate within each hour. Capped hours are excluded. These scores compare observed-time subtotals in '
            'mm SWE. Each row uses identical windows for the model and its reference.</p><section>'+table(scored)+'</section>')


def depth_section(output):
    path = output/'depth.json'
    if not path.is_file():
        return ''
    depth = json.loads(path.read_text())
    rows = []
    for method, scores in depth['scores']['test'].items():
        rows.append([method.replace('_',' '), scores['rmse_inches'],scores['mae_inches']])
    scores = pd.DataFrame(rows, columns=['Depth forecast','Monthly RMSE (in)','Monthly MAE (in)'])
    months = pd.DataFrame(depth['months'])
    test = months[months.split=='test'][['month','observed_inches',depth['method_id'],'climatology_inches']].copy()
    test.columns = ['Month','Observed (in)','Our model (in)','Historical average (in)']
    return (f'<h2>Fresh-snow depth at Main Lodge</h2><p>A separate calibration translates hourly HRRR snow water '
            f'into estimated fresh-snow depth using complete Pacific calendar months of resort reports. '
            f'The validation winter selected {escape(depth["method_id"].replace("_"," "))}. Error was measured '
            f'on {len(test)} held-out months. Hourly resort depth timing comes from the weather model and '
            f'has no independent hourly depth score.</p><section>'+table(scores)+'</section>'
            '<details><summary>Measured and predicted monthly fresh snow</summary>'+table(test)+'</details>')


def figures(rows, out):
    complete = rows.groupby('issue_utc').observed_mm.agg(['count','sum'])
    candidates = complete[(complete['count'] == 48) & (complete['sum'] > 0)].sort_index()
    # One issue per UTC date prevents overlapping updates on stormy days from
    # receiving disproportionate weight in this illustrative selection.
    candidates = candidates[~candidates.index.normalize().duplicated()]
    if len(candidates) < 2:
        raise ValueError('Two complete snowy test dates are required for the comparison plot.')
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(11.7, 6.5), layout='constrained')
    for col, quantile in enumerate((.5,.9)):
        issue = (candidates['sum'] - candidates['sum'].quantile(quantile)).abs().idxmin()
        block = rows[rows.issue_utc == issue].sort_values('hour')
        label = issue.tz_convert('America/Los_Angeles').strftime('%b %d, %Y · %I %p %Z')
        axes[0,col].set_title(('Median wet window' if col == 0 else '90th-percentile wet window') + '\n' + label,
                              loc='left', fontsize=11, fontweight='bold')
        for name, field, color, style in (
            ('Station','observed_mm','#1b252b','-'),('Our model','model_mm','#b45b29','--'),
            ('HRRR','hrrr_mm','#147772',':'),('ECMWF','ecmwf_mm','#4b70b4','-.')):
            if field not in block or block[field].isna().all():
                continue
            axes[0,col].plot(block.hour, block[field], style, color=color, label=name, lw=1.8)
            axes[1,col].plot(np.r_[0,block.hour],np.r_[0,block[field].cumsum(skipna=False)],
                             style,color=color,lw=1.8)
        for ax in axes[:,col]:
            ax.set_xlim(0,48)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=.22)
            ax.spines[['top','right']].set_visible(False)
        axes[0,col].legend(fontsize=8, frameon=False, loc='upper left')
        axes[1,col].set_xlabel('Hours after forecast start')
    axes[0,0].set_ylabel('Hourly snow water (mm)')
    axes[1,0].set_ylabel('Accumulated snow water (mm)')
    fig.savefig(out/'validation.png', dpi=145, bbox_inches='tight')
    plt.close(fig)


def build(root=ROOT):
    out = Path(root)/'outputs/mammoth-hourly'
    rows = pd.read_csv(out/'test_predictions.csv.gz',parse_dates=['issue_utc'],low_memory=False).sort_values(['issue_utc','hour'])
    metrics = pd.read_csv(out/'metrics.csv')
    model = json.loads((out/'model.json').read_text())
    family = model['family']
    figures(rows,out)
    blocks = []
    for field,label in (('hrrr_mm','Native HRRR'),('ecmwf_mm','ECMWF IFS via Open-Meteo'),
                        ('hrrr_tail_baseline_mm','HRRR with a decaying persistence forecast beyond its last hour'),
                        ('climatology_mm','Monthly station average')):
        if field in rows and rows[field].notna().any():
            blocks.append(f'<h3>{escape(label)}</h3>'+table(paired_scores(rows,field)))
    test = metrics[(metrics.split=='test') & (metrics.method==family)]
    err = test[['horizon','n','n_runs','rmse_mm','mae_mm','bias_mm']].copy()
    err.columns = ['Hours','Windows','Weather runs','RMSE','MAE','Mean error']
    err['80% range coverage'] = [f"{100*model['test'][str(h)]['interval_coverage']:.0f}%"
        if 'interval_coverage' in model['test'][str(h)] else '—' for h in err.Hours]
    selection = pd.DataFrame(sorted(model['selection']['scores'].items(), key=lambda e:e[1]),
                             columns=['Candidate','Validation score'])
    split_rows=[]
    for name, entry in model['splits'].items():
        seasons = ', '.join(f'{y}/{str(y+1)[2:]}' for y in entry['season_start_years'])
        split_rows.append([name.capitalize(),seasons,entry['runs'],entry['issues'],entry['issues_with_point_rate']])
    split = pd.DataFrame(split_rows,columns=['Use','Winters','Weather runs','Window starts','With valid point radar'])
    supplemental = minute_check(rows,Path(root),out)
    depth_report = depth_section(out)
    css = """body{font:16px/1.5 system-ui,sans-serif;margin:auto;max-width:1080px;padding:2rem;color:#17383f}
    h1,h2{line-height:1.18}h2{margin-top:2rem}h3{margin-top:1.5rem}p{max-width:85ch}
    h1{font-size:2.2rem}img{width:100%;height:auto}table{border-collapse:collapse;font-size:.9rem}
    td,th{padding:.42rem .6rem;border-bottom:1px solid #dde5e4;text-align:right}th:first-child,td:first-child{text-align:left}
    section{overflow-x:auto}a{color:#146b70}summary{cursor:pointer}"""
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Mammoth hourly model evaluation</title><style>{css}</style><main>
    <p><a href="/?mountain=mammoth&amp;hours=48">Mammoth live forecast</a></p>
    <h1>How well does Snowline predict falling snow at Mammoth?</h1>
    <p>The selected forecast is <strong>{escape(family)}</strong>. Each hourly value predicts the water
    equivalent of falling snow at the Jeff Dozier Snow Study Site, about 1.1 km from Main Lodge.
    An accumulation of 1 mm SWE means 1 litre of water per square metre after melting.</p>
    <h2>Two storms from held-out winters</h2>
    <img src="validation.png" alt="Hourly and accumulated snow-water predictions against station observations in two withheld storm windows">
    <p>Each column starts from a single issued forecast. Examples were chosen by observed 48-hour accumulation,
    using the median and 90th percentile among complete snowy test dates. Only the earliest complete
    snowy start per UTC date was eligible. The tables below use all eligible windows.</p>
    <h2>Forecast errors by horizon</h2>
    <p>Accumulation errors are in mm SWE. RMSE gives large misses more weight. MAE is the mean absolute miss.
    Positive mean error indicates overprediction. The prediction range uses errors from the validation winter.
    Its coverage is measured below on the two later winters.</p><section>{table(err)}</section>
    <h2>Matched comparisons on wet and dry windows</h2>
    <p>Every pair of scores uses the same complete observed hours and forecast windows. Wet means measured
    accumulation greater than zero. Dry means exactly zero throughout the window. A dash indicates that
    the sample was empty. An extended HRRR cycle reaches 48 hours from initialization. A later rolling
    forecast can therefore have fewer than 48 native HRRR hours. The fitted model handles its remaining hours.</p>
    <section>{''.join(blocks)}</section>
    {supplemental}{depth_report}<h2>Model selection</h2><p>Candidate weights were fitted on the first two winters. The next winter
    selected the lowest mean validation RMSE relative to calendar climatology, with equal weight on each
    of the six horizons. Later winters measure out-of-sample performance. Window starts within a
    weather run overlap and should not be counted as independent storms.</p><section>{table(split)}</section>
    <details><summary>Validation scores for all candidates</summary>{table(selection)}</details>
    <p>Complete hours contain 60 measured minutes. Four hours that hit the instrument limit were excluded.
    Other published high rates were retained. The instrument series includes unusually large readings. Model and error values are expressed on its measurement scale. The station estimates frozen precipitation intensity from
    particle measurements using a proprietary conversion. Extreme rates affect RMSE, and the model
    inherits the instrument’s uncertainty. Winter data gaps are substantial at the longer horizons.
    The resort’s season totals measure fresh-snow depth, a separate quantity.</p>
    <p><a href="metrics.csv">Download scores</a> · <a href="https://doi.org/10.5281/zenodo.17179963">Station data</a>
    · <a href="https://registry.opendata.aws/noaa-hrrr-pds/">NOAA HRRR</a>
    · <a href="https://www.nssl.noaa.gov/projects/mrms/">NOAA MRMS</a>
    · <a href="https://open-meteo.com/en/docs/single-runs-api">Open-Meteo single-run archive</a></p></main></html>"""
    (out/'report.html').write_text(html,encoding='utf-8')


if __name__ == '__main__':
    build()
