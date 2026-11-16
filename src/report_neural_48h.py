"""Current 48-hour outlook with native daily neural amounts kept explicit.

Hourly allocation is conditional on a complete, positive provider snowfall
shape over each full reporting day. Dry or missing timing is unavailable, never
silently zero or uniform. A separate uniform scenario is not a forecast.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))


def utc(value):
    t = pd.Timestamp(value)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def overlap_seconds(start, end, starts, ends):
    return np.maximum(0., (np.minimum(utc(end).value, ends.as_unit('ns').asi8)
                          - np.maximum(utc(start).value, starts.as_unit('ns').asi8)) / 1e9)


def day_weights(day_start, day_end, provider_ends, provider_snow, display_ends):
    """Fraction of one daily amount assigned to each displayed clock hour.

    The provider supplies totals for the hour ending at each timestamp.
    Full-day normalization precedes cropping. None means timing unresolved.
    """
    lo, hi = utc(day_start), utc(day_end)
    if hi - lo != pd.Timedelta(hours=24):
        raise ValueError('Neural reporting windows must be 24 hours')
    pe = pd.DatetimeIndex(pd.to_datetime(provider_ends, utc=True))
    de = pd.DatetimeIndex(pd.to_datetime(display_ends, utc=True))
    values = np.asarray(provider_snow, float)
    if len(values) != len(pe) or pe.has_duplicates or not pe.is_monotonic_increasing:
        raise ValueError('Provider arrays must have matching, unique ordered hours')
    if len(pe) > 1 and not np.all(np.diff(pe.as_unit('ns').asi8) == pd.Timedelta(hours=1).value):
        raise ValueError('Provider hours must be contiguous')
    ps = pe - pd.Timedelta(hours=1)
    full = overlap_seconds(lo, hi, ps, pe)
    active = full > 0
    if not np.isclose(full.sum(), 86400) or not np.isfinite(values[active]).all():
        return None, 'incomplete provider reporting-day coverage'
    if (values[active] < 0).any():
        raise ValueError('Negative provider snowfall is invalid')
    total = float(np.sum(values[active] * full[active] / 3600))
    if total <= 0:
        return None, 'provider predicts no snow in this reporting day'
    result = []
    for end in de:
        a, b = max(lo, end - pd.Timedelta(hours=1)), min(hi, end)
        if b <= a:
            result.append(0.)
            continue
        seconds = overlap_seconds(a, b, ps, pe)
        use = seconds > 0
        result.append(float(np.sum(values[use] * seconds[use] / 3600)) / total)
    return np.asarray(result), None


def uniform_weights(day_start, day_end, display_ends):
    ends = pd.DatetimeIndex(pd.to_datetime(display_ends, utc=True))
    return overlap_seconds(day_start, day_end, ends - pd.Timedelta(hours=1), ends) / 86400


def cumulative_distribution(paths, weights):
    paths = np.asarray(paths, float)
    weights = np.asarray(weights, float)
    if paths.ndim != 2 or weights.ndim != 2 or paths.shape[1] != weights.shape[0]:
        raise ValueError('Joint daily paths and day weights do not align')
    if not np.isfinite(paths).all() or (paths < 0).any():
        raise ValueError('Neural paths must be finite nonnegative amounts')
    cumulative = np.cumsum(paths @ weights, axis=1)
    return np.quantile(cumulative, [.1, .9], axis=0)


def build(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo

    folder = Path(folder)
    plan = json.loads((ROOT / 'outputs/neural-48h/current_plan.json').read_text())
    manifest = json.loads((folder / 'daily_inference_manifest.json').read_text())
    daily = pd.read_csv(folder / 'daily_forecasts.csv')
    paths = np.load(folder / 'joint_paths.npz', allow_pickle=False)
    start, end = utc(plan['forecast_start_utc']), utc(plan['forecast_end_utc'])
    ends = pd.date_range(start + pd.Timedelta(hours=1), end, freq='h')
    sites = ['alta', 'vail', 'mansfield', 'jay']
    names = {'alta': 'Alta', 'vail': 'Vail', 'mansfield': 'Stowe', 'jay': 'Jay Peak'}
    forecasts, summaries, daily_audits, hourly_rows = {}, [], [], []
    for site in sites:
        provider = json.loads((folder / f'{site}_weather.json').read_text())
        raw = json.loads((folder / f'{site}_open_meteo_raw.json').read_text())
        if raw['hourly_units']['snowfall'] != 'cm':
            raise ValueError('Unexpected provider snowfall unit')
        pe = pd.to_datetime(raw['hourly']['time'], utc=True)
        pv = np.asarray(raw['hourly']['snowfall'], float) / 2.54
        positions = pe.get_indexer(ends)
        if (positions < 0).any() or not np.isfinite(pv[positions]).all():
            raise ValueError(f'Incomplete provider 48-hour data for {site}')
        g = daily[daily.site.eq(site)].sort_values('window_end_utc').reset_index(drop=True)
        if not len(g):
            raise ValueError(f'Missing neural daily forecasts for {site}')
        values = paths[f'{site}_paths_inches']
        path_ends = pd.to_datetime(paths[f'{site}_window_end_utc'], utc=True)
        wanted = pd.to_datetime(g.window_end_utc, utc=True)
        mapping = path_ends.get_indexer(wanted)
        if (mapping < 0).any():
            raise ValueError('Joint paths do not contain every daily endpoint')
        values = values[:, mapping]
        means = g.mean_inches.to_numpy(float)
        weights, uniforms, unknown_hours = [], [], np.zeros(len(ends), bool)
        lower_total = upper_total = 0.
        reasons = []
        for row in g.itertuples():
            w, reason = day_weights(row.window_start_utc, row.window_end_utc, pe, pv, ends)
            uniform = uniform_weights(row.window_start_utc, row.window_end_utc, ends)
            uniforms.append(uniform)
            if w is None:
                # Nonnegative daily amount bounds a partial-day amount; no
                # timing uncertainty distribution is invented.
                w = np.zeros(len(ends))
                unknown_hours |= (uniform > 0) & (row.mean_inches > 0)
                complete = np.isclose(uniform.sum(), 1.)
                lower_total += row.mean_inches if complete else 0.
                upper_total += row.mean_inches if uniform.sum() > 0 else 0.
                reasons.append(reason)
            else:
                lower_total += row.mean_inches * w.sum()
                upper_total += row.mean_inches * w.sum()
            weights.append(w)
            daily_audits.append({'site': site, 'window_start_utc': row.window_start_utc,
                'window_end_utc': row.window_end_utc, 'timing_status': reason or 'provider snowfall shape',
                'fraction_of_daily_amount_in_display': None if reason else float(w.sum())})
        weights, uniforms = np.asarray(weights), np.asarray(uniforms)
        if not np.allclose(uniforms.sum(axis=0), 1/24):
            raise ValueError('Daily windows do not partition the full requested horizon')
        cum = np.cumsum(means @ weights)
        q10, q90 = cumulative_distribution(values, weights)
        unknown_cum = np.maximum.accumulate(unknown_hours)
        cum[unknown_cum] = np.nan; q10[unknown_cum] = np.nan; q90[unknown_cum] = np.nan
        uniform_mean = np.cumsum(means @ uniforms)
        ulo, uhi = cumulative_distribution(values, uniforms)
        record = dict(provider=provider, daily=g, mean=cum, q10=q10, q90=q90,
            provider_cumulative=np.cumsum(pv[positions]), provider_hourly=pv[positions],
            uniform_mean=uniform_mean, uniform_q10=ulo, uniform_q90=uhi,
            timing_unknown=unknown_hours, paths=values, reasons=sorted(set(reasons)))
        forecasts[site] = record
        summaries.append({'site': site, 'mountain': names[site], 'start_utc': start.isoformat(),
            'end_utc': end.isoformat(), 'provider_48h_inches': float(pv[positions].sum()),
            'neural_48h_expected_lower_timing_bound_inches': float(lower_total),
            'neural_48h_expected_upper_timing_bound_inches': float(upper_total),
            'neural_hourly_timing_resolved': not bool(unknown_hours.any()),
            'uniform_scenario_mean_inches': float(uniform_mean[-1]),
            'uniform_scenario_q10_inches': float(ulo[-1]), 'uniform_scenario_q90_inches': float(uhi[-1]),
            'target': provider['site']['target_label']})
        for i, t in enumerate(ends):
            hourly_rows.append({'site': site, 'hour_end_utc': t.isoformat(),
                'provider_snowfall_inches': float(pv[positions[i]]),
                'provider_cumulative_inches': float(record['provider_cumulative'][i]),
                'neural_allocated_cumulative_inches': float(cum[i]),
                'neural_allocated_cumulative_q10_inches': float(q10[i]),
                'neural_allocated_cumulative_q90_inches': float(q90[i]),
                'uniform_scenario_cumulative_inches': float(uniform_mean[i])})

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.titleweight':'bold', 'axes.labelcolor':'#28434a',
        'text.color':'#19373e', 'axes.edgecolor':'#9fb0b4'})
    blue, orange = '#168291', '#c7792b'
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), gridspec_kw={'width_ratios':[1.35,1]}, layout='constrained')
    for j, site in enumerate(sites):
        d = forecasts[site]; tz = ZoneInfo(d['provider']['site']['timezone'])
        x = [start.to_pydatetime(), *ends.to_pydatetime()]
        ax, bar = axes[j]
        ax.plot(x, np.r_[0,d['provider_cumulative']], lw=2.4, color=blue, label='Weather-provider forecast')
        if np.isfinite(d['mean']).any():
            ax.plot(x, np.r_[0,d['mean']], lw=2.2, color=orange, label='Neural daily amount + weather timing')
            ax.fill_between(x, np.r_[0,d['q10']], np.r_[0,d['q90']], color=orange, alpha=.17,
                label='Conditional neural 10–90% range')
        else:
            ax.text(.04,.8,'Neural hourly timing unavailable\nDaily predictions shown at right',transform=ax.transAxes,
                    color='#87603f',fontsize=10,va='top')
        if d['timing_unknown'].any() and np.isfinite(d['mean']).any():
            ax.text(.98,.04,'Neural curve stops where timing is unresolved',transform=ax.transAxes,
                    ha='right',color='#87603f',fontsize=8)
        ax.set_title(f'{names[site]} · next 48 hours',loc='left')
        ax.set_ylabel('Cumulative snowfall · inches')
        ax.set_xlim(start.to_pydatetime(), end.to_pydatetime())
        ax.set_ylim(bottom=-.025,top=max(.2,ax.get_ylim()[1]))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%I %p',tz=tz))
        ax.grid(alpha=.18);ax.legend(fontsize=8,loc='upper left' if np.isfinite(d['mean']).any() else 'lower right')
        g=d['daily']; idx=np.arange(len(g)); means=g.mean_inches.to_numpy()
        lo,hi=np.quantile(d['paths'],[.1,.9],axis=0)
        bar.bar(idx,means,color=orange,alpha=.8,width=.5,label='Neural daily mean')
        # Percentiles can lie below the mean in a heavy-tailed mixture. Plot
        # them as independent vertical segments, not invalid error bars.
        bar.vlines(idx,lo,hi,color='#795127',lw=2,label='Daily 10–90% range')
        bar.scatter(idx,means,color='#513b25',s=18,zorder=3)
        labels=[utc(t).tz_convert(tz).strftime('Ends %b %d\n%I:%M %p') for t in g.window_end_utc]
        bar.set_xticks(idx,labels);bar.set_ylabel('Full reporting-day snow · inches')
        bar.set_title('Native daily neural output',loc='left');bar.grid(axis='y',alpha=.18)
        bar.set_ylim(bottom=0,top=max(.2,bar.get_ylim()[1]));bar.legend(fontsize=8)
        bar.text(.02,.97,'Full 24-hour windows; edge days extend outside the 48h view',
                 transform=bar.transAxes,va='top',fontsize=7.5,color='#5a6b6e')
    fig.suptitle('48-hour snow outlook · saved neural models and fresh weather guidance\n'
        f'{start:%b %d}–{end:%b %d, %Y} · daily neural amounts are not a 48-hour radar forecast',
        fontsize=17,fontweight='bold')
    fig.savefig(folder/'forecast_48h.png',dpi=160);plt.close(fig)

    # Separate illustrative scenario, never silently substituted into main view.
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,site in zip(axes.flat,sites):
        d=forecasts[site]
        ax.plot(np.arange(49),np.r_[0,d['uniform_mean']],ls='--',color='#6c7377',label='Assumed uniform timing')
        ax.fill_between(np.arange(49),np.r_[0,d['uniform_q10']],np.r_[0,d['uniform_q90']],color='#939b9e',alpha=.2)
        ax.plot(np.arange(49),np.r_[0,d['provider_cumulative']],color=blue,label='Weather-provider forecast')
        ax.set(title=names[site],xlabel='Hours after window starts',ylabel='Cumulative snow · inches',xlim=(0,48))
        ax.set_ylim(bottom=-.02,top=max(.2,ax.get_ylim()[1]));ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Illustrative scenario only · uniform timing within each neural reporting day\n'
        'Not a storm forecast; model-based amount spread excludes timing uncertainty',fontsize=15,fontweight='bold')
    fig.savefig(folder/'uniform_timing_scenario.png',dpi=150);plt.close(fig)

    summary=pd.DataFrame(summaries);summary.to_csv(folder/'forecast_summary.csv',index=False)
    pd.DataFrame(hourly_rows).to_csv(folder/'hourly_comparison.csv',index=False)
    (folder/'allocation_audit.json').write_text(json.dumps({'daily_windows':daily_audits,
        'policy':'Full-reporting-day normalization, exact overlaps; no dry-day timing imputation in primary curves.',
        'quantiles':'Quantiles of accumulated joint daily paths conditional on deterministic timing; not calibrated coverage.',
        'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    cards=[]
    for r in summaries:
        a,b=r['neural_48h_expected_lower_timing_bound_inches'],r['neural_48h_expected_upper_timing_bound_inches']
        amount=f'{a:.2f}' if np.isclose(a,b) else f'{a:.2f}–{b:.2f}'
        cards.append(f"<tr><td>{r['mountain']}</td><td>{r['provider_48h_inches']:.2f}</td><td>{amount}</td><td>{'Available' if r['neural_hourly_timing_resolved'] else 'Some timing unresolved'}</td></tr>")
    daily_table=daily[['site','window_start_utc','window_end_utc','mean_inches','q10_inches','q90_inches']].copy()
    daily_table['site']=daily_table.site.map(names)
    freshness=[]
    for site in sites:
        m=manifest['sites'][site]
        freshness.append({'Mountain':names[site], 'Observed weather through':m['context_last'],
            'Missing context':f'{100*m["context_missing_fraction"]:.1f}%',
            'Forecast origin':m['model_origin_date'], 'Unobserved bridge days':m['bridge_days']})
    freshness_table=pd.DataFrame(freshness).to_html(index=False,border=0)
    sources=' '.join(f'<li>{html.escape(names[s])}: <a href="{html.escape(forecasts[s]["provider"]["source"]["url"],quote=True)}">weather input snapshot source</a>; retrieved {html.escape(forecasts[s]["provider"]["source"]["retrieved_at"])}.</li>' for s in sites)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Snowline · Current 48-hour neural outlook</title><style>
body{{margin:0;background:#edf2ef;color:#203b42;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 28px}}h1{{font-size:40px;line-height:1.15}}h2{{margin-top:0}}section{{background:white;padding:28px;border:1px solid #d8e2dc;border-radius:14px;margin:24px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;border-bottom:1px solid #dfe6e1;padding:10px}}.scroll{{overflow:auto}}.note{{border-left:4px solid #bd7d42;padding:12px 18px;background:#faf4eb}}small{{color:#586f74}}a{{color:#0b737f}}summary{{cursor:pointer;font-weight:650}}.eyebrow{{letter-spacing:.14em;font-size:12px;color:#147582;font-weight:700}}</style>
<main><div class="eyebrow">SNOWLINE RESEARCH · CURRENT INFERENCE</div><h1>Next 48 hours.<br>What the neural models actually predict.</h1>
<p>Window: <b>{start:%B %d, %Y, %H:%M}–{end:%B %d, %Y, %H:%M} UTC</b>. This is {start.tz_convert('America/New_York'):%b %d, %I %p}–{end.tz_convert('America/New_York'):%b %d, %I %p} Eastern and {start.tz_convert('America/Denver'):%b %d, %I %p}–{end.tz_convert('America/Denver'):%b %d, %I %p} Mountain time.</p>
<div class="note"><b>Native neural output is daily.</b> The saved event networks predict full station reporting days. The Alta radar networks were trained for about three hours of liquid precipitation; none of them is a validated 48-hour hourly snow model. This report runs the existing daily neural models independently for all four targets. It does not retrain them or extend radar outputs beyond their trained horizon.</div>
<section><h2>48-hour comparison</h2><div class="scroll"><table><thead><tr><th>Mountain</th><th>Weather-provider total · in</th><th>Neural expected amount / timing bounds · in</th><th>Neural hourly timing</th></tr></thead><tbody>{''.join(cards)}</tbody></table></div>
<p><small>The neural interval in this table is a <b>timing-identification bound, not a prediction interval</b>: full reporting days inside the window contribute their complete mean; unresolved partial days can contribute between none and their full mean. Where provider snow timing exists, that timing fixes the allocated fraction. A single number means the amount can be determined from the daily totals and available timing assumptions; it does not imply certainty.</small></p>
<img src="forecast_48h.png" alt="Four mountain outlooks: 48-hour weather-provider cumulative snowfall and native daily neural predictions">
<p>Blue: fresh weather-provider guidance. Orange hourly curves, when present: neural daily amounts allocated using the provider's snowfall timing over each <b>complete 24-hour reporting window</b>. Orange bars: full native daily neural amounts. Missing timing remains missing. A genuine zero weather forecast is drawn at zero.</p>
<p>Daily and conditional cumulative shading represents model-based 10th–90th percentiles from joint simulated paths. It excludes uncertainty in the hourly timing and has no guaranteed prospective coverage. The rightmost and leftmost daily bars may extend outside the requested 48 hours.</p></section>
<section><h2>Interpretation and target locations</h2><p>Alta targets manual Alta–Collins snowfall. Stowe targets the NOAA Mount Mansfield station. Jay Peak targets a NOAA station around 561 metres elevation, not the summit. Vail predicts a reported Sno24 sensor proxy whose equivalence to fresh snowfall remains unverified.</p>
<p>The daily neural networks use past temperature and precipitation, calendar and lagged ENSO; they do not ingest the future weather forecasts. Fresh weather guidance supplies timing only in the experimental allocation. These neural candidates did not consistently beat climatology or statistical models in earlier benchmarks, so they are not labeled the best overall forecasts.</p>
<p>For a proper 48-hour neural weather forecast, the next model must be trained on issued weather-forecast trajectories and matched 24/48-hour snowfall targets. Radar can condition the first few hours. Native hourly snowfall skill additionally needs independent hourly snowfall observations.</p>
<details><summary>Native daily values and observation freshness</summary><div class="scroll">{daily_table.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)}</div><h3>Actual observations used by the neural models</h3><div class="scroll">{freshness_table}</div><p>Fresh retrieval does not make the observations current: NOAA weather currently reaches November 7, with Stowe precipitation only through November 4. The daily models propagate from their earlier origin across unobserved days. Vail uses hourly observations through this morning. Missing observations stay missing under the training-time preprocessing. Station reporting clocks remain the same assumptions as in the earlier evaluation.</p></details></section>
<section><details><summary>Illustrative uniform-timing scenario</summary><p>This optional scenario spreads each neural daily amount evenly across its reporting day, including days for which weather guidance predicts no snow. It illustrates what the daily model could imply under that assumption. It is <b>not a storm forecast or independently trained hourly prediction</b>, and is excluded from the primary comparison.</p><img src="uniform_timing_scenario.png" alt="Illustrative uniform timing scenario for daily neural snowfall amounts"></details></section>
<section><h2>Sources and exports</h2><ul>{sources}</ul><p><a href="https://open-meteo.com/en/docs">Open-Meteo documentation and attribution</a> · <a href="forecast_summary.csv">48-hour summary</a> · <a href="daily_forecasts.csv">Native neural daily predictions</a> · <a href="hourly_comparison.csv">Hourly comparison</a> · <a href="daily_inference_manifest.json">Input and checkpoint provenance</a> · <a href="daily_inference_audit.json">Neural restoration and timing checks</a> · <a href="allocation_audit.json">Temporal allocation audit</a></p><p><a href="/">Open live dashboard</a> · <a href="/reports/alta/learning-review/report.html">Alta radar training comparison</a></p></section></main></html>'''
    (folder/'report.html').write_text(body)
    public=ROOT/'outputs/nwp-weather/neural-48h';public.mkdir(exist_ok=True)
    for name in ['report.html','forecast_48h.png','uniform_timing_scenario.png','forecast_summary.csv',
                 'daily_forecasts.csv','hourly_comparison.csv','daily_inference_manifest.json','daily_inference_audit.json','allocation_audit.json']:
        shutil.copy2(folder/name,public/name)
    print(summary.to_string(index=False))
    print(public/'report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    build(p.parse_args().folder)
