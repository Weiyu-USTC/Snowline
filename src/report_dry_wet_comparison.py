"""Paired dry/wet test errors against native HRRR and archived provider guidance.

Frozen forecast artifacts only. Missing horizons remain missing; no forecast is
spliced across later issues or scored on a different set of observations.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import os
import numpy as np
import pandas as pd
from dry_wet_provider_inputs import load_provider24, load_provider12

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/nwp-weather/forecast-portfolio/dry-wet'
NAMES = {'alta':'Alta', 'vail':'Vail', 'mansfield':'Stowe', 'jay':'Jay Peak'}
SOURCE_FILES = set()


def read_csv(path):
    source = ROOT/path
    SOURCE_FILES.add(source)
    return pd.read_csv(source)


def rmse(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((p-y)**2))) if len(y) else np.nan


def week_comparison(group, baseline, category, seed, resamples=4000):
    """Resample paired calendar weeks, retaining dry/wet membership by truth.

    Counts and squared errors are aggregated within weeks. Selecting weeks with
    replacement keeps their cases together, including overlapping 48h windows
    within each block. Uncertainty remains conditional on this one test winter.
    """
    dates = pd.to_datetime(group.window_start, utc=True)
    blocks = dates.dt.strftime('%G-%V').to_numpy()
    select = np.ones(len(group), bool) if category == 'all' else (group.observed_cm.to_numpy() == 0 if category == 'dry' else group.observed_cm.to_numpy() > 0)
    keys = np.unique(blocks)
    our_error = (group.our_cm-group.observed_cm).to_numpy()**2
    base_error = (group[baseline]-group.observed_cm).to_numpy()**2
    aggregate = np.array([[int(np.sum(select & (blocks == key))),
                           float(np.sum(our_error[select & (blocks == key)])),
                           float(np.sum(base_error[select & (blocks == key)]))] for key in keys])
    support = int(np.sum(aggregate[:,0] > 0))
    if support < 2:
        return dict(delta_rmse_low_cm=None, delta_rmse_high_cm=None, wet_or_dry_weeks=support, valid_resamples=0)
    rng = np.random.default_rng(seed)
    samples = aggregate[rng.integers(0, len(keys), size=(resamples,len(keys)))].sum(axis=1)
    samples = samples[samples[:,0] > 0]
    delta = np.sqrt(samples[:,1]/samples[:,0])-np.sqrt(samples[:,2]/samples[:,0])
    lower, upper = np.quantile(delta,[.025,.975])
    return dict(delta_rmse_low_cm=float(lower), delta_rmse_high_cm=float(upper),
                wet_or_dry_weeks=support, valid_resamples=len(delta))


def make_cases():
    datasets = []
    d = read_csv('outputs/nwp-weather/alta12-trained/native_asnow_predictions.csv')
    d = d[d.stage.eq('test')].copy()
    manifest_path = ROOT/'outputs/nwp-weather/alta12-trained/manifest.json'
    SOURCE_FILES.add(manifest_path)
    selected = json.loads(manifest_path.read_text())['selection']['modern_missing_history']['name']
    p12 = load_provider12()
    merged = d.merge(p12, on='row', validate='one_to_one', suffixes=('', '_provider'))
    assert len(merged) == len(d), 'Alta12 provider coverage differs; publish a separate common-subset cohort explicitly.'
    assert np.allclose(merged.actual_inches, merged.observed_inches, atol=1e-7)
    assert (pd.to_datetime(merged.start_utc,utc=True) == pd.to_datetime(merged.start_utc_provider,utc=True)).all()
    datasets.append(pd.DataFrame(dict(site='alta', horizon_hours=12, cohort='Alta12 native-HRRR matched subset',
        window_start=merged.start_utc, observed_cm=merged.actual_inches*2.54,
        our_cm=merged[selected]*2.54, hrrr_cm=merged.native_asnow_inches*2.54,
        provider_cm=merged.provider_inches*2.54)))

    d = read_csv('outputs/nwp-weather/forecast-portfolio/hrrr24_predictions.csv')
    d = d[d.split.eq('test')].copy()
    p24 = load_provider24()
    merged = d.merge(p24, on=['row','site'], validate='one_to_one', suffixes=('', '_provider'))
    assert len(merged) == len(d)
    assert np.allclose(merged.truth, merged.observed_inches, atol=1e-7)
    assert (pd.to_datetime(merged.start,utc=True) == pd.to_datetime(merged.start_utc,utc=True)).all()
    datasets.append(pd.DataFrame(dict(site=merged.site, horizon_hours=24, cohort='24h native-HRRR matched subset',
        window_start=merged.start, observed_cm=merged.truth*2.54,
        our_cm=merged.portfolio*2.54, hrrr_cm=merged.hrrr_native*2.54, provider_cm=merged.provider_inches*2.54)))

    d = read_csv('outputs/nwp-weather/forecast-portfolio/dry-wet/heldout_cases.csv')
    d = d[d.horizon_hours.eq(48) & d.method.isin(['Our model','Archived JMA provider'])]
    wide = d.pivot(index=['site','window_start'], columns='method', values='predicted_cm').reset_index()
    ours = d[d.method.eq('Our model')][['site','window_start','observed_cm']]
    provider_truth = d[d.method.eq('Archived JMA provider')][['site','window_start','observed_cm']]
    truth = ours.merge(provider_truth, on=['site','window_start'], validate='one_to_one', suffixes=('', '_provider'))
    assert np.allclose(truth.observed_cm, truth.observed_cm_provider, atol=1e-7)
    wide = wide.merge(ours,on=['site','window_start'],validate='one_to_one')
    datasets.append(pd.DataFrame(dict(site=wide.site, horizon_hours=48, cohort='48h neural test cohort; no full HRRR window',
        window_start=wide.window_start, observed_cm=wide.observed_cm, our_cm=wide['Our model'],
        hrrr_cm=np.nan, provider_cm=wide['Archived JMA provider'])))
    cases = pd.concat(datasets, ignore_index=True)
    assert not cases.duplicated(['site','horizon_hours','window_start']).any()
    assert np.isfinite(cases[['observed_cm','our_cm','provider_cm']]).all().all()
    assert (cases[['observed_cm','our_cm','provider_cm']] >= 0).all().all()
    for hours in [12,24]:
        assert np.isfinite(cases.loc[cases.horizon_hours.eq(hours),'hrrr_cm']).all()
    cases['category'] = np.where(cases.observed_cm.eq(0), 'dry', 'wet')
    return cases


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_FILES.update([OUT/'provider_alignment_audit.json', OUT/'short_audit.json',
                         ROOT/'outputs/nwp-weather/forecast-portfolio/deployment_selection.json'])
    cases = make_cases()
    cases.to_csv(OUT/'comparison_cases.csv',index=False)
    rows, paired = [], []
    for (site, hours), group in cases.groupby(['site','horizon_hours']):
        for category in ['all','dry','wet']:
            d = group if category == 'all' else group[group.category.eq(category)]
            row = dict(site=site,mountain=NAMES[site],horizon_hours=int(hours),category=category,n=len(d),
                cohort=group.cohort.iloc[0],our_rmse_cm=rmse(d.observed_cm,d.our_cm),
                hrrr_rmse_cm=rmse(d.observed_cm,d.hrrr_cm) if d.hrrr_cm.notna().all() else np.nan,
                provider_rmse_cm=rmse(d.observed_cm,d.provider_cm))
            rows.append(row)
            for name,column in [('Native HRRR','hrrr_cm'),('Archived JMA provider','provider_cm')]:
                if not group[column].notna().all(): continue
                error = rmse(d.observed_cm,d[column]); ours = row['our_rmse_cm']
                result = dict(site=site,mountain=NAMES[site],horizon_hours=int(hours),category=category,n=len(d),
                    comparator=name,our_rmse_cm=ours,baseline_rmse_cm=error,delta_rmse_cm=ours-error,
                    rmse_reduction_percent=(100*(error-ours)/error) if error>0 else None,
                    **week_comparison(group,column,category,seed=20261113+int(hours)+list(NAMES).index(site)))
                lo,hi=result['delta_rmse_low_cm'],result['delta_rmse_high_cm']
                result['evidence'] = 'Insufficient weeks' if lo is None else 'Lower within this test winter' if hi<0 else 'Higher within this test winter' if lo>0 else 'Difference unresolved'
                paired.append(result)
    scores = pd.DataFrame(rows).sort_values(['horizon_hours','site','category'], key=lambda s:s.map({v:i for i,v in enumerate(NAMES)}) if s.name=='site' else s)
    effects = pd.DataFrame(paired)
    scores.to_csv(OUT/'comparison_metrics.csv',index=False)
    effects.to_csv(OUT/'paired_comparisons.csv',index=False)
    # A decomposition check catches accidental mismatches in category membership.
    for (site,hours),g in scores.groupby(['site','horizon_hours']):
        d=g.set_index('category'); assert d.loc['dry','n']+d.loc['wet','n']==d.loc['all','n']
        for col in ['our_rmse_cm','hrrr_rmse_cm','provider_rmse_cm']:
            if pd.notna(d.loc['all',col]):
                assert np.isclose(sum(d.loc[k,'n']*d.loc[k,col]**2 for k in ['dry','wet'])/d.loc['all','n'],d.loc['all',col]**2)
    plot(scores)
    availability=[]
    for site in NAMES:
        for h in [1,3,6,12,24,48]:
            reason=('Liquid precipitation, separate short-range appendix' if h<=3 and site!='vail' else
                    'No quality-qualified short-range test observations' if h<=3 else
                    'No verified fresh-snow observations at this horizon' if h==6 or (h==12 and site!='alta') else
                    'Our model and archived provider; native HRRR cannot cover the full future window' if h==48 else
                    'Our model, native HRRR and archived provider on identical windows')
            availability.append(dict(mountain=NAMES[site],hours=h,comparison=reason))
    pd.DataFrame(availability).to_csv(OUT/'comparison_availability.csv',index=False)
    audit=dict(generated_at=datetime.now(timezone.utc).isoformat(), test_winter='2023/24',
        category='Observed window total exactly0 versus positive; recorded traces may be0.',
        units='Centimetres of snowfall; native source inches multiplied by2.54.',
        matched_cases='All displayed methods within each row share exactly the same observations and dates. Alta12 is96 matched cases, not the prior364-case full cohort.',
        provider='Open-Meteo archived JMA GSM previous_day3: each valid hour has a nominal72h-offset source bound, a coarse interpolated mosaic, not a single matched-age run or the live best_match product.',
        hrrr48='No complete issued HRRR48h coverage of the target from the forecast decision time; no padding, future reruns or changed windows used.',
        bootstrap='4000 paired calendar-week block resamples; categories computed from observations. Conditional on one previously inspected test winter; cross-week storm dependence may remain.',
        selection='Unchanged frozen per-horizon selection; no category-specific model retuning.',
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE_FILES)})
    (OUT/'comparison_audit.json').write_text(json.dumps(audit,indent=2))
    render(scores,effects,availability)
    print(scores[scores.category.isin(['dry','wet'])][['mountain','horizon_hours','category','n','our_rmse_cm','hrrr_rmse_cm','provider_rmse_cm']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print(effects[effects.category.eq('wet')][['mountain','horizon_hours','comparator','rmse_reduction_percent','delta_rmse_low_cm','delta_rmse_high_cm','evidence']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))


def plot(scores):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,8),layout='constrained')
    methods=[('Our model','our_rmse_cm','#bc7936'),('Native HRRR','hrrr_rmse_cm','#176b78'),('Archived JMA','provider_rmse_cm','#8a95a1')]
    for ax,category in zip(axes,['dry','wet']):
        d=scores[scores.category.eq(category)].reset_index(drop=True);y=np.arange(len(d))
        for i,(label,column,color) in enumerate(methods):
            available=d[column].notna()
            bars=ax.barh(y[available]+(i-1)*.23,d.loc[available,column],height=.21,label=label,color=color)
            ax.bar_label(bars,labels=[f'{v:.2f}' for v in d.loc[available,column]],padding=3,fontsize=8)
        ax.set_yticks(y,[f'{r.mountain} · {r.horizon_hours}h (n={r.n})' for r in d.itertuples()]);ax.invert_yaxis()
        ax.set_title('Zero recorded snowfall' if category=='dry' else 'Positive recorded snowfall',loc='left',weight='bold')
        ax.set_xlabel('RMSE (cm); lower is better');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_xlim(0,float(d[[m[1] for m in methods]].max().max())*1.18)
        for boundary in [.5,4.5]:ax.axhline(boundary,color='#d8dedd',lw=.7)
    axes[1].legend(loc='lower right',frameon=False)
    fig.suptitle('Dry and snowy forecast errors: model versus baselines',fontsize=18,weight='bold')
    fig.supxlabel('Identical test dates within each row. HRRR at 48h is unavailable. Alta 12h uses the 96-case HRRR-matched subset.\nArchived JMA is older 72h-offset guidance, not the live Open-Meteo product. The two panels have different x-axis ranges.',fontsize=9)
    fig.savefig(OUT/'comparison_rmse.png',dpi=170);plt.close(fig)


def render(scores,effects,availability):
    def table(d):return d.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:.2f}')
    tables=[]
    for category,title in [('wet','Snowy periods'),('dry','Dry periods')]:
        d=scores[scores.category.eq(category)][['mountain','horizon_hours','n','our_rmse_cm','hrrr_rmse_cm','provider_rmse_cm']].copy()
        d.columns=['Mountain','Hours','Cases','Our model (cm)','HRRR (cm)','Archived JMA (cm)']
        tables.append(f'<section><h2>{title}</h2><div class="scroll">{table(d)}</div></section>')
    effect=effects[effects.category.eq('wet')][['mountain','horizon_hours','comparator','n','rmse_reduction_percent','delta_rmse_low_cm','delta_rmse_high_cm','evidence']].copy()
    effect.columns=['Mountain','Hours','Comparator','Wet cases','RMSE reduction %','95% delta low (cm)','95% delta high (cm)','Evidence']
    short=OUT/'short_summary.csv'
    short_section=''
    if short.exists():
        d=pd.read_csv(short);d=d[d.category.isin(['dry','wet'])]
        short_section=f'<section><h2>1–3h appendix: liquid precipitation</h2><p>These errors are millimetres of water, not centimetres of fresh snow. A wet radar interval is not necessarily a snowy interval. Vail has no quality-qualified short-range test cases.</p><div class="scroll">{table(d)}</div><p><a href="short_summary.csv">Short-range metrics</a> · <a href="short_audit.json">Short-range source audit</a></p></section>'
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Snowline · Dry/wet model comparisons</title>
<style>body{{margin:0;background:#f0f4f1;color:#193d43;font:16px/1.6 system-ui}}main{{max-width:1200px;margin:auto;padding:28px}}section{{background:white;border:1px solid #d7e1dd;border-radius:16px;margin:20px 0;padding:24px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;text-align:right;border-bottom:1px solid #dce4e0}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}img{{width:100%;height:auto}}a{{color:#087b82}}h1{{line-height:1.15}}li{{margin-bottom:10px}}</style>
<main><a href="report.html">← Original dry/wet breakdown</a><h1>Are the models better when it snows?</h1><p>Frozen 2023/24 held-out predictions, split by observed zero versus positive accumulation. All methods within a table row use identical dates, observations and accumulation windows. No retraining or model selection used this breakdown.</p>
<section><h2>What this comparison establishes</h2><ul><li>Against HRRR on snowy periods, Alta has 31.8% lower 12h RMSE and 43.4% lower 24h RMSE. Their paired week-resampling intervals remain below zero in this test winter. Vail's 5.5% and Jay's 3.2% 24h improvements are unresolved; Stowe's 24h point score is 29.8% worse.</li><li>Our model usually produces larger false snowfall amounts on dry periods. At Jay 24h, the archived provider also has lower snowy-period RMSE (2.81 cm versus 3.29 cm), although that difference remains unresolved.</li><li>The provider comparison uses archived JMA guidance delivered by Open-Meteo, with a nominal 72h offset for each valid hour. It is a coarse mosaic, not a matched-age live Open-Meteo forecast. Lower errors here do not establish superiority over the current weather provider.</li><li>At 48h, the available native HRRR run cannot cover the entire future target window at the decision time. A dash means unavailable; no later forecast runs, shorter windows or invented tails were substituted.</li></ul></section>
<section><img src="comparison_rmse.png" alt="Paired dry and wet snowfall RMSE against HRRR and archived JMA"></section>{''.join(tables)}
<section><h2>Is the snowy-period difference resolved?</h2><p>RMSE reduction is 100 × (baseline RMSE − model RMSE) / baseline RMSE; positive is better. The 95% interval is for model RMSE minus baseline RMSE, so negative is better. We resample paired calendar weeks 4,000 times. An interval crossing zero does not resolve the difference. These exploratory intervals are not adjusted for multiple comparisons. They describe sampling variation within one previously inspected test winter, not uncertainty across future winters.</p><div class="scroll">{table(effect)}</div></section>
<section><h2>Targets and coverage</h2><p>Alta 12h now uses 96 HRRR-matched cases rather than the 364 cases in the earlier standalone score. The 24h cohort is also a smaller matched calendar subset. 48h windows overlap. Vail is a Sno24 sensor proxy with unverified reset equivalence; Stowe and Jay use lower-elevation town observers. Recorded zeros may include trace or rounded snowfall.</p><div class="scroll">{table(pd.DataFrame(availability))}</div></section>{short_section}
<section><h2>Download the evidence</h2><p><a href="comparison_metrics.csv">Dry/wet/overall metrics</a> · <a href="paired_comparisons.csv">Paired uncertainty and improvements</a> · <a href="comparison_cases.csv">Aligned predictions</a> · <a href="comparison_audit.json">Audit</a> · <a href="provider_alignment_audit.json">Provider alignment</a></p><p>Forecasts and model weights are unchanged. The existing all-mountain refresh continues to use the frozen deployed choices.</p></section></main></html>'''
    (OUT/'comparison.html').write_text(body)


if __name__=='__main__':build()
