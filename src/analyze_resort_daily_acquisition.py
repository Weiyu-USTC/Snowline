"""Audit acquired resort records; exploratory association and two-winter pilot.

CAIC Sno24 remains a source-reported candidate target, not a verified manual
resort snow report. No same-day observed weather is used in the forecast pilot.
"""
from pathlib import Path
import hashlib
import html
import json
import os
import zipfile

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/resort_daily_research'
OUT = ROOT / 'outputs/resort-daily-data'
FEATURES = ['snow_lag1_inches','snow_prev3_inches','snow_prev7_inches',
            'lag1_tmax_celsius','lag1_tmin_celsius','lag1_precipitation_mm',
            'lag1_snow_depth_cm']
CALENDAR = ['sin1','cos1','sin2','cos2']


def correlation(frame, x, y):
    a=frame[[x,y]].dropna()
    return float(a[x].corr(a[y])) if len(a)>2 and a[x].std()>0 and a[y].std()>0 else None


def metric(y,p):
    err=np.asarray(y)-np.asarray(p)
    return {'n':len(err),'rmse_inches':float(np.sqrt(np.mean(err**2))),
            'mae_inches':float(np.mean(np.abs(err))), 'mse_inches_squared':float(np.mean(err**2))}


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    d=pd.read_csv(ROOT/'outputs/dillon_deep_daily_input.csv',parse_dates=['date']).set_index('date')
    assert d.index.is_unique
    sources={
        'Keystone':pd.read_csv(DATA/'keystone_caic_daily_0600.csv',parse_dates=['date']).set_index('date'),
        'A-Basin':pd.read_csv(DATA/'abasin_caic_daily_0600.csv',parse_dates=['date']).set_index('date'),
        'Loveland':pd.read_csv(DATA/'loveland_daily.csv',parse_dates=['report_date']).rename(columns={'report_date':'date'}).set_index('date')}
    summaries={};pairs={};allrows=[]
    for name,raw in sources.items():
        assert raw.index.is_unique
        raw=raw.copy().sort_index()
        target=raw['fresh_snow_inches' if name=='Loveland' else 'Sno24']
        if name=='Loveland': target=target.where(raw.status.eq('reported_numeric'))
        raw['reported_snow_inches']=target
        raw['resort']=name
        raw['measurement_status']={'Keystone':'CAIC Sno24 candidate; protocol/time window unverified',
                                   'A-Basin':'QUARANTINED CAIC Sno24; major quality concerns',
                                   'Loveland':'Official daily resort log; timing and table-entry audit pending'}[name]
        raw['dillon_same_date_inches']=d.snowfall_inches.reindex(raw.index)
        raw['dillon_prior_date_inches']=d.snowfall_inches.shift(1).reindex(raw.index)
        for f in FEATURES:raw['dillon_'+f]=d[f].reindex(raw.index)
        raw['winter_start_year']=np.where(raw.index.month>=10,raw.index.year,raw.index.year-1)
        raw['season']=raw.winter_start_year.astype(str)+'/'+(raw.winter_start_year+1).astype(str)
        raw.to_csv(OUT/(name.lower().replace('-','')+'_dillon_pairs.csv'),float_format='%.8f',index_label='date')
        pairs[name]=raw
        valid=raw.loc[raw.reported_snow_inches.notna()&raw.reported_snow_inches.ge(0)]
        common=valid.dropna(subset=['dillon_same_date_inches','dillon_prior_date_inches'])
        if name=='Keystone':common=common.loc[common.index<'2026-06-01']
        summary={'rows':len(raw),'nonmissing_numeric_nonnegative_rows':len(valid),
                 'first_date':str(raw.index.min().date()),'last_date':str(raw.index.max().date()),
                 'calendar_pairs_with_dillon':len(valid.dropna(subset=['dillon_same_date_inches'])),
                 'common_same_and_prior_day_cohort':len(common),
                 'same_date_pearson_r':correlation(common,'reported_snow_inches','dillon_same_date_inches'),
                 'prior_date_pearson_r':correlation(common,'reported_snow_inches','dillon_prior_date_inches'),
                 'source_status':raw.measurement_status.iloc[0], 'winter_diagnostics':[]}
        for year,g in common.groupby('winter_start_year'):
            summary['winter_diagnostics'].append({'winter_start_year':int(year),'paired_days':len(g),
                'same_date_pearson_r':correlation(g,'reported_snow_inches','dillon_same_date_inches'),
                'prior_date_pearson_r':correlation(g,'reported_snow_inches','dillon_prior_date_inches'),
                'reported_snow_subtotal_inches':float(g.reported_snow_inches.sum()),
                'dillon_same_dates_subtotal_inches':float(g.dillon_same_date_inches.sum())})
        summaries[name]=summary
        allrows.append(raw[['resort','season','reported_snow_inches','measurement_status','dillon_same_date_inches','dillon_prior_date_inches']])
    pd.concat(allrows).reset_index().sort_values(['resort','date']).to_csv(OUT/'resort_daily_observations_with_dillon.csv',index=False,float_format='%.8f')

    # Fixed pilot configuration, without tuning or model selection on held-out data.
    # Each winter is held out in full. Source timing/availability is not verified.
    k=pairs['Keystone'].loc[lambda x:x.winter_start_year.isin([2024,2025])].copy()
    k=k.rename(columns={'dillon_'+f:f for f in FEATURES})
    phase=2*np.pi*(k.index.dayofyear-1)/365.2425
    for harmonic in [1,2]:
        k[f'sin{harmonic}']=np.sin(phase*harmonic); k[f'cos{harmonic}']=np.cos(phase*harmonic)
    cohort=k.dropna(subset=['reported_snow_inches']+FEATURES+CALENDAR).copy()
    cohort=cohort.loc[cohort.reported_snow_inches.ge(0)]
    assert len(cohort)==372 and set(cohort.winter_start_year)=={2024,2025}
    np.testing.assert_allclose(cohort.snow_lag1_inches,d.snowfall_inches.shift(1).reindex(cohort.index))
    forecast_rows=[]
    for year in [2024,2025]:
        train=cohort.loc[cohort.winter_start_year!=year];test=cohort.loc[cohort.winter_start_year==year]
        assert not set(train.index)&set(test.index)
        configs={'calendar_only':CALENDAR,'calendar_plus_prior_Dillon':CALENDAR+FEATURES}
        for name,features in configs.items():
            model=make_pipeline(StandardScaler(),Ridge(alpha=100.0)).fit(train[features],train.reported_snow_inches)
            prediction=np.maximum(0,model.predict(test[features]))
            for date,y,p in zip(test.index,test.reported_snow_inches,prediction):
                forecast_rows.append({'date':date,'held_out_winter':year,'model':name,'actual_inches':float(y),'predicted_inches':float(p)})
    pred=pd.DataFrame(forecast_rows)
    pred.to_csv(OUT/'keystone_exploratory_heldout_predictions.csv',index=False,float_format='%.8f')
    pilot={'target':'CAIC CAKSS reported Sno24 at exact source-clock 06:00',
           'evaluation':'Exploratory leave-one-winter-out, two winters only; not a verified operational hindcast',
           'n_days':len(cohort),'features':FEATURES,'calendar_features':CALENDAR,'ridge_alpha_fixed_without_tuning':100,
           'metrics':{name:metric(g.actual_inches,g.predicted_inches) for name,g in pred.groupby('model')},
           'winter_metrics':[{ 'held_out_winter':int(year),'model':name,**metric(g.actual_inches,g.predicted_inches)} for (year,name),g in pred.groupby(['held_out_winter','model'])],
           'caveats':['No same-date observed weather predictors. Only prior-date Dillon features and calendar enter the pilot.',
                      'Whole winters held out. Scaler fit on the training winter. Hyperparameters fixed, no tuning.',
                      'Two incomplete winters cannot establish stable forecast skill or an ENSO relationship.',
                      'Observation-window offsets, historical publication latency and station protocol remain unverified.',
                      'This experiment does not replace or update the earlier Dillon neural forecasts.',
                      'No A-Basin model trained because source quality checks failed.']}
    report={'prepared_date':'2026-11-04','dillon_observation_cutoff':'2026-11-01','sources':summaries,'keystone_pilot':pilot,
            'sources_unchanged_hashes':{f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['keystone_caic_daily_0600.csv','abasin_caic_daily_0600.csv','loveland_daily.csv']}}
    (OUT/'resort_daily_acquisition_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    plot(cohort,summaries['Keystone'],OUT)
    build_html(report)
    print(json.dumps({'sources':{n:{k:v for k,v in r.items() if k!='winter_diagnostics'} for n,r in summaries.items()},'pilot':pilot},indent=2))


def plot(k,summary,out):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharey=True)
    colors={2024:'#176b91',2025:'#dd8d26'}
    for ax,x,label,r in zip(axes,['dillon_same_date_inches','dillon_prior_date_inches'],
                           ['Dillon snowfall on the same reporting date','Dillon snowfall on the previous reporting date'],
                           [summary['same_date_pearson_r'],summary['prior_date_pearson_r']]):
        for year,g in k.groupby('winter_start_year'):
            ax.scatter(g[x],g.reported_snow_inches,s=28,alpha=.55,color=colors[year],label=f'{year}/{year+1}')
        ax.set_title(f'{label}\nPearson r = {r:.2f} · {len(k)} paired days',loc='left',fontsize=11)
        ax.set_xlabel('Dillon fresh snowfall (inches)');ax.grid(alpha=.15)
        ax.set_xlim(-.2,9);ax.set_ylim(-.3,15);ax.legend(frameon=False)
    axes[0].set_ylabel('Keystone CAKSS reported 24-hour snow (inches)')
    fig.suptitle('Dillon and Keystone: shared storms versus advance information',x=.07,ha='left',fontsize=16,fontweight='bold')
    fig.text(.07,.095,'Keystone mountain snow-study station · two incomplete winters · exact 06:00 source-clock samples\n'
             'Same-date association is not forecast skill. Reporting-window alignment and the measurement protocol remain unverified.',
             fontsize=9,color='#516576',linespacing=1.6)
    fig.subplots_adjust(left=.07,right=.98,top=.8,bottom=.24,wspace=.15)
    for suffix in ['png','svg']:fig.savefig(out/f'keystone_dillon_relationship.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)


def build_html(r):
    def table(head,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
    src=r['sources'];p=r['keystone_pilot'];k=src['Keystone']
    rows=[[n, s['first_date']+' to '+s['last_date'],s['rows'],s['nonmissing_numeric_nonnegative_rows'],s['calendar_pairs_with_dillon'],s['source_status']] for n,s in src.items()]
    metrics=[[n.replace('_',' '),f"{m['rmse_inches']:.3f}",f"{m['mae_inches']:.3f}"] for n,m in p['metrics'].items()]
    winters=[[str(w['held_out_winter'])+'/'+str(w['held_out_winter']+1),w['model'].replace('_',' '),w['n'],f"{w['rmse_inches']:.3f}"] for w in p['winter_metrics']]
    links=[('Keystone + Dillon daily data','keystone_dillon_pairs.csv'),('A-Basin candidate data + Dillon','abasin_dillon_pairs.csv'),('Loveland + Dillon daily data','loveland_dillon_pairs.csv'),('Combined observations','resort_daily_observations_with_dillon.csv'),('Pilot held-out predictions','keystone_exploratory_heldout_predictions.csv'),('Analysis details','resort_daily_acquisition_summary.json')]
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Resort daily snowfall data</title>
    <style>body{{font-family:system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 22px;color:#203545;line-height:1.6}}h1{{font-size:32px}}h2{{margin-top:38px}}table{{width:100%;border-collapse:collapse;margin:20px 0;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dbe3e8;vertical-align:top}}th{{background:#edf3f6}}a{{color:#176b91}}.note{{background:#fff3d9;padding:16px 20px;border-left:4px solid #d8a63b}}img{{max-width:100%}}.muted{{color:#607483}}code{{background:#edf3f6;padding:1px 4px}}</style>
    <h1>Daily snowfall data from the mountains</h1><p class="muted">Acquired November 4, 2026. Paired against the frozen Dillon observations through November 1.</p>
    <p><strong>Keystone has a usable candidate for an initial relationship study.</strong> The CAIC archive provides observations at the VailResort-operated CAKSS snow-study station (11,761 ft), rather than a nearby community station. A-Basin records are available too, but fail important quality checks. Loveland supplies a separate, longer official resort log.</p>
    {table(['Source','Extracted reporting dates','Daily rows','Numeric values','Paired Dillon dates','Interpretation'],rows)}
    <p>Date ranges are not complete daily coverage. A-Basin extraction covers October–May in 2020/21–2025/26. Keystone has 189 readings in 2024/25, 183 in 2025/26 and two in November 2026. Loveland has 10 conflicting dates and six nonnumeric dates kept unavailable.</p>
    <h2>What the records measure</h2><p>CAIC currently labels <code>Sno24</code> “24-hour snowfall (in)” separately from snow height and changes in snow height. We kept the reported field itself, selected exactly one 06:00 source-clock reading per day, and did not calculate snowfall by differencing depth or summing hourly rolling totals. The instrument, clearing schedule, historical timezone convention and equivalence to the resorts’ published manual snow reports remain unverified.</p>
    <p><a href="https://stations.avalanche.state.co.us/precip.php">CAIC variable definitions and station links</a> · <a href="https://skiloveland.com/snowfall-history-archive/">Loveland official archive</a> · <a href="https://skiloveland.com/snowfall-history/">Loveland recent daily log</a></p>
    <div class="note"><strong>A-Basin is quarantined from model training.</strong> Its candidate archive contains 40 inches on April 18, 2022 and 36 inches on October 4, 2025 while reporting zero snow depth. Several winter subtotals are unusually low; 2022/23 has no zero Sno24 readings among its nonmissing daily values. These are diagnostic flags, not proof that every observation is wrong. No arbitrary correction was applied.</div>
    <h2>Dillon and Keystone share a measurable signal</h2><p>On the same common cohort of {k['common_same_and_prior_day_cohort']} days, same-reporting-date correlation is <strong>{k['same_date_pearson_r']:.3f}</strong>. Using Dillon’s previous date gives <strong>{k['prior_date_pearson_r']:.3f}</strong>. The previous-date relationship differs by winter, so proximity and same-storm association alone do not establish forecasting skill.</p>
    <img src="keystone_dillon_relationship.png" alt="Same-date and previous-date Dillon versus Keystone snowfall">
    <p>For Loveland, the official-log same-date association is a separate regional-data result, not validation for Keystone or A-Basin. All correlations are descriptive, include zero-snow dates, and remain subject to reporting-window and source-quality limitations.</p>
    <h2>Small forecast feasibility test</h2><p>Two ridge models used the same held-out winters and the same 372 target days. Both use calendar cycles; one also uses earlier Dillon snowfall, precipitation, snow depth and temperature. Scaling is fitted on the training winter only, penalty is fixed at 100, and predictions are constrained to nonnegative values. No hyperparameter search or same-date observed weather enters this experiment.</p>
    {table(['Model','Daily RMSE (in)','Daily MAE (in)'],metrics)}
    {table(['Held-out winter','Model','Days','Daily RMSE (in)'],winters)}
    <p><strong>This is an exploratory date-indexed test, not an operational hindcast.</strong> Two incomplete winters do not establish reliable generalization. Historical publication times and exact observation windows are unknown. These models do not update the earlier neural forecasts or issue a new resort outlook.</p>
    <h2>Next model</h2><p>First confirm the Keystone measurement protocol and align its daily window with Dillon. Then use resort history plus atmospheric forecast precipitation, temperature, humidity and wind available at the issue time. The longer Loveland record can support regional training, with resort-specific calibration. Validate by whole winters and storm blocks. A-Basin needs corrected or independently verified snowfall labels before it joins that training set.</p>
    <h2>Downloads</h2><ul>{''.join(f'<li><a href="{url}">{label}</a></li>' for label,url in links)}</ul>
    <p class="muted">Missing dates are never treated as zero. Partial sums are not season totals. No future observation is an allowable forecast input. No accounts, paid data or provider contact were used.</p></html>'''
    (OUT/'resort_daily_data_report.html').write_text(body)


if __name__=='__main__':main()
