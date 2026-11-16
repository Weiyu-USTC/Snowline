"""Assemble clearly labeled current-season forecasts and a portable CSV."""
import json
from pathlib import Path
import pandas as pd
from monthly import ROOT,MONTHS,load_training,jackknife_bounds


def main():
    out=ROOT/'outputs'
    frame=load_training('copper')
    predictions=pd.read_csv(out/'monthly_copper_predictions.csv')
    base=predictions.loc[(predictions.family=='climatology')&(predictions.origin==0)]
    folded=[]
    for _,group in base.groupby('start_year'):
        group=group.set_index('target_month').loc[MONTHS[1:]]
        values=group.prediction.to_list()
        folded.append({'prediction':values,'held_prediction':values,'held_actual':group.actual.to_list()})
    baseline={'point_inches':frame[MONTHS[1:]].mean().to_dict(),
              'nov_apr_total_inches':float(frame[MONTHS[1:]].sum(axis=1).mean()),
              **jackknife_bounds(folded,MONTHS[1:])}
    enso=next(iter(json.loads((out/'monthly_copper_missing_october_results.json').read_text())['forecasts'].values()))
    regional=next(iter(json.loads((out/'monthly_copper_regional_october_results.json').read_text())['forecasts'].values()))
    tab=next(iter(json.loads((out/'monthly_tabpfn_results.json').read_text())['experiments']['october_unverified_enso_only']['forecasts'].values()))
    source=json.loads((ROOT/'data/current_2026_inputs.json').read_text())
    summary={'as_of':'2026-11-03','forecast_season':'2026/2027','units':'inches of new snowfall',
             'recommended':'historical_monthly_mean',
             'recommendation_reason':'No evaluated predictor-based procedure demonstrated lower held-out error than monthly climatology; this is a conservative research default, not verified operational skill.',
             'copper_october_observed':None,'copper_october_status':source['copper_october']['status'],
             'regional_dillon_october_observed':source['dillon_october']['value_inches'],
             'lagged_roni':1.7,'lagged_roni_period':'JAS2026',
             'climate_extrapolation':'JAS RONI1.7 exceeds Copper training maximum1.5; only one training season meets JAS RONI>=1.5.',
             'forecasts':{'historical_monthly_mean':baseline,'experimental_enso_only':enso,
                          'experimental_regional_dillon_october':regional,'experimental_tabpfn_enso_only':tab}}
    (out/'forecast_2026_27.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    rows=[]
    for month in MONTHS[1:]:
        row={'month':month,'year':2026 if month in ['nov','dec'] else 2027,
             'recommended_mean_inches':baseline['point_inches'][month],
             'recommended_range_low':baseline['monthly_lower'][month],
             'recommended_range_high':baseline['monthly_upper'][month]}
        for key,forecast in summary['forecasts'].items():
            if key!='historical_monthly_mean':row[key+'_inches']=forecast['point_inches'][month]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'forecast_2026_27.csv',index=False)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__=='__main__':main()
