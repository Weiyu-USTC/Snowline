"""Predeclared clock / horizon and NOAA presumed-zero sensitivity checks."""
from pathlib import Path
import argparse,json
import pandas as pd
from threadpoolctl import threadpool_limits
import nwp_benchmark as b
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/nwp-weather'


def clock(route,site):
    if route=='mos':
        inputs={'main':'mos_daily_features.csv','minus':'mos_daily_features_clock_minus.csv','plus':'mos_daily_features_clock_plus.csv'};folder=ROOT/'data/nwp_archive_probe'
    else:
        inputs={'main':'jma_benchmark_main.csv','minus':'jma_benchmark_minus1h.csv','plus':'jma_benchmark_plus1h.csv'};folder=ROOT/'data/nwp_recent'
    prepared={};history=None
    for name,filename in inputs.items():
        d=pd.read_csv(folder/filename);d=d[d.site==site]
        if route=='mos':d=d[d.lead_day==1] # Mansfield+6h cannot fit a2ndfull24hperiod inside72harchive.
        d,h=b.attach_targets_for_training(d,ROOT);history=h[site];d=b.validate_join(d);d=d[d.target_inches.notna()].copy()
        # Source runtime/target-date/ordinal lead are unchanged across MOSclock
        # hypotheses. JMA issue shifts withclock, so match report-date/lead only.
        key=['site','target_date','lead_day']+(['forecast_reference_utc'] if route=='mos' else [])
        d['_common_key']=d[key].astype(str).agg('|'.join,axis=1)
        if d._common_key.duplicated().any():raise ValueError('Duplicate sensitivity comparison key')
        prepared[name]=d
    common=set.intersection(*(set(d._common_key) for d in prepared.values()))
    root=OUT/f'{route}-clock'/site;root.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for name,d in prepared.items():
        part=d[d._common_key.isin(common)].copy()
        results=b.evaluate(part,root/name,history,model_names=[b.NAMES[-1]])
        for split,r in zip(['crossvalidation','chronological'],results):
            metrics=b.score_predictions(r);metrics['alignment']=name;metrics['split']=split;summaries.append(metrics)
    pd.concat(summaries,ignore_index=True).to_csv(root/'clock_sensitivity_metrics.csv',index=False)
    (root/'clock_plan.json').write_text(json.dumps(dict(site=site,route=route,common_rows=len(common),models=[b.NAMES[0],b.NAMES[-1]],matched_by=key,primary_clock='main',selection='Offsets predeclared; no score-based selection of clock. MOSsensitivityusesordinalday1only forcommonhorizon.'),indent=2))


def flag_sensitivity():
    rows=[];comparisons=[]
    for site in ['mansfield','jay']:
        root=OUT/'mos'/site
        for split in ['crossvalidation','chronological']:
            d=pd.read_csv(root/f'{split}_predictions.csv')
            if 'presumed_zero' not in d:raise ValueError('No sourcePflag retained')
            strict=d[~d.presumed_zero.astype(bool)]
            s=b.score_predictions(strict);s['site']=site;s['split']=split;s['policy']='Exclude presumed-zero evaluation labels only; models unchanged';rows.append(s)
            s=b.paired_comparisons(strict);s['site']=site;s['split']=split;comparisons.append(s)
    pd.concat(rows,ignore_index=True).to_csv(OUT/'noaa_flag_sensitivity_metrics.csv',index=False)
    pd.concat(comparisons,ignore_index=True).to_csv(OUT/'noaa_flag_sensitivity_paired.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--route',choices=['mos','jma']);p.add_argument('--site',choices=['alta','vail','mansfield','jay']);p.add_argument('--flags',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):
        if a.flags:flag_sensitivity()
        else:clock(a.route,a.site)
if __name__=='__main__':main()
