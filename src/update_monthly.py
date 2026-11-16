"""Refit the validated selected configuration using an observed season prefix."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from monthly import ROOT,MONTHS,load_training,fit_predict,make_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observed',required=True,help='Comma-separated actual inches in order: October,November,...')
    p.add_argument('--roni',type=float,required=True,help='Latest safely available lagged RONI at forecast issuance, °C')
    p.add_argument('--family',choices=['selected','climatology','ridge','analog','extra_trees','neural_mlp','tabpfn'],default='selected',help='Use inner-CV selection or a named experimental family; TabPFN requires its optional environment')
    args=p.parse_args()
    try: values=[float(v.strip()) for v in args.observed.split(',')]
    except ValueError: p.error('Observed amounts must be numbers')
    if not 1<=len(values)<=6 or not np.isfinite(values).all() or min(values)<0:
        p.error('Supply 1–6 consecutive observed months, finite and nonnegative')
    if not np.isfinite(args.roni): p.error('RONI must be finite')
    origin=len(values)-1
    results=json.loads((ROOT/'outputs/monthly_copper_results.json').read_text())
    settings=results['final_models'][str(origin)]
    if args.family=='selected': config=settings['selection']['config']
    elif args.family=='tabpfn': config={'family':'tabpfn_v2','enso':True,'n_estimators':2}
    else: config=settings['family_choices'][args.family]['config']
    frame=load_training()
    row={**dict(zip(MONTHS,values)),f'enso_{origin}':args.roni}
    with threadpool_limits(limits=1):
        if args.family=='tabpfn':
            from tabpfn_monthly import predict_tabpfn
            X,_=make_features(frame,origin,True)
            query,_=make_features(pd.DataFrame([row]),origin,True)
            forecast=np.array([max(0.,float(predict_tabpfn(X,frame[m].to_numpy(float),query)[0])) for m in MONTHS[origin+1:]])
        else: forecast=fit_predict(frame,pd.DataFrame([row]),origin,config)[0]
    out={'observed_through':MONTHS[origin],'observed_inches':dict(zip(MONTHS,values)),
         'lagged_roni':args.roni,'selected_config':config,
         'remaining_monthly_mean_inches':dict(zip(MONTHS[origin+1:],map(float,forecast))),
         'remaining_total_inches':float(forecast.sum()),
         'status':'Conditional research estimate from supplied observations; supplied values are not verified by this command.',
         'interval_note':'This update emits point estimates only. Saved November scenarios have separate omitted-season uncertainty bands.'}
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':main()
