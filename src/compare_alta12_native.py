"""Native HRRR ASNOW comparison on exactly matched Alta12h intervals."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from train_alta12 import ROOT,DATA,OUT,metrics,block_bootstrap

def main():
    path=ROOT/'data/hrrr_portfolio/historical_native_endpoints.csv.gz'
    h=pd.read_csv(path);h=h[h.site.eq('alta')].copy()
    h['run_time_utc']=pd.to_datetime(h.run_time_utc,utc=True)
    p=h.pivot_table(index='run_time_utc',columns='forecast_hour',values='snowfall_cumulative_inches',aggfunc='first')
    meta=pd.read_csv(OUT/'case_index.csv');pred=dict(np.load(OUT/'predictions_all.npz'));target=np.load(DATA/'arrays.npz')['snow']
    manifest=json.loads((OUT/'manifest.json').read_text());selected=manifest['selection']['modern']['name']
    operational=dict(np.load(OUT/'predictions_missing_history.npz'))
    opname=manifest['selection']['modern_missing_history']['name']
    pred[opname]=operational[opname]
    rows=[];tiny_negative=[]
    for r in meta[meta.modern].itertuples():
        run=pd.Timestamp(r.hrrr_run_utc)
        if run not in p.index:continue
        lo,hi=(5,17) if r.half==0 else (17,29)
        values=p.loc[run]
        if not np.isfinite(values.get(lo,np.nan)) or not np.isfinite(values.get(hi,np.nan)):continue
        amount=float(values[hi]-values[lo])
        if amount<-.002:raise ValueError('Native cumulative ASNOW decreases by more than0.002in tolerance')
        if amount<0:
            tiny_negative.append({'run_time_utc':str(run),'half':r.half,'increment_inches':amount})
            continue
        rows.append({'row':r.row,'start_utc':r.start_utc,'end_utc':r.end_utc,'run_time_utc':run.isoformat(),'half':r.half,'stage':r.validation_stage,'block':r.seven_day_block,'actual_inches':float(target[r.row]),'native_asnow_inches':max(0,amount)})
    d=pd.DataFrame(rows);train=d[d.stage.eq('train')]
    ratio=float(np.sum(train.native_asnow_inches*train.actual_inches)/max(1e-8,np.sum(train.native_asnow_inches**2)))
    d['native_asnow_train_calibrated']=d.native_asnow_inches*max(0,ratio)
    for k in [selected,opname,'modern_forest','modern_tcn','modern_radar_tcn','broad_forest','broad_tcn','modern_climatology','hrrr_fixed_10to1','hrrr_train_calibrated']:
        d[k]=pred[k][d.row]
    d.to_csv(OUT/'native_asnow_predictions.csv',index=False)
    methods=[c for c in d if c in pred or c.startswith('native_asnow')];records=[]
    for stage,g in d.groupby('stage'):
        for name in methods:records.append({'stage':stage,'method':name,**metrics(g.actual_inches.to_numpy(),g[name].to_numpy())})
    pd.DataFrame(records).to_csv(OUT/'native_asnow_metrics.csv',index=False)
    test=d[d.stage.eq('test')]
    details={'target':'Actual Alta Collins manual12h snow, same observations for every method',
        'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'definition':'HRRR06UTC cumulative ASNOW(f17)-ASNOW(f5) for11–23UTC; ASNOW(f29)-ASNOW(f17) for23–11UTC. Native variable-density model snowfall, parameter(0,1,29), meters converted to inches upstream.',
        'selection':'Frozen calendar every fourth day; not selected by storms or observed snowfall. Primary models are unchanged; ASNOW-calibration multiplicative factor uses training intervals only.',
        'counts':d.groupby('stage').size().to_dict(),'native_training_multiplier':ratio,
        'small_negative_policy':'Tiny non-monotonic archive endpoint differences are excluded from every method on the common subset. They are consistent with packing jitter, but the cause is not proven. Any decrease exceeding0.002in raises an error.',
        'small_negative_increments':tiny_negative,
        'selected_model_vs_native_rmse_difference_95ci':block_bootstrap(test.actual_inches.to_numpy(),test[selected].to_numpy(),test.native_asnow_inches.to_numpy(),test.block.to_numpy()),
        'operational_model_vs_native_rmse_difference_95ci':block_bootstrap(test.actual_inches.to_numpy(),test[opname].to_numpy(),test.native_asnow_inches.to_numpy(),test.block.to_numpy())}
    (OUT/'native_asnow_audit.json').write_text(json.dumps(details,indent=2))
    print(json.dumps(details,indent=2));print(pd.DataFrame(records).query("stage=='test'").sort_values('rmse_inches').to_string(index=False))
if __name__=='__main__':main()
