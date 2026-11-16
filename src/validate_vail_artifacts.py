"""Verify published Vail curves against the fitted checkpoint and frozen inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import vail_hourly_model as v


def main():
    v.seed_everything(v.SEED)
    out=v.OUT
    checkpoint=torch.load(out/'vail_hourly_tcn.pt',weights_only=True)
    _,frame,target,years=v.load_data()
    manifest=json.loads((out/'forecast_manifest.json').read_text())
    assert hashlib.sha256((v.ROOT/'data/vail_hourly/hourly.csv').read_bytes()).hexdigest()==manifest['source_sha256']
    scaler=v.HourlyScaler()
    for name in ['mean','std','future_min','future_max']:
        setattr(scaler,name,np.array(checkpoint['scaler'][name]))
    model=v.VailTCN(2*len(checkpoint['features'])+2,checkpoint['config']['width'],checkpoint['config']['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    origin=pd.Timestamp(manifest['issue_endpoint_source_clock'])
    x=frame.reindex(pd.date_range(end=origin,periods=168,freq='h')).to_numpy('float32')[None]
    p=v.predict(model,scaler,x,v.future_features(origin)[None])[0]
    forecast=pd.read_csv(out/'forecast_daily.csv')
    error=float(np.max(np.abs(v.distribution_mean(p)[:len(forecast)]-forecast.mean_inches.to_numpy())))
    assert error<1e-5,error
    assert len(forecast)==57 and forecast.date.iloc[0]=='2026-11-05' and forecast.date.iloc[-1]=='2026-12-31'
    assert (forecast.q10_inches<=forecast.median_inches).all() and (forecast.median_inches<=forecast.q90_inches).all()
    assert np.isfinite(forecast.select_dtypes('number')).all().all()
    predictions=pd.read_csv(out/'crossvalidation_predictions.csv')
    indices=[set(map(tuple,r[['winter','origin','lead']].to_numpy())) for _,r in predictions.groupby('model')]
    assert all(i==indices[0] for i in indices)
    assert not pd.to_datetime(predictions.date).between('2020-03-20','2020-05-31').any()
    assert not pd.to_datetime(predictions.date).between('2023-10-01','2023-11-28').any()
    assert not (out/'forecast_totals.csv').exists()
    used=scaler.transform_future(v.future_features(origin)[None])
    assert np.isclose(used[0,0,6],manifest['roni_used_by_model'])
    assert used[0,0,7]==0, 'The unobserved strong-positive indicator must not activate arbitrary weights'
    summary={'matched_origin_lead_pairs_per_model':len(indices[0]),'unique_target_dates':predictions.date.nunique(),
             'forecast_endpoints':len(forecast),'checkpoint_max_mean_difference_inches':error,
             'source_hash_matches':True,'quarantine_target_checks_passed':True,
             'future_covariate_guard_restored':True,'roni_observed':manifest['roni_at_origin'],
             'roni_used':manifest['roni_used_by_model'],
             'nominal_interval':'10–90% model-based, not operationally calibrated'}
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))


if __name__=='__main__':main()
