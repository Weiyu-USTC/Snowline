"""Freeze and prepare the Alta radar experiment without selecting on outcomes."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/alta-nowcast'
PLAN = {
    'created_utc': '2026-11-06T03:55:00Z',
    'status': 'Frozen before training and evaluation',
    'seed': 20261105,
    'cohort': 'Every calendar day November1–April30 in winters2020/21–2023/24; no snowfall/event selection.',
    'amendment': 'Initial alternate-day plan expanded to all days after observing faster public-data transfer, before fitting or examining forecast errors.',
    'training_winters': [2020, 2021], 'development_winter': 2022, 'test_winter': 2023,
    'issue_time': '10:50UTC; radar input ends10:40UTC (assumed10min availability); HRRR06UTC forecast assumed available08UTC.',
    'input': '9 radar frames09:20–10:40UTC,10min cadence; rate plus causal quality channel.',
    'output': '20 rate frames10:50–14:00UTC; frames11:10–14:00 integrated for1/2/3h windows beginning11UTC.',
    'target': 'MRMS liquid-equivalent precipitation, not independently measured hourly fresh-snow depth.',
    'daily_target': 'Alta-Collins manual fresh snow11UTC–next11UTC; >=5in defines powder day.',
    'grid': '64x64 cells from4x4 pooling of0.01degree MRMS (~3.4x4.4km atAlta); region about220x285km.',
    'quality': 'Input valid fraction>=0.9 andRQI>=0.1; target valid fraction>=0.9 andRQI>=0.5; future quality is verification only. Four outer cells excluded from spatial metrics.',
    'case_exclusion': 'Missing frames/quality or <10% valid verification pixels; missing daily snow or forecasts excluded only from corresponding matched evaluation.',
    'neural_models': ['ConvLSTM compact', 'DGMR compact adaptation and horizon extension', 'NowcastNet compact independent adaptation'],
    'training': {'steps': 1500, 'batch_size': 2, 'base_channels': 8, 'learning_rate': 0.0003, 'ensemble_members': 8,
                 'selection': 'Lowest quality-masked development MSE; fixed every-fourth development case for checkpoint selection, never test.',
                 'checkpoint_frequency_steps': 150},
    'daily_calibration': 'Model-specific empirical forests trained on development-winter predictions and manual snowfall; matched HRRR-only/control models use identical development cases. Prior MOS model refitted using only winters<=2022.',
    'primary_metrics': ['1h/3h radar accumulation MAE,RMSE,CRPS', '24h fresh-snowMAE,RMSE,CRPS', '>=5in24h Brier score and skill relative to currentMOS model'],
    'secondary_metrics': ['1h/3h event precision,recall,CSI at1mm liquid accumulation', 'Neighborhood fractions skill at1mm andapproximately20km', 'Probability reliability and80% interval coverage'],
    'uncertainty': 'Paired seven-day block bootstrap within the held-out winter; cannot establish between-winter generalization.',
    'model_identity': 'New local compact implementations, no published pretrained checkpoints or full-paper skill claims.',
    'current_curves': 'Fresh current radar+latestavailableHRRR; rolling24h snowfall calibration transfers a model trained on11UTC reporting windows. Hourly snowfall allocation is provisional and has no independent hourly depth verification.'
}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p=OUT/'frozen_plan.json'
    if not p.exists(): p.write_text(json.dumps(PLAN, indent=2))


def prepare():
    freeze()
    src=ROOT/'data/alta_nowcast/radar/radar_sequences.npz'
    z=np.load(src)
    rate=z['rate'].astype('float32');valid=z['valid_fraction'];rqi=z['rqi_hourly']
    times=z['radar_time'].astype('datetime64[s]')
    # All historical windows have identical UTC minute offsets, but build the
    # match from timestamps so quality can never be selected from a future hour.
    qt=z['quality_time'].astype('datetime64[s]')
    hour_index=((times-times[:, :1])/np.timedelta64(1,'m')+20).astype(int)//60
    q=np.take_along_axis(rqi,hour_index[:,:,None,None],axis=1)
    assert np.all(qt[np.arange(len(rate))[:,None],hour_index]<=times)
    inp=rate[:,:9].copy();iq=q[:,:9].copy()
    imask=(valid[:,:9]>=.9)&np.isfinite(iq)&(iq>=.1)
    inp[~imask]=np.nan;iq[~imask]=0
    x=np.stack([inp,iq],axis=2)
    target=rate[:,9:].copy()
    tmask=(valid[:,9:]>=.9)&np.isfinite(q[:,9:])&(q[:,9:]>=.5)
    tmask[:,:,:4,:]=False;tmask[:,:,-4:,:]=False
    tmask[:,:,:,:4]=False;tmask[:,:,:,-4:]=False
    target[~tmask]=np.nan
    complete=z['frame_present'].all(1)&z['quality_present'].all(1)
    eligible=complete&(tmask.mean((1,2,3))>=.1)
    ids=z['case_id'].astype(str)
    frame=pd.DataFrame({'case_id':ids,'date':pd.to_datetime(ids,format='%Y%m%d'),
                        'winter':z['winter'],'split':z['split'], 'radar_eligible':eligible,
                        'target_valid_fraction':tmask.mean((1,2,3)), 'complete_frames':complete})
    frame['target_date']=frame.date+pd.Timedelta(days=1)
    snow=pd.read_csv(ROOT/'data/east_alta_research/alta_hive/alta_collins_daily_paired.csv')
    snow['target_date']=pd.to_datetime(snow.reporting_endpoint_lst).dt.normalize()
    snow=snow[snow.complete_24h_pair.astype(bool)]
    frame=frame.merge(snow[['target_date','snowfall_24h_inches','lpe_24h_inches']],on='target_date',how='left',validate='one_to_one')
    frame.to_csv(OUT/'case_inventory.csv',index=False)
    np.savez_compressed(OUT/'learning_arrays.npz',x=x,y=target,case_id=ids,split=z['split'],winter=z['winter'],eligible=eligible,
                        latitude=z['latitude'],longitude=z['longitude'],radar_time=times,
                        rate_alta_native=z['rate_alta_native'],rqi_alta_native=z['rqi_alta_native'])
    audit={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'shape_x':list(x.shape),'shape_y':list(target.shape),
           'eligible_counts':frame[eligible].groupby('split').size().to_dict(),
           'excluded_missing_or_quality':int((~eligible).sum()),
           'missing_manual_targets':int(frame.snowfall_24h_inches.isna().sum()),
           'future_quality_in_inputs':False,'interpolated_daily_snow_to_hourly':False}
    (OUT/'preparation_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    freeze() if a.freeze_only else prepare()
