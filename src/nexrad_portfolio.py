"""Validation-only DPR/HRRR portfolio weights and honest paired test comparison."""
import json
import numpy as np
import pandas as pd
from nexrad_acquire import ROOT,DATA,utc,iso
from nexrad_train import OUT,metric

WEIGHTS=[0.,.25,.5,.75,1.]

def main():
 select=json.loads((OUT/'selection.json').read_text());cases=pd.read_csv(OUT/'case_predictions.csv',dtype={'case_id':str});targets=pd.read_csv(DATA/'mrms_targets.csv',dtype={'case_id':str})
 plan={'radar_weights':WEIGHTS,'selection':'Validation-only mean of1h RMSE and3h RMSE/3; same scalar weight for3 hours. NEXRAD candidate and Vail radar choice already frozen by validation. No test tuning.','HRRR_window':'06UTC run; forecast hours6,7,8 are amounts ending12,13,14UTC, covering11–14UTC. Radar issue10:50UTC precedes first complete target hour by10minutes. HRRR assumed available09UTC.'}
 (OUT/'portfolio_plan.json').write_text(json.dumps(plan,indent=2))
 selections={};metrics=[];prows=[]
 for key,choice in select.items():
  site='vail' if key.startswith('vail') else key
  df=cases[(cases.site==key)&(cases.model==choice['model'])].copy();weather=[];keep=[]
  for i,r in df.iterrows():
   f=ROOT/'data/hrrr_portfolio/runs'/(r.case_id+'T0600Z')/'hourly.csv'
   if not f.exists():continue
   w=pd.read_csv(f);w=w[(w.site==site)&w.forecast_hour.isin([6,7,8])].sort_values('forecast_hour')
   if len(w)!=3 or not np.isfinite(w.precip_mm).all():continue
   run=pd.to_datetime(w.run_time_utc,utc=True);vt=pd.to_datetime(w.valid_time_utc,utc=True)
   if not (vt==run+pd.to_timedelta(w.forecast_hour,unit='h')).all():raise ValueError('HRRR target-time mismatch')
   if (pd.to_datetime(w.assumed_available_time_utc,utc=True)>pd.Timestamp(r.case_id+'T10:50:00Z')).any():raise ValueError('HRRR future run leakage')
   keep.append(i);weather.append(np.maximum(0,w.precip_mm.to_numpy(float)))
  df=df.loc[keep];hrrr=np.array(weather);native=df[['pred1','pred2','pred3']].to_numpy();truth=df[['truth1','truth2','truth3']].to_numpy();va=df.split.to_numpy()=='validation';te=df.split.to_numpy()=='test'
  if not len(df) or min(va.sum(),te.sum())<5:continue
  scores={w:(metric(truth[va],(w*native+(1-w)*hrrr)[va])['RMSE_1h_mm']+metric(truth[va],(w*native+(1-w)*hrrr)[va])['RMSE_3h_mm']/3)/2 for w in WEIGHTS};weight=min(scores,key=scores.get)
  pred=weight*native+(1-weight)*hrrr
  selections[key]=dict(radar_weight=weight,hrrr_weight=1-weight,native_model=choice['model'],validation_score=scores[weight],validation_n=int(va.sum()),test_n=int(te.sum()),validation_scores={str(k):v for k,v in scores.items()},validation_error_p80_hourly_mm=np.quantile(abs(pred[va]-truth[va]),.8,axis=0).tolist())
  pmap={'NEXRAD_selected':native,'HRRR':hrrr,'NEXRAD_HRRR_portfolio':pred}
  # Same cases and target definition for every comparator.
  for name in ['zero','train_mean','DPR_persistence','DPR_advection']:
   p=cases[(cases.site==key)&(cases.model==name)].set_index('case_id').reindex(df.case_id)[['pred1','pred2','pred3']].to_numpy();pmap[name]=p
  if site=='alta':
   t=targets[targets.site==site].set_index('case_id').reindex(df.case_id)
   for name in ['convlstm','dgmr','nowcastnet']:
    cols=[f'previous_{name}_{h}' for h in [1,2,3]]
    if all(c in t for c in cols):pmap['previous_MRMS_'+name]=t[cols].to_numpy()
  for name,p in pmap.items():
   for split in ['validation','test']:
    mask=(df.split.to_numpy()==split)&np.isfinite(p).all(1)
    if mask.any():metrics.append(dict(site=key,model=name,split=split,**metric(truth[mask],p[mask])))
   for j,(_,r) in enumerate(df.iterrows()):
    if np.isfinite(p[j]).all():prows.append(dict(case_id=r.case_id,site=key,model=name,split=r.split,**{f'pred{k+1}':float(p[j,k]) for k in range(3)},**{f'truth{k+1}':float(truth[j,k]) for k in range(3)}))
 (OUT/'portfolio_selection.json').write_text(json.dumps(selections,indent=2));pd.DataFrame(metrics).to_csv(OUT/'paired_metrics.csv',index=False);pd.DataFrame(prows).to_csv(OUT/'paired_predictions.csv',index=False)
 print(json.dumps(selections,indent=2))

def apply_current():
 path=OUT/'current_forecast.json';result=json.loads(path.read_text());select=json.loads((OUT/'portfolio_selection.json').read_text());weather=pd.read_csv(ROOT/'data/hrrr_portfolio/current_weather.csv');weather['end']=pd.to_datetime(weather.valid_time_utc,utc=True)
 for site,r in result['sites'].items():
  if not r.get('available') or r['model_site_key'] not in select:continue
  s=select[r['model_site_key']];native=r.get('native_nexrad_hourly',r['hourly']);combined=[]
  for h,n in enumerate(native):
   w=weather[(weather.site==site)&(weather.end==pd.Timestamp(n['end']))]
   w=w[pd.to_datetime(w.assumed_available_time_utc,utc=True)<=pd.Timestamp(result['issued_at'])]
   if len(w)!=1:break
   amount=s['radar_weight']*n['liquid_mm']+s['hrrr_weight']*max(0,float(w.iloc[0].precip_mm));spread=s['validation_error_p80_hourly_mm'][h]
   combined.append(dict(start=n['start'],end=n['end'],liquid_mm=amount,liquid_lower_mm=max(0,amount-spread),liquid_upper_mm=amount+spread,nexrad_liquid_mm=n['liquid_mm'],hrrr_liquid_mm=float(w.iloc[0].precip_mm),hrrr_run_time_utc=w.iloc[0].run_time_utc))
  if len(combined)==3:
   r['native_nexrad_hourly']=native;r['hourly']=combined;r['model']='NEXRAD_HRRR_portfolio';r['portfolio']=s
 result['source']='Native NEXRAD LevelIII DPR and NOAA HRRR, validation-selected portfolio weights'
 path.write_text(json.dumps(result,indent=2));print('current portfolio updated')
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--current',action='store_true');a=ap.parse_args();apply_current() if a.current else main()
