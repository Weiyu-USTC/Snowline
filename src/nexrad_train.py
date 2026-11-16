"""Train-only native DPR calibration, frozen winter validation and held-out audit.

Verification is quality-screened MRMS liquid analysis, which shares NEXRAD input
and is not independent hourly snowfall truth. Current output remains liquid mm.
"""
from __future__ import annotations
import argparse, json, pickle, warnings
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.signal import fftconvolve
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from nexrad_acquire import ROOT,DATA,SITES,PIXEL_KM,utc,iso,validate_crop
OUT=ROOT/'outputs/nexrad-short';OUT.mkdir(parents=True,exist_ok=True)
PLAN={'training_winters':[2020,2021],'selection_winter':2022,'held_out_from_current_fit_test_winter':2023,
 'prior_test_exposure':'2023/24 has been inspected in previous project work; it is held out from the current fits and weight selection, not a wholly unseen project dataset.',
 'candidate_models':['zero','train_mean','DPR_persistence','DPR_advection','ridge_alpha1','ridge_alpha10','extra_trees_leaf5','extra_trees_leaf10','MLP_16_8_alpha1','MLP_16_8_alpha10'],
 'selection':'Mean of validation 1h RMSE and 3h RMSE/3, units mm/hour; radar choice for Vail uses the same validation score. Test is excluded from selection.',
 'targets':'Hourly local3x3 MRMS grid mean rate integrated with10min trapezoidal rule; gridcells require source_valid>=0.9, RQI>=0.5; at least5/9 validcells at every endpoint. Shared NEXRAD provenance, not independent gauge truth.',
 'inputs':'Three native NEXRAD DPR scans plus calendar phase only. No future radar or weather observations in inputs.',
 'future_revision':'Any result-driven architecture change requires another independent test winter.'}

def motion(old,new,dt):
 a=gaussian_filter(np.nan_to_num(np.log1p(old)),1);b=gaussian_filter(np.nan_to_num(np.log1p(new)),1)
 if a.std()<.003 or b.std()<.003:return np.zeros(2),0.
 # Remove broad spatial background, taper edges, and bound apparent speed at100km/h.
 window=np.outer(np.hanning(len(a)),np.hanning(len(a)))
 a=(a-gaussian_filter(a,12))*window;b=(b-gaussian_filter(b,12))*window
 corr=fftconvolve(b,a[::-1,::-1],mode='same');center=np.array(corr.shape)//2
 radius=min(int(np.ceil(100*dt/PIXEL_KM)),len(a)//3)
 yy,xx=np.indices(corr.shape);mask=((yy-center[0])**2+(xx-center[1])**2)<=radius**2
 idx=np.array(np.unravel_index(np.argmax(np.where(mask,corr,-np.inf)),corr.shape))
 shift=idx-center;confidence=float(corr[tuple(idx)]/(np.sqrt((a*a).sum()*(b*b).sum())+1e-9))
 if confidence<.15:return np.zeros(2),confidence
 return shift/dt,confidence

def crop_features(path,forecast_start=None):
 with np.load(path) as z:
  grids=z['rate_mm_h'];metas=json.loads(str(z['metadata_json']));case=json.loads(str(z['case_json']));site=json.loads(str(z['site_json']))
 issue=utc(case['forecast_issue_utc']);times=[utc(m['volume_start_utc']) for m in metas]
 if not all(utc(m['product_generated_utc'])<=issue-timedelta(minutes=2) for m in metas):
  raise ValueError('A radar product was unavailable at the forecast issue')
 if forecast_start is None:forecast_start=issue.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)
 dt=(times[-1]-times[-2]).total_seconds()/3600
 velocity,conf=motion(grids[-2],grids[-1],dt)
 mid=grids.shape[1]//2;local=grids[:,mid-2:mid+3,mid-2:mid+3]
 features=[]
 for grid in grids:
  for radius in [2,8,24,48]:
   q=grid[mid-radius:mid+radius+1,mid-radius:mid+radius+1];good=np.isfinite(q)
   features.extend([float(np.nanmean(q)),float(np.nanmax(q)),float((q[good]>.1).mean()),float(good.mean())])
 features.extend([*velocity,conf,(issue-times[-1]).total_seconds()/3600,np.sin(2*np.pi*issue.timetuple().tm_yday/365.25),np.cos(2*np.pi*issue.timetuple().tm_yday/365.25)])
 persistence=np.repeat(np.nanmean(local[-1]),3)
 # Forecast each10min interval endpoint, integrate each complete clock hour.
 leads=np.array([(forecast_start+timedelta(minutes=10*k)-times[-1]).total_seconds()/3600 for k in range(19)])
 yy,xx=np.meshgrid(np.arange(mid-2,mid+3),np.arange(mid-2,mid+3),indexing='ij')
 adv=[];validfractions=[]
 for lead in leads:
  p=map_coordinates(grids[-1],np.stack([yy-velocity[0]*lead,xx-velocity[1]*lead]),order=1,mode='constant',cval=np.nan)
  validfractions.append(float(np.isfinite(p).mean()))
  # Explicit persistence fallback when the upwind source has left the observed domain.
  adv.append(float(np.nanmean(p)) if np.isfinite(p).mean()>=.8 else float(persistence[0]))
 advection=np.array([np.trapezoid(adv[h*6:h*6+7],dx=1/6) for h in range(3)])
 features.extend(list(advection));features.extend(list(persistence));features.append(min(validfractions))
 return np.array(features),persistence,advection,dict(case=case,site=site,scans=metas,motion_pixels_hour=velocity.tolist(),motion_confidence=conf,advection_min_valid_fraction=min(validfractions),forecast_start_utc=iso(forecast_start))

def make_targets():
 dest=DATA/'mrms_targets.csv'
 if dest.exists():return pd.read_csv(dest,dtype={'case_id':str})
 cases=json.loads((DATA/'frozen_schedule.json').read_text())['cases']; rows=[]
 az=np.load(ROOT/'data/alta_nowcast/radar/radar_sequences.npz');
 alta={k:az[k] for k in ['rate','valid_fraction','rqi_hourly','radar_time','quality_time','latitude','longitude','case_id']};az.close()
 lookup={str(k):i for i,k in enumerate(alta['case_id'])}
 neural={};nindex={}
 for model in ['convlstm','dgmr','nowcastnet']:
  p=ROOT/'outputs/alta-nowcast/tuning-20261107/radar_predictions'/f'{model}.npy'
  if p.exists():
   neural[model]=np.load(p,mmap_mode='r');ids=np.load(p.with_name(model+'_index.npz'))['case_id'];nindex[model]={str(k):i for i,k in enumerate(ids)}
 for case in cases:
  for site_key in ['alta','vail','mansfield','jay']:
   site=SITES[site_key if site_key!='vail' else 'vail_ftg']
   if site_key=='alta':
    i=lookup[case['case_id']];rate=alta['rate'][i];valid=alta['valid_fraction'][i];quality=alta['rqi_hourly'][i];rt=alta['radar_time'][i];qt=alta['quality_time'][i];lat=alta['latitude'];lon=alta['longitude']
   else:
    region='vail' if site_key=='vail' else 'vermont';p=ROOT/'outputs/radar-transfer/prepared'/region/'cases'/(case['case_id']+'.npz')
    if not p.exists():continue
    with np.load(p) as z:rate=z['rate_mm_hr'];valid=z['valid_fraction'];quality=z['rqi_hourly'];rt=z['radar_time'];qt=z['quality_time'];lat=z['latitude'];lon=z['longitude']
   iy=int(np.argmin(abs(lat-site['latitude'])));ix=int(np.argmin(abs(lon-site['longitude'])))
   sl=np.s_[:,iy-1:iy+2,ix-1:ix+2];rr=rate[sl];vv=valid[sl];qq=quality[sl]
   qix=np.searchsorted(qt,rt,side='right')-1;qm=qq[qix]
   mask=np.isfinite(rr)&(rr>=0)&(vv>=.9)&np.isfinite(qm)&(qm>=.5)
   count=mask.sum((1,2));local=np.divide(np.where(mask,rr,0).sum((1,2)),count,out=np.full(len(rr),np.nan),where=count>=5)
   target=[np.trapezoid(local[10+h*6:17+h*6],dx=1/6) for h in range(3)]
   row=dict(case_id=case['case_id'],site=site_key,split=case['split'],winter=case['winter'],target1=target[0],target2=target[1],target3=target[2],mrms_persistence=local[8],rqi_target_mean=float(np.nanmean(qm[10:])),qualified_fraction=float(mask[10:].mean()))
   for model,array in neural.items():
    if site_key=='alta' and case['case_id'] in nindex[model]:
     pred=np.asarray(array[nindex[model][case['case_id']],:,1:,iy-1:iy+2,ix-1:ix+2]).mean(0)
     pm=mask[10:];ratepred=np.divide(np.where(pm,pred,0).sum((1,2)),count[10:],out=np.full(19,np.nan),where=count[10:]>=5)
     for h in range(3):row[f'previous_{model}_{h+1}']=np.trapezoid(ratepred[h*6:h*6+7],dx=1/6)
   rows.append(row)
 frame=pd.DataFrame(rows);frame.to_csv(dest,index=False);return frame

def metric(y,p):
 err=p-y
 return {'n':len(y),'MAE_1h_mm':float(np.mean(abs(err[:,0]))),'RMSE_1h_mm':float(np.sqrt(np.mean(err[:,0]**2))),
 'MAE_3h_mm':float(np.mean(abs(err.sum(1)))),'RMSE_3h_mm':float(np.sqrt(np.mean(err.sum(1)**2))),
 'bias_3h_mm':float(err.sum(1).mean()),'wet_1h_cases':int((y[:,0]>.1).sum()),'wet_3h_cases':int((y.sum(1)>.3).sum())}

def candidates():
 d={}
 for a in [1,10]:
  d[f'ridge_alpha{a}']=make_pipeline(SimpleImputer(),StandardScaler(),Ridge(alpha=a))
  d[f'MLP_16_8_alpha{a}']=make_pipeline(SimpleImputer(),StandardScaler(),MLPRegressor(hidden_layer_sizes=(16,8),alpha=a,solver='lbfgs',max_iter=1000,random_state=61111))
 for leaf in [5,10]:d[f'extra_trees_leaf{leaf}']=make_pipeline(SimpleImputer(),ExtraTreesRegressor(n_estimators=200,min_samples_leaf=leaf,max_features=.8,random_state=61111,n_jobs=2))
 return d

def train():
 plan=DATA/'training_plan.json'
 if plan.exists():assert json.loads(plan.read_text())==PLAN
 else:plan.write_text(json.dumps(PLAN,indent=2))
 target=make_targets();metrics=[];prediction_rows=[];winners={};all_models={};coverage=[]
 for site_key in SITES:
  key=SITES[site_key]['site'];frames=[];xx=[];basep=[];basea=[];metadata=[]
  for _,r in target[target.site==key].iterrows():
   p=DATA/'crops'/site_key/(r.case_id+'.npz')
   if p.exists() and np.isfinite(r[['target1','target2','target3']].astype(float)).all():
    f,b,a,m=crop_features(p);frames.append(r);xx.append(f);basep.append(b);basea.append(a);metadata.append(m)
  if not frames:
   cached=sorted((DATA/'crops'/site_key).glob('20*.npz'))
   geometry=None
   if cached:
    with np.load(cached[0]) as z:geometry=json.loads(str(z['metadata_json']))[-1]
   coverage.append(dict(site=site_key,scheduled=len(target[target.site==key]),radar_crops=len(cached),qualified_train=0,qualified_validation=0,qualified_test=0,radar_geometry=geometry,reason='No qualified MRMS verification windows at the frozen RQI>=0.5 threshold; no deployable radar-only model.'))
   continue
  df=pd.DataFrame(frames);x=np.array(xx);y=df[['target1','target2','target3']].to_numpy(float);splits=df.split.to_numpy();tr=splits=='train';va=splits=='validation';te=splits=='test'
  coverage.append(dict(site=site_key,scheduled=len(target[target.site==key]),radar_crops=len(list((DATA/'crops'/site_key).glob('20*.npz'))),qualified_train=int(tr.sum()),qualified_validation=int(va.sum()),qualified_test=int(te.sum()),radar_geometry=metadata[0]['scans'][-1]))
  if min(tr.sum(),va.sum(),te.sum())<5:continue
  predictions={'zero':np.zeros_like(y),'train_mean':np.repeat(y[tr].mean(0)[None],len(y),axis=0),'DPR_persistence':np.array(basep),'DPR_advection':np.array(basea)}
  fitted={}
  for name,model in candidates().items():
   with warnings.catch_warnings(record=True) as ww:
    model.fit(x[tr],y[tr]);predictions[name]=np.maximum(0,model.predict(x));fitted[name]=model
   if ww:print(site_key,name,[str(w.message)[:150] for w in ww],flush=True)
  scores={n:(metric(y[va],p[va])['RMSE_1h_mm']+metric(y[va],p[va])['RMSE_3h_mm']/3)/2 for n,p in predictions.items()}
  selected=min(scores,key=scores.get);winners[site_key]=dict(model=selected,validation_score=scores[selected],train_mean=y[tr].mean(0).tolist())
  all_models[site_key]=fitted
  # Validation residual interval is descriptive; it is not an independent calibrated confidence guarantee.
  residual=np.abs(predictions[selected][va]-y[va]);winners[site_key]['validation_absolute_error_p80_hourly_mm']=np.quantile(residual,.8,axis=0).tolist()
  for name,p in predictions.items():
   for split in ['validation','test']:
    mask=splits==split;metrics.append(dict(site=site_key,model=name,split=split,selected=name==selected,**metric(y[mask],p[mask])))
   for j,(_,r) in enumerate(df.iterrows()):
    prediction_rows.append(dict(case_id=r.case_id,site=site_key,model=name,split=r.split,**{f'pred{h+1}':float(p[j,h]) for h in range(3)},**{f'truth{h+1}':float(y[j,h]) for h in range(3)}))
  # Add old tuned MRMS neural models only where already-produced held-out forecasts exist.
  for name in ['convlstm','dgmr','nowcastnet']:
   cols=[f'previous_{name}_{h+1}' for h in range(3)]
   if all(k in df for k in cols):
    p=df[cols].to_numpy(float);mask=te&np.isfinite(p).all(1)
    if mask.any():metrics.append(dict(site=site_key,model='previous_MRMS_'+name,split='test',selected=False,**metric(y[mask],p[mask])))
  p=np.repeat(df.mrms_persistence.to_numpy()[:,None],3,axis=1);mask=te&np.isfinite(p).all(1)
  if mask.any():metrics.append(dict(site=site_key,model='MRMS_persistence',split='test',selected=False,**metric(y[mask],p[mask])))
  print(site_key,'train/val/test',tr.sum(),va.sum(),te.sum(),'selected',selected,'score',scores[selected],flush=True)
 pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(prediction_rows).to_csv(OUT/'case_predictions.csv',index=False)
 (OUT/'selection.json').write_text(json.dumps(winners,indent=2));(OUT/'coverage.json').write_text(json.dumps(coverage,indent=2));
 with (OUT/'calibrators.pkl').open('wb') as f:pickle.dump(all_models,f)
 report=dict(plan=PLAN,selected=winners,coverage=coverage,warning='Small calendar cohort, often dry. MRMS shares NEXRAD inputs and is liquid analysis rather than hourly snowfall truth. Beam geometry is diagnostic; terrain blockage is not explicitly resolved. Stowe and Jay targets are lower-elevation town stations.',sources=['https://registry.opendata.aws/noaa-nexrad/','https://www.ncei.noaa.gov/products/radar/next-generation-weather-radar','https://www.roc.noaa.gov/public-documents/icds/2620003AE.pdf'])
 (OUT/'report.json').write_text(json.dumps(report,indent=2))
 return winners

def current():
 case=json.loads((DATA/'current_case.json').read_text());selected=json.loads((OUT/'selection.json').read_text())
 with (OUT/'calibrators.pkl').open('rb') as f:models=pickle.load(f)
 result=dict(issued_at=case['forecast_issue_utc'],source='Native NEXRAD LevelIII DPR radar rate sequences',quantity='liquid_equivalent_precipitation',units='mm',sites={})
 for site in ['alta','vail','mansfield','jay']:
  keys=[k for k,s in SITES.items() if s['site']==site and k in selected]
  if not keys:result['sites'][site]={'available':False,'reason':'Insufficient qualified historical cases'};continue
  key=min(keys,key=lambda k:selected[k]['validation_score']);p=DATA/'crops'/key/(case['case_id']+'.npz')
  if not p.exists():result['sites'][site]={'available':False,'reason':'No fresh NEXRAD scan sequence'};continue
  # A failed reacquisition may deliberately preserve the previous cache bytes.
  # Existence alone therefore does not establish that this issue/target is valid.
  try:
   validate_crop(p,case,key)
   x,per,adv,m=crop_features(p)
  except Exception as exc:
   result['sites'][site]={'available':False,'reason':'Invalid NEXRAD scan sequence: '+str(exc)}
   continue
  choice=selected[key];name=choice['model']
  values={'zero':np.zeros(3),'train_mean':np.array(choice['train_mean']),'DPR_persistence':per,'DPR_advection':adv}.get(name)
  if values is None:values=np.maximum(0,models[key][name].predict(x[None])[0])
  start=utc(m['forecast_start_utc']);spread=np.array(choice['validation_absolute_error_p80_hourly_mm'])
  geometry=m['scans'][-1];age=(utc(case['forecast_issue_utc'])-utc(geometry['volume_start_utc'])).total_seconds()/60
  result['sites'][site]=dict(available=age<=30,model=name,radar='K'+SITES[key]['radar'],model_site_key=key,issued_at=case['forecast_issue_utc'],start=iso(start),end=iso(start+timedelta(hours=3)),target=SITES[key],
   hourly=[dict(start=iso(start+timedelta(hours=h)),end=iso(start+timedelta(hours=h+1)),liquid_mm=float(values[h]),liquid_lower_mm=float(max(0,values[h]-spread[h])),liquid_upper_mm=float(values[h]+spread[h])) for h in range(3)],
   coverage=dict(distance_km=geometry['distance_km'],nominal_beam_above_target_m=geometry['beam_above_target_m'],terrain_blockage_resolved=False,grid_fraction=geometry['valid_grid_fraction'],scan_age_minutes=age,advection_valid_fraction=m['advection_min_valid_fraction']),
   provenance=m,selection=choice,limitations=['Liquid precipitation estimate, not an observed fresh-snow amount.','MRMS validation shares NEXRAD inputs; intervals are descriptive validation residual bands.','DPR zero can reflect under-detection, especially in mountain blockage.'])
 (OUT/'current_forecast.json').write_text(json.dumps(result,indent=2));print('current output',OUT/'current_forecast.json',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--current',action='store_true');a=ap.parse_args();current() if a.current else train()
