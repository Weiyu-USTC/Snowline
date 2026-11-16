"""Preplanned mountain-observation ablation for Alta's operational12h model.

Uses existing frozen neural candidates plus newly trained observation forests;
no test-score selection. Sensor snowdepth is only a predictor, never the target.
"""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from train_alta12 import ROOT,DATA,OUT,missing_history,scale_fit,scale_transform,flat_features,metrics,block_bootstrap
from prepare_mountain_hourly import issue_features

SENSORS=['temperature_c','wet_bulb_c','precipitation_mm','snow_depth_cm']
SUFFIXES=['latest','age_hours','mean_3h','n_3h','mean_6h','n_6h','mean_24h','n_24h']
OBS_NAMES=[f'obs_{f}_{s}' for f in SENSORS for s in SUFFIXES]

def observation_vector(observations,issue):
    d=issue_features(observations,issue,'CLN')
    return np.array([d[k] for k in OBS_NAMES],np.float32)

def main():
    source=ROOT/'data/mountain_hourly/alta_cln.csv.gz'
    obs=pd.read_csv(source);meta=pd.read_csv(OUT/'case_index.csv');a=dict(np.load(DATA/'arrays.npz'))
    with open(OUT/'bundle.pkl','rb') as f:b=pickle.load(f)
    full=dict(np.load(OUT/'predictions_all.npz'));pred=dict(np.load(OUT/'predictions_missing_history.npz'))
    a['history']=missing_history(len(meta));modern=meta.modern.to_numpy(bool)
    cache=DATA/'observation_features.npz'
    if cache.exists():
        z=np.load(cache);vectors=z['observations'];assert list(z['names'])==OBS_NAMES
    else:
        vectors=np.full((len(meta),len(OBS_NAMES)),np.nan,np.float32)
        for r in meta[modern].itertuples():vectors[r.row]=observation_vector(obs,r.issue_utc)
        np.savez_compressed(cache,observations=vectors,names=np.array(OBS_NAMES))
    stages=meta.validation_stage.to_numpy();ix={s:np.flatnonzero(modern&(stages==s)) for s in ['train','tune','blend','calibration','test']}
    train,tune,blend,cal,test=[ix[s] for s in ['train','tune','blend','calibration','test']]
    # Sensors with no valid training values remain explicit missing features.
    oscale=scale_fit(vectors[train]);o=scale_transform(vectors,oscale);om=scale_transform(np.full_like(vectors,np.nan),oscale)
    scales=b['models']['modern_forest']['scales'];newnames=[];search=[];prob={}
    for radar in [False,True]:
        base=flat_features(a,'weather',scales,radar);xx=np.concatenate([base,o],1);xm=np.concatenate([base,om],1)
        tx=np.concatenate([xx[train],xm[train]],0);ty=np.tile(a['snow'][train],2);best=None
        for leaf,depth in [(8,8),(20,8),(35,None)]:
            forest=RandomForestRegressor(n_estimators=180,min_samples_leaf=leaf,max_depth=depth,max_features=.8,n_jobs=4,random_state=427)
            forest.fit(tx,ty);p=np.maximum(0,forest.predict(xx));score=float(np.mean((p[tune]-a['snow'][tune])**2))
            search.append({'radar':radar,'leaf':leaf,'depth':depth,'tune_mse':score})
            if best is None or score<best[0]:best=(score,forest,p,leaf,depth)
        score,model,p,leaf,depth=best
        occur=RandomForestRegressor(n_estimators=120,min_samples_leaf=leaf,max_depth=depth,max_features=.8,n_jobs=4,random_state=427);occur.fit(tx,(ty>0).astype(float))
        name='modern_radar_observations_forest' if radar else 'modern_observations_forest';newnames.append(name);pred[name]=p;prob[name]=occur.predict(xx)
        b['models'][name]={'type':'forest_obs','model':model,'occurrence':occur,'kind':'weather','radar':radar,'scales':scales,'obs_scale':oscale,'obs_names':OBS_NAMES}
        print(name,'tuneMSE',score,flush=True)
    candidates=['modern_forest','modern_radar_forest','modern_tcn','modern_radar_tcn','broad_forest','broad_tcn','hrrr_train_calibrated']+newnames
    y=a['snow'];rank=sorted(candidates,key=lambda k:np.mean((pred[k][tune]-y[tune])**2));top=rank[:3];mat=np.stack([pred[k][blend] for k in top],1)
    fit=minimize(lambda w:np.mean((mat@w-y[blend])**2),np.ones(len(top))/len(top),method='SLSQP',bounds=[(0,1)]*len(top),constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-12,'maxiter':1000})
    if not fit.success:raise RuntimeError(fit.message)
    weights={k:float(w) for k,w in zip(top,fit.x) if w>1e-7};weights={k:w/sum(weights.values()) for k,w in weights.items()}
    pp=sum(w*pred[k] for k,w in weights.items());name='modern_observations_portfolio';pred[name]=pp
    ind=min(candidates,key=lambda k:np.mean((pred[k][blend]-y[blend])**2))
    if np.mean((pp[blend]-y[blend])**2)<.995*np.mean((pred[ind][blend]-y[blend])**2):chosen=name
    else:chosen=ind;weights={ind:1.}
    # Compare the proposed sensor route against the already-frozen operational route
    # on blending validation only. If it does not improve, retain the old route.
    existing=b['selection']['modern_missing_history'];previous=pred[existing['name']]
    improving=float(np.mean((pred[chosen][blend]-y[blend])**2))<.995*float(np.mean((previous[blend]-y[blend])**2))
    if not improving:chosen=existing['name'];weights=existing['weights']
    q=float(np.quantile(np.abs(pred[chosen][cal]-y[cal]),min(1,np.ceil((len(cal)+1)*.8)/len(cal)),method='higher'))
    p=pred[chosen][test];actual=y[test]
    selected={'name':chosen,'weights':weights,'conformal_half_width_inches':q,'tune_ranking':rank,'blend_mse':float(np.mean((pred[chosen][blend]-y[blend])**2)),'calibration_n':len(cal),
        'sensor_route_improved_validation':improving,'sensor_models_receive_weight':any(k in newnames for k in weights),
        'history_condition':'Manual snowfall history absent; hourly mountain sensors available with1h assumed publication lag.',
        'test_interval_coverage':float(np.mean((actual>=np.maximum(0,p-q))&(actual<=p+q))),
        'test_rmse_difference_vs_climatology_95ci':block_bootstrap(actual,p,full['modern_climatology'][test],meta.seven_day_block.to_numpy()[test]),
        'test_metrics':metrics(actual,p)}
    b['selection']['modern_observations']=selected;b['manifest']['selection']=b['selection'];b['manifest']['observations']={'source':str(source.relative_to(ROOT)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'fields':OBS_NAMES,'availability':'Observation valid time plus1h; original receipt timestamps unavailable; historical sensor records may be revised.','used_as_target':False,'missing_augmentation':'Every training example has a second copy with the whole observation vector missing. Manual snowfall history absent throughout this ablation.'}
    with open(OUT/'bundle.pkl','wb') as f:pickle.dump(b,f)
    (OUT/'manifest.json').write_text(json.dumps(b['manifest'],indent=2));np.savez_compressed(OUT/'predictions_observations.npz',**pred)
    records=[]
    for stage,ids in ix.items():
        for k in candidates+[name,existing['name']]:records.append({'stage':stage,'method':k,**metrics(y[ids],pred[k][ids])})
    pd.DataFrame(records).drop_duplicates(['stage','method']).to_csv(OUT/'metrics_observations.csv',index=False)
    pd.DataFrame(search).to_csv(OUT/'observation_search.csv',index=False)
    print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
