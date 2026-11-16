"""Train/evaluate genuine Alta 12-hour snowfall using issued MOS/HRRR and past radar.

Selection uses 2022/23 validation weeks, never 2023/24 test outcomes. This is a
fixed 04/16 MST interval experiment, not evidence of hourly or rolling skill.
"""
from __future__ import annotations
from pathlib import Path
import copy, hashlib, json, os, pickle, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import torch
from torch import nn
from torch.nn import functional as F
from neural48_models import TemporalBlock, RadarEncoder, Neural48Config

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/alta12_training'
OUT=ROOT/'outputs/nwp-weather/alta12-trained'
HISTORY_SIZE=18

def missing_history(n):
    x=np.full((n,HISTORY_SIZE),np.nan,np.float32)
    x[:,[3,7,11,15]]=0.;x[:,-1]=1440.
    return x

def scale_fit(x):
    z=x.reshape(-1,x.shape[-1]); count=np.isfinite(z).sum(0)
    mean=np.divide(np.nansum(z,axis=0),count,out=np.zeros(z.shape[1]),where=count>0)
    var=np.divide(np.nansum((z-mean)**2,axis=0),count,out=np.ones(z.shape[1]),where=count>0)
    return {'mean':mean.astype(np.float32),'std':np.maximum(1,np.sqrt(var)).astype(np.float32)}

def scale_transform(x,s):
    valid=np.isfinite(x)
    v=np.where(valid,(x-s['mean'])/s['std'],0.)
    return np.concatenate([np.clip(v,-15,15),valid.astype(np.float32)],axis=-1).astype(np.float32)

def raw_features(a,kind):
    nwp=a['mos'] if kind=='mos' else a['hrrr'] if kind=='hrrr' else np.concatenate([a['mos'],a['hrrr']],axis=-1)
    static=np.concatenate([a['static'],a['history']],axis=-1)
    return nwp,static

def features(a,kind,scales):
    nwp,static=raw_features(a,kind)
    return {'nwp':scale_transform(nwp,scales['nwp']), 'static':scale_transform(static,scales['static']),
        'radar':np.nan_to_num(a['radar']).astype(np.float32),
        'local':scale_transform(a['local'],scales['local'])}

def flat_features(a,kind,scales,radar=False):
    f=features(a,kind,scales);n=f['nwp']
    vals=[n.mean(1),n.max(1),n.min(1),n[:,0],n[:,-1],f['static']]
    if radar:
        vals +=[f['local'].mean(1),f['local'][:,-1],f['radar'].mean((1,3,4))]
    return np.concatenate(vals,axis=-1)

class Alta12TCN(nn.Module):
    def __init__(self,nwp_features,static_features,radar=False,width=32,dropout=.20):
        super().__init__(); self.kwargs=dict(nwp_features=nwp_features,static_features=static_features,radar=radar,width=width,dropout=dropout)
        self.project=nn.Linear(nwp_features+1,width)
        self.blocks=nn.Sequential(*[TemporalBlock(width,d,dropout) for d in [1,2,4,8]])
        self.radar_encoder=RadarEncoder(Neural48Config(nwp_features=nwp_features,local_features=16,variant='weather_radar',radar_width=24,local_width=12)) if radar else None
        d=2*width+static_features+(self.radar_encoder.output_features if radar else 0)
        self.head=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,32),nn.GELU(),nn.Linear(32,2))
        self.register_buffer('lead',torch.linspace(0,1,12).reshape(1,12,1))
    def forward(self,nwp,static,radar=None,local=None):
        x=self.project(torch.cat([nwp,self.lead.expand(len(nwp),-1,-1)],-1)).transpose(1,2)
        x=self.blocks(x).transpose(1,2)
        parts=[x.mean(1),x[:,-1],static]
        if self.radar_encoder:parts.append(self.radar_encoder(radar,local))
        y=self.head(torch.cat(parts,-1))
        return F.softplus(y[:,0]),y[:,1]

def nn_predict(model,features,batch=256):
    pred=[];prob=[];model.eval()
    with torch.no_grad():
        for i in range(0,len(features['nwp']),batch):
            x={k:torch.as_tensor(v[i:i+batch]) for k,v in features.items()}
            p,l=model(**x);pred.extend(p.numpy());prob.extend(torch.sigmoid(l).numpy())
    return np.asarray(pred),np.asarray(prob)

def make_nn(a,idx,kind,radar,scales,seed,lr,tune,steps=900):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    f=features(a,kind,scales)
    model=Alta12TCN(f['nwp'].shape[-1],f['static'].shape[-1],radar)
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    # Fixed monitor subset; all final training scores are evaluated separately.
    monitor=rng.choice(idx,min(1024,len(idx)),replace=False)
    tf={k:torch.as_tensor(v[idx]) for k,v in f.items()}
    y=torch.as_tensor(a['snow'][idx]); denom=float(max(np.mean(y.numpy()**2),1.))
    nohist=np.concatenate([a['static'][idx],missing_history(len(idx))],-1)
    missing_static=torch.as_tensor(scale_transform(nohist,scales['static']))
    best=1e20;state=None;best_step=0;losses=[]
    for step in range(1,steps+1):
        model.train();batch=rng.integers(0,len(idx),size=min(64,len(idx)))
        xx={k:v[batch].clone() for k,v in tf.items()}
        dropout=torch.as_tensor(rng.random(len(batch))<.30)
        xx['static'][dropout]=missing_static[batch][dropout]
        # Radar branch learns absence in addition to all evening contexts.
        if radar:
            mask=torch.as_tensor(rng.random(len(batch))<.10)
            xx['radar'][mask]=0;xx['local'][mask]=0
        mean,logit=model(**xx);target=y[batch]
        loss=F.mse_loss(mean,target)/denom+.08*F.binary_cross_entropy_with_logits(logit,(target>0).float())
        optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);optim.step()
        if step==1 or step%40==0:
            p,_=nn_predict(model,{k:v[tune] for k,v in f.items()})
            pm,_=nn_predict(model,{k:v[monitor] for k,v in f.items()})
            val=float(np.mean((p-a['snow'][tune])**2))
            losses.append({'step':step,'train_monitor_mse':float(np.mean((pm-a['snow'][monitor])**2)),'validation_tune_mse':val,'batch_objective':float(loss.detach()),'learning_rate':lr})
            if val<best-1e-4:best=val;state=copy.deepcopy(model.state_dict());best_step=step
            if step>=400 and step-best_step>=240:break
    model.load_state_dict(state);p,prob=nn_predict(model,f)
    return {'model':model,'state':state,'kwargs':model.kwargs,'best_step':best_step,'best_tune_mse':best,'seed':seed,'lr':lr,'losses':losses,'prediction':p,'probability':prob,'parameters':sum(v.numel() for v in model.parameters())}

def metrics(y,p,prob=None):
    d=p-y;wet=y>0
    out={'n':int(len(y)),'mae_inches':float(np.mean(np.abs(d))),'rmse_inches':float(np.sqrt(np.mean(d*d))),'mse_inches2':float(np.mean(d*d)),'bias_inches':float(np.mean(d)), 'wet_n':int(wet.sum()),'wet_rmse_inches':float(np.sqrt(np.mean(d[wet]**2))) if wet.any() else None}
    if prob is not None:out['snow_occurrence_brier']=float(np.mean((prob-wet)**2))
    return out

def climo(a,meta,train):
    dates=pd.to_datetime(meta.start_utc,utc=True);day=dates.dt.dayofyear.to_numpy()
    y=a['snow'];pred=[];prob=[]; half=meta.half.to_numpy()
    for i in range(len(meta)):
        distance=np.abs(day[train]-day[i]);near=train[(np.minimum(distance,366-distance)<=20)&(half[train]==half[i])]
        if len(near)<10:near=train[half[train]==half[i]]
        pred.append(y[near].mean());prob.append((y[near]>0).mean())
    return np.asarray(pred),np.asarray(prob)

def block_bootstrap(y,pa,pb,groups,seed=630,rep=1200):
    rng=np.random.default_rng(seed);ug=np.unique(groups);sums=[]
    residual_a=(pa-y)**2;residual_b=(pb-y)**2
    for _ in range(rep):
        sample=rng.choice(ug,len(ug),replace=True);ix=np.concatenate([np.flatnonzero(groups==g) for g in sample])
        sums.append(np.sqrt(residual_a[ix].mean())-np.sqrt(residual_b[ix].mean()))
    return [float(v) for v in np.quantile(sums,[.025,.975])]

def predict12(bundle_path,arrays):
    """Raw array inference; all outputs are 12h snowfall INCHES.

    Required mos[N,12,16],hrrr[N,12,7],static[N,7]. Optional history[N,18],
    radar[N,9,5,16,16],local[N,9,8] default explicitly missing. Optional
    observations[N,32] follow manifest observation-field order. HRRR field order
    is in manifest.json. Missing HRRR selects the broad-MOS fallback. All MOS
    slots must be finite. Trained only on fixed 11/23UTC periods; rolling use is
    an extrapolation, not a validated new horizon.
    """
    path=Path(bundle_path)
    with open(path/'bundle.pkl','rb') as f:b=pickle.load(f)
    a={k:np.asarray(v,np.float32) for k,v in arrays.items()};n=len(a['mos'])
    a.setdefault('hrrr',np.full((n,12,7),np.nan,np.float32));a.setdefault('history',missing_history(n))
    a.setdefault('radar',np.zeros((n,9,5,16,16),np.float32));a.setdefault('local',np.full((n,9,8),np.nan,np.float32))
    if not np.isfinite(a['mos']).all():raise ValueError('Alta12 requires complete issued MOS forecast inputs')
    expected={'mos':(n,12,16),'hrrr':(n,12,7),'static':(n,7),'history':(n,18),'radar':(n,9,5,16,16),'local':(n,9,8)}
    for k,shape in expected.items():
        if a[k].shape!=shape:raise ValueError(f'{k}: expected{shape},got{a[k].shape}')
    complete=np.isfinite(a['hrrr']).all((1,2));results={}
    for name,item in b['models'].items():
        if item['type']=='neural':
            model=Alta12TCN(**item['kwargs']);model.load_state_dict(item['state'])
            pred,prob=nn_predict(model,features(a,item['kind'],item['scales']))
        elif item['type'] in {'forest','forest_obs'}:
            xx=flat_features(a,item['kind'],item['scales'],item['radar'])
            if item['type']=='forest_obs':
                obs=a.get('observations',np.full((n,len(item['obs_names'])),np.nan,np.float32))
                if obs.shape!=(n,len(item['obs_names'])):raise ValueError('Invalid observation feature shape')
                xx=np.concatenate([xx,scale_transform(obs,item['obs_scale'])],axis=1)
            pred=item['model'].predict(xx);prob=item['occurrence'].predict(xx)
        elif item['type']=='physical':
            pred=np.maximum(0,np.nan_to_num(a['hrrr'][:,:,1]).sum(1))*item['ratio']/25.4
            prob=1-np.exp(-pred)
        else:continue
        results[name]=(pred,np.clip(prob,0,1))
    mean=np.zeros(n);prob=np.zeros(n);lo=np.zeros(n);hi=np.zeros(n);labels=[];routes=[]
    has_history=np.isfinite(a['history'][:,[0,4,8,12,16]]).any(1)
    obs=a.get('observations')
    has_observations=np.isfinite(obs[:,[0,8,16,24]]).any(1) if obs is not None else np.zeros(n,bool)
    for i in range(n):
        which='modern' if complete[i] else 'broad'
        if not has_history[i] and which+'_missing_history' in b['selection']:which+='_missing_history'
        if complete[i] and not has_history[i] and has_observations[i] and 'modern_observations' in b['selection']:which='modern_observations'
        select=b['selection'][which];routes.append(which)
        mean[i]=sum(w*results[k][0][i] for k,w in select['weights'].items())
        prob[i]=sum(w*results[k][1][i] for k,w in select['weights'].items())
        q=select['conformal_half_width_inches'];lo[i]=max(0,mean[i]-q);hi[i]=mean[i]+q
        labels.append(select['name'])
    return {'mean_inches':mean,'q10_inches':lo,'q90_inches':hi,'snow_probability':prob,'selected_method':labels,'selection_route':routes,
        'lower80_inches':lo,'upper80_inches':hi,'interval_kind':'split-conformal marginal80% prediction interval',
        'components_inches':{k:v[0] for k,v in results.items()}}

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True)
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))
    a=dict(np.load(DATA/'arrays.npz'));meta=pd.read_csv(DATA/'index.csv')
    dates=pd.to_datetime(meta.start_utc,utc=True)
    weeks=((dates-pd.Timestamp('2022-10-01',tz='UTC')).dt.days//7).to_numpy()
    split=meta.split.to_numpy();stages=np.full(len(meta),'train',object)
    stages[split=='test']='test';isv=split=='validation'
    for r,name in [(0,'tune'),(1,'blend'),(2,'calibration')]:stages[isv&(weeks%3==r)]=name
    meta['validation_stage']=stages;meta['seven_day_block']=weeks;meta.to_csv(OUT/'case_index.csv',index=False)
    pred={};prob={};models={};selection={};search=[];lossrows=[]
    manifest={'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'target':'True Alta Collins12hour manualfreshsnowinches',
        'split':'Train winters2004/05–2021/22; validate2022/23; test2023/24. HRRR subset2020/21onward. Validation7dayblocks assigned tune/blend/calibration modulo3; sameAM/PMday stays together.',
        'test_status':'2023/24 already used in preceding project experiments; this is a held-out rerun, not an untouched blind test.',
        'history':'Only source intervals ending at least2h before issue,30% wholehistory dropout; live manualhistory can be missing.',
        'neural_loss':'MSE divided by training mean square target +0.08 binarycrossentropy(snow>0). Eight causal convolutions in four residual TCN blocks, optional radar CNN/GRU.',
        'selection_criterion':'Lowest tune-block12h MSE for hyperparameters. Separate blend-block MSE for choosing constrainedportfolio against individual. Separatecalibrationblock for80% residualinterval.',
        'hrrr_fields':['precip_mm','snow_water_mm','temperature_k','dewpoint_k','wind_u_ms','wind_v_ms','relative_humidity_pct'],
        'limitations':['Fixed11/23UTC starts; arbitraryrolling12hnotvalidated.','PM uses older00UTC MOS and06UTC HRRR; noPMradar.','No freshsnowhourlytargets; cannotprovehourlytimingskill.','Pre2023/24 observationsqualitycontrolled;2023/24 sourceauthorprovidesrawpatrolreports.','MOS forecasts differ from current commercialweatherprovider; not a retrospectiveOpenMeteobenchmark.','2022/23 is only one validationwinter. Temporalblocks reduce but cannotremove stormdependence.'],
        'source_audit':json.loads((DATA/'audit.json').read_text()),'stage_counts':{}}
    # Fit both broad long-history and modern forecast+radar experiments.
    for cohort in ['broad','modern']:
        mask=np.ones(len(meta),bool) if cohort=='broad' else meta.modern.to_numpy(bool)
        ix={s:np.flatnonzero(mask&(stages==s)) for s in ['train','tune','blend','calibration','test']}
        manifest['stage_counts'][cohort]={k:len(v) for k,v in ix.items()}
        print('cohort',cohort,manifest['stage_counts'][cohort],flush=True)
        tr=ix['train'];tune=ix['tune'];kind='mos' if cohort=='broad' else 'weather'
        nwp,stat=raw_features(a,kind)
        scales={'nwp':scale_fit(nwp[tr]),'static':scale_fit(stat[tr]),'local':scale_fit(a['local'][tr])}
        cp,cpr=climo(a,meta,tr);name=f'{cohort}_climatology';pred[name]=cp;prob[name]=cpr
        candidates=[]
        # Forest history dropout is implemented as deterministic training augmentation.
        aa={k:v.copy() if k=='history' else v for k,v in a.items()};aa['history']=missing_history(len(meta))
        for radar in ([False] if cohort=='broad' else [False,True]):
            xx=flat_features(a,kind,scales,radar);xm=flat_features(aa,kind,scales,radar)
            train_x=np.concatenate([xx[tr],xm[tr]],0);train_y=np.tile(a['snow'][tr],2)
            forest_best=None
            for leaf,depth in [(8,8),(20,8),(35,None)]:
                forest=RandomForestRegressor(n_estimators=180,max_depth=depth,min_samples_leaf=leaf,max_features=.8,n_jobs=4,random_state=427)
                forest.fit(train_x,train_y);pp=np.maximum(0,forest.predict(xx));score=float(np.mean((pp[tune]-a['snow'][tune])**2))
                search.append(dict(cohort=cohort,family='radar_forest' if radar else 'forest',leaf=leaf,depth=depth,tune_mse=score))
                if forest_best is None or score<forest_best[0]:forest_best=(score,forest,pp,leaf,depth)
            score,forest,pp,leaf,depth=forest_best
            occur=RandomForestRegressor(n_estimators=120,max_depth=depth,min_samples_leaf=leaf,max_features=.8,n_jobs=4,random_state=427)
            occur.fit(train_x,(train_y>0).astype(float))
            name=f'{cohort}_'+('radar_forest' if radar else 'forest');pred[name]=pp;prob[name]=occur.predict(xx);candidates.append(name)
            models[name]={'type':'forest','model':forest,'occurrence':occur,'kind':kind,'radar':radar,'scales':scales}
            print(name,'tune',round(score,3),flush=True)
        # Deep neural search with validation-only checkpoint and learning-rate selection.
        for radar in ([False] if cohort=='broad' else [False,True]):
            family='radar_tcn' if radar else 'tcn';runs=[]
            for lr in [.0003,.001,.003]:
                r=make_nn(a,tr,kind,radar,scales,17,lr,tune);runs.append(r)
                search.append(dict(cohort=cohort,family=family,lr=lr,seed=17,tune_mse=r['best_tune_mse'],best_step=r['best_step'],parameters=r['parameters']))
                for row in r['losses']:lossrows.append(dict(cohort=cohort,family=family,seed=17,lr=lr,**{k:v for k,v in row.items() if k!='learning_rate'}))
                print(cohort,family,lr,'step',r['best_step'],'tune',round(r['best_tune_mse'],3),flush=True)
            best=min(runs,key=lambda r:r['best_tune_mse'])
            for seed in [43,79]:
                r=make_nn(a,tr,kind,radar,scales,seed,best['lr'],tune);runs.append(r)
                search.append(dict(cohort=cohort,family=family,lr=r['lr'],seed=seed,tune_mse=r['best_tune_mse'],best_step=r['best_step'],parameters=r['parameters']))
                for row in r['losses']:lossrows.append(dict(cohort=cohort,family=family,seed=seed,lr=r['lr'],**{k:v for k,v in row.items() if k!='learning_rate'}))
                print(cohort,family,'seed',seed,'step',r['best_step'],'tune',round(r['best_tune_mse'],3),flush=True)
            best=min(runs,key=lambda r:r['best_tune_mse']);name=f'{cohort}_{family}'
            pred[name]=best['prediction'];prob[name]=best['probability'];candidates.append(name)
            models[name]={'type':'neural','kwargs':best['kwargs'],'state':best['state'],'scales':scales,'kind':kind,'lr':best['lr'],'seed':best['seed'],'best_step':best['best_step']}
            torch.save(models[name],OUT/'checkpoints'/f'{name}.pt')
        if cohort=='modern':
            snowwater=np.nan_to_num(a['hrrr'][:,:,1]).sum(1)/25.4
            # Raw10:1 baseline plus a training-only calibrated snowfall/water conversion.
            denom=float(np.sum(snowwater[tr]**2));ratio=max(0,float(np.sum(snowwater[tr]*a['snow'][tr]))/max(denom,1e-8))
            for name,mult in [('hrrr_fixed_10to1',10.),('hrrr_train_calibrated',ratio)]:
                pred[name]=np.maximum(0,snowwater*mult);prob[name]=1-np.exp(-pred[name]);models[name]={'type':'physical','ratio':mult}
            candidates +=['hrrr_train_calibrated','broad_forest','broad_tcn']
            manifest['hrrr_train_snowwater_ratio']=ratio
        # Restrict mixture to the three individual methods selected on the tune blocks.
        ranked=sorted(candidates,key=lambda k:np.mean((pred[k][tune]-a['snow'][tune])**2));top=ranked[:3]
        blend=ix['blend'];yy=a['snow'][blend];matrix=np.stack([pred[k][blend] for k in top],1)
        fit=minimize(lambda w:np.mean((matrix@w-yy)**2),np.ones(len(top))/len(top),method='SLSQP',bounds=[(0,1)]*len(top),constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-12,'maxiter':1000})
        if not fit.success:raise RuntimeError(fit.message)
        weights={k:float(w) for k,w in zip(top,fit.x) if w>1e-7}
        weights={k:w/sum(weights.values()) for k,w in weights.items()}
        port=sum(w*pred[k] for k,w in weights.items());portprob=sum(w*prob[k] for k,w in weights.items())
        best_ind=min(candidates,key=lambda k:np.mean((pred[k][blend]-yy)**2))
        blend_mse=float(np.mean((port[blend]-yy)**2));ind_mse=float(np.mean((pred[best_ind][blend]-yy)**2))
        portfolio=f'{cohort}_portfolio';pred[portfolio]=port;prob[portfolio]=portprob
        if blend_mse<ind_mse*.995:chosen=portfolio
        else:chosen=best_ind;weights={best_ind:1.}
        cal=ix['calibration'];res=np.abs(pred[chosen][cal]-a['snow'][cal]);q=float(np.quantile(res,min(1,np.ceil((len(res)+1)*.8)/len(res)),method='higher'))
        selection[cohort]={'name':chosen,'weights':weights,'conformal_half_width_inches':q,'tune_ranking':ranked,'blend_mse':float(np.mean((pred[chosen][blend]-yy)**2)),'calibration_n':len(cal)}
        print('SELECT',cohort,selection[cohort],flush=True)
        # Save progress before the next cohort.
        pd.DataFrame(search).to_csv(OUT/'hyperparameter_search.csv',index=False)
        pd.DataFrame(lossrows).to_csv(OUT/'training_curves.csv',index=False)
    # Evaluation only now, after every model/weight/calibration choice has been fixed.
    rows=[];predrows=[]
    for cohort in ['broad','modern']:
        mask=np.ones(len(meta),bool) if cohort=='broad' else meta.modern.to_numpy(bool)
        names=[k for k in pred if k.startswith(cohort)]
        if cohort=='modern':names+=['hrrr_fixed_10to1','hrrr_train_calibrated','broad_forest','broad_tcn']
        for stage in ['train','tune','blend','calibration','test']:
            ids=np.flatnonzero(mask&(stages==stage))
            for k in names:
                rows.append(dict(cohort=cohort,stage=stage,method=k,**metrics(a['snow'][ids],pred[k][ids],prob[k][ids])))
                if stage=='test':
                    for i in ids:predrows.append(dict(row=i,cohort=cohort,method=k,start_utc=meta.start_utc[i],end_utc=meta.end_utc[i],half=int(meta.half[i]),actual_inches=float(a['snow'][i]),prediction_inches=float(pred[k][i]),snow_probability=float(prob[k][i])))
        ids=np.flatnonzero(mask&(stages=='test'));chosen=selection[cohort]['name'];q=selection[cohort]['conformal_half_width_inches']
        p=pred[chosen][ids];actual=a['snow'][ids]
        selection[cohort]['test_interval_coverage']=float(np.mean((actual>=np.maximum(0,p-q))&(actual<=p+q)))
        selection[cohort]['test_rmse_difference_vs_climatology_95ci']=block_bootstrap(actual,p,pred[f'{cohort}_climatology'][ids],weeks[ids])
        selection[cohort]['test_missing_history_metrics']=None
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(predrows).to_csv(OUT/'test_predictions.csv',index=False)
    manifest['selection']=selection;manifest['trained_model_count']=len(models);manifest['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bundle={'models':models,'selection':selection,'manifest':manifest}
    with open(OUT/'bundle.pkl','wb') as f:pickle.dump(bundle,f)
    # Verify saved inference agreement, then quantify the realistic missing-history condition.
    ids=np.flatnonzero(meta.modern.to_numpy(bool)&(stages=='test'))
    infer=predict12(OUT,{k:v[ids] for k,v in a.items() if k in ['mos','hrrr','static','history','radar','local']})
    expected=pred[selection['modern']['name']][ids]
    assert np.allclose(infer['mean_inches'],expected,rtol=1e-5,atol=1e-5)
    args={k:v[ids] for k,v in a.items() if k in ['mos','hrrr','static','radar','local']}
    inf=predict12(OUT,args);manifest['selection']['modern']['test_missing_history_metrics']=metrics(a['snow'][ids],inf['mean_inches'],inf['snow_probability'])
    bundle['manifest']=manifest
    with open(OUT/'bundle.pkl','wb') as f:pickle.dump(bundle,f)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    np.savez_compressed(OUT/'predictions_all.npz',**{k:v for k,v in pred.items()})
    print('DONE',json.dumps(selection,indent=2),flush=True)

if __name__=='__main__':main()
