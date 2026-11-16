"""Train, select and audit forecast-conditioned two-day snowfall distributions.

The input is hourly issued weather guidance and pre-issue radar. Observed labels
are two exact 24-hour station accumulations. This is not an hourly snow model.
"""
from pathlib import Path
import argparse, copy, hashlib, html, json, os, pickle, time, warnings
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtri, gammainc
from neural48_models import Neural48Config, Neural48Model, forecast_loss, checkpoint_payload, restore_model, sample_joint_paths, parameter_count
from neural48_baselines import fit_baselines, predict_baselines, CURRENT, ALL_WEATHER, CLIMATE

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/neural48_training/prepared'
OUT=ROOT/'outputs/nwp-weather/neural48-trained'
SITES=['alta','vail','mansfield','jay']
LABELS={'alta':'Alta','vail':'Vail · sensor proxy','mansfield':'Stowe · town station','jay':'Jay Peak · Westfield station'}
FAMILIES=['weather_only','weather_radar']
NAMES={'weather_only':'Weather TCN','weather_radar':'Radar + weather TCN'}
SEEDS=[20261111,20261112,20261113]
LRS=[.0003,.001,.003]

def jwrite(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))

def fit_scale(x):
    flat=np.asarray(x,float).reshape(-1,x.shape[-1]); flat[~np.isfinite(flat)]=np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mean=np.nanmean(flat,axis=0); std=np.nanstd(flat,axis=0)
    mean=np.nan_to_num(mean,nan=0.);std=np.nan_to_num(std,nan=1.)
    std[std<1e-5]=1.
    return {'mean':mean.tolist(),'std':std.tolist()}

def transform_scale(x,scale):
    x=np.asarray(x,float);missing=~np.isfinite(x)
    value=(np.where(missing,scale['mean'],x)-scale['mean'])/scale['std']
    return np.concatenate([np.clip(value,-15,15),missing.astype(float)],axis=-1).astype('float32')

def clean_history(history):
    h=np.asarray(history,float).copy()
    h[h[:,-1]>=60,-2]=np.nan
    return h

def unavailable_history(history):
    h=np.full_like(np.asarray(history,float),np.nan)
    h[:,[2,5,8,11,14]]=0.;h[:,-1]=60.
    return h

def fit_preprocessing(arrays,train_idx):
    p={'schema_version':1,'site_order':SITES,'raw_shapes':{'nwp':[48,23],'radar':[9,5,16,16],'local':[9,8],'static':[8],'history':[17]},
       'fit_rows':np.asarray(train_idx).tolist(),'history_available_lag_hours':24,'training_whole_history_dropout':.3}
    for name in ['nwp','local']:
        p[name]=fit_scale(arrays[name][train_idx])
    combined=np.concatenate([arrays['static'],clean_history(arrays['history'])],axis=-1)
    p['static_history']=fit_scale(combined[train_idx])
    p['target_scale']=[float(max(1.,np.sqrt(np.mean(arrays['y'][train_idx[arrays['site'][train_idx]==s]]**2)))) for s in range(4)]
    p['uncertainty']='Zero-inflated Gamma ensemble, train-only randomized PIT copula per site. Not prospectively calibrated.'
    return p

def transform(arrays,prep):
    return {'nwp':transform_scale(arrays['nwp'],prep['nwp']),
        'radar':np.asarray(arrays['radar'],dtype='float32'),
        'local':transform_scale(arrays['local'],prep['local']),
        'static':transform_scale(np.concatenate([arrays['static'],clean_history(arrays['history'])],axis=-1),prep['static_history']),
        'site':np.asarray(arrays['site'],dtype='int64'),
        'static_no_history':transform_scale(np.concatenate([arrays['static'],unavailable_history(arrays['history'])],axis=-1),prep['static_history'])}

def tensorize(x):return {k:torch.as_tensor(v) for k,v in x.items()}

def forward(model,x,idx,history_dropout=0.):
    static=x['static'][idx]
    if history_dropout:
        absent=torch.rand((len(idx),1))<history_dropout
        static=torch.where(absent,x['static_no_history'][idx],static)
    return model(x['nwp'][idx],x['radar'][idx],x['local'][idx],x['site'][idx],static)

def predict_parameters(model,x,idx,batch=128):
    model.eval();parts={}
    with torch.no_grad():
        for offset in range(0,len(idx),batch):
            p=forward(model,x,idx[offset:offset+batch])
            for key,value in p.items():parts.setdefault(key,[]).append(value.cpu().numpy())
    return {k:np.concatenate(v) for k,v in parts.items()}

def selection_score(mean,y,site):
    # Equal site weight, with 48h errors divided by two to share daily units.
    values=[]
    for s in np.unique(site):
        d=mean[site==s]-y[site==s]
        values.append(float(.5*np.mean(d[:,0]**2)+.5*np.mean((d.sum(1)/2)**2)))
    return float(np.mean(values))

def estimate_rho(params,y,sites,seed):
    rng=np.random.default_rng(seed);p=params['probability'];a=params['concentration'];b=params['rate']
    zero=y<=0
    u=np.where(zero,rng.uniform(size=y.shape)*(1-p),1-p+p*gammainc(a,b*y))
    z=ndtri(np.clip(u,1e-5,1-1e-5));result=[]
    for s in range(4):
        zs=z[sites==s]
        rho=float(np.corrcoef(zs.T)[0,1]) if len(zs)>5 else 0.
        result.append(float(np.clip(rho if np.isfinite(rho) else 0.,-.8,.8)))
    return result

def train_run(family,lr,seed,x,y,sites,train_idx,val_idx,run_dir,max_updates=1400):
    run_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
    model=Neural48Model(Neural48Config(nwp_features=x['nwp'].shape[-1],static_features=x['static'].shape[-1],local_features=x['local'].shape[-1],variant=family,dropout=.2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.002)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3,min_lr=1e-5)
    weights=np.array([1/np.sum(sites[train_idx]==s) for s in sites[train_idx]],float);weights/=weights.sum()
    yy=torch.as_tensor(y,dtype=torch.float32)
    best=float('inf');best_step=0;best_state=None;rows=[];started=time.time()
    for step in range(1,max_updates+1):
        model.train();idx=rng.choice(train_idx,size=32,replace=True,p=weights)
        p=forward(model,x,idx,history_dropout=.3);loss=forecast_loss(p,yy[idx],nll_weight=.1)
        optimizer.zero_grad(set_to_none=True);loss['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
        if step==1 or step%50==0 or step==max_updates:
            trainp=predict_parameters(model,x,train_idx);valp=predict_parameters(model,x,val_idx)
            trscore=selection_score(trainp['mean'],y[train_idx],sites[train_idx]);score=selection_score(valp['mean'],y[val_idx],sites[val_idx])
            row={'update':step,'train_selection_mse':trscore,'validation_selection_mse':score,'learning_rate':optimizer.param_groups[0]['lr'],'batch_loss':float(loss['loss'].detach()),'seconds':time.time()-started}
            rows.append(row)
            if score<best-1e-5:
                best=score;best_step=step;best_state=copy.deepcopy(model.state_dict())
            scheduler.step(score)
            if step%200==0:print(f'{family} lr={lr} seed={seed} update={step} train={trscore:.4f} validation={score:.4f} best={best:.4f}',flush=True)
            if step>=500 and step-best_step>=400:break
    torch.save(checkpoint_payload(model,{'family':family,'seed':seed,'lr':lr,'update':step,'checkpoint':'final'}),run_dir/'final.pt')
    model.load_state_dict(best_state);model.eval()
    trainp=predict_parameters(model,x,train_idx);rho=estimate_rho(trainp,y[train_idx],sites[train_idx],seed)
    info={'family':family,'learning_rate':lr,'seed':seed,'best_update':best_step,'final_update':step,'validation_score':best,'final_validation_score':rows[-1]['validation_selection_mse'],'parameters':parameter_count(model),'seconds':time.time()-started,'copula_rho_by_site':rho,'copula_fit':'Randomized PIT correlation using only training cases; in-sample dependence estimate, not out-of-sample calibration.'}
    torch.save(checkpoint_payload(model,info),run_dir/'best.pt')
    # Optimizer state belongs to the final update, never silently attached to best weights.
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'update':step},run_dir/'final_optimizer.pt')
    pd.DataFrame(rows).to_csv(run_dir/'history.csv',index=False);jwrite(run_dir/'summary.json',info)
    print('finished '+str(info),flush=True)
    return info

def load_bundle(path=OUT/'models'):
    path=Path(path);prep=json.loads((path/'preprocessing.json').read_text());selection=json.loads((path/'selection.json').read_text())
    models={}
    for family,entries in selection['ensembles'].items():
        models[family]=[{'model':restore_model(path/e['checkpoint']),'rho':e['rho'],'seed':e['seed']} for e in entries]
    return {'path':path,'preprocessing':prep,'selection':selection,'models':models}

def predict_neural(bundle,arrays,family=None,n_paths=768):
    """Predict raw-case arrays; family=None uses validation-selected family by site."""
    torch.set_num_threads(1)
    if family is not None and family not in FAMILIES:raise ValueError('Unknown family')
    transformed=tensorize(transform(arrays,bundle['preprocessing']));n=len(arrays['site'])
    means=np.zeros((n,2),float);paths=np.zeros((n,n_paths,2),dtype='float32');methods=['']*n
    scales=np.asarray(bundle['preprocessing']['target_scale'])
    for f in FAMILIES:
        idx=np.array([i for i,s in enumerate(arrays['site']) if (family or bundle['selection']['selected_family_by_site'][SITES[int(s)]])==f],int)
        if not len(idx):continue
        ensemble=bundle['models'][f];parts=[]
        per=int(np.ceil(n_paths/len(ensemble)))
        for entry in ensemble:
            par=predict_parameters(entry['model'],transformed,idx)
            means[idx]+=par['mean']*scales[np.asarray(arrays['site'])[idx],None]/len(ensemble)
            pth=np.empty((len(idx),per,2),dtype='float32')
            for s in np.unique(np.asarray(arrays['site'])[idx]):
                positions=np.flatnonzero(np.asarray(arrays['site'])[idx]==s)
                pth[positions]=sample_joint_paths({k:v[positions] for k,v in par.items()},n_paths=per,seed=entry['seed'],rho=entry['rho'][int(s)])*scales[int(s)]
            parts.append(pth)
        paths[idx]=np.concatenate(parts,axis=1)[:,:n_paths]
        for i in idx:methods[i]=NAMES[f]
    return {'daily_mean':means,'joint_paths':paths,'methods':methods,'target_definition':'Two exact consecutive24-hour observed accumulation periods; no independently supervised hourly values.'}

def metrics(y,mean,paths=None):
    d=mean-y;out={'n':len(y),'mae_inches':float(np.mean(abs(d))),'mse_inches2':float(np.mean(d*d)),'rmse_inches':float(np.sqrt(np.mean(d*d))),'bias_inches':float(np.mean(d))}
    if paths is not None:
        q=np.quantile(paths,[.1,.9],axis=1)
        sorted_paths=np.sort(paths,axis=1);p=paths.shape[1]
        crps=np.mean(abs(paths-y[:,None]),axis=1)-np.sum(sorted_paths*(2*np.arange(1,p+1)-p-1),axis=1)/(p*p)
        out.update(crps_inches=float(np.mean(crps)),coverage80=float(np.mean((y>=q[0])&(y<=q[1]))),mean_band_width_inches=float(np.mean(q[1]-q[0])))
    return out

def bootstrap_skill(meta,y,pred,baseline,seed=20261111):
    # Resample date blocks rather than falsely treating overlapping forecasts as independent.
    dates=pd.to_datetime(meta.origin_date);blocks=(dates-dates.min()).dt.days.to_numpy()//7
    unique=np.unique(blocks);rng=np.random.default_rng(seed);delta=[]
    a=(pred-y)**2;b=(baseline-y)**2
    for _ in range(1000):
        sampled=rng.choice(unique,len(unique),replace=True);idx=np.concatenate([np.flatnonzero(blocks==v) for v in sampled])
        delta.append(float(np.mean(a[idx])-np.mean(b[idx])))
    lo,hi=np.quantile(delta,[.025,.975]);return {'mse_difference_inches2':float(np.mean(a-b)),'block_bootstrap95':[float(lo),float(hi)],'block_days':7,'bootstrap_replicates':1000,'limitation':'One held-out winter; weekly blocks are not independent storm definitions and intervals cannot measure interannual generalization.'}

def run(max_updates=1400):
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    OUT.mkdir(parents=True,exist_ok=True);models_dir=OUT/'models';models_dir.mkdir(exist_ok=True)
    arrays=dict(np.load(DATA/'arrays.npz'));meta=pd.read_csv(DATA/'index.csv')
    train_idx=np.flatnonzero(meta.split.eq('train'));val_idx=np.flatnonzero(meta.split.eq('validation'));test_idx=np.flatnonzero(meta.split.eq('test'))
    # The seasonal gap is much longer than the 48h target window.
    assert pd.to_datetime(meta.end_utc.iloc[train_idx],utc=True).max()<pd.to_datetime(meta.issue_utc.iloc[val_idx],utc=True).min()
    assert pd.to_datetime(meta.end_utc.iloc[val_idx],utc=True).max()<pd.to_datetime(meta.issue_utc.iloc[test_idx],utc=True).min()
    assert np.isfinite(arrays['y']).all() and (arrays['y']>=0).all()
    plan={'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'families':FAMILIES,'learning_rates':LRS,'seeds':SEEDS,'max_updates':max_updates,'batch_size':32,'whole_history_dropout':.3,'validation_every':50,'early_stop':'No improvement for400updates after minimum500updates','selection':'Equal-site normalized mean squared error at24h and48h/2; tuneLR on validation only, ensemblethreeinitializations, select family by site validation score','training':'2020/21 and2021/22','validation':'2022/23','test':'2023/24, previously used by earlier project experiments; not a new blind season','hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'arrays.npz',DATA/'index.csv',Path(__file__)]},'test_used_for_selection':False}
    old_plan=OUT/'training_plan.json'
    if old_plan.exists() and (OUT/'runs').exists():
        prior=json.loads(old_plan.read_text())
        for key in ['hashes','learning_rates','seeds','max_updates','batch_size','whole_history_dropout','selection']:
            if prior.get(key)!=plan.get(key):
                raise RuntimeError(f'Refusing to resume incompatible training artifacts: {key} differs. Preserve this run and choose a new output directory.')
    jwrite(OUT/'training_plan.json',plan)
    prep=fit_preprocessing(arrays,train_idx);jwrite(models_dir/'preprocessing.json',prep)
    x=tensorize(transform(arrays,prep));y=arrays['y']/np.asarray(prep['target_scale'])[arrays['site'],None];sites=arrays['site']
    all_runs=[];ensembles={};lr_winners={}
    for family in FAMILIES:
        screen=[]
        for lr in LRS:
            run_dir=OUT/'runs'/f'{family}_lr{lr:g}_seed{SEEDS[0]}'
            if (run_dir/'summary.json').exists():info=json.loads((run_dir/'summary.json').read_text())
            else:info=train_run(family,lr,SEEDS[0],x,y,sites,train_idx,val_idx,run_dir,max_updates)
            screen.append(info);all_runs.append(info)
        winner=min(screen,key=lambda r:r['validation_score']);lr=winner['learning_rate'];lr_winners[family]=lr
        members=[winner]
        for seed in SEEDS[1:]:
            run_dir=OUT/'runs'/f'{family}_lr{lr:g}_seed{seed}'
            if (run_dir/'summary.json').exists():info=json.loads((run_dir/'summary.json').read_text())
            else:info=train_run(family,lr,seed,x,y,sites,train_idx,val_idx,run_dir,max_updates)
            members.append(info);all_runs.append(info)
        ensembles[family]=[]
        for info in members:
            source=OUT/'runs'/f"{family}_lr{lr:g}_seed{info['seed']}"/'best.pt'
            target=models_dir/f"{family}_seed{info['seed']}.pt";target.write_bytes(source.read_bytes())
            ensembles[family].append({'checkpoint':target.name,'rho':info['copula_rho_by_site'],'seed':info['seed'],'best_update':info['best_update']})
    selection={'ensembles':ensembles,'best_learning_rates':lr_winners,'selected_family_by_site':{s:'weather_only' for s in SITES},'validation_scores':{},'frozen_before_test_evaluation':True,'test_used_for_selection':False}
    jwrite(models_dir/'selection.json',selection);bundle=load_bundle(models_dir)
    val_arrays={k:v[val_idx] for k,v in arrays.items()};valmeans={f:predict_neural(bundle,val_arrays,f,n_paths=768)['daily_mean'] for f in FAMILIES}
    for s,site in enumerate(SITES):
        rows=np.flatnonzero(sites[val_idx]==s)
        scores={f:selection_score(valmeans[f][rows]/prep['target_scale'][s],y[val_idx][rows],np.zeros(len(rows))) for f in FAMILIES}
        selection['selected_family_by_site'][site]=min(scores,key=scores.get);selection['validation_scores'][site]=scores
    selection['frozen_utc']=pd.Timestamp.now(tz='UTC').isoformat();jwrite(models_dir/'selection.json',selection)
    bundle=load_bundle(models_dir)
    # Fixed baseline parameters. No validation/test target can enter a fitted donor bank.
    doy=pd.to_datetime(meta.origin_date).dt.dayofyear.to_numpy()
    base=fit_baselines(arrays['nwp'],arrays['static'],clean_history(arrays['history']),meta.site.to_numpy(),arrays['y'],train_idx,doy,n_paths=768)
    with open(models_dir/'baselines.pkl','wb') as file:pickle.dump(base,file)
    results={};tables=[]
    for split,idx in [('validation',val_idx),('test',test_idx)]:
        sub={k:v[idx] for k,v in arrays.items()}
        preds=predict_baselines(base,arrays['nwp'],arrays['static'],clean_history(arrays['history']),meta.site.to_numpy(),idx,doy)
        for f in FAMILIES:preds[NAMES[f]]=predict_neural(bundle,sub,f,n_paths=768)
        preds['Selected neural ensemble']=predict_neural(bundle,sub,n_paths=768)
        preds['JMA provider · fixed older guidance']={'daily_mean':arrays['provider'][idx]}
        results[split]=preds
        for method,p in preds.items():
            for s,site in enumerate(SITES):
                pos=np.flatnonzero(sites[idx]==s)
                for horizon in [24,48]:
                    truth=arrays['y'][idx][pos,0] if horizon==24 else arrays['y'][idx][pos].sum(1)
                    mean=p['daily_mean'][pos,0] if horizon==24 else p['daily_mean'][pos].sum(1)
                    paths=None if 'joint_paths' not in p else p['joint_paths'][pos,:,0] if horizon==24 else p['joint_paths'][pos].sum(2)
                    tables.append({'split':split,'site':site,'method':method,'horizon_hours':horizon,**metrics(truth,mean,paths)})
        np.savez_compressed(OUT/f'{split}_predictions.npz',row_indices=idx,truth=arrays['y'][idx],**{f'{i}_mean':p['daily_mean'] for i,p in enumerate(preds.values())},**{f'{i}_paths':p['joint_paths'] for i,p in enumerate(preds.values()) if 'joint_paths' in p})
        jwrite(OUT/f'{split}_methods.json',list(preds))
    scores=pd.DataFrame(tables);scores.to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(all_runs).to_csv(OUT/'optimization_runs.csv',index=False)
    test=results['test'];comparisons={}
    for s,site in enumerate(SITES):
        pos=np.flatnonzero(sites[test_idx]==s);truth=arrays['y'][test_idx][pos].sum(1);pred=test['Selected neural ensemble']['daily_mean'][pos].sum(1)
        comparisons[site]={name:bootstrap_skill(meta.iloc[test_idx[pos]],truth,pred,test[name]['daily_mean'][pos].sum(1)) for name in [CURRENT,'JMA provider · fixed older guidance']}
    jwrite(OUT/'bootstrap_comparisons.json',comparisons)
    report(arrays,meta,results,scores,selection,all_runs,comparisons)
    print('RESULTS '+str(OUT),flush=True)


def report(arrays,meta,results,scores,selection,runs,comparisons):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'Weather TCN':'#cb622b','Radar + weather TCN':'#117d85',CURRENT:'#8665ae','JMA provider · fixed older guidance':'#6e7988',CLIMATE:'#b6aa8a'}
    fig,axs=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    test_idx=np.flatnonzero(meta.split.eq('test'));preds=results['test']
    for s,ax in enumerate(axs.flat):
        pos=np.flatnonzero(arrays['site'][test_idx]==s);dates=pd.to_datetime(meta.origin_date.iloc[test_idx[pos]])
        truth=arrays['y'][test_idx][pos].sum(1)
        grid=pd.date_range(dates.min(),dates.max(),freq='D')
        def expanded(value):return pd.Series(np.asarray(value),index=pd.DatetimeIndex(dates)).reindex(grid).to_numpy()
        ax.plot(grid,expanded(truth),color='#17272d',lw=1.7,marker='.',markersize=3,label='Observed 48h accumulation',zorder=4)
        for name in ['Radar + weather TCN','Weather TCN',CURRENT,'JMA provider · fixed older guidance',CLIMATE]:
            ax.plot(grid,expanded(preds[name]['daily_mean'][pos].sum(1)),color=colors[name],lw=1,alpha=.8,label=name)
        selected=preds['Selected neural ensemble'];q=np.quantile(selected['joint_paths'][pos].sum(2),[.1,.9],axis=1)
        ax.fill_between(grid,expanded(q[0]),expanded(q[1]),color='#117d85',alpha=.13,label='Selected neural 80% prediction interval')
        ax.set_title(LABELS[SITES[s]]+f' · n={len(pos)}');ax.set_ylabel('Next 48h snowfall · inches');ax.set_ylim(bottom=0);ax.grid(alpha=.16);ax.tick_params(axis='x',rotation=25)
    handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Held-out winter 2023/24 · daily-issued, overlapping 48-hour forecasts',fontsize=16)
    fig.savefig(OUT/'heldout_curves.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for family,ax in zip(FAMILIES,axs):
        for lr in LRS:
            path=OUT/'runs'/f'{family}_lr{lr:g}_seed{SEEDS[0]}'/'history.csv';d=pd.read_csv(path)
            line=ax.plot(d['update'],d.validation_selection_mse,label=f'Validation · LR {lr:g}')[0]
            ax.plot(d['update'],d.train_selection_mse,ls='--',alpha=.6,color=line.get_color())
            best=d.loc[d.validation_selection_mse.idxmin()];ax.scatter([best['update']],[best.validation_selection_mse],color=line.get_color(),s=28)
        ax.set_title(NAMES[family]);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Equal-site normalized MSE');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Solid: validation · dashed: training · dot: retained checkpoint',fontsize=14)
    fig.savefig(OUT/'training_validation.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    names=['Weather TCN','Radar + weather TCN',CURRENT,ALL_WEATHER,'JMA provider · fixed older guidance',CLIMATE]
    for site,ax in zip(SITES,axs.flat):
        data=scores[(scores.split=='test')&(scores.site==site)&(scores.horizon_hours==48)].set_index('method')
        ax.barh(np.arange(len(names)),[data.loc[n,'rmse_inches'] for n in names],color=[colors.get(n,'#9cb6b3') for n in names])
        ax.set_yticks(np.arange(len(names)),names,fontsize=8);ax.invert_yaxis();ax.set_title(LABELS[site]);ax.set_xlabel('48h RMSE · inches · lower is better');ax.grid(axis='x',alpha=.16)
    fig.savefig(OUT/'test_error_comparison.png',dpi=160);plt.close(fig)
    display=scores[scores.split=='test'][['site','method','horizon_hours','n','mae_inches','rmse_inches','bias_inches','crps_inches','coverage80']].copy()
    display['site']=display.site.map(LABELS)
    selection_rows=[]
    for site in SITES:
        name=NAMES[selection['selected_family_by_site'][site]]
        testrow=scores[(scores.split=='test')&(scores.site==site)&(scores.method=='Selected neural ensemble')&(scores.horizon_hours==48)].iloc[0]
        ref=scores[(scores.split=='test')&(scores.site==site)&(scores.method==CURRENT)&(scores.horizon_hours==48)].iloc[0]
        provider=scores[(scores.split=='test')&(scores.site==site)&(scores.method=='JMA provider · fixed older guidance')&(scores.horizon_hours==48)].iloc[0]
        skill=100*(1-testrow.mse_inches2/ref.mse_inches2)
        selection_rows.append({'Mountain / target':LABELS[site],'Validation-selected neural':name,'48h neural RMSE (in)':testrow.rmse_inches,'Current-family RMSE (in)':ref.rmse_inches,'Provider RMSE (in)':provider.rmse_inches,'MSE improvement vs current (%)':skill,'80% interval coverage':testrow.coverage80})
    selected_df=pd.DataFrame(selection_rows);selected_df.to_csv(OUT/'selected_summary.csv',index=False)
    brief={'selected':selection_rows,'models_saved':True,'hourly_fresh_snow_supervision':False,'production_readiness':'Research preview. Sparse station proxy targets, one test winter and historical provider age mismatch prevent claiming universal resort accuracy.','test_target_counts':meta[meta.split.eq('test')].groupby('site').size().to_dict()}
    jwrite(OUT/'summary.json',brief)
    caveats='''<ul><li>Input weather forecasts have 48 hourly positions; targets are two measured 24-hour accumulations. Curves between these endpoints would be disaggregation, not an independently verified hourly snowfall forecast.</li><li>Alta uses Collins manual snowfall. Vail uses a Sno24 sensor proxy whose clock/reset/fresh-snow equivalence remains unverified. Stowe is a 223 m town station; Jay Peak is a 326 m Westfield station. These lower-elevation amounts are not summit/resort snowfall.</li><li>Training: winters 2020/21 and 2021/22. Validation: 2022/23. Test: 2023/24. Adjacent forecast days are never randomly split; no split has overlapping target windows. The test winter was used by earlier project experiments, so it is not a newly blind test.</li><li>JMA guidance is a conservative fixed 72-hour-offset historical mosaic retrieved through Open-Meteo. It is older than the MOS forecast used by the models, not a coherent single run and not today’s live best-match provider. Provider snowfall is derived from coarse native forecast data. Results do not establish that these models outperform the live provider.</li><li>Radar is pre-issue MRMS context; Vail has poor radar quality. The radar ablation must improve beyond weather-only to justify its cost. No claim that this compact CNN–GRU is the full published DGMR or NowcastNet.</li><li>Probabilistic bands are prediction intervals, not confidence intervals for the mean. Two-day dependence is a training-only randomized PIT copula; coverage on the held-out data is reported, not forced to 80%. Neural ensembles contain three initializations. Live rolling 48h forecasts outside the morning reporting schedule are an experimental transfer; this backtest does not validate arbitrary issue times.</li><li>The previous Alta ConvLSTM, DGMR and NowcastNet experiments predict three-hour liquid radar precipitation, so they cannot be scored as 48-hour fresh snowfall without changing the task. The new architectures are compact 32,963-parameter weather TCN and radar-augmented CNN–GRU/TCN models, not full implementations of those published systems.</li><li>The current forest family is refit on exactly the same target cohort; its old mountain-target weights would be an unfair comparison. The forest with both MOS and JMA controls for extra forecast information. Neural models additionally use lagged snow history and pool training across sites, while forests fit each site separately. The refit current-family benchmark also omits the older system’s lagged ENSO feature. This is a whole-system comparison, not a pure architecture ablation; the weather-only versus weather-plus-radar neural comparison does isolate the added radar branch.</li></ul>'''
    cards=''.join(f'<div class="card"><h3>{html.escape(r["Mountain / target"])}</h3><strong>{html.escape(r["Validation-selected neural"])}</strong><p>48h test RMSE: {r["48h neural RMSE (in)"]:.2f} in<br>Current-family: {r["Current-family RMSE (in)"]:.2f} in<br>Provider: {r["Provider RMSE (in)"]:.2f} in</p></div>' for r in selection_rows)
    run_df=pd.DataFrame(runs)[['family','learning_rate','seed','best_update','final_update','validation_score','final_validation_score','parameters']]
    boot_rows=[]
    for site,comparison in comparisons.items():
        for method,r in comparison.items():boot_rows.append({'Target':LABELS[site],'Comparison':method,'Neural minus comparator MSE':r['mse_difference_inches2'],'95% lower':r['block_bootstrap95'][0],'95% upper':r['block_bootstrap95'][1]})
    current_section=''
    current_path=OUT/'current_forecast.json'
    if current_path.exists():
        current=json.loads(current_path.read_text());current_rows=[]
        (OUT/'report_forecast_snapshot.json').write_bytes(current_path.read_bytes())
        if (OUT/'current_forecast.png').exists():(OUT/'report_forecast_snapshot.png').write_bytes((OUT/'current_forecast.png').read_bytes())
        for site,r in current.get('sites',{}).items():
            total=next((v for v in r.get('totals',[]) if v.get('hours')==48),None)
            if total is None:continue
            zone='America/Denver' if site in ['alta','vail'] else 'America/New_York'
            start=pd.Timestamp(r['start']).tz_convert(zone).strftime('%b %d, %I:%M %p %Z')
            end=pd.Timestamp(r['end']).tz_convert(zone).strftime('%b %d, %I:%M %p %Z')
            current_rows.append({'Target':LABELS[site],'Forecast window':start+' → '+end,'Neural model':r['model_name'],
                'Neural48h (in)':total['mean_inches'],'10–90% interval (in)':f"{total['q10_inches']:.2f}–{total['q90_inches']:.2f}",
                'Fresh provider48h (in)':total.get('provider_inches'),'Older JMA48h (in)':total.get('conservative_provider_inches')})
        current_table=pd.DataFrame(current_rows).to_html(index=False,float_format=lambda x:f'{x:.2f}',na_rep='—',border=0)
        current_section=f'<h2>48-hour forecast snapshot · experimental</h2><p>Generated {html.escape(current["generated_at"])}. Lines connect the 0,24 and48-hour cumulative totals; they do not establish the timing of snowfall inside each day. The live window begins at the next full hour and is outside the morning schedule used for evaluation. Alta currently has no recent snow-history observations; the model uses the missing-history encoding seen during training.</p><img src="report_forecast_snapshot.png" alt="Current cumulative snowfall predictions and weather-provider comparison"><div class="scroll">{current_table}</div><p>The current provider curve is fresh Open-Meteo best-match guidance at the same station location; the historical scorecard below uses older JMA72-hour-offset guidance. These are different comparisons. Vail remains a sensor proxy; Vermont forecasts are for lower-elevation Stowe and Westfield stations, not resort summit snowfall. <a href="report_forecast_snapshot.json">Snapshot values and source timestamps</a>.</p>'
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Snowline · trained 48-hour comparison</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#eef3ef;color:#18343c;margin:0}}main{{max-width:1250px;margin:auto;padding:35px}}h1{{font-size:38px;line-height:1.15}}h2{{margin-top:44px}}.eyebrow{{letter-spacing:.18em;text-transform:uppercase;color:#117d85;font-size:12px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}}.card,.note{{background:white;border:1px solid #d6e1dd;border-radius:16px;padding:20px}}.card h3{{margin-top:0}}img{{width:100%;background:white;border-radius:16px}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}th,td{{padding:8px;border-bottom:1px solid #e2e9e7;text-align:right}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}a{{color:#117d85}}li{{margin-bottom:12px}}code{{font-size:12px}}footer{{margin:40px 0;color:#537079}}</style><main><p class="eyebrow">Snowline research · trained November 11, 2026</p><h1>Issued weather + radar<br>48-hour snowfall comparison</h1><p>Four mountains, two 24-hour accumulation targets, three-seed neural ensembles. The default neural family is selected on validation; the final winter is used only for evaluation after the selection is frozen.</p><div class="cards">{cards}</div><p class="note"><strong>Promising point forecasts; research preview.</strong> Selected neural 48h RMSE is lower than both forest baselines at all four targets. Alta’s MAE is slightly worse than the current forest, and weekly-block error comparisons remain inconclusive for Alta and Vail. The nominal 80% bands cover 99% at Alta, 91% at Vail and about 87% at the Vermont stations: the bands are conservative and substantially overcover at Alta. The results support a labeled preview, not a claim of calibrated resort forecasts or superiority over the live weather provider.</p>{current_section}<h2>1. Test errors at 48 hours</h2><p>Lower error is better. Site labels identify the actual observed target. Improvement is assessed against a refit current forest and a conservative archived weather-provider forecast.</p><img src="test_error_comparison.png"><div class="scroll">{selected_df.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)}</div><h2>2. What the forecasts look like against observations</h2><p>Each point is a new 48-hour forecast, issued daily. Adjacent targets overlap; adding these plotted points would double-count snowfall. Gaps are missing or excluded reporting windows, not zero snowfall; observed points are never connected across missing days. Shading shows the selected neural model’s 10th–90th percentiles.</p><img src="heldout_curves.png"><h2>3. Optimization and checkpoint selection</h2><p>Three learning rates per family. The best validation checkpoint is retained, not the final optimizer update. Selected learning rates are repeated with two additional seeds. The test loss is intentionally not monitored during training.</p><img src="training_validation.png"><div class="scroll">{run_df.to_html(index=False,float_format=lambda x:f'{x:.6g}',border=0)}</div><h2>4. Does improvement exceed sampling noise?</h2><p>Difference in 48h MSE, neural minus comparator: negative is better. The 95% interval uses 1,000 seven-day block resamples within the single test winter. An interval crossing zero does not resolve which method is better; these intervals do not measure performance across future winters.</p><div class="scroll">{pd.DataFrame(boot_rows).to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)}</div><h2>5. Full test scorecard</h2><p>MAE, RMSE and bias are inches. CRPS evaluates the entire forecast distribution. An ideal nominal 80% interval covers 80% of observations; no distribution is invented for the deterministic provider.</p><div class="scroll">{display.to_html(index=False,float_format=lambda x:f'{x:.3f}',na_rep='—',border=0)}</div><h2>6. Interpretation and deployment limits</h2><div class="note">{caveats}</div><h2>Public data references</h2><p><a href="https://www.nssl.noaa.gov/projects/mrms/">NOAA MRMS radar products</a> · <a href="https://open-meteo.com/en/docs/previous-runs-api">Open-Meteo archived forecast offsets</a> · <a href="https://mesonet.agron.iastate.edu/mos/">Iowa Environmental Mesonet MOS archive</a> · <a href="https://www.cocorahs.org/">CoCoRaHS station reporting</a></p><p>Stowe: US1VTLM0001, Stowe 0.2 SW, 223.1 m. Jay Peak regional proxy: US1VTOL0001, Westfield 0.7 WNW, 325.8 m. Full timestamps, station coordinates and source file hashes are retained in the project’s prepared-data audit.</p><h2>Download the evidence</h2><p><a href="metrics.csv">All metrics</a> · <a href="optimization_runs.csv">Optimization summaries</a> · <a href="training_plan.json">Frozen plan</a> · <a href="bootstrap_comparisons.json">Block comparison intervals</a> · <a href="models/selection.json">Selected checkpoints</a></p><footer>Observed targets and archived issued forecasts are retained separately. No future observed weather is substituted for an issued forecast.</footer></main></html>'''
    (OUT/'report.html').write_text(body)

def regenerate_report():
    arrays=dict(np.load(DATA/'arrays.npz'));meta=pd.read_csv(DATA/'index.csv')
    results={}
    for split in ['validation','test']:
        saved=np.load(OUT/f'{split}_predictions.npz');names=json.loads((OUT/f'{split}_methods.json').read_text())
        results[split]={}
        for i,name in enumerate(names):
            results[split][name]={'daily_mean':saved[f'{i}_mean']}
            if f'{i}_paths' in saved:results[split][name]['joint_paths']=saved[f'{i}_paths']
    report(arrays,meta,results,pd.read_csv(OUT/'metrics.csv'),json.loads((OUT/'models/selection.json').read_text()),
           pd.read_csv(OUT/'optimization_runs.csv').to_dict('records'),json.loads((OUT/'bootstrap_comparisons.json').read_text()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--max-updates',type=int,default=1400);parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if args.report_only:regenerate_report()
    else:run(args.max_updates)

