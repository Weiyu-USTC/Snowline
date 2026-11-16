"""Preplanned validation selection for operationally unavailable manual history."""
from pathlib import Path
import json,pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from train_alta12 import DATA,OUT,Alta12TCN,features,flat_features,missing_history,nn_predict,metrics,block_bootstrap

def main():
    with open(OUT/'bundle.pkl','rb') as f:b=pickle.load(f)
    a=dict(np.load(DATA/'arrays.npz'));meta=pd.read_csv(OUT/'case_index.csv');a['history']=missing_history(len(meta))
    full=dict(np.load(OUT/'predictions_all.npz'));pred={};prob={}
    for name,item in b['models'].items():
        if item['type']=='neural':
            model=Alta12TCN(**item['kwargs']);model.load_state_dict(item['state'])
            pred[name],prob[name]=nn_predict(model,features(a,item['kind'],item['scales']))
        elif item['type']=='forest':
            x=flat_features(a,item['kind'],item['scales'],item['radar']);pred[name]=item['model'].predict(x);prob[name]=item['occurrence'].predict(x)
        elif item['type']=='physical':
            pred[name]=full[name];prob[name]=1-np.exp(-pred[name])
    records=[]
    for cohort in ['broad','modern']:
        mask=np.ones(len(meta),bool) if cohort=='broad' else meta.modern.to_numpy(bool)
        stages=meta.validation_stage.to_numpy();ix={s:np.flatnonzero(mask&(stages==s)) for s in ['train','tune','blend','calibration','test']}
        candidates=['broad_forest','broad_tcn'] if cohort=='broad' else ['modern_forest','modern_radar_forest','modern_tcn','modern_radar_tcn','hrrr_train_calibrated','broad_forest','broad_tcn']
        tune=ix['tune'];blend=ix['blend'];cal=ix['calibration'];test=ix['test'];y=a['snow']
        rank=sorted(candidates,key=lambda k:np.mean((pred[k][tune]-y[tune])**2));top=rank[:3];mat=np.stack([pred[k][blend] for k in top],1)
        fit=minimize(lambda w:np.mean((mat@w-y[blend])**2),np.ones(len(top))/len(top),method='SLSQP',bounds=[(0,1)]*len(top),constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-12,'maxiter':1000})
        if not fit.success:raise RuntimeError(fit.message)
        weights={k:float(w) for k,w in zip(top,fit.x) if w>1e-7};weights={k:w/sum(weights.values()) for k,w in weights.items()}
        port=sum(w*pred[k] for k,w in weights.items());portprob=sum(w*prob[k] for k,w in weights.items())
        ind=min(candidates,key=lambda k:np.mean((pred[k][blend]-y[blend])**2))
        portname=f'{cohort}_operational_portfolio';pred[portname]=port;prob[portname]=portprob
        if np.mean((port[blend]-y[blend])**2)<.995*np.mean((pred[ind][blend]-y[blend])**2):chosen=portname
        else:chosen=ind;weights={ind:1.}
        q=float(np.quantile(np.abs(pred[chosen][cal]-y[cal]),min(1,np.ceil((len(cal)+1)*.8)/len(cal)),method='higher'))
        p=pred[chosen][test];actual=y[test]
        selection={'name':chosen,'weights':weights,'conformal_half_width_inches':q,'tune_ranking':rank,'blend_mse':float(np.mean((pred[chosen][blend]-y[blend])**2)),'calibration_n':len(cal),
            'history_condition':'Manual snowfall/LPE history absent; same trained models with missing masks. Choice/weights/calibration use validation only.',
            'test_interval_coverage':float(np.mean((actual>=np.maximum(0,p-q))&(actual<=p+q))),
            'test_rmse_difference_vs_climatology_95ci':block_bootstrap(actual,p,full[f'{cohort}_climatology'][test],meta.seven_day_block.to_numpy()[test]),
            'test_metrics':metrics(actual,p,prob[chosen][test])}
        b['selection'][f'{cohort}_missing_history']=selection
        for stage,ids in ix.items():
            for k in candidates+[portname]:records.append({'cohort':cohort,'stage':stage,'method':k,**metrics(y[ids],pred[k][ids],prob[k][ids])})
        print(cohort,json.dumps(selection,indent=2))
    b['manifest']['selection']=b['selection'];b['manifest']['operational_selection_note']='Preplanned missing-history ablation; validation-only weights and calibration; benchmark full-history selection retained separately.'
    with open(OUT/'bundle.pkl','wb') as f:pickle.dump(b,f)
    (OUT/'manifest.json').write_text(json.dumps(b['manifest'],indent=2));np.savez_compressed(OUT/'predictions_missing_history.npz',**pred)
    pd.DataFrame(records).to_csv(OUT/'metrics_missing_history.csv',index=False)
if __name__=='__main__':main()
