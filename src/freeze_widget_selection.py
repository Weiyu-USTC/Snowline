"""Freeze per-horizon widget choices using validation scores only.

The original MRMS artifacts include validation predictions and exactly match
the deployed checkpoints. The later tuned MRMS predictions contain test only
and are deliberately not used for deployment selection here.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/nwp-weather/forecast-portfolio'

def h(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paired=pd.read_csv(ROOT/'outputs/nexrad-short/paired_predictions.csv',dtype={'case_id':str})
    metrics=pd.read_csv(ROOT/'outputs/nexrad-short/paired_metrics.csv')
    validation=paired[paired.site.eq('alta')&paired.split.eq('validation')]
    wanted=set(validation.case_id)
    arrays={};lookups={};hashes={}
    for m in ['convlstm','dgmr','nowcastnet']:
        base=ROOT/'outputs/alta-nowcast/radar_predictions';manifest=json.loads((base/f'{m}_manifest.json').read_text())
        checkpoint=ROOT/'outputs/alta-nowcast/models/selected'/f'{m}.pt'
        if h(checkpoint)!=manifest['checkpoint_sha256']:raise ValueError(f'{m}: saved validation predictions do not match deployed checkpoint')
        index=np.load(base/f'{m}_index.npz')
        lookup={str(k):i for i,k in enumerate(index['case_id']) if index['split'][i]=='validation'}
        wanted &=set(lookup);arrays[m]=np.load(base/f'{m}.npy',mmap_mode='r');lookups[m]=lookup;hashes[m]=h(checkpoint)
    # Recreate exactly the quality-qualified MRMS local target used by NEXRAD.
    z=np.load(ROOT/'data/alta_nowcast/radar/radar_sequences.npz')
    ids={str(v):i for i,v in enumerate(z['case_id'])}
    lat,lon=z['latitude'],z['longitude'];iy=int(np.argmin(abs(lat-40.57607)));ix=int(np.argmin(abs(lon+111.63889)))
    rates=z['rate'];valid=z['valid_fraction'];quality=z['rqi_hourly'];rt=z['radar_time'];qt=z['quality_time'];z.close()
    records=[]
    for cid in sorted(wanted):
        i=ids[cid];rr=rates[i,:,iy-1:iy+2,ix-1:ix+2];vv=valid[i,:,iy-1:iy+2,ix-1:ix+2]
        qq=quality[i,:,iy-1:iy+2,ix-1:ix+2];qix=np.searchsorted(qt[i],rt[i],side='right')-1;qq=qq[qix]
        mask=np.isfinite(rr)&(rr>=0)&(vv>=.9)&np.isfinite(qq)&(qq>=.5)
        count=mask.sum((1,2));mask=mask[10:];count=count[10:]
        target=np.divide(np.where(mask,rr[10:],0).sum((1,2)),count,out=np.full(19,np.nan),where=count>=5)
        truth=[float(np.trapezoid(target[k*6:k*6+7],dx=1/6)) for k in range(3)]
        ref=validation[validation.case_id.eq(cid)].iloc[0]
        if not np.allclose(truth,[ref.truth1,ref.truth2,ref.truth3],atol=1e-6):raise ValueError('Target replay differs from paired benchmark')
        for m,a in arrays.items():
            pred=np.asarray(a[lookups[m][cid],:,1:,iy-1:iy+2,ix-1:ix+2]).mean(0)
            rate=np.divide(np.where(mask,pred,0).sum((1,2)),count,out=np.full(19,np.nan),where=count>=5)
            values=[float(np.trapezoid(rate[k*6:k*6+7],dx=1/6)) for k in range(3)]
            records.append(dict(site='alta',model='original_MRMS_'+m,split='validation',case_id=cid,
                **{f'pred{k+1}':values[k] for k in range(3)},**{f'truth{k+1}':truth[k] for k in range(3)}))
    local=validation[validation.case_id.isin(wanted)]
    compare=pd.concat([local,pd.DataFrame(records)],ignore_index=True)
    compare.to_csv(OUT/'deployment_short_validation_predictions.csv',index=False)
    candidates=[]
    for (site,model),g in compare.groupby(['site','model']):
        if model.startswith('previous_MRMS_'):continue
        y=g[['truth1','truth2','truth3']].to_numpy();p=g[['pred1','pred2','pred3']].to_numpy()
        for horizon in [1,3]:
            error=p[:,:horizon].sum(1)-y[:,:horizon].sum(1)
            candidates.append(dict(site=site,method=model,hours=horizon,n=len(g),validation_rmse=float(np.sqrt(np.mean(error*error)))))
    for r in metrics[metrics.split.eq('validation')&~metrics.site.eq('alta')].itertuples():
        for horizon in [1,3]:candidates.append(dict(site=r.site,method=r.model,hours=horizon,n=r.n,validation_rmse=getattr(r,f'RMSE_{horizon}h_mm')))
    cf=pd.DataFrame(candidates);cf.to_csv(OUT/'deployment_short_validation_metrics.csv',index=False)
    eligible={'NEXRAD_selected','NEXRAD_HRRR_portfolio','HRRR','DPR_persistence','DPR_advection',*[f'original_MRMS_{m}' for m in arrays]}
    selection={'schema_version':1,'frozen_at':pd.Timestamp.now(tz='UTC').isoformat(),
        'criterion':'Lowest validation RMSE separately for each horizon among compatible, available candidates. Test scores never used.',
        'validation_winter':'2022/23; already reused for model tuning, not an independent selection season.',
        'display_label':'Our model','sites':{},'checkpoint_sha256':hashes,
        'short_validation_note':f'Original live MRMS checkpoint predictions replayed on {len(wanted)} identical qualified Alta validation dates; newer tuned test-only arrays excluded.',
        'source_files':[dict(path=str(p.relative_to(ROOT)),sha256=h(p)) for p in [ROOT/'outputs/nexrad-short/paired_predictions.csv',ROOT/'outputs/nwp-weather/neural48-trained/metrics.csv',OUT/'hrrr24_selection.json']]}
    neural=pd.read_csv(ROOT/'outputs/nwp-weather/neural48-trained/metrics.csv')
    twentyfour=json.loads((OUT/'hrrr24_selection.json').read_text())
    names={'Weather TCN':'weather_only','Radar + weather TCN':'weather_radar'}
    for site in ['alta','vail','mansfield','jay']:
        choices={}
        for hours in [1,3]:
            g=cf[cf.site.eq(site)&cf.hours.eq(hours)&cf.method.isin(eligible)]
            if len(g):
                best=g.sort_values(['validation_rmse','method']).iloc[0]
                choices[str(hours)]=dict(method_id=best.method,quantity='liquid_precipitation',unit='mm',validation_rmse=float(best.validation_rmse),validation_n=int(best.n),candidates=g.to_dict('records'))
            else:choices[str(hours)]=dict(method_id='HRRR',quantity='liquid_precipitation',unit='mm',selection='Fallback: no quality-qualified radar verification for this mountain.',candidates=[])
        choices['6']=dict(method_id='HRRR_native_snowfall',quantity='snowfall',unit='inches',selection='Native hourly snowfall guidance; no trained6h fresh-snow target available.')
        choices['12']=dict(method_id='Alta12_operational_portfolio' if site=='alta' else 'HRRR_native_snowfall',quantity='snowfall',unit='inches',selection='Fixed11/23UTC Alta12h validation selection; other mountains native hourly HRRR.')
        choices['24']=dict(method_id='HRRR_neural24_portfolio',quantity='snowfall',unit='inches',weights=twentyfour[site])
        g=neural[neural.split.eq('validation')&neural.site.eq(site)&neural.horizon_hours.eq(48)&neural.method.isin(names)]
        best=g.loc[g.rmse_inches.idxmin()]
        choices['48']=dict(method_id='neural48_'+names[best.method],family=names[best.method],quantity='snowfall',unit='inches',validation_rmse=float(best.rmse_inches),validation_n=int(best.n),candidates=g[['method','rmse_inches','n']].to_dict('records'))
        selection['sites'][site]=choices
    (OUT/'deployment_selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False))
    print(json.dumps({s:{k:v['method_id'] for k,v in x.items()} for s,x in selection['sites'].items()},indent=2))

if __name__=='__main__':main()
