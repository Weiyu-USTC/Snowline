"""Live radar forecast and provisional rolling24h Alta snow probabilities."""
from pathlib import Path
import argparse,json,os,hashlib
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.mplconfig'))
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alta_radar_models import load_bundle,predict_ensemble
from evaluate_alta_nowcast import ROOT,OUT,HRRR,LABELS,RADAR_LABELS,weather_features,history_features,local_radar_features
from report_alta_nowcast import COLORS,save


def resample_run(folder,start):
    """Align actual issued HRRR hourly accumulations to a rolling start time.

    An hourly accumulation is distributed uniformly within that hour only for
    fractional-hour overlap. Weather fields use the corresponding hourly value.
    """
    run=pd.to_datetime(folder.name,format='%Y%m%dT%H%MZ',utc=True)
    d={}
    for name in ['precip_mm','snow_water_mm','temperature_k','dewpoint_k','wind_u_ms','wind_v_ms','relative_humidity_pct']:
        z=np.load(folder/(name+'.npz'));p=z['periods']
        ends=(run+pd.to_timedelta(p,unit='h')).as_unit('ns')
        begins=(ends-pd.Timedelta(hours=1)).as_unit('ns')
        vals=[]
        for j in range(24):
            lo=start+pd.Timedelta(hours=j);hi=lo+pd.Timedelta(hours=1)
            seconds=np.maximum(0,np.minimum(ends.asi8,hi.value)-np.maximum(begins.asi8,lo.value))/1e9
            if not np.isclose(seconds.sum(),3600):raise ValueError('Live HRRR horizon does not cover24h')
            vals.append(float(np.sum(z['mean3x3']*seconds/3600)))
        d[name]={'periods':np.arange(1,25),'mean3x3':np.asarray(vals)}
    return d,run


def main(context,run_folder):
    context=Path(context);z=np.load(context)
    # Context filenameparent is a documented10-minute forecastissue.
    issue=pd.to_datetime(context.parent.name,format='%Y%m%d-%H%MZ',utc=True)
    start=issue+pd.Timedelta(minutes=10)
    rate=z['rate'].copy();quality=z['rqi'].copy();valid=z['valid_fraction']
    bad=(valid<.9)|(~np.isfinite(quality))|(quality<.1)
    rate[bad]=np.nan;quality[bad]=0
    x=np.stack([rate,quality],axis=2)
    lat=z['latitude'];lon=z['longitude'];iy=np.argmin(abs(lat-40.57607));ix=np.argmin(abs(lon+111.63889))
    quality_audit={'region_valid_fraction':float(np.isfinite(rate).mean()),
                   'alta_neighborhood_valid_fraction':float(np.isfinite(rate[:,:,iy-1:iy+2,ix-1:ix+2]).mean()),
                   'alta_neighborhood_mean_rqi':float(quality[:,:,iy-1:iy+2,ix-1:ix+2].mean())}
    if quality_audit['region_valid_fraction']<.1:
        raise ValueError('Insufficient current radar coverage; use HRRR-only outlook')
    d,run=resample_run(Path(run_folder),start)
    if run+pd.Timedelta(hours=2)>issue:raise ValueError('HRRR assumedavailableafterissue')
    common={**weather_features(d,start,start_period=1),**history_features(x[0],iy,ix)}
    forecasts={};weight_hashes={}
    for name in LABELS:
        weights=OUT/'models'/'selected'/f'{name}.pt'
        bundle=load_bundle(weights)
        weight_hashes[name]=hashlib.sha256(weights.read_bytes()).hexdigest()
        e=predict_ensemble(bundle,x,members=8,batch_size=1,seed=SEED)[0]
        forecasts[name]=e
        common.update({f'{name}__{k}':v for k,v in local_radar_features(e,iy,ix).items()})
    fits=joblib.load(OUT/'daily_calibrators.joblib')['new_models']
    rows=[]
    for name,b in fits.items():
        xx=pd.DataFrame([common]).reindex(columns=b['columns'])
        p=b['estimator'].predict_distribution(xx)
        rows.append({'model':name,'issue_utc':issue.isoformat(),'start_utc':start.isoformat(),
                     'end_utc':(start+pd.Timedelta(hours=24)).isoformat(),**{k:float(v[0]) for k,v in p.items()}})
    table=pd.DataFrame(rows);table.to_csv(OUT/'current_24h_predictions.csv',index=False)
    arrays={name:e for name,e in forecasts.items()}
    arrays['input_rate']=rate;arrays['input_quality']=quality
    np.savez_compressed(OUT/'current_radar_forecasts.npz',**arrays)
    fig=plt.figure(figsize=(15,5));gs=fig.add_gridspec(1,3,width_ratios=[1.35,1,1],wspace=.55)
    a,b,c=[fig.add_subplot(gs[0,i]) for i in range(3)]
    hours=np.arange(19)/6
    curve_rows=[]
    for name,e in forecasts.items():
        local=e[:,1:20,iy-1:iy+2,ix-1:ix+2].mean((2,3))
        cumulative=np.c_[np.zeros(len(local)),np.cumsum((local[:,1:]+local[:,:-1])/12,axis=1)]
        color=COLORS[RADAR_LABELS[name]]
        a.plot(hours,cumulative.mean(0),color=color,label=RADAR_LABELS[name],lw=2)
        a.fill_between(hours,np.quantile(cumulative,.1,axis=0),np.quantile(cumulative,.9,axis=0),color=color,alpha=.15)
        for j,h in enumerate(hours):curve_rows.append({'model':name,'hours':h,'mean_liquid_mm':cumulative[:,j].mean(),
                                                     'q10_liquid_mm':np.quantile(cumulative[:,j],.1),'q90_liquid_mm':np.quantile(cumulative[:,j],.9)})
    a.plot([0,1,2,3],np.r_[0,np.cumsum(d['precip_mm']['mean3x3'][:3])],color=COLORS['HRRR'],linestyle='--',label='HRRR')
    a.set(xlabel='Hours after forecast window starts',ylabel='Cumulative liquid precipitation (mm)',title='Next 3 hours · radar forecast')
    a.grid(alpha=.2);a.legend(fontsize=8)
    plot_order=['HRRR calibrated (matched)',*LABELS.values()];dd=table.set_index('model').loc[plot_order]
    names=['HRRR calibrated','ConvLSTM + HRRR','DGMR + HRRR','NowcastNet + HRRR']
    yy=np.arange(len(dd))
    for j,(model,r) in enumerate(dd.iterrows()):
        color=COLORS[model]
        b.plot([r.q10,r.q90],[j,j],color=color,lw=5,alpha=.3);b.scatter(r['mean'],j,color=color,s=45)
        c.barh(j,r.p_powder*100,color=color,height=.55)
        c.text(r.p_powder*100+.4,j,f'{r.p_powder:.1%}',va='center',fontsize=9)
    b.set_yticks(yy,names);b.invert_yaxis();b.set(xlabel='Fresh snow (inches / 24h)',title='Next 24 hours · snow amount');b.grid(axis='x',alpha=.2)
    c.set_yticks(yy,names);c.invert_yaxis();c.set(xlabel='Probability (%)',title='Next 24 hours · powder probability')
    c.set_xlim(0,max(5,dd.p_powder.max()*120));c.grid(axis='x',alpha=.2)
    fig.suptitle(f'Experimental Alta forecast · initialized {issue:%b %d, %Y %H:%M UTC}',fontsize=16,fontweight='bold',x=.125,ha='left',y=1.09)
    if d['precip_mm']['mean3x3'].sum()==0:
        fig.text(.125,1.005,'HRRR predicts no precipitation. Positive snow totals below expose a calibration limitation; they are not a validated storm signal.',fontsize=10,color='#974f16')
    fig.text(.125,-.06,'Rolling 24h outlook is provisional: calibration used 11 UTC reporting windows. Radar spread is uncalibrated; daily bars show 10th–90th percentiles.',fontsize=9)
    save(fig,'current_forecast')
    pd.DataFrame(curve_rows).to_csv(OUT/'current_short_curves.csv',index=False)
    manifest={'generated_utc':pd.Timestamp.now(tz='UTC').isoformat(),'forecast_issue_utc':issue.isoformat(),
              'radar_context':str(context),'hrrr_run_utc':run.isoformat(),'start_utc':start.isoformat(),
              'end_utc':(start+pd.Timedelta(hours=24)).isoformat(),'new_training':False,
              'models':'Same frozen neural models and development-calibrated forests as benchmark',
              'measurement_caveat':'Short curves are liquid-equivalent radar precipitation, not observed hourly new snow.',
              'daily_caveat':'Rolling24h target clock differs from fixed11UTC manual training windows; prospective calibration unverified.',
              'hrrr_24h_liquid_mm':float(d['precip_mm']['mean3x3'].sum()),
              'hrrr_24h_snow_water_mm':float(d['snow_water_mm']['mean3x3'].sum()),'radar_quality':quality_audit,
              'checkpoint_sha256':weight_hashes}
    (OUT/'current_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(table.to_string(index=False),flush=True)


SEED=20261105
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);p.add_argument('--hrrr-run',required=True);a=p.parse_args()
    main(a.context,a.hrrr_run)
