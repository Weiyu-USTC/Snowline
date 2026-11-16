"""Assemble separately identified HRRR hourly and learned accumulation forecasts.

Hourly snow remains native HRRR ASNOW, not invented hourly neural labels.
NEXRAD predicts short-range liquid amount; it is not silently called snow depth.
The final few hours beyond the native48h run use an identified provider tail.
"""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/nwp-weather/forecast-portfolio'
SITES=['alta','vail','mansfield','jay']
NAMES={'alta':'Alta','vail':'Vail','mansfield':'Stowe','jay':'Jay Peak'}

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x

def finite_amount(value):
    try:value=float(value)
    except (TypeError,ValueError):return None
    return value if np.isfinite(value) and value>=0 else None

def portfolio24_result(native_hours,model24,selection,end,source_reason=None):
    """A missing component leaves the entire accumulation and range missing."""
    values=[finite_amount(v) for v in native_hours]
    model=finite_amount(model24)
    reason=source_reason
    if reason is None and (len(values)!=24 or any(v is None for v in values)):
        reason='HRRR does not cover the complete 24-hour window.'
    if reason is None and model is None:reason='The neural 24-hour component is unavailable.'
    result=dict(available=False,mean_inches=None,lower_inches=None,upper_inches=None,end=end,
        selection=selection,validation_scope='Fixed reporting24h totals; live rolling interval unvalidated. Residual range descriptive, not a calibrated hourly confidence band.')
    if reason:
        result['reason']=reason
        return result
    total=sum(values)
    if selection['native_hrrr_component']=='hrrr_calibrated':total*=selection['scale_fit_on_train']
    mean=selection['hrrr_weight']*total+selection['prior_neural_weight']*model
    interval=finite_amount(selection.get('validation_residual_p80_inches'))
    result.update(available=True,mean_inches=mean,
        lower_inches=None if interval is None else max(0,mean-interval),
        upper_inches=None if interval is None else mean+interval)
    return result

def native_snapshot(issue):
    """Use the paired files named by the immutable snapshot pointer."""
    native=json.loads((ROOT/'data/hrrr_portfolio/current.json').read_text())
    frames=[]
    for key,legacy in [('weather_path','data/hrrr_portfolio/current_weather.csv'),
                       ('snowfall_path','data/hrrr_portfolio/current_native_snowfall.csv')]:
        path=(ROOT/native.get(key,legacy)).resolve()
        if not path.is_relative_to(ROOT.resolve()):raise ValueError('HRRR source path is outside the project')
        d=pd.read_csv(path)
        if not pd.to_datetime(d.run_time_utc,utc=True).eq(pd.Timestamp(native['run_time_utc'])).all():
            raise ValueError('HRRR source run differs from its snapshot metadata')
        d['valid_time_utc']=pd.to_datetime(d.valid_time_utc,utc=True)
        if d.duplicated(['site','valid_time_utc']).any():raise ValueError('Duplicate native HRRR intervals')
        frames.append(d)
    run=pd.Timestamp(native['run_time_utc']);available=pd.Timestamp(native['assumed_available_utc'])
    reason=None
    if issue-run>pd.Timedelta(hours=12):reason='Cached HRRR initialization is too old; refresh HRRR.'
    elif available>issue:reason='HRRR source is not available at this forecast issue.'
    return native,frames[0],frames[1],reason

def alta12(issue):
    from prepare_alta12 import static_features,HRRR_FIELDS
    from prepare_neural48_nwp import canonical,sequence
    from train_alta12 import predict12
    # The next fixed11/23UTC target, retaining the trained00MOS/06HRRR cycle.
    starts=pd.date_range(issue.normalize(),periods=4,freq='12h')+pd.Timedelta(hours=11)
    start=next(s for s in starts if s>issue)
    run=start.normalize();hrrr_run=run+pd.Timedelta(hours=6)
    if hrrr_run+pd.Timedelta(hours=3)>issue:
        raise ValueError('The trained06UTC HRRR cycle is not available yet')
    path=ROOT/'data/hrrr_portfolio/runs'/hrrr_run.strftime('%Y%m%dT%H00Z')/'hourly.csv'
    h=pd.read_csv(path);h=h[h.site.eq('alta')]
    h['valid_time_utc']=pd.to_datetime(h.valid_time_utc,utc=True)
    ends=pd.date_range(start+pd.Timedelta(hours=1),periods=12,freq='h')
    wx=h.set_index('valid_time_utc').reindex(ends)[HRRR_FIELDS].to_numpy(np.float32)
    if not np.isfinite(wx).all():raise ValueError('Alta12 HRRR window incomplete')
    paths=sorted((ROOT/'data/dashboard/current/raw').glob('station_KSLC_gfs_mos_*.csv'),reverse=True)
    group=None;mospath=None
    for p in paths:
        d=canonical(pd.read_csv(p),cycle_hours=(0,));g=d[d.runtime.eq(run)]
        if len(g):group=g;mospath=p;break
    if group is None:raise ValueError('No matching issued00UTC MOS cycle')
    if run+pd.Timedelta(hours=6)>issue:raise ValueError('MOS not available at issue')
    # Preserve the fixed target-relative coordinates used during training.
    # This is a feature reference, NOT a claimed future issue/availability time.
    # All source-availability checks above use the real issue timestamp.
    reference_issue=start-pd.Timedelta(minutes=10)
    mos=sequence(group,(start-run).total_seconds()/3600,(reference_issue-run).total_seconds()/3600)[:12]
    arrays={'mos':mos[None],'hrrr':wx[None],'static':static_features(start,reference_issue)[None]}
    from train_alta12_observations import observation_vector
    obs_path=ROOT/'data/mountain_hourly/observations.csv.gz'
    if obs_path.exists():
        observations=pd.read_csv(obs_path,low_memory=False)
        arrays['observations']=observation_vector(observations,issue)[None]
    p=predict12(ROOT/'outputs/nwp-weather/alta12-trained',arrays)
    return clean(dict(start=start.isoformat(),end=(start+pd.Timedelta(hours=12)).isoformat(),issued_at=issue.isoformat(),
        mean_inches=p['mean_inches'][0],q10_inches=p['q10_inches'][0],q90_inches=p['q90_inches'][0],
        method=p['selected_method'][0],selection_route=p['selection_route'][0],components={k:v[0] for k,v in p['components_inches'].items()},
        feature_reference_issue=reference_issue.isoformat(),
        mos_run=run.isoformat(),hrrr_run=hrrr_run.isoformat(),mos_source=str(mospath.relative_to(ROOT)),
        history='Recent manual snowfall unavailable; model missing-history configuration.',
        validation_scope='Fixed11/23UTC12h amounts; live issue gap differs from the trained10min gap. No hourly-neural claim.'))

def build():
    OUT.mkdir(parents=True,exist_ok=True);issue=pd.Timestamp.now(tz='UTC')
    old=json.loads((ROOT/'outputs/nwp-weather/neural48-trained/current_forecast.json').read_text())
    radar=json.loads((ROOT/'outputs/nexrad-short/current_forecast.json').read_text())
    selection=json.loads((OUT/'hrrr24_selection.json').read_text())
    native,weather,snow,native_reason=native_snapshot(issue)
    # Match the saved current neural forecast exactly, never retimestamp its totals.
    start=pd.Timestamp(old['sites']['alta']['start'])
    if start<issue:raise ValueError('Refresh the neural forecast first: its window has started.')
    ends=pd.date_range(start+pd.Timedelta(hours=1),periods=48,freq='h')
    provider_path=ROOT/old['sites']['alta']['sources']['provider']['best_match']['path']
    provider=json.loads(provider_path.read_text())
    if len(provider)!=len(SITES):raise ValueError('Provider response site count differs')
    output=dict(generated_at=issue.isoformat(),start=start.isoformat(),end=ends[-1].isoformat(),
        native_hrrr=native,sites={},hourly_snowfall_validated=False,status='ready',warnings=[],
        description='Native HRRR hourly snowfall, short-range liquid radar portfolio, and separate learned accumulation checkpoints.')
    for i,site in enumerate(SITES):
        h=snow[snow.site.eq(site)].set_index('valid_time_utc').reindex(ends)
        w=weather[weather.site.eq(site)].set_index('valid_time_utc').reindex(ends)
        if native_reason:
            h.loc[:,'snowfall_inches']=np.nan
            w.loc[:,['temperature_k','precip_mm']]=np.nan
        pdict=provider[i]
        if pdict['hourly_units']['snowfall']!='cm':raise ValueError('Unexpected provider snowfall unit')
        p=pd.DataFrame(pdict['hourly']);p['time']=pd.to_datetime(p.time,utc=True)
        p=p.set_index('time').reindex(ends)
        neural=old['sites'][site]
        if pd.Timestamp(neural['start'])!=start:raise ValueError('Model forecast starts differ')
        hourly=[]
        for j,end in enumerate(ends):
            hr=float(h.snowfall_inches.iloc[j]);pv=float(p.snowfall.iloc[j])/2.54
            source='HRRR native snowfall' if np.isfinite(hr) else 'Open-Meteo beyond HRRR range'
            if end<=pd.Timestamp(native['forecast_end_utc']) and not np.isfinite(hr):
                source='Stale HRRR forecast' if native_reason else 'Missing HRRR forecast';value=None
            else:value=hr if np.isfinite(hr) else pv
            hourly.append(dict(start=(end-pd.Timedelta(hours=1)).isoformat(),end=end.isoformat(),
                hrrr_snow_inches=hr,provider_snow_inches=pv,display_snow_inches=value,source=source,
                temperature_c=float(w.temperature_k.iloc[j])-273.15,precipitation_mm=float(w.precip_mm.iloc[j])))
        sel=selection[site]
        model24=neural['totals'][0]['mean_inches']
        blend=portfolio24_result(h.snowfall_inches.iloc[:24],model24,sel,ends[23].isoformat(),native_reason)
        if not blend['available']:
            output['status']='partial'
            output['warnings'].append(f"{NAMES[site]} 24h: {blend['reason']}")
        short=radar['sites'].get(site)
        if short and short.get('available'):
            radar_issue=pd.Timestamp(short['issued_at'])
            if (issue-radar_issue).total_seconds()>3600 or radar_issue>issue or pd.Timestamp(short['start'])<issue or pd.Timestamp(short['start'])!=start:
                short={'available':False,'reason':'Cached radar issue/window no longer matches the future forecast; refresh radar.'}
        tail=ends>pd.Timestamp(native['forecast_end_utc'])
        output['sites'][site]=dict(display_name=NAMES[site],target=neural['target'],hourly=hourly,
            native_future_hours=int(h.snowfall_inches.notna().sum()),native_range_hours=int((~tail).sum()),
            available_hrrr_hours=int(h.snowfall_inches.notna().sum()),provider_tail_hours=int(tail.sum()),
            component_sources={'neural':{k:neural.get(k) for k in ['issued_at','start','end','model_name','sources']},
                'weather_provider':neural['sources']['provider']['best_match'],'radar_issued_at':radar.get('issued_at')},
            existing_neural=neural['totals'],short_liquid=short,
            portfolio24=blend)
    try:output['alta12']={'available':True,**alta12(issue)}
    except (ValueError,FileNotFoundError,KeyError) as exc:
        output['alta12']={'available':False,'reason':str(exc),'issued_at':issue.isoformat()}
    for site,station in [('vail','vail_snotel'),('mansfield','stowe_mountain')]:
        path=ROOT/f'outputs/mountain-temperature/{station}_current.csv'
        if path.exists():
            d=pd.read_csv(path)
            output['sites'][site]['mountain_temperature']=dict(station=station,source=str(path.relative_to(ROOT)),
                component_issue_utc=str(d.issue_utc.iloc[0]) if 'issue_utc' in d and len(d) else None,
                note='Station temperature correction only. Snowfall benefit has not been verified. Original component issue times are retained.',rows=d.to_dict('records'))
    cleanout=clean(output);tmp=OUT/'current_forecast.json.tmp';tmp.write_text(json.dumps(cleanout,indent=2,allow_nan=False));tmp.replace(OUT/'current_forecast.json')
    try:plots(cleanout)
    except Exception as exc:
        # A report-plot failure must not discard valid forecasts for other models.
        cleanout['plot_error']=str(exc)
        tmp.write_text(json.dumps(cleanout,indent=2,allow_nan=False));tmp.replace(OUT/'current_forecast.json')
    print(json.dumps({s:{'HRRR_hours':v['native_future_hours'],'provider_tail':v['provider_tail_hours'],'portfolio24_inches':v['portfolio24']['mean_inches']} for s,v in cleanout['sites'].items()},indent=2))
    return cleanout

def plots(data):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(1,49);fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    for ax,site in zip(axes.flat,SITES):
        s=data['sites'][site];h=s['hourly'];native=np.array([r['hrrr_snow_inches'] for r in h],dtype=float);provider=np.array([r['provider_snow_inches'] for r in h],dtype=float);display=np.array([r['display_snow_inches'] for r in h],dtype=float)
        ax.step(np.r_[0,x],np.r_[0,np.cumsum(display)],where='post',color='#2563a6',label='HRRR hourly → provider tail')
        ax.step(np.r_[0,x],np.r_[0,np.cumsum(provider)],where='post',color='#17847c',alpha=.8,label='Weather provider')
        z=s['existing_neural'];ax.plot([0,24,48],[0]+[q['mean_inches'] for q in z],'o:',color='#dc9227',label='Previous neural checkpoints')
        p=s['portfolio24'];mean=finite_amount(p.get('mean_inches'));lower=finite_amount(p.get('lower_inches'));upper=finite_amount(p.get('upper_inches'))
        if mean is not None:
            if lower is not None and upper is not None and lower<=mean<=upper:
                ax.errorbar([24],[mean],yerr=[[mean-lower],[upper-mean]],fmt='D',capsize=4,color='#775da6',label='24h portfolio + residual range')
            else:ax.plot([24],[mean],'D',color='#775da6',label='24h portfolio')
        else:ax.text(.98,.02,'24h portfolio unavailable',transform=ax.transAxes,ha='right',fontsize=8,color='#795b41')
        native_count=s.get('native_range_hours',s['native_future_hours'])
        if native_count<48:ax.axvspan(native_count,48,color='#a4a9ad',alpha=.15,label='Beyond native HRRR range')
        ax.axhline(0,color='#286773',lw=1.5,zorder=5,clip_on=False);ax.set(xlabel='Hours after forecast start',ylabel='Snowfall accumulation (inches)',xlim=(0,48),ylim=(0,None),xticks=[0,12,24,36,48])
        ax.set_title(s['display_name']+' · '+s['target']['name'],fontsize=10);ax.grid(alpha=.15);ax.legend(fontsize=7,loc='upper left')
    start=pd.Timestamp(data['start']);end=pd.Timestamp(data['end'])
    fig.suptitle(f'Forecast comparison · {start:%b %d %H:%M} – {end:%b %d %H:%M UTC}\nHourly curve is weather-model guidance; learned snowfall is shown at accumulation checkpoints',fontsize=12)
    fig.savefig(OUT/'cumulative_forecasts.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7.5),layout='constrained')
    for ax,site in zip(axes.flat,SITES):
        s=data['sites'][site];h=s['hourly'];amount=np.array([r['display_snow_inches'] for r in h],dtype=float);n=s.get('native_range_hours',s['native_future_hours'])
        colors=['#21767f' if r['source']=='HRRR native snowfall' else '#8d969e' for r in h]
        ax.bar(x,amount,color=colors,width=.85);ax.axhline(0,color='#21767f',lw=1.5,zorder=5,clip_on=False)
        ax.axvspan(n,48,color='#a4a9ad',alpha=.12)
        ax.plot(x,[r['provider_snow_inches'] for r in h],color='#d49636',lw=1,label='Open-Meteo')
        peak=float(np.max(amount[np.isfinite(amount)])) if np.isfinite(amount).any() else 0.
        ax.set(title=s['display_name']+' · '+s['target']['name'],xlabel='Hour after forecast start',ylabel='New snowfall (inches / hour)',xlim=(0,49),ylim=(0,max(.06,peak*1.15)));ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Native HRRR hourly snowfall · gray bars mark the provider extension\nStation targets; Stowe and Jay are lower-elevation town proxies',fontsize=12)
    fig.savefig(OUT/'hourly_forecasts.png',dpi=160);plt.close(fig)
    p=data.get('alta12',{})
    if p.get('available'):
        h=data['sites']['alta']['hourly']
        window=[r for r in h if pd.Timestamp(p['start'])<pd.Timestamp(r['end'])<=pd.Timestamp(p['end'])]
        if len(window)==12 and all(r['hrrr_snow_inches'] is not None and r['provider_snow_inches'] is not None for r in window):
            amounts=[p['mean_inches'],sum(r['hrrr_snow_inches'] for r in window),sum(r['provider_snow_inches'] for r in window)]
            fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
            ax.barh([2,1,0],amounts,color=['#d49636','#2563a6','#17847c'],height=.5)
            ax.errorbar([p['mean_inches']],[2],xerr=[[p['mean_inches']-p['q10_inches']],[p['q90_inches']-p['mean_inches']]],fmt='o',color='#875620',capsize=5,label='Nominal 80% model range')
            for y,v in zip([2,1,0],amounts):ax.annotate(f'{v:.3f} in',(v,y),xytext=(5,-20),textcoords='offset points',fontsize=10)
            ax.set(yticks=[2,1,0],yticklabels=['Selected 12h portfolio','Latest HRRR','Weather provider'],xlabel='Snowfall over the exact 12-hour window (inches)',xlim=(0,max(.1,p['q90_inches']*1.12)))
            a=pd.Timestamp(p['start']).tz_convert('America/Denver');b=pd.Timestamp(p['end']).tz_convert('America/Denver')
            native_hour=pd.Timestamp(data['native_hrrr']['run_time_utc']).strftime('%H UTC')
            ax.set_title(f'Alta · {a:%b %d %I:%M %p} – {b:%b %d %I:%M %p %Z}\nModel inputs: 00 UTC MOS / 06 UTC HRRR · Latest HRRR: {native_hour}',fontsize=11)
            ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15)
            fig.savefig(OUT/'alta12_current.png',dpi=160);plt.close(fig)

if __name__=='__main__':build()
