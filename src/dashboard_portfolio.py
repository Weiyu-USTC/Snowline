"""Read-only, per-horizon deployment adapter for Snowline.

Every comparison uses the same target and exact interval as its provider bar.
No optimizer runs here, no daily total is divided into invented hourly labels,
and cached predictions retain their issue/valid times when they go stale.
"""
from __future__ import annotations
import csv,json,math
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SITES=('alta','vail','mansfield','jay')
HORIZONS=(1,3,6,12,24,48)
REPORT='/reports/weather/forecast-portfolio/report.html'
SELECTION='outputs/nwp-weather/forecast-portfolio/deployment_selection.json'


def _dt(value):
    d=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if d.tzinfo is None:raise ValueError('Forecast timestamps must contain their timezone')
    return d.astimezone(timezone.utc)


def _number(v,required=False):
    if v is None or isinstance(v,bool):
        if required:raise ValueError('Forecast amount is missing')
        return None
    try:n=float(v)
    except (ValueError,TypeError):n=float('nan')
    if not math.isfinite(n) or n<0:
        if required:raise ValueError('Forecast amount is invalid or missing')
        return None
    return n


def _load(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def _document(path):
    value=_load(path)
    return value if isinstance(value,dict) else {}


def deployment_neural48_families(root=None):
    data=_document(Path(root or ROOT)/SELECTION)
    return {s:data['sites'][s]['48']['family'] for s in SITES}


def _ceil_hour(now):
    base=now.replace(minute=0,second=0,microsecond=0)
    return base if base==now else base+timedelta(hours=1)


def _integrate(rows,start,end,key,*,end_key='end',start_key='start'):
    """Exact interval overlap; gaps, duplicate intervals and NaNs are errors."""
    parts=[]
    for row in rows:
        b=_dt(row[end_key]);a=_dt(row[start_key]) if start_key in row else b-timedelta(hours=1)
        if b<=a:raise ValueError('Forecast interval has invalid duration')
        lo,hi=max(a,start),min(b,end)
        if hi>lo:
            value=_number(row.get(key),required=True)
            parts.append((lo,hi,value*(hi-lo).total_seconds()/(b-a).total_seconds()))
    cursor=start;total=0.
    for lo,hi,value in sorted(parts):
        if abs((lo-cursor).total_seconds())>1e-4:raise ValueError('Forecast intervals are missing or overlap')
        total+=value;cursor=hi
    if abs((cursor-end).total_seconds())>1e-4:raise ValueError('Forecast does not cover the complete requested interval')
    return total


class Context:
    def __init__(self,site,now,root):
        self.site,self.now,self.root=site,now,root
        self.manifest=_document(root/SELECTION)
        self.selection=self.manifest.get('sites',{}).get(site,{})
        self.neural_snapshot=_document(root/'outputs/nwp-weather/neural48-trained/current_forecast.json')
        self.neural=self.neural_snapshot.get('sites',{}).get(site,{})
        self.portfolio=_document(root/'outputs/nwp-weather/forecast-portfolio/current_forecast.json')
        self.native=_document(root/'data/hrrr_portfolio/current.json')
        self.radar_snapshot=_document(root/'outputs/nexrad-short/current_forecast.json')
        self.radar=self.radar_snapshot.get('sites',{}).get(site,{})
        self.mrms=_document(root/'outputs/dashboard/alta/latest.json') if site=='alta' else {}
        self.target=self.neural.get('target',{})
        self.cache={}

    def csv(self,name):
        if name not in self.cache:
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root.resolve()):raise ValueError('Forecast source path is outside this project')
            try:
                with path.open(newline='') as f:self.cache[name]=list(csv.DictReader(f))
            except OSError:self.cache[name]=[]
        return self.cache[name]

    def provider(self,start,end,quantity):
        meta=self.neural.get('sources',{}).get('provider',{}).get('best_match') or {}
        retrieved=_dt(meta['retrieved_at'])
        if retrieved>self.now or self.now-retrieved>timedelta(hours=6):raise ValueError('Provider snapshot is stale or has a future retrieval time')
        path=(self.root/meta['path']).resolve()
        if not path.is_relative_to(self.root.resolve()):raise ValueError('Provider source path is outside this project')
        responses=_load(path)
        if not isinstance(responses,list) or len(responses)!=4:raise ValueError('Provider four-site response is unavailable')
        response=responses[SITES.index(self.site)]
        if response.get('utc_offset_seconds') not in (None,0) or response.get('timezone') not in (None,'GMT','UTC'):
            raise ValueError('Provider comparison must use UTC timestamps')
        key='snowfall' if quantity=='snowfall' else 'precipitation'
        unit='cm' if quantity=='snowfall' else 'mm'
        if response.get('hourly_units',{}).get(key)!=unit:raise ValueError('Provider accumulation unit differs')
        hourly=response.get('hourly',{});times=hourly.get('time',[]);values=hourly.get(key,[])
        if len(times)!=len(values):raise ValueError('Provider time/value arrays differ')
        rows=[]
        for time,value in zip(times,values):
            stamp=datetime.fromisoformat(str(time).replace('Z','+00:00'))
            if stamp.tzinfo is None:stamp=stamp.replace(tzinfo=timezone.utc)
            amount=_number(value)
            rows.append({'end':stamp.isoformat(),'amount':None if amount is None else amount/(2.54 if quantity=='snowfall' else 1.)})
        return _integrate(rows,start,end,'amount'),meta


def _base(ctx,hours):
    choice=ctx.selection.get(str(hours),{})
    quantity=choice.get('quantity','liquid_precipitation' if hours<=3 else 'snowfall')
    return dict(hours=hours,start=None,end=None,issued_at=None,status='unavailable',
        quantity=quantity,unit='mm' if quantity=='liquid_precipitation' else 'inches',
        model_name='Our model',method_id=choice.get('method_id'),selected_method_id=choice.get('method_id'),
        mean=None,lower=None,upper=None,provider_mean=None,interval_label=None,
        stale=False,window_started=False,expired=False,warnings=[],sources={},target=ctx.target,
        hourly_validated=False,selection=choice)


def _window(record,ctx,start,end,issued,max_age):
    start,end,issued=map(_dt,(start,end,issued))
    record.update(start=start.isoformat(),end=end.isoformat(),issued_at=issued.isoformat(),
        stale=ctx.now-issued>max_age,window_started=start<ctx.now,expired=end<=ctx.now)
    if end-start!=timedelta(hours=record['hours']):raise ValueError('Model window is not the requested horizon')
    if issued>start or issued>ctx.now:raise ValueError('Forecast issue time is in the future or after its window starts')
    if record['stale'] or record['window_started'] or record['expired']:
        raise ValueError('Saved forecast is stale or its accumulation window has begun; refresh for a current window')
    return start,end


def _ready(record,ctx,mean,lower=None,upper=None,interval=None):
    record['mean']=_number(mean,required=True)
    record['lower'],record['upper']=_number(lower),_number(upper)
    if record['lower'] is not None and record['upper'] is not None and record['lower']>record['upper']:
        raise ValueError('Forecast uncertainty range is reversed')
    record['interval_label']=interval;record['status']='ready'
    try:
        value,meta=ctx.provider(_dt(record['start']),_dt(record['end']),record['quantity'])
        record['provider_mean']=value;record['sources']['provider']=meta
    except (KeyError,ValueError,TypeError) as exc:
        record['warnings'].append('Matched provider comparison unavailable: '+str(exc))
    return record


def _hrrr(ctx,record,start=None):
    start=start or _ceil_hour(ctx.now);end=start+timedelta(hours=record['hours'])
    meta=ctx.native;run=_dt(meta['run_time_utc']);available=_dt(meta['assumed_available_utc'])
    if available>ctx.now:raise ValueError('The HRRR run is not yet available at this issue time')
    if ctx.now-run>timedelta(hours=12):raise ValueError('Cached HRRR initialization is too old')
    if end>_dt(meta['forecast_end_utc']):raise ValueError('Native HRRR ends before the requested horizon')
    _window(record,ctx,start,end,available,timedelta(hours=12))
    quantity=record['quantity'];name='current_weather.csv' if quantity=='liquid_precipitation' else 'current_native_snowfall.csv'
    path=meta.get('weather_path' if quantity=='liquid_precipitation' else 'snowfall_path','data/hrrr_portfolio/'+name)
    key='precip_mm' if quantity=='liquid_precipitation' else 'snowfall_inches'
    rows=[r for r in ctx.csv(path) if r.get('site')==ctx.site]
    for row in rows:
        if _dt(row['run_time_utc'])!=run:raise ValueError('HRRR weather and snowfall snapshots refer to different runs')
        if _dt(row['valid_time_utc'])!=run+timedelta(hours=float(row['forecast_hour'])):raise ValueError('HRRR valid time differs from its native forecast hour')
    amount=_integrate(rows,start,end,key,end_key='valid_time_utc')
    record['method_id']='HRRR_liquid' if quantity=='liquid_precipitation' else 'HRRR_native_snowfall'
    record['sources']['hrrr']=meta
    record['warnings'].append('Native HRRR guidance; no independently validated hourly fresh-snow label is available.')
    return _ready(record,ctx,amount)


def _nexrad(ctx,record):
    snap=ctx.radar
    if not snap.get('available'):raise ValueError(snap.get('reason','Fresh NEXRAD forecast is unavailable'))
    if ctx.radar_snapshot.get('units')!='mm' or ctx.radar_snapshot.get('quantity')!='liquid_equivalent_precipitation':
        raise ValueError('NEXRAD quantity/units are not liquid precipitation millimetres')
    start=_dt(snap['start']);end=start+timedelta(hours=record['hours'])
    _window(record,ctx,start,end,snap['issued_at'],timedelta(minutes=60))
    method=record['method_id']
    rows=snap.get('hourly',[])
    if method=='NEXRAD_selected':rows=snap.get('native_nexrad_hourly',rows)
    elif method in ['DPR_persistence','DPR_advection']:
        native=snap.get('native_nexrad_hourly',rows)
        if snap.get('selection',{}).get('model')!=method:raise ValueError('Saved NEXRAD method does not match its selected checkpoint')
        rows=native
    elif method=='NEXRAD_HRRR_portfolio' and snap.get('model')!='NEXRAD_HRRR_portfolio':
        raise ValueError('Native radar snapshot has not been combined with the issued HRRR run')
    amount=_integrate(rows,start,end,'liquid_mm')
    lower=upper=None
    # An hourly residual band cannot be summed into a multi-hour confidence band.
    if record['hours']==1:
        first=next((r for r in rows if _dt(r['start'])==start and _dt(r['end'])==end),{})
        lower,upper=first.get('liquid_lower_mm'),first.get('liquid_upper_mm')
    record['sources']['nexrad']={k:snap.get(k) for k in ['issued_at','radar','model','coverage','provenance','portfolio']}
    record['warnings']+=snap.get('limitations',[])
    return _ready(record,ctx,amount,lower,upper,'Descriptive validation residual range' if record['hours']==1 else None)


def _mrms(ctx,record):
    snap=ctx.mrms;method=record['method_id'].removeprefix('original_MRMS_')
    if snap.get('unit')!='mm liquid equivalent':raise ValueError('Saved MRMS quantity differs')
    original_start=_dt(snap['start'])
    # Use a complete future clock-hour segment already inside this issued
    # trajectory. Never move the trajectory itself to a new issue/start time.
    start=_ceil_hour(max(ctx.now,original_start));end=start+timedelta(hours=record['hours'])
    if end>_dt(snap['end']):raise ValueError('The selected complete clock-hour window exceeds the saved MRMS trajectory')
    _window(record,ctx,start,end,snap['issued_at'],timedelta(minutes=30))
    expected=ctx.manifest.get('checkpoint_sha256',{}).get(method)
    if expected and snap.get('checkpoint_sha256',{}).get(method)!=expected:
        raise ValueError('Saved MRMS checkpoint differs from the validation-selected artifact')
    model=next((r for r in snap.get('models',[]) if r.get('id')==method),None)
    if not model:raise ValueError('Selected MRMS model is absent from the current snapshot')
    points=model.get('hours',[])
    a=next((r for r in points if _dt(r['time'])==start),None)
    b=next((r for r in points if _dt(r['time'])==end),None)
    if not a or not b:raise ValueError('MRMS cumulative endpoints do not cover the selected horizon')
    baseline=_number(a.get('mean_liquid_mm'),required=True)
    mean=_number(b.get('mean_liquid_mm'),required=True)-baseline
    lower=upper=None
    if abs(baseline)<1e-9:
        lower,upper=b.get('q10_liquid_mm'),b.get('q90_liquid_mm')
    record['sources']['mrms']={k:snap.get(k) for k in ['issued_at','fetched_at','source','sources','checkpoint_sha256','hrrr_run_utc']}
    record['warnings']+=snap.get('warnings',[])
    if start!=original_start:
        record['warnings'].append('This complete clock hour is a later segment of the issued radar trajectory; its lead differs from the first-hour validation target.')
    return _ready(record,ctx,mean,lower,upper,'Uncalibrated 80% neural ensemble spread')


def _alta12(ctx,record):
    p=ctx.portfolio.get('alta12',{})
    if not p.get('available'):raise ValueError(p.get('reason','The selected Alta12-hour forecast is unavailable'))
    start,end=_window(record,ctx,p['start'],p['end'],p['issued_at'],timedelta(hours=6))
    if start.hour not in (11,23) or any([start.minute,start.second,start.microsecond]):raise ValueError('Alta12 model is restricted to fixed11/23UTC target windows')
    record['method_id']=p['method'];record['sources']['alta12']=p
    record['warnings']+=[p.get('validation_scope','Fixed reporting12h intervals; earlier issuance unvalidated.'),p.get('history','')]
    return _ready(record,ctx,p['mean_inches'],p.get('q10_inches'),p.get('q90_inches'),'Nominal 80% model prediction range')


def _neural(ctx,record):
    p=ctx.neural
    if not p or p.get('status')=='unavailable' or p.get('site_id')!=ctx.site:raise ValueError('Current neural forecast is unavailable for this mountain')
    desired=ctx.selection['48']['family']
    special=p.get('deployed_48')
    if special is None:
        if p.get('family')!=desired:raise ValueError('Refresh is needed to run the selected48-hour model family')
        special=next((r for r in p.get('totals',[]) if r.get('hours')==48),{})
    actual=special.get('family',p.get('family'))
    fallback=special.get('fallback_reason') or special.get('fallback') or (special.get('reason') if special.get('status')=='fallback' else None)
    if actual!=desired and not fallback:raise ValueError('The saved48-hour family does not match its validation selection')
    _window(record,ctx,special.get('start',p['start']),special['end'],special.get('issued_at',p['issued_at']),timedelta(hours=3))
    record['method_id']='neural48_'+actual
    record['sources']['neural']={k:p.get(k) for k in ['issued_at','family','model_name','sources','target']}
    record['sources']['deployed_48']=special
    record['warnings']+=p.get('warnings',[])
    result=_ready(record,ctx,special['mean_inches'],special.get('q10_inches'),special.get('q90_inches'),special.get('interval_label','Uncalibrated 80% neural prediction range'))
    if actual!=desired:
        result['status']='fallback';result['warnings'].append(str(fallback))
    return result


def _portfolio24(ctx,record):
    p=ctx.neural
    if p.get('site_id')!=ctx.site or p.get('status')=='unavailable':raise ValueError('The neural24-hour portfolio component is unavailable')
    first=next((r for r in p.get('totals',[]) if r.get('hours')==24),{})
    start,end=_window(record,ctx,p['start'],first['end'],p['issued_at'],timedelta(hours=3))
    sel=ctx.selection['24']['weights']
    # Jay48 may use a different family; the untouched first24h total must remain
    # the weather-only component on which the24h portfolio weights were fitted.
    expected={'alta':'weather_radar','vail':'weather_radar','mansfield':'weather_only','jay':'weather_only'}[ctx.site]
    if p.get('family')!=expected:raise ValueError('Neural24-hour component differs from the family used to fit portfolio weights')
    component=_hrrr(ctx,_base(ctx,24),start=start)
    aggregate_issue=max(_dt(p['issued_at']),_dt(component['issued_at']))
    if aggregate_issue>start:raise ValueError('A portfolio component became available after the target interval began')
    record['issued_at']=aggregate_issue.isoformat()
    hrrr=component['mean']*(float(sel['scale_fit_on_train']) if sel['native_hrrr_component']=='hrrr_calibrated' else 1.)
    model=_number(first.get('mean_inches'),required=True)
    mean=float(sel['hrrr_weight'])*hrrr+float(sel['prior_neural_weight'])*model
    spread=_number(sel.get('validation_residual_p80_inches'))
    record['sources']={'hrrr':component['sources']['hrrr'],'neural':{k:p.get(k) for k in ['issued_at','family','model_name','sources']},'weights':sel}
    record['warnings']+=p.get('warnings',[])
    record['warnings'].append('Weights were validated on fixed reporting24h windows; rolling windows remain experimental.')
    return _ready(record,ctx,mean,None if spread is None else max(0,mean-spread),None if spread is None else mean+spread,'Descriptive 80% validation residual range')


def read_deployed(site,now=None,root=None):
    if site not in SITES:raise ValueError('Unknown mountain')
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None:raise ValueError('Current time must include a timezone')
    ctx=Context(site,now.astimezone(timezone.utc),Path(root or ROOT))
    records=[]
    for hours in HORIZONS:
        record=_base(ctx,hours)
        try:
            if str(hours) not in ctx.selection:raise ValueError('The frozen deployment selection is unavailable')
            method=record['method_id']
            if hours in (1,3):
                if method.startswith('original_MRMS_'):record=_mrms(ctx,record)
                elif method=='HRRR':record=_hrrr(ctx,record)
                else:record=_nexrad(ctx,record)
            elif hours==12 and site=='alta':record=_alta12(ctx,record)
            elif hours==24:record=_portfolio24(ctx,record)
            elif hours==48:record=_neural(ctx,record)
            else:record=_hrrr(ctx,record)
        except (KeyError,ValueError,TypeError,OSError) as exc:
            reason=str(exc);saved={k:record.get(k) for k in ['start','end','issued_at','stale','window_started','expired']}
            record=_base(ctx,hours);record['warnings'].append(reason);record['saved_window']=saved
            if hours<48:
                try:
                    record=_hrrr(ctx,record);record['status']='fallback'
                    record['warnings'].insert(0,'Selected model unavailable; using issued HRRR guidance for this complete future window.')
                except (KeyError,ValueError,TypeError,OSError) as fallback_error:
                    record.update(mean=None,lower=None,upper=None,provider_mean=None,status='unavailable')
                    record['warnings'].append('HRRR fallback unavailable: '+str(fallback_error))
            else:
                record.update(saved)
            record['warnings']=list(dict.fromkeys(record['warnings']))
        records.append(record)
    available=sum(r['mean'] is not None for r in records)
    source_times=[r['issued_at'] for r in records if r['issued_at']]
    return dict(site_id=site,status='ready' if available==6 else 'partial' if available else 'unavailable',
        generated_at=max(source_times) if source_times else None,read_at=ctx.now.isoformat(),
        model_name='Our model',target=ctx.target,report_url=REPORT,horizons=records,
        selection_url='/reports/weather/forecast-portfolio/deployment_selection.json',
        warnings=[] if available==6 else ['One or more forecast horizons need fresh inputs.'])


if __name__=='__main__':
    print(json.dumps({s:read_deployed(s) for s in SITES},indent=2,allow_nan=False))
