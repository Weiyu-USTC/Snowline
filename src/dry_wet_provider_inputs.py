"""Align the archived JMA provider with existing held-out snowfall windows.

No new forecasts, training, or model selection. The archive is a previous_day3
mosaic, not the current Open-Meteo best-match forecast at the widget's issue.
"""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/nwp-weather/forecast-portfolio/dry-wet'
PROVIDER_METHOD = 'JMA provider · fixed older guidance'


def _hourly(root):
    d = pd.read_csv(root/'data/neural48_training/provider/historical_hourly.csv.gz')
    for col in ['valid_time_utc','implied_source_bound_utc']:
        d[col] = pd.to_datetime(d[col], utc=True)
    if d.duplicated(['site','valid_time_utc']).any():
        raise ValueError('Duplicate provider source intervals')
    if not ((d.valid_time_utc-d.implied_source_bound_utc)==pd.Timedelta(hours=72)).all():
        raise ValueError('Provider nominal source-age bounds changed')
    # Verify the units in every raw response contributing to the test winter.
    for name in d.loc[d.valid_time_utc.between('2023-10-01','2024-05-04'),'source_file'].unique():
        records = json.loads(gzip.decompress((root/name).read_bytes()))
        if isinstance(records,dict): records = [records]
        for item in records:
            if item['hourly_units']['snowfall_previous_day3'] != 'cm':
                raise ValueError('Raw provider snowfall unit is not centimetres')
    return d


def _amount(frame, start, end, issue):
    """Exact overlap of end-labelled one-hour snowfall forecast amounts."""
    start,end,issue = (pd.Timestamp(x) for x in [start,end,issue])
    if any(x.tzinfo is None for x in [start,end,issue]):
        raise ValueError('Forecast timestamps must contain timezone')
    d = frame[(frame.valid_time_utc>start)&
              (frame.valid_time_utc-pd.Timedelta(hours=1)<end)].sort_values('valid_time_utc')
    cursor=start; value=0.; latest=None
    for r in d.itertuples():
        a=max(start,r.valid_time_utc-pd.Timedelta(hours=1)); b=min(end,r.valid_time_utc)
        if a!=cursor or b<=a: raise ValueError('Provider target interval has a gap or overlap')
        if not np.isfinite(r.snowfall) or r.snowfall<0: raise ValueError('Provider snowfall missing or negative')
        if r.implied_source_bound_utc>issue: raise ValueError('Provider source bound follows model issue')
        value+=float(r.snowfall)*(b-a).total_seconds()/3600/2.54
        cursor=b; latest=r.implied_source_bound_utc
    if cursor!=end: raise ValueError('Incomplete provider window')
    return value, latest


def load_provider24(root=ROOT):
    root=Path(root)
    d=pd.read_csv(root/'outputs/nwp-weather/forecast-portfolio/hrrr24_predictions.csv')
    d=d[d.split.eq('test')].copy()
    z=np.load(root/'outputs/nwp-weather/neural48-trained/test_predictions.npz')
    names=json.loads((root/'outputs/nwp-weather/neural48-trained/test_methods.json').read_text())
    provider=z[f'{names.index(PROVIDER_METHOD)}_mean']
    lookup={int(row):i for i,row in enumerate(z['row_indices'])}
    meta=pd.read_csv(root/'data/neural48_training/prepared/index.csv').set_index('row')
    assert meta.index.is_unique
    hourly=_hourly(root)
    records=[]
    for r in d.itertuples():
        i=lookup[r.row]; m=meta.loc[r.row]
        if m['split']!='test' or m.site!=r.site or pd.Timestamp(m.start_utc)!=pd.Timestamp(r.start):
            raise ValueError('Provider/model case target or test split mismatch')
        if not np.isclose(z['truth'][i,0],r.truth,rtol=1e-7,atol=1e-7):
            raise ValueError('Provider/model observed first-day amount mismatch')
        amount,bound=_amount(hourly[hourly.site.eq(r.site)],m.start_utc,m.midpoint_utc,m.issue_utc)
        # Reproduce the archived daily amount, allowing only original float32 summation error.
        if not np.isclose(amount,float(provider[i,0]),rtol=3e-6,atol=1e-6):
            raise ValueError('Saved provider first-day total differs from source intervals')
        records.append(dict(row=r.row,site=r.site,start_utc=m.start_utc,
            end_utc=m.midpoint_utc,issue_utc=m.issue_utc,observed_inches=r.truth,
            provider_inches=float(provider[i,0]),source_bound_latest_utc=bound.isoformat()))
    result=pd.DataFrame(records)
    assert len(result)==len(d) and not result.duplicated(['site','start_utc']).any()
    return result


def load_provider12(root=ROOT):
    root=Path(root)
    native=pd.read_csv(root/'outputs/nwp-weather/alta12-trained/native_asnow_predictions.csv')
    native=native[native.stage.eq('test')].copy()
    meta=pd.read_csv(root/'outputs/nwp-weather/alta12-trained/case_index.csv').set_index('row')
    assert meta.index.is_unique
    target=np.load(root/'data/alta12_training/arrays.npz')['snow']
    hourly=_hourly(root); hourly=hourly[hourly.site.eq('alta')]
    records=[]
    for r in native.itertuples():
        m=meta.loc[r.row]
        if not bool(m.modern) or m.validation_stage!='test':
            raise ValueError('Alta native/provider cohort mismatch')
        if pd.Timestamp(m.start_utc)!=pd.Timestamp(r.start_utc) or pd.Timestamp(m.end_utc)!=pd.Timestamp(r.end_utc):
            raise ValueError('Alta native/provider interval mismatch')
        if pd.Timestamp(m.end_utc)-pd.Timestamp(m.start_utc)!=pd.Timedelta(hours=12):
            raise ValueError('Alta target is not twelve hours')
        if not np.isclose(target[r.row],r.actual_inches,rtol=1e-7,atol=1e-7):
            raise ValueError('Alta native/provider truth mismatch')
        amount,bound=_amount(hourly,m.start_utc,m.end_utc,m.issue_utc)
        records.append(dict(row=r.row,site='alta',start_utc=m.start_utc,end_utc=m.end_utc,
            issue_utc=m.issue_utc,observed_inches=r.actual_inches,
            provider_inches=amount,source_bound_latest_utc=bound.isoformat()))
    result=pd.DataFrame(records)
    assert len(result)==len(native) and not result.start_utc.duplicated().any()
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p24=load_provider24(); p12=load_provider12()
    p24.to_csv(OUT/'provider24_matched.csv',index=False)
    p12.to_csv(OUT/'provider12_matched.csv',index=False)
    records=[]
    for horizon,frame in [(12,p12),(24,p24)]:
        for site,g in frame.groupby('site'):
            for category,d in [('all',g),('dry',g[g.observed_inches.eq(0)]),('wet',g[g.observed_inches.gt(0)])]:
                records.append(dict(site=site,horizon_hours=horizon,category=category,n=len(d),
                    provider_rmse_cm=float(np.sqrt(np.mean((d.provider_inches-d.observed_inches)**2)))*2.54))
    paths=['outputs/nwp-weather/neural48-trained/test_predictions.npz',
           'outputs/nwp-weather/neural48-trained/test_methods.json',
           'data/neural48_training/prepared/index.csv',
           'outputs/nwp-weather/forecast-portfolio/hrrr24_predictions.csv',
           'outputs/nwp-weather/alta12-trained/native_asnow_predictions.csv',
           'outputs/nwp-weather/alta12-trained/case_index.csv',
           'data/alta12_training/arrays.npz',
           'data/neural48_training/provider/historical_hourly.csv.gz',
           'data/neural48_training/provider/historical_manifest.json']
    audit={'provider':'Open-Meteo archived JMA GSM previous_day3',
        'not_live_best_match':True,
        'caveat':'Fixed nominal72h source-age mosaic; exact issue/receipt timestamps unavailable. Older guidance than the model/HRRR. This is not evidence of superiority over current Open-Meteo best-match.',
        'source_units':'Raw snowfall_previous_day3 in cm; saved neural-provider daily values in inches.',
        'interval_semantics':'Each valid_time_utc is the end of the preceding1h forecast interval. Exact overlap weights applied to fractional clocks; complete coverage required.',
        'checks':['Raw test-winter source units','No duplicate source hours','Exact source interval coverage','Latest nominal source bound before model issue','24h saved provider totals reproduced from source hours','Case row/site/observed target/start/end/test split match'],
        'matched24_counts':p24.groupby('site').size().to_dict(),'matched12_count':len(p12),
        'excluded24':0,'excluded12':0,'metrics':records,
        'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (OUT/'provider_alignment_audit.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__': main()
