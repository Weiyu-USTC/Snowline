"""Dry/wet held-out audit of the frozen deployed 1–3h liquid forecasts.

No training, selection changes or downloads. The provider is the existing
archived JMA72h-offset mosaic, not the current Open-Meteo best-match service.
"""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/nwp-weather/forecast-portfolio/dry-wet'
SOURCES = set()
NAMES = {'alta':'Alta','mansfield':'Stowe','jay':'Jay Peak'}


def source(relative):
    path = ROOT/relative
    SOURCES.add(path)
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for part in iter(lambda: file.read(1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def original_alta(paired, selection):
    """Same target and evaluation mask used by frozen deployment selection."""
    base = 'outputs/alta-nowcast/radar_predictions/'
    manifest = json.loads(source(base+'nowcastnet_manifest.json').read_text())
    checkpoint = source('outputs/alta-nowcast/models/selected/nowcastnet.pt')
    assert sha(checkpoint) == manifest['checkpoint_sha256'] == selection['checkpoint_sha256']['nowcastnet']
    path = source(base+'nowcastnet.npy')
    assert sha(path) == manifest['prediction_sha256']
    index = np.load(source(base+'nowcastnet_index.npz'))
    lookup = {str(cid):i for i,cid in enumerate(index['case_id']) if index['split'][i] == 'test'}
    wanted = set(paired.loc[paired.site.eq('alta'), 'case_id']) & set(lookup)
    arrays = np.load(path, mmap_mode='r')
    with np.load(source('data/alta_nowcast/radar/radar_sequences.npz')) as z:
        ids = {str(cid):i for i,cid in enumerate(z['case_id'])}
        iy = int(np.argmin(abs(z['latitude']-40.57607)))
        ix = int(np.argmin(abs(z['longitude']+111.63889)))
        rates, valid, quality = z['rate'], z['valid_fraction'], z['rqi_hourly']
        rt, qt = z['radar_time'], z['quality_time']
    records = []
    for cid in sorted(wanted):
        i = ids[cid]
        rr = rates[i,:,iy-1:iy+2,ix-1:ix+2]
        vv = valid[i,:,iy-1:iy+2,ix-1:ix+2]
        qq = quality[i,:,iy-1:iy+2,ix-1:ix+2]
        qi = np.searchsorted(qt[i], rt[i], side='right')-1
        assert (qi >= 0).all()
        qq = qq[qi]
        mask = (np.isfinite(rr)&(rr>=0)&(vv>=.9)&np.isfinite(qq)&(qq>=.5))[10:]
        count = mask.sum((1,2))
        target = np.divide(np.where(mask,rr[10:],0).sum((1,2)), count,
                           out=np.full(19,np.nan), where=count>=5)
        truth = np.array([np.trapezoid(target[k*6:k*6+7], dx=1/6) for k in range(3)])
        ref = paired[paired.site.eq('alta')&paired.case_id.eq(cid)].iloc[0]
        assert np.allclose(truth,ref[['truth1','truth2','truth3']].to_numpy(float),atol=1e-6)
        pred = np.asarray(arrays[lookup[cid],:,1:,iy-1:iy+2,ix-1:ix+2]).mean(0)
        rate = np.divide(np.where(mask,pred,0).sum((1,2)), count,
                         out=np.full(19,np.nan), where=count>=5)
        values = [float(np.trapezoid(rate[k*6:k*6+7],dx=1/6)) for k in range(3)]
        records.append(dict(case_id=cid,site='alta',model='original_MRMS_nowcastnet',split='test',
                       **{f'pred{k+1}':values[k] for k in range(3)},
                       **{f'truth{k+1}':truth[k] for k in range(3)}))
    return pd.DataFrame(records)


def provider_inputs():
    # Validate archived accumulation units directly in the raw test-season data.
    for relative in ['data/nwp_recent/raw/jma_gsm_2023_2024.json.gz',
                     'data/neural48_training/provider/raw/vermont_proxy_2023.json.gz']:
        raw = json.loads(gzip.decompress(source(relative).read_bytes()))
        assert all(r['hourly_units']['precipitation_previous_day3']=='mm' for r in raw)
        assert all(r.get('utc_offset_seconds',0)==0 for r in raw)
    frame = pd.read_csv(source('data/neural48_training/provider/historical_hourly.csv.gz'))
    frame['valid_time_utc'] = pd.to_datetime(frame.valid_time_utc, utc=True)
    frame['implied_source_bound_utc'] = pd.to_datetime(frame.implied_source_bound_utc, utc=True)
    assert not frame.duplicated(['site','valid_time_utc']).any()
    return {s:d.set_index('valid_time_utc') for s,d in frame.groupby('site')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads(source('outputs/nwp-weather/forecast-portfolio/deployment_selection.json').read_text())
    paired = pd.read_csv(source('outputs/nexrad-short/paired_predictions.csv'),dtype={'case_id':str})
    paired = paired[paired.split.eq('test')].copy()
    assert not paired.duplicated(['site','model','case_id']).any()
    paired = pd.concat([paired,original_alta(paired,selection)],ignore_index=True)
    providers = provider_inputs()
    rows,exclusions = [],[]
    for site in NAMES:
        for hours in [1,3]:
            identity = selection['sites'][site][str(hours)]['method_id']
            ours = paired[paired.site.eq(site)&paired.model.eq(identity)].set_index('case_id')
            hrrr = paired[paired.site.eq(site)&paired.model.eq('HRRR')].set_index('case_id')
            common = ours.index.intersection(hrrr.index)
            assert len(common)>0
            for cid in common:
                date = pd.Timestamp(cid,tz='UTC')
                issue,start = date+pd.Timedelta(hours=10,minutes=50),date+pd.Timedelta(hours=11)
                ends = pd.date_range(start+pd.Timedelta(hours=1),periods=hours,freq='h')
                wx = providers[site].reindex(ends)
                cols = [f'truth{i}' for i in range(1,hours+1)]
                y = ours.loc[cid,cols].to_numpy(float)
                assert np.allclose(y,hrrr.loc[cid,cols].to_numpy(float),atol=1e-7)
                predcols = [f'pred{i}' for i in range(1,hours+1)]
                model = ours.loc[cid,predcols].to_numpy(float)
                raw = hrrr.loc[cid,predcols].to_numpy(float)
                provider = wx.precipitation.to_numpy(float)
                if not np.isfinite(np.r_[y,model,raw,provider]).all():
                    exclusions.append(dict(site=site,hours=hours,case_id=cid,reason='Common finite coverage missing'))
                    continue
                assert (np.r_[y,model,raw,provider]>=0).all()
                assert wx.implied_source_bound_utc.notna().all()
                assert (wx.implied_source_bound_utc<=issue).all()
                for name,values in [('Our model',model),('Native HRRR',raw),('Archived JMA provider',provider)]:
                    rows.append(dict(site=site,mountain=NAMES[site],horizon_hours=hours,case_id=cid,
                        issue_utc=issue.isoformat(),start_utc=start.isoformat(),end_utc=ends[-1].isoformat(),
                        method=name,frozen_method=identity if name=='Our model' else 'HRRR06UTC' if name=='Native HRRR' else 'JMA_previous_day3',
                        observed_mm=float(y.sum()),predicted_mm=float(values.sum()),
                        category='dry' if y.sum()==0 else 'wet',
                        provider_latest_implied_source_bound_utc=wx.implied_source_bound_utc.max().isoformat()))
    cases = pd.DataFrame(rows)
    assert not cases.duplicated(['site','horizon_hours','method','case_id']).any()
    cases.to_csv(OUT/'short_heldout_cases.csv',index=False)
    metrics = []
    for (site,hours,method),group in cases.groupby(['site','horizon_hours','method']):
        for category in ['all','dry','wet']:
            d = group if category=='all' else group[group.category.eq(category)]
            error = d.predicted_mm-d.observed_mm
            metrics.append(dict(site=site,mountain=NAMES[site],horizon_hours=hours,method=method,category=category,
                n=len(d),rmse_mm=float(np.sqrt(np.mean(error**2))) if len(d) else None,
                mae_mm=float(np.mean(abs(error))) if len(d) else None,bias_mm=float(np.mean(error)) if len(d) else None))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT/'short_metrics.csv',index=False)
    all_summary = metrics.pivot(index=['site','mountain','horizon_hours','category','n'],columns='method',values='rmse_mm').reset_index()
    all_summary.rename(columns={'Our model':'our_rmse_mm','Native HRRR':'hrrr_rmse_mm','Archived JMA provider':'provider_rmse_mm'}).to_csv(OUT/'short_summary.csv',index=False)
    wide = cases.pivot(index=['site','mountain','horizon_hours','case_id','start_utc','end_utc','observed_mm','category'],columns='method',values='predicted_mm').reset_index()
    wide.rename(columns={'Our model':'our_mm','Native HRRR':'hrrr_mm','Archived JMA provider':'provider_mm'}).to_csv(OUT/'short_cases.csv',index=False)
    summary = metrics[~metrics.category.eq('all')].pivot(index=['site','mountain','horizon_hours','category','n'],columns='method',values='rmse_mm').reset_index()
    summary.to_csv(OUT/'short_comparison.csv',index=False)
    for _,g in cases.groupby(['site','horizon_hours','case_id']):
        assert len(g)==3 and g.observed_mm.nunique()==1 and g.category.nunique()==1
    audit = dict(generated_at=datetime.now(timezone.utc).isoformat(),units='mm liquid-equivalent precipitation, NOT cm of snowfall',
        window='Forecast issued10:50UTC; target begins11UTC;1h ends12UTC and3h ends14UTC.',
        target='Quality-qualified local3x3 MRMS analysis mean integrated with10min trapezoidal quadrature; shares NEXRAD input and is not independent fresh-snow truth.',
        categories='Dry recorded MRMS accumulation exactly0, wet>0; wet does not establish precipitation phase or a snowy day.',
        provider='Archived Open-Meteo JMA GSM previous_day3, coarse0.5degree/native6hour guidance interpolated by provider to hourly. A fixed72h-offset mosaic; unavailable original issue/receipt times. Not current Open-Meteo best-match.',
        provider_target='Configured target points match the radar study; provider nearest grid and radar3x3 footprint have different spatial support.',
        hrrr='06UTC run, assumed available09UTC; original hourly amounts ending12/13/14UTC from paired benchmark.',
        vail='No quality-qualified radar verification cases; no dry/wet score can be reported.',
        selection='Frozen deployment_selection.json, never updated using these test results.',
        alta_nowcastnet='Original deployed checkpoint and original prediction SHA256 verified; newer tuned previous_MRMS_nowcastnet comparator deliberately excluded.',
        split='2023/24 winter held out from fitted weights and selection, but previously inspected in project work.',
        exclusions=exclusions,sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(SOURCES)})
    (OUT/'short_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('Saved',len(cases),'matched prediction rows; exclusions',len(exclusions))


if __name__=='__main__':main()
