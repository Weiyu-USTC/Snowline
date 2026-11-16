"""Observed versus forecast one-hour precipitation for the frozen Alta benchmark.

No fitting or checkpoint selection. The illustrative storm is selected only by
observed Alta-cell accumulation, not by a model's performance. Outputs are water
amounts from radar, not independently observed hourly fresh-snow depths.
"""
from pathlib import Path
import json
import os
import hashlib
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/alta-nowcast'
OUT = BASE / 'learning-review'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from evaluate_alta_nowcast import integral, phase_flow, RADAR_LABELS

ORDER = ['Persistence', 'Motion extrapolation', 'HRRR', 'ConvLSTM', 'DGMR adaptation', 'NowcastNet adaptation']
COLORS = dict(zip(ORDER, ['#84979c', '#398c82', '#b44956', '#257db0', '#c17a32', '#8055a6']))
OBS = '#18383e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=175, bbox_inches='tight')
    plt.close(fig)


def cumulative(rate):
    """Seven 10-minute snapshots, 11:00 to 12:00 UTC, integrated as trapezoids."""
    a = np.asarray(rate, float)[..., 1:8, :, :]
    zero = np.zeros_like(a[..., :1, :, :])
    return np.concatenate([zero, np.cumsum((a[..., 1:, :, :] + a[..., :-1, :, :])/12, axis=-3)], axis=-3)


def make_figures(frame, event, lat, lon, iy, ix, run_label):
    dates = pd.date_range(frame.date.min(), frame.date.max(), freq='D')
    # Separate small multiples make actual and forecast amounts easy to compare.
    for scope, title, note in [
        ('regional', 'Observed and forecast one-hour accumulation across the radar region',
         'Regional mean over the same quality-valid cells for every method on each date; available area varies by date.'),
        ('alta', 'Observed and forecast one-hour accumulation at the Alta grid cell',
         'Nearest pooled radar cell: approximately 3.4 × 4.4 km. This is a radar estimate, not a snow-board reading.')]:
        fig, axs = plt.subplots(3, 2, figsize=(14, 8.3), sharex=True, sharey=True, layout='constrained')
        ymax = float(frame[[scope+'_observed_mm',scope+'_predicted_mm']].max().max()) * 1.08
        for ax, name in zip(axs.flat, ORDER):
            d = frame[frame.model == name].set_index('date').reindex(dates)
            truth, pred = d[scope+'_observed_mm'], d[scope+'_predicted_mm']
            ax.fill_between(dates, 0, truth, color=OBS, alpha=.12)
            ax.plot(dates, truth, color=OBS, lw=1.4, label='Observed MRMS')
            ax.plot(dates, pred, color=COLORS[name], lw=1.3, alpha=.9, label='Forecast')
            err = pred-truth
            ax.set_title(f'{name}  ·  RMSE {np.sqrt((err**2).mean()):.3f} mm', fontsize=11, loc='left')
            ax.set_ylim(0, ymax)
            ax.grid(alpha=.15)
            ax.axhline(0, color='#9baeb2', lw=.65)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.legend(frameon=False, fontsize=8, loc='upper left')
        for ax in axs[:,0]: ax.set_ylabel('One-hour water amount · mm')
        for ax in axs[-1,:]: ax.set_xlabel('Forecast date · winter 2023/24')
        fig.suptitle(title, fontsize=16, fontweight='bold', y=.99)
        fig.get_layout_engine().set(rect=(0, .115, 1, .79))
        fig.text(.055, .027, run_label+'\nEach date shows 11:00–12:00 UTC only; connecting points does not imply continuous hourly coverage.\n'+note, fontsize=10)
        save(fig, 'observed_1h_'+scope)

    # A visual definition of the accumulation: the area under rate in one hour.
    cid = event['case_id']; date = pd.Timestamp(cid).strftime('%b %d, %Y')
    mask, fields, rates = event['mask'], event['fields'], event['rates']
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), layout='constrained')
    minutes = np.arange(0, 61, 10)
    curve_rows = []
    for ax, scope in zip(axs, ['regional', 'alta']):
        for name, rate in [('Observed MRMS', event['truth_rate']), *rates.items()]:
            c = cumulative(rate)
            if c.ndim == 4: c = c.mean(0)
            v = c[:,mask].mean(1) if scope == 'regional' else c[:,iy,ix]
            ax.plot(minutes, v, marker='o', markersize=3, color=OBS if name == 'Observed MRMS' else COLORS[name],
                    lw=2.7 if name == 'Observed MRMS' else 1.5, label=name)
            for minute, value in zip(minutes, v):
                curve_rows.append(dict(case_id=cid, scope=scope, model=name, minutes_since_1100_utc=minute, accumulation_mm=value))
        h = fields['HRRR'];v = float(h[mask].mean()) if scope == 'regional' else float(h[iy,ix])
        ax.scatter([60], [v], marker='D', s=75, color=COLORS['HRRR'], zorder=6, label='HRRR · hourly endpoint')
        curve_rows.append(dict(case_id=cid, scope=scope, model='HRRR', minutes_since_1100_utc=60, accumulation_mm=v))
        ax.set(xlabel='Minutes after 11:00 UTC', ylabel='Accumulated water · mm', xticks=minutes,
               title='Regional mean' if scope=='regional' else 'Alta grid cell', ylim=(0,None))
        ax.grid(alpha=.15)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc='lower center', fontsize=9)
    fig.suptitle(f'What one-hour accumulation looks like · {date}, 11:00–12:00 UTC', fontsize=15, fontweight='bold')
    fig.get_layout_engine().set(rect=(0, .23, 1, .66))
    fig.text(.06, .125, run_label+'\nIntegrate seven 10-minute rate snapshots over 60 minutes. HRRR supplies an hourly total, so no within-hour curve is invented.\nStorm selected by highest observed Alta-cell 1h amount. Curves use ensemble means; this single event does not rank models.', fontsize=9)
    save(fig, 'one_hour_event_curves')
    pd.DataFrame(curve_rows).to_csv(OUT/'one_hour_event_curves.csv', index=False)

    fig, axs = plt.subplots(2, 4, figsize=(15, 8), layout='constrained')
    all_fields = {'Observed MRMS':event['truth'], **fields}
    vmax = max(float(a[mask].max()) for a in all_fields.values())
    for ax, (name, a) in zip(axs.flat, all_fields.items()):
        value = np.where(mask, a, np.nan)
        im = ax.pcolormesh(lon, lat, value, vmin=0, vmax=vmax, cmap='YlGnBu', shading='nearest')
        ax.scatter([-111.63889], [40.57607], color='#d22e59', marker='*', s=85, edgecolors='white', linewidths=.5)
        ax.set_title(name, fontsize=11)
        ax.set(xlabel='Longitude', ylabel='Latitude', xlim=(lon[4],lon[-5]), ylim=(lat[-5],lat[4]))
        ax.tick_params(labelsize=8)
        ax.set_aspect(1/np.cos(np.deg2rad(40.57)))
    axs.flat[-1].axis('off')
    axs.flat[-1].text(.02,.93,
        f'{date}\n11:00–12:00 UTC\n\n★ Alta\n\nSame color scale and\nvalid cells for all maps.\nWhite = excluded by quality.\n\nWater equivalent, mm.\nNot fresh-snow depth.',
        va='top', fontsize=11, color=OBS, transform=axs.flat[-1].transAxes)
    fig.colorbar(im, ax=list(axs.flat[:-1]), label='One-hour accumulated water · mm', shrink=.75, pad=.02)
    fig.suptitle('Observed storm structure versus the forecasts', fontsize=16, fontweight='bold')
    fig.get_layout_engine().set(rect=(0,.075,1,.82))
    fig.text(.04,.013,run_label+'\nIllustration selected only by observed Alta-cell accumulation, not by any forecast error. All methods share the quality mask.', fontsize=10)
    save(fig, 'one_hour_event_maps')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir', type=Path, default=BASE/'radar_predictions')
    parser.add_argument('--metrics-path', type=Path, default=BASE/'short_case_metrics.csv')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--run-label', default='Original selected checkpoints · initial 1,500-update training budget')
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    source = args.metrics_path
    old = pd.read_csv(source, dtype={'case_id':str})
    old = old[old.hours==1].copy()
    ids = sorted(old.case_id.unique())
    assert len(ids)==180 and set(old.model)==set(ORDER)
    with np.load(BASE/'learning_arrays.npz') as a:
        cid_lookup = {str(cid):i for i,cid in enumerate(a['case_id'])}
        indices = np.array([cid_lookup[cid] for cid in ids])
        x, y = a['x'][indices], a['y'][indices]
        lat, lon = a['latitude'], a['longitude']
        times = a['radar_time'][indices]
    iy, ix = int(np.argmin(abs(lat-40.57607))), int(np.argmin(abs(lon+111.63889)))
    assert np.all(times[:,10].astype('datetime64[m]').astype(str)==np.array([pd.Timestamp(cid).strftime('%Y-%m-%dT11:00') for cid in ids]))
    assert np.all(times[:,16]-times[:,10]==np.timedelta64(60,'m'))
    preds, lookups = {}, {}
    for name in RADAR_LABELS:
        preds[name] = np.load(args.predictions_dir/f'{name}.npy', mmap_mode='r')
        with np.load(args.predictions_dir/f'{name}_index.npz') as z:
            lookups[name] = {str(cid):i for i,cid in enumerate(z['case_id'])}
    rows, event = [], None
    old_index = old.set_index(['case_id','model'])
    for j, cid in enumerate(ids):
        truth = integral(y[j],1)
        rates = {
            'Persistence':np.broadcast_to(np.nan_to_num(x[j,-1,0]),(1,20,64,64)),
            'Motion extrapolation':phase_flow(x[j])[None],
            **{label:np.asarray(preds[name][lookups[name][cid]],float) for name,label in RADAR_LABELS.items()}}
        fields = {name:integral(e,1).mean(0) for name,e in rates.items()}
        with np.load(ROOT/'data/alta_nowcast/hrrr/runs'/(cid+'T0600Z')/'precip_mm.npz') as h:
            fields['HRRR'] = h['mrms64'][h['periods']==6].sum(0)
        fields = {name:fields[name] for name in ORDER}
        mask = np.isfinite(y[j,1:8]).all(0)
        for a in fields.values(): mask &= np.isfinite(a)
        for name, predicted in fields.items():
            row = old_index.loc[(cid,name)]
            error = (predicted-truth)[mask]
            assert mask.sum() == row.pixels
            assert np.isclose(np.mean(error**2),row.mse,rtol=2e-6,atol=1e-9),(cid,name,'MSE')
            assert np.isclose(truth[iy,ix],row.alta_truth_mm,rtol=2e-6,atol=1e-9)
            assert np.isclose(predicted[iy,ix],row.alta_predicted_mm,rtol=2e-6,atol=1e-9)
            rows.append(dict(case_id=cid,date=pd.Timestamp(cid),model=name,
                regional_observed_mm=float(truth[mask].mean()), regional_predicted_mm=float(predicted[mask].mean()),
                alta_observed_mm=float(truth[iy,ix]), alta_predicted_mm=float(predicted[iy,ix]),
                valid_pixels=int(mask.sum()), regional_field_mse_mm2=float(np.mean(error**2))))
        observed_mean = float(truth[mask].mean())
        if event is None or float(truth[iy,ix]) > event['alta_observed_mm']:
            event = dict(case_id=cid,regional_observed_mm=observed_mean,alta_observed_mm=float(truth[iy,ix]),truth=truth.copy(),
                         truth_rate=y[j].copy(),rates={k:v.copy() for k,v in rates.items()},
                         fields={k:v.copy() for k,v in fields.items()},mask=mask.copy())
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'observed_1h_accumulation.csv',index=False)
    metrics=[]
    for model,d in frame.groupby('model',sort=False):
        for scope in ['regional','alta']:
            delta=d[scope+'_predicted_mm']-d[scope+'_observed_mm']
            metrics.append(dict(model=model,scope=scope,cases=len(d),mse_mm2=float((delta**2).mean()),
                mae_mm=float(abs(delta).mean()),rmse_mm=float(np.sqrt((delta**2).mean())),bias_mm=float(delta.mean())))
    pd.DataFrame(metrics).to_csv(OUT/'observed_1h_amount_metrics.csv',index=False)
    make_figures(frame,event,lat,lon,iy,ix,args.run_label)
    audit=dict(run_label=args.run_label,predictions_directory=str(args.predictions_dir),cases=len(ids),period_start='2023-11-01',period_end='2024-04-30',
        hourly_window_utc='11:00–12:00',case_frequency='one morning window per available day; not continuous hourly coverage',
        target='quality-masked MRMS radar liquid-equivalent precipitation accumulation; not independent hourly fresh-snow depth',
        integration='trapezoidal integration of seven 10-minute rate samples, 11:00 through 12:00 UTC inclusive',
        regional_metric='mean across common valid verification cells; area varies by date; not the spatial MSE used for model selection',
        alta_cell={'latitude':float(lat[iy]),'longitude':float(lon[ix]),'pooled_degrees':.04},
        illustrative_event={'case_id':event['case_id'],'criterion':'largest observed Alta-cell 1h accumulation, independent of all model errors',
            'regional_observed_mm':event['regional_observed_mm'], 'alta_observed_mm':float(event['truth'][iy,ix])},
        endpoint_fields_reproduced_original_metrics=True,models=list(ORDER),
        source_metrics_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        files=['observed_1h_regional.png','observed_1h_alta.png','one_hour_event_curves.png','one_hour_event_maps.png'])
    (OUT/'observed_1h_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
