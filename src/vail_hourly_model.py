"""Independent Vail hourly-observation / daily reported-Sno24 experiment.

All times retain CAIC's display clock. Targets are one 04:00 reading/day;
hourly Sno24 readings are NEVER summed. Each origin uses only its past 168 h.
The target is a provisional station proxy, not a verified manual resort total.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from advanced_snow_models import TemporalResidualBlock, ChannelLayerNorm
from deep_dillon import calendar_features, lagged_roni, distribution_mean, distribution_quantile, seed_everything, logloss
from vail_baselines import fit_predict_baselines, tabular_summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/vail-hourly'
CONTEXT, HORIZON, SEED = 168, 60, 20261104
STATIONS = ['CAVMM', 'CAVPQ', 'CAVBS', 'CAVCB']
CONFIGS = [dict(width=24, dropout=.25, lr=.001, weight_decay=.01),
           dict(width=32, dropout=.35, lr=.0006, weight_decay=.03)]


def load_data():
    raw = pd.read_csv(ROOT/'data/vail_hourly/hourly.csv', low_memory=False)
    raw['timestamp'] = pd.to_datetime(raw.datetime_display)
    assert not raw.duplicated(['station_id', 'timestamp']).any()
    features = {}
    for st in STATIONS:
        part = raw[raw.station_id == st].set_index('timestamp').sort_index()
        for col, low, high in [('Temp', -70, 110), ('DewP', -90, 100), ('RH', 0, 100), ('Spd', 0, 150), ('Gst', 0, 200)]:
            values = part[col].where(part[col].between(low, high))
            features[f'{st}_{col}'] = values
        direction = part.Dir.where(part.Dir.between(0, 360))*np.pi/180
        speed = features[f'{st}_Spd']
        features[f'{st}_wind_u'] = -speed*np.sin(direction)
        features[f'{st}_wind_v'] = -speed*np.cos(direction)
    mid = raw[raw.station_id == 'CAVMM'].set_index('timestamp').sort_index()
    # No arbitrary positive-trace threshold; retain possible offsets in labels.
    snow = mid.Sno24.where(mid.Sno24 >= 0)
    # Pre-fit source QA found unreset accumulation after March 20, 2020.
    # Mask the same invalid regime in context features and target labels.
    snow.loc['2020-03-20':'2020-05-31'] = np.nan
    # Pre-opening 2023 repeated decaying board readings before Nov29 reset.
    snow.loc['2023-10-01':'2023-11-28'] = np.nan
    features['CAVMM_Sno24'] = np.log1p(snow)
    phq = raw[raw.station_id == 'CAVPQ'].set_index('timestamp').sort_index()
    for col, low, high in [('Pcp1', 0, 10), ('MSLP', 850, 1100)]:
        features[f'CAVPQ_{col}'] = phq[col].where(phq[col].between(low, high))
    features['temperature_difference_PQ_MM'] = features['CAVPQ_Temp']-features['CAVMM_Temp']
    features['temperature_difference_BS_MM'] = features['CAVBS_Temp']-features['CAVMM_Temp']
    frame = pd.DataFrame(features).sort_index()
    target = snow[snow.index.hour == 4].rename('reported_sno24_inches')
    target.to_csv(OUT/'daily_target_0400.csv', index_label='reporting_endpoint_source_clock')
    coverage = []
    for year in range(frame.index.min().year, 2026):
        vals = target.loc[f'{year}-10-01':f'{year+1}-04-30']
        coverage.append(dict(winter=year, reported_days=int(vals.notna().sum()),
                             mean=float(vals.mean()), max=float(vals.max()),
                             negative_raw=int((mid.Sno24.loc[f'{year}-10-01':f'{year+1}-04-30'] < 0).sum()),
                             positive_days=int((vals > 0).sum()), one_inch_days=int((vals >= 1).sum())))
    years = [r['winter'] for r in coverage if r['reported_days'] >= 100]
    (OUT/'target_coverage.json').write_text(json.dumps(coverage, indent=2))
    return raw, frame, target, years


def make_arrays(frame, target, years):
    records, xs, fs, ys = [], [], [], []
    for year in years:
        for origin in pd.date_range(f'{year}-10-08 04:00', f'{year+1}-02-28 04:00', freq='2D'):
            x = frame.reindex(pd.date_range(end=origin, periods=CONTEXT, freq='h')).to_numpy(float)
            dates = pd.date_range(origin+pd.Timedelta(days=1), periods=HORIZON)
            y = target.reindex(dates).to_numpy(float)
            # Quality-based eligibility is declared before fitting, common to every model.
            if np.isfinite(y).sum() < 45 or np.isfinite(x[:, 0]).sum() < CONTEXT*.5:
                continue
            records.append(dict(winter=year, origin=str(origin)))
            xs.append(x); fs.append(future_features(origin)); ys.append(y)
    return pd.DataFrame(records), np.asarray(xs, dtype='float32'), np.asarray(fs, dtype='float32'), np.asarray(ys, dtype='float32')


def future_features(origin):
    dates = pd.date_range(origin+pd.Timedelta(days=1), periods=HORIZON)
    lead = np.arange(1, HORIZON+1)/HORIZON
    roni = lagged_roni(origin)
    known = [roni if np.isfinite(roni) else 0., float(roni >= 1.5), float(roni <= -1.5), float(not np.isfinite(roni))]
    return np.column_stack([calendar_features(dates), lead, lead**2,
                            np.tile(known, (HORIZON, 1))]).astype('float32')


class HourlyScaler:
    def fit(self, x):
        # Context windows overlap; weights are repeated equally across this fit split only.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.mean = np.nanmean(x, axis=(0, 1))
            self.std = np.nanstd(x, axis=(0, 1))
        self.mean = np.nan_to_num(self.mean, nan=0.)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-6), self.std, 1.)
        return self

    def fit_future(self, f):
        self.future_min = f.min(axis=(0, 1))
        self.future_max = f.max(axis=(0, 1))
        return self

    def transform_future(self, f):
        result = f.copy()
        if hasattr(self, 'future_min'):
            constant = self.future_max-self.future_min < 1e-7
            # A never-varying feature has no learned effect. Do not activate
            # arbitrary decoder weights when a new category first appears.
            result[..., constant] = self.future_min[constant]
            # Avoid unsupported ENSO magnitude extrapolation in this short archive.
            result[..., 6] = np.clip(result[..., 6], self.future_min[6], self.future_max[6])
        return result

    def transform(self, x):
        missing = ~np.isfinite(x)
        scaled = (x-self.mean)/self.std
        scaled[missing] = 0.
        # Same within-day hour at every daily issue; cycle remains known.
        hours = np.arange(4-CONTEXT+1, 5) % 24
        cycles = np.tile(np.stack([np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24)], -1), (len(x), 1, 1))
        return np.concatenate([scaled, missing, cycles], axis=-1).astype('float32')

    def as_dict(self):
        result = dict(mean=self.mean.tolist(), std=self.std.tolist(), policy='fit-context mean imputation plus missingness channels; no future fill')
        if hasattr(self, 'future_min'):
            result.update(future_min=self.future_min.tolist(), future_max=self.future_max.tolist(),
                          future_policy='constant training covariates held at their training value; RONI magnitude clipped to training range; fit split only')
        return result


class VailTCN(nn.Module):
    def __init__(self, channels, width=24, dropout=.25):
        super().__init__()
        self.projection = nn.Conv1d(channels, width, 1)
        self.blocks = nn.Sequential(*[TemporalResidualBlock(width, 3, 2**i, dropout) for i in range(6)])
        self.norm = ChannelLayerNorm(width)
        self.decoder = nn.Sequential(nn.Linear(2*width+10, 96), nn.SiLU(), nn.Dropout(dropout),
                                     nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 3))
        with torch.no_grad():
            self.decoder[-1].bias.copy_(torch.tensor([.5, -.3, .2]))

    def forward(self, x, future):
        z = self.norm(self.blocks(self.projection(x.transpose(1, 2))))
        state = torch.cat([z[:, :, -1], z.mean(-1)], -1)[:, None, :].expand(-1, future.shape[1], -1)
        raw = self.decoder(torch.cat([state, future], -1))
        # Wider log-amount support accommodates the station's small positive offsets.
        return raw[..., 0], 5*torch.tanh(raw[..., 1]/5), .2+1.8*torch.sigmoid(raw[..., 2])


def train_model(x, f, y, fit, valid, config, epochs=35, seed=SEED):
    seed_everything(seed)
    scaler = HourlyScaler().fit(x[fit]).fit_future(f[fit])
    xx = scaler.transform(x[fit])
    yy = np.nan_to_num(y[fit], nan=0.)
    mm = np.isfinite(y[fit]).astype('float32')
    tensors = [torch.tensor(a, dtype=torch.float32) for a in [xx, scaler.transform_future(f[fit]), yy, mm]]
    loader = DataLoader(TensorDataset(*tensors), batch_size=48, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = VailTCN(xx.shape[-1], config['width'], config['dropout'])
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    best, best_epoch, state, logs = float('inf'), 1, None, []
    for epoch in range(1, epochs+1):
        model.train(); running = 0.
        for bx, bf, by, bm in loader:
            opt.zero_grad(); params = model(bx, bf)
            loss = logloss(params, by, bm)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); running += float(loss.detach())*len(bx)
        row = dict(epoch=epoch, train_nll=running/len(fit))
        if valid is not None:
            pp = predict(model, scaler, x[valid], f[valid])
            val_y = y[valid]; ok = np.isfinite(val_y)
            # Validation selects by predictive mean squared error, matching requested benchmark.
            score = float(np.mean((distribution_mean(pp)[ok]-val_y[ok])**2))
            row['validation_mse'] = score
            if score < best-1e-6:
                best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        else:
            best_epoch = epoch
        logs.append(row)
        if valid is not None and epoch >= 12 and epoch-best_epoch >= 8:
            break
    if state is not None: model.load_state_dict(state)
    return model, scaler, logs, best_epoch, best


def predict(model, scaler, x, f):
    model.eval(); results = []
    with torch.no_grad():
        for i in range(0, len(x), 96):
            logits, mu, sigma = model(torch.tensor(scaler.transform(x[i:i+96])), torch.tensor(scaler.transform_future(f[i:i+96])))
            results.append(np.stack([torch.sigmoid(logits).numpy(), mu.numpy(), sigma.numpy()], -1))
    return np.concatenate(results)


def climate(target, years, origins):
    values = pd.concat([target.loc[f'{year}-10-01':f'{year+1}-04-30'] for year in years]).dropna()
    # Month/day mapped onto a leap year gives a stable circular calendar distance.
    day = np.array([pd.Timestamp(2000, d.month, d.day).dayofyear for d in values.index])
    lookup = {}
    for d in range(1, 367):
        delta = np.abs(day-d); sample = values.to_numpy()[np.minimum(delta, 366-delta) <= 15]
        if len(sample) == 0: sample = values.to_numpy()
        lookup[d] = [np.mean(sample), np.quantile(sample, .1), np.quantile(sample, .9)]
    return np.array([[lookup[pd.Timestamp(2000, d.month, d.day).dayofyear]
                     for d in pd.date_range(pd.Timestamp(origin)+pd.Timedelta(days=1), periods=HORIZON)] for origin in origins])


def get_issue_snow(x, columns):
    # Only exact issue reading; absent values remain absent for baseline imputation.
    return np.expm1(x[:, -1, columns.index('CAVMM_Sno24')])


def select_and_fit(meta, x, f, y, train_idx, tag):
    years = sorted(map(int, meta.iloc[train_idx].winter.unique()))
    rng = np.random.default_rng(SEED+sum(years))
    val_years = sorted(rng.choice(years, size=max(1, len(years)//4), replace=False).tolist())
    valid = train_idx[meta.iloc[train_idx].winter.isin(val_years).to_numpy()]
    fit = train_idx[~meta.iloc[train_idx].winter.isin(val_years).to_numpy()]
    trials = []
    for c, config in enumerate(CONFIGS):
        model, scaler, logs, epoch, score = train_model(x, f, y, fit, valid, config, seed=SEED+c)
        trials.append(dict(config=config, best_epoch=epoch, validation_mse=score, logs=logs))
        print(json.dumps(dict(stage=tag, config=c, epochs=epoch, validation_mse=score)), flush=True)
    chosen = min(trials, key=lambda t:t['validation_mse'])
    model, scaler, logs, _, _ = train_model(x, f, y, train_idx, None, chosen['config'], epochs=chosen['best_epoch'])
    record = dict(tag=tag, training_winters=years, inner_validation_winters=val_years,
                  selected_config=chosen['config'], selected_epochs=chosen['best_epoch'],
                  trials=trials, refit_logs=logs, parameter_count=sum(p.numel() for p in model.parameters()))
    (OUT/f'{tag}_training.json').write_text(json.dumps(record, indent=2))
    return model, scaler, record


def evaluate_fold(meta, x, f, y, target, columns, train_idx, test_idx, tag):
    model, scaler, record = select_and_fit(meta, x, f, y, train_idx, tag)
    params = predict(model, scaler, x[test_idx], f[test_idx])
    years = sorted(map(int, meta.iloc[train_idx].winter.unique()))
    cl = climate(target, years, meta.iloc[test_idx].origin)
    summaries = tabular_summary(x)
    result = fit_predict_baselines(summaries[train_idx], f[train_idx], y[train_idx], np.isfinite(y[train_idx]),
                                  summaries[test_idx], f[test_idx], get_issue_snow(x[train_idx], columns),
                                  get_issue_snow(x[test_idx], columns))
    means = result[0] if isinstance(result, tuple) else result
    means.update({'Climatology':cl[..., 0], 'Hourly TCN':distribution_mean(params)})
    records = []
    for name, pred in means.items():
        for j, ix in enumerate(test_idx):
            for h in range(HORIZON):
                if not np.isfinite(y[ix, h]): continue
                row = dict(split=tag, model=name, winter=int(meta.iloc[ix].winter), origin=meta.iloc[ix].origin,
                           date=str((pd.Timestamp(meta.iloc[ix].origin)+pd.Timedelta(days=h+1)).date()),
                           lead=h+1, actual_inches=float(y[ix, h]), predicted_inches=float(pred[j, h]))
                if name == 'Hourly TCN':
                    row.update(q10_inches=float(distribution_quantile(params[j, h], .1)), q90_inches=float(distribution_quantile(params[j, h], .9)))
                elif name == 'Climatology': row.update(q10_inches=float(cl[j, h, 1]), q90_inches=float(cl[j, h, 2]))
                records.append(row)
    pd.DataFrame(records).to_csv(OUT/f'{tag}_predictions.csv', index=False)
    return pd.DataFrame(records), record


def metrics(predictions):
    result = []
    bands = [(1,60,'1–60'), (1,1,'1'), (2,3,'2–3'), (4,7,'4–7'), (8,14,'8–14'), (15,30,'15–30'), (31,60,'31–60')]
    for name, rows in predictions.groupby('model'):
        for low, high, label in bands:
            r = rows[rows.lead.between(low, high)]; err = r.predicted_inches-r.actual_inches
            record = dict(model=name, leads=label, pairs=len(r), winters=r.winter.nunique(), unique_days=r.date.nunique(),
                          mse=float(np.mean(err**2)), rmse=float(np.sqrt(np.mean(err**2))), mae=float(np.mean(abs(err))), bias=float(err.mean()))
            if r.q10_inches.notna().all():
                a,b = r.q10_inches,r.q90_inches
                record.update(coverage80=float(((r.actual_inches>=a)&(r.actual_inches<=b)).mean()),
                              width80=float((b-a).mean()), interval_score80=float((b-a+10*np.maximum(a-r.actual_inches,0)+10*np.maximum(r.actual_inches-b,0)).mean()))
            result.append(record)
    return pd.DataFrame(result)


def paired_bootstrap(rows):
    keys = ['winter', 'origin', 'lead']
    neural = rows[rows.model=='Hourly TCN'].set_index(keys)
    rng = np.random.default_rng(SEED); result = []
    for name in rows.model.unique():
        if name == 'Hourly TCN': continue
        other = rows[rows.model==name].set_index(keys)
        assert neural.index.equals(other.index)
        delta = (neural.predicted_inches-neural.actual_inches)**2-(other.predicted_inches-other.actual_inches)**2
        agg = delta.groupby(level='winter').agg(['sum','count'])
        selections = rng.integers(0,len(agg),(4000,len(agg)))
        draws = agg['sum'].to_numpy()[selections].sum(1)/agg['count'].to_numpy()[selections].sum(1)
        result.append(dict(comparison='TCN minus '+name, mse_difference=float(delta.mean()), ci95_low=float(np.quantile(draws,.025)), ci95_high=float(np.quantile(draws,.975)), clusters=len(agg)))
    return result


def forecast_paths(target, years, first, params, n=2048):
    templates = []
    for year in years:
        anchor = pd.Timestamp(year, first.month, first.day, 4)
        for offset in range(-14,15):
            vals = target.reindex(pd.date_range(anchor+pd.Timedelta(days=offset), periods=len(params))).to_numpy(float)
            if np.isfinite(vals).all(): templates.append(vals)
    if not templates: raise ValueError('No complete observed trajectories; do not silently interpolate target')
    rng = np.random.default_rng(SEED)
    sample = np.array(templates)[rng.integers(len(templates),size=n)]
    ranks = np.empty_like(sample)
    for h in range(sample.shape[1]):
        order = np.lexsort((rng.random(n),sample[:,h])); ranks[order,h] = np.arange(n)+.5
    return distribution_quantile(params[None,:,:],ranks/n),len(templates)


def issue_forecast(meta,x,f,y,frame,target,years):
    model,scaler,record = select_and_fit(meta,x,f,y,np.arange(len(meta)),'final')
    latest = frame.index.max()
    origin = latest.normalize()+pd.Timedelta(hours=4)
    if origin > latest: origin -= pd.Timedelta(days=1)
    fx = frame.reindex(pd.date_range(end=origin, periods=CONTEXT, freq='h')).to_numpy('float32')[None]
    ff = future_features(origin)[None]
    params = predict(model,scaler,fx,ff)[0]
    dates = pd.date_range(origin+pd.Timedelta(days=1),periods=HORIZON)
    keep = dates <= pd.Timestamp('2026-12-31 04:00')
    dates,params = dates[keep],params[keep]
    cl = climate(target,years,[origin])[0][keep]
    paths,n_templates = forecast_paths(target,years,dates[0],params)
    forecast = pd.DataFrame(dict(date=dates.date, reporting_endpoint_source_clock=dates, lead=np.arange(1,len(dates)+1),
                                 mean_inches=distribution_mean(params), median_inches=distribution_quantile(params,.5),
                                 q10_inches=distribution_quantile(params,.1), q90_inches=distribution_quantile(params,.9),
                                 probability_at_least_one_inch=params[:,0]*(1-.5*(1+__import__('scipy').special.erf((0-params[:,1])/(params[:,2]*np.sqrt(2))))),
                                 historical_mean_inches=cl[:,0], historical_q10_inches=cl[:,1], historical_q90_inches=cl[:,2]))
    forecast.to_csv(OUT/'forecast_daily.csv',index=False)
    np.savez_compressed(OUT/'forecast_paths.npz',paths_inches=paths,dates=dates.astype(str).to_numpy(dtype=str))
    # Summing endpoint board readings is not validated fresh snowfall.
    # Keep simulated proxy paths for research, but publish no monthly snow total.
    checkpoint=dict(state_dict=model.state_dict(),scaler=scaler.as_dict(),config=record['selected_config'],features=list(frame.columns),training_winters=years,context_hours=CONTEXT,horizon_days=HORIZON)
    torch.save(checkpoint,OUT/'vail_hourly_tcn.pt')
    manifest=dict(issue_endpoint_source_clock=str(origin),last_hour_available_source_clock=str(latest),
                  first_reporting_endpoint=str(dates[0]),last_reporting_endpoint=str(dates[-1]),
                  target='CAVMM CAIC-reported Sno24 at 04:00; provisional sensor endpoint proxy, NOT verified 24-hour fresh snowfall; resets observed at other hours',
                  timestamp_policy='CAIC display clock; historical UTC offset, DST and publication latency unverified',
                  training_winters=years,parameters=record['parameter_count'],architecture='6 residual blocks / 12 causal convolutions; 253-hour receptive field; 168-hour observed input; shared direct daily hurdle-lognormal decoder',
                  input_columns=list(frame.columns),excluded_inputs='Snow depth excluded for suspicious plateaus; CAVMM Sno24 March20–May31 2020 and October1–November28 2023 masked after source QA; no future observed weather; no future ENSO observations',
                  roni_at_origin=float(lagged_roni(origin)),roni_used_by_model=float(scaler.transform_future(ff)[0,0,6]),
                  roni_training_range=[float(scaler.future_min[6]),float(scaler.future_max[6])],
                  roni_note='Lagged revised RONI, not release-vintage archive; fixed for all forecast leads. Neural ENSO magnitude clipped to fit range; any indicator constant in training is held at its training value. Strong El Niño effect is unidentifiable in these six winters.',
                  paths=2048,historical_dependence_templates=n_templates,interval_policy='Uncalibrated model-based pointwise 10–90% predictive ranges; historical rank-template copula gives sensor-proxy paths only; no validated monthly fresh-snow total',
                  forecast_input_missing_fraction=float(np.mean(~np.isfinite(fx))),source_sha256=hashlib.sha256((ROOT/'data/vail_hourly/hourly.csv').read_bytes()).hexdigest())
    (OUT/'forecast_manifest.json').write_text(json.dumps(manifest,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','run','forecast'],default='run');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    raw,frame,target,years=load_data()
    meta,x,f,y=make_arrays(frame,target,years)
    meta.to_csv(OUT/'origins.csv',index=False)
    np.savez_compressed(OUT/'training_arrays.npz',x=x,f=f,y=y)
    plan=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),source_sha256=hashlib.sha256((ROOT/'data/vail_hourly/hourly.csv').read_bytes()).hexdigest(),
              target='exact04:00 nonnegative CAVMM reported Sno24 endpoint proxy; no hourly summation; no trace correction; March20–May31 2020 and October1–November28 2023 excluded after source QA',years=years,
              origins=len(meta),channels=list(frame.columns),context_hours=168,horizon_days=60,origin_stride_days=2,
              outer_folds=5 if len(years)>=8 else 3,seed=SEED,configs=CONFIGS,
              future_covariate_policy='Neural covariates constant within fit held at training value; RONI magnitude clipped to fit range, including inner validation and final forecast',
              validation='Shuffled whole-winter outer folds; one inner whole-winter validation split for two configurations and epoch selection; refit all outer-training winters; separate latest-two-winter chronological stress test',
              interval='Uncalibrated hurdle-lognormal predictive distribution; test coverage reported, never adjusted using test labels')
    if args.mode!='forecast': (OUT/'frozen_plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(stage='prepared',years=years,origins=len(meta),shape=x.shape,valid_labels=int(np.isfinite(y).sum()))),flush=True)
    if args.mode=='prepare':return
    if args.mode=='run':
        folds=KFold(n_splits=plan['outer_folds'],shuffle=True,random_state=SEED)
        all_rows=[]
        for fold,(tr,te) in enumerate(folds.split(years)):
            train_idx=np.flatnonzero(meta.winter.isin(np.array(years)[tr]));test_idx=np.flatnonzero(meta.winter.isin(np.array(years)[te]))
            rows,record=evaluate_fold(meta,x,f,y,target,list(frame.columns),train_idx,test_idx,f'fold{fold}')
            all_rows.append(rows)
        predictions=pd.concat(all_rows,ignore_index=True)
        predictions.to_csv(OUT/'crossvalidation_predictions.csv',index=False)
        metrics(predictions).to_csv(OUT/'crossvalidation_metrics.csv',index=False)
        (OUT/'paired_error_bootstrap.json').write_text(json.dumps(paired_bootstrap(predictions),indent=2))
        train_idx=np.flatnonzero(meta.winter.isin(years[:-2]));test_idx=np.flatnonzero(meta.winter.isin(years[-2:]))
        rows,_=evaluate_fold(meta,x,f,y,target,list(frame.columns),train_idx,test_idx,'chronological')
        metrics(rows).to_csv(OUT/'chronological_metrics.csv',index=False)
    issue_forecast(meta,x,f,y,frame,target,years)
    print('DONE',flush=True)


if __name__=='__main__':main()
