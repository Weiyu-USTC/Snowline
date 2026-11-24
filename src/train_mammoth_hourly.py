"""Forecast hourly Parsivel2 snowfall water equivalent at the Jeff Dozier site.

Inference needs NumPy and a small, portable JSON artifact. Training uses only
issued HRRR forecasts, earlier weather radar, and the historical station target.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/mammoth-hourly'
HORIZONS = (1, 3, 6, 12, 24, 48)
WEATHER = ('snow_water_mm', 'precip_mm', 'temperature_c')
RADAR = ('point', 'local', 'regional', 'west', 'quality')
TARGET_LAT, TARGET_LON = 37.6431, -119.0291
VERSION = 1


def _time(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Forecast times must have a UTC offset')
    return value.astimezone(timezone.utc)


def radar_features(path):
    """Read quality-screened CUES radar rates from one matched, dated MRMS crop."""
    with np.load(path, allow_pickle=False) as crop:
        meta = json.loads(crop['metadata'].item())
        if _time(meta['timestamp']) != _time(meta['quality_timestamp']):
            raise ValueError('MRMS timestamps differ')
        rates, quality = crop['precip_rate_mm_h'], crop['radar_quality_index']
        lat, lon = crop['latitude'], crop['longitude']
        i, j = np.abs(lat - TARGET_LAT).argmin(), np.abs(lon - TARGET_LON).argmin()
        if abs(lat[i] - TARGET_LAT) > .008 or abs(lon[j] - TARGET_LON) > .008:
            raise ValueError('Radar crop does not cover the Jeff Dozier site')
        usable = np.isfinite(quality) & (quality >= .1) & (quality <= 1) & np.isfinite(rates) & (rates >= 0)

        def area(rows, cols):
            valid = usable[rows, cols]
            return float(rates[rows, cols][valid].mean()) if valid.size and valid.mean() >= .5 else None

        return {'timestamp': meta['timestamp'],
                'point': float(rates[i, j]) if usable[i, j] else None,
                'quality': float(quality[i, j]) if 0 <= quality[i, j] <= 1 else None,
                'local': area(slice(i-2, i+3), slice(j-2, j+3)),
                'regional': area(slice(i-10, i+11), slice(j-10, j+11)),
                'west': area(slice(i-10, i+11), slice(j-20, j-2))}


def _weather_array(weather):
    if len(weather) != 48:
        raise ValueError('Mammoth inference needs exactly 48 hourly weather slots')
    values = np.asarray([[row.get(key) for key in WEATHER] for row in weather], dtype=float)
    if np.any(np.isinf(values)) or np.any(values[:, :2] < 0):
        raise ValueError('Invalid hourly HRRR values')
    if not np.isfinite(values[:36]).all():
        raise ValueError('Mammoth requires 36 contiguous hours of native HRRR guidance')
    return values


def _features(issue, raw, radar, clim):
    """Build a coherent 48-hour trajectory from observations available at issue."""
    from datetime import timedelta
    issue = _time(issue)
    ts = [issue + timedelta(hours=i) for i in range(48)]
    climate = np.array([clim[t.month-1] for t in ts])
    # Values beyond HRRR's native endpoint use a receding mean of the last
    # three *native* hours. Age, missingness, and persistence are predictors.
    w = np.array(raw, copy=True)
    tail = np.zeros(48)
    for k in range(3):
        for i in np.flatnonzero(~np.isfinite(raw[:, k])):
            prior = np.flatnonzero(np.isfinite(raw[:i, k]))
            if not len(prior):
                raise ValueError('HRRR has no earlier hour for this predictor')
            gap = i - prior[-1]
            tail[i] = max(tail[i], gap)
            w[i, k] = np.mean(raw[prior[-3:], k]) * (np.exp(-gap / 6.) if k < 2 else 1.)
    snow, precip, temp = w.T
    radar = radar or {}
    age = (issue - _time(radar['timestamp'])).total_seconds() / 3600 if radar.get('timestamp') else np.nan
    available = np.isfinite(age) and (5/60 <= age <= 1.5)
    rad = np.asarray([radar.get(k) for k in RADAR], dtype=float) if available else np.full(5, np.nan)
    decay = np.exp(-(np.arange(48) + (age if available else 0)) / 3.)
    # Neighbouring hours of the same issued NWP run are also known at issue.
    smooth = np.convolve(np.pad(snow, 2, mode='edge'), np.ones(5)/5, mode='valid')
    phase = np.clip((2.-temp)/4., 0., 1.)
    annual = np.asarray([2 * np.pi * (t.timetuple().tm_yday-1)/365.25 for t in ts])
    hour = np.asarray([2 * np.pi * t.hour / 24 for t in ts])
    base = np.column_stack([snow, np.sqrt(snow), precip, precip*phase, temp, smooth, climate,
                            np.sin(annual), np.cos(annual), np.sin(hour), np.cos(hour),
                            np.arange(1, 49)/48, tail, tail > 0])
    rcols = [np.full(48, float(np.isfinite(rad[0]))) * decay]
    rcols.extend((value if np.isfinite(value) else 0.) * decay for value in rad)
    return np.column_stack([base, *rcols]), climate, snow


def _run(estimator, X, climate, native_tail):
    family = estimator['kind']
    if family == 'climatology':
        return climate
    if family == 'hrrr_tail':
        return native_tail
    if family == 'scaled_hrrr':
        return np.power(native_tail, estimator.get('power',1.)) * estimator['scale']
    if family == 'radar_update':
        # Only a new, quality-screened echo changes the short-range guidance.
        # Poor or missing radar leaves the fitted weather-only baseline intact.
        return native_tail * estimator['scale'] + np.maximum(X[:, 16:19], 0.) @ estimator['gains']
    X = X[:, :estimator['features']]
    z = (X - np.asarray(estimator['center'])) / np.asarray(estimator['spread'])
    if family == 'ridge':
        result = z @ np.asarray(estimator['weights']) + estimator['intercept']
    else:
        for index, (weights, bias) in enumerate(zip(estimator['weights'], estimator['biases'])):
            z = z @ np.asarray(weights) + np.asarray(bias)
            if index < len(estimator['weights'])-1:
                z = np.maximum(z, 0)
        result = z.ravel() * estimator['target_scale']
    return np.maximum(0., result)


def predict_hourly(model, issue, weather, radar=None):
    """Predict 48 hourly instrument SWE totals in mm, starting at issue (UTC)."""
    start = _time(issue)
    if start.minute or start.second or start.microsecond or model['version'] != VERSION:
        raise ValueError('Unsupported model or forecast start')
    X, climate, tail = _features(start, _weather_array(weather), radar, model['monthly_means_mm'])
    return _run(model['estimator'], X, climate, tail).tolist()


def forecast_interval(model, hours, total):
    q = model.get('calibration', {}).get(str(hours))
    if not q:
        return None, None
    return max(0., total+q['lower_offset_mm']), max(0., total+q['upper_offset_mm'])


def _radar_history(root):
    import pandas as pd
    archive = pd.read_csv(root/'data/mammoth/radar/historical_points.csv')
    rows = [radar_features(root/path) for path in archive.archive_path]
    return pd.DataFrame(rows).set_index('timestamp').set_axis(
        pd.to_datetime([r['timestamp'] for r in rows], utc=True)).sort_index()


def _training_data(root, weather_path):
    import pandas as pd
    ground = pd.read_csv(root/'data/mammoth/research/cues_disdrometer_hourly.csv.gz')
    ground.index = pd.to_datetime(ground.start_utc, utc=True)
    ground['target'] = ground.snow_swe_mm.where((ground.snow_valid_minutes == 60) & (ground.snow_rate_at_cap_minutes == 0))
    # Keep observed, genuine zeros. Remove incomplete and cap-contaminated hours.
    base = ground.loc['2020-10-01':'2022-09-30', 'target'].dropna()
    cl = base.groupby(base.index.month).agg(['sum', 'count'])
    clim = ((cl['sum'] + 48 * base.mean()) / (cl['count'] + 48)).reindex(range(1,13)).fillna(base.mean()).values
    radar = _radar_history(root)
    hrrr = pd.read_csv(weather_path)
    hrrr['run_utc'] = pd.to_datetime(hrrr.run_utc, utc=True)
    hrrr['start_utc'] = pd.to_datetime(hrrr.start_utc, utc=True)
    hrrr['end_utc'] = pd.to_datetime(hrrr.end_utc, utc=True)
    hrrr['available_utc'] = pd.to_datetime(hrrr.available_utc, utc=True)
    if hrrr.duplicated(['run_utc', 'start_utc']).any() or not (hrrr.end_utc-hrrr.start_utc).eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Duplicate or non-hourly HRRR window')
    hrrr = hrrr.sort_values(['run_utc','start_utc'])
    windows = []
    for run, frame in hrrr.groupby('run_utc'):
        native = frame.set_index('start_utc')[list(WEATHER)]
        # Recreate each hour for which this six-hourly extended cycle is the
        # latest available (+3 h publication allowance, until next one at +9).
        for offset in range(4, 10):
            issue = run + pd.Timedelta(hours=offset)
            issued_at = issue - pd.Timedelta(minutes=30)
            if frame.available_utc.max() > issued_at:
                continue
            times = pd.date_range(issue, periods=48, freq='h')
            year = issue.year - int(issue.month < 10)
            if any((t.year - int(t.month < 10)) != year for t in times):
                continue
            weather = native.reindex(times).to_numpy(dtype=float)
            if not np.isfinite(weather[0]).all() or not np.isfinite(weather[:36]).all():
                continue
            observed = ground.target.reindex(times).to_numpy()
            if not np.isfinite(observed).any():
                continue
            # As-of search includes a 5-minute publication allowance. Rain-rate
            # measurements from within the verification window are inaccessible.
            idx = radar.index.searchsorted(issued_at - pd.Timedelta(minutes=5), side='right') - 1
            scan = None
            if idx >= 0 and issue-radar.index[idx] <= pd.Timedelta(minutes=90):
                scan = dict(radar.iloc[idx], timestamp=radar.index[idx].isoformat())
            x, climate, tail = _features(issue, weather, scan, clim)
            windows.append(dict(run=run, issue=issue, issued_at=issued_at, season=year, x=x, obs=observed, climate=climate,
                                hrrr=weather[:, 0], tail=tail,
                                radar=float(scan['point']) if scan else np.nan,
                                radar_utc=scan['timestamp'] if scan else None))
    return windows, clim


def _fit_candidates(train, seed=104):
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    X = np.concatenate([v['x'] for v in train])
    y = np.concatenate([v['obs'] for v in train])
    valid = np.isfinite(y)
    X, y = X[valid], y[valid]
    native = X[:, 0]
    scale = float(np.dot(native, y)/np.dot(native, native)) if np.dot(native, native)>0 else 1.
    yield 'Calendar climatology', {'kind': 'climatology'}
    yield 'HRRR with tail persistence', {'kind': 'hrrr_tail'}
    yield 'Calibrated HRRR', {'kind': 'scaled_hrrr', 'scale': max(0.,scale), 'power':1.}
    for power in (.75,1.25):
        feature = np.power(native,power)
        factor = float(np.dot(feature,y)/np.dot(feature,feature)) if np.dot(feature,feature)>0 else 1.
        yield f'Power-calibrated HRRR {power:g}', {'kind':'scaled_hrrr','power':power,'scale':max(0.,factor)}
    from scipy.optimize import nnls
    echo = X[:, 16:19]
    gains, _ = nnls(echo, y - native * scale)
    yield 'Calibrated HRRR + radar update', {'kind':'radar_update','scale':max(0.,scale),'gains':gains.tolist()}
    for nfeat, name in ((14,'Weather'), (X.shape[1],'Weather + radar')):
        scaler = StandardScaler().fit(X[:, :nfeat])
        z = scaler.transform(X[:, :nfeat])
        base = {'features': nfeat, 'center': scaler.mean_.tolist(), 'spread': scaler.scale_.tolist()}
        for penalty in (.03, .3):
            fit = Ridge(alpha=penalty * len(y)).fit(z, y)
            yield f'{name} ridge {penalty:g}', dict(base,kind='ridge',weights=fit.coef_.tolist(),intercept=float(fit.intercept_))
        for scale_penalty in (.1, 1.):
            yscale = max(1.,float(y.std()))
            fit = MLPRegressor(hidden_layer_sizes=(24,12), activation='relu', solver='adam',
                               learning_rate_init=.002, alpha=scale_penalty, max_iter=75,
                               n_iter_no_change=12, tol=.0001, random_state=seed, batch_size=256)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                fit.fit(z, y / yscale)
            yield f'{name} neural network {scale_penalty:g}',dict(base, kind='network',
                     weights=[w.tolist() for w in fit.coefs_], biases=[b.tolist() for b in fit.intercepts_],
                     target_scale=yscale, epochs=int(fit.n_iter_))


def _matrix(windows, estimator):
    return np.array([_run(estimator, row['x'], row['climate'], row['tail']) for row in windows])


def _score(windows, prediction, method, split):
    observed = np.array([row['obs'] for row in windows])
    metrics = []
    for h in HORIZONS:
        if h > observed.shape[1]:
            continue
        y, p = observed[:, :h].sum(axis=1), prediction[:, :h].sum(axis=1)
        keep = np.isfinite(y) & np.isfinite(p)
        err, target = (p-y)[keep], y[keep]
        def rmse(values):
            return float(np.sqrt(np.mean(values**2))) if len(values) else None
        metrics.append(dict(split=split, method=method, horizon=h, n=int(keep.sum()),
                            n_runs=len({v['run'] for flag,v in zip(keep, windows) if flag}),
                            n_wet=int((target>0).sum()), n_dry=int((target==0).sum()),
                            rmse_mm=rmse(err), mae_mm=float(np.abs(err).mean()) if len(err) else None,
                            wet_rmse_mm=rmse(err[target>0]), dry_rmse_mm=rmse(err[target==0]),
                            bias_mm=float(err.mean()) if len(err) else None))
    return metrics


def _calibrate(val, xp):
    ys = np.array([v['obs'] for v in val])
    offsets = {}
    for h in HORIZONS:
        resids = ys[:, :h].sum(axis=1) - xp[:, :h].sum(axis=1)
        keep = np.isfinite(resids)
        sample = resids[keep]
        distinct = {v['run'] for v, ok in zip(val,keep) if ok}
        if len(sample) >= 20 and len(distinct) >= 8:
            lo,hi = np.quantile(sample, [.1,.9])
            offsets[str(h)] = dict(lower_offset_mm=float(min(0.,lo)),upper_offset_mm=float(max(0.,hi)),
                                     n=len(sample),runs=len(distinct))
    return offsets


def provider_benchmark(output, weather_dir):
    """Add provider comparisons on identical observed windows without refitting."""
    import pandas as pd
    output = Path(output)
    pred = pd.read_csv(output/'test_predictions.csv.gz',low_memory=False)
    ecmwf_file = Path(weather_dir)/'ecmwf_history.csv.gz'
    provider = 'Open-Meteo ECMWF issued forecast'
    if ecmwf_file.is_file():
        frame = pd.read_csv(ecmwf_file)
        for key in ('run_utc','available_utc','start_utc'):
            frame[key] = pd.to_datetime(frame[key], utc=True)
        cycles = {r:f.set_index('start_utc') for r,f in frame.groupby('run_utc') if not f.start_utc.duplicated().any()}
        available = sorted((f.available_utc.max(),r) for r,f in cycles.items())
        import bisect
        times = [item[0] for item in available]
        pred['ecmwf_mm'] = np.nan
        pred['ecmwf_run_utc'] = ''
        for _,rows in pred.groupby('issue_utc', sort=False):
            idx = bisect.bisect_right(times, pd.Timestamp(rows.issued_at.iloc[0])) - 1
            if idx < 0:
                continue
            run = available[idx][1]
            if pd.Timestamp(rows.issued_at.iloc[0])-run > pd.Timedelta(hours=36):
                continue
            values = cycles[run].snow_water_mm.reindex(pd.to_datetime(rows.start_utc,utc=True))
            if values.notna().any():
                pred.loc[rows.index,'ecmwf_mm']=values.to_numpy()
                pred.loc[rows.index,'ecmwf_run_utc']=run.isoformat()
        pred.to_csv(output/'test_predictions.csv.gz',index=False,float_format='%.8g')
    groups = list(pred.groupby('issue_utc',sort=False))
    cases = [dict(obs=g.observed_mm.to_numpy(),run=g.run_utc.iloc[0]) for _,g in groups]
    model = np.array([g.model_mm for _,g in groups])
    common = []
    mac = []
    for label,field in [('HRRR native','hrrr_mm'), (provider,'ecmwf_mm'),('Radar persistence','radar_persistence_mm')]:
        if field not in pred:
            continue
        other = np.array([g[field] for _,g in groups])
        a = _score(cases,np.where(np.isfinite(other),model,np.nan),'Our model','test')
        b = _score(cases,other,label,'test')
        for ra, rb in zip(a,b):
            assert ra['n'] == rb['n']
            common.extend([dict(ra,reference=label),dict(rb,reference=label)])
        if label==provider:
            mac.extend(b)
    pd.DataFrame(common).to_csv(output/'matched_metrics.csv',index=False)
    if mac:
        metrics=pd.read_csv(output/'metrics.csv')
        metrics=metrics[metrics.method!=provider]
        pd.concat([metrics,pd.DataFrame(mac)],ignore_index=True).to_csv(output/'metrics.csv',index=False)
    # From 2024 onward the disdrometer regularly lacks one minute at 12 UTC.
    # Compare *only* the same observed hours, with a separate target name.
    coverage_results = []
    obs = np.array([case['obs'] for case in cases])
    for h in (24,48):
        measured = np.isfinite(obs[:, :h])
        min_hours = 23 if h==24 else 46
        enough = measured.sum(axis=1) >= min_hours
        for label, field in [('Calendar climatology','climatology_mm'),('HRRR native','hrrr_mm'),
                             ('HRRR with tail persistence','hrrr_tail_baseline_mm'),(provider,'ecmwf_mm')]:
            if field not in pred:
                continue
            ref = np.array([g[field] for _,g in groups])[:, :h]
            keep = enough & np.all(~measured | np.isfinite(ref), axis=1)
            ok = np.flatnonzero(keep)
            subsets = [dict(obs=np.where(measured[i],obs[i,:h],0.),run=cases[i]['run']) for i in ok]
            if not subsets:
                continue
            for name, forecasts in [('Our model',model[:, :h]),(label,ref)]:
                masked = np.where(measured[keep],forecasts[keep],0.)
                entry = next(rec for rec in _score(subsets,masked,name,'test') if rec['horizon']==h)
                entry.update(reference=label, min_complete_hours=min_hours,
                             mean_observed_hours=float(measured[keep].sum(axis=1).mean()),
                             target='Sum over observed hours in window')
                coverage_results.append(entry)
    pd.DataFrame(coverage_results).to_csv(output/'observed_subtotal_metrics.csv',index=False)
    return common


def train(root=ROOT, weather_path=None, output=None):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    root = Path(root)
    output = Path(output) if output else root/'outputs/mammoth-hourly'
    weather_path = Path(weather_path) if weather_path else root/'data/mammoth/weather/history.csv.gz'
    inputs = [weather_path, root/'data/mammoth/research/cues_disdrometer_hourly.csv.gz',
              root/'data/mammoth/radar/historical_points.csv']
    fingerprints = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    windows, clim = _training_data(root, weather_path)
    splits = {'train': [w for w in windows if 2020 <= w['season'] <= 2021],
              'validation': [w for w in windows if w['season'] == 2022],
              'test': [w for w in windows if w['season'] >= 2023]}
    for name, rows in splits.items():
        if len({w['run'] for w in rows}) < 5:
            raise ValueError(f'Insufficient distinct HRRR initializations in {name} split')
    output.mkdir(parents=True, exist_ok=True)
    ys_key = 'Calendar climatology'
    with threadpool_limits(limits=1):
        candidates = dict(_fit_candidates(splits['train']))
    val = splits['validation']
    metrics, ranks, base = [], {}, None
    for name, candidate in candidates.items():
        pred = _matrix(val, candidate)
        rec = _score(val, pred, name, 'validation')
        metrics.extend(rec)
        if name == ys_key:
            base = {r['horizon']:r['rmse_mm'] for r in rec}
        # Equal weight for the six cumulative horizons, after dividing out scale.
        # Ranking is decided before scoring any held-out case.
        ratios = [r['rmse_mm']/base[r['horizon']] for r in rec
                  if r['rmse_mm'] is not None and base.get(r['horizon'],0)]
        if len(ratios) != len(HORIZONS):
            raise ValueError('No validation case for one or more complete forecast windows')
        ranks[name] = float(np.mean(ratios))
    winner = min(ranks, key=ranks.get)
    model = {'version':VERSION, 'family':winner, 'target':'Jeff Dozier Parsivel2 frozen precipitation water equivalent, mm',
             'uses_radar': winner.startswith('Weather + radar') or candidates[winner]['kind']=='radar_update',
             'estimator':candidates[winner], 'monthly_means_mm':clim.tolist(),
             'input_sha256':fingerprints,
             'feature_order':['native_snow_swe_or_tail','sqrt_snow_swe','total_precip','cold_phase_precip','temperature_c',
                              'five_hour_smooth_snow','monthly_target_mean','annual_sin','annual_cos','hour_sin','hour_cos',
                              'lead_over_48','extrapolated_hours','extrapolated_flag','point_available_decay',
                              'radar_point_decay','radar_local_decay','radar_regional_decay','radar_west_decay','radar_quality_decay'],
             'selection':{'rule':'Lowest average validation RMSE relative to monthly climatology across six accumulation horizons',
                          'scores':ranks}, 'historical_weather':'NOAA HRRR archived forecasts at the CUES grid point',
             'radar':'Matched MRMS at the CUES cell and nearby regions, 5–90 minutes before the window start',
             'training_scope':'November–April issue dates in five winter seasons. Historical issue time is 30 minutes before the next whole hour. HRRR publication assumed three hours after initialization. A radar scan must have been observed at least five minutes before simulated issuance.',
             'proxy_note':'Parsivel2 estimates frozen precipitation using a proprietary conversion. The archive retains large estimates even after removing capped minutes. Values describe the instrument response; calibrating to it does not convert SWE to fresh-snow depth.',
             'training_target_qc':'Complete 60-minute hours only. Exclude four hours touching the instrument cap. Retain other published extremes.',
             'calibration':_calibrate(val,_matrix(val,candidates[winner])),
             'calibration_note':'Central 80% empirical offsets from the model-selection winter. Correlated windows and winter-to-winter shifts limit calibration.',
             'splits':{}, 'validation':{}, 'test':{}}
    for split,rows in splits.items():
        model['splits'][split]={'season_start_years':sorted({r['season'] for r in rows}),
            'issues':len(rows), 'runs':len({r['run'] for r in rows}),
            'runs_by_winter':{str(y):len({r['run'] for r in rows if r['season']==y}) for y in sorted({r['season'] for r in rows})},
            'unique_issue_dates':len({r['issue'].date() for r in rows}),
            'first_issue':min(r['issue'] for r in rows).isoformat(), 'last_issue':max(r['issue'] for r in rows).isoformat(),
            'issues_with_radar':sum(r['radar_utc'] is not None for r in rows),
            'issues_with_point_rate':int(sum(np.isfinite(r['radar']) for r in rows))}
        selected = _matrix(rows,candidates[winner])
        if split == 'test':
            for name,candidate in candidates.items():
                metrics.extend(_score(rows,_matrix(rows,candidate),name,split))
        if split == 'train':
            continue
        raw = np.asarray([r['hrrr'] for r in rows])
        persistence = np.asarray([np.full(48,r['radar']) for r in rows])
        for name, pred in [('HRRR native',raw),('Radar persistence',persistence)]:
            metrics.extend(_score(rows,pred,name,split))
        model[split] = {str(m['horizon']):m for m in _score(rows,selected,winner,split)}
        if split=='test':
            count_rows = []
            for idx,r in enumerate(rows):
                for j in range(48):
                    count_rows.append(dict(issue_utc=r['issue'].isoformat(),
                            start_utc=(r['issue']+pd.Timedelta(hours=j)).isoformat(),issued_at=r['issued_at'].isoformat(),hour=j+1,
                            observed_mm=r['obs'][j],model_mm=selected[idx,j],hrrr_mm=r['hrrr'][j],
                            climatology_mm=r['climate'][j], radar_persistence_mm=r['radar'],
                            hrrr_tail_baseline_mm=r['tail'][j], radar_utc=r['radar_utc'],run_utc=r['run'].isoformat()))
            pd.DataFrame(count_rows).to_csv(output/'test_predictions.csv.gz', index=False, float_format='%.8g')
            observed=np.array([r['obs'] for r in rows])
            for h in HORIZONS:
                y,p=observed[:,:h].sum(axis=1), selected[:,:h].sum(axis=1)
                ok=np.isfinite(y)
                offsets=model['calibration'].get(str(h))
                if offsets and ok.any():
                    lower=np.maximum(0.,p[ok]+offsets['lower_offset_mm'])
                    upper=np.maximum(0.,p[ok]+offsets['upper_offset_mm'])
                    model['test'][str(h)]['interval_coverage']=float(np.mean((y[ok]>=lower)&(y[ok]<=upper)))
    pd.DataFrame(metrics).to_csv(output/'metrics.csv', index=False)
    tmp=output/'model.json.tmp'
    tmp.write_text(json.dumps(model, indent=2, allow_nan=False)+'\n')
    tmp.replace(output/'model.json')
    provider_benchmark(output, root/'data/mammoth/weather')
    print('Selected:',winner)
    print('Validation:',ranks)
    print('Splits:',model['splits'])
    print('Held out:',[(h,v['n'],v['rmse_mm']) for h,v in model['test'].items()])
    return model


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--weather',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--benchmark-only', action='store_true')
    args=parser.parse_args()
    if args.benchmark_only:
        provider_benchmark(args.output or args.root/'outputs/mammoth-hourly', args.root/'data/mammoth/weather')
    else:
        train(args.root,args.weather,args.output)
