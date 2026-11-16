"""Optional frozen TabPFN-v2 in-context regression for monthly snowfall.

Use the project-specific .venv-tabpfn interpreter. The checkpoint lives in
.model_cache/tabpfn-v2 and is intentionally not bundled with the small archive.
The pretrained weights are frozen: fit() supplies a labeled context, it does not
gradient-train the network on the snowfall observations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('SKB_DATA_DIRECTORY', str(ROOT / '.model_cache/skrub'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.model_cache/matplotlib'))
MODEL = ROOT / '.model_cache/tabpfn-v2/tabpfn-v2-regressor-v2_default.ckpt'
MODEL_SHA256 = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
_ESTIMATOR = None


def download_model():
    """Fetch the public, explicitly named V2 artifact; reject a changed checkpoint."""
    import urllib.request
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    base='https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/'
    for name,expected in [(MODEL.name,MODEL_SHA256),('LICENSE.txt','08b1851b4b6136b5fb0cb124d3258e0d91e3e1623f1be5904b77e4d270d112d8')]:
        destination=MODEL.parent/name
        temporary=destination.with_suffix(destination.suffix+'.download')
        with urllib.request.urlopen(base+name,timeout=30) as response,temporary.open('wb') as sink:
            count=0
            while block:=response.read(1024*1024):
                count+=len(block)
                if count>50_000_000:
                    raise ValueError('Unexpectedly large model download; refusing replacement')
                sink.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected:
            temporary.unlink()
            raise ValueError(f'{name} changed from the recorded source snapshot')
        temporary.replace(destination)
        print(f'Saved {name}: {destination.stat().st_size:,} bytes')


def model_metadata():
    return {
        'name': 'TabPFN-v2 frozen in-context regressor',
        'package': 'tabpfn', 'package_version': importlib.metadata.version('tabpfn'),
        'checkpoint': MODEL.name, 'checkpoint_bytes': MODEL.stat().st_size,
        'checkpoint_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        'source_url': 'https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor-v2_default.ckpt',
        'license_url': 'https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt',
        'license': 'Prior Labs License 1.1, Apache-derived with enhanced attribution',
        'pretraining': 'Synthetic tabular prediction tasks; no local snowfall gradient training',
        'device': 'cpu', 'n_estimators': 2, 'random_state': 20261103,
        'selection': 'Fixed configuration; no tuning of these hyperparameters on outer test seasons',
    }


def predict_tabpfn(X_train, y_train, X_test):
    """Return predictions from fixed pretrained weights conditioned on training rows.

    Arrays must contain the same issue-time-available features for train and test.
    Feature construction and outer train/test isolation remain the caller's job.
    No cross-fold prediction cache or target-value clipping is performed here.
    """
    global _ESTIMATOR
    import torch
    from tabpfn import TabPFNRegressor
    torch.set_num_threads(1)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError('Training and test features must have matching two-dimensional shapes')
    if len(X_train) != len(y_train) or len(y_train) < 3:
        raise ValueError('At least three matching labeled rows are required')
    if not all(np.isfinite(v).all() for v in [X_train, X_test, y_train]):
        raise ValueError('This experiment requires finite observed features and targets')
    if _ESTIMATOR is None:
        if not MODEL.exists():
            raise FileNotFoundError(f'Download the explicitly versioned public checkpoint to {MODEL}')
        if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('Checkpoint hash differs from the version used for this experiment')
        _ESTIMATOR = TabPFNRegressor(
            model_path=str(MODEL), device='cpu', n_estimators=2,
            random_state=20261103, n_preprocessing_jobs=1,
            inference_precision=torch.float32, enable_torch_compile=False,
        )
    _ESTIMATOR.fit(X_train, y_train)
    return np.asarray(_ESTIMATOR.predict(X_test), dtype=float)


def run_benchmark():
    """Fixed-model LOSO comparator, separate from the tuned model-selection pipeline."""
    import pandas as pd
    import monthly
    frame = monthly.load_training('copper')
    months = monthly.MONTHS
    roni = pd.read_csv(ROOT / 'data/roni_noaa.csv').set_index('year')
    current_enso = float(roni.loc[2026, 'JAS'])
    scenario_values = {f'October scenario {v} inches':v for v in [0,5,10,20,30,40]}
    starts = time.monotonic()
    rows, forecast_modes = [], {}
    for mode, snow_context in [('observed_prefix',True),('october_unverified_enso_only',False)]:
        live = monthly.live_frame(scenario_values if snow_context else {'October unverified; ENSO information only':np.nan},current_enso)
        tasks = ([(0,t) for t in range(1,7)] + [(origin,origin+1) for origin in range(1,6)]) if snow_context else [(0,t) for t in range(1,7)]
        # The first six tasks collectively form one October-issued multimonth forecast.
        current_predictions = np.empty((len(live),6))
        live_omitted = {j: {i: {'prediction':[None]*6,'held_prediction':[None]*6,
                                'held_actual':[None]*6} for i in range(len(frame))} for j in range(len(live))}
        for task_count,(origin,target) in enumerate(tasks,1):
            X,names = monthly.make_features(frame,origin,True,snow_context)
            y = frame[months[target]].to_numpy(float)
            live_X = monthly.make_features(live,origin,True,snow_context)[0] if origin==0 else None
            for held in range(len(frame)):
                keep = np.arange(len(frame)) != held
                queries = np.vstack([X[held:held+1],live_X]) if live_X is not None else X[held:held+1]
                values = np.maximum(0.,predict_tabpfn(X[keep],y[keep],queries))
                common = {'season':str(frame.season.iloc[held]),'start_year':int(frame.start_year.iloc[held]),
                          'mode':mode,'origin':origin,'observed_through':months[origin] if snow_context else 'unknown',
                          'target_month':months[target],'horizon':target-origin,
                          'actual':float(y[held]),'training_seasons':int(keep.sum()),'features':','.join(names)}
                rows.append({**common,'family':'tabpfn_v2','prediction':float(values[0])})
                rows.append({**common,'family':'climatology','prediction':float(y[keep].mean())})
                if origin==0:
                    for j in range(len(live)):
                        record=live_omitted[j][held]
                        record['prediction'][target-1]=float(values[j+1])
                        record['held_prediction'][target-1]=float(values[0])
                        record['held_actual'][target-1]=float(y[held])
            if origin==0:
                current_predictions[:,target-1]=np.maximum(0.,predict_tabpfn(X,y,live_X))
            print(f'TabPFN {mode}: task {task_count}/{len(tasks)}, {months[origin]} → {months[target]}, elapsed {time.monotonic()-starts:.1f}s',flush=True)
        forecasts={}
        for j,scenario in enumerate(live.scenario):
            bounds=monthly.jackknife_bounds(list(live_omitted[j].values()),months[1:])
            bounds['method']='Fixed pretrained TabPFN-v2 omitted-season jackknife+ finite-rank range, alpha=.10 (nominal90%; theoretical lower bound80% only under required exchangeability/symmetry assumptions). Exploratory marginal ranges, not simultaneous or empirically calibrated coverage.'
            forecasts[str(scenario)]={'point_inches':dict(zip(months[1:],current_predictions[j].tolist())),
                'nov_apr_total_inches':float(current_predictions[j].sum()),**bounds}
        forecast_modes[mode]={'forecasts':forecasts,'feature_configuration':'Observed prefix plus lagged RONI and ±1.5 strong indicators' if snow_context else 'Lagged RONI and ±1.5 strong indicators only; October not substituted',
                              'tasks':tasks}
    table=pd.DataFrame(rows)
    metrics=[]
    for mode,part in table.groupby('mode',sort=False):
        scopes={'october_issued_all_six':part.loc[part.origin==0]}
        if mode=='observed_prefix': scopes['sequential_next_month']=part.loc[part.horizon==1]
        for scope,group in scopes.items():
            for family,g in group.groupby('family',sort=False):
                err=g.prediction-g.actual
                metrics.append({'mode':mode,'scope':scope,'family':family,'n_predictions':len(g),
                                'n_seasons':g.season.nunique(),'mae_inches':float(abs(err).mean()),
                                'rmse_inches':float(np.sqrt((err**2).mean()))})
    table.to_csv(ROOT/'outputs/monthly_tabpfn_predictions.csv',index=False)
    output={'as_of':'2026-11-03','forecast_season':'2026/2027','metadata':model_metadata(),
            'evaluation':'Fixed-model leave-one-season-out, with all labeled rows from the held-out season excluded from context. Diagnostic challenger; not a member of the nested family-selection procedure.',
            'year_assumption':'Independent/exchangeable seasons as requested. Not a chronological operational backtest.',
            'enso_feature_current':'JAS2026 RONI','enso_value_current':current_enso,
            'copper_october_status':'Not verified. Values0,5,10,20,30,40 are scenarios only. ENSO-only forecast omits October.',
            'metrics':metrics,'experiments':forecast_modes,
            'runtime_seconds':time.monotonic()-starts,'n_seasons':len(frame),
            'years':frame.start_year.astype(int).tolist(),
            'model_uncertainty':'Pretrained model is not snowfall-validated; intervals require exchangeability and are marginal rather than simultaneous.',
            'source_data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/copper_monthly_user.csv',ROOT/'data/roni_noaa.csv']}}
    (ROOT/'outputs/monthly_tabpfn_results.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--download-model', action='store_true',help='Download the public V2 checkpoint and license, verifying their recorded hashes')
    args = parser.parse_args()
    if args.download_model:
        download_model()
    elif args.benchmark:
        run_benchmark()
    elif args.smoke_test:
        import pandas as pd
        frame = pd.read_csv(ROOT / 'data/copper_monthly_user.csv').iloc[:20]
        start = time.monotonic()
        value = predict_tabpfn(frame[['oct']].iloc[:-1], frame['nov'].iloc[:-1], frame[['oct']].iloc[-1:])
        print(json.dumps({'smoke_prediction_november':value.tolist(), 'elapsed_seconds':time.monotonic()-start,
                          'metadata':model_metadata()},indent=2))
    else:
        parser.error('Use --download-model, --benchmark, --smoke-test or import predict_tabpfn.')


if __name__ == '__main__':
    main()
