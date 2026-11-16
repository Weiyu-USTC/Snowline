"""Frozen, retrospective rolling-winter architecture benchmark for Dillon.

Architecture settings and stopping are selected inside each outer training
partition. Previously examined winters are not described as fresh confirmation.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import deep_dillon as base
from advanced_snow_models import make_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
SEED = 20261104
CONFIGS = [
    {'family': 'gru', 'hidden': 32, 'dropout': .25},
    {'family': 'gru', 'hidden': 64, 'dropout': .35},
    {'family': 'tcn', 'width': 32, 'blocks': 5, 'dropout': .2},
    {'family': 'tcn', 'width': 48, 'blocks': 5, 'dropout': .35},
    {'family': 'patch_transformer', 'd_model': 48, 'layers': 3, 'nhead': 4, 'patch_days': 5, 'ff_multiplier': 3, 'dropout': .2},
    {'family': 'patch_transformer', 'd_model': 64, 'layers': 3, 'nhead': 4, 'patch_days': 5, 'ff_multiplier': 3, 'dropout': .35},
]
FAMILIES = ['gru', 'tcn', 'patch_transformer']
FOLDS = [{'id': i, 'fit_years': list(range(2005, start)), 'test_years': list(range(start, start+3))}
         for i, start in enumerate([2017, 2020, 2023])]
PLAN = {
    'created_client_date': '2026-11-04', 'purpose': 'Architecture comparison, not a fresh confirmatory test',
    'input': 'outputs/dillon_deep_daily_input.csv', 'input_cutoff': '2026-11-01',
    'context_days': 60, 'horizon_days': 61, 'configs': CONFIGS, 'outer_folds': FOLDS,
    'inner_validation': 'Last three fit winters in each fold; fit all earlier winters. Scalers fit on inner training only.',
    'selection': 'Within each family choose lowest inner-validation zero-inflated-lognormal NLL and its best epoch. Refit all outer-fit winters for that fixed epoch count. Also score the validation-selected family and fixed equal-family mean ensemble.',
    'training': {'max_epochs': 50, 'minimum_training_epochs': 12, 'early_stopping_patience': 8,
                 'learning_rate': .001, 'weight_decay': .01, 'batch_size': 64, 'gradient_norm_limit': 1.,
                 'loss': 'Same masked zero-inflated-lognormal negative log likelihood for all architectures',
                 'training_origin_stride': 3, 'validation_test_origin_stride': 7, 'cpu_threads': 2},
    'primary_metrics': ['daily RMSE, MAE, NLL/Brier/interval score', 'fixed-Nov1 November/December total RMSE and MAE'],
    'secondary_metrics': ['all complete month-start totals', 'lead-group daily scores'],
    'data_policy': 'Same five observed station variables, missingness, calendar, and conservatively lagged RONI. Never future observed weather. Original publication timestamps/revised ENSO vintage unavailable.',
    'adaptive_reuse_disclosure': 'Several winters were inspected in earlier experiments. This expanded chronological comparison is retrospective and exploratory even though the new fitting code keeps each outer block out of its own selection.',
    'ensemble': 'Arithmetic mean of the three fitted-family conditional means, fixed a priori; no outer-score weighting.',
    'final_fit': 'After evaluation, independently select settings on 2023–25 using 2005–22, then fit all2005–25 for each family. No new current forecast is issued by this script.',
}


def split_inner(fold):
    return fold['fit_years'][:-3], fold['fit_years'][-3:]


def train(frame, fit_origins, config, valid_origins=None, epochs=50, seed=SEED):
    base.seed_everything(seed)
    scaler = base.Scaler().fit(frame, fit_origins)
    data = base.arrays(frame, fit_origins, scaler)
    valid = base.arrays(frame, valid_origins, scaler) if valid_origins else None
    model = make_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    batches = DataLoader(TensorDataset(*data), batch_size=64, shuffle=True,
                         generator=torch.Generator().manual_seed(seed))
    best, best_epoch, state, history = float('inf'), 0, None, []
    for epoch in range(1, epochs+1):
        model.train()
        total, count = 0., 0
        for x, future, y, mask in batches:
            optimizer.zero_grad()
            loss = base.logloss(model(x, future), y, mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss: {config}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * float(mask.sum())
            count += float(mask.sum())
        row = {'epoch': epoch, 'train_nll': total/count}
        if valid is not None:
            model.eval()
            with torch.no_grad():
                params = model(valid[0], valid[1])
                score = float(base.logloss(params, valid[2], valid[3]))
                mean = torch.sigmoid(params[0])*torch.exp(params[1]+.5*params[2]**2)
                rmse = float(torch.sqrt((((mean-valid[2])**2)*valid[3]).sum()/valid[3].sum()))
            row.update(validation_nll=score, validation_rmse_inches=rmse)
            if score < best-1e-5:
                best, best_epoch = score, epoch
                state = copy.deepcopy(model.state_dict())
        else:
            best_epoch = epoch
        history.append(row)
        if valid is not None and epoch >= 12 and epoch-best_epoch >= 8:
            break
    if state is not None:
        model.load_state_dict(state)
    return model, scaler, {'config': config, 'best_epoch': best_epoch, 'validation_nll': best if valid is not None else None,
                           'parameter_count': sum(p.numel() for p in model.parameters()), 'history': history}


def point_metrics(part):
    part = part.dropna(subset=['actual_inches', 'predicted_inches'])
    error = part.predicted_inches - part.actual_inches
    return {'n': len(part), 'rmse_inches': float(np.sqrt(np.mean(error**2))),
            'mae_inches': float(np.abs(error).mean()), 'mse_square_inches': float(np.mean(error**2)),
            'bias_inches': float(error.mean())}


def bootstrap_difference(rows, family, baseline, keys, repeats=10000):
    pair = rows.loc[rows.model.isin([family, baseline])].pivot(index=keys, columns='model', values=['actual_inches', 'predicted_inches']).dropna()
    y = pair.actual_inches[family]
    diff = (pair.predicted_inches[family]-y)**2 - (pair.predicted_inches[baseline]-y)**2
    grouped = diff.groupby(level='season_start').agg(['sum', 'count'])
    choices = np.random.default_rng(SEED).integers(0, len(grouped), (repeats, len(grouped)))
    estimates = grouped['sum'].to_numpy()[choices].sum(1)/grouped['count'].to_numpy()[choices].sum(1)
    return {'model_minus_baseline_mse_square_inches': float(diff.mean()),
            'whole_winter_bootstrap95': np.quantile(estimates, [.025, .975]).tolist(),
            'winter_clusters': len(grouped), 'paired_predictions': len(pair),
            'interpretation': 'Negative favors model; descriptive retrospective interval, not independent confirmation.'}


def append_derived(rows, selected_family, keys):
    chosen = rows.loc[rows.model.eq(selected_family)].copy()
    chosen['model'] = 'validation_selected_network'
    point_columns = keys + ['actual_inches', 'predicted_inches']
    ensembled = rows.loc[rows.model.isin(FAMILIES), point_columns].groupby(keys, as_index=False).agg(
        actual_inches=('actual_inches', 'first'), predicted_inches=('predicted_inches', 'mean'))
    ensembled['model'] = 'equal_family_mean'
    return pd.concat([rows, chosen, ensembled], ignore_index=True)


def main():
    started = time.time()
    plan_path = OUT/'advanced_snow_frozen_plan.json'
    encoded = json.dumps(PLAN, indent=2)+'\n'
    if plan_path.exists() and plan_path.read_text() != encoded:
        raise ValueError('Frozen plan changed. Archive and explicitly version a new experiment instead.')
    plan_path.write_text(encoded)
    input_path = ROOT/PLAN['input']
    frame = pd.read_csv(input_path, parse_dates=['date']).set_index('date')
    model_dir = OUT/'advanced_models'
    model_dir.mkdir(exist_ok=True)
    all_daily, all_monthly, fold_details = [], [], []
    for fold in FOLDS:
        inner_fit, inner_valid = split_inner(fold)
        fit_origins = base.make_origins(frame, inner_fit, 3)
        valid_origins = base.make_origins(frame, inner_valid, 7)
        trials = []
        for config in CONFIGS:
            _, _, trial = train(frame, fit_origins, config, valid_origins, seed=SEED+fold['id'])
            trials.append(trial)
            print(json.dumps({'stage': 'inner_selection', 'fold': fold['id'], 'family': config['family'],
                              'params': trial['parameter_count'], 'best_epoch': trial['best_epoch'], 'validation_nll': trial['validation_nll']}), flush=True)
        choices = {family: min([t for t in trials if t['config']['family'] == family], key=lambda t: t['validation_nll']) for family in FAMILIES}
        selected_family = min(choices, key=lambda f: choices[f]['validation_nll'])
        outer_origins = base.make_origins(frame, fold['fit_years'], 3)
        test_origins = base.make_origins(frame, fold['test_years'], 7)
        daily_parts, month_parts, fitted = [], [], {}
        for family in FAMILIES:
            choice = choices[family]
            model, scaler, fit = train(frame, outer_origins, choice['config'], epochs=choice['best_epoch'], seed=SEED+fold['id'])
            checkpoint = model_dir/f'{family}_fold{fold["id"]}.pt'
            base.save_checkpoint(checkpoint, model, scaler, choice['config'], fold['fit_years'], choice['best_epoch'])
            # The base saver has an older default seed; correct this manifest field.
            saved = torch.load(checkpoint, weights_only=True)
            saved['seed'] = SEED+fold['id']
            torch.save(saved, checkpoint)
            daily, monthly = base.evaluate(frame, model, scaler, test_origins, fold['fit_years'])
            daily['model'] = daily.model.replace({'multivariate_gru': family})
            monthly['model'] = monthly.model.replace({'multivariate_gru': family})
            if family != FAMILIES[0]:
                daily = daily.loc[daily.model.eq(family)]
                monthly = monthly.loc[monthly.model.eq(family)]
            daily_parts.append(daily)
            month_parts.append(monthly)
            params = base.predict(model, base.arrays(frame, outer_origins, scaler))
            _, _, y, mask = base.arrays(frame, outer_origins, scaler)
            valid = mask.numpy().astype(bool)
            fit['training_rmse_inches'] = float(np.sqrt(np.mean((base.distribution_mean(params)[valid]-y.numpy()[valid])**2)))
            fit['scaler'] = scaler.as_dict()
            fitted[family] = fit
        daily = append_derived(pd.concat(daily_parts), selected_family, ['season_start', 'origin', 'date', 'lead_day'])
        monthly = append_derived(pd.concat(month_parts), selected_family, ['season_start', 'origin', 'month', 'fixed_nov1'])
        daily['outer_fold'] = monthly['outer_fold'] = fold['id']
        all_daily.append(daily)
        all_monthly.append(monthly)
        detail = {**fold, 'inner_fit_years': inner_fit, 'inner_validation_years': inner_valid,
                  'trials': trials, 'choices': choices, 'selected_family': selected_family, 'refits': fitted}
        fold_details.append(detail)
        (OUT/f'advanced_snow_fold{fold["id"]}.json').write_text(json.dumps(detail, indent=2))
        print(json.dumps({'stage': 'outer_fold_complete', 'fold': fold['id'], 'selected_inside_training': selected_family,
                          'seconds': round(time.time()-started, 1)}), flush=True)
    daily = pd.concat(all_daily, ignore_index=True)
    monthly = pd.concat(all_monthly, ignore_index=True)
    daily.to_csv(OUT/'advanced_snow_daily_predictions.csv', index=False, float_format='%.8f')
    monthly.to_csv(OUT/'advanced_snow_monthly_predictions.csv', index=False, float_format='%.8f')
    fixed = monthly.loc[monthly.fixed_nov1.eq(True)]
    metrics = {'daily': {name: point_metrics(p) for name, p in daily.groupby('model')},
               'fixed_nov1_monthly': {name: point_metrics(p) for name, p in fixed.groupby('model')},
               'all_monthstart_monthly': {name: point_metrics(p) for name, p in monthly.groupby('model')}}
    probabilistic = {name: base.score_rows(p) for name, p in daily.groupby('model') if name != 'equal_family_mean'}
    bootstraps = {family: {
        'daily': bootstrap_difference(daily, family, 'daily_calendar_climatology', ['season_start', 'origin', 'date']),
        'fixed_nov1_monthly': bootstrap_difference(fixed, family, 'direct_monthly_climatology', ['season_start', 'origin', 'month'])
    } for family in FAMILIES+['validation_selected_network', 'equal_family_mean']}
    results = {'plan': PLAN, 'plan_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
               'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
               'roni_sha256': hashlib.sha256(base.RONI_PATH.read_bytes()).hexdigest(),
               'seed': SEED, 'torch_version': torch.__version__, 'folds': fold_details,
               'metrics': metrics, 'daily_probabilistic_metrics': probabilistic, 'paired_winter_bootstrap': bootstraps,
               'elapsed_seconds': time.time()-started,
               'limits': ['Retrospective adaptive research: outer code isolation does not erase prior inspection of these winters.',
                          'Each outer block uses one fixed earlier fitted model; no observations from other test winters enter fitting.',
                          'Only one random initialization per configuration and fold. Not a broad hyperparameter search.',
                          'Each architecture has the same probabilistic head/objective and inputs, but capacity, regularization and optimization differ.',
                          'No numerical-weather forecasts, original release-time vintages, or new resort observations.',
                          'Equal-family ensemble evaluated for point forecasts only; no interval calibration claim.']}
    # Freeze outer scores before production fitting; do not select using these scores.
    (OUT/'advanced_snow_results.json').write_text(json.dumps(results, indent=2))
    current_choices = {}
    for family in FAMILIES:
        trials = []
        for config in [c for c in CONFIGS if c['family'] == family]:
            _, _, trial = train(frame, base.make_origins(frame, list(range(2005, 2023)), 3), config,
                                base.make_origins(frame, [2023, 2024, 2025], 7), seed=SEED+10)
            trials.append(trial)
        choice = min(trials, key=lambda t: t['validation_nll'])
        model, scaler, fit = train(frame, base.make_origins(frame, list(range(2005, 2026)), 3), choice['config'],
                                   epochs=choice['best_epoch'], seed=SEED+10)
        checkpoint = model_dir/f'{family}_all_completed_winters.pt'
        base.save_checkpoint(checkpoint, model, scaler, choice['config'], list(range(2005, 2026)), choice['best_epoch'])
        saved = torch.load(checkpoint, weights_only=True)
        saved['seed'] = SEED+10
        torch.save(saved, checkpoint)
        current_choices[family] = {'selection_trials': trials, 'choice': choice, 'refit': fit, 'checkpoint': str(checkpoint.relative_to(ROOT))}
    results['final_models'] = current_choices
    results['elapsed_seconds'] = time.time()-started
    (OUT/'advanced_snow_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({'complete': True, 'metrics': metrics, 'seconds': results['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
