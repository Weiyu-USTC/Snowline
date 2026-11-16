"""Frozen shuffled WHOLE-WINTER nested cross-validation of snowfall networks.

Exchangeable-winter estimation differs from the earlier chronological benchmark:
later winters may train a model evaluated on an earlier winter. Individual days
never cross the winter-group boundary between fitting, selection and evaluation.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch

import deep_dillon as base
import benchmark_advanced_snow as advanced

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
SEED = 20261104
YEARS = list(range(2005, 2026))
CONFIGS = copy.deepcopy(advanced.CONFIGS)
FAMILIES = list(advanced.FAMILIES)


def make_folds(years=YEARS, seed=SEED):
    """Only season identifiers and a prespecified seed determine assignments."""
    values = np.asarray(sorted(years), dtype=int)
    folds = []
    for fold_id, (fit_index, test_index) in enumerate(KFold(5, shuffle=True, random_state=seed).split(values)):
        fit, test = values[fit_index], values[test_index]
        inner = []
        for inner_id, (train_index, validation_index) in enumerate(KFold(2, shuffle=True, random_state=seed+101+fold_id).split(fit)):
            inner.append({'id': inner_id, 'fit_years': fit[train_index].tolist(),
                          'validation_years': fit[validation_index].tolist()})
        folds.append({'id': fold_id, 'fit_years': fit.tolist(), 'test_years': test.tolist(), 'inner_folds': inner})
    return folds


FOLDS = make_folds()
PLAN = {
    'created_client_date': '2026-11-04', 'seed': SEED,
    'input': 'outputs/dillon_deep_daily_input.csv', 'input_cutoff': '2026-11-01',
    'season_start_years': YEARS, 'outer_folds': FOLDS, 'configs': CONFIGS,
    'grouping': 'Entire Aug-Apr winter context-plus-target groups, never randomly shuffled individual days.',
    'assignment': 'Five outer KFold groups, shuffled with a fixed seed independently of weather outcomes and ENSO. Two inner shuffled whole-winter folds within every outer training partition.',
    'selection': 'Within each family, arithmetic mean of the two inner validation NLL values chooses one of two configurations. Refit epoch count is the median of its two best-epoch values, rounded half up. Refit all outer training winters before scoring its test winters.',
    'selected_family': 'Minimum mean inner validation NLL across the family winners, evaluated as a separate selection rule.',
    'ensemble': 'Fixed equal arithmetic mean of three family conditional means; evaluated for point forecasts only, never weighted by outer errors.',
    'training': copy.deepcopy(advanced.PLAN['training']),
    'context_days': 60, 'horizon_days': 61,
    'primary_metrics': ['Daily RMSE/MAE and probabilistic scores', 'Fixed-November-origin November/December total RMSE/MAE'],
    'secondary_metrics': ['All complete month-start totals', 'Lead-bin metrics', 'Post-assignment ENSO/snowfall fold-balance audit'],
    'balance_policy': 'No search for favorable fold seeds and no stratification on held-out snowfall. Inspect JAS RONI phase and complete winter/Nov-Dec totals only after fixed assignment. Only two warm JAS seasons are available, so every one of five folds cannot contain a warm season.',
    'information_contract': 'Same observed-only weather, missingness, calendar and origin-known lagged RONI as the prior architecture benchmark. Future winter ENSO and future observed weather never enter forecast inputs.',
    'interpretation': 'Retrospective exploratory evaluation under an exchangeable-winter assumption; not a chronological operational hindcast or fresh independent confirmation. Later calendar winters may train earlier test winters.',
    'cohort_note': 'Use all eligible origins in 2005-2025; incomplete 2005 early context can remove its November1 origin. Every model is scored on the same eligible targets. Compare models within this cohort, not numerically against the earlier nine-winter benchmark.',
    'production_fit': 'After outer scoring is frozen, use two shuffled whole-winter inner folds across all21 winters with random_state=seed+9999. Each family selects a configuration by mean validation NLL and a median rounded-half-up epoch count, then refits all21 winters. Outer score rankings never select production settings.',
}


def choose_trials(trials):
    """The selector has no access to outer test outcomes."""
    choices = {}
    for family in FAMILIES:
        choices[family] = min([t for t in trials if t['config']['family'] == family],
                              key=lambda t: t['mean_validation_nll'])
    selected = min(choices, key=lambda family: choices[family]['mean_validation_nll'])
    return choices, selected


def source_hashes():
    paths = [Path(__file__), ROOT/'src/benchmark_advanced_snow.py', ROOT/'src/advanced_snow_models.py',
             ROOT/'src/deep_dillon.py', ROOT/PLAN['input'], base.RONI_PATH]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def balance_audit(frame, folds=FOLDS):
    assignment = {year: fold['id'] for fold in folds for year in fold['test_years']}
    rows = []
    for year in YEARS:
        roni = float(base.RONI.loc[year, 'JAS'])
        winter = frame.reindex(pd.date_range(f'{year}-10-01', f'{year+1}-04-30')).snowfall_inches
        novdec = frame.reindex(pd.date_range(f'{year}-11-01', f'{year}-12-31')).snowfall_inches
        rows.append({'season_start': year, 'outer_fold': assignment[year], 'jas_roni': roni,
                     'jas_enso_phase': 'warm' if roni >= .5 else ('cold' if roni <= -.5 else 'neutral'),
                     'strong_warm_jas': roni >= 1.5, 'strong_cold_jas': roni <= -1.5,
                     'oct_apr_snow_usable_days': int(winter.notna().sum()), 'oct_apr_days': len(winter),
                     'oct_apr_total_inches': float(winter.sum()) if winter.notna().all() else None,
                     'nov_dec_total_inches': float(novdec.sum()) if novdec.notna().all() else None})
    seasons = pd.DataFrame(rows)
    summaries = []
    for fold, part in seasons.groupby('outer_fold'):
        row = {'outer_fold': int(fold), 'seasons': len(part), 'season_start_years': part.season_start.tolist(),
               'warm_jas_seasons': int(part.jas_enso_phase.eq('warm').sum()),
               'neutral_jas_seasons': int(part.jas_enso_phase.eq('neutral').sum()),
               'cold_jas_seasons': int(part.jas_enso_phase.eq('cold').sum()),
               'strong_warm_jas_seasons': int(part.strong_warm_jas.sum()),
               'strong_cold_jas_seasons': int(part.strong_cold_jas.sum())}
        for column in ['oct_apr_total_inches', 'nov_dec_total_inches']:
            values = part[column].dropna()
            row.update({column+'_n_complete': len(values), column+'_mean': float(values.mean()) if len(values) else None,
                        column+'_min': float(values.min()) if len(values) else None,
                        column+'_max': float(values.max()) if len(values) else None,
                        column+'_std': float(values.std(ddof=1)) if len(values)>1 else None})
        summaries.append(row)
    return seasons, summaries


def main():
    started = time.time()
    plan_path = OUT/'shuffled_snow_frozen_plan.json'
    encoded = json.dumps(PLAN, indent=2)+'\n'
    if plan_path.exists() and plan_path.read_text() != encoded:
        raise ValueError('Frozen shuffled plan changed. Explicitly version a new experiment instead.')
    plan_path.write_text(encoded)
    initial_hashes = source_hashes()
    frame = pd.read_csv(ROOT/PLAN['input'], parse_dates=['date']).set_index('date')
    assignment, balance = balance_audit(frame)
    assignment.to_csv(OUT/'shuffled_snow_fold_assignment.csv', index=False, float_format='%.8f')
    pd.DataFrame(balance).to_csv(OUT/'shuffled_snow_fold_balance.csv', index=False, float_format='%.8f')
    (OUT/'shuffled_snow_fold_balance.json').write_text(json.dumps(balance, indent=2))
    model_dir = OUT/'shuffled_models'
    model_dir.mkdir(exist_ok=True)
    all_daily, all_monthly, details = [], [], []
    for fold in FOLDS:
        trials = []
        for config in CONFIGS:
            inner_trials = []
            for inner in fold['inner_folds']:
                fit_origins = base.make_origins(frame, inner['fit_years'], 3)
                valid_origins = base.make_origins(frame, inner['validation_years'], 7)
                _, scaler, trial = advanced.train(frame, fit_origins, config, valid_origins,
                                                   seed=SEED+1000+fold['id']*10+inner['id'])
                inner_trials.append({**inner, 'fit_origin_count': len(fit_origins),
                                     'validation_origin_count': len(valid_origins),
                                     'scaler': scaler.as_dict(), 'trial': trial})
            mean_nll = float(np.mean([t['trial']['validation_nll'] for t in inner_trials]))
            epochs = int(math.floor(float(np.median([t['trial']['best_epoch'] for t in inner_trials]))+.5))
            trials.append({'config': config, 'mean_validation_nll': mean_nll, 'refit_epochs': epochs,
                           'inner_trials': inner_trials})
            (OUT/f'shuffled_snow_fold{fold["id"]}_inner_progress.json').write_text(json.dumps(trials, indent=2))
            print(json.dumps({'stage': 'inner', 'fold': fold['id'], 'family': config['family'],
                              'mean_validation_nll': mean_nll, 'refit_epochs': epochs,
                              'elapsed_seconds': round(time.time()-started, 1)}), flush=True)
        choices, selected_family = choose_trials(trials)
        outer_origins = base.make_origins(frame, fold['fit_years'], 3)
        test_origins = base.make_origins(frame, fold['test_years'], 7)
        daily_parts, monthly_parts, refits = [], [], {}
        for family in FAMILIES:
            choice = choices[family]
            fit_seed = SEED+2000+fold['id']
            model, scaler, fit = advanced.train(frame, outer_origins, choice['config'],
                                                 epochs=choice['refit_epochs'], seed=fit_seed)
            checkpoint = model_dir/f'{family}_fold{fold["id"]}.pt'
            torch.save({'state_dict': model.state_dict(), 'scaler': scaler.as_dict(), 'config': choice['config'],
                        'fit_season_starts': fold['fit_years'], 'epochs': choice['refit_epochs'],
                        'context_days': base.CONTEXT, 'forecast_horizon_days': base.HORIZON,
                        'seed': fit_seed}, checkpoint)
            daily, monthly = base.evaluate(frame, model, scaler, test_origins, fold['fit_years'])
            daily['model'] = daily.model.replace({'multivariate_gru': family})
            monthly['model'] = monthly.model.replace({'multivariate_gru': family})
            if family != FAMILIES[0]:
                daily, monthly = daily.loc[daily.model.eq(family)], monthly.loc[monthly.model.eq(family)]
            daily_parts.append(daily)
            monthly_parts.append(monthly)
            fit.update(scaler=scaler.as_dict(), checkpoint=str(checkpoint.relative_to(ROOT)))
            refits[family] = fit
        daily = advanced.append_derived(pd.concat(daily_parts), selected_family,
                                        ['season_start', 'origin', 'date', 'lead_day'])
        monthly = advanced.append_derived(pd.concat(monthly_parts), selected_family,
                                          ['season_start', 'origin', 'month', 'fixed_nov1'])
        daily['outer_fold'] = monthly['outer_fold'] = fold['id']
        all_daily.append(daily)
        all_monthly.append(monthly)
        detail = {**fold, 'trials': trials, 'choices': choices, 'selected_family': selected_family,
                  'fit_origin_count': len(outer_origins), 'test_origin_count': len(test_origins), 'refits': refits}
        details.append(detail)
        (OUT/f'shuffled_snow_fold{fold["id"]}.json').write_text(json.dumps(detail, indent=2))
        print(json.dumps({'stage': 'outer_complete', 'fold': fold['id'], 'selected_family': selected_family,
                          'elapsed_seconds': round(time.time()-started, 1)}), flush=True)
    daily, monthly = pd.concat(all_daily, ignore_index=True), pd.concat(all_monthly, ignore_index=True)
    daily.to_csv(OUT/'shuffled_snow_daily_predictions.csv', index=False, float_format='%.8f')
    monthly.to_csv(OUT/'shuffled_snow_monthly_predictions.csv', index=False, float_format='%.8f')
    fixed = monthly.loc[monthly.fixed_nov1.eq(True)]
    metrics = {'daily': {name: advanced.point_metrics(p) for name, p in daily.groupby('model')},
               'fixed_nov1_monthly': {name: advanced.point_metrics(p) for name, p in fixed.groupby('model')},
               'all_monthstart_monthly': {name: advanced.point_metrics(p) for name, p in monthly.groupby('model')}}
    probabilistic = {name: base.score_rows(p) for name, p in daily.groupby('model') if name != 'equal_family_mean'}
    lead_metrics = {name: {f'{lo}-{hi}': advanced.point_metrics(p.loc[p.lead_day.between(lo, hi)])
                          for lo, hi in [(1, 7), (8, 14), (15, 30), (31, 61)]}
                    for name, p in daily.groupby('model')}
    bootstraps = {family: {
        'daily': advanced.bootstrap_difference(daily, family, 'daily_calendar_climatology', ['season_start', 'origin', 'date']),
        'fixed_nov1_monthly': advanced.bootstrap_difference(fixed, family, 'direct_monthly_climatology', ['season_start', 'origin', 'month'])
    } for family in FAMILIES+['validation_selected_network', 'equal_family_mean']}
    unchanged = source_hashes() == initial_hashes
    if not unchanged:
        raise RuntimeError('Training source or inputs changed during the frozen run.')
    result = {'plan': PLAN, 'plan_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
              'source_and_input_sha256': initial_hashes, 'sources_verified_unchanged_after_run': unchanged,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'torch_version': torch.__version__,
              'folds': details, 'fold_balance': balance, 'metrics': metrics,
              'daily_probabilistic_metrics': probabilistic, 'lead_group_metrics': lead_metrics,
              'paired_winter_bootstrap': bootstraps,
              'coverage': {'daily_unique_test_winters': int(daily.season_start.nunique()),
                           'daily_unique_target_dates': int(daily.date.nunique()),
                           'fixed_nov1_test_winters': sorted(fixed.season_start.unique().tolist()),
                           'fixed_nov1_months_per_model': fixed.groupby('model').size().to_dict(),
                           'all_monthstart_unique_target_months': int(monthly.month.nunique())},
              'benchmark_fit_count': 60+15, 'elapsed_seconds': time.time()-started,
              'limitations': ['Winter exchangeability is assumed for this shuffled estimate; climate drift and historical operational timing are not tested.',
                              'Some evaluated winters were examined in earlier experiments; results are retrospective exploratory evidence.',
                              'Only21 winter groups,20 with eligible November1 contexts; many daily targets overlap within each winter.',
                              'JAS RONI has only two warm seasons and one strong-warm season, preventing balanced warm coverage across five folds.',
                              'No test-outcome stratification, seed search, or outer-score hyperparameter/ensemble weighting.',
                              'One initialization per configuration and inner fold; two configurations per family are a limited search.',
                              'No new current forecast or production selection is made from the outer score ranking by this script.']}
    (OUT/'shuffled_snow_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'outer_evaluation_complete': True, 'metrics': metrics, 'coverage': result['coverage'],
                      'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    # Production selection has its own prespecified folds and never reads the
    # outer performance ranking. This stage permits all completed winters.
    production_trials = []
    years = np.asarray(YEARS)
    production_folds = [{'id': i, 'fit_years': years[fit].tolist(), 'validation_years': years[valid].tolist()}
                        for i, (fit, valid) in enumerate(KFold(2, shuffle=True, random_state=SEED+9999).split(years))]
    for config in CONFIGS:
        inner_trials = []
        for inner in production_folds:
            _, scaler, trial = advanced.train(frame, base.make_origins(frame, inner['fit_years'], 3), config,
                                               base.make_origins(frame, inner['validation_years'], 7),
                                               seed=SEED+3000+inner['id'])
            inner_trials.append({**inner, 'scaler': scaler.as_dict(), 'trial': trial})
        production_trials.append({'config': config,
                                  'mean_validation_nll': float(np.mean([t['trial']['validation_nll'] for t in inner_trials])),
                                  'refit_epochs': int(math.floor(float(np.median([t['trial']['best_epoch'] for t in inner_trials]))+.5)),
                                  'inner_trials': inner_trials})
        print(json.dumps({'stage': 'production_selection', 'family': config['family'],
                          'elapsed_seconds': round(time.time()-started, 1)}), flush=True)
    production_choices, production_selected_family = choose_trials(production_trials)
    production_models = {}
    for family in FAMILIES:
        choice = production_choices[family]
        fit_seed = SEED+4000
        model, scaler, fit = advanced.train(frame, base.make_origins(frame, YEARS, 3), choice['config'],
                                             epochs=choice['refit_epochs'], seed=fit_seed)
        checkpoint = model_dir/f'{family}_all_completed_winters.pt'
        torch.save({'state_dict': model.state_dict(), 'scaler': scaler.as_dict(), 'config': choice['config'],
                    'fit_season_starts': YEARS, 'epochs': choice['refit_epochs'], 'context_days': base.CONTEXT,
                    'forecast_horizon_days': base.HORIZON, 'seed': fit_seed}, checkpoint)
        production_models[family] = {'choice': choice, 'refit': fit, 'scaler': scaler.as_dict(),
                                     'checkpoint': str(checkpoint.relative_to(ROOT))}
    if source_hashes() != initial_hashes:
        raise RuntimeError('Training source or inputs changed during production fitting.')
    result.update(production_folds=production_folds, production_trials=production_trials,
                  production_models=production_models, production_selected_family=production_selected_family,
                  production_fit_count=15, total_fit_count=90, elapsed_seconds=time.time()-started)
    (OUT/'shuffled_snow_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'complete': True, 'production_selected_family': production_selected_family,
                      'total_fit_count': 90, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
