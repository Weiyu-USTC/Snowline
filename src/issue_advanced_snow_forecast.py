"""Compare saved advanced-model forecasts using the same frozen observations.

Prepared November 4, 2026; inputs stop November 1. This is an architecture
comparison, not a new weather-data update. Model weights are never modified.
"""
from __future__ import annotations

import hashlib
import json
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import deep_dillon as base
from advanced_snow_models import architecture_metadata, make_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
FAMILIES = ['gru', 'tcn', 'patch_transformer']
LABELS = {'gru': 'GRU benchmark', 'tcn': 'Residual temporal convolutional network',
          'patch_transformer': 'Patch Transformer with cross-attention'}
COLORS = {'gru': '#176b91', 'tcn': '#16856d', 'patch_transformer': '#8051a4'}
PREPARED_DATE = '2026-11-04'
DISPLAY_START, DISPLAY_END = '2026-11-04', '2026-12-31'
PATH_SEED = 20261104
PATH_COUNT = 4096


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model_checkpoint(path):
    """Safely restore locally generated model weights and fitted preprocessing."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model(saved['config'])
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    metadata = saved['scaler']
    scaler = base.Scaler()
    scaler.mean = np.asarray(metadata['mean'])
    scaler.std = np.asarray(metadata['std'])
    scaler.fit_days = metadata['fit_unique_context_days']
    scaler.fit_min, scaler.fit_max = metadata['fit_context_start'], metadata['fit_context_end']
    scaler.roni_mean, scaler.roni_std = metadata['roni_mean'], metadata['roni_std']
    scaler.roni_fit_range = metadata['roni_fit_range']
    return model, scaler, saved


def plot_curves(daily, model_details, output_prefix='advanced_forecast'):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#182d3c', 'axes.labelcolor': '#182d3c',
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    shown = daily.loc[daily.date.between(DISPLAY_START, DISPLAY_END)]
    daily_max = max(shown.q90_inches.max(), shown.predicted_inches.max(), shown.historical_daily_mean_inches.max()) * 1.13
    cumulative_max = max(shown.display_cumulative_q90_inches.max(), shown.display_cumulative_mean_inches.max()) * 1.1
    fig, axes = plt.subplots(3, 2, figsize=(14.2, 10.2), sharex=True)
    for row, family in enumerate(FAMILIES):
        data = shown.loc[shown.model.eq(family)]
        color = COLORS[family]
        dates = data.date
        left, right = axes[row]
        label = LABELS[family]
        parameters = model_details[family]['architecture']['trainable_parameters']
        left.set_title(f'{label}\n{parameters:,} trainable parameters', loc='left', fontsize=10.5)
        right.set_title(f'{label}\nAccumulation from November 4', loc='left', fontsize=10.5)
        left.fill_between(dates, data.q10_inches, data.q90_inches, step='mid', color=color, alpha=.17,
                          label='Model-based q10–q90 prediction range')
        left.step(dates, data.predicted_inches, where='mid', color=color, lw=2, label='Expected daily snow')
        left.step(dates, data.historical_daily_mean_inches, where='mid', color='#727980', alpha=.5,
                  linestyle='--', lw=1.7, label='Historical mean (alpha = 0.5)')
        left.set_ylim(0, daily_max)
        left.set_ylabel('Daily new snow (inches)')
        right.fill_between(dates, data.display_cumulative_q10_inches, data.display_cumulative_q90_inches,
                           step='mid', color=color, alpha=.17, label='Range from sums of correlated paths')
        right.step(dates, data.display_cumulative_mean_inches, where='mid', color=color, lw=2,
                   label='Expected cumulative snow')
        right.step(dates, data.historical_daily_mean_inches.cumsum(), where='mid', color='#727980', alpha=.5,
                   linestyle='--', lw=1.7, label='Cumulative historical mean')
        right.set_ylim(0, cumulative_max)
        right.set_ylabel('Cumulative new snow (inches)')
        for axis in (left, right):
            axis.grid(axis='y', alpha=.17)
            axis.axvline(pd.Timestamp('2026-12-01'), color='#727980', linestyle=':', lw=1)
            axis.margins(x=.007)
            axis.legend(frameon=False, fontsize=7.3, loc='upper left')
    for axis in axes[-1]:
        axis.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[4, 15, 25]))
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.suptitle('Dillon snowfall · comparison of trained neural architectures', x=.065, y=.985,
                 ha='left', fontsize=18, fontweight='bold')
    fig.text(.065, .945, 'November 4–December 31, 2026 · identical observations through November 1 · common axes for fair comparison', fontsize=11)
    fig.text(.065, .043, 'Prepared November 4 using frozen inputs: November 2–3 are unobserved bridge predictions; November 4 itself is also a prediction.\n'
             'Bands are pointwise model-based q10–q90 ranges, not calibrated guarantees. No future observed weather or atmospheric forecasts enter these models.\n'
             'The GRU here is the benchmark refit, with newly selected settings and weights; it is not the earlier GRU forecast checkpoint.',
             fontsize=8.5, color='#53616c', linespacing=1.5)
    fig.subplots_adjust(left=.07, right=.985, top=.89, bottom=.135, hspace=.5, wspace=.19)
    for suffix in ['png', 'svg']:
        fig.savefig(OUT / f'{output_prefix}_curves.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default='outputs/advanced_models')
    parser.add_argument('--output-prefix', default='advanced_forecast')
    parser.add_argument('--benchmark-results', default=None,
                        help='Optional explicit path to matching benchmark results JSON')
    args = parser.parse_args()
    if Path(args.output_prefix).name != args.output_prefix:
        raise ValueError('output-prefix is a filename prefix, not a path')
    prefix = args.output_prefix
    model_directory = ROOT / args.model_dir
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    input_path = OUT / 'dillon_deep_daily_input.csv'
    default_results = 'shuffled_snow_results.json' if model_directory.name == 'shuffled_models' else 'advanced_snow_results.json'
    results_path = ROOT / args.benchmark_results if args.benchmark_results else OUT / default_results
    benchmark = json.loads(results_path.read_text())
    source_hashes = benchmark.get('source_and_input_sha256', {})
    input_hash = benchmark.get('input_sha256', source_hashes.get('outputs/dillon_deep_daily_input.csv'))
    roni_hash = benchmark.get('roni_sha256', source_hashes.get('data/roni_noaa.csv'))
    if sha256(input_path) != input_hash or sha256(base.RONI_PATH) != roni_hash:
        raise ValueError('Forecast comparison must use the benchmark input snapshots unchanged')
    frame = pd.read_csv(input_path, parse_dates=['date']).set_index('date').sort_index()
    cutoff = frame.index.max()
    if cutoff != pd.Timestamp('2026-11-01'):
        raise ValueError('This comparison is explicitly frozen to November 1 observations')
    origin = cutoff + pd.Timedelta(days=1)
    all_dates = pd.date_range(origin, periods=base.HORIZON)
    keep = all_dates <= pd.Timestamp(DISPLAY_END)
    dates = all_dates[keep]
    display = dates >= pd.Timestamp(DISPLAY_START)
    all_daily, all_monthly, all_paths, details = [], [], [], {}
    for family in FAMILIES:
        checkpoint = model_directory / f'{family}_all_completed_winters.pt'
        model, scaler, saved = load_model_checkpoint(checkpoint)
        fit_years = saved['fit_season_starts']
        if fit_years != list(range(2005, 2026)):
            raise ValueError(f'{family} is not the full-completed-history forecast model')
        params = base.predict(model, base.arrays(frame, [(2026, origin)], scaler))[0][keep]
        means = base.distribution_mean(params)
        paths, templates = base.rank_template_paths(frame, fit_years, origin, params, PATH_SEED, PATH_COUNT)
        all_paths.append(paths)
        historical = base.climatology_parameters(frame, fit_years, dates)
        cumulative = paths.cumsum(axis=1)
        cumlo, cumhi = np.quantile(cumulative, [.1, .9], axis=0)
        shown_cumulative = paths[:, display].cumsum(axis=1)
        shown_lo, shown_hi = np.quantile(shown_cumulative, [.1, .9], axis=0)
        data = pd.DataFrame({'model': family, 'date': dates, 'lead_day': np.arange(1, len(dates)+1),
                             'predicted_inches': means, 'p_snow': params[:, 0],
                             'positive_log_mu': params[:, 1], 'positive_log_sigma': params[:, 2],
                             'q10_inches': base.distribution_quantile(params, .1),
                             'q90_inches': base.distribution_quantile(params, .9),
                             'q05_inches': base.distribution_quantile(params, .05),
                             'q95_inches': base.distribution_quantile(params, .95),
                             'historical_daily_mean_inches': historical[0],
                             'historical_q10_inches': historical[2], 'historical_q90_inches': historical[3],
                             'cumulative_predicted_inches': means.cumsum(),
                             'cumulative_q10_inches': cumlo, 'cumulative_q90_inches': cumhi,
                             'forecast_status': ['unobserved_bridge' if d < pd.Timestamp(DISPLAY_START) else
                                                 ('current_day_forecast' if d == pd.Timestamp(PREPARED_DATE) else 'future_forecast') for d in dates]})
        for column in ['display_cumulative_mean_inches', 'display_cumulative_q10_inches', 'display_cumulative_q90_inches']:
            data[column] = np.nan
        data.loc[display, 'display_cumulative_mean_inches'] = means[display].cumsum()
        data.loc[display, 'display_cumulative_q10_inches'] = shown_lo
        data.loc[display, 'display_cumulative_q90_inches'] = shown_hi
        all_daily.append(data)
        family_months = []
        for month in dates.to_period('M').unique():
            month_mask = dates.to_period('M') == month
            observed = frame.loc[frame.index.to_period('M') == month, 'snowfall_inches']
            if observed.isna().any():
                raise ValueError('Cannot silently replace a missing observed prefix day with zero')
            observed_sum = float(observed.sum())
            totals = paths[:, month_mask].sum(axis=1) + observed_sum
            record = {'model': family, 'month': str(month), 'observed_prefix_days': len(observed),
                      'observed_prefix_inches': observed_sum,
                      'predicted_remaining_inches': float(means[month_mask].sum()),
                      'predicted_full_month_inches': float(means[month_mask].sum() + observed_sum),
                      'q10_full_month_inches': float(np.quantile(totals, .1)),
                      'q90_full_month_inches': float(np.quantile(totals, .9)),
                      'q05_full_month_inches': float(np.quantile(totals, .05)),
                      'q95_full_month_inches': float(np.quantile(totals, .95))}
            family_months.append(record)
            all_monthly.append(record)
        details[family] = {'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': sha256(checkpoint),
                           'config': saved['config'], 'epochs': saved['epochs'], 'training_seed': saved['seed'],
                           'fit_season_starts': fit_years, 'architecture': architecture_metadata(saved['config'], model),
                           'template_blocks': templates, 'forecast_months': family_months,
                           'display_mean_total_inches': float(means[display].sum()),
                           'display_total_q10_q90_inches': [float(shown_lo[-1]), float(shown_hi[-1])]}
    daily = pd.concat(all_daily, ignore_index=True)
    monthly = pd.DataFrame(all_monthly)
    daily.to_csv(OUT / f'{prefix}_daily.csv', index=False, float_format='%.10f')
    monthly.to_csv(OUT / f'{prefix}_monthly.csv', index=False, float_format='%.10f')
    np.savez_compressed(OUT / f'{prefix}_paths.npz', models=np.array(FAMILIES, dtype='U20'),
                        dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'), paths_inches=np.stack(all_paths))
    manifest = {'prepared_client_date': PREPARED_DATE, 'timezone': 'America/New_York',
                'purpose': 'Matched architecture forecast comparison using frozen observations; not an updated weather-data forecast',
                'observation_cutoff': str(cutoff.date()), 'forecast_origin': str(origin.date()),
                'display_start': DISPLAY_START, 'display_end': DISPLAY_END,
                'current_day_is_forecast': True, 'unobserved_bridge_dates': ['2026-11-02', '2026-11-03'],
                'input_path': str(input_path.relative_to(ROOT)), 'input_sha256': sha256(input_path),
                'roni_sha256': sha256(base.RONI_PATH), 'benchmark_results_sha256': sha256(results_path),
                'benchmark_results_path': str(results_path.relative_to(ROOT)),
                'model_directory': str(model_directory.relative_to(ROOT)),
                'model_selection_provenance': benchmark.get('plan', benchmark.get('protocol', {})),
                'current_lagged_roni': base.lagged_roni(origin), 'source_module_sha256': sha256(Path(__file__)),
                'cpu_threads': 1, 'path_seed': PATH_SEED, 'path_count_per_model': PATH_COUNT,
                'models': details,
                'uncertainty': 'Marginal zero-inflated lognormal q10–q90 and q05–q95 model-based ranges. Monthly/cumulative intervals are quantiles of sums of historical-rank-coupled daily paths. Same historical blocks and random seed for all models. Never sums of daily interval endpoints.',
                'limits': ['Ranges have not been calibrated to guarantee coverage; no simultaneous-path coverage claim.',
                           'Historical rank dependence is assumed to persist; parameter uncertainty is not fully represented.',
                           'No future observed weather or numerical weather forecasts are used.',
                           'Zero-mass distributions can yield closed q10–q90 coverage above80% even when correctly specified.',
                           'This GRU was selected/refit in the advanced benchmark and differs from the earlier small GRU.',
                           'No claim that more complicated models are more accurate follows from different forecast curves.']}
    (OUT / f'{prefix}_manifest.json').write_text(json.dumps(manifest, indent=2))
    plot_curves(daily, details, prefix)
    print(json.dumps({'prepared_date': PREPARED_DATE, 'observation_cutoff': str(cutoff.date()),
                      'forecast_months': all_monthly, 'figure': f'outputs/{prefix}_curves.png'}, indent=2))


if __name__ == '__main__':
    main()
