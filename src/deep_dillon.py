"""Train and evaluate a multivariate, direct 61-day Dillon snowfall GRU.

No future observed weather is passed to the model. Whole winter groups separate
training, validation and testing. Model-based daily marginals are coupled with
contiguous historical rank templates before computing cumulative/monthly ranges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from scipy.special import ndtri
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['snowfall_inches', 'precipitation_mm', 'snow_depth_cm', 'tmax_celsius', 'tmin_celsius']
CONTEXT, HORIZON, SEED = 60, 61, 20261103
TRAIN_YEARS = list(range(2005, 2017))
VALID_YEARS = list(range(2017, 2021))
TEST_YEARS = list(range(2021, 2026))
RONI_PATH = ROOT/'data/roni_noaa.csv'
RONI = pd.read_csv(RONI_PATH).set_index('year')
RONI_PERIODS = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']


def lagged_roni(origin):
    # The centered three-month period ends two calendar months before the
    # origin month: November uses JAS, December uses ASO, January uses SON.
    # Values are revised historical RONI, not archived release vintages.
    center = origin.to_period('M')-3
    return float(RONI.loc[center.year, RONI_PERIODS[center.month-1]])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def calendar_features(dates):
    phase = 2 * np.pi * (pd.DatetimeIndex(dates).dayofyear.to_numpy() - 1) / 365.2425
    return np.column_stack([np.sin(phase), np.cos(phase), np.sin(2*phase), np.cos(2*phase)]).astype('float32')


def transformed_weather(frame):
    values = frame[FEATURES].to_numpy(dtype=float, copy=True)
    values[:, :3] = np.log1p(values[:, :3])
    return values


def make_origins(frame, years, stride=3):
    """Each context and target lies in its own Aug--Apr winter group."""
    dates = []
    for year in years:
        start, end = pd.Timestamp(f'{year}-10-01'), pd.Timestamp(f'{year+1}-02-28')
        candidates = set(pd.date_range(start, end, freq=f'{stride}D'))
        candidates.update(pd.date_range(start, end, freq='MS'))
        for origin in sorted(candidates):
            if origin-pd.Timedelta(days=CONTEXT) < frame.index.min():
                continue
            if origin+pd.Timedelta(days=HORIZON-1) > frame.index.max():
                continue
            labels = frame.reindex(pd.date_range(origin, periods=HORIZON)).snowfall_inches
            if labels.notna().sum() >= HORIZON-3:
                dates.append((year, origin))
    return dates


class Scaler:
    def fit(self, frame, origins):
        # Fit only on unique historical context rows belonging to this fit split.
        days = sorted(set(day for _, origin in origins for day in pd.date_range(origin-pd.Timedelta(days=CONTEXT), periods=CONTEXT)))
        values = transformed_weather(frame.reindex(days))
        self.mean = np.nanmean(values, axis=0)
        self.std = np.nanstd(values, axis=0)
        self.std = np.where(self.std > 1e-6, self.std, 1.)
        self.fit_days = len(days)
        self.fit_min, self.fit_max = str(days[0].date()), str(days[-1].date())
        roni = np.array([lagged_roni(origin) for _, origin in origins])
        self.roni_mean, self.roni_std = float(np.nanmean(roni)), float(np.nanstd(roni))
        self.roni_std = max(self.roni_std, 1e-6)
        self.roni_fit_range = [float(np.nanmin(roni)), float(np.nanmax(roni))]
        return self

    def transform(self, frame):
        values = transformed_weather(frame)
        missing = ~np.isfinite(values)
        scaled = (values-self.mean)/self.std
        scaled[missing] = 0.
        return np.column_stack([scaled, missing.astype(float), calendar_features(frame.index)]).astype('float32')

    def as_dict(self):
        return {'features': FEATURES, 'nonnegative_transform': 'log1p for snow, precipitation and depth',
                'mean': self.mean.tolist(), 'std': self.std.tolist(), 'fit_unique_context_days': self.fit_days,
                'fit_context_start': self.fit_min, 'fit_context_end': self.fit_max,
                'roni_mean': self.roni_mean, 'roni_std': self.roni_std, 'roni_fit_range': self.roni_fit_range,
                'missing_policy': 'training-context mean in transformed space, with separate missingness indicator'}


def arrays(frame, origins, scaler):
    xx, ff, yy, mm = [], [], [], []
    for _, origin in origins:
        context_dates = pd.date_range(origin-pd.Timedelta(days=CONTEXT), periods=CONTEXT)
        target_dates = pd.date_range(origin, periods=HORIZON)
        xx.append(scaler.transform(frame.reindex(context_dates)))
        lead = np.arange(1, HORIZON+1, dtype='float32')/HORIZON
        roni = lagged_roni(origin)
        roni_features = [(roni-scaler.roni_mean)/scaler.roni_std if np.isfinite(roni) else 0.,
                         float(roni >= 1.5), float(roni <= -1.5), float(not np.isfinite(roni))]
        ff.append(np.column_stack([calendar_features(target_dates), lead, lead**2,
                                   np.tile(roni_features, (HORIZON, 1))]))
        y = frame.reindex(target_dates).snowfall_inches.to_numpy(dtype='float32')
        mm.append(np.isfinite(y).astype('float32'))
        yy.append(np.nan_to_num(y, nan=0.))
    return tuple(torch.tensor(np.asarray(a), dtype=torch.float32) for a in [xx, ff, yy, mm])


class SnowGRU(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.gru = nn.GRU(14, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.decoder = nn.Sequential(nn.Linear(hidden+10, 128), nn.SiLU(), nn.Dropout(dropout),
                                     nn.Linear(128, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, 3))
        # Start near a reasonable snowfall occurrence rate, not 50%.
        with torch.no_grad():
            self.decoder[-1].bias[:] = torch.tensor([-1.2, -.1, .3])

    def forward(self, x, future_calendar):
        _, h = self.gru(x)
        state = h[-1, :, None, :].expand(-1, HORIZON, -1)
        raw = self.decoder(torch.cat([state, future_calendar], dim=-1))
        logits = raw[..., 0]
        mu = 3 * torch.tanh(raw[..., 1]/3)
        sigma = .15 + 1.35 * torch.sigmoid(raw[..., 2])
        return logits, mu, sigma


def logloss(params, y, mask):
    logits, mu, sigma = params
    positive = (y > 0).float()
    log_y = torch.log(torch.clamp(y, min=1e-5))
    occurrence = nn.functional.binary_cross_entropy_with_logits(logits, positive, reduction='none')
    amount = .5*((log_y-mu)/sigma)**2 + torch.log(sigma) + log_y + .5*math.log(2*math.pi)
    return ((occurrence+positive*amount)*mask).sum()/mask.sum()


def predict(model, data):
    model.eval()
    all_params = []
    with torch.no_grad():
        for i in range(0, len(data[0]), 128):
            logits, mu, sigma = model(data[0][i:i+128], data[1][i:i+128])
            all_params.append(np.stack([torch.sigmoid(logits).numpy(), mu.numpy(), sigma.numpy()], -1))
    return np.concatenate(all_params)


def distribution_mean(params):
    return params[..., 0]*np.exp(params[..., 1]+.5*params[..., 2]**2)


def distribution_quantile(params, q):
    p, mu, sigma = params[..., 0], params[..., 1], params[..., 2]
    q = np.asarray(q)
    positive_q = np.clip((q-(1-p))/np.maximum(p, 1e-8), 1e-7, 1-1e-7)
    return np.where(q <= 1-p, 0., np.exp(mu+sigma*ndtri(positive_q)))


def train(frame, train_origins, config, valid_origins=None, epochs=70, seed=SEED):
    seed_everything(seed)
    scaler = Scaler().fit(frame, train_origins)
    data = arrays(frame, train_origins, scaler)
    validation = arrays(frame, valid_origins, scaler) if valid_origins else None
    loader = DataLoader(TensorDataset(*data), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = SnowGRU(config['hidden'], config['dropout'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    logs, best, best_epoch, best_state = [], float('inf'), 0, None
    for epoch in range(1, epochs+1):
        model.train()
        total, count, squared_error, target_count = 0., 0, 0., 0.
        for x, future, y, mask in loader:
            optimizer.zero_grad()
            params = model(x, future)
            loss = logloss(params, y, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(x)
            count += len(x)
            with torch.no_grad():
                mean = torch.sigmoid(params[0])*torch.exp(params[1]+.5*params[2]**2)
                squared_error += float((((mean-y)**2)*mask).sum())
                target_count += float(mask.sum())
        row = {'epoch': epoch, 'train_nll': total/count,
               'training_batch_rmse_inches': float(np.sqrt(squared_error/target_count))}
        if validation:
            model.eval()
            with torch.no_grad():
                params = model(validation[0], validation[1])
                score = float(logloss(params, validation[2], validation[3]))
                mean = torch.sigmoid(params[0])*torch.exp(params[1]+.5*params[2]**2)
                row['validation_rmse_inches'] = float(torch.sqrt((((mean-validation[2])**2)*validation[3]).sum()/validation[3].sum()))
            row['validation_nll'] = score
            if score < best-1e-5:
                best, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            best_epoch = epoch
        logs.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({'config_hidden': config['hidden'], **row}), flush=True)
        if validation and epoch >= 15 and epoch-best_epoch >= 10:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, logs, best_epoch, best


def training_snow(frame, years):
    pieces = [frame.loc[f'{year}-10-01':f'{year+1}-04-30', ['snowfall_inches']] for year in years]
    return pd.concat(pieces)


def climatology_parameters(frame, years, dates):
    snow = training_snow(frame, years).dropna()
    day = snow.index.dayofyear.to_numpy()
    means, probabilities, lows, highs = [], [], [], []
    for date in dates:
        delta = np.abs(day-date.dayofyear)
        values = snow.snowfall_inches.to_numpy()[np.minimum(delta, 366-delta) <= 15]
        means.append(float(values.mean()))
        probabilities.append(float((values > 0).mean()))
        lows.append(float(np.quantile(values, .1)))
        highs.append(float(np.quantile(values, .9)))
    return np.array(means), np.array(probabilities), np.array(lows), np.array(highs)


def rank_template_paths(frame, years, origin, params, seed, n_paths=1024):
    """Schaake-style marginal rank reordering of contiguous historical blocks.

    This assumes historical calendar-conditioned temporal dependence persists.
    Tied-zero ranks are randomized, reproducibly. It is not a calibration claim.
    """
    templates = []
    for year in years:
        source_year = year if origin.month >= 10 else year+1
        anchor = pd.Timestamp(year=source_year, month=origin.month, day=min(origin.day, 28) if origin.month == 2 else origin.day)
        for shift in range(-14, 15):
            dates = pd.date_range(anchor+pd.Timedelta(days=shift), periods=len(params))
            values = frame.reindex(dates).snowfall_inches.to_numpy(float)
            if np.isfinite(values).all():
                templates.append(values)
    if not templates:
        raise ValueError('No complete historical templates')
    rng = np.random.default_rng(seed)
    selected = np.asarray(templates)[rng.integers(0, len(templates), size=n_paths)]
    # Perturbations only resolve ties; physical observations are not altered.
    ranks = np.empty_like(selected)
    for lead in range(selected.shape[1]):
        order = np.lexsort((rng.random(n_paths), selected[:, lead]))
        ranks[order, lead] = np.arange(1, n_paths+1)
    uniforms = (ranks-.5)/n_paths
    paths = distribution_quantile(params[None, :, :], uniforms)
    return paths, len(templates)


def score_rows(rows):
    data = rows.dropna(subset=['actual_inches'])
    error = data.predicted_inches-data.actual_inches
    covered = (data.actual_inches >= data.q10_inches) & (data.actual_inches <= data.q90_inches)
    interval_score = data.q90_inches-data.q10_inches+10*np.maximum(data.q10_inches-data.actual_inches, 0)+10*np.maximum(data.actual_inches-data.q90_inches, 0)
    result = {'origin_lead_pairs': len(data), 'unique_target_days': data.date.nunique(),
            'rmse_inches': float(np.sqrt(np.mean(error**2))), 'mse_square_inches': float(np.mean(error**2)),
            'mae_inches': float(np.mean(np.abs(error))), 'bias_inches': float(error.mean()),
            'snow_occurrence_brier': float(np.mean((data.p_snow-(data.actual_inches > 0).astype(float))**2)),
            'central80_coverage': float(covered.mean()),
            'central80_snow_day_coverage': float(covered.loc[data.actual_inches > 0].mean()),
            'central80_interval_score_inches': float(interval_score.mean()),
            'central80_mean_width_inches': float(np.mean(data.q90_inches-data.q10_inches))}
    if 'positive_log_mu' in data and data.positive_log_mu.notna().all():
        positive = (data.actual_inches > 0).to_numpy()
        log_y = np.log(np.maximum(data.actual_inches.to_numpy(), 1e-5))
        probability = np.clip(data.p_snow.to_numpy(), 1e-8, 1-1e-8)
        mu, sigma = data.positive_log_mu.to_numpy(), data.positive_log_sigma.to_numpy()
        nll = -(positive*np.log(probability)+(1-positive)*np.log(1-probability))
        nll += positive*(.5*((log_y-mu)/sigma)**2+np.log(sigma)+log_y+.5*np.log(2*np.pi))
        result['mean_negative_log_likelihood'] = float(nll.mean())
    return result


def evaluate(frame, model, scaler, origins, fit_years):
    data = arrays(frame, origins, scaler)
    predicted = predict(model, data)
    rows, monthly = [], []
    for i, (year, origin) in enumerate(origins):
        params = predicted[i]
        dates = pd.date_range(origin, periods=HORIZON)
        means = distribution_mean(params)
        low, high = distribution_quantile(params, .1), distribution_quantile(params, .9)
        actual = frame.reindex(dates).snowfall_inches.to_numpy(float)
        baseline = climatology_parameters(frame, fit_years, dates)
        for j, date in enumerate(dates):
            base = {'season_start': year, 'origin': str(origin.date()), 'date': str(date.date()),
                    'lead_day': j+1, 'actual_inches': actual[j]}
            rows.append({**base, 'model': 'multivariate_gru', 'predicted_inches': means[j], 'p_snow': params[j, 0],
                         'positive_log_mu': params[j, 1], 'positive_log_sigma': params[j, 2], 'q10_inches': low[j], 'q90_inches': high[j]})
            rows.append({**base, 'model': 'daily_calendar_climatology', 'predicted_inches': baseline[0][j],
                         'p_snow': baseline[1][j], 'q10_inches': baseline[2][j], 'q90_inches': baseline[3][j]})
        if origin.day != 1:
            continue
        paths, templates = rank_template_paths(frame, fit_years, origin, params, SEED+year+origin.month)
        for month in dates.to_period('M').unique():
            in_month = dates.to_period('M') == month
            if int(in_month.sum()) != month.days_in_month or not np.isfinite(actual[in_month]).all():
                continue
            historical_totals = []
            for fit_year in fit_years:
                calendar_year = fit_year if month.month >= 10 else fit_year+1
                month_dates = pd.date_range(f'{calendar_year}-{month.month:02d}-01', periods=pd.Period(f'{calendar_year}-{month.month:02d}', freq='M').days_in_month)
                history = frame.reindex(month_dates).snowfall_inches
                if history.notna().all():
                    historical_totals.append(float(history.sum()))
            metadata = {'season_start': year, 'origin': str(origin.date()), 'month': str(month),
                        'actual_inches': float(actual[in_month].sum()), 'forecast_lead_start': int(np.where(in_month)[0][0]+1),
                        'fixed_nov1': origin.month == 11, 'template_blocks': templates}
            totals = paths[:, in_month].sum(axis=1)
            for name, mean, lo, hi in [
                ('multivariate_gru', means[in_month].sum(), *np.quantile(totals, [.1, .9])),
                ('daily_calendar_climatology', baseline[0][in_month].sum(), *np.quantile(historical_totals, [.1, .9])),
                ('direct_monthly_climatology', np.mean(historical_totals), *np.quantile(historical_totals, [.1, .9]))]:
                monthly.append({**metadata, 'model': name, 'predicted_inches': float(mean), 'q10_inches': float(lo), 'q90_inches': float(hi),
                                'interval_method': 'sum of coherent model paths' if name == 'multivariate_gru' else 'empirical complete historical month totals'})
    return pd.DataFrame(rows), pd.DataFrame(monthly)


def monthly_scores(rows):
    result = {}
    for name, part in rows.groupby('model'):
        error = part.predicted_inches-part.actual_inches
        result[name] = {'month_predictions': len(part), 'unique_target_months': part.month.nunique(),
                        'rmse_inches': float(np.sqrt(np.mean(error**2))), 'mae_inches': float(np.abs(error).mean()),
                        'bias_inches': float(error.mean()),
                        'central80_coverage': float(((part.actual_inches >= part.q10_inches) & (part.actual_inches <= part.q90_inches)).mean()),
                        'central80_interval_score_inches': float((part.q90_inches-part.q10_inches+10*np.maximum(part.q10_inches-part.actual_inches, 0)+10*np.maximum(part.actual_inches-part.q90_inches, 0)).mean()),
                        'central80_mean_width_inches': float((part.q90_inches-part.q10_inches).mean())}
    return result


def bootstrap_mse_difference(rows, seed=SEED, repeats=10000):
    """Resample whole winters, jointly weighting squared-error sums and counts."""
    pivot = rows.pivot(index=['season_start', 'origin', 'date'], columns='model', values=['predicted_inches', 'actual_inches']).dropna()
    y = pivot['actual_inches']['multivariate_gru']
    diff = (pivot['predicted_inches']['multivariate_gru']-y)**2-(pivot['predicted_inches']['daily_calendar_climatology']-y)**2
    group = diff.groupby(level='season_start').agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(group), size=(repeats, len(group)))
    estimates = group['sum'].to_numpy()[choices].sum(axis=1)/group['count'].to_numpy()[choices].sum(axis=1)
    return {'definition': 'neural MSE minus calendar-climatology MSE; negative favors neural',
            'difference_square_inches': float(diff.mean()), 'whole_winter_bootstrap95': np.quantile(estimates, [.025, .975]).tolist(),
            'winter_clusters': len(group), 'bootstrap_replicates': repeats,
            'caution': 'Only five independent test winters; overlapping forecast origins are not independent samples.'}


def monthly_bootstrap_differences(rows, seed=SEED, repeats=10000):
    pivot = rows.pivot(index=['season_start', 'origin', 'month'], columns='model', values=['predicted_inches', 'actual_inches']).dropna()
    truth = pivot['actual_inches']['multivariate_gru']
    neural_squared_error = (pivot['predicted_inches']['multivariate_gru']-truth)**2
    result = {}
    for baseline in ['daily_calendar_climatology', 'direct_monthly_climatology']:
        diff = neural_squared_error-(pivot['predicted_inches'][baseline]-truth)**2
        group = diff.groupby(level='season_start').agg(['sum', 'count'])
        choices = np.random.default_rng(seed).integers(0, len(group), size=(repeats, len(group)))
        estimates = group['sum'].to_numpy()[choices].sum(axis=1)/group['count'].to_numpy()[choices].sum(axis=1)
        result[baseline] = {'neural_minus_baseline_mse_square_inches': float(diff.mean()),
                            'whole_winter_bootstrap95': np.quantile(estimates, [.025, .975]).tolist(),
                            'winter_clusters': len(group), 'paired_months': len(diff), 'bootstrap_replicates': repeats}
    return result


def save_checkpoint(path, model, scaler, config, years, epochs):
    torch.save({'state_dict': model.state_dict(), 'scaler': scaler.as_dict(), 'config': config,
                'fit_season_starts': years, 'epochs': epochs, 'context_days': CONTEXT,
                'forecast_horizon_days': HORIZON, 'seed': SEED}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='outputs/dillon_deep_daily_input.csv')
    parser.add_argument('--epochs', type=int, default=70)
    args = parser.parse_args()
    start_clock = time.time()
    path = ROOT/args.input
    frame = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
    assert frame.index.is_unique
    frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), name='date'))
    outputs = ROOT/'outputs'
    model_dir = outputs/'deep_models'
    model_dir.mkdir(exist_ok=True)
    train_origins = make_origins(frame, TRAIN_YEARS, 3)
    valid_origins = make_origins(frame, VALID_YEARS, 7)
    test_origins = make_origins(frame, TEST_YEARS, 7)
    configs = [{'hidden': 32, 'dropout': .25, 'weight_decay': .01, 'learning_rate': .001},
               {'hidden': 64, 'dropout': .35, 'weight_decay': .01, 'learning_rate': .001}]
    trials = []
    for config in configs:
        model, scaler, log, epoch, score = train(frame, train_origins, config, valid_origins, args.epochs)
        trials.append({'config': config, 'best_epoch': epoch, 'validation_nll': score,
                       'parameter_count': sum(p.numel() for p in model.parameters()), 'logs': log})
        (outputs/f'deep_dillon_training_width{config["hidden"]}.json').write_text(json.dumps(trials[-1], indent=2))
    selected = min(trials, key=lambda t: t['validation_nll'])
    config, epochs = selected['config'], selected['best_epoch']
    print(json.dumps({'selected_config': config, 'fixed_refit_epochs': epochs}), flush=True)
    evaluation_years = TRAIN_YEARS+VALID_YEARS
    evaluation_origins = make_origins(frame, evaluation_years, 3)
    model, scaler, evaluation_log, _, _ = train(frame, evaluation_origins, config, epochs=epochs)
    save_checkpoint(model_dir/'dillon_gru_evaluation.pt', model, scaler, config, evaluation_years, epochs)
    test_rows, monthly = evaluate(frame, model, scaler, test_origins, evaluation_years)
    test_rows.to_csv(outputs/'deep_dillon_test_daily.csv', index=False, float_format='%.8f')
    monthly.to_csv(outputs/'deep_dillon_test_monthly.csv', index=False, float_format='%.8f')
    daily_scores = {name: score_rows(part) for name, part in test_rows.groupby('model')}
    lead_scores = {}
    for name, part in test_rows.groupby('model'):
        lead_scores[name] = {f'{lo}-{hi}': score_rows(part.loc[part.lead_day.between(lo, hi)]) for lo, hi in [(1,7),(8,14),(15,30),(31,61)]}
    # Archive fitted training performance to distinguish optimization from generalization.
    train_params = predict(model, arrays(frame, evaluation_origins, scaler))
    train_y = arrays(frame, evaluation_origins, scaler)[2].numpy()
    train_mask = arrays(frame, evaluation_origins, scaler)[3].numpy().astype(bool)
    training_rmse = float(np.sqrt(np.mean((distribution_mean(train_params)[train_mask]-train_y[train_mask])**2)))
    final_years = evaluation_years+TEST_YEARS
    final_origins = make_origins(frame, final_years, 3)
    final_model, final_scaler, final_log, _, _ = train(frame, final_origins, config, epochs=epochs)
    save_checkpoint(model_dir/'dillon_gru_forecast.pt', final_model, final_scaler, config, final_years, epochs)
    cutoff = frame.index.max()
    origin = cutoff+pd.Timedelta(days=1)
    params = predict(final_model, arrays(frame, [(cutoff.year, origin)], final_scaler))[0]
    dates = pd.date_range(origin, periods=HORIZON)
    keep = dates <= pd.Timestamp(f'{cutoff.year}-12-31')
    params, dates = params[keep], dates[keep]
    paths, template_count = rank_template_paths(frame, final_years, origin, params, SEED, 4096)
    np.savez_compressed(outputs/'deep_dillon_forecast_paths.npz', paths_inches=paths, dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'))
    baseline = climatology_parameters(frame, final_years, dates)
    forecast = pd.DataFrame({'date': dates, 'lead_day': np.arange(1, len(dates)+1),
                             'predicted_inches': distribution_mean(params), 'p_snow': params[:, 0],
                             'positive_log_mu': params[:, 1], 'positive_log_sigma': params[:, 2],
                             'q10_inches': distribution_quantile(params, .1), 'q90_inches': distribution_quantile(params, .9),
                             'q05_inches': distribution_quantile(params, .05), 'q95_inches': distribution_quantile(params, .95),
                             'historical_daily_mean_inches': baseline[0], 'historical_q10_inches': baseline[2], 'historical_q90_inches': baseline[3]})
    cumulative = np.cumsum(paths, axis=1)
    forecast['cumulative_predicted_inches'] = np.cumsum(forecast.predicted_inches)
    forecast['cumulative_q10_inches'], forecast['cumulative_q90_inches'] = np.quantile(cumulative, [.1, .9], axis=0)
    forecast.to_csv(outputs/'deep_dillon_forecast_daily.csv', index=False, float_format='%.8f')
    current_months = []
    for month in dates.to_period('M').unique():
        in_month = dates.to_period('M') == month
        observed_prefix = frame.loc[str(month)].snowfall_inches if str(month) in frame.index.strftime('%Y-%m') else pd.Series(dtype=float)
        prefix_total = float(observed_prefix.sum())
        month_paths = paths[:, in_month].sum(axis=1)+prefix_total
        current_months.append({'month': str(month), 'observed_prefix_days': len(observed_prefix),
                               'observed_prefix_inches': prefix_total, 'predicted_remaining_inches': float(forecast.predicted_inches.to_numpy()[in_month].sum()),
                               'predicted_full_month_inches': float(forecast.predicted_inches.to_numpy()[in_month].sum()+prefix_total),
                               'q10_full_month_inches': float(np.quantile(month_paths, .1)), 'q90_full_month_inches': float(np.quantile(month_paths, .9))})
    pd.DataFrame(current_months).to_csv(outputs/'deep_dillon_forecast_monthly.csv', index=False, float_format='%.8f')
    metadata = {
        'station': 'NOAA GHCN-D USC00052281, Dillon 1 E', 'issue_date': '2026-11-03', 'observation_cutoff': str(cutoff.date()),
        'forecast_origin': str(origin.date()), 'requested_display_period': ['2026-11-04', '2026-12-31'],
        'input_path': str(path.relative_to(ROOT)), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'python': __import__('sys').version, 'torch': torch.__version__, 'numpy': np.__version__, 'seed': SEED, 'cpu_threads': 2,
        'architecture': 'Two-layer GRU historical encoder; two-layer SiLU decoder conditioned on future calendar and forecast lead; zero-inflated lognormal daily marginals',
        'context_days': CONTEXT, 'horizon_days': HORIZON, 'historical_input_features': FEATURES,
        'historical_input_dimensions': 14, 'future_inputs': ['calendar sin/cos first and second annual harmonic', 'lead/61', '(lead/61)^2',
                                                            'origin-known lagged RONI, strong warm/cold flags, missing flag'],
        'no_future_observed_weather': True, 'enso_predictor': True,
        'enso_contract': 'Origin month minus3 centered RONI period, ending two calendar months before origin month. Nov uses JAS, Dec ASO, Jan SON. RONI held constant for all future leads. Historical values are revised and not archived issue-time releases: pseudo-real-time availability only.',
        'roni_sha256': hashlib.sha256(RONI_PATH.read_bytes()).hexdigest(),
        'current_lagged_roni': lagged_roni(origin),
        'training_season_starts': TRAIN_YEARS, 'validation_season_starts': VALID_YEARS, 'test_season_starts': TEST_YEARS,
        'selection': 'Minimum separate-winter validation mean zero-inflated-lognormal negative log likelihood. Early stopping patience10, minimum15epochs; two prespecified configurations.',
        'hyperparameter_trials': trials, 'selected_config': config, 'selected_epochs': epochs,
        'parameter_count': selected['parameter_count'], 'validation_model_scalers_fit_on_training_only': True,
        'evaluation_fit_season_starts': evaluation_years, 'final_fit_season_starts': final_years,
        'refit_contract': 'After validation selection, refit train+validation for fixed selected epochs before untouched test scoring. Then refit all completed winters, including test winters, for the production forecast. Test metrics belong to the earlier evaluation checkpoint only.',
        'window_counts': {'train': len(train_origins), 'validation': len(valid_origins), 'test': len(test_origins),
                          'evaluation_refit_train': len(evaluation_origins), 'final_refit_train': len(final_origins)},
        'origin_policy': 'October1 through February28, stride3 training / stride7 validation and test, plus every month start. Each61day target and60day context belongs to one Aug-Apr winter group. Target masks retain missing labels; windows need at least58observed targetdays.',
        'split_purge': 'Latest preceding-split targets endApril29; next split earliest contexts beginAugust2. No training context/target shares a date with validation/test context/target.',
        'daily_test_scores': daily_scores, 'lead_group_test_scores': lead_scores,
        'fixed_nov1_monthly_test_scores': monthly_scores(monthly.loc[monthly.fixed_nov1]),
        'all_monthstart_monthly_test_scores': monthly_scores(monthly), 'daily_mse_winter_bootstrap': bootstrap_mse_difference(test_rows),
        'fixed_nov1_monthly_mse_winter_bootstrap': monthly_bootstrap_differences(monthly.loc[monthly.fixed_nov1]),
        'training_rmse_inches': training_rmse, 'evaluation_training_log': evaluation_log, 'final_training_log': final_log,
        'final_scaler': final_scaler.as_dict(), 'forecast_months': current_months,
        'uncertainty': {'daily': 'Central80% and90% intervals from the fitted zero-inflated lognormal marginal distribution; model-based, not calibrated.',
                        'paths': '4096 marginal-quantile paths reordered by ranks of complete contiguous historical blocks, matching origin calendar ±14days. Tied-zero ranks randomized reproducibly; this assumes historical temporal dependence persists.',
                        'monthly': 'Quantiles of sums of coherent daily paths, plus known observed month prefix. Never sums of marginal daily endpoints.',
                        'test_paths': '1024 paths per monthly test origin; templates restricted to 2005–2020 fit winters.',
                        'final_template_blocks': template_count,
                        'zero_mass_note': 'Central q10-q90 intervals have lower bound0 when dry-day probability exceeds10%; because of the discrete atom at0, their closed-set coverage can exceed80% even under the true model.',
                        'limitations': 'No posterior parameter uncertainty or numerical-weather-prediction inputs. The copula dependence assumption is not separately calibrated; small5winter evaluation limits coverage estimates. Daily width is not constrained to increase with lead.'},
        'limitations': ['Dillon valley station cannot directly stand in for Copper, A-Basin or Keystone snowfall.',
                        'Calendar-date weather history gives a statistical outlook, not reliable exact storm timing two months ahead.',
                        'Only21winter groups despite thousands of daily rows; overlapping target windows are dependent.',
                        'Lagged RONI is revised historical data, not archived release vintages; availability is only approximated conservatively. ENSO feature importance is not established by inclusion.',
                        'Observation reporting/publication times are not fully archived; the backtest is date-indexed, not a verified issue-time operational hindcast.'],
        'elapsed_seconds': time.time()-start_clock}
    (outputs/'deep_dillon_results.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'completed': True, 'seconds': metadata['elapsed_seconds'], 'daily_scores': daily_scores,
                      'fixed_nov1_monthly_scores': metadata['fixed_nov1_monthly_test_scores'], 'forecast_months': current_months}), flush=True)


if __name__ == '__main__':
    main()
